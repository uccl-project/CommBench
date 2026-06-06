#!/usr/bin/env python3
"""
empty_vllm_shm_broadcast_object.py

Reference implementation and benchmark for vLLM's POSIX shared-memory
ring-buffer broadcast (vllm.distributed.device_communicators.shm_broadcast.
MessageQueue).

Purpose:
- Demonstrate the production "single-node driver→workers control-plane
  broadcast" path that vLLM uses to ship Python objects (sampling params,
  request metadata, scheduler decisions) without going through a
  torch.distributed object broadcast — the latter pickles+collects+IPCs
  on every call, while MessageQueue pickles once into a shared ring
  buffer with zero-copy reads.
- Verify each reader receives exactly the byte sequence the writer
  enqueued (no oracle needed: the writer's input is the ground truth).
- Benchmark per-message round-trip latency and message throughput across
  payload sizes from 1 KiB to 1 MiB.
- Print exactly one JSON object to stdout (rank 0 only).

Units:
    data_size_unit  KB        (object payload size, KiB)
    latency_unit    us        (per-message round-trip)
    throughput_unit msg_per_s

Hint for the model:
- The relevant vLLM class is
  `vllm.distributed.device_communicators.shm_broadcast.MessageQueue`.
- Use the static helper `MessageQueue.create_from_process_group(pg, ...)`
  with `writer_rank=0, blocking=True` — it handles all handle exchange
  through the supplied Gloo group.
- Writer side calls `enqueue(obj)`; reader side calls `dequeue(timeout=...)`.
- Cleanup uses `shutdown()`.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch.distributed as dist


# ============================================================
# Distributed Environment
# ============================================================

class DistributedEnv:
    """
    Sets up a Gloo process group (no NCCL, no GPU dependency required).
    The MessageQueue helper bootstraps its handle exchange through this
    group and validates that all ranks live on the same node.
    """

    gloo_group = None

    @staticmethod
    def Init() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Single-node only — MessageQueue's local-reader path requires
        # all readers to share /dev/shm with the writer.
        assert world_size == local_world_size, "multi-node runs are not supported"
        assert rank == local_rank, "multi-node runs are not supported"
        assert world_size >= 2, "shm_broadcast example requires at least 2 ranks"

        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
        )
        DistributedEnv.gloo_group = dist.group.WORLD
        return rank, world_size

    @staticmethod
    def Destroy() -> None:
        dist.destroy_process_group()

    @staticmethod
    def Barrier() -> None:
        dist.barrier()


# ============================================================
# Core Implementation (no test/benchmark logic here)
# ============================================================

class VllmShmBroadcast:
    """
    Wrapper around vLLM's MessageQueue that exposes role-aware
    `Send(obj)` / `Recv()` methods for benchmarking a 1-writer / N-reader
    broadcast pattern.

    Construction is collective: every rank in `pg` must call __init__.
    Rank `writer_rank` becomes the producer; all other ranks become
    consumers.  Handle exchange happens through `pg`.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        writer_rank: int = 0,
        max_chunk_bytes: int = 4 * 1024 * 1024,  # 4 MiB
        max_chunks: int = 16,
    ):
        # TODO: bring up the underlying MessageQueue via
        # MessageQueue.create_from_process_group(...) with the supplied
        # `writer_rank` and `blocking=True`.  Cache the resulting handle
        # on `self._mq`, the rank on `self.rank`, and a boolean
        # `self.is_writer = (self.rank == writer_rank)`.
        self.pg = pg
        self.writer_rank = writer_rank
        self.rank = dist.get_rank(pg)
        self.is_writer = (self.rank == writer_rank)
        self._mq = None
        raise NotImplementedError

    def Send(self, obj) -> None:
        """Writer side: enqueue one object into the ring buffer."""
        # TODO: assert is_writer and call the underlying enqueue.
        raise NotImplementedError

    def Recv(self, timeout: Optional[float] = 30.0):
        """Reader side: dequeue one object."""
        # TODO: assert NOT is_writer and call the underlying dequeue
        # with the provided timeout.
        raise NotImplementedError

    def Close(self) -> None:
        # TODO: call shutdown() on the underlying MessageQueue.
        raise NotImplementedError


# ============================================================
# Evaluation
# ============================================================

@dataclass
class ShmBroadcastCase:
    payload_bytes: int
    n_messages: int


def _make_payload(seq: int, payload_bytes: int) -> Dict:
    """Deterministic payload — both writer and reader can regenerate it.

    Uses a SHA-256 stream keyed by `seq` so corrupted bytes are detected
    even when the size happens to match.  We expand the digest to fill
    `payload_bytes` so pickle overhead is small relative to the payload.
    """
    h = hashlib.sha256(str(seq).encode()).digest()  # 32 bytes
    blob = (h * (payload_bytes // 32 + 1))[:payload_bytes]
    return {"seq": seq, "blob": blob}


class ShmBroadcastEvaluator:
    """Drives correctness + benchmark for the writer/reader pair on one case."""

    def __init__(self, comm: VllmShmBroadcast, world_size: int):
        self.comm = comm
        self.world_size = world_size
        self.n_readers = world_size - 1

    # --------------------------------------------------------
    # Correctness — every reader must receive every payload in order
    # --------------------------------------------------------
    def CheckCorrectness(self, case: ShmBroadcastCase) -> bool:
        if self.comm.is_writer:
            for seq in range(case.n_messages):
                self.comm.Send(_make_payload(seq, case.payload_bytes))
            DistributedEnv.Barrier()
            return True

        # Reader always drains the full sequence so leftover messages
        # don't pollute the next case's ring buffer; bytes from the ring
        # may come back as a memoryview, so coerce before comparing.
        ok = True
        for seq in range(case.n_messages):
            got = self.comm.Recv()
            expected = _make_payload(seq, case.payload_bytes)
            if got["seq"] != expected["seq"] or bytes(got["blob"]) != expected["blob"]:
                ok = False
        DistributedEnv.Barrier()
        return ok

    # --------------------------------------------------------
    # Benchmark — wall-clock round-trip, averaged
    # --------------------------------------------------------
    def Benchmark(
        self,
        case: ShmBroadcastCase,
        warmup_iters: int,
    ) -> Dict[str, float]:
        n = case.n_messages

        # Pre-build payloads outside the timed region so we measure the
        # IPC pipeline (pickle + ring-buffer write + signal + read), not
        # SHA-256 / bytes multiplication.
        if self.comm.is_writer:
            warmup_payloads = [
                _make_payload(seq, case.payload_bytes) for seq in range(warmup_iters)
            ]
            timed_payloads = [
                _make_payload(seq, case.payload_bytes) for seq in range(n)
            ]

        # Warmup
        if self.comm.is_writer:
            for p in warmup_payloads:
                self.comm.Send(p)
        else:
            for _ in range(warmup_iters):
                self.comm.Recv()
        DistributedEnv.Barrier()

        if self.comm.is_writer:
            t0 = time.perf_counter()
            for p in timed_payloads:
                self.comm.Send(p)
            t1 = time.perf_counter()
            DistributedEnv.Barrier()
        else:
            t0 = time.perf_counter()
            for _ in range(n):
                self.comm.Recv()
            t1 = time.perf_counter()
            DistributedEnv.Barrier()

        elapsed_s = t1 - t0
        latency_us = (elapsed_s / n) * 1e6
        throughput_msg_per_s = n / elapsed_s

        return {
            "data_size": case.payload_bytes / 1024.0,  # KB
            "latency_avg": latency_us,
            "throughput_avg": throughput_msg_per_s,
        }


# ============================================================
# Test orchestration
# ============================================================

def runTest():
    rank, world_size = DistributedEnv.Init()

    correctness = "PASS"
    metrics: List[Dict[str, float]] = []

    comm: Optional[VllmShmBroadcast] = None
    try:
        comm = VllmShmBroadcast(
            pg=DistributedEnv.gloo_group,
            writer_rank=0,
            max_chunk_bytes=4 * 1024 * 1024,
            max_chunks=16,
        )
        evaluator = ShmBroadcastEvaluator(comm=comm, world_size=world_size)

        payload_sizes = [
            1 * 1024,           # 1 KiB
            16 * 1024,          # 16 KiB
            64 * 1024,          # 64 KiB
            256 * 1024,         # 256 KiB
            1 * 1024 * 1024,    # 1 MiB
        ]
        n_messages_per_size = {
            1 * 1024: 1000,
            16 * 1024: 500,
            64 * 1024: 200,
            256 * 1024: 100,
            1 * 1024 * 1024: 50,
        }

        cases = [
            ShmBroadcastCase(payload_bytes=p, n_messages=n_messages_per_size[p])
            for p in payload_sizes
        ]

        for case in cases:
            if not evaluator.CheckCorrectness(case):
                correctness = "FAIL"
                break

        if correctness == "PASS":
            for case in cases:
                m = evaluator.Benchmark(case, warmup_iters=10)
                metrics.append({k: round(v, 4) for k, v in m.items()})

    finally:
        if comm is not None:
            comm.Close()
        DistributedEnv.Barrier()
        DistributedEnv.Destroy()

    if rank == 0:
        out = {
            "Correctness": correctness,
            "data_size_unit": "KB",
            "throughput_unit": "msg_per_s",
            "latency_unit": "us",
            "metrics": metrics,
        }
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    runTest()
