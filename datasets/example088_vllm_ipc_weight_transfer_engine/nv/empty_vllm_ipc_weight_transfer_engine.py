#!/usr/bin/env python3
"""
ref_vllm_ipc_weight_transfer_engine.py

Reference implementation and benchmark for vLLM's CUDA-IPC weight
transfer engine — the single-node analogue of NCCLWeightTransferEngine.

Background:
    vLLM's IPCWeightTransferEngine (vllm/distributed/weight_transfer/
    ipc_engine.py) is the transport used when the trainer and
    inference workers run as separate processes that share a physical
    GPU (e.g. a co-located trainer on the same node, or an MPS-style
    multi-process launch).  Instead of copying tensor bytes through
    NCCL, the trainer hands receivers a CUDA IPC handle pickle —
    `torch.multiprocessing.reductions.reduce_tensor` produces
    `(rebuild_cuda_tensor, args)` — and the receiver calls
    `rebuild_cuda_tensor(*args)` to map the trainer's GPU memory into
    its own address space.  No bytes cross the device.

Why we wrap reduce_tensor + the static helpers (not the engine):
    `IPCWeightTransferEngine.__init__` requires a `WeightTransferConfig`
    + `ParallelConfig` and `trainer_send_weights` ships handles via
    Ray RPC or HTTP POST — heavy dependencies for a kernel benchmark.
    The interesting wire pattern IS the
    `reduce_tensor(weight) -> (func, args)` step on the trainer side
    plus the receiver's `func(*args)` call (with the device index
    patched to the receiver's local device, exactly as the engine
    does at line 187 of ipc_engine.py).  We exchange the (func, args)
    tuples over a gloo `broadcast_object_list` instead of Ray/HTTP —
    same on-the-wire payload (pickle of the dict), no Ray cluster
    required.

Hardware floor / shape:
    CUDA IPC requires the trainer and inference processes to be on the
    same node, and the engine's UUID-keyed dict assumes they share the
    same physical GPU.  We pin BOTH ranks to LOCAL_RANK=0 (cuda:0) so
    `props.uuid` matches between trainer and inference — the canonical
    deployment shape (e.g. trainer + an MPS-multiplexed inference
    worker on the same GPU).

Purpose:
- Demonstrate the production "trainer hands inference IPC handles to
  10 named buckets without copying device bytes" path.
- Verify byte equality on the rebuilt tensors against the trainer's
  in-place pattern.
- Benchmark per-update latency (handle export + IPC open + load) and
  the apparent aggregate "bandwidth" so it can be compared against
  example087's NCCL transport.
- Print exactly one JSON object to stdout (rank 0 only).

Launch:
    torchrun --nproc_per_node=2 ref_vllm_ipc_weight_transfer_engine.py
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


# ============================================================
# Distributed Environment
# ============================================================

class DistributedEnv:
    """
    Initializes a Gloo-backed world group used purely as a CPU rendezvous
    for the IPC-handle pickle exchange.  No NCCL group: CUDA IPC moves
    no bytes through any collective, just exports a memory handle the
    peer maps directly.

    Both ranks are pinned to cuda:0 — IPCWeightTransferEngine's
    receiver-side lookup keys on `props.uuid`, so trainer and worker
    must share the same physical GPU UUID for the production code path
    to be exercised faithfully.
    """

    @staticmethod
    def Init() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        assert world_size == local_world_size, "multi-node runs are not supported"
        assert rank == local_rank, "multi-node runs are not supported"
        assert world_size == 2, (
            "IPC weight transfer example pins world_size=2 "
            "(one trainer + one inference worker sharing cuda:0)"
        )

        # Both ranks pinned to GPU 0 — matches the engine's UUID lookup
        # contract.  Each rank still gets its own CUDA context.
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
        )
        torch.manual_seed(1234 + local_rank)
        return local_rank, world_size

    @staticmethod
    def Destroy() -> None:
        dist.destroy_process_group()

    @staticmethod
    def Barrier() -> None:
        dist.barrier()


# ============================================================
# Core Implementation (no test/benchmark logic here)
# ============================================================

@dataclass
class WeightSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype


# `gpu_id` lives once per host; both ranks read it lazily.
def _local_gpu_uuid(device_index: int = 0) -> str:
    return str(torch.cuda.get_device_properties(device_index).uuid)


class VllmIpcWeightTransfer:
    """
    Wrapper around vLLM's IPCWeightTransferEngine helpers that exposes a
    SendWeights / RecvWeights pair driven by torch's CUDA-IPC reductions
    plus a gloo `broadcast_object_list` rendezvous.

    `SendWeights(named_tensors)` (rank 0 / trainer):
        For each (name, tensor): call
        `reduce_tensor(tensor.detach().contiguous())` → `(func, args)`,
        store under `{gpu_uuid: (func, args)}`.  Build the
        IPCWeightTransferUpdateInfo-shaped dict and broadcast it to all
        receivers via `dist.broadcast_object_list` on the gloo group.

        Note we keep references to the source tensors alive on the
        sender side; CUDA IPC handles are tied to the source storage,
        and freeing the source while the receiver still holds the
        rebuilt tensor would invalidate the mapping.

    `RecvWeights()` (rank != 0 / inference):
        Receive the broadcast payload, then for each `(name, dtype,
        shape, ipc_handle_dict)`: look up by trainer's gpu UUID, patch
        the device index in args[6] to local device, call
        `func(*args)` to rebuild the CUDA tensor.  Returns the list
        `load_weights` would receive in production.

    `ReleaseSent()`:
        Drop the keep-alive references the trainer holds on the source
        tensors.  Useful in benchmarks so the sender doesn't accumulate
        exports across iterations.
    """

    def __init__(self, device: torch.device):
        # TODO
        raise NotImplementedError

    def SendWeights(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        group: dist.ProcessGroup,
    ) -> None:
        """Trainer side: build an UpdateInfo-equivalent dict and broadcast
        it to all receivers via the supplied gloo group."""
        # TODO
        raise NotImplementedError

    def RecvWeights(
        self, group: dist.ProcessGroup
    ) -> List[Tuple[str, torch.Tensor]]:
        """Inference side: receive the broadcast and rebuild tensors."""
        # TODO
        raise NotImplementedError

    def ReleaseSent(self) -> None:
        # TODO
        raise NotImplementedError

    def Sync(self) -> None:
        # TODO
        raise NotImplementedError

    def Close(self) -> None:
        # TODO
        raise NotImplementedError


# ============================================================
# Weight-spec construction & deterministic patterns
# ============================================================

_DTYPE_ROTATION: List[torch.dtype] = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.int64,
]


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _build_specs(total_bytes: int, num_buckets: int) -> List[WeightSpec]:
    per_bucket_bytes = total_bytes // num_buckets
    specs: List[WeightSpec] = []
    for i in range(num_buckets):
        dtype = _DTYPE_ROTATION[i % len(_DTYPE_ROTATION)]
        numel = max(1, per_bucket_bytes // _itemsize(dtype))
        specs.append(WeightSpec(
            name=f"layer_{i:02d}.weight", shape=(numel,), dtype=dtype,
        ))
    return specs


def _expected_pattern(spec: WeightSpec, device: torch.device) -> torch.Tensor:
    spec_idx = int(spec.name.split("_")[1].split(".")[0])
    idx = torch.arange(spec.shape[0], device=device)
    base = (idx + spec_idx) % 64
    return base.to(spec.dtype)


def _build_named_tensors(
    specs: List[WeightSpec], device: torch.device
) -> List[Tuple[str, torch.Tensor]]:
    return [(s.name, _expected_pattern(s, device)) for s in specs]


def _verify_received(
    received: List[Tuple[str, torch.Tensor]],
    specs: List[WeightSpec],
    device: torch.device,
) -> bool:
    if len(received) != len(specs):
        return False
    by_name = dict(received)
    for s in specs:
        if s.name not in by_name:
            return False
        got = by_name[s.name]
        if got.dtype != s.dtype or tuple(got.shape) != tuple(s.shape):
            return False
        expected = _expected_pattern(s, device)
        if not torch.equal(got, expected):
            return False
    return True


# ============================================================
# Evaluation
# ============================================================

@dataclass
class WeightTransferCase:
    name: str
    total_bytes: int
    num_buckets: int = 10


class WeightTransferEvaluator:
    """Runs correctness check and wall-clock benchmark for a single case.

    Apparent BW is bytes_total / wall_time, where `bytes_total` is the
    nominal payload — IPC moves no bytes across the device, so this is
    a *zero-copy throughput* metric for direct comparison with NCCL
    (example087).  Expect the IPC number to be much higher than NCCL's
    because the actual transfer is a per-tensor IPC handle map.
    """

    def __init__(
        self,
        comm: VllmIpcWeightTransfer,
        local_rank: int,
        world_size: int,
        group: dist.ProcessGroup,
    ):
        self.comm = comm
        self.local_rank = local_rank
        self.world_size = world_size
        self.group = group
        self.device = comm.device
        self.is_trainer = local_rank == 0

    def _build(self, case: WeightTransferCase
               ) -> Tuple[List[WeightSpec], Optional[List[Tuple[str, torch.Tensor]]]]:
        specs = _build_specs(case.total_bytes, case.num_buckets)
        if self.is_trainer:
            payload = _build_named_tensors(specs, self.device)
        else:
            payload = None
        return specs, payload

    def CheckCorrectness(self, case: WeightTransferCase) -> bool:
        specs, payload = self._build(case)

        if self.is_trainer:
            self.comm.SendWeights(payload, group=self.group)
            self.comm.Sync()
            ok_local = True
        else:
            received = self.comm.RecvWeights(group=self.group)
            self.comm.Sync()
            ok_local = _verify_received(received, specs, self.device)

        # All-reduce over gloo expects a CPU tensor; the world group is
        # gloo here (no NCCL needed for the CPU rendezvous).
        flag = torch.tensor(
            [1 if ok_local else 0], dtype=torch.int32
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.group)
        return bool(flag.item() == 1)

    def Benchmark(
        self,
        case: WeightTransferCase,
        warmup_iters: int,
        iters: int,
    ) -> Dict[str, float]:
        specs, payload = self._build(case)

        for _ in range(warmup_iters):
            if self.is_trainer:
                self.comm.SendWeights(payload, group=self.group)
            else:
                _ = self.comm.RecvWeights(group=self.group)
        self.comm.Sync()
        dist.barrier(group=self.group)

        if self.is_trainer:
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                self.comm.SendWeights(payload, group=self.group)
                self.comm.Sync()
            elapsed_ns = time.perf_counter_ns() - t0
        else:
            for _ in range(iters):
                _ = self.comm.RecvWeights(group=self.group)
                self.comm.Sync()
            elapsed_ns = 0

        dist.barrier(group=self.group)

        latency_us = elapsed_ns / iters / 1e3 if elapsed_ns > 0 else 0.0
        actual_bytes = sum(
            s.shape[0] * _itemsize(s.dtype) for s in specs
        )
        size_mb = actual_bytes / (1024.0 * 1024.0)
        if latency_us > 0:
            throughput_gbps = (
                (actual_bytes * 8.0) / (latency_us * 1e-6) / 1e9
            )
        else:
            throughput_gbps = 0.0

        return {
            "data_size": size_mb,
            "latency_avg": latency_us,
            "throughput_avg": throughput_gbps,
        }


# ============================================================
# Test orchestration
# ============================================================

def runTest():
    local_rank, world_size = DistributedEnv.Init()
    device = torch.device("cuda:0")  # both ranks pinned to cuda:0

    correctness = "PASS"
    metrics: List[Dict[str, float]] = []

    comm: Optional[VllmIpcWeightTransfer] = None
    try:
        comm = VllmIpcWeightTransfer(device=device)
        evaluator = WeightTransferEvaluator(
            comm=comm,
            local_rank=local_rank,
            world_size=world_size,
            group=dist.group.WORLD,
        )

        # Same sweep as example087 so direct comparisons are 1:1.
        # Expect IPC apparent-throughput to be 1-2 orders of magnitude
        # higher because no device bytes move per update.
        cases = [
            WeightTransferCase(
                name="1MiB",   total_bytes=1 * 1024 * 1024,
            ),
            WeightTransferCase(
                name="16MiB",  total_bytes=16 * 1024 * 1024,
            ),
            WeightTransferCase(
                name="64MiB",  total_bytes=64 * 1024 * 1024,
            ),
            WeightTransferCase(
                name="256MiB", total_bytes=256 * 1024 * 1024,
            ),
        ]

        for case in cases:
            if not evaluator.CheckCorrectness(case):
                correctness = "FAIL"
                break

        if correctness == "PASS":
            for case in cases:
                m = evaluator.Benchmark(case, warmup_iters=2, iters=5)
                metrics.append({k: round(v, 4) for k, v in m.items()})

    finally:
        if comm is not None:
            comm.Close()
        DistributedEnv.Barrier()
        DistributedEnv.Destroy()

    if int(os.environ.get("RANK", 0)) == 0:
        out = {
            "Correctness": correctness,
            "data_size_unit": "MB",
            "throughput_unit": "Gbps",
            "latency_unit": "us",
            "metrics": metrics,
        }
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    runTest()
