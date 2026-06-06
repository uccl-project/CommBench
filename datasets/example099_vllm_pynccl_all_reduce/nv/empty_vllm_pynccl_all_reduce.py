#!/usr/bin/env python3
"""
empty_vllm_pynccl_all_reduce.py

Reference implementation and benchmark for vLLM's direct-ctypes NCCL
wrapper (vllm.distributed.device_communicators.pynccl.PyNcclCommunicator).

Purpose:
- Demonstrate the production "all-reduce driven on an externally-supplied
  CUDA stream" path that vLLM uses when capturing a CUDA graph or running
  inside a torch.compile region — torch.distributed.all_reduce can't be
  used there because its `Work` future doesn't survive graph capture.
- Verify bit-exact correctness against torch.distributed.all_reduce on
  the same set of ranks (both paths drive the same NCCL build).
- Benchmark latency and bus bandwidth across fp16 message sizes that
  exercise both small (latency-bound) and large (BW-bound) regimes,
  with all-reduce calls issued on a dedicated non-default torch.cuda.Stream.
- Print exactly one JSON object to stdout (rank 0 only).

Hint for the model:
- The relevant vLLM class is
  `vllm.distributed.device_communicators.pynccl.PyNcclCommunicator`.
- It must be bound to a non-NCCL process group (Gloo is wired up below).
- Issue every all_reduce call on `self.stream`, NOT the default stream;
  this is the configuration vLLM hits during CUDA-graph capture.
- Cleanup uses `destroy()`, not `close()`.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


# ============================================================
# Distributed Environment
# ============================================================

class DistributedEnv:
    """
    Initializes:
      * the default (NCCL) process group used as the correctness oracle, and
      * a parallel Gloo subgroup that PyNcclCommunicator binds to (it
        explicitly forbids being attached to a NCCL-backed group).
    """

    nccl_group = None
    gloo_group = None

    @staticmethod
    def Init() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Single-node only.  PyNcclCommunicator works cross-node too, but
        # the example wires up rendezvous via torchrun's local launcher.
        assert world_size == local_world_size, "multi-node runs are not supported"
        assert rank == local_rank, "multi-node runs are not supported"

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=local_rank,
            rank=rank,
            world_size=world_size,
        )
        DistributedEnv.nccl_group = dist.group.WORLD
        DistributedEnv.gloo_group = dist.new_group(
            ranks=list(range(world_size)), backend="gloo"
        )
        torch.manual_seed(1234 + local_rank)
        return local_rank, world_size

    @staticmethod
    def Destroy() -> None:
        if DistributedEnv.gloo_group is not None:
            dist.destroy_process_group(DistributedEnv.gloo_group)
            DistributedEnv.gloo_group = None
        dist.destroy_process_group()

    @staticmethod
    def Barrier() -> None:
        dist.barrier()


# ============================================================
# Core Implementation (no test/benchmark logic here)
# ============================================================

class VllmPyNccl:
    """
    Wrapper around vLLM's PyNcclCommunicator that exposes a minimal
    `Run(input, out)` API for benchmarking.

    The wrapper owns a dedicated non-default CUDA stream and issues every
    all_reduce on it.  Callers are expected to synchronize via the
    returned stream, not via the default-stream synchronize, so the
    benchmark measures the NCCL kernel time rather than implicit
    cross-stream waits.
    """

    def __init__(
        self,
        gloo_group: dist.ProcessGroup,
        device: torch.device,
    ):
        # TODO: build a PyNcclCommunicator on the supplied Gloo group +
        # device.  Raise RuntimeError if the resulting communicator
        # reports `disabled`.  Allocate a dedicated, non-default
        # torch.cuda.Stream on `device` and store it on `self.stream`.
        self._impl = None
        self.device = device
        self.stream = None
        raise NotImplementedError

    def Run(self, input: torch.Tensor, out: torch.Tensor) -> None:
        """
        Out-of-place all-reduce sum across the bound process group on the
        wrapper's dedicated stream.
        """
        # TODO: have self.stream wait_stream(default) before invoking
        # NCCL — otherwise NCCL on self.stream can race with the
        # default-stream producer of `input` for large inputs.
        # Then call the vLLM communicator's all_reduce, passing
        # `out_tensor=out` and `stream=self.stream`.
        raise NotImplementedError

    def Sync(self) -> None:
        """Block the host until queued NCCL work on the comm stream is done."""
        # TODO
        raise NotImplementedError

    def Close(self) -> None:
        # TODO: call the underlying communicator's destroy() and clear
        # the stored impl.
        raise NotImplementedError


# ============================================================
# Evaluation
# ============================================================

@dataclass
class PyNcclCase:
    numel: int                          # fp16 element count
    dtype: torch.dtype = torch.float16


class PyNcclEvaluator:
    """Runs correctness check and CUDA-event-timed benchmark for a single case."""

    def __init__(
        self,
        comm: VllmPyNccl,
        nccl_group: dist.ProcessGroup,
        local_rank: int,
        world_size: int,
    ):
        self.comm = comm
        self.nccl_group = nccl_group
        self.device = torch.device(f"cuda:{local_rank}")
        self.world_size = world_size

    # --------------------------------------------------------
    # Tensor construction
    # --------------------------------------------------------
    def _build_input(self, case: PyNcclCase) -> torch.Tensor:
        # Small integers in fp16 storage so the per-rank sum lands well
        # inside the dtype's representable range and the two NCCL
        # invocations (PyNccl + torch.distributed) match bit-for-bit.
        return torch.randint(
            1, 8, (case.numel,), dtype=case.dtype, device=self.device
        )

    # --------------------------------------------------------
    # Correctness — bit-exact match against torch.distributed.all_reduce
    # --------------------------------------------------------
    def CheckCorrectness(self, case: PyNcclCase) -> bool:
        inp = self._build_input(case)

        # Reference: in-place NCCL all-reduce on a clone.
        ref = inp.clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=self.nccl_group)
        torch.cuda.synchronize(self.device)

        # PyNccl: out-of-place into a fresh tensor, on the wrapper's stream.
        out = torch.empty_like(inp)
        self.comm.Run(inp, out)
        self.comm.Sync()

        return bool(torch.equal(out, ref))

    # --------------------------------------------------------
    # Benchmark — CUDA-event timing on the comm stream
    # --------------------------------------------------------
    def Benchmark(
        self,
        case: PyNcclCase,
        warmup_iters: int,
        iters: int,
    ) -> Dict[str, float]:
        inp = self._build_input(case)
        out = torch.empty_like(inp)

        for _ in range(warmup_iters):
            self.comm.Run(inp, out)
        self.comm.Sync()
        dist.barrier(group=self.nccl_group)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record(self.comm.stream)
        for _ in range(iters):
            self.comm.Run(inp, out)
        end_evt.record(self.comm.stream)
        self.comm.Sync()

        elapsed_ms = start_evt.elapsed_time(end_evt) / iters
        latency_us = elapsed_ms * 1e3

        size_bytes = inp.numel() * inp.element_size()
        bus_factor = 2.0 * (self.world_size - 1) / self.world_size
        throughput_gbps = (
            (bus_factor * size_bytes * 8.0)
            / (elapsed_ms * 1e-3)
            / 1e9
        )

        return {
            "data_size": size_bytes / (1024.0 * 1024.0),  # MB
            "latency_avg": latency_us,
            "throughput_avg": throughput_gbps,
        }


# ============================================================
# Test orchestration
# ============================================================

def runTest():
    local_rank, world_size = DistributedEnv.Init()
    device = torch.device(f"cuda:{local_rank}")

    correctness = "PASS"
    metrics: List[Dict[str, float]] = []

    comm: Optional[VllmPyNccl] = None
    try:
        comm = VllmPyNccl(
            gloo_group=DistributedEnv.gloo_group,
            device=device,
        )
        evaluator = PyNcclEvaluator(
            comm=comm,
            nccl_group=DistributedEnv.nccl_group,
            local_rank=local_rank,
            world_size=world_size,
        )

        element_counts = [
            8 * 1024,                # 16 KiB
            128 * 1024,              # 256 KiB
            2 * 1024 * 1024,         # 4 MiB
            8 * 1024 * 1024,         # 16 MiB
            32 * 1024 * 1024,        # 64 MiB
        ]

        cases = [
            PyNcclCase(numel=n, dtype=torch.float16)
            for n in element_counts
        ]

        for case in cases:
            if not evaluator.CheckCorrectness(case):
                correctness = "FAIL"
                break

        if correctness == "PASS":
            for case in cases:
                m = evaluator.Benchmark(case, warmup_iters=10, iters=50)
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
