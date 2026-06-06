#!/usr/bin/env python3
"""
empty_vllm_custom_all_reduce.py

Reference implementation and benchmark for vLLM's custom NVLink-P2P
all-reduce (vllm.distributed.device_communicators.custom_all_reduce.
CustomAllreduce).

Purpose:
- Demonstrate the production-quality "small-message TP all-reduce" path that
  vLLM uses to bypass NCCL on intra-node NVLink fabrics.
- Verify correctness against torch.distributed.all_reduce (NCCL) on the same
  process group.
- Benchmark latency and bus bandwidth across message sizes that fit inside
  the custom-AR registered IPC buffer (default 8 MiB).
- Print exactly one JSON object to stdout (rank 0 only).

Design:
- Core implementation lives in `VllmCustomAllReduce`, a thin wrapper that
  owns the `CustomAllreduce` handle and exposes a single `Run(input, out)`
  entry point matching the vLLM API.
- Evaluation logic is isolated in `CustomAllReduceEvaluator`.
- runTest(...) sweeps a fixed set of message sizes and emits the schema
  expected by the dataset framework.

Launch:
    torchrun --nproc_per_node=2 empty_vllm_custom_all_reduce.py

Hint for the model:
- The relevant vLLM class is
  `vllm.distributed.device_communicators.custom_all_reduce.CustomAllreduce`.
- It must be bound to a non-NCCL process group (Gloo is wired up below).
- Use `all_reduce(input, out=..., registered=False)` for the eager path that
  copies into the pre-registered IPC buffer.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist


# ============================================================
# Distributed Environment
# ============================================================

class DistributedEnv:
    """
    Initializes:
      * the default (NCCL) process group used as the correctness oracle, and
      * a parallel Gloo subgroup that the vLLM CustomAllreduce binds to for
        IPC handle exchange (CustomAllreduce explicitly forbids NCCL groups).
    """

    nccl_group = None
    gloo_group = None

    @staticmethod
    def Init() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Single-node only (CustomAllreduce is intra-node by construction).
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
        # Parallel Gloo group across the same ranks for the CustomAllreduce
        # control plane (IPC handle exchange + graph-buffer broadcast).
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

class VllmCustomAllReduce:
    """
    Wrapper around vLLM's CustomAllreduce that exposes a minimal
    `Run(input, out)` API for benchmarking.

    The constructor is responsible for IPC buffer allocation and rank-to-rank
    handle exchange via the supplied Gloo group.  After Run(...) the output
    tensor holds the reduced (sum) values across all ranks.

    NOTE: CustomAllreduce performs an out-of-place reduce.  The caller owns
    both input and output tensors.
    """

    def __init__(
        self,
        gloo_group: dist.ProcessGroup,
        device: torch.device,
        max_size: int = 8 * 1024 * 1024,
    ):
        # TODO: instantiate the vLLM CustomAllreduce on `gloo_group`/`device`
        # with the requested `max_size`.  Raise RuntimeError if the resulting
        # communicator reports `disabled` (e.g. unsupported world size or
        # missing P2P).  Cache the realized max_size on `self.max_size`.
        raise NotImplementedError

    def Run(self, input: torch.Tensor, out: torch.Tensor) -> None:
        """
        Out-of-place all-reduce sum across the bound process group.
        Falls back via the wrapper's internal eager copy when the input is
        not pre-registered (which is the path benchmarked here).
        """
        # TODO
        raise NotImplementedError

    def ShouldCustom(self, input: torch.Tensor) -> bool:
        """Mirrors vLLM's runtime gate: True iff custom path is eligible."""
        # TODO
        raise NotImplementedError

    def Close(self) -> None:
        # TODO: release IPC buffers / dispose the underlying handle.
        raise NotImplementedError


# ============================================================
# Evaluation
# ============================================================

@dataclass
class CustomAllReduceCase:
    numel: int                      # number of fp16 elements
    dtype: torch.dtype = torch.float16


class CustomAllReduceEvaluator:
    """Runs correctness check and CUDA-event-timed benchmark for a single case."""

    def __init__(
        self,
        comm: VllmCustomAllReduce,
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
    def _build_input(self, case: CustomAllReduceCase) -> torch.Tensor:
        # Use small integers so the reduce result lands well inside fp16
        # representable range (no overflow for world_size <= 8).
        return torch.randint(
            1, 8, (case.numel,), dtype=case.dtype, device=self.device
        )

    # --------------------------------------------------------
    # Correctness — compare against NCCL allreduce
    # --------------------------------------------------------
    def CheckCorrectness(self, case: CustomAllReduceCase) -> bool:
        inp = self._build_input(case)

        # Reference: in-place NCCL all-reduce on a clone.
        ref = inp.clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=self.nccl_group)
        torch.cuda.synchronize(self.device)

        # Custom: out-of-place into a fresh tensor.
        out = torch.empty_like(inp)
        self.comm.Run(inp, out)
        torch.cuda.synchronize(self.device)

        # Inputs are integral; result must match exactly.
        return bool(torch.equal(out, ref))

    # --------------------------------------------------------
    # Benchmark — CUDA-event timing, averaged
    # --------------------------------------------------------
    def Benchmark(
        self,
        case: CustomAllReduceCase,
        warmup_iters: int,
        iters: int,
    ) -> Dict[str, float]:
        inp = self._build_input(case)
        out = torch.empty_like(inp)

        # Warmup
        for _ in range(warmup_iters):
            self.comm.Run(inp, out)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.nccl_group)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        for _ in range(iters):
            self.comm.Run(inp, out)
        end_evt.record()
        torch.cuda.synchronize(self.device)

        elapsed_ms = start_evt.elapsed_time(end_evt) / iters
        latency_us = elapsed_ms * 1e3

        # Bus bandwidth for ring all-reduce: 2*(W-1)/W * size_bytes / time.
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

    comm = None
    try:
        comm = VllmCustomAllReduce(
            gloo_group=DistributedEnv.gloo_group,
            device=device,
        )
        evaluator = CustomAllReduceEvaluator(
            comm=comm,
            nccl_group=DistributedEnv.nccl_group,
            local_rank=local_rank,
            world_size=world_size,
        )

        # Element counts are kept under the 8 MiB IPC buffer max
        # (8 MiB / 2 bytes = 4 Mi fp16 elements).  Each value is also a
        # multiple of 8 so the byte count is a multiple of 16 (a hard
        # requirement of the kernel).
        element_counts = [
            2 * 1024,           # 4 KiB
            16 * 1024,          # 32 KiB
            128 * 1024,         # 256 KiB
            1024 * 1024,        # 2 MiB
            3 * 1024 * 1024,    # 6 MiB (under the 8 MiB max_size)
        ]

        cases = [
            CustomAllReduceCase(numel=n, dtype=torch.float16)
            for n in element_counts
        ]

        # Correctness
        for case in cases:
            if not evaluator.CheckCorrectness(case):
                correctness = "FAIL"
                break

        # Benchmark (only if correctness passed — emit empty metrics on FAIL).
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
