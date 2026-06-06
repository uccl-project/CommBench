#!/usr/bin/env python3
"""
ref_vllm_symm_mem_all_reduce.py

Reference implementation and benchmark for vLLM's CUDA symmetric-memory
all-reduce (vllm.distributed.device_communicators.symm_mem.
SymmMemCommunicator).

Purpose:
- Demonstrate the CUDA-native small-/medium-message TP all-reduce path
  that vLLM ships for NVIDIA SM90/SM100/SM103 hosts. The kernel uses
  torch.distributed._symmetric_memory + multimem (when the world size
  matches the device's multimem table) or the two-shot fallback otherwise.
- Verify correctness against torch.distributed.all_reduce on the same
  process group.
- Benchmark latency and bus bandwidth across a sweep of message sizes
  that fit under SymmMemCommunicator's per-(capability, world_size)
  buffer limit (see SYMM_MEM_ALL_REDUCE_MAX_SIZES).
- Print exactly one JSON object to stdout (rank 0 only).

Hardware floor:
    SymmMemCommunicator initializes only when:
      - the host has CUDA + a recent torch with
        torch.distributed._symmetric_memory,
      - the device capability appears in
        SYMM_MEM_ALL_REDUCE_MAX_SIZES (9.0, 10.0, 10.3),
      - the rendezvous handle reports multicast support.
    If any of those gates fails, runTest() emits a Correctness="SKIPPED"
    payload and exits cleanly.

Design:
- Core implementation lives in `VllmSymmMemAllReduce`, a thin wrapper
  that owns the SymmMemCommunicator handle and exposes `Run(input, out)`.
- Evaluation logic is isolated in `SymmMemAllReduceEvaluator`.
- runTest(...) sweeps a fixed set of message sizes and emits the schema
  expected by the dataset framework.

Launch:
    torchrun --nproc_per_node=2 ref_vllm_symm_mem_all_reduce.py
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
    Initializes the default NCCL process group used by SymmMemCommunicator
    for rendezvous AND a separate NCCL subgroup used as the correctness
    oracle (so the reference all-reduce does not collide with the
    accelerated one on the same group).
    """

    nccl_group = None
    ref_group = None

    @staticmethod
    def Init() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

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
        DistributedEnv.ref_group = dist.new_group(
            ranks=list(range(world_size)), backend="nccl"
        )
        torch.manual_seed(1234 + local_rank)
        return local_rank, world_size

    @staticmethod
    def Destroy() -> None:
        if DistributedEnv.ref_group is not None:
            dist.destroy_process_group(DistributedEnv.ref_group)
            DistributedEnv.ref_group = None
        dist.destroy_process_group()

    @staticmethod
    def Barrier() -> None:
        dist.barrier()


# ============================================================
# Core Implementation (no test/benchmark logic here)
# ============================================================

class VllmSymmMemAllReduce:
    """
    Wrapper around vLLM's SymmMemCommunicator that exposes a minimal
    `Run(input, out)` API for benchmarking.

    Construction is allowed to soft-fail: if the underlying
    SymmMemCommunicator reports `disabled=True` (wrong platform, wrong
    capability, unsupported world size, multicast not available, etc.),
    the wrapper records a `skip_reason` and `Run(...)` is unsupported.
    Callers should check `disabled` before using the wrapper.
    """

    DTYPE = torch.bfloat16  # SymmMemCommunicator pins the buffer dtype.

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
    ):
        self._impl = None
        self.disabled: bool = True
        self.skip_reason: Optional[str] = None
        self.max_size: Optional[int] = None
        self.algorithm: Optional[str] = None  # "multimem" or "two_shot"
        # TODO
        raise NotImplementedError

    def Run(self, input: torch.Tensor, out: torch.Tensor) -> None:
        """Symmetric-memory all-reduce sum across the bound process group."""
        assert not self.disabled, "VllmSymmMemAllReduce is disabled — see skip_reason"
        # TODO
        raise NotImplementedError

    def ShouldUse(self, input: torch.Tensor) -> bool:
        """Mirrors SymmMemCommunicator's runtime eligibility gate."""
        if self.disabled:
            return False
        # TODO
        raise NotImplementedError

    def Close(self) -> None:
        # TODO
        self._impl = None
        self.disabled = True


# ============================================================
# Evaluation
# ============================================================

@dataclass
class SymmMemAllReduceCase:
    numel: int                          # number of elements
    dtype: torch.dtype = torch.bfloat16


class SymmMemAllReduceEvaluator:
    """Runs correctness check and CUDA-event-timed benchmark for a single case."""

    # bfloat16 has only ~8 mantissa bits; absorb per-rank rounding plus
    # any reduction-order divergence between NCCL and the symm_mem kernels.
    _TOL = {"atol": 5e-2, "rtol": 5e-2}

    def __init__(
        self,
        comm: VllmSymmMemAllReduce,
        ref_group: dist.ProcessGroup,
        local_rank: int,
        world_size: int,
    ):
        self.comm = comm
        self.ref_group = ref_group
        self.device = torch.device(f"cuda:{local_rank}")
        self.world_size = world_size

    # --------------------------------------------------------
    # Tensor construction
    # --------------------------------------------------------
    def _build_input(self, case: SymmMemAllReduceCase) -> torch.Tensor:
        return torch.randn(case.numel, dtype=case.dtype, device=self.device) * 0.1

    # --------------------------------------------------------
    # Correctness — compare against a separate NCCL allreduce
    # --------------------------------------------------------
    def CheckCorrectness(self, case: SymmMemAllReduceCase) -> bool:
        inp = self._build_input(case)

        ref = inp.clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=self.ref_group)
        torch.cuda.synchronize(self.device)

        out = torch.empty_like(inp)
        self.comm.Run(inp, out)
        torch.cuda.synchronize(self.device)

        return bool(torch.allclose(out.float(), ref.float(), **self._TOL))

    # --------------------------------------------------------
    # Benchmark — CUDA-event timing, averaged
    # --------------------------------------------------------
    def Benchmark(
        self,
        case: SymmMemAllReduceCase,
        warmup_iters: int,
        iters: int,
    ) -> Dict[str, float]:
        inp = self._build_input(case)
        out = torch.empty_like(inp)

        for _ in range(warmup_iters):
            self.comm.Run(inp, out)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.ref_group)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        for _ in range(iters):
            self.comm.Run(inp, out)
        end_evt.record()
        torch.cuda.synchronize(self.device)

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

def _emit(rank: int, payload: Dict) -> None:
    if rank == 0:
        print(json.dumps(payload, indent=2))


def _select_element_counts(cap_bytes: int) -> List[int]:
    """Pick a bfloat16 element-count sweep that fits comfortably under the
    SymmMemCommunicator buffer cap. bfloat16 is 2 bytes per element; the
    `should_use_symm_mem` gate requires the byte size to be strictly less
    than max_size, so leave headroom."""
    candidate_counts = [
        128 * 1024,       # 256 KiB
        256 * 1024,       # 512 KiB
        512 * 1024,       # 1 MiB
        1024 * 1024,      # 2 MiB
        1536 * 1024,      # 3 MiB
    ]
    counts = [n for n in candidate_counts if n * 2 < cap_bytes]
    if len(counts) < 2:
        # Buffer cap is tiny; fall back to two scaled fractions of it.
        per_elem = 2
        counts = [
            max(per_elem, (cap_bytes // 4) // per_elem * per_elem // per_elem),
            max(per_elem, (cap_bytes // 2) // per_elem * per_elem // per_elem),
        ]
    return counts


def runTest():
    local_rank, world_size = DistributedEnv.Init()
    rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    comm: Optional[VllmSymmMemAllReduce] = None
    payload: Dict
    try:
        comm = VllmSymmMemAllReduce(
            group=DistributedEnv.nccl_group,
            device=device,
        )

        if comm.disabled:
            payload = {
                "Correctness": "SKIPPED",
                "skip_reason": comm.skip_reason or "SymmMemCommunicator disabled",
            }
            return payload

        evaluator = SymmMemAllReduceEvaluator(
            comm=comm,
            ref_group=DistributedEnv.ref_group,
            local_rank=local_rank,
            world_size=world_size,
        )

        cap_bytes = comm.max_size or (4 * 1024 * 1024)
        element_counts = _select_element_counts(cap_bytes)
        cases = [
            SymmMemAllReduceCase(numel=n, dtype=torch.bfloat16)
            for n in element_counts
        ]

        correctness = "PASS"
        for case in cases:
            if not evaluator.CheckCorrectness(case):
                correctness = "FAIL"
                break

        metrics: List[Dict[str, float]] = []
        if correctness == "PASS":
            for case in cases:
                m = evaluator.Benchmark(case, warmup_iters=10, iters=50)
                metrics.append({k: round(v, 4) for k, v in m.items()})

        payload = {
            "Correctness": correctness,
            "algorithm": comm.algorithm,
            "data_size_unit": "MB",
            "throughput_unit": "Gbps",
            "latency_unit": "us",
            "metrics": metrics,
        }
        return payload

    finally:
        if comm is not None:
            comm.Close()
        DistributedEnv.Barrier()
        DistributedEnv.Destroy()


if __name__ == "__main__":
    out = runTest()
    _emit(int(os.environ.get("RANK", 0)), out)
