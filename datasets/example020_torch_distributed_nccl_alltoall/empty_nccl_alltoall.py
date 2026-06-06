#!/usr/bin/env python3
"""
ref_nccl_alltoall.py

Reference implementation and benchmark for intra-node NCCL AllToAll using
torch.distributed.all_to_all_single.

Purpose:
- Provide a clean, class-based reference for AllToAll communication patterns
  commonly used in long-context attention tensor reshaping.
- Include correctness tests and performance benchmarks.
- Print exactly one JSON object to stdout (rank 0 only).

Design:
- Core implementation is isolated in NcclAllToAll (no test/bench logic inside).
- Evaluation logic lives in AllToAllEvaluator.
- runTest(...) orchestrates test + benchmark and emits JSON.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


# ============================================================
# Distributed Environment
# ============================================================

class DistributedEnv:
    @staticmethod
    def Init() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Single-node only (as in your original common.py).
        assert world_size == local_world_size, "multi-node runs are not supported"
        assert rank == local_rank, "multi-node runs are not supported"

        torch.distributed.init_process_group(
            backend="nccl",
            device_id=local_rank,
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)
        torch.manual_seed(1234 + local_rank)
        return local_rank, local_world_size

    @staticmethod
    def Destroy() -> None:
        torch.distributed.destroy_process_group()

    @staticmethod
    def Barrier() -> None:
        torch.distributed.barrier()


# ============================================================
# Core Implementation (No test/benchmark logic here)
# ============================================================

class NcclAllToAll:
    """
    Core NCCL AllToAll implementation built on torch.distributed.all_to_all_single.

    Supported patterns (matching your original code):
    1) scatter_idx=2, gather_idx=1
       input:  [B, N_per_rank, H, D]
       output: [B, N, H_per_rank, D] where N=N_per_rank*W, H_per_rank=H/W

    2) scatter_idx=1, gather_idx=2
       input:  [B, N, H_per_rank, D] where N divisible by W, H_per_rank=H/W
       output: [B, N_per_rank, H, D]
    """

    @staticmethod
    def Run(
        output: torch.Tensor,
        input: torch.Tensor,
        world_size: int,
        scatter_idx: int,
        gather_idx: int,
    ) -> None:
        # TODO


# ============================================================
# Benchmark helpers (evaluation side)
# ============================================================

class CudaTimer:
    @staticmethod
    def Benchmark(
        fn,
        warmup_iters: int,
        iters: int,
    ) -> float:
        for _ in range(warmup_iters):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iters
        return avg_ms


@dataclass
class AllToAllCase:
    N: int
    H: int
    D: int
    scatter_axis: int
    gather_axis: int


class AllToAllEvaluator:
    def __init__(self, local_rank: int, world_size: int):
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")

    def _build_tensors(self, case: AllToAllCase) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create input/output tensors with correct shapes for the chosen pattern.
        Use BF16 to match your original benchmark.
        """
        B = 1
        W = self.world_size
        N = case.N
        H = case.H
        D = case.D

        if case.scatter_axis == 2 and case.gather_axis == 1:
            # input:  [B, N/W, H, D]
            # output: [B, N, H/W, D]
            assert N % W == 0
            assert H % W == 0
            x = torch.randn((B, N // W, H, D), device=self.device, dtype=torch.bfloat16)
            y = torch.empty((B, N, H // W, D), device=self.device, dtype=torch.bfloat16)
            return x, y

        if case.scatter_axis == 1 and case.gather_axis == 2:
            # input:  [B, N, H/W, D]
            # output: [B, N/W, H, D]
            assert N % W == 0
            assert H % W == 0
            x = torch.randn((B, N, H // W, D), device=self.device, dtype=torch.bfloat16)
            y = torch.empty((B, N // W, H, D), device=self.device, dtype=torch.bfloat16)
            return x, y

        raise RuntimeError("Unsupported scatter/gather axis")

    def CheckCorrectness(self, case: AllToAllCase) -> bool:
        """
        Meaningful correctness:
        - Run AllToAll
        - Verify output equals reference computed by explicit per-rank packing using all_gather
          is expensive; instead do a strong invariant:
            1) Input is random but we also compute a checksum on GPU before/after inverse pattern.
            2) Apply forward pattern then inverse pattern and verify we recover original input.
        This is non-trivial and catches layout mistakes robustly.
        """
        x, y = self._build_tensors(case)

        # Build the inverse case
        inv = AllToAllCase(
            N=case.N,
            H=case.H,
            D=case.D,
            scatter_axis=case.gather_axis,
            gather_axis=case.scatter_axis,
        )

        # Forward
        NcclAllToAll.Run(y, x, self.world_size, case.scatter_axis, case.gather_axis)

        # Inverse output shape should match x
        x2 = torch.empty_like(x)
        NcclAllToAll.Run(x2, y, self.world_size, inv.scatter_axis, inv.gather_axis)

        # Compare in fp32 for robustness
        return torch.allclose(x.float(), x2.float(), atol=1e-2, rtol=1e-2)

    def Benchmark(self, case: AllToAllCase, warmup_iters: int, iters: int) -> Dict:
        x, y = self._build_tensors(case)

        def run_once():
            NcclAllToAll.Run(y, x, self.world_size, case.scatter_axis, case.gather_axis)

        DistributedEnv.Barrier()
        avg_ms = CudaTimer.Benchmark(run_once, warmup_iters, iters)

        # Communication volume model (same as your original):
        # chunk_size_bytes = (N/W)*(H/W)*D*2 (bf16=2 bytes) and each rank exchanges (W-1) chunks
        W = self.world_size
        N = case.N
        H = case.H
        D = case.D
        bytes_per_elem = 2  # bf16
        chunk_size_bytes = (N // W) * (H // W) * D * bytes_per_elem
        per_rank_comm_bytes = chunk_size_bytes * (W - 1)

        # Throughput in Gbps (bits/sec)
        throughput_gbps = (per_rank_comm_bytes * 8.0) / (avg_ms * 1e6)

        # data_size in MB: use "per-rank communicated bytes" as data size (meaningful and consistent)
        data_size_mb = per_rank_comm_bytes / (1024.0 * 1024.0)

        return {
            "data_size": int(round(data_size_mb)),
            "throughput_avg": round(throughput_gbps, 3),
            "latency_avg": round(avg_ms * 1000.0, 3),  # us
        }


# ============================================================
# runTest entry (single JSON)
# ============================================================

def runTest():
    local_rank, world_size = DistributedEnv.Init()

    # Meaningful cases: multiple N (token length), fixed H,D typical attention dims.
    H = 128
    D = 128
    scatter_axis = 2
    gather_axis = 1

    cases = [
        AllToAllCase(N=16384,  H=H, D=D, scatter_axis=scatter_axis, gather_axis=gather_axis),
        AllToAllCase(N=32768,  H=H, D=D, scatter_axis=scatter_axis, gather_axis=gather_axis),
        AllToAllCase(N=65536,  H=H, D=D, scatter_axis=scatter_axis, gather_axis=gather_axis),
        AllToAllCase(N=131072, H=H, D=D, scatter_axis=scatter_axis, gather_axis=gather_axis),
        AllToAllCase(N=262144, H=H, D=D, scatter_axis=scatter_axis, gather_axis=gather_axis),
    ]

    evaluator = AllToAllEvaluator(local_rank, world_size)

    # Correctness: forward+inverse recovery per case.
    correctness = True
    for case in cases[:3]:  # keep correctness meaningful but bounded cost
        correctness &= evaluator.CheckCorrectness(case)

    # Benchmark
    metrics: List[Dict] = []
    for case in cases:
        metrics.append(evaluator.Benchmark(case, warmup_iters=1, iters=5))

    if local_rank == 0:
        result = {
            "Correctness": "PASS" if correctness else "FAIL",
            "data_size_unit": "MB",
            "throughput_unit": "Gbps",
            "latency_unit": "us",
            "metrics": metrics,
        }
        print(json.dumps(result, indent=2))

    DistributedEnv.Destroy()


if __name__ == "__main__":
    runTest()