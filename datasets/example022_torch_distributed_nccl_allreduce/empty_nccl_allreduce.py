#!/usr/bin/env python3
"""
Reference implementation and benchmark for NCCL AllReduce (SUM).

This file provides:
1. A clean, class-based NCCL AllReduce implementation
2. Correctness verification
3. Performance benchmarking over multiple data sizes
4. Exactly one JSON object printed to stdout (rank 0 only)

All implementation logic is separated from evaluation logic.
"""

import os
import json
from typing import List, Dict

import torch

class DistributedEnvironment:
    @staticmethod
    def Init():
        #TODO

    @staticmethod
    def Destroy():
        torch.distributed.destroy_process_group()


# ============================================================
# Core NCCL AllReduce Implementation
# ============================================================

class NcclAllReduce:
    """
    Core NCCL AllReduce (SUM) implementation.
    No correctness or benchmarking logic is embedded here.
    """

    @staticmethod
    def Run(tensor: torch.Tensor) -> None:
       #TODO: Implement NCCL AllReduce using torch.distributed


# ============================================================
# Evaluation: Correctness + Performance
# ============================================================

class AllReduceEvaluator:
    def __init__(self, local_rank: int, world_size: int):
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")

    def CheckCorrectness(self, N: int) -> bool:
        """
        Correctness check:
        After all-reduce(sum), tensor should equal original * world_size.
        """
        # Seed identically across ranks so every rank starts from the same x;
        # otherwise allreduce(sum) != x * world_size.
        gen = torch.Generator(device=self.device).manual_seed(N)
        x = torch.randn(
            (N, N),
            generator=gen,
            device=self.device,
            dtype=torch.float32
        )
        ref = x * self.world_size

        NcclAllReduce.Run(x)

        return torch.allclose(x, ref, atol=1e-5, rtol=1e-5)

    def Benchmark(
        self,
        N: int,
        warmup_iters: int,
        iters: int
    ) -> Dict:
        """
        Measure latency and effective bandwidth.
        """
        tensor = torch.randn(
            (N, N),
            device=self.device,
            dtype=torch.bfloat16
        )

        # Warmup
        for _ in range(warmup_iters):
            NcclAllReduce.Run(tensor)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            NcclAllReduce.Run(tensor)
        end.record()
        torch.cuda.synchronize()

        latency_ms = start.elapsed_time(end) / iters
        latency_us = latency_ms * 1000.0

        # Ring all-reduce traffic model
        bytes_tensor = tensor.numel() * tensor.element_size()
        bytes_per_rank = (
            bytes_tensor * 2.0 * (self.world_size - 1) / self.world_size
        )

        throughput_gbps = (
            bytes_per_rank * 8.0
        ) / (latency_ms * 1e6)

        data_size_mb = bytes_tensor / (1024 * 1024)

        return {
            "data_size": int(data_size_mb),
            "throughput_avg": round(throughput_gbps, 3),
            "latency_avg": round(latency_us, 3),
        }


# ============================================================
# Unified Test Entry
# ============================================================

def runTest():
    local_rank, world_size = DistributedEnvironment.Init()
    evaluator = AllReduceEvaluator(local_rank, world_size)

    test_sizes = [2048, 4096, 8192, 16384, 32768]
    metrics: List[Dict] = []

    correctness = True
    for N in test_sizes:
        correctness &= evaluator.CheckCorrectness(N)

    for N in test_sizes:
        metrics.append(
            evaluator.Benchmark(
                N=N,
                warmup_iters=1,
                iters=5
            )
        )

    if local_rank == 0:
        result = {
            "Correctness": "PASS" if correctness else "FAIL",
            "data_size_unit": "MB",
            "throughput_unit": "Gbps",
            "latency_unit": "us",
            "metrics": metrics,
        }
        print(json.dumps(result, indent=2))

    DistributedEnvironment.Destroy()


if __name__ == "__main__":
    runTest()