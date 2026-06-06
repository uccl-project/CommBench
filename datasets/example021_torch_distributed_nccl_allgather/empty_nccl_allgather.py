#!/usr/bin/env python3
"""
Reference implementation and benchmark for NCCL AllGather along the last tensor dimension.

This file provides:
1) A clean, class-based implementation of NCCL AllGather
2) Correctness verification
3) Performance benchmarking
4) A single JSON result printed to stdout (rank 0 only)

The design separates core functionality from testing and benchmarking logic.
"""

import os
import json
from time import perf_counter
from typing import List, Dict

import torch
torch.set_printoptions(sci_mode=False)


class DistributedEnv:
    @staticmethod
    def init():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        assert world_size == local_world_size, "Multi-node runs are not supported"
        assert rank == local_rank, "Multi-node runs are not supported"

        torch.distributed.init_process_group(
            backend="nccl",
            device_id=local_rank,
            rank=rank,
            world_size=world_size
        )

        torch.cuda.set_device(local_rank)
        torch.manual_seed(local_rank)

        return local_rank, local_world_size

    @staticmethod
    def destroy():
        torch.distributed.destroy_process_group()

class NcclAllGather:
    """
    Core NCCL AllGather implementation.
    No benchmarking or testing logic is embedded here.
    """

    @staticmethod
    def all_gather_last_dim(
        output: torch.Tensor,
        input: torch.Tensor,
        world_size: int
    ) -> None:
         # TODO



class AllGatherBenchmark:
    def __init__(self, local_rank: int, world_size: int):
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")

    def _benchmark(self, fn, warmup: int, iters: int) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / iters  # ms

    def run_single_test(
        self,
        N: int,
        warmup: int,
        iters: int
    ) -> Dict:
        input_tensor = torch.randn(
            (N, N // self.world_size),
            device=self.device,
            dtype=torch.bfloat16
        )

        output_tensor = torch.empty(
            (N, N),
            device=self.device,
            dtype=torch.bfloat16
        )

        torch.distributed.barrier()

        def run():
            NcclAllGather.all_gather_last_dim(
                output_tensor, input_tensor, self.world_size
            )

        latency_ms = self._benchmark(run, warmup, iters)
        latency_us = latency_ms * 1000

        # Effective data size per rank (MB)
        bytes_moved = input_tensor.numel() * input_tensor.element_size() * self.world_size
        data_size_mb = bytes_moved / (1024 * 1024)

        throughput_gbps = (bytes_moved * 8) / (latency_ms * 1e6)

        return {
            "data_size": int(data_size_mb),
            "throughput_avg": round(throughput_gbps, 3),
            "latency_avg": round(latency_us, 3)
        }

    def check_correctness(self, N: int) -> bool:
        input_tensor = torch.randn(
            (N, N // self.world_size),
            device=self.device,
            dtype=torch.float32
        )

        output_nccl = torch.empty(
            (N, N),
            device=self.device,
            dtype=torch.float32
        )

        NcclAllGather.all_gather_last_dim(
            output_nccl, input_tensor, self.world_size
        )

        # Reference: explicit gather
        gathered = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, input_tensor)
        ref = torch.cat(gathered, dim=-1)

        return torch.allclose(output_nccl, ref, atol=1e-5, rtol=1e-5)


def runTest():
    local_rank, world_size = DistributedEnv.init()
    bench = AllGatherBenchmark(local_rank, world_size)

    test_sizes =[1024, 4096, 8192, 16384, 32768, 65536]
    metrics: List[Dict] = []

    correctness = True
    for N in test_sizes:
        correctness &= bench.check_correctness(N)

    for N in test_sizes:
        metrics.append(
            bench.run_single_test(
                N=N,
                warmup=1,
                iters=5
            )
        )

    if local_rank == 0:
        result = {
            "Correctness": "PASS" if correctness else "FAIL",
            "data_size_unit": "MB",
            "throughput_unit": "Gbps",
            "latency_unit": "us",
            "metrics": metrics
        }
        print(json.dumps(result, indent=2))

    DistributedEnv.destroy()


if __name__ == "__main__":
    runTest()