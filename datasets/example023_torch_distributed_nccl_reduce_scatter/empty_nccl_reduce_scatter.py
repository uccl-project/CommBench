#!/usr/bin/env python3
"""
ref_nccl_reduce_scatter.py

Purpose:
  Reference implementation of NCCL ReduceScatter (SUM) using torch.distributed.
  This file cleanly separates:
    1) Core ReduceScatter functionality.
    2) Correctness testing and performance benchmarking.
  The program prints EXACTLY ONE JSON object to stdout.
"""

import json
import torch
import torch.distributed as dist


# ==============================================================================
# Core functionality (NO test / benchmark logic here)
# ==============================================================================

class NcclReduceScatter:
    """Core NCCL ReduceScatter(SUM) implementation."""

    def __init__(self, world_size: int):
        self.world_size = world_size

    def run(self, output: torch.Tensor, input: torch.Tensor) -> None:
         # TODO

# ==============================================================================
# Evaluation logic (correctness + performance)
# ==============================================================================

def _cpu_reference_reduce_scatter(
    local_input: torch.Tensor,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    """
    Reference for ReduceScatter:
      - gather all ranks' input (on GPU via NCCL)
      - sum corresponding chunks
      - return the chunk for this rank on CPU
    """
    gathered = [torch.empty_like(local_input) for _ in range(world_size)]
    dist.all_gather(gathered, local_input)

    total = torch.stack(gathered, dim=0).sum(dim=0)

    chunk = local_input.numel() // world_size
    return total[rank * chunk : (rank + 1) * chunk].cpu()


def runTest(
    data_sizes_mb,
    num_warmup_iters: int = 5,
    num_iters: int = 20,
):
    """
    Dedicated correctness + performance test.

    Returns:
      dict that matches the mandatory JSON output schema.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    rs = NcclReduceScatter(world_size)

    correctness_pass = True
    metrics = []

    for size_mb in data_sizes_mb:
        # Each rank contributes size_mb MB * world_size input
        total_bytes = size_mb * 1024 * 1024
        total_elems = total_bytes // 4  # float32

        assert total_elems % world_size == 0
        chunk_elems = total_elems // world_size

        input_tensor = torch.randn(
            total_elems,
            device=device,
            dtype=torch.float32,
        )
        output_tensor = torch.empty(
            chunk_elems,
            device=device,
            dtype=torch.float32,
        )

        # ---------------- correctness ----------------
        rs.run(output_tensor, input_tensor)

        ref = _cpu_reference_reduce_scatter(
            input_tensor, world_size, rank
        )

        if not torch.allclose(
            output_tensor.cpu(), ref, rtol=1e-4, atol=1e-4
        ):
            correctness_pass = False

        # ---------------- warmup ----------------
        for _ in range(num_warmup_iters):
            rs.run(output_tensor, input_tensor)

        torch.cuda.synchronize()
        dist.barrier()

        # ---------------- benchmark ----------------
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        for _ in range(num_iters):
            rs.run(output_tensor, input_tensor)
        end_evt.record()

        torch.cuda.synchronize()
        dist.barrier()

        avg_ms = start_evt.elapsed_time(end_evt) / num_iters
        avg_us = avg_ms * 1000.0

        # Effective throughput:
        # total reduced data = size_mb * world_size MB
        throughput_gbps = (
            total_bytes * world_size * 8
        ) / (avg_ms * 1e6)

        if rank == 0:
            metrics.append({
                "data_size": size_mb,
                "throughput_avg": round(throughput_gbps, 3),
                "latency_avg": round(avg_us, 3),
            })

    result = {
        "Correctness": "PASS" if correctness_pass else "FAIL",
    }

    if metrics:
        result.update({
            "data_size_unit": "MB",
            "throughput_unit": "Gbps",
            "latency_unit": "us",
            "metrics": metrics,
        })

    return result


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    data_sizes_mb = [16, 64, 256, 512]

    result = runTest(
        data_sizes_mb=data_sizes_mb,
        num_warmup_iters=5,
        num_iters=20,
    )

    # EXACTLY one JSON object
    if dist.get_rank() == 0:
        print(json.dumps(result, indent=2))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()