#!/usr/bin/env python3
"""
empty_vllm_agrs_moe_all_to_all.py

Reference implementation and benchmark for the all-gather + reduce-scatter
"naive" MoE all-to-all primitive that vLLM's AgRsAll2AllManager wraps.

Background:
    vLLM's AgRsAll2AllManager (vllm/distributed/device_communicators/all2all.py)
    implements MoE expert dispatch as an all-gather over per-rank token
    chunks of varying size (each rank may carry a different number of
    routed tokens).  Combine is the symmetric reduce-scatter back.  The
    underlying NCCL primitive is the *variable-size* all-gather and
    reduce-scatter exposed on vLLM's PyNcclCommunicator
    (`all_gatherv` / `reduce_scatterv`) — they're implemented as a single
    NCCL group of N broadcasts / N reduces with grouped semantics.

Why we wrap pynccl directly (not AgRsAll2AllManager):
    AgRsAll2AllManager.dispatch() reads `get_forward_context().dp_metadata`
    and `get_dp_group()`, both of which require a full vLLM config +
    parallel-state initialization.  The interesting thing being
    benchmarked here is the *kernel-level dispatch/combine pattern*, and
    that pattern is exactly `PyNcclCommunicator.{all_gatherv,
    reduce_scatterv}` — same NCCL calls, same wire format, no
    forward-context plumbing.

Hint for the model:
- The relevant vLLM class is
  `vllm.distributed.device_communicators.pynccl.PyNcclCommunicator`.
- It must be bound to a non-NCCL process group (Gloo is wired up below).
- Use `all_gatherv(output, input, sizes, stream=...)` for Dispatch and
  `reduce_scatterv(output, input, sizes, stream=...)` for Combine; both
  must run on the wrapper's dedicated stream.
- Cleanup uses `destroy()`.
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
      * the default (NCCL) process group used as both the correctness
        oracle and as the device group for vLLM's PyNcclCommunicator
        (which wants to attach to a non-NCCL group), and
      * a parallel Gloo subgroup that the PyNcclCommunicator binds to.
    """

    nccl_group = None
    gloo_group = None

    @staticmethod
    def Init() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Single-node only.
        assert world_size == local_world_size, "multi-node runs are not supported"
        assert rank == local_rank, "multi-node runs are not supported"
        assert world_size >= 2, "MoE all-to-all needs at least 2 ranks"

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

class VllmAgRsAll2All:
    """
    Wrapper around vLLM's PyNcclCommunicator that exposes a Dispatch /
    Combine pair matching the AgRsAll2AllManager contract.

    `Dispatch(local_tokens, sizes)`:
        local_tokens has shape [sizes[rank], hidden]; the returned
        global_tokens has shape [sum(sizes), hidden] with rank `r`'s
        contribution at rows sum(sizes[:r]) .. sum(sizes[:r+1]).

    `Combine(global_tokens, sizes)`:
        global_tokens has shape [sum(sizes), hidden]; each rank's local
        view sums all ranks' contributions per-row, then keeps the
        slice it owns: returned tensor has shape [sizes[rank], hidden].

    All NCCL work is queued on a dedicated non-default torch.cuda.Stream
    so the benchmark measures only the queued kernel time.
    """

    def __init__(
        self,
        gloo_group: dist.ProcessGroup,
        device: torch.device,
    ):
        # TODO: import PyNcclCommunicator and instantiate it on the
        # supplied Gloo group + device.  Raise RuntimeError if it
        # reports `disabled`.  Cache:
        #   - self.world_size, self.rank from the impl
        #   - self.device
        #   - self.stream = torch.cuda.Stream(device=device)
        self._impl = None
        self.device = device
        self.world_size = None
        self.rank = None
        self.stream = None
        raise NotImplementedError

    def Dispatch(
        self,
        local_tokens: torch.Tensor,
        sizes: List[int],
    ) -> torch.Tensor:
        """All-gather variable-sized token chunks into a global tensor."""
        # TODO: allocate an output tensor of shape (sum(sizes), *trailing).
        # IMPORTANT: have self.stream wait_stream(default) before issuing
        # NCCL — otherwise NCCL on self.stream can race with default-stream
        # producers of `local_tokens` for large inputs and read garbage.
        # Then call self._impl.all_gatherv(out, local_tokens, sizes,
        # stream=self.stream).  Return out.
        raise NotImplementedError

    def Combine(
        self,
        global_tokens: torch.Tensor,
        sizes: List[int],
    ) -> torch.Tensor:
        """Reduce-scatter the global tensor back to per-rank chunks."""
        # TODO: allocate an output of shape (sizes[rank], *trailing).
        # As in Dispatch, have self.stream wait_stream(default) before
        # invoking NCCL.  Then call self._impl.reduce_scatterv(out,
        # global_tokens, sizes, stream=self.stream).  Return out.
        raise NotImplementedError

    def Sync(self) -> None:
        # TODO: synchronize on self.stream.
        raise NotImplementedError

    def Close(self) -> None:
        # TODO: call self._impl.destroy() and clear the handle.
        raise NotImplementedError


# ============================================================
# Evaluation
# ============================================================

@dataclass
class MoeAll2AllCase:
    mean_tokens_per_rank: int
    hidden: int
    dtype: torch.dtype = torch.float16


def _imbalanced_sizes(world_size: int, mean: int) -> List[int]:
    """Return per-rank token counts whose mean is `mean`, with deterministic
    +/- 25% imbalance to mimic real MoE routing skew.

    The pattern repeats every 4 ranks: [+25%, ±0, -25%, ±0].  Any leftover
    tokens are folded into rank 0 so the total stays exactly W * mean.
    """
    skew = [1.25, 1.0, 0.75, 1.0]
    sizes = [int(round(mean * skew[i % 4])) for i in range(world_size)]
    diff = world_size * mean - sum(sizes)
    sizes[0] += diff
    return sizes


class MoeAll2AllEvaluator:
    """Runs correctness check and CUDA-event-timed benchmark for one case."""

    def __init__(
        self,
        comm: VllmAgRsAll2All,
        nccl_group: dist.ProcessGroup,
        local_rank: int,
        world_size: int,
    ):
        self.comm = comm
        self.nccl_group = nccl_group
        self.device = torch.device(f"cuda:{local_rank}")
        self.world_size = world_size
        self.rank = comm.rank

    # --------------------------------------------------------
    # Tensor construction
    # --------------------------------------------------------
    def _make_local_tokens(
        self, rank: int, n_rows: int, hidden: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Deterministic per-rank tokens — every rank can regenerate every
        other rank's tokens for oracle construction.  Stored as small
        ints in fp16 so values are exactly representable."""
        row_base = (torch.arange(n_rows, device=self.device) % 16).unsqueeze(1)
        col_idx = torch.arange(hidden, device=self.device).unsqueeze(0)
        val = (rank * 32 + row_base + (col_idx % 4)).to(dtype)
        return val

    def _make_global_combine_input(
        self, rank: int, total_rows: int, hidden: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Per-rank contribution to the combine input.  Each rank fills a
        full [total_rows, hidden] tensor with values keyed by its own
        rank, so the reduce-scatter sum across ranks is predictable."""
        row_idx = torch.arange(total_rows, device=self.device).unsqueeze(1)
        col_idx = torch.arange(hidden, device=self.device).unsqueeze(0)
        val = (rank + 1) * (1 + (row_idx % 8) + (col_idx % 4))
        return val.to(dtype)

    # --------------------------------------------------------
    # Correctness — Dispatch
    # --------------------------------------------------------
    def CheckDispatch(self, case: MoeAll2AllCase, sizes: List[int]) -> bool:
        local = self._make_local_tokens(
            self.rank, sizes[self.rank], case.hidden, case.dtype
        )
        out = self.comm.Dispatch(local, sizes)
        self.comm.Sync()

        expected_chunks = [
            self._make_local_tokens(r, sizes[r], case.hidden, case.dtype)
            for r in range(self.world_size)
        ]
        expected = torch.cat(expected_chunks, dim=0)
        return bool(torch.equal(out, expected))

    # --------------------------------------------------------
    # Correctness — Combine
    # --------------------------------------------------------
    def CheckCombine(self, case: MoeAll2AllCase, sizes: List[int]) -> bool:
        total = sum(sizes)
        full_in = self._make_global_combine_input(
            self.rank, total, case.hidden, case.dtype
        )
        out = self.comm.Combine(full_in, sizes)
        self.comm.Sync()

        oracle_full = full_in.clone()
        dist.all_reduce(oracle_full, op=dist.ReduceOp.SUM, group=self.nccl_group)
        start = sum(sizes[: self.rank])
        end = start + sizes[self.rank]
        return bool(torch.equal(out, oracle_full[start:end]))

    # --------------------------------------------------------
    # Benchmark — dispatch + combine round-trip on the comm stream
    # --------------------------------------------------------
    def Benchmark(
        self,
        case: MoeAll2AllCase,
        sizes: List[int],
        warmup_iters: int,
        iters: int,
    ) -> Dict[str, float]:
        local = self._make_local_tokens(
            self.rank, sizes[self.rank], case.hidden, case.dtype
        )

        for _ in range(warmup_iters):
            global_tokens = self.comm.Dispatch(local, sizes)
            _ = self.comm.Combine(global_tokens, sizes)
        self.comm.Sync()
        dist.barrier(group=self.nccl_group)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record(self.comm.stream)
        for _ in range(iters):
            global_tokens = self.comm.Dispatch(local, sizes)
            _ = self.comm.Combine(global_tokens, sizes)
        end_evt.record(self.comm.stream)
        self.comm.Sync()

        elapsed_ms = start_evt.elapsed_time(end_evt) / iters
        latency_us = elapsed_ms * 1e3

        per_rank_bytes = case.mean_tokens_per_rank * case.hidden * local.element_size()
        size_mb = per_rank_bytes / (1024.0 * 1024.0)

        rest_rows = sum(sizes) - sizes[self.rank]
        bytes_moved = 2 * rest_rows * case.hidden * local.element_size()
        throughput_gbps = (bytes_moved * 8.0) / (elapsed_ms * 1e-3) / 1e9

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
    device = torch.device(f"cuda:{local_rank}")

    correctness = "PASS"
    metrics: List[Dict[str, float]] = []

    comm: Optional[VllmAgRsAll2All] = None
    try:
        comm = VllmAgRsAll2All(
            gloo_group=DistributedEnv.gloo_group,
            device=device,
        )
        evaluator = MoeAll2AllEvaluator(
            comm=comm,
            nccl_group=DistributedEnv.nccl_group,
            local_rank=local_rank,
            world_size=world_size,
        )

        cases = [
            MoeAll2AllCase(mean_tokens_per_rank=64,   hidden=4096),
            MoeAll2AllCase(mean_tokens_per_rank=256,  hidden=4096),
            MoeAll2AllCase(mean_tokens_per_rank=1024, hidden=4096),
            MoeAll2AllCase(mean_tokens_per_rank=4096, hidden=4096),
        ]

        sized_cases: List[Tuple[MoeAll2AllCase, List[int]]] = [
            (c, _imbalanced_sizes(world_size, c.mean_tokens_per_rank))
            for c in cases
        ]

        for case, sizes in sized_cases:
            if not evaluator.CheckDispatch(case, sizes):
                correctness = "FAIL"
                break
            if not evaluator.CheckCombine(case, sizes):
                correctness = "FAIL"
                break

        if correctness == "PASS":
            for case, sizes in sized_cases:
                m = evaluator.Benchmark(case, sizes, warmup_iters=5, iters=30)
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
