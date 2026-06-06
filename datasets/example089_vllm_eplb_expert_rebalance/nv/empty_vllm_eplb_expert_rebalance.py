#!/usr/bin/env python3
"""
ref_vllm_eplb_expert_rebalance.py

Reference implementation and benchmark for vLLM's expert-parallel
load-balancer (EPLB) weight rearrangement — the cross-rank machinery
that migrates MoE expert tensors when routing-load statistics decide a
new physical→logical assignment.

Background:
    vLLM's `rearrange_expert_weights_inplace`
    (vllm/distributed/eplb/rebalance_execute.py) takes the current
    `old_global_expert_indices` and the desired
    `new_global_expert_indices` and shuffles each layer's expert
    weight tensors so that, after the call, slot E on rank R holds the
    weights for the logical expert id `new_global[layer, R*L + E]`
    (where L is the number of local physical experts per rank).
    Internally it computes per-expert send/recv plans, stages local
    moves through an intermediate buffer, and dispatches the cross-rank
    transfers via an `EplbCommunicator` — we use the simplest backend,
    `TorchDistNcclEplbCommunicator`, which lowers to
    `batch_isend_irecv` over the supplied process group.

Why we wrap the rearrange function directly (not EplbState):
    `EplbState` and the policy layer above it consume vLLM's full
    forward-context state (model config, MoE layer indices, route
    statistics).  The kernel-level pattern this benchmark cares about
    is the rearrange + communicator dispatch — we drive that
    directly with a synthetic permutation plan.

Purpose:
- Demonstrate the production "rearrange MoE expert tensors after a
  rebalance plan" path end-to-end: plan → buffer staging → P2P sends
  → buffer drain.
- Verify each rank's post-rebalance tensors match the deterministic
  recipe pattern for the new logical expert id at every slot.
- Benchmark wall-clock and effective bandwidth across a sweep of
  (num_experts, hidden_dim) sizes.
- Print exactly one JSON object to stdout (rank 0 only).

Launch:
    torchrun --nproc_per_node=4 ref_vllm_eplb_expert_rebalance.py
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


# ============================================================
# Distributed Environment
# ============================================================

class DistributedEnv:
    """
    Initializes the default NCCL world group used as the ep_group for
    `rearrange_expert_weights_inplace`.  The TorchDistNccl EPLB
    backend lowers to `batch_isend_irecv` on this group.
    """

    nccl_group = None

    @staticmethod
    def Init() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Single-node only.  EPLB rebalance can run cross-node, but our
        # rendezvous is purely torchrun-launched.
        assert world_size == local_world_size, "multi-node runs are not supported"
        assert rank == local_rank, "multi-node runs are not supported"
        assert world_size >= 2, (
            "EPLB example requires world_size >= 2; defaults to 4 so "
            "the rotation pattern actually exercises cross-rank sends"
        )

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=local_rank,
            rank=rank,
            world_size=world_size,
        )
        DistributedEnv.nccl_group = dist.group.WORLD
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
class EplbCase:
    """One benchmark case.

    `num_local_experts` is per rank; the global physical-experts count
    is `world_size * num_local_experts`.
    """
    name: str
    num_layers: int
    num_local_experts: int
    hidden_size: int
    weights_per_layer: int = 2  # up + down projection, like a real MoE


class VllmEplbRebalance:
    """
    Wrapper around `rearrange_expert_weights_inplace` driving the
    `TorchDistNcclEplbCommunicator` backend.

    `Allocate(case)`:
        Allocate `expert_weights[layer][kind]` tensors of shape
        `(num_local_experts, hidden_size)`.  Re-callable across cases.

    `FillFromIndices(global_indices)`:
        Fill local weights so slot E on rank R holds the deterministic
        recipe of logical expert id `global_indices[layer, R*L+E]`
        for every (layer, kind, slot, h).  Used both to seed the
        sender's data (with the OLD layout) and to compute the receiver-
        side oracle (with the NEW layout).

    `Rebalance(old_global, new_global)`:
        Drive the production rearrange call.

    The wrapper also exposes the ep_rank and totals so the test can
    construct deterministic plans.
    """

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        device: torch.device,
    ):
        # TODO
        raise NotImplementedError

    def Allocate(self, case: EplbCase, dtype: torch.dtype = torch.bfloat16) -> None:
        # TODO
        raise NotImplementedError

    def FillFromIndices(self, global_indices: torch.Tensor) -> None:
        """Fill local weights to match `global_indices` row-by-row."""
        # TODO
        raise NotImplementedError

    def ExpectedAfter(self, new_global: torch.Tensor) -> List[List[torch.Tensor]]:
        """Build the oracle: what each rank's tensors should look like
        after a rebalance into `new_global`."""
        # TODO
        raise NotImplementedError

    def Rebalance(
        self,
        old_global: torch.Tensor,
        new_global: torch.Tensor,
    ) -> None:
        # TODO
        raise NotImplementedError

    def Sync(self) -> None:
        # TODO
        raise NotImplementedError

    def Close(self) -> None:
        # TODO
        raise NotImplementedError

    # --------------------------------------------------------
    # Recipe — deterministic per (logical_id, layer, kind, hidden)
    # --------------------------------------------------------
    def _recipe_row(
        self, logical_id: int, layer: int, kind: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """One row of length self.hidden_size for the given (id, L, K)."""
        # TODO
        raise NotImplementedError


# ============================================================
# Plan construction
# ============================================================

def _identity_indices(
    num_layers: int, num_global_physical: int, device: torch.device,
) -> torch.Tensor:
    """old_global = identity per layer."""
    row = torch.arange(num_global_physical, dtype=torch.int64, device=device)
    return row.unsqueeze(0).expand(num_layers, -1).contiguous()


def _rotated_indices(
    num_layers: int, num_global_physical: int,
    num_local_experts: int, device: torch.device,
) -> torch.Tensor:
    """new_global = identity rotated by one rank.

    Slot p ends up holding logical_id `(p + L) % G` where L is the
    per-rank local count and G is the global physical count.  Every
    layer rotates by the same amount, which still exercises every
    rank's send + recv path.
    """
    row = torch.arange(num_global_physical, dtype=torch.int64, device=device)
    rotated = torch.roll(row, shifts=num_local_experts, dims=0)
    return rotated.unsqueeze(0).expand(num_layers, -1).contiguous()


# ============================================================
# Evaluation
# ============================================================

class EplbEvaluator:
    """Runs correctness check and wall-clock benchmark for one case."""

    def __init__(
        self,
        comm: VllmEplbRebalance,
        local_rank: int,
        world_size: int,
    ):
        self.comm = comm
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = comm.device

    def _make_plan(
        self, case: EplbCase,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        num_global_physical = self.world_size * case.num_local_experts
        old = _identity_indices(case.num_layers, num_global_physical, self.device)
        new = _rotated_indices(
            case.num_layers, num_global_physical,
            case.num_local_experts, self.device,
        )
        return old, new, num_global_physical

    def CheckCorrectness(self, case: EplbCase) -> bool:
        self.comm.Allocate(case)
        old, new, _ = self._make_plan(case)

        # Seed: each rank fills its local tensors as if the OLD plan
        # were in effect.  After Rebalance, the receiver's data should
        # match the recipe under the NEW plan.
        self.comm.FillFromIndices(old)
        self.comm.Sync()
        dist.barrier()

        # Build the oracle BEFORE rebalance — the recipe is just a
        # function of (logical_id, layer, kind, h), so the answer
        # under `new` is stable regardless of the rebalance result.
        expected = self.comm.ExpectedAfter(new)

        self.comm.Rebalance(old, new)
        self.comm.Sync()
        dist.barrier()

        ok = True
        for layer in range(self.comm.num_layers):
            for kind in range(self.comm.weights_per_layer):
                got = self.comm.expert_weights[layer][kind]
                ref = expected[layer][kind]
                if not torch.equal(got, ref):
                    ok = False
                    break
            if not ok:
                break

        flag = torch.tensor(
            [1 if ok else 0], device=self.device, dtype=torch.int32,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item() == 1)

    def Benchmark(
        self,
        case: EplbCase,
        warmup_iters: int,
        iters: int,
    ) -> Dict[str, float]:
        self.comm.Allocate(case)
        old, new, num_global = self._make_plan(case)

        # Slow Python-level fill once, then snapshot the seeded weights
        # so the benchmark loop can restore in O(weights) tensor copies
        # instead of doing the python-loop fill every iter.
        self.comm.FillFromIndices(old)
        self.comm.Sync()
        snapshot = [
            [w.clone() for w in layer] for layer in self.comm.expert_weights
        ]

        def _restore_from_snapshot() -> None:
            for layer_w, layer_s in zip(self.comm.expert_weights, snapshot):
                for w, s in zip(layer_w, layer_s):
                    w.copy_(s)

        dist.barrier()

        # Warmup — restore + rebalance.  Sync between to avoid stream
        # races between the restore copy and the next rebalance plan.
        for _ in range(warmup_iters):
            _restore_from_snapshot()
            self.comm.Sync()
            self.comm.Rebalance(old, new)
            self.comm.Sync()
        dist.barrier()

        # Timed region — accumulate only the rebalance call's wall-clock.
        # Restore happens outside the timer so the benchmark isolates the
        # rearrange machinery (the production hot path).
        elapsed_ns_total = 0
        for _ in range(iters):
            _restore_from_snapshot()
            self.comm.Sync()
            dist.barrier()
            if self.local_rank == 0:
                t0 = time.perf_counter_ns()
                self.comm.Rebalance(old, new)
                self.comm.Sync()
                elapsed_ns_total += time.perf_counter_ns() - t0
            else:
                self.comm.Rebalance(old, new)
                self.comm.Sync()
        elapsed_ns = elapsed_ns_total

        dist.barrier()

        # Effective bytes moved per rebalance.  Each layer migrates
        # roughly `num_global_physical * weights_per_layer * hidden *
        # itemsize` bytes across the cluster (every slot rotates), so
        # per-rank traffic is `(num_local * weights * hidden * itemsize)
        # * 2` (one send, one recv per slot).  For the BW number we
        # report the per-rank one-sided traffic — matches the plan's
        # rule for send/recv (algorithmic BW, half-duplex).
        per_rank_one_sided_bytes = (
            case.num_layers
            * case.weights_per_layer
            * case.num_local_experts
            * case.hidden_size
            * 2  # bf16 itemsize
        )
        total_data_bytes = (
            case.num_layers
            * case.weights_per_layer
            * num_global
            * case.hidden_size
            * 2
        )

        latency_us = elapsed_ns / iters / 1e3 if elapsed_ns > 0 else 0.0
        size_mb = total_data_bytes / (1024.0 * 1024.0)
        if latency_us > 0:
            throughput_gbps = (
                (per_rank_one_sided_bytes * 8.0) / (latency_us * 1e-6) / 1e9
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
    device = torch.device(f"cuda:{local_rank}")

    correctness = "PASS"
    metrics: List[Dict[str, float]] = []

    comm: Optional[VllmEplbRebalance] = None
    try:
        comm = VllmEplbRebalance(
            ep_group=DistributedEnv.nccl_group,
            device=device,
        )
        evaluator = EplbEvaluator(
            comm=comm,
            local_rank=local_rank,
            world_size=world_size,
        )

        # Sweep dimensions.  num_local_experts = 4 means 16 physical
        # experts globally on a 4-rank run; hidden ramps from 256 →
        # 4096.  Layers fixed at 2 so per-iteration time stays under a
        # second on B300 NVLink.
        cases = [
            EplbCase(
                name="L2_E4_H256",
                num_layers=2, num_local_experts=4, hidden_size=256,
            ),
            EplbCase(
                name="L2_E4_H1024",
                num_layers=2, num_local_experts=4, hidden_size=1024,
            ),
            EplbCase(
                name="L2_E4_H4096",
                num_layers=2, num_local_experts=4, hidden_size=4096,
            ),
            EplbCase(
                name="L4_E8_H4096",
                num_layers=4, num_local_experts=8, hidden_size=4096,
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
