#!/usr/bin/env python3
"""
ref_vllm_pp_send_recv_tensor_dict.py

Reference implementation and benchmark for vLLM's pipeline-parallel
point-to-point exchange of a heterogeneous tensor dict.

Background:
    vLLM's GroupCoordinator (vllm/distributed/parallel_state.py) ships a
    `send_tensor_dict` / `recv_tensor_dict` pair that pipeline-parallel
    stages use to forward activations between layers.  The dict can mix
    tensors of different dtypes, shapes and device locations alongside
    pickle-able non-tensor metadata; the implementation transports the
    metadata over the group's CPU (Gloo) sub-group and the tensor payload
    over the device (NCCL) sub-group.

Why we wrap GroupCoordinator directly (not initialize_model_parallel):
    `initialize_model_parallel` reads `get_current_vllm_config()` and
    therefore needs a fully-formed vLLM config + parallel-state init,
    which is heavyweight scaffolding for a kernel-level benchmark.  The
    interesting wire pattern is exactly what `GroupCoordinator.
    {send,recv}_tensor_dict` does, and `GroupCoordinator(...)` accepts a
    plain `group_ranks` list — no vLLM config required.  Same NCCL/Gloo
    call sequence, no forward-context plumbing.

Purpose:
- Demonstrate the production "PP stage hand-off" send/recv path with a
  heterogeneous (dtype, shape) dict including non-tensor metadata.
- Verify byte equality across a full round-trip echo.
- Benchmark single-tensor sweep and a metadata-heavy mode that stresses
  the pickle/Gloo metadata side of the pipeline.
- Print exactly one JSON object to stdout (rank 0 only).

Launch:
    torchrun --nproc_per_node=2 ref_vllm_pp_send_recv_tensor_dict.py
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


# ============================================================
# Distributed Environment
# ============================================================

class DistributedEnv:
    """
    Initializes the default NCCL world group.  GroupCoordinator builds
    its own NCCL + Gloo sub-groups internally on whatever ranks we hand
    it, so we do not need to set up a separate Gloo group up front.
    """

    @staticmethod
    def Init() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Single-node only.
        assert world_size == local_world_size, "multi-node runs are not supported"
        assert rank == local_rank, "multi-node runs are not supported"
        assert world_size == 2, (
            "PP send/recv tensor-dict example pins world_size=2 "
            "(a minimal 2-stage pipeline)"
        )

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=local_rank,
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

class VllmPpTensorDict:
    """
    Wrapper around vLLM's GroupCoordinator that exposes a Send / Recv
    pair matching the PP stage hand-off contract.

    `Send(tensor_dict, dst=None)`:
        Send a heterogeneous {str: tensor|any} dict to peer `dst`
        (default: next rank in the group).  Tensor payload over NCCL,
        pickled metadata over Gloo.  Blocking.

    `Recv(src=None)`:
        Allocate fresh tensors on the local device matching the sender's
        metadata, populate them, and return the new dict.  Blocking.

    The wrapper itself is stream-agnostic — the underlying torch.distributed
    isend/irecv attaches to the default CUDA stream; vLLM relies on this
    for its PP plumbing and we stay faithful.
    """

    def __init__(self, local_rank: int, world_size: int, device: torch.device):
        # TODO
        raise NotImplementedError

    def Send(
        self,
        tensor_dict: Dict[str, Any],
        dst: Optional[int] = None,
    ) -> None:
        """Blocking send to local-rank-in-group `dst` (default: next rank)."""
        # TODO
        raise NotImplementedError

    def Recv(self, src: Optional[int] = None) -> Dict[str, Any]:
        """Blocking recv from local-rank-in-group `src` (default: prev rank)."""
        # TODO
        raise NotImplementedError

    def Sync(self) -> None:
        # TODO
        raise NotImplementedError

    def Close(self) -> None:
        # TODO
        raise NotImplementedError


# ============================================================
# Tensor-dict recipes
# ============================================================

@dataclass
class TensorDictCase:
    """One benchmark/correctness case.

    Either `single_tensor_bytes` is set (one big tensor under the
    "hidden_states" key, fp16) or `meta_tensors` is set (many small
    tensors of mixed dtype, modelling a real PP hand-off with KV
    metadata, sequence lengths, attention masks, etc.).
    """
    name: str
    single_tensor_bytes: Optional[int] = None
    meta_tensors: Optional[int] = None
    meta_tensor_numel: int = 64
    label_bytes: Optional[int] = None  # carried in JSON only

    def total_payload_bytes(self) -> int:
        if self.single_tensor_bytes is not None:
            return self.single_tensor_bytes
        # meta_tensors mode — see _build_recipe for dtype mix.
        if self.meta_tensors is None:
            return 0
        # Average across the dtype rotation: fp16(2) bf16(2) fp32(4) int32(4) int64(8) -> avg 4
        avg_dtype_bytes = 4
        return self.meta_tensors * self.meta_tensor_numel * avg_dtype_bytes


def _build_recipe(case: TensorDictCase, device: torch.device) -> Dict[str, Any]:
    """Construct the dict that rank 0 sends.  Deterministic so a rank
    that never saw the original can still rebuild it for comparison.
    """
    out: Dict[str, Any] = {}
    if case.single_tensor_bytes is not None:
        # fp16 single big tensor.  Use deterministic small-int values so
        # the sum across the round-trip stays exactly representable.
        numel = case.single_tensor_bytes // 2
        idx = torch.arange(numel, device=device)
        out["hidden_states"] = ((idx % 16).to(torch.float16))
        # A piece of non-tensor metadata to exercise the pickle path.
        out["seq_len"] = numel
        out["dtype_tag"] = "fp16"
        return out

    if case.meta_tensors is not None:
        n = case.meta_tensors
        m = case.meta_tensor_numel
        # Rotate through 5 dtypes so metadata pickling has variety.
        dtype_rotation = [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int32,
            torch.int64,
        ]
        for i in range(n):
            dtype = dtype_rotation[i % len(dtype_rotation)]
            base = (torch.arange(m, device=device) + i) % 64
            if dtype.is_floating_point:
                out[f"k_{i:03d}"] = base.to(dtype)
            else:
                out[f"k_{i:03d}"] = base.to(dtype)
        out["batch_size"] = n
        out["layer_idx"] = 17
        return out

    raise ValueError("TensorDictCase must set single_tensor_bytes or meta_tensors")


def _dicts_byte_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a.keys():
        va, vb = a[k], b[k]
        if isinstance(va, torch.Tensor) != isinstance(vb, torch.Tensor):
            return False
        if isinstance(va, torch.Tensor):
            if va.dtype != vb.dtype or tuple(va.shape) != tuple(vb.shape):
                return False
            if not torch.equal(va.cpu(), vb.cpu()):
                return False
        else:
            if va != vb:
                return False
    return True


# ============================================================
# Evaluation
# ============================================================

class PpTensorDictEvaluator:
    """Runs round-trip correctness check and wall-clock benchmark."""

    def __init__(
        self,
        comm: VllmPpTensorDict,
        local_rank: int,
        world_size: int,
    ):
        self.comm = comm
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.rank_in_group = comm.rank_in_group

    # --------------------------------------------------------
    # Correctness — round-trip echo
    # --------------------------------------------------------
    def CheckRoundTrip(self, case: TensorDictCase) -> bool:
        """Rank 0 sends a recipe-built dict, rank 1 echoes it back, rank 0
        compares to the original byte-for-byte."""
        if self.rank_in_group == 0:
            payload = _build_recipe(case, self.device)
            self.comm.Send(payload, dst=1)
            echoed = self.comm.Recv(src=1)
            self.comm.Sync()
            ok_local = _dicts_byte_equal(payload, echoed)
        else:
            received = self.comm.Recv(src=0)
            self.comm.Send(received, dst=0)
            self.comm.Sync()
            # Receiver has no separate oracle; the round-trip on rank 0
            # is the contract.  Receiver also rebuilds the recipe and
            # cross-checks to catch a one-way corruption that the echo
            # would otherwise mask.
            expected = _build_recipe(case, self.device)
            ok_local = _dicts_byte_equal(received, expected)

        # Both ranks must agree.
        flag = torch.tensor([1 if ok_local else 0], device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item() == 1)

    # --------------------------------------------------------
    # Benchmark — round-trip on the default stream, report half
    # --------------------------------------------------------
    def Benchmark(
        self,
        case: TensorDictCase,
        warmup_iters: int,
        iters: int,
    ) -> Dict[str, float]:
        # Pre-build payload outside the timed region so we measure the
        # send/recv pipeline, not test-side tensor allocation.  Receiver
        # has nothing to pre-build (Recv allocates fresh tensors itself
        # — that allocation is part of what we want to measure).
        if self.rank_in_group == 0:
            payload = _build_recipe(case, self.device)

        # Warmup
        for _ in range(warmup_iters):
            if self.rank_in_group == 0:
                self.comm.Send(payload, dst=1)
                _ = self.comm.Recv(src=1)
            else:
                received = self.comm.Recv(src=0)
                self.comm.Send(received, dst=0)
        self.comm.Sync()
        dist.barrier()

        # Timed region
        if self.rank_in_group == 0:
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                self.comm.Send(payload, dst=1)
                _ = self.comm.Recv(src=1)
            torch.cuda.synchronize(self.device)
            elapsed_ns = time.perf_counter_ns() - t0
        else:
            for _ in range(iters):
                received = self.comm.Recv(src=0)
                self.comm.Send(received, dst=0)
            self.comm.Sync()
            elapsed_ns = 0  # unused on rank 1

        dist.barrier()

        # Round-trip per iter; one-way is half.
        roundtrip_us = elapsed_ns / iters / 1e3
        oneway_us = roundtrip_us / 2.0

        payload_bytes = case.total_payload_bytes()
        size_mb = payload_bytes / (1024.0 * 1024.0)
        if oneway_us > 0:
            throughput_gbps = (payload_bytes * 8.0) / (oneway_us * 1e-6) / 1e9
        else:
            throughput_gbps = 0.0

        return {
            "data_size": size_mb,           # MB, single-direction payload
            "latency_avg": oneway_us,       # us, one-way send+recv
            "throughput_avg": throughput_gbps,  # Gbps, algorithmic
        }


# ============================================================
# Test orchestration
# ============================================================

def runTest():
    local_rank, world_size = DistributedEnv.Init()
    device = torch.device(f"cuda:{local_rank}")

    correctness = "PASS"
    metrics: List[Dict[str, float]] = []

    comm: Optional[VllmPpTensorDict] = None
    try:
        comm = VllmPpTensorDict(
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )
        evaluator = PpTensorDictEvaluator(
            comm=comm,
            local_rank=local_rank,
            world_size=world_size,
        )

        # Three single-tensor sizes spanning small to medium PP payloads
        # plus one metadata-heavy case (50 small tensors of mixed dtype).
        # The metadata-heavy case stresses the pickle/Gloo metadata path
        # the single-tensor cases barely exercise.
        cases = [
            TensorDictCase(name="single_16KiB",
                           single_tensor_bytes=16 * 1024),
            TensorDictCase(name="single_1MiB",
                           single_tensor_bytes=1 * 1024 * 1024),
            TensorDictCase(name="single_16MiB",
                           single_tensor_bytes=16 * 1024 * 1024),
            TensorDictCase(name="meta50",
                           meta_tensors=50, meta_tensor_numel=64),
        ]

        for case in cases:
            if not evaluator.CheckRoundTrip(case):
                correctness = "FAIL"
                break

        if correctness == "PASS":
            for case in cases:
                m = evaluator.Benchmark(case, warmup_iters=3, iters=20)
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
