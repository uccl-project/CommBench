#!/usr/bin/env python3
"""
ref_vllm_nccl_weight_transfer_engine.py

Reference implementation and benchmark for vLLM's NCCL-based weight
transfer engine — the in-place RL-style "trainer broadcasts a fresh
named-tensor state-dict to inference workers" path.

Background:
    vLLM's NCCLWeightTransferEngine (vllm/distributed/weight_transfer/
    nccl_engine.py) wraps the trainer-side `trainer_send_weights` and
    the worker-side `receive_weights` calls.  Both end up driving
    PyNcclCommunicator.broadcast over a dedicated process group so the
    update can run in parallel with the inference engine's main NCCL
    group without serializing on it.  Two transport flavors exist —
    one-by-one (`packed=False`) and double/triple-buffered packed
    broadcast (`packed=True`).  This example exercises the one-by-one
    flavor; it is the canonical "trainer pushes 10 named buckets to
    inference" pattern and exposes per-tensor overhead unobscured by
    the buffered packing layer.

Why we wrap PyNcclCommunicator + the static helpers (not the engine):
    `NCCLWeightTransferEngine.__init__` requires a full
    `WeightTransferConfig` + `ParallelConfig`, and `init_transfer_engine`
    expects a `StatelessProcessGroup` rendezvous over a separate
    master_addr/port.  Both are heavy scaffolding for a kernel
    benchmark.  The interesting wire pattern IS the
    `PyNcclCommunicator.broadcast` loop driven by
    `NCCLTrainerSendWeightsArgs(packed=False)` + the dual loop in
    `receive_weights` that allocates `torch.empty(shape, dtype)` and
    broadcasts into it.  The wrapper drives that loop directly and
    delegates the trainer side to `NCCLWeightTransferEngine.
    trainer_send_weights` so the example exercises the same code path
    that runs in production.

Purpose:
- Demonstrate the production "remote in-place named-parameter update"
  path for an RL trainer-inference pair.
- Verify byte equality across a 10-bucket named state-dict.
- Benchmark total transfer time + average per-tensor overhead for
  total payload sizes ramping from 1 MiB up to 256 MiB.
- Print exactly one JSON object to stdout (rank 0 only).

Launch:
    torchrun --nproc_per_node=2 ref_vllm_nccl_weight_transfer_engine.py
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist


# ============================================================
# Distributed Environment
# ============================================================

class DistributedEnv:
    """
    Initializes the default NCCL world group plus a parallel Gloo group
    that PyNcclCommunicator binds to (it explicitly forbids being bound
    to a NCCL-backed group).
    """

    nccl_group = None
    gloo_group = None

    @staticmethod
    def Init() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Single-node only.  The trainer/worker split here is logical:
        # rank 0 plays "trainer" and ranks > 0 play "inference."
        assert world_size == local_world_size, "multi-node runs are not supported"
        assert rank == local_rank, "multi-node runs are not supported"
        assert world_size == 2, (
            "NCCL weight transfer example pins world_size=2 "
            "(one trainer + one inference worker)"
        )

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

# A single weight bucket in the synthetic "state dict": name, shape,
# dtype.  The wire payload is the row-major flattened tensor.
@dataclass
class WeightSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype


class VllmNcclWeightTransfer:
    """
    Wrapper around vLLM's NCCLWeightTransferEngine helpers that exposes a
    SendWeights / RecvWeights pair driven by PyNcclCommunicator.broadcast
    on a dedicated CUDA stream.

    `SendWeights(named_tensors)` (rank 0 / trainer):
        Iterate named tensors and broadcast each on the wrapper's stream.
        Delegates to `NCCLWeightTransferEngine.trainer_send_weights` with
        `packed=False` so we exercise the production loop.

    `RecvWeights(specs)` (rank != 0 / inference):
        For each (name, shape, dtype), allocate a fresh device tensor
        and broadcast into it.  Returns the populated `[(name, tensor),
        ...]` list — what `receive_weights` hands to `load_weights` in
        production.

    A non-default `torch.cuda.Stream` is used so the broadcast doesn't
    serialize behind unrelated default-stream work; we always
    `wait_stream` on the default stream before the first send/recv to
    avoid the cross-stream race spelled out in rule #11 of the
    examples plan.
    """

    def __init__(
        self,
        gloo_group: dist.ProcessGroup,
        device: torch.device,
    ):
        # TODO
        raise NotImplementedError

    def SendWeights(self, named_tensors: List[Tuple[str, torch.Tensor]]) -> None:
        """Trainer (src=0) broadcast of a named tensor list, packed=False."""
        # TODO
        raise NotImplementedError

    def RecvWeights(
        self, specs: List[WeightSpec]
    ) -> List[Tuple[str, torch.Tensor]]:
        """Inference (src=0) recv loop: allocate per spec, broadcast in."""
        # TODO
        raise NotImplementedError

    def Sync(self) -> None:
        """Block the host until queued NCCL work on the comm stream is done."""
        # TODO
        raise NotImplementedError

    def Close(self) -> None:
        # TODO
        raise NotImplementedError


# ============================================================
# Weight-spec construction & deterministic patterns
# ============================================================

# Five-dtype rotation matches NCCLWeightTransferEngine's real workload —
# RL training state dicts intermix fp16/bf16 weights with fp32 optimizer
# state and integer routing/index tensors.
_DTYPE_ROTATION: List[torch.dtype] = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.int64,
]


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _build_specs(total_bytes: int, num_buckets: int) -> List[WeightSpec]:
    """Build `num_buckets` 1-D specs whose total raw bytes are
    `total_bytes`.  Each bucket gets `total_bytes // num_buckets` bytes,
    rounded up to the bucket's dtype boundary so numel is exact."""
    per_bucket_bytes = total_bytes // num_buckets
    specs: List[WeightSpec] = []
    for i in range(num_buckets):
        dtype = _DTYPE_ROTATION[i % len(_DTYPE_ROTATION)]
        numel = max(1, per_bucket_bytes // _itemsize(dtype))
        specs.append(WeightSpec(
            name=f"layer_{i:02d}.weight", shape=(numel,), dtype=dtype,
        ))
    return specs


def _expected_pattern(spec: WeightSpec, device: torch.device) -> torch.Tensor:
    """Deterministic recipe: index + spec_index, modulo 64, cast to
    the bucket's dtype.  Both ranks can recompute this independently
    so we have a true oracle, not a round-trip echo."""
    spec_idx = int(spec.name.split("_")[1].split(".")[0])
    idx = torch.arange(spec.shape[0], device=device)
    base = (idx + spec_idx) % 64
    return base.to(spec.dtype)


def _build_named_tensors(
    specs: List[WeightSpec], device: torch.device
) -> List[Tuple[str, torch.Tensor]]:
    return [(s.name, _expected_pattern(s, device)) for s in specs]


def _verify_received(
    received: List[Tuple[str, torch.Tensor]],
    specs: List[WeightSpec],
    device: torch.device,
) -> bool:
    if len(received) != len(specs):
        return False
    by_name = dict(received)
    for s in specs:
        if s.name not in by_name:
            return False
        got = by_name[s.name]
        if got.dtype != s.dtype or tuple(got.shape) != tuple(s.shape):
            return False
        expected = _expected_pattern(s, device)
        if not torch.equal(got, expected):
            return False
    return True


# ============================================================
# Evaluation
# ============================================================

@dataclass
class WeightTransferCase:
    name: str
    total_bytes: int
    num_buckets: int = 10


class WeightTransferEvaluator:
    """Runs correctness check and wall-clock benchmark for a single case.

    Wall-clock — not CUDA events — because the benchmark covers a python
    iterator + multiple kernel launches; the per-tensor overhead we
    care about is host-side.
    """

    def __init__(
        self,
        comm: VllmNcclWeightTransfer,
        local_rank: int,
        world_size: int,
    ):
        self.comm = comm
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.is_trainer = local_rank == 0

    def _build(self, case: WeightTransferCase
               ) -> Tuple[List[WeightSpec], Optional[List[Tuple[str, torch.Tensor]]]]:
        specs = _build_specs(case.total_bytes, case.num_buckets)
        if self.is_trainer:
            payload = _build_named_tensors(specs, self.device)
        else:
            payload = None
        return specs, payload

    # --------------------------------------------------------
    # Correctness — broadcast 10 named tensors, verify pattern
    # --------------------------------------------------------
    def CheckCorrectness(self, case: WeightTransferCase) -> bool:
        specs, payload = self._build(case)

        if self.is_trainer:
            self.comm.SendWeights(payload)
            self.comm.Sync()
            ok_local = True
        else:
            received = self.comm.RecvWeights(specs)
            self.comm.Sync()
            ok_local = _verify_received(received, specs, self.device)

        flag = torch.tensor(
            [1 if ok_local else 0], device=self.device, dtype=torch.int32
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item() == 1)

    # --------------------------------------------------------
    # Benchmark — full broadcast cycle, wall-clock on rank 0
    # --------------------------------------------------------
    def Benchmark(
        self,
        case: WeightTransferCase,
        warmup_iters: int,
        iters: int,
    ) -> Dict[str, float]:
        specs, payload = self._build(case)

        # Warmup
        for _ in range(warmup_iters):
            if self.is_trainer:
                self.comm.SendWeights(payload)
            else:
                _ = self.comm.RecvWeights(specs)
        self.comm.Sync()
        dist.barrier()

        if self.is_trainer:
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                self.comm.SendWeights(payload)
                self.comm.Sync()
            elapsed_ns = time.perf_counter_ns() - t0
        else:
            for _ in range(iters):
                _ = self.comm.RecvWeights(specs)
                self.comm.Sync()
            elapsed_ns = 0  # only rank 0 reports

        dist.barrier()

        latency_us = elapsed_ns / iters / 1e3 if elapsed_ns > 0 else 0.0
        # True transferred bytes (sum over the actual specs we built —
        # a few bytes off `case.total_bytes` because of dtype rounding).
        actual_bytes = sum(
            s.shape[0] * _itemsize(s.dtype) for s in specs
        )
        size_mb = actual_bytes / (1024.0 * 1024.0)
        if latency_us > 0:
            # Algorithmic BW = bytes / time.  Per-broadcast point-to-point
            # rule #6 of the plan: send/recv → algo BW.
            throughput_gbps = (
                (actual_bytes * 8.0) / (latency_us * 1e-6) / 1e9
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

    comm: Optional[VllmNcclWeightTransfer] = None
    try:
        comm = VllmNcclWeightTransfer(
            gloo_group=DistributedEnv.gloo_group,
            device=device,
        )
        evaluator = WeightTransferEvaluator(
            comm=comm,
            local_rank=local_rank,
            world_size=world_size,
        )

        # Plan-driven sweep: 1 MiB → 256 MiB in 4 steps; 10 buckets per
        # case.  We stop at 256 MiB because a 1 GiB total is dominated
        # by a single broadcast and obscures the per-tensor overhead
        # this example is meant to surface.
        cases = [
            WeightTransferCase(
                name="1MiB",   total_bytes=1 * 1024 * 1024,
            ),
            WeightTransferCase(
                name="16MiB",  total_bytes=16 * 1024 * 1024,
            ),
            WeightTransferCase(
                name="64MiB",  total_bytes=64 * 1024 * 1024,
            ),
            WeightTransferCase(
                name="256MiB", total_bytes=256 * 1024 * 1024,
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
