"""
Shared performance-verdict logic + per-example metric registry.

Each example produces its own metric dict in summary.json, but historically
each build_and_run.py also had its own ad-hoc rule for turning those into a
`summary["performance"]` label (some 3-tier "same/better/worse", some 4-tier
"better/on_par/degraded/severely_degraded", a few one-off schemes). This
module is the single source of truth:

  * 4-tier verdict vocabulary:  better / on_par / degraded / severely_degraded
  * Uniform thresholds: ±5% on_par, ≥+20% better, ≤−40% severely degraded,
    everything in between (i.e., −40..−5) degraded.
  * One primary metric per example (the registry below).
  * A "did-not-measure" guard: if the primary metric is lower-is-better and
    ref > 0 while gen == 0, we treat that as ~−100% improvement (severely
    degraded). This blocks the common failure mode where generated code
    reports zero latency because the timing logic is broken — without this
    guard the naive (ref − 0)/ref formula would award +100% improvement and
    the model could "win" by simply not implementing the timer.

Each build_and_run.py is expected to call `compute_unified_verdict(...)` by
default. A `--legacy-perf-verdict` CLI flag (added in each build_and_run.py)
can still pin the old per-example logic for backward compat / debugging.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 4-tier thresholds
# ---------------------------------------------------------------------------
# Improvement is computed as a signed % delta of the *primary* metric only,
# in the direction declared in the registry. So positive = generated is
# better than reference. Bands:
#
#   imp >= +20%   → better
#   imp >= -5%    → on_par     (covers [-5%, +20%): mild change either way)
#   imp >= -40%   → degraded
#   else          → severely_degraded
#
# Asymmetric on purpose: small positive deltas are noise (still on_par),
# small negative deltas count as a real regression.
BETTER_PCT             =  20.0
ON_PAR_LOWER_PCT       =  -5.0
DEGRADED_LOWER_PCT     = -40.0

# Core 4-tier verdicts (real perf comparison happened):
VERDICT_BETTER             = "better"
VERDICT_ON_PAR             = "on_par"
VERDICT_DEGRADED           = "degraded"
VERDICT_SEVERELY_DEGRADED  = "severely_degraded"

# Non-comparison verdicts (NO real perf comparison happened — kept distinct
# from on_par so a CSV reader can immediately tell "this row didn't actually
# measure anything" from "model truly held parity with ref"):
VERDICT_INFO_ONLY          = "info_only"        # registry says no perf metric exists
VERDICT_NO_GEN_METRICS     = "no_gen_metrics"   # gen didn't produce the primary metric
VERDICT_NO_REF_METRICS     = "no_ref_metrics"   # ref didn't produce the primary metric (registry probably wrong)
VERDICT_UNKNOWN            = "unknown"          # example missing from PERF_METRICS registry

ALL_VERDICTS = (
    VERDICT_BETTER,
    VERDICT_ON_PAR,
    VERDICT_DEGRADED,
    VERDICT_SEVERELY_DEGRADED,
    VERDICT_INFO_ONLY,
    VERDICT_NO_GEN_METRICS,
    VERDICT_NO_REF_METRICS,
    VERDICT_UNKNOWN,
)

# Verdicts that should NOT trigger a perf-fix retry in generate_eval_one's
# main loop. Includes the "good enough" comparisons (on_par, better) and
# every non-comparison status (info_only/no_gen_metrics/no_ref_metrics/
# unknown) because retrying won't make a missing metric appear.
ACCEPTABLE_VERDICTS = frozenset({
    VERDICT_BETTER,
    VERDICT_ON_PAR,
    VERDICT_INFO_ONLY,
    VERDICT_NO_GEN_METRICS,
    VERDICT_NO_REF_METRICS,
    VERDICT_UNKNOWN,
})


# ---------------------------------------------------------------------------
# Per-example metric registry
# ---------------------------------------------------------------------------
# Entry forms:
#   "info_only"                  → example does not have a numeric perf metric.
#                                  Verdict is always on_par when correctness PASS;
#                                  retry never fires for perf reasons.
#   {"primary": KEY,
#    "direction": "higher"|"lower"}
#                                → use this single key from metrics_comparison.{ref,generated}
#                                  with this direction to compute the % delta.
#
# Keys in PERF_METRICS are example names as they appear under datasets/
# (e.g. "example003_fifo_device2host"). Examples not in the registry fall
# back to "unknown" → on_par (i.e., no perf retry, no false-positive
# degraded). When you add a new example, add an entry here.
#
# The choices below are best-effort initial guesses based on the metric
# names that show up in each example's summary.json on the current GH200
# host. Review and correct any that don't match what the example actually
# measures.

PERF_METRICS: Dict[str, Any] = {
    # ── flat layout (build_and_run.py at top of example dir) ──
    "example011_rdma_write_rc_with_imm":     {"primary": "latency_avg",      "direction": "lower"},
    "example013_rdma_write_zero_message":    {"primary": "latency_avg",      "direction": "lower"},
    "example014_rdma_write_inline":          {"primary": "latency_avg",      "direction": "lower"},
    "example024_rdma_atomic_cmp_swp":        {"primary": "latency_avg",      "direction": "lower"},
    "example025_rdma_atomic_fetch_add":      {"primary": "latency_avg",      "direction": "lower"},
    "example026_rdma_ibv_get_async_event":   "info_only",
    "example027_rdma_ibv_wr_bind":           {"primary": "latency_avg",      "direction": "lower"},
    "example028_ibv_query_rt_values_ex":     {"primary": "latency_avg",      "direction": "lower"},
    "example029_ibv_query_qp_data_in_order": "info_only",
    "example072_GPU_barrier_within_CTA":     {"primary": "latency_avg",      "direction": "lower"},
    "example075_GPU_d2h_copy_bandwidth":     {"primary": "throughput",       "direction": "higher"},
    # NB: example076 is a NANOSLEEP-ACCURACY benchmark, not a latency one.
    # The right primary metric is the *error* in measured sleep time, which
    # is `abs_error_ns_avg` (lower = more accurate timer).
    "example076_GPU_globaltimer_nanosleep":  {"primary": "abs_error_ns_avg", "direction": "lower"},
    "example077_GPU_vectorized_copy_widths": {"primary": "throughput_avg",   "direction": "higher"},
    # NB: TMA bulk copy reports `aggregate_throughput`, not `throughput_avg`.
    "example094_TMA_bulk_copy_gmem_to_smem": {"primary": "aggregate_throughput", "direction": "higher"},

    # ── platform-subdir layout (build_and_run.py under nv/ib/amd_nv/roce) ──
    # NB: example002 emits `throughput`, not `throughput_avg`.
    "example002_rdma_loopback":                       {"primary": "throughput",     "direction": "higher"},
    "example003_fifo_device2host":                    {"primary": "throughput_MBps","direction": "higher"},
    "example004_rdma_nic_info":                       "info_only",
    "example005_fifo_host2device":                    {"primary": "throughput_avg", "direction": "higher"},
    "example006_rdma_read_write_rc":                  {"primary": "throughput_gbps","direction": "higher"},
    "example007_rdma_send_recv_uc":                   {"primary": "throughput_avg", "direction": "higher"},
    "example008_rdma_write_shared_receive_queue":     {"primary": "throughput_avg", "direction": "higher"},
    # NB: example009 emits `bandwidth_gbps`, not `bandwidth`.
    "example009_rdma_recv_and_send":                  {"primary": "bandwidth_gbps", "direction": "higher"},
    "example010_list_gpu_attributes":                 "info_only",
    "example015_memory_pool_with_registered_region":  "info_only",
    "example016_rdma_send_recv_ud":                   {"primary": "throughput_avg", "direction": "higher"},
    "example019_rdma_write_scatter_gather_lists":     {"primary": "throughput_avg", "direction": "higher"},

    # ── b300 batch (see batch_scripts_b300/_run_passed.txt) ──
    # Convention: this batch's build_and_run.py scripts emit `throughput`
    # (without `_avg`) in metrics_comparison.{ref,generated} after averaging
    # the per-record `throughput_avg`. The mscclpp variant emits
    # `throughput_avg` directly. Each entry was set by reading the actual
    # _metrics_avg() function in that example's build_and_run.py.
    #
    # Standard layout — keys=[data_size_avg, latency_avg, throughput]
    "example001_gpu_comm_single_process":             {"primary": "throughput",     "direction": "higher"},
    "example012_gpu_ipc_comm_with_NVLink":            {"primary": "throughput",     "direction": "higher"},
    "example017_intranode_gpu_barrier":               {"primary": "throughput",     "direction": "higher"},
    "example020_torch_distributed_nccl_alltoall":     {"primary": "throughput",     "direction": "higher"},
    "example021_torch_distributed_nccl_allgather":    {"primary": "throughput",     "direction": "higher"},
    "example022_torch_distributed_nccl_allreduce":    {"primary": "throughput",     "direction": "higher"},
    "example023_torch_distributed_nccl_reduce_scatter":{"primary": "throughput",    "direction": "higher"},
    "example032_nvshmem_atomic_fetch_inc":            {"primary": "throughput",     "direction": "higher"},
    "example033_nvshmem_put_and_signal":              {"primary": "throughput",     "direction": "higher"},
    "example034_nvshmem_broadcast":                   {"primary": "throughput",     "direction": "higher"},
    "example035_nvshmem_tiled_produce_consume":       {"primary": "throughput",     "direction": "higher"},
    "example042_deepep_dispatch_layout":              {"primary": "throughput",     "direction": "higher"},
    "example043_nccl_device_api_intranode_allreduce": {"primary": "throughput",     "direction": "higher"},
    # nccl_device_api 044/045/091 emit {count, latency_avg, algbw_avg, busbw_avg}
    # — busbw_avg (effective bus bandwidth) is the canonical NCCL primary.
    "example044_nccl_device_api_intranode_alltoall":  {"primary": "busbw_avg",      "direction": "higher"},
    "example045_nccl_device_api_allgather_intra":     {"primary": "busbw_avg",      "direction": "higher"},
    "example046_thunderkitten_all_gather_easy":       {"primary": "throughput",     "direction": "higher"},
    "example047_deepep_intranode_notify_dispatch":    {"primary": "throughput",     "direction": "higher"},
    "example048_thunderkitten_all_reduce_easy":       {"primary": "throughput",     "direction": "higher"},
    "example049_thunderkitten_alltoall_easy":         {"primary": "throughput",     "direction": "higher"},
    "example050_thunderkitten_reduce_scatter_easy":   {"primary": "throughput",     "direction": "higher"},
    "example051_thunderkitten_ag_gemm_easy":          {"primary": "throughput",     "direction": "higher"},
    "example052_thunderkitten_ag_gemm_fp8_easy":      {"primary": "throughput",     "direction": "higher"},
    "example053_thunderkitten_gemm_ar_easy":          {"primary": "throughput",     "direction": "higher"},
    "example054_thunderkitten_gemm_rs_easy":          {"primary": "throughput",     "direction": "higher"},
    "example055_thunderkitten_gemm_rs_fp8_easy":      {"primary": "throughput",     "direction": "higher"},
    "example056_thunderkitten_moe_dispatch_gemm_easy":{"primary": "throughput",     "direction": "higher"},
    "example057_thunderkitten_ring_attn_easy":        {"primary": "throughput",     "direction": "higher"},
    "example058_thunderkitten_ulysses_attn_easy":     {"primary": "throughput",     "direction": "higher"},
    "example059_thunderkitten_all_gather_hard":       {"primary": "throughput",     "direction": "higher"},
    "example060_thunderkitten_all_reduce_hard":       {"primary": "throughput",     "direction": "higher"},
    "example061_thunderkitten_alltoall_hard":         {"primary": "throughput",     "direction": "higher"},
    "example062_thunderkitten_reduce_scatter_hard":   {"primary": "throughput",     "direction": "higher"},
    "example063_thunderkitten_ag_gemm_hard":          {"primary": "throughput",     "direction": "higher"},
    "example064_thunderkitten_ag_gemm_fp8_hard":      {"primary": "throughput",     "direction": "higher"},
    "example065_thunderkitten_gemm_ar_hard":          {"primary": "throughput",     "direction": "higher"},
    "example066_thunderkitten_gemm_rs_hard":          {"primary": "throughput",     "direction": "higher"},
    "example067_thunderkitten_gemm_rs_fp8_hard":      {"primary": "throughput",     "direction": "higher"},
    "example068_thunderkitten_moe_dispatch_gemm_hard":{"primary": "throughput",     "direction": "higher"},
    "example069_thunderkitten_ring_attn_hard":        {"primary": "throughput",     "direction": "higher"},
    "example070_thunderkitten_ulysses_attn_hard":     {"primary": "throughput",     "direction": "higher"},
    # NB: example073 is a pingpong-LATENCY benchmark. Although it also emits
    # `throughput` (msg/s), the headline metric is round-trip latency.
    "example073_GPU_p2p_pingpong_latency":            {"primary": "latency_avg",    "direction": "lower"},
    "example074_GPU_p2p_d2d_bandwidth":               {"primary": "throughput",     "direction": "higher"},
    "example084_vllm_agrs_moe_all_to_all":            {"primary": "throughput",     "direction": "higher"},
    "example085_vllm_pp_send_recv_tensor_dict":       {"primary": "throughput",     "direction": "higher"},
    "example087_vllm_nccl_weight_transfer_engine":    {"primary": "throughput",     "direction": "higher"},
    "example088_vllm_ipc_weight_transfer_engine":     {"primary": "throughput",     "direction": "higher"},
    "example089_vllm_eplb_expert_rebalance":          {"primary": "throughput",     "direction": "higher"},
    "example091_nccl_device_api_reducescatter_from_blockers":{"primary":"busbw_avg","direction":"higher"},
    "example097_vllm_custom_all_reduce":              {"primary": "throughput",     "direction": "higher"},
    "example098_vllm_symm_mem_all_reduce":            {"primary": "throughput",     "direction": "higher"},
    "example099_vllm_pynccl_all_reduce":              {"primary": "throughput",     "direction": "higher"},
    "example100_vllm_shm_broadcast_object":           {"primary": "throughput",     "direction": "higher"},

    # mscclpp variant — _metrics_avg emits keys=[count, latency_avg, throughput_avg]
    "example018_mscclpp_alltoall_easy":               {"primary": "throughput_avg", "direction": "higher"},
    "example030_mscclpp_allgather_fullmesh_easy":     {"primary": "throughput_avg", "direction": "higher"},
    "example031_msclpp_allreduce_rsag_zero_easy":     {"primary": "throughput_avg", "direction": "higher"},
    "example037_mscclpp_reducescatter_easy":          {"primary": "throughput_avg", "direction": "higher"},
    "example038_mscclpp_mc_put_easy":                 {"primary": "throughput_avg", "direction": "higher"},
    "example039_mscclpp_mc_get_easy":                 {"primary": "throughput_avg", "direction": "higher"},
    "example040_mscclpp_relaxedsingal_easy":          {"primary": "throughput_avg", "direction": "higher"},
    "example086_mscclpp_alltoall_hard":               {"primary": "throughput_avg", "direction": "higher"},
    "example093_mscclpp_reducescatter_hard":          {"primary": "throughput_avg", "direction": "higher"},
    "example095_msclpp_allreduce_rsag_zero_hard":     {"primary": "throughput_avg", "direction": "higher"},
    "example096_mscclpp_mc_put_hard":                 {"primary": "throughput_avg", "direction": "higher"},

    # Previously non-conformant; now converted to emit metrics_comparison
    # flat dict. deepep_jit _metrics_avg returns {data_size_avg, latency_avg,
    # throughput} → primary=throughput. sglang_qknorm now computes a mscclpp-
    # style {count, latency_avg, throughput_avg} → primary=throughput_avg.
    "example078_deepep_jit_dispatch_deterministic_prologue": {"primary": "throughput",     "direction": "higher"},
    "example079_deepep_jit_dispatch_notify":          {"primary": "throughput",     "direction": "higher"},
    "example080_deepep_jit_dispatch_payload":         {"primary": "throughput",     "direction": "higher"},
    "example081_deepep_jit_dispatch_copy_epilogue":   {"primary": "throughput",     "direction": "higher"},
    "example082_deepep_jit_combine":                  {"primary": "throughput",     "direction": "higher"},
    "example083_deepep_jit_combine_reduce_epilogue":  {"primary": "throughput",     "direction": "higher"},
    "example101_sglang_qknorm_fused_easy":            {"primary": "throughput_avg", "direction": "higher"},
    "example102_sglang_qknorm_fused_hard":            {"primary": "throughput_avg", "direction": "higher"},

    # ── Discovered during the b300 batch re-run (not in original
    # _run_passed.txt). All four now emit a proper `metrics_comparison`
    # flat dict per datasets/readme.md.
    #   example090: mscclpp variant — keys=[count, latency_avg, throughput_avg]
    #   example092: standard layout — keys=[data_size_avg, latency_avg, throughput]
    #               (currently fails on this host with PMIX UNREACHABLE
    #               because it's an internode test; entry is still correct
    #               for when cluster comms are configured)
    #   example105/109: mscclpp-style — keys=[count, latency_avg, throughput_avg]
    "example090_mscclpp_mc_get_hard":                 {"primary": "throughput_avg", "direction": "higher"},
    "example092_nccl_device_api_internode_allreduce": {"primary": "throughput",     "direction": "higher"},
    "example105_sglang_custom_allreduce_easy":        {"primary": "throughput_avg", "direction": "higher"},
    "example109_sglang_custom_allreduce_hard":        {"primary": "throughput_avg", "direction": "higher"},
}


def _infer_example_name(hint: str) -> Optional[str]:
    """Given a filesystem path under datasets/, walk up to find the
    top-level exampleNNN_xxx directory name. Used by build_and_run.py to
    look up its own registry entry without being told its example name.
    """
    p = os.path.abspath(hint)
    while p and p != "/":
        base = os.path.basename(p)
        if base.startswith("example") and len(base) > len("example"):
            # crude: require the next two chars to be digits
            if len(base) >= 10 and base[7:10].isdigit():
                return base
        p = os.path.dirname(p)
    return None


def _improvement_pct(
    ref: float,
    gen: float,
    direction: str,
) -> Tuple[float, bool, Optional[str]]:
    """Return (improvement_pct, better_or_equal, note).

    The cheating guard fires only when direction is 'lower' (latency-style)
    AND ref > 0 but gen == 0 — the typical "model didn't actually implement
    timing" failure mode. We pin improvement to −100 in that case so the
    verdict reflects "this metric wasn't really measured" instead of the
    misleading naive +100% improvement.
    """
    if direction == "lower" and ref > 0 and gen == 0:
        return -100.0, False, "gen=0 with ref>0 treated as did-not-measure"
    if ref == 0:
        return 0.0, gen == 0, "ref=0; treated as no-op"
    if direction == "lower":
        imp = (ref - gen) / ref * 100.0
        better = gen <= ref
    else:  # higher_is_better
        imp = (gen - ref) / ref * 100.0
        better = gen >= ref
    return imp, better, None


def _classify(improvement_pct: float) -> str:
    """Bucket an improvement % into the 4-tier vocabulary."""
    if improvement_pct >= BETTER_PCT:
        return VERDICT_BETTER
    if improvement_pct >= ON_PAR_LOWER_PCT:
        return VERDICT_ON_PAR
    if improvement_pct >= DEGRADED_LOWER_PCT:
        return VERDICT_DEGRADED
    return VERDICT_SEVERELY_DEGRADED


def compute_unified_verdict(
    example_name: str,
    ref_metrics: Dict[str, Any],
    gen_metrics: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Compute the 4-tier verdict for one example.

    Args:
        example_name: e.g. "example025_rdma_atomic_fetch_add". If not in the
            registry, we default to on_par (no retry, no false positives).
        ref_metrics: flat dict of reference metrics (the same dict that
            usually ends up under summary.json["metrics_comparison"]["ref"]).
        gen_metrics: flat dict of generated metrics.

    Returns:
        (verdict, detail) where:
            verdict ∈ {better, on_par, degraded, severely_degraded}
            detail  has primary_metric, direction, ref, generated,
                    improvement_pct, and any note from the cheating guard.
    """
    entry = PERF_METRICS.get(example_name)

    if entry is None:
        # Distinct from on_par so audit reports can flag "you forgot to add
        # this example to the registry" vs. "model genuinely held parity".
        return VERDICT_UNKNOWN, {
            "primary_metric": None,
            "reason": f"example '{example_name}' not in PERF_METRICS registry; "
                      f"add an entry to silence this warning",
        }

    if entry == "info_only":
        # Distinct from on_par so we can tell "no perf metric by design"
        # apart from "model matched ref to within ±5%".
        return VERDICT_INFO_ONLY, {
            "primary_metric": None,
            "reason": "registry marks example as info_only (no perf metric)",
        }

    primary = entry["primary"]
    direction = entry["direction"]
    if direction not in ("higher", "lower"):
        return VERDICT_UNKNOWN, {
            "primary_metric": primary,
            "reason": f"registry direction must be 'higher'|'lower', got "
                      f"{direction!r}; treating as unknown",
        }

    ref_val = ref_metrics.get(primary)
    gen_val = gen_metrics.get(primary)
    ref_ok = isinstance(ref_val, (int, float))
    gen_ok = isinstance(gen_val, (int, float))
    if not ref_ok or not gen_ok:
        # Split into two verdicts. We check gen-side FIRST because most
        # "no data" cases trace back to a gen-side failure (compile/run
        # crashed → some compare()s bail out before running ref properly,
        # which leaves *both* dicts empty). Reporting that as
        # no_gen_metrics matches the actionable root cause. Pure ref-side
        # absence (gen has the metric, ref doesn't) is a registry bug —
        # we picked a primary the ref doesn't actually emit.
        if not gen_ok:
            v = VERDICT_NO_GEN_METRICS
            why = (f"generated dict has no numeric {primary!r} "
                   f"(got {gen_val!r}); compare likely did not run "
                   f"successfully")
        else:
            v = VERDICT_NO_REF_METRICS
            why = (f"ref dict has no numeric {primary!r} "
                   f"(got {ref_val!r}); registry's primary is wrong")
        return v, {
            "primary_metric": primary,
            "direction": direction,
            "ref": ref_val,
            "generated": gen_val,
            "reason": why,
        }

    imp, _better, note = _improvement_pct(float(ref_val), float(gen_val), direction)
    verdict = _classify(imp)
    detail: Dict[str, Any] = {
        "primary_metric": primary,
        "direction": direction,
        "ref": ref_val,
        "generated": gen_val,
        "improvement_pct": round(imp, 2),
    }
    if note:
        detail["note"] = note
    return verdict, detail


def override_summary_verdict(result_dir: str, verbose: bool = False) -> Optional[str]:
    """Read summary.json under result_dir, recompute its `performance` field
    using compute_unified_verdict(), and write it back.

    Preserves the original verdict at `performance_legacy`. Adds
    `performance_detail` and `verdict_scheme` for audit. No-ops silently
    when the file is absent / malformed / not in the registry — never
    breaks the surrounding pipeline.

    Returns the new verdict string on success, None on no-op.
    """
    summary_path = os.path.join(result_dir, "summary.json")
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except Exception:
        return None

    if not isinstance(summary, dict):
        return None
    mc = summary.get("metrics_comparison")
    if not isinstance(mc, dict):
        return None
    ref = mc.get("ref") or {}
    gen = mc.get("generated") or {}
    if not isinstance(ref, dict) or not isinstance(gen, dict):
        return None

    example_name = _infer_example_name(result_dir)
    if not example_name:
        return None

    verdict, detail = compute_unified_verdict(example_name, ref, gen)
    legacy = summary.get("performance")
    summary["performance"] = verdict
    summary["performance_detail"] = detail
    summary["verdict_scheme"] = "unified_perf_verdict"
    if legacy is not None and legacy != verdict:
        summary["performance_legacy"] = legacy

    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"[perf_verdict] {example_name}: "
                  f"legacy={legacy!r} → unified={verdict!r}")
        return verdict
    except Exception as e:
        if verbose:
            print(f"[perf_verdict] WARN: failed to write {summary_path}: {e}")
        return None
