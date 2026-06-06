#!/usr/bin/env python3
"""
Build/run/compare helper for example044 — NCCL Device API Intranode AllToAll.

Uses MPI (mpicxx) as the host compiler and mpirun to launch.

Build and run reference:
        python build_and_run.py --source ref_nccl_device_api_intranode_alltoall.cu (default 8 GPUs)
        python build_and_run.py --source ref_nccl_device_api_intranode_alltoall.cu --num-processes 8

Compare reference vs generated:
        python build_and_run.py --compare ref_nccl_device_api_intranode_alltoall.cu generated_nccl_device_api_intranode_alltoall.cu --num-processes 8

Build only:
    python build_and_run.py --source ref_nccl_device_api_intranode_alltoall.cu --build-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── unified perf-verdict bootstrap ────────────────────────────────────────
# Walk up from this file to locate run_eval/perf_verdict.py, add the repo
# root to sys.path, and import override_summary_verdict. Defaults to None
# if perf_verdict.py is unavailable so this build_and_run.py still works
# with the legacy verdict.
_HERE = os.path.dirname(os.path.abspath(__file__))
_repo = _HERE
while _repo and _repo != "/":
    if os.path.isfile(os.path.join(_repo, "run_eval", "perf_verdict.py")):
        if _repo not in sys.path:
            sys.path.insert(0, _repo)
        break
    _repo = os.path.dirname(_repo)
try:
    from run_eval.perf_verdict import override_summary_verdict as _override_summary_verdict
except ImportError:
    _override_summary_verdict = None
# ─────────────────────────────────────────────────────────────────────────


_DEFAULT_ARCH = "sm_90"
_DEFAULT_NGPUS = 8
_DEFAULT_TIMEOUT_SEC = 900


def get_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@dataclass
class BuildResult:
    success: bool
    source_file: str
    output_file: str
    return_code: int
    stdout: str
    stderr: str
    command: List[str]
    error_message: Optional[str] = None


@dataclass
class RunResult:
    success: bool
    executable: str
    return_code: int
    stdout: str
    stderr: str
    command: List[str]
    parsed_output: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class BuildAndRunResult:
    build_result: Optional[BuildResult]
    run_result: Optional[RunResult]

    @property
    def success(self) -> bool:
        build_ok = self.build_result is None or self.build_result.success
        run_ok = self.run_result is None or self.run_result.success
        return build_ok and run_ok


def _resolve_source_path(path: str) -> str:
    module_dir = get_module_dir()
    if os.path.isabs(path):
        if os.path.isfile(path):
            return path
    else:
        cand = os.path.join(module_dir, path)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError(f"Source file '{path}' not found")


def _resolve_output_path(output: Optional[str], source_abs: str) -> str:
    if output:
        if os.path.isabs(output):
            return output
        return os.path.join(get_module_dir(), output)
    stem = os.path.splitext(os.path.basename(source_abs))[0]
    return os.path.join(get_module_dir(), stem)


def _detect_nvcc() -> str:
    env_nvcc = os.environ.get("NVCC", "").strip()
    if env_nvcc and os.path.isfile(env_nvcc) and os.access(env_nvcc, os.X_OK):
        return env_nvcc
    candidates = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.2/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
    ]
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    raise RuntimeError("nvcc not found. Set $NVCC or add nvcc to PATH.")


def _detect_mpicxx() -> str:
    mpicxx = shutil.which("mpicxx")
    if mpicxx:
        return mpicxx
    candidates = ["/usr/bin/mpicxx", "/usr/local/bin/mpicxx"]
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise RuntimeError("mpicxx not found. Install OpenMPI or set PATH.")


def _detect_mpirun() -> str:
    mpirun = shutil.which("mpirun")
    if mpirun:
        return mpirun
    candidates = ["/usr/bin/mpirun", "/usr/local/bin/mpirun"]
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise RuntimeError("mpirun not found. Install OpenMPI or set PATH.")


def _parse_json_output(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    continue
    return {}


def _save_metrics_csv(metrics: List[Dict[str, Any]], csv_path: str) -> None:
    if not metrics:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    keys = list(metrics[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "count": float(n),
        "latency_avg": sum(float(m.get("latency_avg", 0.0)) for m in metrics) / n,
        "algbw_avg": sum(float(m.get("algbw", 0.0)) for m in metrics) / n,
        "busbw_avg": sum(float(m.get("busbw", 0.0)) for m in metrics) / n,
    }


def build(
    source_file: str,
    output_file: Optional[str] = None,
    compiler: Optional[str] = None,
    arch: str = _DEFAULT_ARCH,
    debug: bool = False,
    verbose: bool = True,
) -> BuildResult:
    try:
        source_abs = _resolve_source_path(source_file)
    except FileNotFoundError as exc:
        if verbose:
            print(f"[build] ERROR: {exc}", file=sys.stderr)
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file or "",
            return_code=-1, stdout="", stderr="", command=[],
            error_message=str(exc),
        )

    out_abs = _resolve_output_path(output_file, source_abs)

    try:
        nvcc = compiler or _detect_nvcc()
        mpicxx = _detect_mpicxx()
    except RuntimeError as exc:
        return BuildResult(
            success=False, source_file=source_file, output_file=out_abs,
            return_code=-1, stdout="", stderr="", command=[],
            error_message=str(exc),
        )

    opt_flags = ["-O0", "-g"] if debug else ["-O3"]
    cmd = [
        nvcc,
        *opt_flags,
        "-std=c++17",
        f"-arch={arch}",
        "--extended-lambda",
        "-DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE",
        "-ccbin", mpicxx,
        source_abs,
        "-lnccl",
        "-o", out_abs,
    ]

    env = os.environ.copy()
    env["OMPI_CXX"] = os.environ.get("OMPI_CXX", "g++-12")

    if verbose:
        print(f"[build] OMPI_CXX={env['OMPI_CXX']} {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if verbose and result.stdout:
        print(result.stdout, end="")
    if verbose and result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    return BuildResult(
        success=result.returncode == 0,
        source_file=source_file,
        output_file=out_abs,
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=cmd,
        error_message=None if result.returncode == 0 else "Compilation failed",
    )


def run(
    executable: str,
    num_processes: Optional[int] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    verbose: bool = True,
) -> RunResult:
    exe_abs = executable if os.path.isabs(executable) else os.path.join(get_module_dir(), executable)
    if not os.path.isfile(exe_abs):
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=[],
            error_message=f"Executable '{executable}' not found",
        )

    try:
        mpirun = _detect_mpirun()
    except RuntimeError as exc:
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=[],
            error_message=str(exc),
        )

    process_count = num_processes if num_processes is not None else _DEFAULT_NGPUS

    cmd = [mpirun, "--allow-run-as-root", "-np", str(process_count), exe_abs]

    if verbose:
        print(f"[run] {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout=exc.stdout or "", stderr=exc.stderr or "", command=cmd,
            error_message=f"Run timed out after {timeout_sec}s",
        )

    if verbose and result.stdout:
        print(result.stdout, end="")
    if verbose and result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    parsed = _parse_json_output(result.stdout) if result.returncode == 0 else {}

    return RunResult(
        success=result.returncode == 0,
        executable=executable,
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=cmd,
        parsed_output=parsed,
        error_message=None if result.returncode == 0 else "Execution failed",
    )


def build_and_run(
    source_file: str,
    output_file: Optional[str] = None,
    compiler: Optional[str] = None,
    arch: str = _DEFAULT_ARCH,
    debug: bool = False,
    build_only: bool = False,
    run_only: bool = False,
    num_processes: Optional[int] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    verbose: bool = True,
) -> BuildAndRunResult:
    build_result: Optional[BuildResult] = None
    run_result: Optional[RunResult] = None

    if run_only and not output_file:
        output_file = os.path.splitext(os.path.basename(source_file))[0]

    if not run_only:
        build_result = build(
            source_file=source_file,
            output_file=output_file,
            compiler=compiler,
            arch=arch,
            debug=debug,
            verbose=verbose,
        )
        if not build_result.success:
            return BuildAndRunResult(build_result=build_result, run_result=None)

    if not build_only:
        exe = build_result.output_file if build_result else output_file
        if not exe:
            exe = os.path.splitext(os.path.basename(source_file))[0]
        run_result = run(
            executable=exe,
            num_processes=num_processes,
            timeout_sec=timeout_sec,
            verbose=verbose,
        )

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str,
    compiler: Optional[str] = None,
    arch: str = _DEFAULT_ARCH,
    debug: bool = False,
    num_processes: Optional[int] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    verbose: bool = True,
) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)
    bin_dir = os.path.join(results_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    def _label(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    ref_label = _label(src_ref)
    gen_label = _label(src_gen)

    ref_out = os.path.join(bin_dir, ref_label)
    gen_out = os.path.join(bin_dir, gen_label)
    if os.path.abspath(ref_out) == os.path.abspath(gen_out):
        gen_out = gen_out + "_gen"

    if verbose:
        print(f"[compare] reference: {src_ref}")
        print(f"[compare] generated: {src_gen}")

    ref_res = build_and_run(
        source_file=src_ref, output_file=ref_out, compiler=compiler,
        arch=arch, debug=debug, num_processes=num_processes,
        timeout_sec=timeout_sec,
        verbose=verbose,
    )
    gen_res = build_and_run(
        source_file=src_gen, output_file=gen_out, compiler=compiler,
        arch=arch, debug=debug, num_processes=num_processes,
        timeout_sec=timeout_sec,
        verbose=verbose,
    )

    ref_ok = ref_res.run_result is not None and ref_res.run_result.success
    gen_ok = gen_res.run_result is not None and gen_res.run_result.success

    ref_parsed = ref_res.run_result.parsed_output if ref_ok else {}
    gen_parsed = gen_res.run_result.parsed_output if gen_ok else {}

    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])

    if ref_metrics:
        _save_metrics_csv(ref_metrics, os.path.join(results_dir, f"{ref_label}_metrics.csv"))
    if gen_metrics:
        _save_metrics_csv(gen_metrics, os.path.join(results_dir, f"{gen_label}_metrics.csv"))

    ref_avg = _metrics_avg(ref_metrics)
    gen_avg = _metrics_avg(gen_metrics)

    ref_by_size = {m.get("data_size"): m for m in ref_metrics}
    gen_by_size = {m.get("data_size"): m for m in gen_metrics}
    common_sizes = sorted(set(ref_by_size.keys()) & set(gen_by_size.keys()))

    per_size: List[Dict[str, Any]] = []
    for size in common_sizes:
        r = ref_by_size[size]
        g = gen_by_size[size]
        r_lat = float(r.get("latency_avg", 0.0))
        g_lat = float(g.get("latency_avg", 0.0))
        r_busbw = float(r.get("busbw", 0.0))
        g_busbw = float(g.get("busbw", 0.0))
        lat_impr = ((r_lat - g_lat) / r_lat * 100.0) if r_lat > 0 else 0.0
        bw_impr = ((g_busbw - r_busbw) / r_busbw * 100.0) if r_busbw > 0 else 0.0
        per_size.append({
            "data_size": size,
            "ref_latency_avg": r_lat,
            "gen_latency_avg": g_lat,
            "latency_improvement_pct": round(lat_impr, 3),
            "ref_busbw": r_busbw,
            "gen_busbw": g_busbw,
            "busbw_improvement_pct": round(bw_impr, 3),
        })

    ref_compile_ok = ref_res.build_result.success if ref_res.build_result else False
    gen_compile_ok = gen_res.build_result.success if gen_res.build_result else False
    summary: Dict[str, Any] = {
        "ref_source": os.path.basename(src_ref),
        "generated_source": os.path.basename(src_gen),
        "model": "",
        "pass_iteration": 1,
        "improvement_iteration": 1,
        "data_size_unit": "MB",
        "bandwidth_unit": "GB/s",
        "latency_unit": "us",
        "metrics_comparison": {
            "ref":       {"compile_success": ref_compile_ok, "run_success": ref_ok, **ref_avg},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_ok, **gen_avg},
        },
        "common_points": len(common_sizes),
        "per_size_comparison": per_size,
    }

    if ref_avg and gen_avg:
        r_lat = ref_avg.get("latency_avg", 0.0)
        g_lat = gen_avg.get("latency_avg", 0.0)
        r_bw = ref_avg.get("busbw_avg", 0.0)
        g_bw = gen_avg.get("busbw_avg", 0.0)
        summary["latency_improvement_pct"] = round(((r_lat - g_lat) / r_lat * 100.0) if r_lat > 0 else 0.0, 3)
        summary["busbw_improvement_pct"] = round(((g_bw - r_bw) / r_bw * 100.0) if r_bw > 0 else 0.0, 3)

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"[compare] summary saved to: {summary_path}")
        mc = summary["metrics_comparison"]
        print(f"[compare] ref run success: {mc['ref']['run_success']}")
        print(f"[compare] gen run success: {mc['generated']['run_success']}")
        if "busbw_improvement_pct" in summary and "latency_improvement_pct" in summary:
            print(f"[compare] busbw improvement: {summary['busbw_improvement_pct']:+.2f}%")
            print(f"[compare] latency improvement: {summary['latency_improvement_pct']:+.2f}%")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and run NCCL Device API Intranode AllToAll benchmark")
    parser.add_argument("--source", "-s", default="ref_nccl_device_api_intranode_alltoall.cu",
                        help="Source file (default: ref_nccl_device_api_intranode_alltoall.cu)")
    parser.add_argument("--compare", nargs=2, metavar=("SRC_REF", "SRC_GEN"),
                        help="Compare two source files")
    parser.add_argument("--output", "-o", default=None, help="Output executable path")
    parser.add_argument("--build-only", "-b", action="store_true", help="Compile only")
    parser.add_argument("--run-only", "-r", action="store_true",
                        help="Run only (expects existing executable)")
    parser.add_argument("--compiler", "-c", default=None, help="Path to nvcc")
    parser.add_argument("--arch", "-a", default=_DEFAULT_ARCH,
                        help=f"GPU arch (default: {_DEFAULT_ARCH})")
    parser.add_argument("--debug", action="store_true", help="Build with debug flags")
    parser.add_argument("--num-processes", type=int, default=None,
                        help=f"Number of MPI processes for mpirun (default: {_DEFAULT_NGPUS})")
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT_SEC,
                        help="Run timeout seconds")
    parser.add_argument("--results-dir", default=None,
                        help="Results directory (default: ./results)")
    parser.add_argument("--quiet", action="store_true", help="Less console output")
    parser.add_argument(
        "--legacy-perf-verdict",
        action="store_true",
        help="Use this example's local verdict logic instead of the shared "
             "4-tier scheme in run_eval/perf_verdict.py."
    )

    args = parser.parse_args()

    # Override summary.json's `performance` field with the shared 4-tier
    # verdict after this script finishes (atexit), unless --legacy-perf-verdict
    # was given. We scan the most likely output dirs and rewrite every
    # summary.json we find. See run_eval/perf_verdict.py.
    if not getattr(args, "legacy_perf_verdict", False) and _override_summary_verdict is not None:
        import atexit as _atexit
        def _apply_unified_verdict():
            _candidates = []
            if getattr(args, "results_dir", None):
                _candidates.append(args.results_dir)
            if getattr(args, "plot_dir", None):
                _candidates.append(args.plot_dir)
            _candidates.append(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "results"))
            _seen = set()
            for _root in _candidates:
                if not _root or _root in _seen:
                    continue
                _seen.add(_root)
                if os.path.isfile(os.path.join(_root, "summary.json")):
                    _override_summary_verdict(_root)
                if os.path.isdir(_root):
                    for _name in os.listdir(_root):
                        _sub = os.path.join(_root, _name)
                        if os.path.isdir(_sub) and \
                           os.path.isfile(os.path.join(_sub, "summary.json")):
                            _override_summary_verdict(_sub)
        _atexit.register(_apply_unified_verdict)

    verbose = not args.quiet
    results_dir = args.results_dir or os.path.join(get_module_dir(), "results")

    if args.compare:
        src_ref, src_gen = args.compare
        summary = compare(
            src_ref=src_ref, src_gen=src_gen, results_dir=results_dir,
            compiler=args.compiler, arch=args.arch, debug=args.debug,
            num_processes=args.num_processes,
            timeout_sec=args.timeout, verbose=verbose,
        )
        mc = summary.get("metrics_comparison", {})
        ref_mc = mc.get("ref", {}); gen_mc = mc.get("generated", {})
        ok = ref_mc.get("compile_success") and ref_mc.get("run_success")
        ok = ok and gen_mc.get("compile_success") and gen_mc.get("run_success")
        sys.exit(0 if ok else 1)

    result = build_and_run(
        source_file=args.source, output_file=args.output,
        compiler=args.compiler, arch=args.arch, debug=args.debug,
        build_only=args.build_only, run_only=args.run_only,
        num_processes=args.num_processes,
        timeout_sec=args.timeout, verbose=verbose,
    )

    if result.run_result and result.run_result.success and result.run_result.parsed_output:
        metrics = result.run_result.parsed_output.get("metrics", [])
        if metrics:
            label = os.path.splitext(os.path.basename(args.source))[0]
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            _save_metrics_csv(metrics, csv_path)
            if verbose:
                print(f"[run] metrics csv saved to: {csv_path}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
