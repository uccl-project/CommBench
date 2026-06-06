#!/usr/bin/env python3
"""
RDMA UC SEND/RECV – Build, Run, and Compare Module

Compiles and runs RDMA UC SEND/RECV benchmark programs using libibverbs.
Can be used as a library (import build, run, build_and_run, compare) or
as a CLI tool.

Usage examples:
  python build_and_run.py --source ref_rdma_uc_send_recv.cpp
  python build_and_run.py --build-only --source ref_rdma_uc_send_recv.cpp
  python build_and_run.py --compare ref_rdma_uc_send_recv.cpp generated_rdma_uc_send_recv.cpp
"""

import subprocess
import sys
import os
import json
import csv
import argparse
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

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


# ============================================================================
# Result data classes
# ============================================================================

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


# ============================================================================
# Helpers
# ============================================================================

def get_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _detect_compiler(platform: Optional[str] = None):
    """Auto-detect compiler.
    Use g++ for RDMA + CUDA runtime API sources (no CUDA kernels are needed,
    so nvcc is not required; linking -lcudart is sufficient).
    """
    compiler = shutil.which("g++") or "g++"
    return compiler, platform or "ib"


def _parse_json_output(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    brace_depth = 0
    json_start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and json_start is not None:
                try:
                    return json.loads(text[json_start:i + 1])
                except json.JSONDecodeError:
                    continue
    return {}


# ============================================================================
# Core API
# ============================================================================

def build(
    source_file: str,
    output_file: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    verbose: bool = True,
) -> BuildResult:
    """
    Compile an RDMA UC SEND/RECV source file.

    Args:
        source_file: Path to the source file (.cpp)
        output_file: Path to the output executable
        compiler:    Compiler path (auto-detect g++ if None)
        platform:    Platform hint ("ib" or "efa"); informational only
        debug:       Enable debug build (-g)
        arch:        Unused for RDMA; accepted for API compatibility
        verbose:     Print build progress
    """
    wd = get_module_dir()
    src_path = source_file if os.path.isabs(source_file) else os.path.join(wd, source_file)

    if not os.path.exists(src_path):
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file,
            return_code=-1, stdout="", stderr="", command=[],
            error_message=f"Source file '{source_file}' not found",
        )

    if compiler is None:
        compiler, _ = _detect_compiler(platform)

    flags = ["-g"] if debug else ["-O2"]
    flags += ["-std=c++17", "-Wall"]
    # Link CUDA runtime (for cudaMalloc/cudaFree/cudaMemcpy/cudaMemset) and
    # ibverbs + pthread for RDMA.  No CUDA kernels are used, so g++ suffices.
    link_flags = ["-libverbs", "-pthread", "-lcudart"]

    out_path = output_file if os.path.isabs(output_file) else os.path.join(wd, output_file)
    cmd = [compiler] + flags + [src_path, "-o", out_path] + link_flags

    if verbose:
        print("===================================")
        print("Building (RDMA / libibverbs)")
        print("===================================")
        print(f"Source:  {source_file}")
        print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    success = result.returncode == 0

    if verbose:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print("===================================")
        print("Build successful!" if success else "Build failed!")
        print("===================================")

    return BuildResult(
        success=success, source_file=source_file, output_file=output_file,
        return_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
        command=cmd,
        error_message=None if success else "Compilation failed",
    )


def run(executable: str, verbose: bool = True) -> RunResult:
    """
    Run a compiled RDMA UC SEND/RECV benchmark executable.

    The executable auto-detects the UC-capable RDMA device and uses
    built-in defaults for all parameters.

    Args:
        executable: Path to the compiled executable
        verbose:    Print run progress and stderr diagnostics
    """
    wd = get_module_dir()
    exe_path = executable if os.path.isabs(executable) else os.path.join(wd, executable)

    if not os.path.exists(exe_path):
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=[],
            error_message=f"Executable '{executable}' not found",
        )

    cmd = [exe_path]

    if verbose:
        print(f"Running: {exe_path}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    parsed = _parse_json_output(result.stdout)
    correctness = parsed.get("Correctness", "") if parsed else ""
    success = result.returncode == 0 and correctness == "PASS"

    if verbose and result.stderr:
        for line in result.stderr.strip().splitlines():
            print(f"  [stderr] {line}", file=sys.stderr)

    return RunResult(
        success=success, executable=executable,
        return_code=result.returncode,
        stdout=result.stdout, stderr=result.stderr,
        command=cmd, parsed_output=parsed,
        error_message=None if success else "Run failed or correctness FAIL",
    )


def build_and_run(
    source_file: str,
    output_file: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    build_only: bool = False,
    run_only: bool = False,
    verbose: bool = True,
) -> BuildAndRunResult:
    """Build then run an RDMA UC SEND/RECV benchmark source file."""
    build_result = None
    run_result = None

    if not run_only:
        build_result = build(
            source_file=source_file, output_file=output_file,
            compiler=compiler, platform=platform,
            debug=debug, arch=arch, verbose=verbose,
        )
        if not build_result.success:
            return BuildAndRunResult(build_result=build_result, run_result=None)

    if not build_only:
        if verbose:
            print()
            print("===================================")
            print("Running benchmark")
            print("===================================")
        run_result = run(executable=output_file, verbose=verbose)

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


# ============================================================================
# Compare helpers
# ============================================================================

def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "data_size_avg":  sum(m.get("data_size", 0)      for m in metrics) / n,
        "latency_avg":    sum(m.get("latency_avg", 0)    for m in metrics) / n,
        "throughput_avg": sum(m.get("throughput_avg", 0) for m in metrics) / n,
    }


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str) -> None:
    if not metrics:
        return
    keys = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def _plot_comparison(
    ref_metrics, gen_metrics, ref_label, gen_label, results_dir,
    data_size_unit="Bytes", latency_unit="us", throughput_unit="Gbps",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed – skipping plots", file=sys.stderr)
        return

    os.makedirs(results_dir, exist_ok=True)

    for key, ylabel, fname in [
        ("latency_avg",    f"Latency ({latency_unit})",       "latency_comparison.png"),
        ("throughput_avg", f"Throughput ({throughput_unit})", "throughput_comparison.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, data, marker in [
            (ref_label, ref_metrics, "o"),
            (gen_label, gen_metrics, "s"),
        ]:
            xs = [d["data_size"] for d in data if key in d]
            ys = [d[key]         for d in data if key in d]
            if xs:
                ax.plot(xs, ys, marker=marker, label=label, linewidth=2)
        ax.set_xlabel(f"Data size ({data_size_unit})")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_title(f"Example 7 (UC SEND/RECV) – {ylabel} vs data size")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        path = os.path.join(results_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


# ============================================================================
# Compare
# ============================================================================

def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str = "results",
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    no_plot: bool = False,
    verbose: bool = True,
    show_raw_output: bool = False,
) -> Dict[str, Any]:
    """Build & run ref and generated sources, compare and save results."""
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]

    build_run_verbose = show_raw_output

    # --- Reference ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"[compare] Building & running: {src_ref}  ({ref_label})")
        print(f"{'='*60}")
    ref_result = build_and_run(
        source_file=src_ref, output_file="ref_exec",
        compiler=compiler, platform=platform,
        debug=debug, arch=arch, verbose=build_run_verbose,
    )
    ref_compile_ok = ref_result.build_result is not None and ref_result.build_result.success
    ref_run_ok     = ref_result.run_result   is not None and ref_result.run_result.success
    ref_parsed     = ref_result.run_result.parsed_output if ref_run_ok else {}

    # --- Generated ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"[compare] Building & running: {src_gen}  ({gen_label})")
        print(f"{'='*60}")
    gen_result = build_and_run(
        source_file=src_gen, output_file="gen_exec",
        compiler=compiler, platform=platform,
        debug=debug, arch=arch, verbose=build_run_verbose,
    )
    gen_compile_ok = gen_result.build_result is not None and gen_result.build_result.success
    gen_run_ok     = gen_result.run_result   is not None and gen_result.run_result.success
    gen_parsed     = gen_result.run_result.parsed_output if gen_run_ok else {}

    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])

    data_size_unit  = ref_parsed.get("data_size_unit",  gen_parsed.get("data_size_unit",  "Bytes"))
    latency_unit    = ref_parsed.get("latency_unit",    gen_parsed.get("latency_unit",    "us"))
    throughput_unit = ref_parsed.get("throughput_unit", gen_parsed.get("throughput_unit", "Gbps"))

    # Save CSV
    if ref_metrics:
        ref_csv = os.path.join(results_dir, f"{ref_label}_metrics.csv")
        _save_metrics_csv(ref_metrics, ref_csv)
        if verbose:
            print(f"Raw metrics saved to {ref_csv}")
    if gen_metrics:
        gen_csv = os.path.join(results_dir, f"{gen_label}_metrics.csv")
        _save_metrics_csv(gen_metrics, gen_csv)
        if verbose:
            print(f"Raw metrics saved to {gen_csv}")

    has_perf = bool(ref_metrics) and bool(gen_metrics)
    ref_avg  = _metrics_avg(ref_metrics)
    gen_avg  = _metrics_avg(gen_metrics)

    # Improvement percentages
    lat_imp = 0.0
    thr_imp = 0.0
    if has_perf:
        if ref_avg.get("latency_avg", 0) != 0:
            lat_imp = (ref_avg["latency_avg"] - gen_avg["latency_avg"]) / ref_avg["latency_avg"] * 100
        if ref_avg.get("throughput_avg", 0) != 0:
            thr_imp = (gen_avg["throughput_avg"] - ref_avg["throughput_avg"]) / ref_avg["throughput_avg"] * 100

    if abs(lat_imp) < 5 and abs(thr_imp) < 5:
        performance = "same"
    elif lat_imp > 0 and thr_imp > 0:
        performance = "better"
    elif lat_imp < -5 or thr_imp < -5:
        performance = "worse"
    else:
        performance = "same"

    # Build summary
    summary: Dict[str, Any] = {
        "generated_source":       os.path.basename(src_gen),
        "ref_source":             os.path.basename(src_ref),
        "model":                  "unknown",
        "pass_iteration":         1,
        "improvement_iteration":  1,
        "data_size_unit":         data_size_unit,
        "latency_unit":           latency_unit,
        "throughput_unit":        throughput_unit,
        "metrics_comparison": {
            "ref": {
                "compile_success": ref_compile_ok,
                "run_success":     ref_run_ok,
                **ref_avg,
            },
            "generated": {
                "compile_success": gen_compile_ok,
                "run_success":     gen_run_ok,
                **gen_avg,
            },
        },
        "latency_improvement_pct":    round(lat_imp, 2),
        "throughput_improvement_pct": round(thr_imp, 2),
        "performance":                performance,
    }

    # Save summary JSON
    json_path = os.path.join(results_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"Summary saved to {json_path}")

    # Plots
    if not no_plot and has_perf:
        _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label,
                         results_dir, data_size_unit, latency_unit, throughput_unit)

    # Required printed comparison summary
    last_gen = gen_metrics[-1] if gen_metrics else {}
    print(f"\nPERFORMANCE COMPARISON (ref vs generated)\n")
    print(f"[+] data_size_avg: {last_gen.get('data_size', 'N/A')} {data_size_unit}")
    print(f"[+] throughput: {last_gen.get('throughput_avg', 'N/A')} {throughput_unit}")
    print(f"[+] latency_avg: {last_gen.get('latency_avg', 'N/A')} {latency_unit}")
    print(f"Performance: {performance}\n")
    print("=" * 60)
    mc = summary["metrics_comparison"]
    print(f"ref       compile_success: {mc['ref']['compile_success']}")
    print(f"ref       run_success:     {mc['ref']['run_success']}")
    print(f"generated compile_success: {mc['generated']['compile_success']}")
    print(f"generated run_success:     {mc['generated']['run_success']}")
    print(f"performance:               {performance}")
    print("=" * 60)

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build and run the RDMA UC SEND/RECV benchmark"
    )
    parser.add_argument("--source", "-s", default="ref_rdma_uc_send_recv.cpp",
                        help="Source file to compile")
    parser.add_argument("--output", "-o", default="rdma_uc_send_recv",
                        help="Output executable name")
    parser.add_argument("--arch", default=None,
                        help="Target architecture hint (unused for RDMA; kept for API compatibility)")
    parser.add_argument("--build-only", action="store_true",
                        help="Compile only, do not run")
    parser.add_argument("--run-only", action="store_true",
                        help="Run existing executable without compiling")
    parser.add_argument("--compiler", default=None,
                        help="C++ compiler path (default: auto-detect g++)")
    parser.add_argument("--platform", default=None,
                        choices=["ib", "efa"],
                        help="RDMA NIC platform hint (ib or efa)")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Generate performance plots after running")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--results-dir", default="results",
                        help="Directory for saving CSV, plots, and summary JSON")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "GEN"),
                        help="Compare mode: --compare ref.cpp generated.cpp")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug build")
    parser.add_argument("--show-raw-output", action="store_true",
                        help="In compare mode, also print raw stdout/stderr from each run")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--quiet", "-q", action="store_true")

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
    # summary.json we find — runs whether compare() was invoked from CLI,
    # Python API, or anywhere else in this script. See run_eval/perf_verdict.py.
    if not args.legacy_perf_verdict and _override_summary_verdict is not None:
        import atexit as _atexit
        def _apply_unified_verdict():
            _candidates = []
            if getattr(args, "results_dir", None):
                _candidates.append(args.results_dir)
            if getattr(args, "plot_dir", None):
                _candidates.append(args.plot_dir)
            _candidates.append(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "results"))
            # Also walk one level deep — many scripts write into
            # results/<subdir>/summary.json — to catch nested layouts.
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
    wd = get_module_dir()
    results_dir = args.results_dir if os.path.isabs(args.results_dir) \
                  else os.path.join(wd, args.results_dir)

    if args.compare:
        src_ref, src_gen = args.compare
        compare(
            src_ref=src_ref, src_gen=src_gen,
            results_dir=results_dir,
            compiler=args.compiler, platform=args.platform,
            debug=args.debug, arch=args.arch,
            no_plot=not args.plot,
            verbose=verbose,
            show_raw_output=args.show_raw_output,
        )
        sys.exit(0)

    result = build_and_run(
        source_file=args.source,
        output_file=args.output,
        compiler=args.compiler,
        platform=args.platform,
        debug=args.debug,
        arch=args.arch,
        build_only=args.build_only,
        run_only=args.run_only,
        verbose=verbose,
    )

    br = result.build_result
    rr = result.run_result

    if br is not None:
        print("Build successful" if br.success else "Compilation failed")
        if not br.success and br.stderr:
            print(br.stderr, file=sys.stderr)

    if rr is not None:
        if rr.success:
            print("PASS")
            if rr.stdout:
                print(rr.stdout)
        else:
            print("FAIL")
            if rr.stderr:
                print(rr.stderr, file=sys.stderr)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
