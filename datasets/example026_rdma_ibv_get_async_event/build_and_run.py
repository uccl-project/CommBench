#!/usr/bin/env python3
"""
RDMA ibv_get_async_event Build and Run Module

Compiles and runs RDMA async event monitoring programs using g++ with libibverbs.
Can be used as a standalone script or imported as a module.

Usage as module:
    from build_and_run import build, run, build_and_run, compare

Usage as script:
    # Single file mode
    python build_and_run.py --source ref_rdma_ibv_get_async_event.cpp

    # Compare mode
    python build_and_run.py --compare ref_rdma_ibv_get_async_event.cpp \\
                                      generated_rdma_ibv_get_async_event_xxx.cpp
"""

import subprocess
import sys
import os
import json
import csv
import shutil
import argparse
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


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class BuildResult:
    """Result of a build operation."""
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
    """Result of a run operation."""
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
    """Result of combined build and run operation."""
    build_result: Optional[BuildResult]
    run_result: Optional[RunResult]

    @property
    def success(self) -> bool:
        build_ok = self.build_result is None or self.build_result.success
        run_ok   = self.run_result   is None or self.run_result.success
        return build_ok and run_ok


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_module_dir() -> str:
    """Return the directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


def _detect_compiler(compiler: Optional[str] = None) -> str:
    """Return the C++ compiler to use (g++ preferred, falls back to c++)."""
    if compiler:
        return compiler
    for cc in ("g++", "c++", "clang++"):
        if shutil.which(cc):
            return cc
    return "g++"


def _parse_json_output(text: str) -> Dict[str, Any]:
    """Extract the first complete JSON object from program stdout."""
    if not text:
        return {}
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    continue
    return {}


# ── Core API ─────────────────────────────────────────────────────────────────

def build(
    source_file: str,
    output_file: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,   # accepted but ignored (always C++)
    debug: bool = False,
    arch: Optional[str] = None,        # accepted but ignored for C++
    verbose: bool = True,
) -> BuildResult:
    """
    Compile a C++ RDMA source file with libibverbs.

    Args:
        source_file: Path to the .cpp source file
        output_file: Name of the output executable
        compiler:    Compiler to use (auto-detected if None)
        platform:    Ignored for C++ files (kept for API compatibility)
        debug:       Enable -g debug build
        arch:        Ignored for C++ files (kept for API compatibility)
        verbose:     Print build progress

    Returns:
        BuildResult with success status and details
    """
    wd = get_module_dir()

    src_path = source_file if os.path.isabs(source_file) \
               else os.path.join(wd, source_file)
    if not os.path.exists(src_path):
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file,
            return_code=-1, stdout="", stderr="", command=[],
            error_message=f"Source file '{source_file}' not found",
        )

    cc = _detect_compiler(compiler)

    flags = ["-g"] if debug else ["-O2"]
    flags += ["-std=c++17", "-Wall", "-pthread"]

    out_path = output_file if os.path.isabs(output_file) \
               else os.path.join(wd, output_file)

    cmd = [cc] + flags + [src_path, "-o", out_path, "-libverbs"]

    if verbose:
        print("=" * 60)
        print("Building (C++ / libibverbs)")
        print("=" * 60)
        print(f"Source : {source_file}")
        print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    success = result.returncode == 0
    if verbose:
        print("Build successful!" if success else "Build FAILED!")
        print("=" * 60)

    return BuildResult(
        success=success,
        source_file=source_file,
        output_file=output_file,
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=cmd,
        error_message=None if success else "Compilation failed",
    )


def run(executable: str, verbose: bool = True) -> RunResult:
    """
    Run a compiled RDMA executable and parse its JSON output.

    Args:
        executable: Path to the executable
        verbose:    Print run progress

    Returns:
        RunResult with success status, stdout/stderr, and parsed JSON metrics
    """
    wd = get_module_dir()
    exe_path = executable if os.path.isabs(executable) \
               else os.path.join(wd, executable)

    if not os.path.exists(exe_path):
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=[],
            error_message=f"Executable '{executable}' not found",
        )

    cmd = [exe_path]
    if verbose:
        print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    success = result.returncode == 0
    parsed  = _parse_json_output(result.stdout) if success else {}

    return RunResult(
        success=success,
        executable=executable,
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=cmd,
        parsed_output=parsed,
        error_message=None if success else "Execution failed",
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
    """Build and optionally run a C++ RDMA source file."""
    build_result = None
    run_result   = None

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
            print("=" * 60)
            print("Running program")
            print("=" * 60)
        run_result = run(executable=output_file, verbose=verbose)

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


# ── Compare mode helpers ──────────────────────────────────────────────────────

def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute averages of latency_avg (and data_size) from a metrics list."""
    if not metrics:
        return {}
    n = len(metrics)
    data_size_avg = sum(m.get("data_size", 0)    for m in metrics) / n
    latency_avg   = sum(m.get("latency_avg", 0)  for m in metrics) / n
    return {
        "data_size_avg": data_size_avg,
        "latency_avg":   latency_avg,
        "throughput":    0.0,   # not applicable; kept for API compatibility
    }


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str):
    """Save a metrics list to CSV."""
    if not metrics:
        return
    keys = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label,
                     results_dir, latency_unit="us"):
    """Plot latency comparison (detection latency vs trial)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        return

    os.makedirs(results_dir, exist_ok=True)

    ref_trials = [m["data_size"]   for m in ref_metrics]
    ref_lat    = [m["latency_avg"] for m in ref_metrics]
    gen_trials = [m["data_size"]   for m in gen_metrics]
    gen_lat    = [m["latency_avg"] for m in gen_metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ref_trials, ref_lat, marker="o", label=ref_label, linewidth=2)
    ax.plot(gen_trials, gen_lat, marker="s", label=gen_label, linewidth=2)
    ax.set_xlabel("Trial")
    ax.set_ylabel(f"Event Detection Latency ({latency_unit})")
    ax.set_title("Async Event Detection Latency Comparison")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    lat_path = os.path.join(results_dir, "latency_comparison.png")
    fig.savefig(lat_path, dpi=150)
    plt.close(fig)
    print(f"Saved {lat_path}")

    # Throughput plot (placeholder — not applicable here, saved as empty)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Throughput (N/A for async event benchmark)")
    ax.text(0.5, 0.5, "Not applicable", transform=ax.transAxes,
            ha="center", va="center", fontsize=14, color="grey")
    fig.tight_layout()
    thr_path = os.path.join(results_dir, "throughput_comparison.png")
    fig.savefig(thr_path, dpi=150)
    plt.close(fig)
    print(f"Saved {thr_path}")


def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    no_plot: bool = False,
    verbose: bool = True,
    show_raw_output: bool = False,
) -> Dict[str, Any]:
    """
    Build and run both reference and generated files, compare latency metrics,
    save CSV, plots, and a summary.json.

    Returns the summary dict.
    """
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]

    bav = show_raw_output  # build_and_run verbose

    # ── Build & run reference ────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"[compare] Building & running: {src_ref}")
        print(f"{'='*60}")
    ref_result = build_and_run(
        source_file=src_ref, output_file="ref_exec",
        compiler=compiler, platform=platform, debug=debug, arch=arch,
        verbose=bav,
    )
    ref_compile_ok = ref_result.build_result is not None and ref_result.build_result.success
    ref_run_ok     = ref_result.run_result   is not None and ref_result.run_result.success
    ref_parsed     = ref_result.run_result.parsed_output if ref_run_ok else {}

    if not ref_compile_ok and verbose:
        print(f"[compare] BUILD FAILED for {src_ref}")
    elif not ref_run_ok and verbose:
        print(f"[compare] RUN FAILED for {src_ref}")

    # ── Build & run generated ────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"[compare] Building & running: {src_gen}")
        print(f"{'='*60}")
    gen_result = build_and_run(
        source_file=src_gen, output_file="gen_exec",
        compiler=compiler, platform=platform, debug=debug, arch=arch,
        verbose=bav,
    )
    gen_compile_ok = gen_result.build_result is not None and gen_result.build_result.success
    gen_run_ok     = gen_result.run_result   is not None and gen_result.run_result.success
    gen_parsed     = gen_result.run_result.parsed_output if gen_run_ok else {}

    if not gen_compile_ok and verbose:
        print(f"[compare] BUILD FAILED for {src_gen}")
    elif not gen_run_ok and verbose:
        print(f"[compare] RUN FAILED for {src_gen}")

    # ── Extract metrics ──────────────────────────────────────────────────────
    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])
    latency_unit = ref_parsed.get("latency_unit",
                   gen_parsed.get("latency_unit", "us"))

    has_perf = bool(ref_metrics) and bool(gen_metrics)

    # Save CSV files
    if ref_metrics:
        ref_csv = os.path.join(results_dir, f"{ref_label}_metrics.csv")
        _save_metrics_csv(ref_metrics, ref_csv)
        if verbose:
            print(f"Saved {ref_csv}")
    if gen_metrics:
        gen_csv = os.path.join(results_dir, f"{gen_label}_metrics.csv")
        _save_metrics_csv(gen_metrics, gen_csv)
        if verbose:
            print(f"Saved {gen_csv}")

    # ── Build summary ────────────────────────────────────────────────────────
    summary: Dict[str, Any] = {
        "generated_source": os.path.basename(src_gen),
        "ref_source":       os.path.basename(src_ref),
        "model":            "",
        "pass_iteration":   1,
    }

    if has_perf:
        ref_avg = _metrics_avg(ref_metrics)
        gen_avg = _metrics_avg(gen_metrics)

        summary["improvement_iteration"] = 1
        summary["latency_unit"]          = latency_unit
        summary["data_size_unit"]        = "trial"
        summary["throughput_unit"]       = "N/A"

        summary["metrics_comparison"] = {
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
        }

        # Latency improvement (lower is better)
        ref_lat = ref_avg.get("latency_avg", 0)
        gen_lat = gen_avg.get("latency_avg", 0)
        if ref_lat != 0:
            lat_imp = (ref_lat - gen_lat) / ref_lat * 100
        else:
            lat_imp = 0.0
        summary["latency_improvement_pct"]  = round(lat_imp, 2)
        summary["throughput_improvement_pct"] = 0.0

        if abs(lat_imp) < 5:
            summary["performance"] = "same"
        elif lat_imp > 0:
            summary["performance"] = "better"
        else:
            summary["performance"] = "worse"

        # Generate plots
        if not no_plot and ref_metrics and gen_metrics:
            _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label,
                             results_dir, latency_unit)

        # Print comparison table
        if verbose and ref_avg and gen_avg:
            print(f"\nPERFORMANCE COMPARISON (event detection latency, {latency_unit})")
            print(f"\n  [+] data_size_avg: {gen_avg['data_size_avg']:.1f} trial")
            flag = "+" if lat_imp >= 0 else "-"
            print(f"  [{flag}] latency_avg:   {gen_avg['latency_avg']:.2f} {latency_unit}"
                  f" vs ref {ref_avg['latency_avg']:.2f} {latency_unit}"
                  f" ({lat_imp:+.1f}%)")
            print(f"  Performance: {summary['performance']}")

    else:
        summary["metrics_comparison"] = {
            "ref": {
                "compile_success": ref_compile_ok,
                "run_success":     ref_run_ok,
            },
            "generated": {
                "compile_success": gen_compile_ok,
                "run_success":     gen_run_ok,
            },
        }

    # Save summary.json
    json_path = os.path.join(results_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"\nSummary saved to {json_path}")

    mc = summary["metrics_comparison"]
    print(f"\n{'='*60}")
    print(f"  ref       compile_success: {mc['ref']['compile_success']}")
    print(f"  ref       run_success:     {mc['ref']['run_success']}")
    print(f"  generated compile_success: {mc['generated']['compile_success']}")
    print(f"  generated run_success:     {mc['generated']['run_success']}")
    print(f"  performance:               {summary.get('performance', 'N/A')}")
    print(f"{'='*60}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build and run RDMA ibv_get_async_event benchmark"
    )
    parser.add_argument(
        "--source", "-s",
        default="ref_rdma_ibv_get_async_event.cpp",
        help="Source file to compile (default: ref_rdma_ibv_get_async_event.cpp)"
    )
    parser.add_argument(
        "--output", "-o",
        default="rdma_ibv_get_async_event",
        help="Output executable name"
    )
    parser.add_argument(
        "--arch", "-a",
        default=None,
        help="GPU architecture (not used; kept for API compatibility)"
    )
    parser.add_argument(
        "--build-only", "-b",
        action="store_true",
        help="Only compile, do not run"
    )
    parser.add_argument(
        "--run-only", "-r",
        action="store_true",
        help="Only run an already-compiled executable"
    )
    parser.add_argument(
        "--compiler", "-c",
        default=None,
        help="C++ compiler path (auto-detected if not specified)"
    )
    parser.add_argument(
        "--platform", "-p",
        default=None,
        help="Ignored for C++ RDMA builds; kept for API compatibility"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate latency plots after running (single-file mode)"
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory to save results (default: ./results)"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("SRC_REF", "SRC_GEN"),
        default=None,
        help="Compare two source files: ref and generated"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug build (-g)"
    )
    parser.add_argument(
        "--show-raw-output",
        action="store_true",
        help="In compare mode, print raw stdout/stderr from each run"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )

    parser.add_argument(
        "--legacy-perf-verdict",
        action="store_true",
        help="Use this example's local verdict logic instead of the shared "
             "4-tier scheme in run_eval/perf_verdict.py."
    )
    args    = parser.parse_args()

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
    results_dir = args.results_dir or os.path.join(get_module_dir(), "results")

    # ── compare mode ─────────────────────────────────────────────────────────
    if args.compare:
        src_a, src_b = args.compare
        wd = get_module_dir()
        for src in (src_a, src_b):
            path = src if os.path.isabs(src) else os.path.join(wd, src)
            if not os.path.exists(path):
                print(f"Error: source file '{src}' not found!")
                sys.exit(1)
        compare(
            src_ref=src_a, src_gen=src_b,
            results_dir=results_dir,
            compiler=args.compiler,
            platform=args.platform,
            debug=args.debug,
            arch=args.arch,
            verbose=verbose,
            show_raw_output=args.show_raw_output,
        )
        sys.exit(0)

    # ── single-file mode ──────────────────────────────────────────────────────
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

    if args.plot and result.run_result and result.run_result.parsed_output:
        parsed  = result.run_result.parsed_output
        metrics = parsed.get("metrics", [])
        if metrics:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib not installed — skipping plot.")
                sys.exit(0 if result.success else 1)

            os.makedirs(results_dir, exist_ok=True)
            label   = os.path.splitext(os.path.basename(args.source))[0]
            trials  = [m["data_size"]   for m in metrics]
            lats    = [m["latency_avg"] for m in metrics]
            lat_unit = parsed.get("latency_unit", "us")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(trials, lats, marker="o", label=label, linewidth=2)
            ax.set_xlabel("Trial")
            ax.set_ylabel(f"Detection Latency ({lat_unit})")
            ax.set_title("Async Event Detection Latency")
            ax.legend()
            ax.grid(True, ls="--", alpha=0.5)
            fig.tight_layout()
            lat_path = os.path.join(results_dir, "latency.png")
            fig.savefig(lat_path, dpi=150)
            plt.close(fig)
            print(f"Saved {lat_path}")

            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            _save_metrics_csv(metrics, csv_path)
            print(f"Saved {csv_path}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
