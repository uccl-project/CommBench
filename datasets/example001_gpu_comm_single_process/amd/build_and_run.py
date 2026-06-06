#!/usr/bin/env python3
"""
GPU P2P Communication Build and Run Module for AMD HIP

This module compiles and runs GPU P2P communication programs.
Auto-detects compiler/platform (hipcc/nvcc) and supports both HIP and CUDA;
defaults to HIP on AMD systems.
Can be used as a standalone script or imported as a module.

Usage as module:
    from build_and_run import build, run, build_and_run, compare
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
        run_ok = self.run_result is None or self.run_result.success
        return build_ok and run_ok


def get_module_dir() -> str:
    """Get the directory where this module is located."""
    return os.path.dirname(os.path.abspath(__file__))


def _detect_compiler(platform: Optional[str] = None) -> tuple:
    """Auto-detect compiler and platform. Returns (compiler, platform)."""
    if platform == "cuda" or platform is None:
        nvcc = shutil.which("nvcc")
        if nvcc:
            return nvcc, "cuda"
    if platform == "hip" or platform is None:
        hipcc = shutil.which("hipcc")
        if hipcc:
            return hipcc, "hip"
    if platform == "cuda":
        return "nvcc", "cuda"
    if platform == "hip":
        return "hipcc", "hip"
    return "nvcc", "cuda"


def _parse_json_output(text: str) -> Dict[str, Any]:
    """Parse JSON output from stdout. The program prints exactly one JSON object."""
    if not text:
        return {}
    # Try to find JSON in the output (may have extra text before/after)
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
    Build a GPU source file.

    Args:
        source_file: Path to the source file
        output_file: Path to the output executable
        compiler: Compiler path (auto-detect if None)
        platform: "cuda" or "hip" (auto-detect if None)
        debug: Enable debug build (-g -G)
        arch: GPU architecture (e.g., sm_80 for NVIDIA, gfx90a for AMD)
        verbose: Print build progress

    Returns:
        BuildResult with success status and details
    """
    wd = get_module_dir()

    # Resolve source file path
    src_path = source_file if os.path.isabs(source_file) else os.path.join(wd, source_file)
    if not os.path.exists(src_path):
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file,
            return_code=-1, stdout="", stderr="",
            command=[], error_message=f"Source file '{source_file}' not found"
        )

    # Detect compiler/platform
    if compiler is None:
        compiler, detected_platform = _detect_compiler(platform)
        if platform is None:
            platform = detected_platform
    elif platform is None:
        if "hipcc" in compiler:
            platform = "hip"
        else:
            platform = "cuda"

    # Build flags
    flags = []
    if debug:
        flags.extend(["-g", "-G"] if platform == "cuda" else ["-g"])
    else:
        flags.append("-O2")

    if platform == "cuda":
        flags.append("-std=c++11")
        if arch:
            flags.extend(["-arch", arch])
    elif platform == "hip":
        flags.append("-std=c++11")
        if arch:
            flags.append(f"--offload-arch={arch}")

    # Resolve output path
    out_path = output_file if os.path.isabs(output_file) else os.path.join(wd, output_file)

    cmd = [compiler] + flags + [src_path, "-o", out_path]

    if verbose:
        print("===================================")
        print(f"Building ({platform.upper()})")
        print("===================================")
        print(f"Source: {source_file}")
        print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    success = result.returncode == 0
    if verbose:
        print("===================================")
        print("Build successful!" if success else "Build failed!")
        print("===================================")

    return BuildResult(
        success=success, source_file=source_file, output_file=output_file,
        return_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
        command=cmd, error_message=None if success else "Compilation failed"
    )


def run(executable: str, verbose: bool = True) -> RunResult:
    """
    Run a compiled executable.

    Args:
        executable: Path to the executable
        verbose: Print run progress

    Returns:
        RunResult with success status, output, and parsed JSON metrics
    """
    wd = get_module_dir()
    exe_path = executable if os.path.isabs(executable) else os.path.join(wd, executable)

    if not os.path.exists(exe_path):
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=[],
            error_message=f"Executable '{executable}' not found"
        )

    cmd = [exe_path]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    success = result.returncode == 0
    parsed = _parse_json_output(result.stdout) if success else {}

    return RunResult(
        success=success, executable=executable, return_code=result.returncode,
        stdout=result.stdout, stderr=result.stderr, command=cmd,
        parsed_output=parsed,
        error_message=None if success else "Execution failed"
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
    """
    Build and run a GPU source file.

    Returns:
        BuildAndRunResult containing both build and run results
    """
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
            print("Running program")
            print("===================================")
        run_result = run(executable=output_file, verbose=verbose)

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute averages of data_size, latency_avg, throughput_avg from metrics list."""
    if not metrics:
        return {}
    n = len(metrics)
    data_size_avg = sum(m.get("data_size", 0) for m in metrics) / n
    latency_avg = sum(m.get("latency_avg", 0) for m in metrics) / n
    throughput_avg = sum(m.get("throughput_avg", 0) for m in metrics) / n
    return {
        "data_size_avg": data_size_avg,
        "latency_avg": latency_avg,
        "throughput": throughput_avg,
    }


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str):
    """Save metrics list to CSV."""
    if not metrics:
        return
    keys = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label, results_dir,
                     data_size_unit="MB", latency_unit="us", throughput_unit="Gbps"):
    """Generate two comparison plots: latency and throughput vs data_size."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed - cannot generate plots.")
        return

    os.makedirs(results_dir, exist_ok=True)

    ref_sizes = [m["data_size"] for m in ref_metrics]
    ref_lat = [m["latency_avg"] for m in ref_metrics]
    ref_thr = [m["throughput_avg"] for m in ref_metrics]

    gen_sizes = [m["data_size"] for m in gen_metrics]
    gen_lat = [m["latency_avg"] for m in gen_metrics]
    gen_thr = [m["throughput_avg"] for m in gen_metrics]

    # Latency plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ref_sizes, ref_lat, marker="o", label=ref_label, linewidth=2)
    ax.plot(gen_sizes, gen_lat, marker="s", label=gen_label, linewidth=2)
    ax.set_xlabel(f"Data Size ({data_size_unit})")
    ax.set_ylabel(f"Latency ({latency_unit})")
    ax.set_title(f"Latency vs Data Size")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    lat_path = os.path.join(results_dir, "latency_comparison.png")
    fig.savefig(lat_path, dpi=150)
    plt.close(fig)
    print(f"Saved {lat_path}")

    # Throughput plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ref_sizes, ref_thr, marker="o", label=ref_label, linewidth=2)
    ax.plot(gen_sizes, gen_thr, marker="s", label=gen_label, linewidth=2)
    ax.set_xlabel(f"Data Size ({data_size_unit})")
    ax.set_ylabel(f"Throughput ({throughput_unit})")
    ax.set_title(f"Throughput vs Data Size")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    thr_path = os.path.join(results_dir, "throughput_comparison.png")
    fig.savefig(thr_path, dpi=150)
    plt.close(fig)
    print(f"Saved {thr_path}")


def _plot_single(metrics, label, results_dir,
                 data_size_unit="MB", latency_unit="us", throughput_unit="Gbps"):
    """Generate two plots for a single source: latency and throughput vs data_size."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed - cannot generate plots.")
        return

    os.makedirs(results_dir, exist_ok=True)

    sizes = [m["data_size"] for m in metrics]
    lat = [m["latency_avg"] for m in metrics]
    thr = [m["throughput_avg"] for m in metrics]

    # Latency plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, lat, marker="o", label=label, linewidth=2)
    ax.set_xlabel(f"Data Size ({data_size_unit})")
    ax.set_ylabel(f"Latency ({latency_unit})")
    ax.set_title(f"Latency vs Data Size")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    lat_path = os.path.join(results_dir, "latency.png")
    fig.savefig(lat_path, dpi=150)
    plt.close(fig)
    print(f"Saved {lat_path}")

    # Throughput plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, thr, marker="o", label=label, linewidth=2)
    ax.set_xlabel(f"Data Size ({data_size_unit})")
    ax.set_ylabel(f"Throughput ({throughput_unit})")
    ax.set_title(f"Throughput vs Data Size")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    thr_path = os.path.join(results_dir, "throughput.png")
    fig.savefig(thr_path, dpi=150)
    plt.close(fig)
    print(f"Saved {thr_path}")


def _compare_metrics(
    ref_metrics: Dict[str, Any],
    gen_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare generated metrics against reference metrics."""
    result: Dict[str, Any] = {
        "ref": ref_metrics,
        "generated": gen_metrics,
        "comparison": {},
        "summary": {},
    }
    if not ref_metrics or not gen_metrics:
        result["summary"]["status"] = "incomplete"
        return result

    common_keys = set(ref_metrics.keys()) & set(gen_metrics.keys())
    for key in common_keys:
        ref_val = ref_metrics[key]
        gen_val = gen_metrics[key]
        if isinstance(ref_val, (int, float)) and isinstance(gen_val, (int, float)) and ref_val != 0:
            ratio = gen_val / ref_val
            lower_key = key.lower()
            is_lower_better = ("latency" in lower_key or "time" in lower_key
                               or lower_key.startswith("lat_")
                               or lower_key.startswith("wall_"))
            if is_lower_better:
                improvement = (ref_val - gen_val) / ref_val * 100
                better = gen_val <= ref_val
            else:
                improvement = (gen_val - ref_val) / ref_val * 100
                better = gen_val >= ref_val
            result["comparison"][key] = {
                "ref": ref_val,
                "generated": gen_val,
                "ratio": round(ratio, 4),
                "improvement_pct": round(improvement, 2),
                "better_or_equal": better,
            }

    if result["comparison"]:
        improvements = [v["improvement_pct"] for v in result["comparison"].values()]
        worst = min(improvements)
        all_better = all(v["better_or_equal"] for v in result["comparison"].values())
        if all_better and worst >= 5:
            perf = "better"
        elif worst >= -5:
            perf = "on_par"
        elif worst >= -30:
            perf = "degraded"
        else:
            perf = "severely_degraded"
        result["summary"]["status"] = perf
        result["summary"]["all_metrics_pass"] = all_better
        result["summary"]["worst_improvement_pct"] = round(worst, 2)
    else:
        result["summary"]["status"] = "no_common_metrics"
    return result


def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    no_plot: bool = False,
    block_size: int = 256,
    verbose: bool = True,
    show_raw_output: bool = False,
) -> Dict[str, Any]:
    """Compare a generated file against the reference file.

    Builds and runs both files once, parses their JSON output,
    saves CSV, plots, and a comparison summary JSON.

    Args:
        show_raw_output: If True, print the raw stdout/stderr from each
            program execution.  Defaults to False (only comparison results
            are printed).

    Returns the summary dict.
    """
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]

    # Use show_raw_output to control whether build_and_run prints raw output
    build_run_verbose = show_raw_output

    # Build and run reference
    if verbose:
        print(f"\n{'='*60}")
        print(f"[compare] Building & running: {src_ref}  ({ref_label})")
        print(f"{'='*60}")
    ref_result = build_and_run(
        source_file=src_ref, output_file="ref_exec",
        compiler=compiler, platform=platform, debug=debug, arch=arch,
        verbose=build_run_verbose,
    )
    ref_compile_ok = ref_result.build_result is not None and ref_result.build_result.success
    ref_run_ok = ref_result.run_result is not None and ref_result.run_result.success
    ref_parsed = ref_result.run_result.parsed_output if ref_run_ok else {}

    if not ref_compile_ok and verbose:
        print(f"[compare] BUILD FAILED for {src_ref}")
    elif not ref_run_ok and verbose:
        print(f"[compare] RUN FAILED for {src_ref}")

    # Build and run generated
    if verbose:
        print(f"\n{'='*60}")
        print(f"[compare] Building & running: {src_gen}  ({gen_label})")
        print(f"{'='*60}")
    gen_result = build_and_run(
        source_file=src_gen, output_file="gen_exec",
        compiler=compiler, platform=platform, debug=debug, arch=arch,
        verbose=build_run_verbose,
    )
    gen_compile_ok = gen_result.build_result is not None and gen_result.build_result.success
    gen_run_ok = gen_result.run_result is not None and gen_result.run_result.success
    gen_parsed = gen_result.run_result.parsed_output if gen_run_ok else {}

    if not gen_compile_ok and verbose:
        print(f"[compare] BUILD FAILED for {src_gen}")
    elif not gen_run_ok and verbose:
        print(f"[compare] RUN FAILED for {src_gen}")

    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])

    # Determine units from output
    data_size_unit = ref_parsed.get("data_size_unit", gen_parsed.get("data_size_unit", "MB"))
    latency_unit = ref_parsed.get("latency_unit", gen_parsed.get("latency_unit", "us"))
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

    # Check if performance measurement is available
    has_perf = bool(ref_metrics) and bool(gen_metrics)

    # Build summary
    summary: Dict[str, Any] = {
        "generated_source": os.path.basename(src_gen),
        "ref_source": os.path.basename(src_ref),
        "model": "",
        "pass_iteration": 1,
    }

    if has_perf:
        summary["improvement_iteration"] = 1
        summary["data_size_unit"] = data_size_unit
        summary["latency_unit"] = latency_unit
        summary["throughput_unit"] = throughput_unit

        ref_avg = _metrics_avg(ref_metrics)
        gen_avg = _metrics_avg(gen_metrics)

        summary["metrics_comparison"] = {
            "ref": {
                "compile_success": ref_compile_ok,
                "run_success": ref_run_ok,
                **ref_avg,
            },
            "generated": {
                "compile_success": gen_compile_ok,
                "run_success": gen_run_ok,
                **gen_avg,
            },
        }

        # Compute improvement percentages
        if ref_avg.get("latency_avg", 0) != 0:
            lat_imp = (ref_avg["latency_avg"] - gen_avg["latency_avg"]) / ref_avg["latency_avg"] * 100
            summary["latency_improvement_pct"] = round(lat_imp, 2)
        else:
            summary["latency_improvement_pct"] = 0.0

        if ref_avg.get("throughput", 0) != 0:
            thr_imp = (gen_avg["throughput"] - ref_avg["throughput"]) / ref_avg["throughput"] * 100
            summary["throughput_improvement_pct"] = round(thr_imp, 2)
        else:
            summary["throughput_improvement_pct"] = 0.0

        # Determine performance label
        lat_imp = summary["latency_improvement_pct"]
        thr_imp = summary["throughput_improvement_pct"]
        if abs(lat_imp) < 5 and abs(thr_imp) < 5:
            summary["performance"] = "same"
        elif lat_imp > 0 and thr_imp > 0:
            summary["performance"] = "better"
        elif lat_imp < -5 or thr_imp < -5:
            summary["performance"] = "worse"
        else:
            summary["performance"] = "same"

        # Plot
        if not no_plot and ref_metrics and gen_metrics:
            _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label,
                             results_dir, data_size_unit, latency_unit, throughput_unit)

        # Print PERFORMANCE COMPARISON
        if verbose and ref_avg and gen_avg:
            # Build metric -> unit mapping from the program output
            metric_units = {}
            metric_units["data_size_avg"] = data_size_unit
            metric_units["latency_avg"] = latency_unit
            metric_units["throughput"] = throughput_unit

            comparison = _compare_metrics(ref_avg, gen_avg)
            if comparison.get("comparison"):
                print(f"\nPERFORMANCE COMPARISON (averaged over "
                      f"{len(ref_metrics)} {ref_label} / "
                      f"{len(gen_metrics)} {gen_label} records)")
                for metric, comp in comparison["comparison"].items():
                    flag = "+" if comp["better_or_equal"] else "-"
                    unit = metric_units.get(metric, "")
                    unit_str = f" {unit}" if unit else ""
                    print(f"  [{flag}] {metric}: {comp['generated']:.4f}{unit_str} vs "
                          f"ref {comp['ref']:.4f}{unit_str} "
                          f"(ratio: {comp['ratio']:.2f}, "
                          f"{comp['improvement_pct']:+.1f}%)")
                print(f"  Performance: {summary['performance']}")
    else:
        # No performance measurement
        summary["metrics_comparison"] = {
            "ref": {
                "compile_success": ref_compile_ok,
                "run_success": ref_run_ok,
            },
            "generated": {
                "compile_success": gen_compile_ok,
                "run_success": gen_run_ok,
            },
        }

    # Save summary JSON
    json_path = os.path.join(results_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"Summary saved to {json_path}")

        mc = summary["metrics_comparison"]
        print(f"\n{'='*60}")
        print(f"  ref       compile_success: {mc['ref']['compile_success']}")
        print(f"  ref       run_success:     {mc['ref']['run_success']}")
        print(f"  generated compile_success: {mc['generated']['compile_success']}")
        print(f"  generated run_success:     {mc['generated']['run_success']}")
        print(f"  performance:               {summary.get('performance', 'N/A')}")
        print(f"{'='*60}")

    return summary


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build and run GPU P2P communication program"
    )
    parser.add_argument(
        "--source", "-s",
        default="ref_gpu_p2p_comm.cpp",
        help="Source file to compile (default: ref_gpu_p2p_comm.cpp)"
    )
    parser.add_argument(
        "--output", "-o",
        default="gpu_p2p_comm",
        help="Output executable name (default: gpu_p2p_comm)"
    )
    parser.add_argument(
        "--arch", "-a",
        default=None,
        help="GPU architecture (e.g., gfx90a for AMD, sm_80 for NVIDIA)"
    )
    parser.add_argument(
        "--build-only", "-b",
        action="store_true",
        help="Only build, do not run"
    )
    parser.add_argument(
        "--run-only", "-r",
        action="store_true",
        help="Only run (assume already built)"
    )
    parser.add_argument(
        "--compiler", "-c",
        default=None,
        help="Specify compiler path (auto-detect if not specified)"
    )
    parser.add_argument(
        "--platform", "-p",
        choices=["hip", "cuda"],
        default=None,
        help="Force platform (hip or cuda)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate benchmark plots after running"
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory to save results (default: ./results)"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("SRC_A", "SRC_B"),
        default=None,
        help="Compare two source files. Builds and runs both, then generates "
             "comparison plots. Implies --plot.  "
             "Example: --compare ref_gpu_p2p_comm.cpp generated_gpu_p2p_comm.cpp"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug build"
    )
    parser.add_argument(
        "--show-raw-output",
        action="store_true",
        help="In compare mode, also print the raw stdout/stderr from each "
             "program execution (default: only comparison results are shown)"
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

    # --compare mode
    if args.compare:
        src_a, src_b = args.compare
        wd = get_module_dir()
        for src in (src_a, src_b):
            path = src if os.path.isabs(src) else os.path.join(wd, src)
            if not os.path.exists(path):
                print(f"Error: Source file '{src}' not found!")
                sys.exit(1)
        compare(
            src_ref=src_a, src_gen=src_b,
            results_dir=results_dir,
            compiler=args.compiler, platform=args.platform,
            debug=args.debug, arch=args.arch,
            verbose=verbose,
            show_raw_output=args.show_raw_output,
        )
        sys.exit(0)

    # Normal build-and-run
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

    # --plot mode for single source
    if args.plot and result.run_result and result.run_result.parsed_output:
        parsed = result.run_result.parsed_output
        metrics = parsed.get("metrics", [])
        if metrics:
            label = os.path.splitext(os.path.basename(args.source))[0]
            _plot_single(
                metrics, label, results_dir,
                parsed.get("data_size_unit", "MB"),
                parsed.get("latency_unit", "us"),
                parsed.get("throughput_unit", "Gbps"),
            )
            # Save CSV
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            os.makedirs(results_dir, exist_ok=True)
            _save_metrics_csv(metrics, csv_path)
            if verbose:
                print(f"Saved {csv_path}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
