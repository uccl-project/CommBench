#!/usr/bin/env python3
"""
GPU Globaltimer Nanosleep Build and Run Module for NVIDIA CUDA.

Compiles and runs the example73_GPU_globaltimer_nanosleep reference /
generated sources, parses their JSON metrics, saves CSV + plots +
summary.json, and prints a comparison summary in compare mode.

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


def get_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


# Prefer a newer CUDA toolchain if it is installed alongside an older system one,
# because the example uses Hopper/Blackwell mbarrier PTX that benefits from a
# recent ptxas.
_NVCC_CANDIDATES = [
    "/usr/local/cuda-13.2/bin/nvcc",
    "/usr/local/cuda-13/bin/nvcc",
    "/usr/local/cuda-12.4/bin/nvcc",
    "/usr/local/cuda/bin/nvcc",
]


def _detect_compiler(platform: Optional[str] = None) -> tuple:
    if platform == "cuda" or platform is None:
        for cand in _NVCC_CANDIDATES:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand, "cuda"
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


def _cuda_compat_isystem_args() -> list:
    """Return -isystem flags pointing at the bundled stub for
    /usr/include/aarch64-linux-gnu/bits/math-vector.h that nvcc < 12.4
    cannot parse on aarch64. Empty list when the stub is unavailable."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "_cuda_compat"),
        os.path.join(here, "..", "_cuda_compat"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "bits", "math-vector.h")):
            return ["-isystem", os.path.abspath(c)]
    return []


def build(
    source_file: str,
    output_file: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    verbose: bool = True,
) -> BuildResult:
    wd = get_module_dir()

    src_path = source_file if os.path.isabs(source_file) else os.path.join(wd, source_file)
    if not os.path.exists(src_path):
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file,
            return_code=-1, stdout="", stderr="",
            command=[], error_message=f"Source file '{source_file}' not found"
        )

    if compiler is None:
        compiler, detected_platform = _detect_compiler(platform)
        if platform is None:
            platform = detected_platform
    elif platform is None:
        platform = "hip" if "hipcc" in compiler else "cuda"

    flags: List[str] = []
    if debug:
        flags.extend(["-g", "-G"] if platform == "cuda" else ["-g"])
    else:
        flags.append("-O2")

    if platform == "cuda":
        flags.extend(["-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"])
        # Default arch: sm_90 — only generic system-scope acquire/release PTX
        # is used; the binary JITs forward onto Blackwell at load time.
        flags.extend(["-arch", arch or "sm_90"])
        flags += _cuda_compat_isystem_args()
    elif platform == "hip":
        flags.append("-std=c++17")
        if arch:
            flags.append(f"--offload-arch={arch}")

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
    if not metrics:
        return {}
    n = len(metrics)
    requested_ns_avg = sum(m.get("requested_ns", 0) for m in metrics) / n
    measured_ratio_avg = sum(m.get("measured_ratio", 0) for m in metrics) / n
    error_ns_avg = sum(m.get("error_ns", 0) for m in metrics) / n
    abs_error_ns_avg = sum(abs(m.get("error_ns", 0)) for m in metrics) / n
    return {
        "requested_ns_avg": requested_ns_avg,
        "measured_ratio_avg": measured_ratio_avg,
        "error_ns_avg": error_ns_avg,
        "abs_error_ns_avg": abs_error_ns_avg,
    }


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str):
    if not metrics:
        return
    keys = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label, results_dir,
                     requested_ns_unit="ns", measured_ratio_unit="ratio", error_ns_unit="ns"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed - cannot generate plots.")
        return

    os.makedirs(results_dir, exist_ok=True)

    ref_req = [m["requested_ns"] for m in ref_metrics]
    ref_ratio = [m["measured_ratio"] for m in ref_metrics]
    ref_err = [m["error_ns"] for m in ref_metrics]

    gen_req = [m["requested_ns"] for m in gen_metrics]
    gen_ratio = [m["measured_ratio"] for m in gen_metrics]
    gen_err = [m["error_ns"] for m in gen_metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ref_req, ref_ratio, marker="o", label=ref_label, linewidth=2)
    ax.plot(gen_req, gen_ratio, marker="s", label=gen_label, linewidth=2)
    ax.axhline(1.0, ls=":", color="gray", label="ideal (ratio=1)")
    ax.set_xlabel(f"Requested duration ({requested_ns_unit})")
    ax.set_ylabel(f"Measured / Requested ({measured_ratio_unit})")
    ax.set_title("Measured/Requested ratio vs Requested duration")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    ratio_path = os.path.join(results_dir, "ratio_comparison.png")
    fig.savefig(ratio_path, dpi=150)
    plt.close(fig)
    print(f"Saved {ratio_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ref_req, ref_err, marker="o", label=ref_label, linewidth=2)
    ax.plot(gen_req, gen_err, marker="s", label=gen_label, linewidth=2)
    ax.axhline(0, ls=":", color="gray", label="ideal (error=0)")
    ax.set_xlabel(f"Requested duration ({requested_ns_unit})")
    ax.set_ylabel(f"Error = measured - requested ({error_ns_unit})")
    ax.set_title("Error vs Requested duration")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    err_path = os.path.join(results_dir, "error_comparison.png")
    fig.savefig(err_path, dpi=150)
    plt.close(fig)
    print(f"Saved {err_path}")


def _plot_single(metrics, label, results_dir,
                 requested_ns_unit="ns", measured_ratio_unit="ratio", error_ns_unit="ns"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed - cannot generate plots.")
        return

    os.makedirs(results_dir, exist_ok=True)

    req = [m["requested_ns"] for m in metrics]
    ratio = [m["measured_ratio"] for m in metrics]
    err = [m["error_ns"] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(req, ratio, marker="o", label=label, linewidth=2)
    ax.axhline(1.0, ls=":", color="gray", label="ideal (ratio=1)")
    ax.set_xlabel(f"Requested duration ({requested_ns_unit})")
    ax.set_ylabel(f"Measured / Requested ({measured_ratio_unit})")
    ax.set_title("Measured/Requested ratio vs Requested duration")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    ratio_path = os.path.join(results_dir, "ratio.png")
    fig.savefig(ratio_path, dpi=150)
    plt.close(fig)
    print(f"Saved {ratio_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(req, err, marker="o", label=label, linewidth=2)
    ax.axhline(0, ls=":", color="gray", label="ideal (error=0)")
    ax.set_xlabel(f"Requested duration ({requested_ns_unit})")
    ax.set_ylabel(f"Error = measured - requested ({error_ns_unit})")
    ax.set_title("Error vs Requested duration")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    err_path = os.path.join(results_dir, "error.png")
    fig.savefig(err_path, dpi=150)
    plt.close(fig)
    print(f"Saved {err_path}")


def _compare_metrics(
    ref_metrics: Dict[str, Any],
    gen_metrics: Dict[str, Any],
) -> Dict[str, Any]:
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
                               or "error" in lower_key
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
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]

    build_run_verbose = show_raw_output

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

    requested_ns_unit = ref_parsed.get("requested_ns_unit", gen_parsed.get("requested_ns_unit", "ns"))
    measured_ratio_unit = ref_parsed.get("measured_ratio_unit", gen_parsed.get("measured_ratio_unit", "ratio"))
    error_ns_unit = ref_parsed.get("error_ns_unit", gen_parsed.get("error_ns_unit", "ns"))

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

    summary: Dict[str, Any] = {
        "generated_source": os.path.basename(src_gen),
        "ref_source": os.path.basename(src_ref),
        "model": "",
        "pass_iteration": 1,
    }

    if has_perf:
        summary["improvement_iteration"] = 1
        summary["requested_ns_unit"] = requested_ns_unit
        summary["measured_ratio_unit"] = measured_ratio_unit
        summary["error_ns_unit"] = error_ns_unit

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

        # Accuracy metric: smaller mean |error_ns| = better.
        if ref_avg.get("abs_error_ns_avg", 0) != 0:
            acc_imp = ((ref_avg["abs_error_ns_avg"] - gen_avg["abs_error_ns_avg"])
                       / ref_avg["abs_error_ns_avg"] * 100)
            summary["accuracy_improvement_pct"] = round(acc_imp, 2)
        else:
            summary["accuracy_improvement_pct"] = 0.0

        acc_imp = summary["accuracy_improvement_pct"]
        if abs(acc_imp) < 5:
            summary["performance"] = "same"
        elif acc_imp > 0:
            summary["performance"] = "better"
        else:
            summary["performance"] = "worse"

        if not no_plot and ref_metrics and gen_metrics:
            _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label,
                             results_dir, requested_ns_unit, measured_ratio_unit, error_ns_unit)

        if verbose and ref_avg and gen_avg:
            metric_units = {
                "requested_ns_avg": requested_ns_unit,
                "measured_ratio_avg": measured_ratio_unit,
                "error_ns_avg": error_ns_unit,
                "abs_error_ns_avg": error_ns_unit,
            }
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
    parser = argparse.ArgumentParser(
        description="Build and run P2P pingpong latency example"
    )
    parser.add_argument("--source", "-s",
                        default="ref_globaltimer_nanosleep.cu",
                        help="Source file to compile")
    parser.add_argument("--output", "-o",
                        default="globaltimer_nanosleep",
                        help="Output executable name")
    parser.add_argument("--arch", "-a", default=None,
                        help="GPU architecture (default: sm_90 for CUDA)")
    parser.add_argument("--build-only", "-b", action="store_true",
                        help="Only build, do not run")
    parser.add_argument("--run-only", "-r", action="store_true",
                        help="Only run (assume already built)")
    parser.add_argument("--compiler", "-c", default=None,
                        help="Specify compiler path (auto-detect if omitted)")
    parser.add_argument("--platform", "-p", choices=["hip", "cuda"],
                        default=None, help="Force platform (hip or cuda)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate benchmark plots after running")
    parser.add_argument("--results-dir", default=None,
                        help="Directory to save results (default: ./results)")
    parser.add_argument("--compare", nargs=2, metavar=("SRC_A", "SRC_B"),
                        default=None,
                        help="Compare two source files. Implies plot generation.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug build")
    parser.add_argument("--show-raw-output", action="store_true",
                        help="In compare mode, also print raw program stdout/stderr")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

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
    results_dir = args.results_dir or os.path.join(get_module_dir(), "results")

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
        parsed = result.run_result.parsed_output
        metrics = parsed.get("metrics", [])
        if metrics:
            label = os.path.splitext(os.path.basename(args.source))[0]
            _plot_single(
                metrics, label, results_dir,
                parsed.get("requested_ns_unit", "ns"),
                parsed.get("measured_ratio_unit", "ratio"),
                parsed.get("error_ns_unit", "ns"),
            )
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            os.makedirs(results_dir, exist_ok=True)
            _save_metrics_csv(metrics, csv_path)
            if verbose:
                print(f"Saved {csv_path}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
