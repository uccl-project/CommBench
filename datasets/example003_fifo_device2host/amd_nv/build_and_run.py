#!/usr/bin/env python3
"""
Build and run script for FIFO Device-to-Host Communication Test
Supports both AMD (HIP) and NVIDIA (CUDA) GPUs.
"""

import subprocess
import os
import sys
import shutil
import json
import csv
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Allow importing the plotting module from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from metrics_plot import plot_metrics, plot_metrics_compare

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
    error_message: Optional[str] = None


@dataclass
class BuildAndRunResult:
    """Result of combined build and run operation."""
    build_result: Optional[BuildResult]
    run_result: Optional[RunResult]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Returns True if both build and run succeeded."""
        build_ok = self.build_result is None or self.build_result.success
        run_ok = self.run_result is None or self.run_result.success
        return build_ok and run_ok


def _parse_output_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse the single JSON object printed to stdout by the program.

    The program outputs exactly one JSON object to stdout.  We find it by
    looking for the outermost '{' ... '}' block in the text.
    """
    if not text:
        return None
    # Find the first '{' and last '}' to extract the JSON envelope
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _extract_performance_metrics(text: str) -> Dict[str, Any]:
    """
    Extract the first metrics entry from the single JSON object printed by the program.
    Only supports the new single-JSON format.
    """
    if not text:
        return {}

    obj = _parse_output_json(text)
    if obj and "metrics" in obj and isinstance(obj["metrics"], list):
        return obj["metrics"][0] if obj["metrics"] else {}

    return {}

def parse_all_metrics(text: str) -> List[Dict[str, Any]]:
    """
    Extract all metric entries from the single JSON output.
    Only supports the new format with a top-level 'metrics' array.
    """
    if not text:
        return []

    obj = _parse_output_json(text)
    if obj and "metrics" in obj and isinstance(obj["metrics"], list):
        return obj["metrics"]

    return []



def _average_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average numeric values across a list of metric dicts."""
    if not records:
        return {}
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


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


def find_compiler():
    """Find available GPU compiler (hipcc for AMD, nvcc for NVIDIA)."""
    # Check for hipcc (AMD)
    hipcc = shutil.which("hipcc")
    if hipcc:
        return hipcc, "hip"

    # Check for nvcc (NVIDIA)
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc, "cuda"

    return None, None


def get_hip_arch():
    """Get the AMD GPU architecture using rocminfo."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Look for gfx architecture in output
        for line in result.stdout.split("\n"):
            if "gfx" in line.lower():
                parts = line.split()
                for part in parts:
                    if part.startswith("gfx"):
                        return part
        # Default to common architectures
        return "gfx906,gfx908,gfx90a,gfx942"
    except Exception:
        return "gfx906,gfx908,gfx90a,gfx942"


def build(source_file, output_file, compiler, platform, debug=False, arch=None, verbose=True) -> BuildResult:
    """Build the source file using the appropriate compiler."""
    if verbose:
        print(f"Building with {compiler} ({platform})...")

    if platform == "hip":
        # HIP compilation for AMD GPUs
        if arch is None:
            arch = get_hip_arch()
        if verbose:
            print(f"  Target architecture: {arch}")

        cmd = [
            compiler,
            source_file,
            "-o", output_file,
            "-std=c++17",
            "-D__HIP_PLATFORM_AMD__",
            "-lpthread",
        ]

        # Add architecture flags
        for a in arch.split(","):
            cmd.append(f"--offload-arch={a.strip()}")

        if debug:
            cmd.extend(["-g", "-DDEBUG_BUILD"])
        else:
            cmd.append("-O3")

    else:
        # CUDA compilation for NVIDIA GPUs
        cmd = [
            compiler,
            source_file,
            "-o", output_file,
            "-std=c++17",
            "-lpthread",
        ]

        if arch:
            cmd.append(f"-arch={arch}")
        else:
            # Default to common architectures
            cmd.append("-arch=sm_70")

        # Inject the bundled aarch64 stub for bits/math-vector.h when
        # available; nvcc < 12.4 cannot parse the system header on Grace
        # Hopper otherwise.
        here = os.path.dirname(os.path.abspath(__file__))
        for c in (os.path.join(here, "..", "..", "_cuda_compat"),
                  os.path.join(here, "..", "_cuda_compat")):
            if os.path.exists(os.path.join(c, "bits", "math-vector.h")):
                cmd.extend(["-isystem", os.path.abspath(c)])
                break

        if debug:
            cmd.extend(["-g", "-G", "-DDEBUG_BUILD"])
        else:
            cmd.append("-O3")

    if verbose:
        print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        success = result.returncode == 0
        if not success and verbose:
            print(f"\nCompilation failed!")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
        elif verbose:
            print(f"  Build successful: {output_file}")

        return BuildResult(
            success=success,
            source_file=source_file,
            output_file=output_file,
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            command=cmd,
            error_message=None if success else "Compilation failed"
        )

    except subprocess.TimeoutExpired:
        if verbose:
            print("Compilation timed out!")
        return BuildResult(
            success=False,
            source_file=source_file,
            output_file=output_file,
            return_code=-1,
            stdout="",
            stderr="Compilation timed out",
            command=cmd,
            error_message="Compilation timed out"
        )
    except Exception as e:
        if verbose:
            print(f"Compilation error: {e}")
        return BuildResult(
            success=False,
            source_file=source_file,
            output_file=output_file,
            return_code=-1,
            stdout="",
            stderr=str(e),
            command=cmd,
            error_message=str(e)
        )


def run(executable, verbose=True) -> RunResult:
    """Run the compiled executable."""
    if verbose:
        print(f"\nRunning {executable}...\n")
        print("=" * 50)

    env = os.environ.copy()
    # Set HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES if needed
    if "HIP_VISIBLE_DEVICES" not in env and "CUDA_VISIBLE_DEVICES" not in env:
        env["HIP_VISIBLE_DEVICES"] = "0"
        env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [f"./{executable}"]

    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(executable)) or ".",
            env=env,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            print("=" * 50)

        success = result.returncode == 0
        return RunResult(
            success=success,
            executable=executable,
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            command=cmd,
            error_message=None if success else "Execution failed"
        )

    except subprocess.TimeoutExpired:
        if verbose:
            print("Execution timed out!")
        return RunResult(
            success=False,
            executable=executable,
            return_code=-1,
            stdout="",
            stderr="Execution timed out",
            command=cmd,
            error_message="Execution timed out"
        )
    except Exception as e:
        if verbose:
            print(f"Execution error: {e}")
        return RunResult(
            success=False,
            executable=executable,
            return_code=-1,
            stdout="",
            stderr=str(e),
            command=cmd,
            error_message=str(e)
        )


def build_and_run(
    source_file: str,
    output_file: str,
    compiler: Optional[str],
    platform: Optional[str],
    debug: bool = False,
    arch: Optional[str] = None,
    build_only: bool = False,
    run_only: bool = False,
    verbose: bool = True
) -> BuildAndRunResult:
    """Build and run the FIFO Device-to-Host test."""
    build_result = None
    run_result = None

    if not run_only:
        build_result = build(
            source_file=source_file,
            output_file=output_file,
            compiler=compiler,
            platform=platform,
            debug=debug,
            arch=arch,
            verbose=verbose
        )
        if not build_result.success:
            return BuildAndRunResult(build_result=build_result, run_result=None)

    if not build_only:
        run_result = run(output_file, verbose=verbose)

    # Extract metrics from run output
    metrics = {}
    if run_result is not None:
        metrics_text = "\n".join([run_result.stdout, run_result.stderr]).strip()
        metrics = _extract_performance_metrics(metrics_text)

    return BuildAndRunResult(
        build_result=build_result,
        run_result=run_result,
        performance_metrics=metrics
    )


def _build_summary(
    ref_compile_ok: bool,
    ref_run_ok: bool,
    ref_avg_metrics: Dict[str, Any],
    gen_compile_ok: bool,
    gen_run_ok: bool,
    gen_avg_metrics: Dict[str, Any],
    latency_key: str = "lat_avg_ns",
    throughput_key: str = "throughput_MBps",
) -> Dict[str, Any]:
    """Build a standardized summary dict for comparison results.

    Verdict in summary["performance"] is computed by this example's own
    _compare_metrics() — the shared 4-tier scheme in run_eval/perf_verdict.py
    overrides it at process exit via the atexit hook installed in main(),
    unless --legacy-perf-verdict was given. Keeping the local computation
    here means --legacy-perf-verdict produces a coherent legacy verdict.
    """
    summary: Dict[str, Any] = {
        "metrics_comparison": {
            "ref": {"compile_success": ref_compile_ok, "run_success": ref_run_ok, **ref_avg_metrics},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok, **gen_avg_metrics},
        },
        "performance": None,
        "model": "",
        "pass_iteration": None,
        "improvement_iteration": None,
    }
    if gen_compile_ok and gen_run_ok and ref_avg_metrics and gen_avg_metrics:
        if latency_key in ref_avg_metrics and latency_key in gen_avg_metrics and ref_avg_metrics[latency_key] != 0:
            summary["latency_improvement_pct"] = round(
                (ref_avg_metrics[latency_key] - gen_avg_metrics[latency_key]) / ref_avg_metrics[latency_key] * 100, 2)
        if throughput_key in ref_avg_metrics and throughput_key in gen_avg_metrics and ref_avg_metrics[throughput_key] != 0:
            summary["throughput_improvement_pct"] = round(
                (gen_avg_metrics[throughput_key] - ref_avg_metrics[throughput_key]) / ref_avg_metrics[throughput_key] * 100, 2)
        comparison = _compare_metrics(ref_avg_metrics, gen_avg_metrics)
        summary["performance"] = comparison["summary"].get("status", None)
    return summary


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
    """Compare two source files: build, run, save CSV + summary.json + plots.

    Args:
        show_raw_output: If True, print the raw stdout/stderr from each
            program execution.  Defaults to False (only comparison results
            are printed).

    Returns the summary dict.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if compiler is None:
        compiler, platform = find_compiler()
    if compiler is None:
        raise RuntimeError("No GPU compiler found")

    for src in (src_ref, src_gen):
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file '{src}' not found")

    label_ref = os.path.splitext(src_ref)[0]
    label_gen = os.path.splitext(src_gen)[0]
    os.makedirs(results_dir, exist_ok=True)

    named_metrics: Dict[str, list] = {}
    build_results: Dict[str, bool] = {}
    run_results: Dict[str, bool] = {}

    # Use show_raw_output to control whether build_and_run prints raw output
    build_run_verbose = show_raw_output

    for label, src in ((label_ref, src_ref), (label_gen, src_gen)):
        exe = f"fifo_test_{label}"
        if verbose:
            print(f"\n{'='*60}")
            print(f"[compare] Building & running: {src}  ({label})")
            print(f"{'='*60}")
        res = build_and_run(
            source_file=src,
            output_file=exe,
            compiler=compiler,
            platform=platform,
            debug=debug,
            arch=arch,
            build_only=False,
            run_only=False,
            verbose=build_run_verbose,
        )
        compile_ok = res.build_result is not None and res.build_result.success
        run_ok = res.run_result is not None and res.run_result.success
        build_results[label] = compile_ok
        run_results[label] = run_ok

        if not compile_ok:
            if verbose:
                print(f"[compare] BUILD FAILED for {src}")
            named_metrics[label] = []
            continue
        if not run_ok:
            if verbose:
                print(f"[compare] RUN FAILED for {src}")
            named_metrics[label] = []
            continue

        combined = "\n".join(
            filter(None, [res.run_result.stdout, res.run_result.stderr])
        )
        mlist = parse_all_metrics(combined)
        if not mlist and verbose:
            print(f"[compare] WARNING: no 'metrics' field found in JSON output from {src}")
        named_metrics[label] = mlist

    # Save raw metrics as CSV for each source
    for label in (label_ref, label_gen):
        mlist = named_metrics.get(label, [])
        if not mlist:
            continue
        all_keys = list(dict.fromkeys(k for m in mlist for k in m.keys()))
        csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(mlist)
        if verbose:
            print(f"Raw metrics saved to {csv_path}")

    # Reference must succeed; if it doesn't we cannot evaluate
    if not build_results.get(label_ref) or not run_results.get(label_ref):
        if verbose:
            print(f"[compare] Reference ({src_ref}) failed – cannot evaluate.")
        sys.exit(1)

    ref_compile_ok = build_results.get(label_ref, False)
    ref_run_ok = run_results.get(label_ref, False)
    gen_compile_ok = build_results.get(label_gen, False)
    gen_run_ok = run_results.get(label_gen, False)

    avg_ref = _average_metrics(named_metrics.get(label_ref, []))
    avg_gen = _average_metrics(named_metrics.get(label_gen, []))

    summary = _build_summary(
        ref_compile_ok, ref_run_ok, avg_ref,
        gen_compile_ok, gen_run_ok, avg_gen,
        latency_key="lat_avg_ns",
        throughput_key="throughput_MBps",
    )

    # Generate comparison plots only if both succeeded
    if gen_compile_ok and gen_run_ok:
        if not no_plot:
            saved = plot_metrics_compare(
                named_metrics, output_dir=results_dir,
                fixed_block_size=block_size,
            )
            if verbose:
                print(f"\n{len(saved)} comparison plot(s) saved to {results_dir}")

        if verbose and avg_ref and avg_gen:
            # Infer unit from metric name suffix
            def _unit_for(name: str) -> str:
                lower = name.lower()
                if lower.endswith("_ns"):
                    return "ns"
                if lower.endswith("_us"):
                    return "us"
                if lower.endswith("_ms"):
                    return "ms"
                if lower.endswith("_mbps") or lower.endswith("_mibps"):
                    return "MB/s"
                if lower.endswith("_gbps"):
                    return "Gbps"
                return ""

            comparison = _compare_metrics(avg_ref, avg_gen)
            if comparison.get("comparison"):
                print(f"\nPERFORMANCE COMPARISON (averaged over "
                      f"{len(named_metrics[label_ref])} {label_ref} / "
                      f"{len(named_metrics[label_gen])} {label_gen} records)")
                for metric, comp in comparison["comparison"].items():
                    flag = "+" if comp["better_or_equal"] else "-"
                    unit = _unit_for(metric)
                    unit_str = f" {unit}" if unit else ""
                    print(f"  [{flag}] {metric}: {comp['generated']:.4f}{unit_str} vs "
                          f"ref {comp['ref']:.4f}{unit_str} "
                          f"(ratio: {comp['ratio']:.2f}, "
                          f"{comp['improvement_pct']:+.1f}%)")
                print(f"  Performance: {summary['performance']}")

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
        print(f"  performance:               {summary['performance']}")
        print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build and run FIFO Device-to-Host test"
    )
    parser.add_argument(
        "--source", "-s",
        default="ref_fifo_test_unified.cu",
        help="Source file to compile (default: ref_fifo_test_unified.cu)"
    )
    parser.add_argument(
        "--output", "-o",
        default="fifo_test",
        help="Output executable name (default: fifo_test)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Build with debug flags"
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
        "--plot-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("SRC_A", "SRC_B"),
        default=None,
        help="Compare two source files. Builds and runs both, then generates "
             "comparison plots. Implies --plot.  "
             "Example: --compare ref_fifo_test_unified.cu generated_fifo_test_unified.cu"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Fixed blockSize for comparison plots (default: 256)"
    )
    parser.add_argument(
        "--compare-no-plot",
        action="store_true",
        help="Skip generating plot images (CSV and summary JSON are still saved)"
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
        help="Use this example's local per-metric worst-case verdict logic "
             "instead of the shared 4-tier scheme in run_eval/perf_verdict.py. "
             "Default is the shared logic."
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

    # --plot-dir is a hidden alias for --results-dir
    results_dir = args.results_dir or args.plot_dir

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Find or use specified compiler
    if args.compiler:
        compiler = args.compiler
        platform = args.platform or ("hip" if "hip" in compiler.lower() else "cuda")
    else:
        compiler, platform = find_compiler()

    if args.platform:
        platform = args.platform

    if not args.run_only:
        if not compiler:
            print("Error: No GPU compiler found!")
            print("  For AMD GPUs: Install ROCm and ensure 'hipcc' is in PATH")
            print("  For NVIDIA GPUs: Install CUDA and ensure 'nvcc' is in PATH")
            sys.exit(1)

        print(f"Detected compiler: {compiler} ({platform})")

    # ── Compare mode ──────────────────────────────────────────────────────
    if args.compare:
        src_a, src_b = args.compare
        rd = results_dir or os.path.join(script_dir, "results")
        summary = compare(
            src_ref=src_a,
            src_gen=src_b,
            results_dir=rd,
            compiler=compiler,
            platform=platform,
            debug=args.debug,
            arch=args.arch,
            no_plot=args.compare_no_plot,
            block_size=args.block_size,
            verbose=not args.quiet,
            show_raw_output=args.show_raw_output,
        )
        print("\nDone!")
        sys.exit(0)

    # ── Single-source mode ────────────────────────────────────────────────
    source_file = args.source
    output_file = args.output

    # Check source file exists
    if not args.run_only and not os.path.exists(source_file):
        print(f"Error: Source file '{source_file}' not found!")
        sys.exit(1)

    # Use build_and_run for unified execution
    result = build_and_run(
        source_file=source_file,
        output_file=output_file,
        compiler=compiler,
        platform=platform,
        debug=args.debug,
        arch=args.arch,
        build_only=args.build_only,
        run_only=args.run_only,
        verbose=not args.quiet
    )

    if args.build_only:
        print("\nBuild-only mode: skipping execution.")
        sys.exit(0 if result.success else 1)

    if not result.success:
        sys.exit(1)

    # --- Plotting ----------------------------------------------------------
    if args.plot and result.run_result is not None:
        combined_output = "\n".join(
            filter(None, [result.run_result.stdout, result.run_result.stderr])
        )
        all_metrics = parse_all_metrics(combined_output)
        if all_metrics:
            rd = results_dir or os.path.join(script_dir, "results")
            saved = plot_metrics(all_metrics, output_dir=rd)
            print(f"\n{len(saved)} plot(s) saved to {rd}")
        else:
            print("\nNo 'metrics' field found in program JSON - skipping plots.")

    print("\nDone!")


if __name__ == "__main__":
    main()
