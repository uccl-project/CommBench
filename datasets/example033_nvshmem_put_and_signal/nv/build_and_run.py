#!/usr/bin/env python3
"""
NVSHMEM Put-and-Signal Build and Run Module

Compiles and runs NVSHMEM + MPI programs for the put+signal benchmark.
Can be used as a standalone script or imported as a module.

Usage:
    python build_and_run.py --source ref_nvshmem_put_and_signal.cu
    python build_and_run.py --compare ref_nvshmem_put_and_signal.cu generated_nvshmem_put_and_signal.cu
"""

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

# ─── Defaults ────────────────────────────────────────────────────────────────

_DEFAULT_SOURCE   = "ref_nvshmem_put_and_signal.cu"
_DEFAULT_ARCH     = "compute_100,code=sm_100"
_DEFAULT_NUM_PES  = 2
_DEFAULT_TIMEOUT  = 600

_NVCC             = "/usr/local/cuda-13.2/bin/nvcc"
_NVSHMEM_INCLUDE  = "/usr/include/nvshmem_13"
_NVSHMEM_LIB      = "/usr/lib/x86_64-linux-gnu"
_MPI_HOME         = "/usr/mpi/gcc/openmpi-4.1.9a1"
_MPIRUN           = os.path.join(_MPI_HOME, "bin", "mpirun")


# ─── Data classes ─────────────────────────────────────────────────────────────

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
        run_ok   = self.run_result   is None or self.run_result.success
        return build_ok and run_ok


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()
    extra = f"{_NVSHMEM_LIB}:{os.path.join(_MPI_HOME, 'lib')}"
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{extra}:{existing}" if existing else extra
    return env


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
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    continue
    return {}


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str) -> None:
    if not metrics:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "data_size_avg":   sum(float(m.get("data_size",      0)) for m in metrics) / n,
        "latency_avg":     sum(float(m.get("latency_avg",    0)) for m in metrics) / n,
        "throughput":      sum(float(m.get("throughput_avg", 0)) for m in metrics) / n,
    }


def _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label, results_dir,
                     data_size_unit="", latency_unit="us", throughput_unit=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping plots.")
        return

    os.makedirs(results_dir, exist_ok=True)

    def _extract(metrics, key):
        return [m.get(key, 0) for m in metrics]

    x_key = "data_size" if any("data_size" in m for m in ref_metrics) else "pe"

    ref_x   = _extract(ref_metrics, x_key)
    gen_x   = _extract(gen_metrics, x_key)
    ref_lat = _extract(ref_metrics, "latency_avg")
    gen_lat = _extract(gen_metrics, "latency_avg")
    ref_thr = _extract(ref_metrics, "throughput_avg")
    gen_thr = _extract(gen_metrics, "throughput_avg")

    for data, ylabel, unit, fname in [
        ((ref_lat, gen_lat), "Latency",    latency_unit,    "latency_comparison.png"),
        ((ref_thr, gen_thr), "Throughput", throughput_unit, "throughput_comparison.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ref_x, data[0], marker="o", label=ref_label, linewidth=2)
        ax.plot(gen_x, data[1], marker="s", label=gen_label, linewidth=2)
        ax.set_xlabel(f"{x_key} ({data_size_unit})" if data_size_unit else x_key)
        ax.set_ylabel(f"{ylabel} ({unit})" if unit else ylabel)
        ax.set_xscale("log")
        ax.set_title(f"{ylabel} Comparison")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout()
        out = os.path.join(results_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


# ─── Core API ─────────────────────────────────────────────────────────────────

def build(
    source_file: str,
    output_file: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = "cuda",
    debug: bool = False,
    arch: Optional[str] = _DEFAULT_ARCH,
    verbose: bool = True,
) -> BuildResult:
    wd = get_module_dir()
    src_path = source_file if os.path.isabs(source_file) else os.path.join(wd, source_file)
    out_path = output_file if os.path.isabs(output_file) else os.path.join(wd, output_file)

    if not os.path.exists(src_path):
        msg = f"Source file '{source_file}' not found"
        if verbose:
            print(f"[build] ERROR: {msg}", file=sys.stderr)
        return BuildResult(success=False, source_file=source_file, output_file=output_file,
                           return_code=-1, stdout="", stderr="", command=[], error_message=msg)

    nvcc = compiler or _NVCC
    if not os.path.exists(nvcc):
        nvcc_found = shutil.which("nvcc")
        if nvcc_found:
            nvcc = nvcc_found
        else:
            msg = f"nvcc not found at '{nvcc}'. Pass --compiler."
            if verbose:
                print(f"[build] ERROR: {msg}", file=sys.stderr)
            return BuildResult(success=False, source_file=source_file, output_file=output_file,
                               return_code=-1, stdout="", stderr="", command=[], error_message=msg)

    cmd = [
        nvcc,
        "-rdc=true",
        "-ccbin", "/usr/bin/g++",
        f"-gencode=arch={arch}" if arch else f"-gencode=arch={_DEFAULT_ARCH}",
        "-O0" if debug else "-O3",
        "-I", _NVSHMEM_INCLUDE,
        "-I", os.path.join(_MPI_HOME, "include"),
        src_path,
        "-o", out_path,
        "-L", _NVSHMEM_LIB,
        "-lnvshmem_host", "-lnvshmem_device",
        "-L", os.path.join(_MPI_HOME, "lib"),
        "-lmpi", "-lcuda",
    ]

    if verbose:
        print("=" * 60)
        print("Building (CUDA + NVSHMEM + MPI)")
        print("=" * 60)
        print(f"Source : {source_file}")
        print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=_build_env())

    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print("=" * 60)
        print("Build successful!" if result.returncode == 0 else "Build failed!")
        print("=" * 60)

    return BuildResult(
        success=result.returncode == 0,
        source_file=source_file, output_file=output_file,
        return_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
        command=cmd, error_message=None if result.returncode == 0 else "Compilation failed",
    )


def run(
    executable: str,
    num_pes: int = _DEFAULT_NUM_PES,
    verbose: bool = True,
    timeout: int = _DEFAULT_TIMEOUT,
) -> RunResult:
    wd = get_module_dir()
    exe_path = executable if os.path.isabs(executable) else os.path.join(wd, executable)

    if not os.path.exists(exe_path):
        msg = f"Executable '{executable}' not found"
        if verbose:
            print(f"[run] ERROR: {msg}", file=sys.stderr)
        return RunResult(success=False, executable=executable, return_code=-1,
                         stdout="", stderr="", command=[], error_message=msg)

    mpirun = _MPIRUN if os.path.exists(_MPIRUN) else (shutil.which("mpirun") or "mpirun")
    cmd = [mpirun, "--allow-run-as-root", "-n", str(num_pes), exe_path]

    if verbose:
        print(f"[run] {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env=_build_env())
    except subprocess.TimeoutExpired:
        msg = f"Run timed out after {timeout}s"
        if verbose:
            print(f"[run] ERROR: {msg}", file=sys.stderr)
        return RunResult(success=False, executable=executable, return_code=-1,
                         stdout="", stderr="", command=cmd, error_message=msg)

    if verbose:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

    parsed = _parse_json_output(result.stdout) if result.returncode == 0 else {}

    return RunResult(
        success=result.returncode == 0,
        executable=executable, return_code=result.returncode,
        stdout=result.stdout, stderr=result.stderr,
        command=cmd, parsed_output=parsed,
        error_message=None if result.returncode == 0 else "Execution failed",
    )


def build_and_run(
    source_file: str,
    output_file: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = "cuda",
    debug: bool = False,
    arch: Optional[str] = _DEFAULT_ARCH,
    build_only: bool = False,
    run_only: bool = False,
    num_pes: int = _DEFAULT_NUM_PES,
    verbose: bool = True,
) -> BuildAndRunResult:
    build_result: Optional[BuildResult] = None
    run_result:   Optional[RunResult]   = None

    if not run_only:
        build_result = build(source_file=source_file, output_file=output_file,
                             compiler=compiler, platform=platform,
                             debug=debug, arch=arch, verbose=verbose)
        if not build_result.success:
            return BuildAndRunResult(build_result=build_result, run_result=None)

    if not build_only:
        if verbose:
            print()
            print("=" * 60)
            print("Running program")
            print("=" * 60)
        run_result = run(executable=output_file, num_pes=num_pes, verbose=verbose)

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = "cuda",
    debug: bool = False,
    arch: Optional[str] = _DEFAULT_ARCH,
    num_pes: int = _DEFAULT_NUM_PES,
    no_plot: bool = False,
    verbose: bool = True,
    show_raw_output: bool = False,
) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]
    build_verbose = show_raw_output

    def _build_run(src, out_name):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[compare] Building & running: {src}")
            print(f"{'='*60}")
        return build_and_run(source_file=src, output_file=out_name,
                             compiler=compiler, platform=platform,
                             debug=debug, arch=arch, num_pes=num_pes,
                             verbose=build_verbose)

    ref_res = _build_run(src_ref, "ref_exec")
    gen_res = _build_run(src_gen, "gen_exec")

    ref_compile_ok = ref_res.build_result is not None and ref_res.build_result.success
    ref_run_ok     = ref_res.run_result   is not None and ref_res.run_result.success
    gen_compile_ok = gen_res.build_result is not None and gen_res.build_result.success
    gen_run_ok     = gen_res.run_result   is not None and gen_res.run_result.success

    ref_parsed = ref_res.run_result.parsed_output if ref_run_ok else {}
    gen_parsed = gen_res.run_result.parsed_output if gen_run_ok else {}

    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])

    data_size_unit  = ref_parsed.get("data_size_unit",  gen_parsed.get("data_size_unit",  "bytes"))
    latency_unit    = ref_parsed.get("latency_unit",    gen_parsed.get("latency_unit",    "us"))
    throughput_unit = ref_parsed.get("throughput_unit", gen_parsed.get("throughput_unit", "GB/s"))

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
        "ref_source":       os.path.basename(src_ref),
        "model":            "",
        "pass_iteration":   1,
    }

    if has_perf:
        ref_avg = _metrics_avg(ref_metrics)
        gen_avg = _metrics_avg(gen_metrics)

        summary.update({
            "improvement_iteration": 1,
            "data_size_unit":        data_size_unit,
            "latency_unit":          latency_unit,
            "throughput_unit":       throughput_unit,
            "metrics_comparison": {
                "ref":       {"compile_success": ref_compile_ok, "run_success": ref_run_ok, **ref_avg},
                "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok, **gen_avg},
            },
        })

        r_lat = ref_avg.get("latency_avg", 0.0)
        g_lat = gen_avg.get("latency_avg", 0.0)
        r_thr = ref_avg.get("throughput",  0.0)
        g_thr = gen_avg.get("throughput",  0.0)

        lat_imp = ((r_lat - g_lat) / r_lat * 100.0) if r_lat else 0.0
        thr_imp = ((g_thr - r_thr) / r_thr * 100.0) if r_thr else 0.0
        summary["latency_improvement_pct"]    = round(lat_imp, 2)
        summary["throughput_improvement_pct"] = round(thr_imp, 2)

        if abs(lat_imp) < 5 and abs(thr_imp) < 5:
            summary["performance"] = "same"
        elif lat_imp > 0 and thr_imp > 0:
            summary["performance"] = "better"
        elif lat_imp < -5 or thr_imp < -5:
            summary["performance"] = "worse"
        else:
            summary["performance"] = "same"

        if not no_plot:
            _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label,
                             results_dir, data_size_unit, latency_unit, throughput_unit)

        if verbose:
            print(f"\nPERFORMANCE COMPARISON")
            print(f"  [+] data_size_avg : {ref_avg['data_size_avg']:.4f} (ref) vs "
                  f"{gen_avg['data_size_avg']:.4f} (gen)")
            print(f"  [+] latency_avg   : {r_lat:.4f} {latency_unit} (ref) vs "
                  f"{g_lat:.4f} {latency_unit} (gen)  ({lat_imp:+.2f}%)")
            print(f"  [+] throughput    : {r_thr:.4f} {throughput_unit} (ref) vs "
                  f"{g_thr:.4f} {throughput_unit} (gen)  ({thr_imp:+.2f}%)")
            print(f"  Performance: {summary['performance']}")
    else:
        summary["metrics_comparison"] = {
            "ref":       {"compile_success": ref_compile_ok, "run_success": ref_run_ok},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok},
        }

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


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and run NVSHMEM put-and-signal benchmark"
    )
    parser.add_argument("--source", "-s", default=_DEFAULT_SOURCE,
                        help=f"Source file (default: {_DEFAULT_SOURCE})")
    parser.add_argument("--output", "-o", default=None,
                        help="Output executable name (default: derived from source)")
    parser.add_argument("--arch", "-a", default=_DEFAULT_ARCH,
                        help=f"GPU arch for nvcc (default: {_DEFAULT_ARCH})")
    parser.add_argument("--build-only", "-b", action="store_true",
                        help="Compile only, do not run")
    parser.add_argument("--run-only", "-r", action="store_true",
                        help="Run only (skip compile)")
    parser.add_argument("--compiler", "-c", default=None,
                        help=f"Compiler path (default: {_NVCC})")
    parser.add_argument("--platform", "-p", choices=["cuda", "hip"], default="cuda",
                        help="Platform (default: cuda)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots after run")
    parser.add_argument("--results-dir", default=None,
                        help="Directory for CSV/plots/summary (default: ./results)")
    parser.add_argument("--compare", nargs=2, metavar=("SRC_REF", "SRC_GEN"),
                        help="Compare two source files")
    parser.add_argument("--debug", action="store_true", help="Debug build")
    parser.add_argument("--num-pes", "-n", type=int, default=_DEFAULT_NUM_PES,
                        help=f"Number of MPI PEs (default: {_DEFAULT_NUM_PES})")
    parser.add_argument("--show-raw-output", action="store_true",
                        help="Print raw program stdout/stderr in compare mode")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
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
        compare(src_ref=src_ref, src_gen=src_gen,
                results_dir=results_dir,
                compiler=args.compiler, platform=args.platform,
                debug=args.debug, arch=args.arch,
                num_pes=args.num_pes,
                verbose=verbose, show_raw_output=args.show_raw_output)
        sys.exit(0)

    output = args.output or os.path.splitext(os.path.basename(args.source))[0]

    result = build_and_run(
        source_file=args.source, output_file=output,
        compiler=args.compiler, platform=args.platform,
        debug=args.debug, arch=args.arch,
        build_only=args.build_only, run_only=args.run_only,
        num_pes=args.num_pes, verbose=verbose,
    )

    if args.plot and result.run_result and result.run_result.parsed_output:
        metrics = result.run_result.parsed_output.get("metrics", [])
        if metrics:
            label    = os.path.splitext(os.path.basename(args.source))[0]
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            os.makedirs(results_dir, exist_ok=True)
            _save_metrics_csv(metrics, csv_path)
            if verbose:
                print(f"Saved {csv_path}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
