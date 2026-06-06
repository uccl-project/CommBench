#!/usr/bin/env python3
"""
Build, run, and compare the NCCL Device API inter-node AllReduce example.

Usage as module:
    from build_and_run import build, run, build_and_run, compare

Usage as script:
    python build_and_run.py --source ref_nccl_device_api_internode_allreduce.cu
    python build_and_run.py --compare ref_nccl_device_api_internode_allreduce.cu generated_nccl_device_api_internode_allreduce.cu
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
import os  # noqa: E402,F811
import sys  # noqa: E402,F811
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


_DEFAULT_SOURCE = "ref_nccl_device_api_internode_allreduce.cu"
_DEFAULT_ARCH = "sm_90"
_DEFAULT_COUNT = 0  # 0 → binary sweeps a fixed multi-size list; >0 pins to one size
_DEFAULT_ITERS = 20
_DEFAULT_NPROC = 2
_DEFAULT_TIMEOUT_SEC = 900
_MPI_ROOT = "/usr/mpi/gcc/openmpi-4.1.9a1"
_CUDA_ROOT = "/usr/local/cuda-13.2"


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


def _env() -> Dict[str, str]:
    env = os.environ.copy()
    cuda_home = env.get("CUDA_HOME") or _CUDA_ROOT
    env["CUDA_HOME"] = cuda_home
    env["OMPI_CXX"] = env.get("OMPI_CXX", "g++-12")

    path_parts = [
        os.path.join(cuda_home, "bin"),
        os.path.join(_MPI_ROOT, "bin"),
        env.get("PATH", ""),
    ]
    ld_parts = [
        os.path.join(cuda_home, "lib64"),
        os.path.join(_MPI_ROOT, "lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["PATH"] = ":".join(p for p in path_parts if p)
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ld_parts if p)
    # Force NCCL to use the active IB port so ncclDevCommCreate with
    # NCCL_GIN_CONNECTION_FULL succeeds on a single-node setup.
    env.setdefault("NCCL_IB_HCA", "mlx5_5")
    return env


def _detect_nvcc(compiler: Optional[str]) -> Optional[str]:
    if compiler:
        return compiler
    env = _env()
    for cand in (
        os.path.join(env["CUDA_HOME"], "bin", "nvcc"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.2/bin/nvcc",
        shutil.which("nvcc", path=env["PATH"]),
    ):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _parse_json_output(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    depth = 0
    start = None
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


def _binary_path(source_abs: str, output_file: Optional[str]) -> str:
    if output_file:
        return output_file if os.path.isabs(output_file) else os.path.join(get_module_dir(), output_file)
    stem = os.path.splitext(os.path.basename(source_abs))[0]
    return os.path.join(get_module_dir(), stem)


def build(
    source_file: str,
    output_file: Optional[str] = None,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    verbose: bool = True,
) -> BuildResult:
    if platform not in (None, "cuda"):
        return BuildResult(False, source_file, output_file or "", -1, "", "",
                           [], f"Unsupported platform '{platform}', expected cuda")

    wd = get_module_dir()
    src_abs = source_file if os.path.isabs(source_file) else os.path.join(wd, source_file)
    if not os.path.isfile(src_abs):
        return BuildResult(False, source_file, output_file or "", -1, "", "",
                           [], f"Source file '{source_file}' not found")

    nvcc = _detect_nvcc(compiler)
    if not nvcc:
        return BuildResult(False, source_file, output_file or "", -1, "", "",
                           [], "nvcc not found. Set --compiler or CUDA_HOME.")

    out_abs = _binary_path(src_abs, output_file)
    flags = ["-O0", "-g", "-G"] if debug else ["-O3"]
    cmd = [
        nvcc,
        *flags,
        "-std=c++17",
        f"-arch={arch or _DEFAULT_ARCH}",
        "--extended-lambda",
        "-DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE",
        "-ccbin",
        "mpicxx",
        src_abs,
        "-lnccl",
        "-o",
        out_abs,
    ]

    if verbose:
        print(f"[build] {' '.join(cmd)}")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=_env())
    if verbose and proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    return BuildResult(
        success=proc.returncode == 0,
        source_file=source_file,
        output_file=out_abs,
        return_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        command=cmd,
        error_message=None if proc.returncode == 0 else "Compilation failed",
    )


def _run_with_args(
    executable: str,
    count: int,
    iters: int,
    nproc: int,
    verbose: bool = True,
) -> RunResult:
    wd = get_module_dir()
    exe_abs = executable if os.path.isabs(executable) else os.path.join(wd, executable)
    if not os.path.isfile(exe_abs):
        return RunResult(False, executable, -1, "", "", [],
                         error_message=f"Executable '{executable}' not found")

    env = _env()
    mpirun = shutil.which("mpirun", path=env["PATH"]) or os.path.join(_MPI_ROOT, "bin", "mpirun")
    cmd = [
        mpirun,
        "--allow-run-as-root",
        "-np",
        str(nproc),
        exe_abs,
        str(count),
        str(iters),
    ]

    if verbose:
        print(f"[run] {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=_DEFAULT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as e:
        return RunResult(False, executable, -1, e.stdout or "", e.stderr or "",
                         cmd, error_message=f"Run timed out after {_DEFAULT_TIMEOUT_SEC}s")

    parsed = _parse_json_output(proc.stdout)
    success = proc.returncode == 0 and parsed.get("Correctness") == "PASS"
    if verbose and proc.stderr and not success:
        print(proc.stderr, end="", file=sys.stderr)

    return RunResult(
        success=success,
        executable=executable,
        return_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        command=cmd,
        parsed_output=parsed,
        error_message=None if success else "Execution failed",
    )


def run(executable: str, verbose: bool = True) -> RunResult:
    return _run_with_args(
        executable=executable,
        count=_DEFAULT_COUNT,
        iters=_DEFAULT_ITERS,
        nproc=_DEFAULT_NPROC,
        verbose=verbose,
    )


def build_and_run(
    source_file: str,
    output_file: Optional[str] = None,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    build_only: bool = False,
    run_only: bool = False,
    verbose: bool = True,
    count: int = _DEFAULT_COUNT,
    iters: int = _DEFAULT_ITERS,
    nproc: int = _DEFAULT_NPROC,
) -> BuildAndRunResult:
    build_result: Optional[BuildResult] = None
    run_result: Optional[RunResult] = None

    if not run_only:
        build_result = build(
            source_file=source_file,
            output_file=output_file,
            compiler=compiler,
            platform=platform,
            debug=debug,
            arch=arch,
            verbose=verbose,
        )
        if not build_result.success:
            return BuildAndRunResult(build_result, None)

    if not build_only:
        executable = build_result.output_file if build_result else (output_file or source_file)
        run_result = _run_with_args(
            executable=executable,
            count=count,
            iters=iters,
            nproc=nproc,
            verbose=verbose,
        )

    return BuildAndRunResult(build_result, run_result)


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str) -> None:
    if not metrics:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "data_size_avg": sum(float(m.get("data_size", 0.0)) for m in metrics) / n,
        "latency_avg": sum(float(m.get("latency_avg", 0.0)) for m in metrics) / n,
        "throughput": sum(float(m.get("throughput_avg", 0.0)) for m in metrics) / n,
    }


def _plot_comparison(
    ref_metrics: List[Dict[str, Any]],
    gen_metrics: List[Dict[str, Any]],
    ref_label: str,
    gen_label: str,
    results_dir: str,
    data_size_unit: str,
    latency_unit: str,
    throughput_unit: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots.")
        return

    os.makedirs(results_dir, exist_ok=True)
    for key, ylabel, unit, filename in (
        ("latency_avg", "Latency", latency_unit, "latency_comparison.png"),
        ("throughput_avg", "Throughput", throughput_unit, "throughput_comparison.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([m["data_size"] for m in ref_metrics],
                [m[key] for m in ref_metrics], marker="o", label=ref_label)
        ax.plot([m["data_size"] for m in gen_metrics],
                [m[key] for m in gen_metrics], marker="s", label=gen_label)
        ax.set_xlabel(f"Data Size ({data_size_unit})")
        ax.set_ylabel(f"{ylabel} ({unit})")
        ax.set_title(f"{ylabel} vs Data Size")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, filename), dpi=150)
        plt.close(fig)


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
    count: int = _DEFAULT_COUNT,
    iters: int = _DEFAULT_ITERS,
    nproc: int = _DEFAULT_NPROC,
) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)
    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]
    inner_verbose = show_raw_output

    if verbose:
        print(f"[compare] reference: {src_ref}")
    ref_res = build_and_run(src_ref, compiler=compiler, platform=platform,
                            debug=debug, arch=arch, verbose=inner_verbose,
                            count=count, iters=iters, nproc=nproc)
    if verbose:
        print(f"[compare] generated: {src_gen}")
    gen_res = build_and_run(src_gen, compiler=compiler, platform=platform,
                            debug=debug, arch=arch, verbose=inner_verbose,
                            count=count, iters=iters, nproc=nproc)

    ref_compile_ok = ref_res.build_result is not None and ref_res.build_result.success
    ref_run_ok = ref_res.run_result is not None and ref_res.run_result.success
    gen_compile_ok = gen_res.build_result is not None and gen_res.build_result.success
    gen_run_ok = gen_res.run_result is not None and gen_res.run_result.success
    ref_parsed = ref_res.run_result.parsed_output if ref_run_ok else {}
    gen_parsed = gen_res.run_result.parsed_output if gen_run_ok else {}
    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])

    data_size_unit = ref_parsed.get("data_size_unit", gen_parsed.get("data_size_unit", "MiB"))
    latency_unit = ref_parsed.get("latency_unit", gen_parsed.get("latency_unit", "us"))
    throughput_unit = ref_parsed.get("throughput_unit", gen_parsed.get("throughput_unit", "GB/s"))

    if ref_metrics:
        _save_metrics_csv(ref_metrics, os.path.join(results_dir, f"{ref_label}_metrics.csv"))
    if gen_metrics:
        _save_metrics_csv(gen_metrics, os.path.join(results_dir, f"{gen_label}_metrics.csv"))

    ref_avg = _metrics_avg(ref_metrics)
    gen_avg = _metrics_avg(gen_metrics)
    summary: Dict[str, Any] = {
        "generated_source": os.path.basename(src_gen),
        "ref_source": os.path.basename(src_ref),
        "model": "",
        "pass_iteration": 1,
        "improvement_iteration": 1,
        "data_size_unit": data_size_unit,
        "latency_unit": latency_unit,
        "throughput_unit": throughput_unit,
        "metrics_comparison": {
            "ref": {"compile_success": ref_compile_ok, "run_success": ref_run_ok, **ref_avg},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok, **gen_avg},
        },
        "latency_improvement_pct": 0.0,
        "throughput_improvement_pct": 0.0,
        "performance": "same",
    }

    if ref_avg and gen_avg:
        if ref_avg["latency_avg"]:
            summary["latency_improvement_pct"] = round(
                (ref_avg["latency_avg"] - gen_avg["latency_avg"]) / ref_avg["latency_avg"] * 100.0, 2)
        if ref_avg["throughput"]:
            summary["throughput_improvement_pct"] = round(
                (gen_avg["throughput"] - ref_avg["throughput"]) / ref_avg["throughput"] * 100.0, 2)
        lat_imp = summary["latency_improvement_pct"]
        thr_imp = summary["throughput_improvement_pct"]
        if lat_imp > 5 and thr_imp > 5:
            summary["performance"] = "better"
        elif lat_imp < -5 or thr_imp < -5:
            summary["performance"] = "worse"

        if not no_plot:
            _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label,
                             results_dir, data_size_unit, latency_unit, throughput_unit)

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("\nPERFORMANCE COMPARISON")
        if ref_avg and gen_avg:
            for metric, unit in (
                ("data_size_avg", data_size_unit),
                ("throughput", throughput_unit),
                ("latency_avg", latency_unit),
            ):
                print(f"  [+] {metric}: {gen_avg[metric]:.4f} {unit} vs ref {ref_avg[metric]:.4f} {unit}")
            print(f"  Performance: {summary['performance']}")
        mc = summary["metrics_comparison"]
        print(f"\n{'=' * 60}")
        print(f"  ref       compile_success: {mc['ref']['compile_success']}")
        print(f"  ref       run_success:     {mc['ref']['run_success']}")
        print(f"  generated compile_success: {mc['generated']['compile_success']}")
        print(f"  generated run_success:     {mc['generated']['run_success']}")
        print(f"  performance:               {summary['performance']}")
        print(f"{'=' * 60}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="NCCL Device API inter-node AllReduce build/run helper")
    parser.add_argument("--source", "-s", default=_DEFAULT_SOURCE)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--arch", "-a", default=_DEFAULT_ARCH)
    parser.add_argument("--build-only", "-b", action="store_true")
    parser.add_argument("--run-only", "-r", action="store_true")
    parser.add_argument("--compiler", "-c", default=None)
    parser.add_argument("--platform", default="cuda", choices=["cuda"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--compare", nargs=2, metavar=("SRC_REF", "SRC_GEN"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--count", type=int, default=_DEFAULT_COUNT)
    parser.add_argument("--iters", type=int, default=_DEFAULT_ITERS)
    parser.add_argument("--nproc", type=int, default=_DEFAULT_NPROC)
    parser.add_argument("--show-raw-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
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
        summary = compare(
            args.compare[0],
            args.compare[1],
            results_dir=results_dir,
            compiler=args.compiler,
            platform=args.platform,
            debug=args.debug,
            arch=args.arch,
            verbose=verbose,
            show_raw_output=args.show_raw_output,
            count=args.count,
            iters=args.iters,
            nproc=args.nproc,
        )
        sys.exit(0 if summary["metrics_comparison"]["generated"]["run_success"] else 1)

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
        count=args.count,
        iters=args.iters,
        nproc=args.nproc,
    )

    if args.plot and result.run_result and result.run_result.parsed_output:
        parsed = result.run_result.parsed_output
        metrics = parsed.get("metrics", [])
        label = os.path.splitext(os.path.basename(args.source))[0]
        _save_metrics_csv(metrics, os.path.join(results_dir, f"{label}_metrics.csv"))
        _plot_comparison(metrics, metrics, label, label, results_dir,
                         parsed.get("data_size_unit", "MiB"),
                         parsed.get("latency_unit", "us"),
                         parsed.get("throughput_unit", "GB/s"))

    if result.run_result is not None and result.run_result.success:
        print(json.dumps(result.run_result.parsed_output, indent=None if args.quiet else 2))
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
