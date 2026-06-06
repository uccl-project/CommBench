#!/usr/bin/env python3
"""
DeepEP standalone CUDA dataset build/run helper.

This helper is used by the DeepEP CUDA dataset example in this directory.

It supports three common paths:

1. Build one ref/empty/generated .cu source file.
2. Run the executable and parse the JSON object printed on stdout.
3. In compare mode, run reference and generated sources, then write CSV files,
   optional plots, and summary.json.

The public functions and CLI flags follow the other dataset helpers so
scripts/generate_eval_one.py can call this file directly.
"""

import argparse
import csv
import glob
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


def _find_default_source() -> str:
    for name in sorted(os.listdir(get_module_dir())):
        if name.startswith("ref_") and name.endswith(".cu"):
            return name
    return "ref_example.cu"


_DEFAULT_SOURCE = _find_default_source()


def _detect_compiler() -> str:
    nvcc = shutil.which("nvcc", path=_build_env().get("PATH"))
    if nvcc:
        return "nvcc"

    candidates = ["/usr/local/cuda/bin/nvcc"]
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return "nvcc"


def _relative_to_module(path: str) -> str:
    """Return a module-directory-relative path when that is practical."""
    if not os.path.isabs(path):
        return path
    try:
        rel = os.path.relpath(path, get_module_dir())
        if not rel.startswith(".."):
            return rel
    except ValueError:
        pass
    return path


def _relative_executable(path: str) -> str:
    rel = _relative_to_module(path)
    if os.path.isabs(rel):
        return rel
    if os.path.dirname(rel):
        return rel
    return f".{os.sep}{rel}"


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()

    path_parts = ["/usr/local/nvidia/bin", "/usr/local/cuda-12.8/bin", "/usr/local/cuda/bin"]
    existing_path = env.get("PATH", "")
    env["PATH"] = ":".join(path_parts + ([existing_path] if existing_path else []))

    lib_parts = ["/usr/local/nvidia/lib64", "/usr/local/cuda-12.8/lib64", "/usr/local/cuda/lib64"]
    existing_lib = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(lib_parts + ([existing_lib] if existing_lib else []))

    return env


def _detect_nvidia_smi() -> str:
    for candidate in (
        shutil.which("nvidia-smi"),
        "/usr/local/nvidia/bin/nvidia-smi",
        "/usr/bin/nvidia-smi",
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return "nvidia-smi"


def _detect_cuda_arch() -> Optional[str]:
    try:
        result = subprocess.run(
            [_detect_nvidia_smi(), "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            env=_build_env(),
            timeout=10,
        )
        cap = result.stdout.strip().splitlines()[0].strip()
        if not cap:
            return None
        major, minor = cap.split(".")
        return f"sm_{major}{minor}"
    except Exception:
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
        "data_size_avg": sum(float(m.get("data_size", 0)) for m in metrics) / n,
        "latency_avg": sum(float(m.get("latency_avg", 0)) for m in metrics) / n,
        "throughput": sum(float(m.get("throughput_avg", 0)) for m in metrics) / n,
    }


def _plot_comparison(
    ref_metrics,
    gen_metrics,
    ref_label,
    gen_label,
    results_dir,
    data_size_unit="MB",
    latency_unit="us",
    throughput_unit="Gbps",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    os.makedirs(results_dir, exist_ok=True)
    ref_sizes = [m["data_size"] for m in ref_metrics]
    gen_sizes = [m["data_size"] for m in gen_metrics]

    plots = [
        (
            [m["latency_avg"] for m in ref_metrics],
            [m["latency_avg"] for m in gen_metrics],
            "Latency",
            latency_unit,
            "latency_comparison.png",
        ),
        (
            [m["throughput_avg"] for m in ref_metrics],
            [m["throughput_avg"] for m in gen_metrics],
            "Throughput",
            throughput_unit,
            "throughput_comparison.png",
        ),
    ]

    for y_ref, y_gen, ylabel, unit, filename in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ref_sizes, y_ref, marker="o", label=ref_label, linewidth=2)
        ax.plot(gen_sizes, y_gen, marker="s", label=gen_label, linewidth=2)
        ax.set_xlabel(f"Data Size ({data_size_unit})")
        ax.set_ylabel(f"{ylabel} ({unit})")
        ax.set_title(f"{ylabel} Comparison")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, filename), dpi=150)
        plt.close(fig)


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
        return BuildResult(False, source_file, output_file, -1, "", "", [], f"source not found: {source_file}")

    compiler = compiler or _detect_compiler()
    out_path = output_file if os.path.isabs(output_file) else os.path.join(wd, output_file)
    flags = ["-std=c++17"]
    flags.extend(["-g", "-G"] if debug else ["-O3"])
    arch = arch or _detect_cuda_arch()
    if arch:
        flags.extend(["-arch", arch])

    src_arg = source_file if not os.path.isabs(source_file) else _relative_to_module(src_path)
    out_arg = output_file if not os.path.isabs(output_file) else _relative_to_module(out_path)
    cmd = [compiler] + flags + [src_arg, "-o", out_arg]
    if verbose:
        print(f"[build] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=_build_env(), cwd=wd)
    success = result.returncode == 0
    if verbose:
        print("Build successful!" if success else "Build failed!")
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
    return BuildResult(
        success,
        source_file,
        output_file,
        result.returncode,
        result.stdout,
        result.stderr,
        cmd,
        None if success else "Compilation failed",
    )


def run(executable: str, verbose: bool = True) -> RunResult:
    wd = get_module_dir()
    exe_path = executable if os.path.isabs(executable) else os.path.join(wd, executable)
    if not os.path.exists(exe_path):
        return RunResult(False, executable, -1, "", "", [], {}, f"executable not found: {executable}")

    cmd = [_relative_executable(exe_path)]
    if verbose:
        print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=_build_env(), cwd=wd)
    parsed = _parse_json_output(result.stdout)
    success = result.returncode == 0 and bool(parsed)
    if verbose:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
    return RunResult(
        success,
        executable,
        result.returncode,
        result.stdout,
        result.stderr,
        cmd,
        parsed,
        None if success else "Execution failed or JSON output missing",
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
        build_result = build(source_file, output_file, compiler, platform, debug, arch, verbose)
        if not build_result.success:
            return BuildAndRunResult(build_result, None)
    if not build_only:
        run_result = run(output_file, verbose)
    return BuildAndRunResult(build_result, run_result)


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
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]

    def run_one(src: str, label: str):
        output = os.path.join(results_dir, f"{label}.out")
        res = build_and_run(src, output, compiler, platform, debug, arch, verbose=show_raw_output)
        compile_ok = res.build_result is not None and res.build_result.success
        run_ok = res.run_result is not None and res.run_result.success
        parsed = res.run_result.parsed_output if run_ok else {}
        if verbose:
            print(f"[compare] {label}: compile={compile_ok}, run={run_ok}")
            if not run_ok and res.run_result:
                print(res.run_result.stderr[-2000:], file=sys.stderr)
        return compile_ok, run_ok, parsed

    ref_compile_ok, ref_run_ok, ref_parsed = run_one(src_ref, ref_label)
    gen_compile_ok, gen_run_ok, gen_parsed = run_one(src_gen, gen_label)

    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])
    data_size_unit = ref_parsed.get("data_size_unit", gen_parsed.get("data_size_unit", ""))
    latency_unit = ref_parsed.get("latency_unit", gen_parsed.get("latency_unit", ""))
    throughput_unit = ref_parsed.get("throughput_unit", gen_parsed.get("throughput_unit", ""))

    if ref_metrics:
        _save_metrics_csv(ref_metrics, os.path.join(results_dir, f"{ref_label}_metrics.csv"))
    if gen_metrics:
        _save_metrics_csv(gen_metrics, os.path.join(results_dir, f"{gen_label}_metrics.csv"))

    summary: Dict[str, Any] = {
        "generated_source": os.path.basename(src_gen),
        "ref_source": os.path.basename(src_ref),
        "model": "",
        "pass_iteration": 1,
        "metrics_comparison": {
            "ref": {"compile_success": ref_compile_ok, "run_success": ref_run_ok},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok},
        },
    }

    if ref_metrics and gen_metrics:
        ref_avg = _metrics_avg(ref_metrics)
        gen_avg = _metrics_avg(gen_metrics)
        summary.update(
            {
                "improvement_iteration": 1,
                "data_size_unit": data_size_unit,
                "latency_unit": latency_unit,
                "throughput_unit": throughput_unit,
                "metrics_comparison": {
                    "ref": {"compile_success": ref_compile_ok, "run_success": ref_run_ok, **ref_avg},
                    "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok, **gen_avg},
                },
            }
        )
        lat_imp = (
            (ref_avg["latency_avg"] - gen_avg["latency_avg"]) / ref_avg["latency_avg"] * 100.0
            if ref_avg.get("latency_avg")
            else 0.0
        )
        thr_imp = (
            (gen_avg["throughput"] - ref_avg["throughput"]) / ref_avg["throughput"] * 100.0
            if ref_avg.get("throughput")
            else 0.0
        )
        summary["latency_improvement_pct"] = round(lat_imp, 2)
        summary["throughput_improvement_pct"] = round(thr_imp, 2)
        if abs(lat_imp) < 5 and abs(thr_imp) < 5:
            summary["performance"] = "same"
        elif lat_imp > 5 and thr_imp > 5:
            summary["performance"] = "better"
        elif lat_imp < -5 or thr_imp < -5:
            summary["performance"] = "worse"
        else:
            summary["performance"] = "same"
        if not no_plot:
            _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label, results_dir, data_size_unit, latency_unit, throughput_unit)

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/run/compare standalone CUDA DeepEP dataset example")
    parser.add_argument("--source", "-s", default=_DEFAULT_SOURCE)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--arch", "-a", default=None)
    parser.add_argument("--build-only", "-b", action="store_true")
    parser.add_argument("--run-only", "-r", action="store_true")
    parser.add_argument("--compiler", "-c", default=None)
    parser.add_argument("--platform", "-p", choices=["cuda"], default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--compare", nargs=2, metavar=("SRC_REF", "SRC_GEN"), default=None)
    parser.add_argument("--compare-no-plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
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
        src_ref, src_gen = args.compare
        compare(
            src_ref,
            src_gen,
            results_dir,
            compiler=args.compiler,
            platform=args.platform,
            debug=args.debug,
            arch=args.arch,
            no_plot=args.compare_no_plot,
            verbose=verbose,
            show_raw_output=args.show_raw_output,
        )
        return 0

    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.source))[0]
        output = os.path.join(get_module_dir(), stem)

    result = build_and_run(
        args.source,
        output,
        compiler=args.compiler,
        platform=args.platform,
        debug=args.debug,
        arch=args.arch,
        build_only=args.build_only,
        run_only=args.run_only,
        verbose=verbose,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
