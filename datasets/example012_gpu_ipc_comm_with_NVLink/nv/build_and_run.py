#!/usr/bin/env python3
"""
GPU IPC Communication Build and Run Module for NVIDIA CUDA

This module compiles and runs GPU IPC communication programs that require
two cooperating processes (sender + receiver). The sender exports a CUDA IPC
memory handle and the receiver imports it to benchmark D2D copy throughput.

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
import time
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


_DEFAULT_SOURCE = "ref_gpu_ipc_comm.cu"


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


def _detect_gpu_devices() -> tuple:
    """Detect available GPUs and return (sender_dev, receiver_dev).

    If two or more GPUs are available, use devices 0 and 1 so that the
    D2D copy travels over NVLink. Otherwise, both processes share device 0.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = [int(x.strip()) for x in result.stdout.strip().split("\n")
                if x.strip()]
        if len(gpus) >= 2:
            return 0, 1
    except Exception:
        pass
    return 0, 0


def _parse_json_output(text: str) -> Dict[str, Any]:
    """Parse JSON output from stdout."""
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


def build(
    source_file: str,
    output_file: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    verbose: bool = True,
) -> BuildResult:
    """Build a CUDA IPC source file."""
    wd = get_module_dir()

    src_path = source_file if os.path.isabs(source_file) else os.path.join(wd, source_file)
    if not os.path.exists(src_path):
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file,
            return_code=-1, stdout="", stderr="",
            command=[], error_message=f"Source file '{source_file}' not found",
        )

    if compiler is None:
        compiler, detected_platform = _detect_compiler(platform)
        if platform is None:
            platform = detected_platform
    elif platform is None:
        platform = "hip" if "hipcc" in compiler else "cuda"

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
        return_code=result.returncode, stdout=result.stdout,
        stderr=result.stderr, command=cmd,
        error_message=None if success else "Compilation failed",
    )


def run(executable: str, verbose: bool = True) -> RunResult:
    """Run the IPC program as two cooperating processes (sender + receiver).

    The executable is invoked twice with different ``--role`` arguments.
    The sender is started first (it creates and listens on a Unix socket),
    then the receiver connects and the two communicate via CUDA IPC handles.
    JSON output is captured from the receiver's stdout.
    """
    wd = get_module_dir()
    exe_path = executable if os.path.isabs(executable) else os.path.join(wd, executable)

    if not os.path.exists(exe_path):
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=[],
            error_message=f"Executable '{executable}' not found",
        )

    sender_dev, receiver_dev = _detect_gpu_devices()
    sock_path = f"/tmp/cuda_ipc_{os.getpid()}.sock"

    # Clean up stale socket from a previous failed run
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    sender_cmd = [exe_path, "--role", "sender",
                  "--dev", str(sender_dev), "--sock", sock_path]
    receiver_cmd = [exe_path, "--role", "receiver",
                    "--dev", str(receiver_dev), "--sock", sock_path]

    if verbose:
        print(f"Running sender:   {' '.join(sender_cmd)}")
        print(f"Running receiver: {' '.join(receiver_cmd)}")

    # Launch sender (it creates the socket and blocks on accept)
    sender_proc = subprocess.Popen(
        sender_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    # Wait for the socket file to appear (sender has called bind+listen)
    for _ in range(100):
        if os.path.exists(sock_path):
            break
        time.sleep(0.05)
    else:
        sender_proc.kill()
        sender_proc.wait()
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="sender did not create socket in time",
            command=sender_cmd,
            error_message="Sender timed out waiting for socket",
        )

    # Launch receiver
    receiver_proc = subprocess.Popen(
        receiver_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    # Wait for both processes
    try:
        recv_stdout, recv_stderr = receiver_proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        receiver_proc.kill()
        sender_proc.kill()
        receiver_proc.wait()
        sender_proc.wait()
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="receiver timed out",
            command=receiver_cmd, error_message="Receiver timed out",
        )

    try:
        send_stdout, send_stderr = sender_proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        sender_proc.kill()
        sender_proc.wait()
        send_stdout, send_stderr = "", ""

    # Clean up socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    success = receiver_proc.returncode == 0 and sender_proc.returncode == 0
    all_stderr = (send_stderr + recv_stderr).strip()

    if verbose:
        if recv_stdout:
            print(recv_stdout)
        if all_stderr:
            print(all_stderr, file=sys.stderr)

    parsed = _parse_json_output(recv_stdout) if success else {}

    return RunResult(
        success=success, executable=executable,
        return_code=receiver_proc.returncode,
        stdout=recv_stdout, stderr=all_stderr,
        command=receiver_cmd, parsed_output=parsed,
        error_message=None if success else
            f"sender rc={sender_proc.returncode}, "
            f"receiver rc={receiver_proc.returncode}",
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
    """Build and run a CUDA IPC source file."""
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


# ── Metrics / plotting helpers ───────────────────────────────────────────────

def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "data_size_avg": sum(m.get("data_size", 0) for m in metrics) / n,
        "latency_avg": sum(m.get("latency_avg", 0) for m in metrics) / n,
        "throughput": sum(m.get("throughput_avg", 0) for m in metrics) / n,
    }


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str):
    if not metrics:
        return
    keys = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label,
                     results_dir, data_size_unit="MB", latency_unit="us",
                     throughput_unit="Gbps"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    os.makedirs(results_dir, exist_ok=True)

    ref_sizes = [m["data_size"] for m in ref_metrics]
    ref_lat = [m["latency_avg"] for m in ref_metrics]
    ref_thr = [m["throughput_avg"] for m in ref_metrics]
    gen_sizes = [m["data_size"] for m in gen_metrics]
    gen_lat = [m["latency_avg"] for m in gen_metrics]
    gen_thr = [m["throughput_avg"] for m in gen_metrics]

    for y_ref, y_gen, ylabel, unit, fname in [
        (ref_lat, gen_lat, "Latency", latency_unit, "latency_comparison.png"),
        (ref_thr, gen_thr, "Throughput", throughput_unit,
         "throughput_comparison.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ref_sizes, y_ref, marker="o", label=ref_label, linewidth=2)
        ax.plot(gen_sizes, y_gen, marker="s", label=gen_label, linewidth=2)
        ax.set_xlabel(f"Data Size ({data_size_unit})")
        ax.set_ylabel(f"{ylabel} ({unit})")
        ax.set_title(f"{ylabel} vs Data Size")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout()
        out = os.path.join(results_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


def _plot_single(metrics, label, results_dir, data_size_unit="MB",
                 latency_unit="us", throughput_unit="Gbps"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    os.makedirs(results_dir, exist_ok=True)
    sizes = [m["data_size"] for m in metrics]

    for values, ylabel, unit, fname in [
        ([m["latency_avg"] for m in metrics], "Latency", latency_unit,
         "latency.png"),
        ([m["throughput_avg"] for m in metrics], "Throughput", throughput_unit,
         "throughput.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sizes, values, marker="o", label=label, linewidth=2)
        ax.set_xlabel(f"Data Size ({data_size_unit})")
        ax.set_ylabel(f"{ylabel} ({unit})")
        ax.set_title(f"{ylabel} vs Data Size")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout()
        out = os.path.join(results_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


def _compare_metrics(ref_metrics, gen_metrics):
    result = {"ref": ref_metrics, "generated": gen_metrics, "comparison": {},
              "summary": {}}
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
            is_lower_better = ("latency" in lower_key or "time" in lower_key)
            if is_lower_better:
                improvement = (ref_val - gen_val) / ref_val * 100
                better = gen_val <= ref_val
            else:
                improvement = (gen_val - ref_val) / ref_val * 100
                better = gen_val >= ref_val
            result["comparison"][key] = {
                "ref": ref_val, "generated": gen_val,
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


# ── Compare ──────────────────────────────────────────────────────────────────

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
    """Compare a generated file against the reference file."""
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]

    def _run_one(src, label, exe_name):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[compare] Building & running: {src}  ({label})")
            print(f"{'='*60}")
        res = build_and_run(
            source_file=src, output_file=exe_name,
            compiler=compiler, platform=platform, debug=debug, arch=arch,
            verbose=show_raw_output,
        )
        compile_ok = res.build_result is not None and res.build_result.success
        run_ok = res.run_result is not None and res.run_result.success
        parsed = res.run_result.parsed_output if run_ok else {}
        if verbose:
            if not compile_ok:
                print(f"[compare] BUILD FAILED for {src}")
                if res.build_result:
                    print(res.build_result.stderr[-2000:], file=sys.stderr)
            elif not run_ok:
                print(f"[compare] RUN FAILED for {src}")
                if res.run_result:
                    print(res.run_result.stderr[-2000:], file=sys.stderr)
        return compile_ok, run_ok, parsed

    ref_compile_ok, ref_run_ok, ref_parsed = _run_one(src_ref, ref_label,
                                                       "ref_exec")
    gen_compile_ok, gen_run_ok, gen_parsed = _run_one(src_gen, gen_label,
                                                       "gen_exec")

    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])

    data_size_unit = ref_parsed.get("data_size_unit",
                                    gen_parsed.get("data_size_unit", "MB"))
    latency_unit = ref_parsed.get("latency_unit",
                                  gen_parsed.get("latency_unit", "us"))
    throughput_unit = ref_parsed.get("throughput_unit",
                                    gen_parsed.get("throughput_unit", "Gbps"))

    # Save CSV
    for metrics, label in [(ref_metrics, ref_label), (gen_metrics, gen_label)]:
        if metrics:
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            _save_metrics_csv(metrics, csv_path)
            if verbose:
                print(f"Raw metrics saved to {csv_path}")

    has_perf = bool(ref_metrics) and bool(gen_metrics)

    summary: Dict[str, Any] = {
        "generated_source": os.path.basename(src_gen),
        "ref_source": os.path.basename(src_ref),
        "model": "",
        "pass_iteration": 1,
    }

    if has_perf:
        ref_avg = _metrics_avg(ref_metrics)
        gen_avg = _metrics_avg(gen_metrics)

        summary["improvement_iteration"] = 1
        summary["data_size_unit"] = data_size_unit
        summary["latency_unit"] = latency_unit
        summary["throughput_unit"] = throughput_unit
        summary["metrics_comparison"] = {
            "ref": {"compile_success": ref_compile_ok,
                    "run_success": ref_run_ok, **ref_avg},
            "generated": {"compile_success": gen_compile_ok,
                          "run_success": gen_run_ok, **gen_avg},
        }

        lat_imp = (
            (ref_avg["latency_avg"] - gen_avg["latency_avg"])
            / ref_avg["latency_avg"] * 100
            if ref_avg.get("latency_avg") else 0.0
        )
        thr_imp = (
            (gen_avg["throughput"] - ref_avg["throughput"])
            / ref_avg["throughput"] * 100
            if ref_avg.get("throughput") else 0.0
        )
        summary["latency_improvement_pct"] = round(lat_imp, 2)
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
                             results_dir, data_size_unit, latency_unit,
                             throughput_unit)

        if verbose:
            print(f"\nPERFORMANCE COMPARISON (averaged over "
                  f"{len(ref_metrics)} {ref_label} / "
                  f"{len(gen_metrics)} {gen_label} records)")
            flag_thr = "+" if thr_imp >= 0 else "-"
            flag_lat = "+" if lat_imp >= 0 else "-"
            print(f"  [+] data_size_avg: {ref_avg['data_size_avg']:.1f}"
                  f" {data_size_unit}")
            print(f"  [{flag_thr}] throughput: {gen_avg['throughput']:.3f}"
                  f" vs ref {ref_avg['throughput']:.3f} {throughput_unit}"
                  f" ({thr_imp:+.1f}%)")
            print(f"  [{flag_lat}] latency_avg: {gen_avg['latency_avg']:.3f}"
                  f" vs ref {ref_avg['latency_avg']:.3f} {latency_unit}"
                  f" ({lat_imp:+.1f}%)")
            print(f"  Performance: {summary['performance']}")
    else:
        summary["metrics_comparison"] = {
            "ref": {"compile_success": ref_compile_ok,
                    "run_success": ref_run_ok},
            "generated": {"compile_success": gen_compile_ok,
                          "run_success": gen_run_ok},
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
        print(f"  generated compile_success:"
              f" {mc['generated']['compile_success']}")
        print(f"  generated run_success:    "
              f" {mc['generated']['run_success']}")
        print(f"  performance:              "
              f" {summary.get('performance', 'N/A')}")
        print(f"{'='*60}")

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build and run GPU IPC communication program"
    )
    parser.add_argument(
        "--source", "-s", default=_DEFAULT_SOURCE,
        help=f"Source file to compile (default: {_DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output", "-o", default="gpu_ipc_comm",
        help="Output executable name (default: gpu_ipc_comm)",
    )
    parser.add_argument("--arch", "-a", default=None,
                        help="GPU architecture (e.g., sm_80)")
    parser.add_argument("--build-only", "-b", action="store_true",
                        help="Only build, do not run")
    parser.add_argument("--run-only", "-r", action="store_true",
                        help="Only run (assume already built)")
    parser.add_argument("--compiler", "-c", default=None,
                        help="Specify compiler path")
    parser.add_argument("--platform", "-p", choices=["hip", "cuda"],
                        default=None, help="Force platform")
    parser.add_argument("--plot", action="store_true",
                        help="Generate benchmark plots after running")
    parser.add_argument("--results-dir", default=None,
                        help="Directory to save results")
    parser.add_argument(
        "--compare", nargs=2, metavar=("SRC_A", "SRC_B"), default=None,
        help="Compare two source files",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug build")
    parser.add_argument("--show-raw-output", action="store_true",
                        help="Print raw stdout/stderr in compare mode")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")
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
            os.makedirs(results_dir, exist_ok=True)
            _plot_single(
                metrics, label, results_dir,
                parsed.get("data_size_unit", "MB"),
                parsed.get("latency_unit", "us"),
                parsed.get("throughput_unit", "Gbps"),
            )
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            _save_metrics_csv(metrics, csv_path)
            if verbose:
                print(f"Saved {csv_path}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
