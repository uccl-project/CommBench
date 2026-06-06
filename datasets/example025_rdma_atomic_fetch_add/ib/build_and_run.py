#!/usr/bin/env python3
"""
RDMA Atomic Fetch-and-Add Build and Run Module

Compiles and runs an RDMA atomic fetch-and-add correctness test.
Two RC QPs on the client both issue FAA against the same remote counter;
the final value must equal the sum of both increments.

Requires g++ and libibverbs.

Usage as module:
    from build_and_run import build, run, build_and_run, compare

Usage as script:
    python build_and_run.py --source ref_rdma_atomic_fetch_add.cpp
    python build_and_run.py --compare ref_rdma_atomic_fetch_add.cpp generated.cpp
"""

import subprocess
import sys
import os
import json
import csv
import time
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


def _parse_json_output(text: str) -> Dict[str, Any]:
    """Find and parse the first JSON object in text."""
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
                    start = None
    return {}


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
    return {k: round(sums[k] / counts[k], 6) for k in sums}


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
            is_lower_better = ("latency" in lower_key or "time" in lower_key)
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


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str):
    """Save metrics list to CSV."""
    if not metrics:
        return
    all_keys = list(dict.fromkeys(k for m in metrics for k in m.keys()))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(metrics)


def _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label, results_dir,
                     latency_key="latency_avg", throughput_key="throughput_avg",
                     latency_unit="us", throughput_unit="ops/s"):
    """Generate latency_comparison.png for a single-point comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed - cannot generate plots.")
        return

    os.makedirs(results_dir, exist_ok=True)

    labels = [ref_label, gen_label]
    lat_vals = [
        ref_metrics[0].get(latency_key, 0) if ref_metrics else 0,
        gen_metrics[0].get(latency_key, 0) if gen_metrics else 0,
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, lat_vals)
    ax.set_ylabel(f"Latency ({latency_unit})")
    ax.set_title("Latency Comparison")
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    fig.tight_layout()
    lat_path = os.path.join(results_dir, "latency_comparison.png")
    fig.savefig(lat_path, dpi=150)
    plt.close(fig)
    print(f"Saved {lat_path}")


def check_gxx(verbose: bool = False) -> tuple:
    """Check if g++ is available in PATH."""
    try:
        result = subprocess.run(
            ["g++", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if verbose:
                print("g++: found")
            return True, "g++ found"
        return False, "g++ returned non-zero"
    except FileNotFoundError:
        return False, "g++ not found in PATH"


def check_ibverbs(verbose: bool = False) -> tuple:
    """Check if libibverbs is available."""
    header_paths = [
        "/usr/include/infiniband/verbs.h",
        "/usr/local/include/infiniband/verbs.h",
    ]
    header_found = any(os.path.exists(p) for p in header_paths)
    if not header_found:
        return False, "infiniband/verbs.h not found"

    lib_paths = [
        "/usr/lib/x86_64-linux-gnu/libibverbs.so",
        "/usr/lib64/libibverbs.so",
        "/usr/lib/libibverbs.so",
        "/usr/lib/aarch64-linux-gnu/libibverbs.so",
    ]
    lib_found = any(os.path.exists(p) for p in lib_paths)
    if not lib_found:
        try:
            result = subprocess.run(
                ["pkg-config", "--exists", "libibverbs"],
                capture_output=True,
            )
            lib_found = result.returncode == 0
        except FileNotFoundError:
            pass

    if lib_found:
        if verbose:
            print("libibverbs: found")
        return True, "libibverbs found"
    return False, "libibverbs not found"


def list_rdma_devices() -> List[str]:
    """List available RDMA devices."""
    try:
        result = subprocess.run(["ibv_devices"], capture_output=True, text=True)
        if result.returncode != 0:
            return []
        devices = []
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if parts:
                devices.append(parts[0])
        return devices
    except FileNotFoundError:
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
    """Build an RDMA atomic fetch-and-add source file using g++.

    Args:
        source_file: Path to the source file
        output_file: Path to the output executable
        compiler: Compiler path (default: g++)
        platform: Accepted but not used for RDMA/IB builds
        debug: Enable debug build (-g)
        arch: Accepted but not used for RDMA/IB builds
        verbose: Print build progress
    """
    wd = get_module_dir()

    src_path = source_file if os.path.isabs(source_file) else os.path.join(wd, source_file)
    if not os.path.exists(src_path):
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file,
            return_code=-1, stdout="", stderr="",
            command=[], error_message=f"Source file '{source_file}' not found",
        )

    if compiler is None:
        compiler = "g++"

    gxx_available, gxx_msg = check_gxx(verbose)
    if not gxx_available:
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file,
            return_code=-1, stdout="", stderr=gxx_msg, command=[],
            error_message=f"g++ not found: {gxx_msg}",
        )

    ibverbs_available, ibverbs_msg = check_ibverbs(verbose)
    if not ibverbs_available:
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file,
            return_code=-1, stdout="", stderr=ibverbs_msg, command=[],
            error_message=f"libibverbs not found: {ibverbs_msg}",
        )

    flags = ["-O2", "-std=c++17", "-Wall"]
    if debug:
        flags = ["-g", "-std=c++17", "-Wall"]

    out_path = output_file if os.path.isabs(output_file) else os.path.join(wd, output_file)
    cmd = [compiler] + flags + [src_path, "-o", out_path, "-libverbs"]

    if verbose:
        print("===================================")
        print("Building RDMA Atomic Fetch-and-Add Test (g++)")
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
        if success:
            print(f"Executable: {output_file}")
        print("===================================")

    return BuildResult(
        success=success, source_file=source_file, output_file=output_file,
        return_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
        command=cmd, error_message=None if success else "Compilation failed",
    )


def run(
    executable: str,
    verbose: bool = True,
    server_nic: int = 0,
    client_nic: int = 0,
    port: int = 9999,
    server_addr: str = "127.0.0.1",
    gid_index: Optional[int] = None,
    timeout_sec: int = 60,
) -> RunResult:
    """Run the RDMA atomic fetch-and-add program (starts server and client processes).

    Args:
        executable: Path to the executable
        verbose: Print run progress
        server_nic: NIC index for server (default: 0)
        client_nic: NIC index for client (default: 0)
        port: TCP port for metadata exchange (default: 9999)
        server_addr: Server address for client connection (default: 127.0.0.1)
        gid_index: GID index (auto-detect if None)
        timeout_sec: Timeout in seconds (default: 60)
    """
    wd = get_module_dir()
    exe_path = executable if os.path.isabs(executable) else os.path.join(wd, executable)

    if not os.path.exists(exe_path):
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=[],
            error_message=f"Executable '{executable}' not found",
        )

    server_cmd = [exe_path, "server", "--nic", str(server_nic), "--port", str(port)]
    client_cmd = [exe_path, "client", "--nic", str(client_nic),
                  "--server", server_addr, "--port", str(port)]

    if gid_index is not None:
        server_cmd += ["--gid-index", str(gid_index)]
        client_cmd += ["--gid-index", str(gid_index)]

    if verbose:
        print(f"Server: {' '.join(server_cmd)}")
        print(f"Client: {' '.join(client_cmd)}")

    server_proc = subprocess.Popen(
        server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    time.sleep(1)

    try:
        client_proc = subprocess.Popen(
            client_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        try:
            client_stdout, client_stderr = client_proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            client_proc.kill()
            client_stdout, client_stderr = client_proc.communicate()
            server_proc.kill()
            server_proc.communicate()
            return RunResult(
                success=False, executable=executable, return_code=-1,
                stdout="", stderr="Client timed out",
                command=client_cmd, error_message="Client process timed out",
            )

        try:
            server_stdout, server_stderr = server_proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_stdout, server_stderr = server_proc.communicate()
            return RunResult(
                success=False, executable=executable, return_code=-1,
                stdout=client_stdout, stderr="Server timed out",
                command=server_cmd, error_message="Server process timed out",
            )

    except Exception as e:
        server_proc.kill()
        server_proc.communicate()
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr=str(e), command=client_cmd,
            error_message=str(e),
        )

    combined_stdout = (
        f"[server]\n{server_stdout.strip()}\n"
        f"[client]\n{client_stdout.strip()}"
    )
    combined_stderr = ""
    if server_stderr.strip():
        combined_stderr += f"[server stderr]\n{server_stderr.strip()}\n"
    if client_stderr.strip():
        combined_stderr += f"[client stderr]\n{client_stderr.strip()}"

    if verbose:
        print(combined_stdout)
        if combined_stderr:
            print(combined_stderr, file=sys.stderr)

    server_ok = server_proc.returncode == 0
    client_ok = client_proc.returncode == 0
    success = server_ok and client_ok

    parsed = _parse_json_output(client_stdout) if client_ok else {}
    if not parsed:
        parsed = _parse_json_output(server_stdout) if server_ok else {}

    if success and "PASS" not in server_stdout and "PASS" not in client_stdout:
        success = False

    return RunResult(
        success=success, executable=executable,
        return_code=0 if success else (client_proc.returncode or server_proc.returncode or 1),
        stdout=combined_stdout, stderr=combined_stderr,
        command=client_cmd, parsed_output=parsed,
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
    server_nic: int = 0,
    client_nic: int = 0,
    port: int = 9999,
    server_addr: str = "127.0.0.1",
    gid_index: Optional[int] = None,
) -> BuildAndRunResult:
    """Build and run an RDMA atomic fetch-and-add source file."""
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
            print("Running RDMA Atomic Fetch-and-Add Test")
            print("===================================")
        run_result = run(
            executable=output_file, verbose=verbose,
            server_nic=server_nic, client_nic=client_nic,
            port=port, server_addr=server_addr, gid_index=gid_index,
        )

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    server_nic: int = 0,
    client_nic: int = 0,
    gid_index: Optional[int] = None,
    no_plot: bool = False,
    verbose: bool = True,
    show_raw_output: bool = False,
) -> Dict[str, Any]:
    """Compare two source files by building and running each once.

    Returns the summary dict.
    """
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]

    all_results: Dict[str, List[Dict[str, Any]]] = {ref_label: [], gen_label: []}
    sources = {ref_label: (src_ref, "ref_exec"), gen_label: (src_gen, "gen_exec")}

    build_status: Dict[str, bool] = {}
    run_status: Dict[str, bool] = {}

    base_port = 9999
    port_offset = 0
    for label, (src, exe_name) in sources.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"[compare] Building: {src}")
            print(f"{'='*60}")
        br = build(
            source_file=src, output_file=exe_name,
            compiler=compiler, platform=platform,
            debug=debug, arch=arch, verbose=show_raw_output,
        )
        build_status[label] = br.success
        if not br.success:
            if verbose:
                print(f"[compare] BUILD FAILED for {src}")
            run_status[label] = False
            continue

        port = base_port + port_offset
        port_offset += 1
        rr = run(
            executable=exe_name, verbose=show_raw_output,
            server_nic=server_nic, client_nic=client_nic,
            port=port, gid_index=gid_index,
        )
        run_status[label] = rr.success
        if rr.success and rr.parsed_output:
            all_results[label].append(rr.parsed_output)

    ref_metrics = all_results[ref_label]
    gen_metrics = all_results[gen_label]

    for label, pts in all_results.items():
        if pts:
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            _save_metrics_csv(pts, csv_path)
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
        summary["improvement_iteration"] = 1
        summary["latency_unit"] = "us"
        summary["throughput_unit"] = "ops/s"

        ref_avg = _average_metrics(ref_metrics)
        gen_avg = _average_metrics(gen_metrics)

        summary["metrics_comparison"] = {
            "ref": {
                "compile_success": build_status.get(ref_label, False),
                "run_success": run_status.get(ref_label, False),
                **ref_avg,
            },
            "generated": {
                "compile_success": build_status.get(gen_label, False),
                "run_success": run_status.get(gen_label, False),
                **gen_avg,
            },
        }

        lat_key = "latency_avg"
        thr_key = "throughput_avg"

        if ref_avg.get(lat_key, 0) != 0:
            lat_imp = (ref_avg[lat_key] - gen_avg.get(lat_key, 0)) / ref_avg[lat_key] * 100
            summary["latency_improvement_pct"] = round(lat_imp, 2)
        else:
            summary["latency_improvement_pct"] = 0.0

        if ref_avg.get(thr_key, 0) != 0:
            thr_imp = (gen_avg.get(thr_key, 0) - ref_avg[thr_key]) / ref_avg[thr_key] * 100
            summary["throughput_improvement_pct"] = round(thr_imp, 2)
        else:
            summary["throughput_improvement_pct"] = 0.0

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

        if not no_plot:
            _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label, results_dir)

        if verbose:
            comparison = _compare_metrics(ref_avg, gen_avg)
            if comparison.get("comparison"):
                print(f"\nPERFORMANCE COMPARISON")
                for metric, comp in comparison["comparison"].items():
                    flag = "+" if comp["better_or_equal"] else "-"
                    print(f"  [{flag}] {metric}: {comp['generated']:.4f} vs "
                          f"ref {comp['ref']:.4f} "
                          f"(ratio: {comp['ratio']:.2f}, "
                          f"{comp['improvement_pct']:+.1f}%)")
                print(f"  Performance: {summary['performance']}")
    else:
        summary["metrics_comparison"] = {
            "ref": {
                "compile_success": build_status.get(ref_label, False),
                "run_success": run_status.get(ref_label, False),
            },
            "generated": {
                "compile_success": build_status.get(gen_label, False),
                "run_success": run_status.get(gen_label, False),
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
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build and run RDMA atomic fetch-and-add program (g++ + libibverbs)"
    )
    parser.add_argument(
        "--source", "-s", default="ref_rdma_atomic_fetch_add.cpp",
        help="Source file to compile (default: ref_rdma_atomic_fetch_add.cpp)",
    )
    parser.add_argument(
        "--output", "-o", default="rdma_fetch_add",
        help="Output executable name (default: rdma_fetch_add)",
    )
    parser.add_argument(
        "--arch", "-a", default=None,
        help="Architecture (not used for RDMA/IB builds)",
    )
    parser.add_argument(
        "--build-only", "-b", action="store_true",
        help="Only build, do not run",
    )
    parser.add_argument(
        "--run-only", "-r", action="store_true",
        help="Only run (assume already built)",
    )
    parser.add_argument(
        "--compiler", "-c", default=None,
        help="Specify compiler path (default: g++)",
    )
    parser.add_argument(
        "--platform", "-p", choices=["hip", "cuda"],
        default=None,
        help="Force platform (not used for RDMA/IB builds)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Run and generate benchmark plots",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("SRC_A", "SRC_B"), default=None,
        help="Compare two source files: build & run both, plot. "
             "Example: --compare ref_rdma_atomic_fetch_add.cpp generated.cpp",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug build")
    parser.add_argument(
        "--server-nic", type=int, default=0,
        help="NIC index for server (default: 0)",
    )
    parser.add_argument(
        "--client-nic", type=int, default=0,
        help="NIC index for client (default: 0)",
    )
    parser.add_argument(
        "--port", type=int, default=9999,
        help="TCP port for metadata exchange (default: 9999)",
    )
    parser.add_argument(
        "--server-addr", default="127.0.0.1",
        help="Server address for client connection (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--gid-index", type=int, default=None,
        help="GID index (auto-detect if not set)",
    )
    parser.add_argument(
        "--show-raw-output", action="store_true",
        help="Print raw stdout/stderr from program execution",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List RDMA devices and exit",
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

    if args.list_devices:
        devices = list_rdma_devices()
        if devices:
            print("Available RDMA devices:")
            for i, dev in enumerate(devices):
                print(f"  [{i}] {dev}")
        else:
            print("No RDMA devices found")
        return

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
            server_nic=args.server_nic, client_nic=args.client_nic,
            gid_index=args.gid_index,
            verbose=verbose,
            show_raw_output=args.show_raw_output,
        )
        sys.exit(0)

    if args.plot:
        label = os.path.splitext(os.path.basename(args.source))[0]
        os.makedirs(results_dir, exist_ok=True)

        br = build(
            source_file=args.source, output_file=args.output,
            compiler=args.compiler, platform=args.platform,
            debug=args.debug, arch=args.arch, verbose=verbose,
        )
        if not br.success:
            sys.exit(1)

        rr = run(
            executable=args.output, verbose=verbose,
            server_nic=args.server_nic, client_nic=args.client_nic,
            port=args.port, server_addr=args.server_addr, gid_index=args.gid_index,
        )
        if rr.success and rr.parsed_output:
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            _save_metrics_csv([rr.parsed_output], csv_path)
            print(f"Saved {csv_path}")
        sys.exit(0 if rr.success else 1)

    result = build_and_run(
        source_file=args.source, output_file=args.output,
        compiler=args.compiler, platform=args.platform,
        debug=args.debug, arch=args.arch,
        build_only=args.build_only, run_only=args.run_only,
        verbose=verbose,
        server_nic=args.server_nic, client_nic=args.client_nic,
        port=args.port, server_addr=args.server_addr, gid_index=args.gid_index,
    )

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
