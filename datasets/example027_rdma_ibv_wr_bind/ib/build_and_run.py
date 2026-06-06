#!/usr/bin/env python3
"""
RDMA Memory Window Bind/Revoke Build and Run Module

Compiles and runs the RDMA MW-bind benchmark (ref_rdma_ibv_wr_bind.cpp).
The program uses ibv_wr_bind_mw / ibv_wr_local_inv with GPU-registered memory
(GPUDirect RDMA via nvidia-peermem).

Execution model: two processes on the same host share one IB device.
  Server: ./rdma_ibv_wr_bind          (listens on TCP 12345, grants/revokes MW)
  Client: ./rdma_ibv_wr_bind 127.0.0.1 (connects, writes, verifies block)

Requires: g++, libibverbs, libcudart (nvidia-peermem for GPU memory)

Usage as module:
    from build_and_run import build, run, build_and_run, compare

Usage as script:
    python build_and_run.py --source ref_rdma_ibv_wr_bind.cpp
    python build_and_run.py --compare ref_rdma_ibv_wr_bind.cpp generated.cpp
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
        run_ok   = self.run_result   is None or self.run_result.success
        return build_ok and run_ok


# ============================================================================
# Helpers
# ============================================================================

def get_module_dir() -> str:
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
    if not records:
        return {}
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, (int, float)):
                sums[k]   = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
    return {k: round(sums[k] / counts[k], 6) for k in sums}


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str) -> None:
    if not metrics:
        return
    all_keys = list(dict.fromkeys(k for m in metrics for k in m.keys()))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(metrics)


def _plot_comparison(
    ref_metrics: List[Dict[str, Any]],
    gen_metrics: List[Dict[str, Any]],
    ref_label: str, gen_label: str, results_dir: str,
    latency_key: str = "latency_avg",
    latency_unit: str = "us",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed – skipping plots", file=sys.stderr)
        return

    os.makedirs(results_dir, exist_ok=True)

    ref_lat = [m[latency_key] for m in ref_metrics if latency_key in m]
    gen_lat = [m[latency_key] for m in gen_metrics if latency_key in m]
    ref_x   = list(range(1, len(ref_lat) + 1))
    gen_x   = list(range(1, len(gen_lat) + 1))

    # Latency comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    if ref_x and ref_lat:
        ax.bar([x - 0.2 for x in ref_x], ref_lat, width=0.4, label=ref_label)
    if gen_x and gen_lat:
        ax.bar([x + 0.2 for x in gen_x], gen_lat, width=0.4, label=gen_label)
    ax.set_xlabel("Run index")
    ax.set_ylabel(f"Latency ({latency_unit})")
    ax.set_title(f"Example 27 (ibv_wr_bind_mw) – write_latency per run")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5, axis="y")
    fig.tight_layout()
    lat_path = os.path.join(results_dir, "latency_comparison.png")
    fig.savefig(lat_path, dpi=150)
    plt.close(fig)
    print(f"Saved {lat_path}")

    # Throughput placeholder (MW bind has no meaningful throughput curve,
    # so we plot latency again with a different title for tooling compatibility)
    fig, ax = plt.subplots(figsize=(8, 5))
    if ref_x and ref_lat:
        ax.plot(ref_x, ref_lat, marker="o", label=ref_label, linewidth=2)
    if gen_x and gen_lat:
        ax.plot(gen_x, gen_lat, marker="s", label=gen_label, linewidth=2)
    ax.set_xlabel("Run index")
    ax.set_ylabel(f"Latency ({latency_unit})")
    ax.set_title("Example 27 – bind/revoke round-trip latency")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    thr_path = os.path.join(results_dir, "throughput_comparison.png")
    fig.savefig(thr_path, dpi=150)
    plt.close(fig)
    print(f"Saved {thr_path}")


def check_compiler(compiler: str = "g++", verbose: bool = False) -> tuple:
    try:
        result = subprocess.run([compiler, "--version"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            if verbose:
                print(f"{compiler}: found")
            return True, f"{compiler} found"
        return False, f"{compiler} returned non-zero"
    except FileNotFoundError:
        return False, f"{compiler} not found in PATH"


def check_ibverbs(verbose: bool = False) -> tuple:
    header_paths = [
        "/usr/include/infiniband/verbs.h",
        "/usr/local/include/infiniband/verbs.h",
    ]
    if not any(os.path.exists(p) for p in header_paths):
        return False, "infiniband/verbs.h not found"
    if verbose:
        print("libibverbs: found")
    return True, "libibverbs found"


def list_rdma_devices() -> List[str]:
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
    """Compile an RDMA ibv_wr_bind source file.

    Args:
        source_file: Path to the source file (.cpp)
        output_file: Path to the output executable
        compiler:    C++ compiler (default: g++)
        platform:    Accepted for API compatibility; not used
        debug:       Enable debug build (-g)
        arch:        Accepted for API compatibility; not used
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
        compiler = "g++"

    ok, msg = check_compiler(compiler, verbose)
    if not ok:
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file,
            return_code=-1, stdout="", stderr=msg, command=[],
            error_message=msg,
        )

    ok, msg = check_ibverbs(verbose)
    if not ok:
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file,
            return_code=-1, stdout="", stderr=msg, command=[],
            error_message=msg,
        )

    flags = ["-g"] if debug else ["-O2"]
    flags += ["-std=c++17", "-Wall"]

    out_path = output_file if os.path.isabs(output_file) else os.path.join(wd, output_file)
    cmd = [compiler] + flags + [src_path, "-o", out_path, "-libverbs", "-pthread", "-lcudart"]

    if verbose:
        print("===================================")
        print("Building RDMA ibv_wr_bind Test (g++ + libibverbs + lcudart)")
        print("===================================")
        print(f"Source:  {source_file}")
        print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    success = result.returncode == 0

    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
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
    server_addr: str = "127.0.0.1",
    server_gpu: int = 0,
    client_gpu: int = 0,
    timeout_sec: int = 60,
) -> RunResult:
    """Run the ibv_wr_bind benchmark (server + client on the same host).

    Starts the server process in the background, waits briefly for it to
    bind the TCP port, then launches the client.  Both processes share the
    same IB device; the client connects to server_addr on TCP 12345.

    Args:
        executable:  Path to the compiled executable
        verbose:     Print run progress and diagnostics
        server_addr: IP address the client connects to (default: 127.0.0.1)
        server_gpu:  CUDA device index for the server (default: 0)
        client_gpu:  CUDA device index for the client (default: 0)
        timeout_sec: Per-process timeout in seconds (default: 60)
    """
    wd = get_module_dir()
    exe_path = executable if os.path.isabs(executable) else os.path.join(wd, executable)

    if not os.path.exists(exe_path):
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=[],
            error_message=f"Executable '{executable}' not found",
        )

    server_cmd = [exe_path]
    client_cmd = [exe_path, server_addr]

    server_env = os.environ.copy()
    server_env["SERVER_GPU"] = str(server_gpu)

    client_env = os.environ.copy()
    client_env["CLIENT_GPU"] = str(client_gpu)

    if verbose:
        print(f"Server: SERVER_GPU={server_gpu} {' '.join(server_cmd)}")
        print(f"Client: CLIENT_GPU={client_gpu} {' '.join(client_cmd)}")

    server_proc = subprocess.Popen(
        server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=server_env,
    )

    # Give the server time to bind the TCP port
    time.sleep(1)

    try:
        client_proc = subprocess.Popen(
            client_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=client_env,
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
            stdout="", stderr=str(e), command=client_cmd, error_message=str(e),
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
        # Don't echo combined_stdout to our own stdout — that would mix
        # the wrapper's diagnostic noise with the canonical JSON. Send the
        # raw [server]/[client] dump to stderr so the caller can still see
        # it without polluting stdout.
        if combined_stdout.strip():
            print(combined_stdout, file=sys.stderr)
        if combined_stderr:
            print(combined_stderr, file=sys.stderr)

    server_ok = server_proc.returncode == 0
    client_ok  = client_proc.returncode == 0
    success    = server_ok and client_ok

    # Parse metrics from server JSON (it carries the latency measurement)
    server_parsed = _parse_json_output(server_stdout)
    client_parsed = _parse_json_output(client_stdout)
    # The reference binary now routes the server-side status to stderr so
    # the canonical "exactly one JSON object on stdout" comes from the
    # client only. Fall back to the server's stderr to keep the
    # correctness check working.
    if not server_parsed:
        server_parsed = _parse_json_output(server_stderr)

    # Prefer server metrics; merge client latency if server has none
    parsed: Dict[str, Any] = {}
    if server_parsed:
        parsed.update(server_parsed)
        metrics = server_parsed.get("metrics", [])
        if metrics:
            parsed["latency_avg"] = metrics[0].get("latency_avg", 0.0)
    if client_parsed and "latency_avg" not in parsed:
        metrics = client_parsed.get("metrics", [])
        if metrics:
            parsed["latency_avg"] = metrics[0].get("latency_avg", 0.0)

    # Require PASS in both outputs
    if success:
        server_pass = server_parsed.get("Correctness", "") == "PASS"
        client_pass = client_parsed.get("Correctness", "") == "PASS"
        success = server_pass and client_pass

    rc = 0 if success else (client_proc.returncode or server_proc.returncode or 1)
    return RunResult(
        success=success, executable=executable, return_code=rc,
        stdout=combined_stdout, stderr=combined_stderr,
        command=client_cmd, parsed_output=parsed,
        error_message=None if success else "Execution failed or correctness FAIL",
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
    server_addr: str = "127.0.0.1",
    server_gpu: int = 0,
    client_gpu: int = 0,
) -> BuildAndRunResult:
    """Build and run the RDMA ibv_wr_bind benchmark."""
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
            print("===================================")
            print("Running RDMA ibv_wr_bind Test")
            print("===================================")
        run_result = run(
            executable=output_file, verbose=verbose,
            server_addr=server_addr,
            server_gpu=server_gpu, client_gpu=client_gpu,
        )

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


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
    server_addr: str = "127.0.0.1",
    server_gpu: int = 0,
    client_gpu: int = 0,
    no_plot: bool = False,
    verbose: bool = True,
    show_raw_output: bool = False,
) -> Dict[str, Any]:
    """Build & run ref and generated, compare latency, save CSV/plots/summary."""
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]

    build_status: Dict[str, bool] = {}
    run_status:   Dict[str, bool] = {}
    all_results:  Dict[str, List[Dict[str, Any]]] = {ref_label: [], gen_label: []}

    for label, src, exe_name in [
        (ref_label, src_ref, "ref_exec"),
        (gen_label, src_gen, "gen_exec"),
    ]:
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

        if verbose:
            print(f"[compare] Running: {src}")

        rr = run(
            executable=exe_name, verbose=show_raw_output,
            server_addr=server_addr,
            server_gpu=server_gpu, client_gpu=client_gpu,
        )
        run_status[label] = rr.success
        if rr.success and rr.parsed_output:
            all_results[label].append(rr.parsed_output)

        time.sleep(0.5)

    ref_metrics = all_results[ref_label]
    gen_metrics = all_results[gen_label]

    # Save CSVs
    for label, pts in all_results.items():
        if pts:
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            _save_metrics_csv(pts, csv_path)
            if verbose:
                print(f"Raw metrics saved to {csv_path}")

    has_perf = bool(ref_metrics) and bool(gen_metrics)

    ref_avg = _average_metrics(ref_metrics)
    gen_avg = _average_metrics(gen_metrics)

    lat_key = "latency_avg"
    lat_imp = 0.0
    if has_perf and ref_avg.get(lat_key, 0) != 0:
        lat_imp = (ref_avg[lat_key] - gen_avg.get(lat_key, 0)) / ref_avg[lat_key] * 100

    if abs(lat_imp) < 5:
        performance = "same"
    elif lat_imp > 0:
        performance = "better"
    else:
        performance = "worse"

    summary: Dict[str, Any] = {
        "generated_source":       os.path.basename(src_gen),
        "ref_source":             os.path.basename(src_ref),
        "model":                  "unknown",
        "pass_iteration":         1,
        "improvement_iteration":  1,
        "data_size_unit":         "Bytes",
        "latency_unit":           "us",
        "throughput_unit":        "Gbps",
        "metrics_comparison": {
            "ref": {
                "compile_success": build_status.get(ref_label, False),
                "run_success":     run_status.get(ref_label, False),
            },
            "generated": {
                "compile_success": build_status.get(gen_label, False),
                "run_success":     run_status.get(gen_label, False),
            },
        },
        "latency_improvement_pct":    round(lat_imp, 2),
        "throughput_improvement_pct": 0.0,
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
        _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label, results_dir)

    # Required printed comparison summary
    last_gen = gen_metrics[-1] if gen_metrics else {}
    print(f"\nPERFORMANCE COMPARISON (ref vs generated)\n")
    print(f"[+] latency_avg: {last_gen.get('latency_avg', 'N/A')} us")
    print(f"[+] latency_improvement_pct: {lat_imp:.2f}%")
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
        description="Build and run the RDMA ibv_wr_bind_mw benchmark"
    )
    parser.add_argument("--source", "-s", default="ref_rdma_ibv_wr_bind.cpp",
                        help="Source file to compile")
    parser.add_argument("--output", "-o", default="rdma_ibv_wr_bind",
                        help="Output executable name")
    parser.add_argument("--arch", default=None,
                        help="GPU architecture hint (unused; kept for API compatibility)")
    parser.add_argument("--build-only", action="store_true",
                        help="Compile only, do not run")
    parser.add_argument("--run-only", action="store_true",
                        help="Run existing executable without compiling")
    parser.add_argument("--compiler", default=None,
                        help="C++ compiler path (default: g++)")
    parser.add_argument("--platform", default=None,
                        choices=["ib", "efa"],
                        help="RDMA NIC platform hint (unused; kept for API compatibility)")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Generate performance plots after running")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--results-dir", default="results",
                        help="Directory for saving CSV, plots, and summary JSON")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "GEN"),
                        help="Compare mode: --compare ref.cpp generated.cpp")
    parser.add_argument("--debug", action="store_true", help="Enable debug build")
    parser.add_argument("--server-addr", default="127.0.0.1",
                        help="Server IP address for client to connect to (default: 127.0.0.1)")
    parser.add_argument("--server-gpu", type=int, default=0,
                        help="CUDA device index for server (default: 0)")
    parser.add_argument("--client-gpu", type=int, default=0,
                        help="CUDA device index for client (default: 0)")
    parser.add_argument("--show-raw-output", action="store_true",
                        help="Print raw stdout/stderr from program execution")
    parser.add_argument("--list-devices", action="store_true",
                        help="List RDMA devices and exit")
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
        src_ref, src_gen = args.compare
        for src in (src_ref, src_gen):
            path = src if os.path.isabs(src) else os.path.join(wd, src)
            if not os.path.exists(path):
                print(f"Error: Source file '{src}' not found!")
                sys.exit(1)
        compare(
            src_ref=src_ref, src_gen=src_gen,
            results_dir=results_dir,
            compiler=args.compiler, platform=args.platform,
            debug=args.debug, arch=args.arch,
            server_addr=args.server_addr,
            server_gpu=args.server_gpu, client_gpu=args.client_gpu,
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
        server_addr=args.server_addr,
        server_gpu=args.server_gpu,
        client_gpu=args.client_gpu,
    )

    br = result.build_result
    rr = result.run_result

    if br is not None:
        print("Build successful" if br.success else "Compilation failed")
        if not br.success and br.stderr:
            print(br.stderr, file=sys.stderr)

    if rr is not None:
        # Forward the canonical JSON (the client's) to our own stdout so
        # the wrapper presents exactly one JSON object on stdout. Status
        # text and raw [server]/[client] dumps go to stderr.
        sys.stderr.write("PASS\n" if rr.success else "FAIL\n")
        if rr.parsed_output:
            json.dump(rr.parsed_output, sys.stdout, indent=2)
            sys.stdout.write("\n")
        elif rr.stdout:
            sys.stdout.write(rr.stdout)
            if not rr.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if not rr.success and rr.stderr:
            print(rr.stderr, file=sys.stderr)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
