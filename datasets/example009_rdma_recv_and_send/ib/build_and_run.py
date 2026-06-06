#!/usr/bin/env python3
"""
RDMA RC recv/send (pingpong-style) Build & Run Harness

Meets repo requirements:
- Single File Mode: build/run one source
- Compare Mode: build/run two sources, save CSV + plots + summary.json
- Do NOT print raw stdout by default
- Save all outputs under --results-dir
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

# ---------------- Data models ----------------

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
    role: str
    executable: str
    return_code: int
    stdout: str
    stderr: str
    command: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class BuildAndRunResult:
    build_result: Optional[BuildResult]
    run_result: Optional[RunResult]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        build_ok = (self.build_result is None) or self.build_result.success
        run_ok = (self.run_result is None) or self.run_result.success
        return build_ok and run_ok


# ---------------- Metrics parsing ----------------

_BW_RE = re.compile(r"Bandwidth:\s*([0-9]*\.?[0-9]+)\s*([A-Za-z/]+)")
_LAT_RE = re.compile(r"Round-trip latency:\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)")
_PASS_RE = re.compile(r"\bPASSED\b")
_FAIL_RE = re.compile(r"\bFAILED\b")

def _extract_metrics(text: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    bw_all = _BW_RE.findall(text or "")
    lat_all = _LAT_RE.findall(text or "")

    if bw_all:
        val, unit = bw_all[-1]
        try:
            m["throughput_value"] = float(val)
            m["throughput_unit"] = unit
        except ValueError:
            pass

    if lat_all:
        val, unit = lat_all[-1]
        try:
            m["latency_value"] = float(val)
            m["latency_unit"] = unit
        except ValueError:
            pass

    m["passed"] = bool(_PASS_RE.search(text or ""))
    m["failed"] = bool(_FAIL_RE.search(text or ""))
    return m


# ---------------- Utilities ----------------

def get_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_text(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def safe_stem(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def normalize_units(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize to:
      latency_unit = "us"
      throughput_unit = "Gbps"
    Based on typical output patterns you used earlier (usec, Gbit/s).
    """
    out = dict(metrics)

    # Latency
    lat = out.get("latency_value")
    lat_u = (out.get("latency_unit") or "").lower()
    if lat is not None:
        if lat_u in ("usec", "us", "µs"):
            out["latency_avg"] = float(lat)
            out["latency_unit_norm"] = "us"
        elif lat_u in ("msec", "ms"):
            out["latency_avg"] = float(lat) * 1000.0
            out["latency_unit_norm"] = "us"
        elif lat_u in ("sec", "s"):
            out["latency_avg"] = float(lat) * 1_000_000.0
            out["latency_unit_norm"] = "us"
        else:
            # unknown, pass through
            out["latency_avg"] = float(lat)
            out["latency_unit_norm"] = out.get("latency_unit")

    # Throughput
    thr = out.get("throughput_value")
    thr_u = (out.get("throughput_unit") or "")
    if thr is not None:
        u = thr_u.strip().lower()
        # common: "Gbit/s", "Gbps"
        if u in ("gbit/s", "gbps"):
            out["throughput"] = float(thr)
            out["throughput_unit_norm"] = "Gbps"
        elif u in ("mbit/s", "mbps"):
            out["throughput"] = float(thr) / 1000.0
            out["throughput_unit_norm"] = "Gbps"
        elif u in ("kbit/s", "kbps"):
            out["throughput"] = float(thr) / 1_000_000.0
            out["throughput_unit_norm"] = "Gbps"
        else:
            out["throughput"] = float(thr)
            out["throughput_unit_norm"] = out.get("throughput_unit")

    return out


# ---------------- Build ----------------
# REQUIRED SIGNATURE:
# def build(source_file, output_file, compiler, platform, debug=False, arch=None, verbose=True) -> BuildResult

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
    Build the C++ RDMA tool with libibverbs.
    platform/arch are accepted for signature compliance but not used for RDMA.
    """
    wd = get_module_dir()
    if compiler is None:
        compiler = "g++"

    cxx_flags = ["-std=c++17", "-Wall"]
    if debug:
        cxx_flags += ["-O0", "-g"]
    else:
        cxx_flags += ["-O2"]

    cmd = [compiler] + cxx_flags + [source_file, "-o", output_file, "-libverbs"]

    if not os.path.exists(os.path.join(wd, source_file)):
        return BuildResult(
            success=False,
            source_file=source_file,
            output_file=output_file,
            return_code=-1,
            stdout="",
            stderr="",
            command=cmd,
            error_message=f"Source file not found: {source_file}",
        )

    p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True)
    ok = (p.returncode == 0)

    # By default, do not print raw stdout/stderr. Caller will save them.
    if verbose:
        print(f"[build] {'OK' if ok else 'FAIL'}: {source_file} -> {output_file}")

    return BuildResult(
        success=ok,
        source_file=source_file,
        output_file=output_file,
        return_code=p.returncode,
        stdout=p.stdout or "",
        stderr=p.stderr or "",
        command=cmd,
        error_message=None if ok else "Compilation failed",
    )


# ---------------- Run ----------------
# REQUIRED SIGNATURE:
# def run(executable, verbose=True) -> RunResult

def _start_process_group(cmd: List[str], *, cwd: str) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )

def _kill_group(p: subprocess.Popen, *, grace_s: float = 0.3) -> None:
    try:
        pgid = os.getpgid(p.pid)
    except Exception:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(grace_s)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass

def _run_pair(
    executable: str,
    server_dev: str,
    client_dev: str,
    server_ip: str,
    msg_size: int,
    server_start_delay_s: float,
    server_timeout_s: int,
    client_timeout_s: int,
) -> Tuple[RunResult, RunResult]:
    wd = get_module_dir()
    exe_path = os.path.join(wd, executable)
    if not os.path.exists(exe_path):
        rr = RunResult(
            success=False,
            role="client",
            executable=executable,
            return_code=-1,
            stdout="",
            stderr="",
            command=[],
            metrics={"passed": False},
            error_message=f"Executable not found: {executable}",
        )
        return rr, rr

    server_cmd = [f"./{executable}", server_dev]
    client_cmd = [f"./{executable}", client_dev, server_ip]
    if msg_size != 4096:
        server_cmd += ["--msg-size", str(msg_size)]
        client_cmd += ["--msg-size", str(msg_size)]

    srv = _start_process_group(server_cmd, cwd=wd)
    time.sleep(server_start_delay_s)

    # Run client
    try:
        cp = subprocess.run(
            client_cmd,
            cwd=wd,
            capture_output=True,
            text=True,
            timeout=client_timeout_s,
        )
        out = cp.stdout or ""
        err = cp.stderr or ""
        metrics = _extract_metrics(out + "\n" + err)
        ok = (cp.returncode == 0) and metrics.get("passed", False)
        client_res = RunResult(
            success=ok,
            role="client",
            executable=executable,
            return_code=cp.returncode,
            stdout=out,
            stderr=err,
            command=client_cmd,
            metrics=metrics,
            error_message=None if ok else "Client failed",
        )
    except subprocess.TimeoutExpired:
        client_res = RunResult(
            success=False,
            role="client",
            executable=executable,
            return_code=-1,
            stdout="",
            stderr=f"Timeout after {client_timeout_s} seconds",
            command=client_cmd,
            metrics={"passed": False},
            error_message="Client timed out",
        )

    # Wait server
    try:
        so, se = srv.communicate(timeout=server_timeout_s)
        srv_rc = srv.returncode
        srv_timed = False
    except subprocess.TimeoutExpired:
        so, se = "", ""
        srv_rc = -1
        srv_timed = True
        _kill_group(srv)

    so = so or ""
    se = se or ""
    sm = _extract_metrics(so + "\n" + se)
    srv_ok = (srv_rc == 0) and sm.get("passed", False)
    server_res = RunResult(
        success=srv_ok,
        role="server",
        executable=executable,
        return_code=srv_rc,
        stdout=so,
        stderr=se,
        command=server_cmd,
        metrics=sm,
        error_message=None if srv_ok else ("Server timeout" if srv_timed else "Server failed"),
    )

    return server_res, client_res


def _list_rdma_devices() -> List[str]:
    """Return RDMA device names visible to ibverbs (e.g. ['mlx5_0'])."""
    try:
        r = subprocess.run(["ibv_devices"], capture_output=True, text=True, timeout=5)
        names = []
        for ln in r.stdout.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith(("device", "----")):
                continue
            names.append(ln.split()[0])
        return names
    except Exception:
        return []


def _autoselect_dev(role: str) -> str:
    """Pick a sensible default RDMA device for `role` ('server' or 'client').

    With multiple devices we use server=devs[0], client=devs[1].
    With a single device (the common single-NIC test box) both endpoints
    share that device for loopback.
    """
    devs = _list_rdma_devices()
    if not devs:
        return "ionic_0" if role == "server" else "ionic_1"
    if role == "server":
        return devs[0]
    return devs[1] if len(devs) > 1 else devs[0]


def run(executable: str, verbose: bool = True) -> RunResult:
    """
    Required signature by README.
    NOTE: We allow configuration through env vars to keep signature fixed.
    """
    server_dev = os.environ.get("RDMA_SERVER_DEV", _autoselect_dev("server"))
    client_dev = os.environ.get("RDMA_CLIENT_DEV", _autoselect_dev("client"))
    server_ip  = os.environ.get("RDMA_SERVER_IP", "127.0.0.1")
    msg_size   = int(os.environ.get("RDMA_MSG_SIZE", "4096"))

    server_start_delay_s = float(os.environ.get("RDMA_SERVER_DELAY_S", "0.5"))
    server_timeout_s = int(os.environ.get("RDMA_SERVER_TIMEOUT_S", "120"))
    client_timeout_s = int(os.environ.get("RDMA_CLIENT_TIMEOUT_S", "120"))

    if verbose:
        print(f"[run] exe={executable} server_dev={server_dev} client_dev={client_dev} ip={server_ip} msg_size={msg_size}")

    _, cli = _run_pair(
        executable=executable,
        server_dev=server_dev,
        client_dev=client_dev,
        server_ip=server_ip,
        msg_size=msg_size,
        server_start_delay_s=server_start_delay_s,
        server_timeout_s=server_timeout_s,
        client_timeout_s=client_timeout_s,
    )
    return cli


# ---------------- Build+Run ----------------
# REQUIRED SIGNATURE: def build_and_run(...) -> BuildAndRunResult

def build_and_run(
    source_file: str,
    output_file: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    verbose: bool = True,
    build_only: bool = False,
    run_only: bool = False,
) -> BuildAndRunResult:
    b: Optional[BuildResult] = None
    r: Optional[RunResult] = None

    if not run_only:
        b = build(
            source_file=source_file,
            output_file=output_file,
            compiler=compiler,
            platform=platform,
            debug=debug,
            arch=arch,
            verbose=verbose,
        )
        if not b.success:
            return BuildAndRunResult(build_result=b, run_result=None, performance_metrics={})

    if not build_only:
        r = run(output_file, verbose=verbose)
        perf = normalize_units(r.metrics if r else {})
    else:
        perf = {}

    return BuildAndRunResult(build_result=b, run_result=r, performance_metrics=perf)


# ---------------- Compare ----------------
# REQUIRED SIGNATURE: def compare(...) -> Dict[str, Any]

def _write_metrics_csv(path: str, *, metrics: Dict[str, Any], source: str) -> None:
    # Minimal required fields + whatever else is present
    row = {
        "source": source,
        "passed": bool(metrics.get("passed", False)),
        "latency_avg": metrics.get("latency_avg"),
        "latency_unit": metrics.get("latency_unit_norm", metrics.get("latency_unit")),
        "throughput": metrics.get("throughput"),
        "throughput_unit": metrics.get("throughput_unit_norm", metrics.get("throughput_unit")),
    }
    # include raw parsed too
    for k, v in metrics.items():
        if k not in row:
            row[k] = v

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

def _plot_two_bars(path: str, title: str, a_label: str, a_val: float, b_label: str, b_val: float, y_label: str) -> None:
    # Spec requires plots; keep dependencies minimal: matplotlib
    import matplotlib.pyplot as plt

    labels = [a_label, b_label]
    vals = [a_val, b_val]

    plt.figure()
    plt.bar(labels, vals)
    plt.title(title)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str = "results",
    verbose: bool = True,
    server_dev: str = "",
    client_dev: str = "",
    server_ip: str = "127.0.0.1",
    msg_size: int = 4096,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare mode:
    - build+run both
    - save all outputs under results_dir
    - write CSV per run
    - write two plots
    - write summary.json
    - print required PERFORMANCE COMPARISON block
    """
    ensure_dir(results_dir)

    # Configure run() via env while keeping signature fixed
    os.environ["RDMA_SERVER_DEV"] = server_dev or _autoselect_dev("server")
    os.environ["RDMA_CLIENT_DEV"] = client_dev or _autoselect_dev("client")
    os.environ["RDMA_SERVER_IP"] = server_ip
    os.environ["RDMA_MSG_SIZE"] = str(msg_size)

    ref_exe = "rdma_recv_send_ref"
    gen_exe = "rdma_recv_send_gen"

    ref = build_and_run(
        source_file=src_ref,
        output_file=ref_exe,
        compiler=compiler,
        platform=platform,
        debug=debug,
        arch=arch,
        verbose=verbose,
    )
    gen = build_and_run(
        source_file=src_gen,
        output_file=gen_exe,
        compiler=compiler,
        platform=platform,
        debug=debug,
        arch=arch,
        verbose=verbose,
    )

    # Save raw outputs (logs) under results_dir (required: save outputs there)
    # build logs
    if ref.build_result:
        write_text(os.path.join(results_dir, f"ref_{safe_stem(src_ref)}_build_stdout.txt"), ref.build_result.stdout)
        write_text(os.path.join(results_dir, f"ref_{safe_stem(src_ref)}_build_stderr.txt"), ref.build_result.stderr)
    if gen.build_result:
        write_text(os.path.join(results_dir, f"generated_{safe_stem(src_gen)}_build_stdout.txt"), gen.build_result.stdout)
        write_text(os.path.join(results_dir, f"generated_{safe_stem(src_gen)}_build_stderr.txt"), gen.build_result.stderr)

    # run logs
    if ref.run_result:
        write_text(os.path.join(results_dir, f"ref_{safe_stem(src_ref)}_run_stdout.txt"), ref.run_result.stdout)
        write_text(os.path.join(results_dir, f"ref_{safe_stem(src_ref)}_run_stderr.txt"), ref.run_result.stderr)
    if gen.run_result:
        write_text(os.path.join(results_dir, f"generated_{safe_stem(src_gen)}_run_stdout.txt"), gen.run_result.stdout)
        write_text(os.path.join(results_dir, f"generated_{safe_stem(src_gen)}_run_stderr.txt"), gen.run_result.stderr)

    ref_ok = bool(ref.performance_metrics.get("passed", False)) and (ref.run_result is not None and ref.run_result.success)
    gen_ok = bool(gen.performance_metrics.get("passed", False)) and (gen.run_result is not None and gen.run_result.success)

    # CSV metrics (required)
    ref_csv = os.path.join(results_dir, f"ref_{safe_stem(src_ref)}_metrics.csv")
    gen_csv = os.path.join(results_dir, f"generated_{safe_stem(src_gen)}_metrics.csv")
    _write_metrics_csv(ref_csv, metrics=ref.performance_metrics, source=src_ref)
    _write_metrics_csv(gen_csv, metrics=gen.performance_metrics, source=src_gen)

    # Plots (required)
    # If missing metrics, plot 0.0 to keep pipeline consistent.
    ref_lat = float(ref.performance_metrics.get("latency_avg") or 0.0)
    gen_lat = float(gen.performance_metrics.get("latency_avg") or 0.0)
    ref_thr = float(ref.performance_metrics.get("throughput") or 0.0)
    gen_thr = float(gen.performance_metrics.get("throughput") or 0.0)

    latency_png = os.path.join(results_dir, "latency_comparison.png")
    throughput_png = os.path.join(results_dir, "throughput_comparison.png")

    _plot_two_bars(
        latency_png,
        "Latency comparison",
        "ref",
        ref_lat,
        "generated",
        gen_lat,
        "latency (us)",
    )
    _plot_two_bars(
        throughput_png,
        "Throughput comparison",
        "ref",
        ref_thr,
        "generated",
        gen_thr,
        "throughput (Gbps)",
    )

    # Summary JSON (required format)
    summary: Dict[str, Any] = {
        "generated_source": src_gen,
        "ref_source": src_ref,
        "model": "",
        "pass_iteration": 1,
        "improvement_iteration": 1,
        "data_size_unit": "bytes",
        "latency_unit": "us",
        "throughput_unit": "Gbps",
        "metrics_comparison": {
            "ref": {
                "compile_success": bool(ref.build_result and ref.build_result.success),
                "run_success": bool(ref.run_result and ref.run_result.success),
            },
            "generated": {
                "compile_success": bool(gen.build_result and gen.build_result.success),
                "run_success": bool(gen.run_result and gen.run_result.success),
            },
        },
        "latency_improvement_pct": 0.0,
        "throughput_improvement_pct": 0.0,
        "performance": "same",
    }

    if ref_ok and gen_ok and ref_lat > 0 and ref_thr > 0:
        summary["latency_improvement_pct"] = ((ref_lat - gen_lat) / ref_lat) * 100.0 if gen_lat > 0 else 0.0
        summary["throughput_improvement_pct"] = ((gen_thr - ref_thr) / ref_thr) * 100.0 if gen_thr > 0 else 0.0

        # simple label
        if gen_lat < ref_lat and gen_thr > ref_thr:
            summary["performance"] = "better"
        elif gen_lat > ref_lat and gen_thr < ref_thr:
            summary["performance"] = "worse"
        else:
            summary["performance"] = "same"

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Required printed comparison summary
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"[+] data_size_avg:        {msg_size} bytes")
    print(f"[+] throughput:           ref={ref_thr:.6g} Gbps, generated={gen_thr:.6g} Gbps")
    print(f"[+] latency_avg:          ref={ref_lat:.6g} us, generated={gen_lat:.6g} us")
    print(f"Performance: {summary['performance']}")
    print("\n" + "=" * 60)
    print(f"ref       compile_success: {summary['metrics_comparison']['ref']['compile_success']}")
    print(f"ref       run_success:     {summary['metrics_comparison']['ref']['run_success']}")
    print(f"generated compile_success: {summary['metrics_comparison']['generated']['compile_success']}")
    print(f"generated run_success:     {summary['metrics_comparison']['generated']['run_success']}")
    print(f"performance:               {summary['performance']}")
    print("=" * 60)
    print(f"Saved: {results_dir}")

    return summary


# ---------------- CLI ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build & run RDMA recv/send harness")

    ap.add_argument("--source", "-s", default="ref_rdma_recv_send.cpp", help="Source file to compile")
    ap.add_argument("--output", "-o", default="rdma_recv_send", help="Output executable name")
    ap.add_argument("--arch", "-a", default=None, help="Arch (unused for RDMA; for API compat)")
    ap.add_argument("--compiler", default=None, help="Compiler (default: g++)")
    ap.add_argument("--platform", default=None, help="Platform (unused for RDMA; for API compat)")
    ap.add_argument("--build-only", action="store_true", help="Only build")
    ap.add_argument("--run-only", action="store_true", help="Only run (assumes executable exists)")
    ap.add_argument("--debug", action="store_true", help="Build with debug flags")

    # RDMA args (used by compare; also export to env for run())
    ap.add_argument("--server-dev", default="",
                    help="Server RDMA device name (default: auto-detect via ibv_devices)")
    ap.add_argument("--client-dev", default="",
                    help="Client RDMA device name (default: auto-detect via ibv_devices)")
    ap.add_argument("--server-ip", default="127.0.0.1", help="Server IP for client to connect to")
    ap.add_argument("--msg-size", type=int, default=4096, help="Message size in bytes")

    # Compare mode + results dir
    ap.add_argument("--compare", nargs=2, metavar=("SRC_A", "SRC_B"),
                    help="Build both sources and compare extracted metrics")
    ap.add_argument("--results-dir", "-r", default="results", help="Directory for results")

    # plotting flag exists for API but we always create required plots in compare mode
    ap.add_argument("--plot", action="store_true", help="Generate plots (compare mode always does)")
    ap.add_argument("--quiet", action="store_true", help="Less printing (still prints required compare summary)")

    ap.add_argument(
        "--legacy-perf-verdict",
        action="store_true",
        help="Use this example's local verdict logic instead of the shared "
             "4-tier scheme in run_eval/perf_verdict.py."
    )
    args = ap.parse_args()

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

    if args.compare:
        compare(
            src_ref=args.compare[0],
            src_gen=args.compare[1],
            results_dir=args.results_dir,
            verbose=verbose,
            server_dev=args.server_dev,
            client_dev=args.client_dev,
            server_ip=args.server_ip,
            msg_size=args.msg_size,
            compiler=args.compiler,
            platform=args.platform,
            debug=args.debug,
            arch=args.arch,
        )
        sys.exit(0)

    # single-file mode: must compile and run a single source file
    # Save outputs under --results-dir
    ensure_dir(args.results_dir)

    # configure run() via env
    os.environ["RDMA_SERVER_DEV"] = args.server_dev or _autoselect_dev("server")
    os.environ["RDMA_CLIENT_DEV"] = args.client_dev or _autoselect_dev("client")
    os.environ["RDMA_SERVER_IP"] = args.server_ip
    os.environ["RDMA_MSG_SIZE"] = str(args.msg_size)

    r = build_and_run(
        source_file=args.source,
        output_file=args.output,
        compiler=args.compiler,
        platform=args.platform,
        debug=args.debug,
        arch=args.arch,
        verbose=verbose,
        build_only=args.build_only,
        run_only=args.run_only,
    )

    # Save build/run outputs + metrics CSV
    if r.build_result:
        write_text(os.path.join(args.results_dir, f"single_{safe_stem(args.source)}_build_stdout.txt"), r.build_result.stdout)
        write_text(os.path.join(args.results_dir, f"single_{safe_stem(args.source)}_build_stderr.txt"), r.build_result.stderr)
    if r.run_result:
        write_text(os.path.join(args.results_dir, f"single_{safe_stem(args.source)}_run_stdout.txt"), r.run_result.stdout)
        write_text(os.path.join(args.results_dir, f"single_{safe_stem(args.source)}_run_stderr.txt"), r.run_result.stderr)

    # metrics csv (single file mode)
    single_csv = os.path.join(args.results_dir, f"{safe_stem(args.source)}_metrics.csv")
    _write_metrics_csv(single_csv, metrics=r.performance_metrics, source=args.source)

    # Optional: you can add plotting for single-file mode if you want (spec says optional)
    if verbose:
        print(f"[single] saved results under: {args.results_dir}")

    # Forward the reference binary's stdout (a single JSON object per spec)
    # to our stdout so downstream tooling can parse it directly.
    if r.run_result and r.run_result.stdout:
        sys.stdout.write(r.run_result.stdout)
        if not r.run_result.stdout.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    sys.exit(0 if r.success else 1)


if __name__ == "__main__":
    main()
