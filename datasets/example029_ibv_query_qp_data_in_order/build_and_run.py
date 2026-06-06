#!/usr/bin/env python3
"""
RDMA QP Data In Order Query Build and Run Module

Compiles and runs a single-process program that queries data ordering guarantees
for different QP transport types and RDMA opcodes using ibv_query_qp_data_in_order.

No server/client processes needed — this is a purely local NIC query.

Usage as module:
    from build_and_run import build, run, build_and_run, compare

Usage as script:
    python3 build_and_run.py --source ref_ibv_query_qp_data_in_order.cpp
    python3 build_and_run.py --compare ref_ibv_query_qp_data_in_order.cpp generated.cpp
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


def _extract_metrics(text: str) -> Dict[str, Any]:
    """Extract metrics from METRICS_JSON: line in output, or parse raw JSON."""
    if not text:
        return {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("METRICS_JSON:"):
            json_str = stripped[len("METRICS_JSON:"):].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
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


def check_ibverbs(verbose: bool = False) -> tuple:
    """Check if libibverbs is available."""
    header_paths = [
        "/usr/include/infiniband/verbs.h",
        "/usr/local/include/infiniband/verbs.h",
    ]
    header_found = any(os.path.exists(p) for p in header_paths)
    if not header_found:
        return False, "infiniband/verbs.h not found"

    # Check for versioned .so as well since the unversioned symlink may be missing
    lib_paths = [
        "/usr/lib/x86_64-linux-gnu/libibverbs.so",
        "/usr/lib/x86_64-linux-gnu/libibverbs.so.1",
        "/usr/lib64/libibverbs.so",
        "/usr/lib64/libibverbs.so.1",
        "/usr/lib/libibverbs.so",
        "/usr/lib/libibverbs.so.1",
    ]
    lib_found = any(os.path.exists(p) for p in lib_paths)
    if not lib_found:
        try:
            result = subprocess.run(["pkg-config", "--exists", "libibverbs"], capture_output=True)
            lib_found = result.returncode == 0
        except FileNotFoundError:
            pass

    if lib_found:
        if verbose:
            print("libibverbs: found")
        return True, "libibverbs found"
    return False, "libibverbs not found"


def _find_libibverbs() -> Optional[str]:
    """Find the libibverbs shared library to link against directly."""
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libibverbs.so",
        "/usr/lib/x86_64-linux-gnu/libibverbs.so.1",
        "/usr/lib64/libibverbs.so",
        "/usr/lib64/libibverbs.so.1",
        "/usr/lib/libibverbs.so",
        "/usr/lib/libibverbs.so.1",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


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
    """Build the ibv_query_qp_data_in_order program using g++."""
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

    # Link directly against the .so file since the unversioned symlink may be missing
    libibverbs_path = _find_libibverbs()
    if libibverbs_path:
        cmd = [compiler] + flags + [src_path, libibverbs_path, "-o", out_path]
    else:
        cmd = [compiler] + flags + [src_path, "-o", out_path, "-libverbs"]

    if verbose:
        print("===================================")
        print("Building ibv_query_qp_data_in_order")
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
    nic: int = 0,
    gid_index: Optional[int] = None,
    timeout_sec: int = 60,
) -> RunResult:
    """Run the ibv_query_qp_data_in_order program (single process, no server/client).

    Args:
        executable:  Path to the executable
        verbose:     Print run progress
        nic:         NIC index to use (default: 0)
        gid_index:   GID index (auto-detect if None)
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

    cmd = [exe_path, "--nic", str(nic)]
    if gid_index is not None:
        cmd += ["--gid-index", str(gid_index)]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="Timed out", command=cmd,
            error_message="Process timed out",
        )

    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    success = result.returncode == 0
    parsed = _extract_metrics(result.stdout) if success else {}

    if success and "PASS" not in result.stdout:
        success = False

    return RunResult(
        success=success, executable=executable,
        return_code=result.returncode,
        stdout=result.stdout, stderr=result.stderr,
        command=cmd, parsed_output=parsed,
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
    nic: int = 0,
    gid_index: Optional[int] = None,
) -> BuildAndRunResult:
    """Build and run the ibv_query_qp_data_in_order program."""
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
            print("Running ibv_query_qp_data_in_order")
            print("===================================")
        run_result = run(
            executable=output_file, verbose=verbose,
            nic=nic, gid_index=gid_index,
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
    nic: int = 0,
    gid_index: Optional[int] = None,
    no_plot: bool = False,
    verbose: bool = True,
    show_raw_output: bool = False,
) -> Dict[str, Any]:
    """Compare two source files: build both, run once each, compare metrics."""
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]

    all_results: Dict[str, List[Dict[str, Any]]] = {ref_label: [], gen_label: []}
    sources = {ref_label: (src_ref, "ref_exec"), gen_label: (src_gen, "gen_exec")}

    build_status: Dict[str, bool] = {}
    run_status: Dict[str, bool] = {}

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

        rr = run(
            executable=exe_name, verbose=show_raw_output,
            nic=nic, gid_index=gid_index,
        )
        run_status[label] = rr.success

        if rr.success and rr.parsed_output:
            parsed = rr.parsed_output
            metrics_list = parsed.get("metrics", [])
            if metrics_list:
                for m in metrics_list:
                    all_results[label].append(m)
            else:
                all_results[label].append(parsed)

        time.sleep(0.5)

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
        "ref_source":       os.path.basename(src_ref),
        "model":            "",
        "pass_iteration":   1,
    }

    if has_perf:
        ref_avg = _average_metrics(ref_metrics)
        gen_avg = _average_metrics(gen_metrics)

        summary["metrics_comparison"] = {
            "ref": {
                "compile_success": build_status.get(ref_label, False),
                "run_success":     run_status.get(ref_label, False),
                **ref_avg,
            },
            "generated": {
                "compile_success": build_status.get(gen_label, False),
                "run_success":     run_status.get(gen_label, False),
                **gen_avg,
            },
        }

        # For this benchmark, correctness is the primary metric
        ref_in_order = ref_avg.get("in_order", 0)
        gen_in_order = gen_avg.get("in_order", 0)
        summary["performance"] = "same" if ref_in_order == gen_in_order else "worse"

        if verbose:
            comparison = _compare_metrics(ref_avg, gen_avg)
            if comparison.get("comparison"):
                print(f"\nMETRICS COMPARISON")
                for metric, comp in comparison["comparison"].items():
                    flag = "+" if comp["better_or_equal"] else "-"
                    print(f"  [{flag}] {metric}: generated={comp['generated']} "
                          f"vs ref={comp['ref']}")
                print(f"  Performance: {summary['performance']}")
    else:
        summary["metrics_comparison"] = {
            "ref": {
                "compile_success": build_status.get(ref_label, False),
                "run_success":     run_status.get(ref_label, False),
            },
            "generated": {
                "compile_success": build_status.get(gen_label, False),
                "run_success":     run_status.get(gen_label, False),
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
        description="Build and run ibv_query_qp_data_in_order benchmark. "
                    "Single-process — no server/client needed."
    )
    parser.add_argument(
        "--source", "-s", default="ref_ibv_query_qp_data_in_order.cpp",
        help="Source file to compile (default: ref_ibv_query_qp_data_in_order.cpp)",
    )
    parser.add_argument(
        "--output", "-o", default="ref_ibv_query_qp_data_in_order",
        help="Output executable name (default: ref_ibv_query_qp_data_in_order)",
    )
    parser.add_argument(
        "--arch", "-a", default=None,
        help="GPU architecture (not used for RDMA/IB builds)",
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
        "--compiler", "-c", default="g++",
        help="Specify compiler path (default: g++)",
    )
    parser.add_argument(
        "--platform", "-p", default=None,
        help="Ignored for RDMA/IB builds",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("SRC_A", "SRC_B"), default=None,
        help="Compare two source files: build & run both.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug build")
    parser.add_argument(
        "--nic", type=int, default=0,
        help="NIC index to use (default: 0)",
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
            nic=args.nic, gid_index=args.gid_index,
            verbose=verbose,
            show_raw_output=args.show_raw_output,
        )
        sys.exit(0)

    result = build_and_run(
        source_file=args.source, output_file=args.output,
        compiler=args.compiler, platform=args.platform,
        debug=args.debug, arch=args.arch,
        build_only=args.build_only, run_only=args.run_only,
        verbose=verbose,
        nic=args.nic, gid_index=args.gid_index,
    )

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()