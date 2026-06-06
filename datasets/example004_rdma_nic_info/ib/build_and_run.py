#!/usr/bin/env python3
"""
RDMA NIC Info Build and Run Module

This module compiles and runs rdma_nic_info programs.
Can be used as a standalone script or imported as a module.

Usage as module:
    from build_and_run import build, run, build_and_run

    result = build("ref_rdma_nic_info.cpp", "rdma_nic_info")
    if result.success:
        run_result = run("rdma_nic_info")
"""

import subprocess
import sys
import os
import json
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
    error_message: Optional[str] = None


@dataclass
class BuildAndRunResult:
    """Result of combined build and run operation."""
    build_result: Optional[BuildResult]
    run_result: Optional[RunResult]
    performance_metrics: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Returns True if both build and run succeeded."""
        build_ok = self.build_result is None or self.build_result.success
        run_ok = self.run_result is None or self.run_result.success
        return build_ok and run_ok


def get_module_dir() -> str:
    """Get the directory where this module is located."""
    return os.path.dirname(os.path.abspath(__file__))


def _extract_all_metrics(text: str) -> List[Dict[str, Any]]:
    """Extract all METRICS_JSON lines from text.

    Returns a list of dicts, one per METRICS_JSON line found.
    """
    metrics_list: List[Dict[str, Any]] = []
    if not text:
        return metrics_list

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("METRICS_JSON:"):
            json_str = line[len("METRICS_JSON:"):].strip()
            try:
                metrics = json.loads(json_str)
                metrics_list.append(metrics)
            except json.JSONDecodeError:
                continue

    return metrics_list


def check_compiler(verbose: bool = False) -> tuple:
    """Check if g++ is available."""
    try:
        result = subprocess.run(
            ["g++", "--version"],
            capture_output=True,
            text=True
        )
        if verbose and result.stdout:
            print("Compiler version:")
            print(result.stdout.split('\n')[0])
        return result.returncode == 0, result.stdout, result.stderr
    except FileNotFoundError:
        return False, "", "g++ not found"


def check_ibverbs(verbose: bool = False) -> tuple:
    """Check if libibverbs is available."""
    header_paths = [
        "/usr/include/infiniband/verbs.h",
        "/usr/local/include/infiniband/verbs.h"
    ]
    header_found = any(os.path.exists(p) for p in header_paths)
    if not header_found:
        return False, "infiniband/verbs.h not found"

    lib_paths = [
        "/usr/lib/x86_64-linux-gnu/libibverbs.so",
        "/usr/lib64/libibverbs.so",
        "/usr/lib/libibverbs.so"
    ]
    lib_found = any(os.path.exists(p) for p in lib_paths)
    if not lib_found:
        try:
            result = subprocess.run(
                ["pkg-config", "--exists", "libibverbs"],
                capture_output=True
            )
            lib_found = result.returncode == 0
        except FileNotFoundError:
            pass

    if lib_found:
        if verbose:
            print("libibverbs: found")
        return True, "libibverbs found"
    return False, "libibverbs not found"


def build(
    source_file: str = "ref_rdma_nic_info.cpp",
    output_file: str = "rdma_nic_info",
    cxx_flags: Optional[List[str]] = None,
    working_dir: Optional[str] = None,
    verbose: bool = False
) -> BuildResult:
    """Build the RDMA NIC info program."""
    if working_dir is None:
        working_dir = get_module_dir()

    original_dir = os.getcwd()
    try:
        os.chdir(working_dir)

        if verbose:
            print("===================================")
            print("Building RDMA NIC Info Tool")
            print("===================================")

        if not os.path.exists(source_file):
            return BuildResult(
                success=False, source_file=source_file, output_file=output_file,
                return_code=-1, stdout="", stderr="", command=[],
                error_message=f"Source file '{source_file}' not found"
            )

        if verbose:
            print(f"Using source file: {source_file}")

        compiler_available, compiler_stdout, compiler_stderr = check_compiler(verbose)
        if not compiler_available:
            return BuildResult(
                success=False, source_file=source_file, output_file=output_file,
                return_code=-1, stdout=compiler_stdout, stderr=compiler_stderr,
                command=[], error_message="g++ not found"
            )

        ibverbs_available, ibverbs_msg = check_ibverbs(verbose)
        if not ibverbs_available:
            return BuildResult(
                success=False, source_file=source_file, output_file=output_file,
                return_code=-1, stdout="", stderr=ibverbs_msg, command=[],
                error_message=f"libibverbs not found: {ibverbs_msg}"
            )

        if cxx_flags is None:
            cxx_flags = ["-O2", "-std=c++17", "-Wall"]

        cmd = ["g++"] + cxx_flags + [source_file, "-o", output_file, "-libverbs"]

        if verbose:
            print(f"\nCompiling {source_file}...")
            print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

        success = result.returncode == 0

        if verbose:
            print("\n===================================")
            print("Build successful!" if success else "Build failed!")
            if success:
                print(f"Executable: {output_file}")
            print("===================================")

        return BuildResult(
            success=success, source_file=source_file, output_file=output_file,
            return_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
            command=cmd, error_message=None if success else "Compilation failed"
        )
    finally:
        os.chdir(original_dir)


def run(
    executable: str = "rdma_nic_info",
    working_dir: Optional[str] = None,
    verbose: bool = False
) -> RunResult:
    """Run the RDMA NIC info program."""
    if working_dir is None:
        working_dir = get_module_dir()

    original_dir = os.getcwd()
    try:
        os.chdir(working_dir)

        if not os.path.exists(executable):
            return RunResult(
                success=False, executable=executable, return_code=-1,
                stdout="", stderr="FAIL", command=[],
                error_message=f"Executable '{executable}' not found"
            )

        cmd = [f"./{executable}"]

        if verbose:
            print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

        success = result.returncode == 0 and "PASS" in result.stdout

        return RunResult(
            success=success, executable=executable, return_code=result.returncode,
            stdout=result.stdout, stderr=result.stderr, command=cmd,
            error_message=None if success else "Execution failed"
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="Timeout after 30 seconds", command=[f"./{executable}"],
            error_message="Execution timed out"
        )
    finally:
        os.chdir(original_dir)


def build_and_run(
    source_file: str = "ref_rdma_nic_info.cpp",
    output_file: str = "rdma_nic_info",
    cxx_flags: Optional[List[str]] = None,
    working_dir: Optional[str] = None,
    verbose: bool = False,
    build_only: bool = False,
    run_only: bool = False
) -> BuildAndRunResult:
    """Build and run the RDMA NIC info program."""
    build_result = None
    run_result = None

    if working_dir is None:
        working_dir = get_module_dir()

    if not run_only:
        build_result = build(
            source_file=source_file,
            output_file=output_file,
            cxx_flags=cxx_flags,
            working_dir=working_dir,
            verbose=verbose
        )
        if not build_result.success:
            return BuildAndRunResult(build_result=build_result, run_result=None)

    if not build_only:
        if verbose:
            print("\n===================================")
            print("Running RDMA NIC Info Tool")
            print("===================================")

        run_result = run(
            executable=output_file,
            working_dir=working_dir,
            verbose=verbose
        )

    metrics_text = ""
    if run_result is not None:
        metrics_text = "\n".join([run_result.stdout, run_result.stderr]).strip()
    performance_metrics = _extract_all_metrics(metrics_text)
    return BuildAndRunResult(
        build_result=build_result,
        run_result=run_result,
        performance_metrics={"devices": performance_metrics}
    )


def _build_summary(
    ref_compile_ok: bool,
    ref_run_ok: bool,
    ref_avg_metrics: Dict[str, Any],
    gen_compile_ok: bool,
    gen_run_ok: bool,
    gen_avg_metrics: Dict[str, Any],
    latency_key: str = "latency_ms",
    throughput_key: str = "throughput_gbps",
) -> Dict[str, Any]:
    """Build a standardized summary dict for comparison results."""
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
    return summary


def compare_outputs(src_a: str, src_b: str, verbose: bool = False) -> bool:
    """Build and run two source files, compare their METRICS_JSON output field-by-field.

    Args:
        src_a: Path to first source file (typically the reference)
        src_b: Path to second source file (typically the generated)
        verbose: Print detailed output

    Returns:
        True if all METRICS_JSON fields match between the two programs.
    """
    working_dir = get_module_dir()

    # Build and run source A
    if verbose:
        print(f"\n>>> Building and running: {src_a}")
    result_a = build_and_run(
        source_file=src_a,
        output_file="nic_info_a",
        working_dir=working_dir,
        verbose=verbose
    )
    if not result_a.success:
        print(f"FAIL: {src_a} failed to build or run")
        if result_a.build_result and not result_a.build_result.success:
            print(f"  Build error: {result_a.build_result.error_message}")
            if result_a.build_result.stderr:
                print(f"  Stderr: {result_a.build_result.stderr}")
        if result_a.run_result and not result_a.run_result.success:
            print(f"  Run error: {result_a.run_result.error_message}")
        return False

    # Build and run source B
    if verbose:
        print(f"\n>>> Building and running: {src_b}")
    result_b = build_and_run(
        source_file=src_b,
        output_file="nic_info_b",
        working_dir=working_dir,
        verbose=verbose
    )
    if not result_b.success:
        print(f"FAIL: {src_b} failed to build or run")
        if result_b.build_result and not result_b.build_result.success:
            print(f"  Build error: {result_b.build_result.error_message}")
            if result_b.build_result.stderr:
                print(f"  Stderr: {result_b.build_result.stderr}")
        if result_b.run_result and not result_b.run_result.success:
            print(f"  Run error: {result_b.run_result.error_message}")
        return False

    # Extract METRICS_JSON lines from both
    stdout_a = result_a.run_result.stdout if result_a.run_result else ""
    stdout_b = result_b.run_result.stdout if result_b.run_result else ""

    metrics_a = _extract_all_metrics(stdout_a)
    metrics_b = _extract_all_metrics(stdout_b)

    if len(metrics_a) != len(metrics_b):
        print(f"FAIL: Different number of METRICS_JSON lines: "
              f"{len(metrics_a)} vs {len(metrics_b)}")
        return False

    if len(metrics_a) == 0:
        print("FAIL: No METRICS_JSON lines found in output")
        return False

    # Compare field-by-field for each device
    all_match = True
    for i, (ma, mb) in enumerate(zip(metrics_a, metrics_b)):
        device_name = ma.get("device_name", f"device_{i}")
        if verbose:
            print(f"\nComparing device: {device_name}")

        # Get all keys from both
        all_keys = sorted(set(ma.keys()) | set(mb.keys()))

        for key in all_keys:
            val_a = ma.get(key)
            val_b = mb.get(key)

            if val_a != val_b:
                print(f"  MISMATCH [{device_name}] {key}: {val_a!r} vs {val_b!r}")
                all_match = False
            elif verbose:
                print(f"  OK [{device_name}] {key}: {val_a!r}")

    if all_match:
        print(f"\nPASS: All {len(metrics_a)} device(s) match field-by-field")
    else:
        print(f"\nFAIL: Mismatches detected")

    return all_match


def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare two source files: build, run, field-by-field compare, save summary.json.

    Returns the summary dict.
    """
    os.makedirs(results_dir, exist_ok=True)
    working_dir = get_module_dir()

    # Build and run reference
    result_ref = build_and_run(
        source_file=src_ref, output_file="nic_info_a",
        working_dir=working_dir, verbose=verbose
    )
    ref_compile_ok = result_ref.build_result is not None and result_ref.build_result.success
    ref_run_ok = result_ref.run_result is not None and result_ref.run_result.success

    # Build and run generated
    result_gen = build_and_run(
        source_file=src_gen, output_file="nic_info_b",
        working_dir=working_dir, verbose=verbose
    )
    gen_compile_ok = result_gen.build_result is not None and result_gen.build_result.success
    gen_run_ok = result_gen.run_result is not None and result_gen.run_result.success

    # Non-performance summary (no latency/throughput metrics)
    summary = _build_summary(
        ref_compile_ok, ref_run_ok, {},
        gen_compile_ok, gen_run_ok, {},
    )

    # If both succeeded, do field-by-field comparison
    if ref_run_ok and gen_run_ok:
        stdout_a = result_ref.run_result.stdout if result_ref.run_result else ""
        stdout_b = result_gen.run_result.stdout if result_gen.run_result else ""
        metrics_a = _extract_all_metrics(stdout_a)
        metrics_b = _extract_all_metrics(stdout_b)

        all_match = True
        if len(metrics_a) != len(metrics_b) or len(metrics_a) == 0:
            all_match = False
        else:
            for ma, mb in zip(metrics_a, metrics_b):
                all_keys = set(ma.keys()) | set(mb.keys())
                for key in all_keys:
                    if ma.get(key) != mb.get(key):
                        all_match = False
                        break
                if not all_match:
                    break

        # For non-perf examples: pass = fields match, fail = mismatch
        summary["performance"] = "better" if all_match else "severely_degraded"

    json_path = os.path.join(results_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"Summary saved to {json_path}")

    return summary


def list_rdma_devices() -> List[str]:
    """List available RDMA devices."""
    try:
        result = subprocess.run(["ibv_devices"], capture_output=True, text=True)
        if result.returncode != 0:
            return []
        devices = []
        for line in result.stdout.strip().split('\n')[1:]:
            parts = line.split()
            if parts:
                devices.append(parts[0])
        return devices
    except FileNotFoundError:
        return []


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build and run RDMA NIC info tool"
    )
    parser.add_argument("--source", "-s", type=str, default="ref_rdma_nic_info.cpp",
                        help="Source file to compile (default: ref_rdma_nic_info.cpp)")
    parser.add_argument("--output", "-o", type=str, default="rdma_nic_info",
                        help="Output executable name (default: rdma_nic_info)")
    parser.add_argument("--build-only", action="store_true", help="Only build")
    parser.add_argument("--run-only", action="store_true", help="Only run")
    parser.add_argument("--list-devices", action="store_true", help="List RDMA devices and exit")
    parser.add_argument("--compare", nargs=2, metavar=("SRC_A", "SRC_B"),
                        help="Build both sources and compare METRICS_JSON output field-by-field")
    parser.add_argument("--results-dir", default=None,
                        help="Directory to save results (default: ./results)")
    parser.add_argument("--plot-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--compare-no-plot", action="store_true",
                        help=argparse.SUPPRESS)
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
        verbose = not args.quiet
        results_dir = args.results_dir or args.plot_dir
        if results_dir:
            summary = compare(
                src_ref=args.compare[0],
                src_gen=args.compare[1],
                results_dir=results_dir,
                verbose=verbose,
            )
            perf = summary.get("performance")
            sys.exit(0 if perf == "better" else 1)
        else:
            success = compare_outputs(args.compare[0], args.compare[1], verbose=verbose)
            sys.exit(0 if success else 1)

    result = build_and_run(
        source_file=args.source,
        output_file=args.output,
        verbose=not args.quiet,
        build_only=args.build_only,
        run_only=args.run_only
    )

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
