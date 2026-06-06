#!/usr/bin/env python3
"""
Memory Pool with Registered Region Build and Run Module

Compiles and runs programs that allocate a CUDA memory pool and register
it as an RDMA memory region (ibv_reg_mr).  Falls back to pinned host
memory when direct GPU MR registration is unsupported.

Usage as module:
    from build_and_run import build, run, build_and_run, compare

Usage as script:
    python build_and_run.py --source ref_mempool_reg_region.cpp
    python build_and_run.py --source ref_mempool_reg_region.cpp --pool-size-mb 512
    python build_and_run.py --build-only
    python build_and_run.py --run-only
    python build_and_run.py --compare ref_mempool_reg_region.cpp generated_mempool_reg_region.cpp
    python build_and_run.py --compare ref_mempool_reg_region.cpp generated_mempool_reg_region.cpp --results-dir ./results
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

    @property
    def success(self) -> bool:
        build_ok = self.build_result is None or self.build_result.success
        run_ok = self.run_result is None or self.run_result.success
        return build_ok and run_ok


def get_module_dir() -> str:
    """Get the directory where this module is located."""
    return os.path.dirname(os.path.abspath(__file__))


def check_compiler(verbose: bool = False) -> tuple:
    """Check if g++ is available."""
    try:
        result = subprocess.run(["g++", "--version"], capture_output=True, text=True)
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
        "/usr/local/include/infiniband/verbs.h",
    ]
    header_found = any(os.path.exists(p) for p in header_paths)
    if not header_found:
        return False, "infiniband/verbs.h not found"

    lib_paths = [
        "/usr/lib/x86_64-linux-gnu/libibverbs.so",
        "/usr/lib64/libibverbs.so",
        "/usr/lib/libibverbs.so",
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


def check_cuda(verbose: bool = False) -> tuple:
    """Check if CUDA toolkit is available.

    Tries the canonical installer layout (CUDA_HOME / /usr/local/cuda) first,
    then falls back to the system-package layout where headers and libraries
    are installed under /usr/include and /usr/lib/<triplet>.
    """
    cuda_paths = ["/usr/local/cuda", "/usr/cuda"]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        cuda_paths.insert(0, cuda_home)

    for base in cuda_paths:
        inc = os.path.join(base, "include", "cuda_runtime.h")
        lib = os.path.join(base, "lib64", "libcudart.so")
        if os.path.exists(inc) and os.path.exists(lib):
            if verbose:
                print(f"CUDA toolkit: found at {base}")
            return True, base

    # System-package fallback (Ubuntu nvidia-cuda-toolkit, etc.)
    sys_inc = "/usr/include/cuda_runtime.h"
    sys_libs = [
        "/usr/lib/aarch64-linux-gnu/libcudart.so",
        "/usr/lib/x86_64-linux-gnu/libcudart.so",
        "/usr/lib64/libcudart.so",
    ]
    if os.path.exists(sys_inc) and any(os.path.exists(p) for p in sys_libs):
        if verbose:
            print("CUDA toolkit: found in system include/lib")
        return True, "/usr"

    return False, "CUDA toolkit not found"


def build(
    source_file: str = "ref_mempool_reg_region.cpp",
    output_file: str = "gpu_mr_pool",
    cxx_flags: Optional[List[str]] = None,
    working_dir: Optional[str] = None,
    verbose: bool = False,
) -> BuildResult:
    """Build the memory pool program."""
    if working_dir is None:
        working_dir = get_module_dir()

    original_dir = os.getcwd()
    try:
        os.chdir(working_dir)

        if verbose:
            print("===================================")
            print("Building Memory Pool Test")
            print("===================================")

        if not os.path.exists(source_file):
            return BuildResult(
                success=False, source_file=source_file, output_file=output_file,
                return_code=-1, stdout="", stderr="", command=[],
                error_message=f"Source file '{source_file}' not found",
            )

        compiler_available, compiler_stdout, compiler_stderr = check_compiler(verbose)
        if not compiler_available:
            return BuildResult(
                success=False, source_file=source_file, output_file=output_file,
                return_code=-1, stdout=compiler_stdout, stderr=compiler_stderr,
                command=[], error_message="g++ not found",
            )

        ibverbs_available, ibverbs_msg = check_ibverbs(verbose)
        if not ibverbs_available:
            return BuildResult(
                success=False, source_file=source_file, output_file=output_file,
                return_code=-1, stdout="", stderr=ibverbs_msg, command=[],
                error_message=f"libibverbs not found: {ibverbs_msg}",
            )

        cuda_available, cuda_path = check_cuda(verbose)
        if not cuda_available:
            return BuildResult(
                success=False, source_file=source_file, output_file=output_file,
                return_code=-1, stdout="", stderr=cuda_path, command=[],
                error_message="CUDA toolkit not found",
            )

        if cxx_flags is None:
            cxx_flags = ["-O2", "-std=c++17"]

        # Resolve include/lib dirs that exist for both the canonical CUDA
        # installer layout (cuda_path/include + cuda_path/lib64) and Ubuntu's
        # nvidia-cuda-toolkit packaging (/usr/include + /usr/lib/<triplet>).
        cuda_include_candidates = [
            os.path.join(cuda_path, "include"),
            "/usr/include",
        ]
        cuda_lib_candidates = [
            os.path.join(cuda_path, "lib64"),
            "/usr/lib/aarch64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
        ]
        cuda_inc = next((d for d in cuda_include_candidates
                         if os.path.exists(os.path.join(d, "cuda_runtime.h"))),
                        cuda_include_candidates[0])
        cuda_lib = next((d for d in cuda_lib_candidates
                         if os.path.exists(os.path.join(d, "libcudart.so"))),
                        cuda_lib_candidates[0])

        cmd = (
            ["g++"]
            + cxx_flags
            + [source_file, "-o", output_file]
            + [f"-I{cuda_inc}", f"-L{cuda_lib}", "-lcudart"]
            + ["-libverbs"]
        )

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
            command=cmd, error_message=None if success else "Compilation failed",
        )
    finally:
        os.chdir(original_dir)


def run(
    executable: str = "gpu_mr_pool",
    working_dir: Optional[str] = None,
    verbose: bool = False,
    pool_size_mb: int = 256,
) -> RunResult:
    """Run the memory pool program."""
    if working_dir is None:
        working_dir = get_module_dir()

    original_dir = os.getcwd()
    try:
        os.chdir(working_dir)

        if not os.path.exists(executable):
            return RunResult(
                success=False, executable=executable, return_code=-1,
                stdout="", stderr="FAIL", command=[],
                error_message=f"Executable '{executable}' not found",
            )

        cmd = [f"./{executable}"]
        if pool_size_mb != 256:
            cmd += ["--pool-size-mb", str(pool_size_mb)]

        if verbose:
            print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

        success = (
            result.returncode == 0
            and (
                '"Correctness":"PASS"' in result.stdout
                or '"Correctness": "PASS"' in result.stdout
            )
        )

        return RunResult(
            success=success, executable=executable, return_code=result.returncode,
            stdout=result.stdout, stderr=result.stderr, command=cmd,
            error_message=None if success else "Execution failed or verification failed",
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="Timed out after 60s", command=[f"./{executable}"],
            error_message="Execution timed out",
        )
    finally:
        os.chdir(original_dir)


def build_and_run(
    source_file: str = "ref_mempool_reg_region.cpp",
    output_file: str = "gpu_mr_pool",
    cxx_flags: Optional[List[str]] = None,
    working_dir: Optional[str] = None,
    verbose: bool = False,
    build_only: bool = False,
    run_only: bool = False,
    pool_size_mb: int = 256,
) -> BuildAndRunResult:
    """Build and run the memory pool program."""
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
            verbose=verbose,
        )
        if not build_result.success:
            return BuildAndRunResult(build_result=build_result, run_result=None)

    if not build_only:
        if verbose:
            print("\n===================================")
            print("Running Memory Pool Test")
            print("===================================")

        run_result = run(
            executable=output_file,
            working_dir=working_dir,
            verbose=verbose,
            pool_size_mb=pool_size_mb,
        )

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare two source files: build and run both, save summary.json."""
    wd = get_module_dir()
    if results_dir is None:
        results_dir = os.path.join(wd, "results")
    os.makedirs(results_dir, exist_ok=True)

    summary: Dict[str, Any] = {}
    for label, src in [("ref", src_ref), ("generated", src_gen)]:
        br = build(source_file=src, output_file="gpu_mr_pool", working_dir=wd, verbose=verbose)
        rr = None
        if br.success:
            rr = run(executable="gpu_mr_pool", working_dir=wd, verbose=verbose)
        summary[label] = {
            "compile_success": br.success,
            "run_success": rr.success if rr else False,
            "stdout": rr.stdout if rr else "",
        }

    json_path = os.path.join(results_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"Summary saved to {json_path}")
    return summary


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build and run memory pool with registered region test"
    )
    parser.add_argument("--source", "-s", type=str, default="ref_mempool_reg_region.cpp",
                        help="Source file to compile (default: ref_mempool_reg_region.cpp)")
    parser.add_argument("--output", "-o", type=str, default="gpu_mr_pool",
                        help="Output executable name (default: gpu_mr_pool)")
    parser.add_argument("--build-only", action="store_true", help="Only build")
    parser.add_argument("--run-only", action="store_true", help="Only run")
    parser.add_argument("--compare", nargs=2, metavar=("SRC_A", "SRC_B"), default=None,
                        help="Compare two source files: build & run both. "
                             "Example: --compare ref_mempool_reg_region.cpp generated_mempool_reg_region.cpp")
    parser.add_argument("--results-dir", default=None,
                        help="Directory to save results (default: ./results)")
    parser.add_argument("--pool-size-mb", type=int, default=256,
                        help="Memory pool size in MB (default: 256)")
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

    if args.compare:
        src_a, src_b = args.compare
        wd = get_module_dir()
        for src in (src_a, src_b):
            if not os.path.exists(os.path.join(wd, src)):
                print(f"Error: Source file '{src}' not found!")
                sys.exit(1)
        compare(
            src_ref=src_a, src_gen=src_b,
            results_dir=args.results_dir or os.path.join(wd, "results"),
            verbose=not args.quiet,
        )
        sys.exit(0)

    result = build_and_run(
        source_file=args.source,
        output_file=args.output,
        verbose=not args.quiet,
        build_only=args.build_only,
        run_only=args.run_only,
        pool_size_mb=args.pool_size_mb,
    )

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
