#!/usr/bin/env python3
"""
Module for compiling and running GPU communication programs.
Provides functionality to execute build_and_run.py scripts in dataset folders.
Includes utilities for finding source files and analyzing results.
Supports automatic platform detection and subdirectory selection.
"""

import re
import subprocess
import os
import signal
import json
import sys
import functools
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any


def _killpg_safe(pgid: int, sig: int) -> None:
    """SIGNAL the entire process group; swallow ESRCH/EPERM."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


_SHUANGMA_ENV_PREFIX = "/home/uccl/miniconda3/envs/shuangma_env"

# env passed to nvcc --version / python -c "import torch" probes — strips the
# conda flags that cause spurious "incompatible redefinition" warnings.
_PROBE_ENV_STRIP = ("NVCC_PREPEND_FLAGS", "NVCC_PREPEND_FLAGS_BACKUP")


def _nvcc_cuda_major(nvcc_path: str) -> Optional[int]:
    """Return CUDA major from `nvcc --version`, or None on failure."""
    probe_env = {k: v for k, v in os.environ.items() if k not in _PROBE_ENV_STRIP}
    try:
        r = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True,
                           timeout=10, env=probe_env)
        for line in (r.stdout + r.stderr).splitlines():
            if "release" in line:
                m = re.search(r"release\s+(\d+)\.", line)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    return None


@functools.lru_cache(maxsize=None)
def _torch_cuda_major() -> Optional[int]:
    """Return torch.version.cuda major as int (cached per process), or None."""
    probe_env = {k: v for k, v in os.environ.items() if k not in _PROBE_ENV_STRIP}
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch; v=torch.version.cuda; print(v.split('.')[0] if v else '')"],
            capture_output=True, text=True, timeout=15, env=probe_env,
        )
        s = r.stdout.strip()
        return int(s) if r.returncode == 0 and s.isdigit() else None
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def _cuda_home_matching_torch() -> Optional[str]:
    """Return the first CUDA_HOME whose nvcc major matches torch's CUDA major.

    Checks shuangma_env first (fast path when env and torch agree), then falls
    back to versioned system paths.  Returns None if no match is found, which
    lets each build_and_run.py do its own detection.
    """
    torch_major = _torch_cuda_major()
    if torch_major is None:
        return None
    candidates = [
        _SHUANGMA_ENV_PREFIX,
        f"/usr/local/cuda-{torch_major}.2",
        f"/usr/local/cuda-{torch_major}.1",
        f"/usr/local/cuda-{torch_major}.0",
        "/usr/local/cuda",
    ]
    for c in candidates:
        nvcc = os.path.join(c, "bin", "nvcc")
        if os.path.isfile(nvcc) and _nvcc_cuda_major(nvcc) == torch_major:
            return c
    return None


def _scrubbed_build_env() -> dict:
    """Return a copy of os.environ with conda-toolchain landmines removed.

    Anaconda's "anaconda standard" activation script sets several env vars
    that quietly break our nvcc-based builds. We also pin a few values so
    `torch.utils.cpp_extension` (DeepEP JIT) and plain nvcc agree on the
    same CUDA/host-compiler combo.

      * NVCC_PREPEND_FLAGS=-ccbin=.../x86_64-conda-linux-gnu-c++
            Forces nvcc to use the conda-shipped host g++ whose sysroot
            has GLIBC_PRIVATE symbol-version mismatches with the system
            libc -> `collect2: error: ld returned 1 exit status`.
      * NVCC_PREPEND_FLAGS_BACKUP
            Sibling of the above; conda restores it on env deactivation.
      * CUDAARCHS=...;101-real;...
            CMake equivalent of TORCH_CUDA_ARCH_LIST; rejected by torch.
      * TORCH_CUDA_ARCH_LIST=...;10.1;...
            torch.utils.cpp_extension rejects `10.1` (no such arch). We
            replace with the B300-compatible `10.0a` instead of deleting,
            so the extension actually targets the right SM.
      * CUDA_HOME
            Must point to a CUDA toolchain whose major version matches the
            torch wheel (e.g. cu130 → CUDA 13.x). shuangma_env ships nvcc
            12.8 while torch may be cu130; using the wrong one makes
            torch.utils.cpp_extension raise a CUDA-version-mismatch error.
            We detect the right home at startup (cached) and set it here so
            every child build process sees a consistent, correct value.
      * CC / CXX
            CUDA 12/13 accept host gcc up to 13.x; conda's bundled g++ is
            14.3 and its sysroot breaks linking. Point at system g++-12.
            torch.utils.cpp_extension honours CXX when emitting `-ccbin`.

    Per-example build_and_run.py scripts inherit this env when we Popen
    them; this scrub is therefore the single chokepoint that fixes every
    nvcc / cpp_extension build, not just one example.
    """
    env = os.environ.copy()
    for k in ("NVCC_PREPEND_FLAGS", "NVCC_PREPEND_FLAGS_BACKUP", "CUDAARCHS"):
        env.pop(k, None)
    if "10.1" in env.get("TORCH_CUDA_ARCH_LIST", ""):
        env["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    # Pin CUDA_HOME to whichever CUDA toolchain matches torch's CUDA major.
    # If detection fails, clear CUDA_HOME so each build_and_run.py auto-detects.
    cuda_home = _cuda_home_matching_torch()
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
    else:
        env.pop("CUDA_HOME", None)
    # Prepend the correct CUDA bin dir to PATH so that shutil.which("nvcc")
    # in any build_and_run.py finds the right nvcc before conda's shuangma_env
    # nvcc (12.8), which doesn't support B300 (compute_103a).
    if cuda_home:
        cuda_bin = os.path.join(cuda_home, "bin")
        path_parts = [cuda_bin] + [p for p in env.get("PATH", "").split(os.pathsep)
                                   if p != cuda_bin]
        env["PATH"] = os.pathsep.join(path_parts)
    # Use system g++-12: compatible with CUDA 12.x and 13.x, no conda sysroot.
    if os.path.isfile("/usr/bin/g++-12"):
        env["CXX"] = "/usr/bin/g++-12"
    if os.path.isfile("/usr/bin/gcc-12"):
        env["CC"] = "/usr/bin/gcc-12"
    return env


def _run_in_new_session(cmd, *, cwd=None, timeout=None):
    """`subprocess.run`-like helper that runs `cmd` in a fresh POSIX session
    and tears the entire session down on timeout / on exit.

    Why this exists: many of our build_and_run.py scripts launch multi-rank
    binaries (mscclpp / NCCL / mpirun) that themselves fork rank workers.
    With plain `subprocess.run(..., timeout=...)` the immediate child gets
    SIGKILLed but those rank workers get re-parented to init and keep
    busy-spinning on the GPU (we've observed them pegging cores and holding
    600+ MiB of VRAM for hours). Putting the child in its own session via
    `start_new_session=True` makes the child a session/group leader, so we
    can `killpg` the whole tree.

    The env passed to the child is filtered by `_scrubbed_build_env()` to
    remove conda-toolchain landmines (see that function's docstring).

    Returns a `subprocess.CompletedProcess`. Raises `subprocess.TimeoutExpired`
    after tearing down the session group, exactly like `subprocess.run`.
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
        env=_scrubbed_build_env(),
    )
    pgid = proc.pid  # start_new_session makes the child its own session/pgrp leader
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            _killpg_safe(pgid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                _killpg_safe(pgid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(cmd, rc, stdout, stderr)
    finally:
        # Even on a clean exit, sweep any rank workers the binary left behind.
        # The immediate child is already dead at this point, so SIGKILLing the
        # group hits only its descendants (and is a no-op if there are none).
        _killpg_safe(pgid, signal.SIGKILL)

# Import platform detection
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from platform_detect import detect_platform, get_platform_string, PlatformInfo
from perf_verdict import override_summary_verdict


class CompileRunResult:
    """Container for compile and run results."""

    def __init__(self,
                 compile_success: bool,
                 compile_stdout: str,
                 compile_stderr: str,
                 compile_returncode: int,
                 run_success: bool = False,
                 run_stdout: str = "",
                 run_stderr: str = "",
                 run_returncode: int = -1,
                 metrics: Optional[Dict[str, Any]] = None):
        self.compile_success = compile_success
        self.compile_stdout = compile_stdout
        self.compile_stderr = compile_stderr
        self.compile_returncode = compile_returncode
        self.run_success = run_success
        self.run_stdout = run_stdout
        self.run_stderr = run_stderr
        self.run_returncode = run_returncode
        self.metrics = metrics or {}

    def __str__(self):
        result = []
        result.append("=" * 60)
        result.append("COMPILATION RESULT")
        result.append("=" * 60)
        result.append(f"Success: {self.compile_success}")
        result.append(f"Return Code: {self.compile_returncode}")
        if self.compile_stdout:
            result.append(f"\nStdout:\n{self.compile_stdout}")
        if self.compile_stderr:
            result.append(f"\nStderr:\n{self.compile_stderr}")

        result.append("\n" + "=" * 60)
        result.append("EXECUTION RESULT")
        result.append("=" * 60)
        result.append(f"Success: {self.run_success}")
        result.append(f"Return Code: {self.run_returncode}")
        if self.run_stdout:
            result.append(f"\nStdout:\n{self.run_stdout}")
        if self.run_stderr:
            result.append(f"\nStderr:\n{self.run_stderr}")

        return "\n".join(result)

    def to_dict(self) -> Dict:
        """Convert result to dictionary format."""
        return {
            'compile': {
                'success': self.compile_success,
                'stdout': self.compile_stdout,
                'stderr': self.compile_stderr,
                'returncode': self.compile_returncode
            },
            'run': {
                'success': self.run_success,
                'stdout': self.run_stdout,
                'stderr': self.run_stderr,
                'returncode': self.run_returncode
            },
            'metrics': self.metrics
        }


def get_platform_subdir(dataset_dir: str, platform_info: Optional[PlatformInfo] = None) -> Optional[str]:
    """
    Get the appropriate platform subdirectory based on detected hardware.

    Returns the first candidate subdirectory that (a) exists and (b) contains
    a build_and_run.py, walking a priority-ordered list keyed on the host's
    platform:

        1. ``amd_nv`` — the unified variant, preferred when present.
        2. The GPU-vendor-specific dir (``amd`` or ``nv``).
        3. The network-type-specific dirs (``ib``/``roce`` or ``efa``).

    GPU dirs come before network dirs because some examples (e.g.
    example016_rdma_send_recv_ud) split per-GPU variants across an ``amd/``
    dir and a network-named dir, where the network-named dir actually holds
    the NVIDIA variant; on an AMD host we still want ``amd/`` to win.

    Network dirs as a final fallback catch examples that ship only a
    network-named variant (e.g. example015_memory_pool_with_registered_region
    which only has ``roce/``).

    Args:
        dataset_dir: Path to the dataset directory
        platform_info: Optional PlatformInfo object (will detect if not provided)

    Returns:
        Path to the platform subdirectory, or None if no candidate exists
        (i.e., the dataset uses the old flat structure or supports a
        platform other than this host).

    Example:
        >>> subdir = get_platform_subdir("/path/to/example1_ipc_gpu_comm")
        >>> print(subdir)
        /path/to/example1_ipc_gpu_comm/nv
    """
    dataset_path = Path(dataset_dir)

    if platform_info is None:
        platform_info = detect_platform()

    candidates = ["amd_nv"]

    gpu_vendor = getattr(getattr(platform_info, "gpu", None), "vendor", None)
    if gpu_vendor == "AMD":
        candidates.append("amd")
    elif gpu_vendor == "NVIDIA":
        candidates.append("nv")

    net_type = getattr(getattr(platform_info, "network", None), "type", None)
    if net_type == "EFA":
        candidates.append("efa")
    elif net_type in ("IB", "RoCE"):
        candidates.extend(["ib", "roce"])

    for name in candidates:
        sub = dataset_path / name
        if sub.is_dir() and (sub / "build_and_run.py").is_file():
            return str(sub)

    return None


def find_empty_file(dataset_dir: str, platform_info: Optional[PlatformInfo] = None) -> Optional[str]:
    """
    Find the empty_* source file in the dataset directory.
    Supports .cpp, .cu, and other source file extensions.
    Automatically selects platform subdirectory if present.

    Args:
        dataset_dir: Path to the dataset directory
        platform_info: Optional PlatformInfo for platform detection

    Returns:
        Path to the empty_* source file, or None if not found

    Example:
        >>> empty_file = find_empty_file("/path/to/dataset")
        >>> print(empty_file)
        /path/to/dataset/nv/empty_gpu_p2p_comm.cpp
    """
    # Check for platform subdirectory
    platform_subdir = get_platform_subdir(dataset_dir, platform_info)
    search_dir = platform_subdir if platform_subdir else dataset_dir

    dataset_path = Path(search_dir)

    # Find all files starting with "empty_" with supported extensions.
    # `.py` is included so Python-fixture examples (torch.distributed,
    # vllm, etc.) are not rejected at rounds=0 with "No empty_* source file
    # found"; their build_and_run.py knows how to run the .py file directly.
    supported_extensions = ['.cpp', '.cu', '.cxx', '.cc', '.hip', '.py']
    empty_files = []

    for ext in supported_extensions:
        empty_files.extend(dataset_path.glob(f"empty_*{ext}"))

    if not empty_files:
        return None

    # Return the first match (should only be one)
    return str(empty_files[0])


def find_ref_file(dataset_dir: str, platform_info: Optional[PlatformInfo] = None,
                  empty_file: Optional[str] = None) -> Optional[str]:
    """
    Find the ref_* source file in the dataset directory.
    If empty_file is provided, prefer the ref whose name matches
    (e.g. empty_rdma_loopback.cpp -> ref_rdma_loopback.cpp).
    """
    # Check for platform subdirectory
    platform_subdir = get_platform_subdir(dataset_dir, platform_info)
    search_dir = platform_subdir if platform_subdir else dataset_dir

    dataset_path = Path(search_dir)

    # Find all files starting with "ref_" with supported extensions
    # (`.py` covers torch.distributed / vllm fixtures — same as in
    # find_empty_file).
    supported_extensions = ['.cpp', '.cu', '.cxx', '.cc', '.hip', '.py']
    ref_files = []

    for ext in supported_extensions:
        ref_files.extend(dataset_path.glob(f"ref_*{ext}"))

    if not ref_files:
        return None

    if empty_file:
        base = os.path.basename(empty_file)
        suffix = base.replace("empty_", "", 1) 
        for rf in ref_files:
            if rf.name == f"ref_{suffix}":
                return str(rf)

    return str(ref_files[0])


def extract_metrics_from_output(output: str) -> Dict[str, Any]:
    """
    Extract performance metrics from program output.
    Looks for lines starting with 'METRICS_JSON:' and parses the JSON.

    Args:
        output: Program stdout/stderr output

    Returns:
        Dict with metric names as keys and their values

    Example:
        >>> output = "Running test...\\nMETRICS_JSON: {\"throughput_gbps\": 0.85}\\nDone!"
        >>> metrics = extract_metrics_from_output(output)
        >>> print(metrics)
        {'throughput_gbps': 0.85}
    """
    metrics: Dict[str, Any] = {}
    if not output:
        return metrics

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("METRICS_JSON:"):
            json_str = line[len("METRICS_JSON:"):].strip()
            try:
                metrics = json.loads(json_str)
                break
            except json.JSONDecodeError:
                continue

    return metrics


def compare_metrics(
    ref_metrics: Dict[str, Any],
    gen_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare generated code metrics against reference metrics.

    Args:
        ref_metrics: Metrics from reference implementation
        gen_metrics: Metrics from generated implementation

    Returns:
        Dict with comparison results including:
        - ref: reference metrics
        - generated: generated metrics
        - comparison: per-metric comparison (ratio, improvement)
        - summary: overall assessment

    Example:
        >>> ref = {"throughput_gbps": 1.0, "latency_ms": 100}
        >>> gen = {"throughput_gbps": 0.9, "latency_ms": 110}
        >>> comparison = compare_metrics(ref, gen)
    """
    result = {
        "ref": ref_metrics,
        "generated": gen_metrics,
        "comparison": {},
        "summary": {}
    }

    if not ref_metrics or not gen_metrics:
        result["summary"]["status"] = "incomplete"
        result["summary"]["message"] = "Missing metrics for comparison"
        return result

    # Compare common metrics
    common_keys = set(ref_metrics.keys()) & set(gen_metrics.keys())

    for key in common_keys:
        ref_val = ref_metrics[key]
        gen_val = gen_metrics[key]

        if isinstance(ref_val, (int, float)) and isinstance(gen_val, (int, float)) and ref_val != 0:
            ratio = gen_val / ref_val

            # Determine if higher or lower is better based on metric name
            # throughput: higher is better, latency: lower is better
            if "latency" in key.lower() or "time" in key.lower():
                improvement = (ref_val - gen_val) / ref_val * 100  # positive = better
                better = gen_val <= ref_val
            else:
                improvement = (gen_val - ref_val) / ref_val * 100  # positive = better
                better = gen_val >= ref_val

            result["comparison"][key] = {
                "ref": ref_val,
                "generated": gen_val,
                "ratio": round(ratio, 4),
                "improvement_pct": round(improvement, 2),
                "better_or_equal": better
            }

    # Generate summary
    if result["comparison"]:
        all_better = all(v["better_or_equal"] for v in result["comparison"].values())
        result["summary"]["status"] = "pass" if all_better else "degraded"
        result["summary"]["all_metrics_pass"] = all_better
    else:
        result["summary"]["status"] = "no_common_metrics"

    return result


def analyze_result(result: CompileRunResult) -> Tuple[bool, str]:
    """
    Analyze the compilation and execution result to determine if it passed.

    Args:
        result: CompileRunResult object

    Returns:
        Tuple of (passed: bool, reason: str)

    Example:
        >>> result = compile_and_run(folder, source)
        >>> passed, reason = analyze_result(result)
        >>> if passed:
        ...     print("Test passed!")
    """
    # Check if compilation failed
    if not result.compile_success:
        return False, f"Compilation failed with return code {result.compile_returncode}"

    # Check if execution failed
    if not result.run_success:
        return False, f"Execution failed with return code {result.run_returncode}"

    # Check for "FAIL" in output
    if "FAIL" in result.run_stdout or "FAIL" in result.run_stderr:
        return False, "Test marked as FAIL in output"

    # Check for "Verification PASSED" or similar success indicators
    if "PASS" in result.run_stdout or "success" in result.run_stdout.lower():
        return True, "Test passed successfully"

    # If no explicit failure and execution succeeded, consider it passed
    if result.run_returncode == 0:
        return True, "Execution completed successfully"

    return False, "Unknown failure"


def get_error_details(result: CompileRunResult) -> str:
    """
    Extract error details from a failed compilation or execution result.

    Args:
        result: CompileRunResult object

    Returns:
        str: Error details for feedback to LLM

    Example:
        >>> result = compile_and_run(folder, source)
        >>> passed, reason = analyze_result(result)
        >>> if not passed:
        ...     error_details = get_error_details(result)
        ...     print(error_details)
    """
    error_parts = []

    if not result.compile_success:
        error_parts.append("=== COMPILATION ERROR ===")
        if result.compile_stderr:
            error_parts.append(result.compile_stderr)
        if result.compile_stdout:
            error_parts.append(result.compile_stdout)
        error_parts.append(f"Return code: {result.compile_returncode}")
    elif not result.run_success:
        error_parts.append("=== EXECUTION ERROR ===")
        if result.run_stderr:
            error_parts.append(result.run_stderr)
        if result.run_stdout:
            error_parts.append(result.run_stdout)
        error_parts.append(f"Return code: {result.run_returncode}")
    else:
        # Test failed but execution succeeded
        error_parts.append("=== TEST FAILURE ===")
        if result.run_stdout:
            error_parts.append(result.run_stdout)
        if result.run_stderr:
            error_parts.append(result.run_stderr)

    return "\n".join(error_parts)


def compile_and_run(
    folder_path: str,
    source_file: str,
    extra_args: Optional[List[str]] = None,
    timeout: int = 300,
    verbose: bool = False
) -> CompileRunResult:
    """
    Compile and run a GPU communication program using the folder's build_and_run.py.

    Args:
        folder_path: Path to the folder containing build_and_run.py
                     (e.g., /home/yangzhou/shuangma/llm-for-gpu-comm/datasets/example1_ipc_gpu_comm)
        source_file: Name of the source file to compile
                     (e.g., ref_gpu_p2p_comm.cpp)
        extra_args: Optional list of extra arguments to pass to build_and_run.py
        timeout: Timeout in seconds for the command (default: 300)

    Returns:
        CompileRunResult: Object containing compilation and execution results

    Example:
        >>> result = compile_and_run(
        ...     folder_path='/path/to/example1_ipc_gpu_comm',
        ...     source_file='ref_gpu_p2p_comm.cpp'
        ... )
        >>> print(result)
        >>> if result.compile_success and result.run_success:
        ...     print("Success!")
    """
    # Convert to absolute path
    folder_path = os.path.abspath(folder_path)

    # Validate folder exists
    if not os.path.isdir(folder_path):
        return CompileRunResult(
            compile_success=False,
            compile_stdout="",
            compile_stderr=f"Folder not found: {folder_path}",
            compile_returncode=-1
        )

    # Check for build_and_run.py in the folder
    build_script = os.path.join(folder_path, "build_and_run.py")
    if not os.path.isfile(build_script):
        return CompileRunResult(
            compile_success=False,
            compile_stdout="",
            compile_stderr=f"build_and_run.py not found in: {folder_path}",
            compile_returncode=-1
        )

    # Check source file exists
    source_path = os.path.join(folder_path, source_file)
    if not os.path.isfile(source_path):
        return CompileRunResult(
            compile_success=False,
            compile_stdout="",
            compile_stderr=f"Source file not found: {source_path}",
            compile_returncode=-1
        )

    # Build command: python build_and_run.py --source <source_file>
    # Intentionally do NOT pass `--quiet`. Most build_and_run.py scripts gate
    # the g++ stdout/stderr re-print on their own `verbose` flag, so `--quiet`
    # makes them silently swallow the compiler's real error message. We use
    # capture_output=True below so the subprocess output is captured into
    # `result.compile_stdout/stderr` and surfaced to the LLM via
    # `get_error_details`; it does NOT pollute our own tty regardless.
    cmd = ["python3", build_script, "--source", source_file]
    if extra_args:
        cmd.extend(extra_args)

    if verbose:
        print(f"Folder: {folder_path}")
        print(f"Source: {source_file}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)

    try:
        # Run in a fresh session so we can kill the *entire* multi-rank tree
        # (mscclpp / NCCL / mpirun launchers fork workers that survive plain
        # subprocess.run timeout cleanup).
        result = _run_in_new_session(
            cmd,
            cwd=folder_path,
            timeout=timeout,
        )

        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode

        # Parse output to determine compile vs run success
        # Look for indicators in the output
        compile_success = False
        run_success = False

        # Check for compilation success indicators
        if "Build successful" in stdout or "Compilation successful" in stdout:
            compile_success = True
        elif "Compilation failed" in stdout or "error:" in stderr.lower():
            compile_success = False
        elif returncode == 0:
            # If no explicit message but success, assume both worked
            compile_success = True

        # Check for run success
        if compile_success:
            if returncode == 0:
                run_success = True
            elif "PASS" in stdout or "Success" in stdout:
                run_success = True
            elif "FAIL" in stdout or "Error" in stdout:
                run_success = False

        # Extract metrics from output
        metrics = {}
        if compile_success and run_success:
            combined_output = stdout + "\n" + stderr
            metrics = extract_metrics_from_output(combined_output)

        # For combined output, put everything in compile_stdout
        # and determine success based on return code
        return CompileRunResult(
            compile_success=compile_success,
            compile_stdout=stdout,
            compile_stderr=stderr,
            compile_returncode=returncode if not compile_success else 0,
            run_success=run_success,
            run_stdout=stdout if compile_success else "",
            run_stderr=stderr if compile_success else "",
            run_returncode=returncode,
            metrics=metrics
        )

    except subprocess.TimeoutExpired:
        return CompileRunResult(
            compile_success=False,
            compile_stdout="",
            compile_stderr=f"Command timed out after {timeout} seconds",
            compile_returncode=-1
        )
    except Exception as e:
        return CompileRunResult(
            compile_success=False,
            compile_stdout="",
            compile_stderr=f"Error: {str(e)}",
            compile_returncode=-1
        )


def compare_save_results(
    working_dir: str,
    ref_filename: str,
    generated_filename: str,
    result_dir: str,
    verbose: bool = True,
    timeout: int = 600,
    no_plot: bool = False,
) -> bool:
    """Run build_and_run.py --compare to generate comparison results.

    Args:
        working_dir: Directory containing build_and_run.py and source files.
        ref_filename: Reference source filename (e.g. ref_gpu_p2p_comm.cpp).
        generated_filename: Generated source filename.
        result_dir: Directory to save results (CSV, summary.json, plots).
        verbose: Print progress.
        timeout: Subprocess timeout in seconds.
        no_plot: Skip generating plot images (CSV and summary JSON are still saved).

    Returns:
        True if the compare command succeeded.
    """
    build_script = os.path.join(working_dir, "build_and_run.py")
    if not os.path.isfile(build_script):
        if verbose:
            print(f"build_and_run.py not found in {working_dir}")
        return False

    cmd = [
        sys.executable, build_script,
        "--compare", ref_filename, generated_filename,
        "--results-dir", result_dir,
    ]
    if no_plot:
        cmd.append("--compare-no-plot")
    if not verbose:
        cmd.append("--quiet")

    try:
        # Same rationale as in compile_and_run(): tear down the whole session
        # group on timeout / completion so leaked rank workers can't survive.
        cp = _run_in_new_session(cmd, cwd=working_dir, timeout=timeout)
        if verbose:
            if cp.stdout:
                print(cp.stdout)
            if cp.stderr:
                print(cp.stderr, file=sys.stderr)
        if cp.returncode != 0 and verbose:
            print(f"--compare exited with code {cp.returncode}")

        # Post-process summary.json to overwrite `performance` with the
        # unified 4-tier verdict computed from the shared registry, so
        # every example produces consistent verdicts regardless of which
        # legacy scheme its own build_and_run.py used. The original verdict
        # is preserved at summary["performance_legacy"] for audit. The
        # actual rewrite logic lives in run_eval.perf_verdict so each
        # build_and_run.py can call it too (after its own --compare).
        if cp.returncode == 0:
            override_summary_verdict(result_dir, verbose=verbose)

        return cp.returncode == 0
    except Exception as e:
        if verbose:
            print(f"Comparison plots failed: {e}")
        return False


def main():
    """Example usage of the compile_and_run function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile and run GPU programs using build_and_run.py"
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing build_and_run.py"
    )
    parser.add_argument(
        "source",
        help="Source file name to compile"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--extra-args", "-e",
        nargs="*",
        help="Extra arguments to pass to build_and_run.py"
    )

    args = parser.parse_args()

    # Run compile and execute
    result = compile_and_run(
        folder_path=args.folder,
        source_file=args.source,
        extra_args=args.extra_args,
        timeout=args.timeout
    )

    # Print results
    print(result)

    # Summary
    if result.compile_success and result.run_success:
        print("\n✓ Both compilation and execution succeeded!")
    elif result.compile_success:
        print("\n✓ Compilation succeeded, but execution failed.")
    else:
        print("\n✗ Compilation failed.")

    return 0 if (result.compile_success and result.run_success) else 1


if __name__ == "__main__":
    exit(main())
