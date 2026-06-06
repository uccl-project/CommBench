#!/usr/bin/env python3
"""
Build/run/compare helper for example37_mscclpp_reducescatter.

Intended usage:

# Build and run reference
python build_and_run.py --source ref_mscclpp_reducescatter.cu

# Compare reference vs generated
python build_and_run.py --compare ref_mscclpp_reducescatter.cu empty_easy.cu

# Build only
python build_and_run.py --source ref_mscclpp_reducescatter.cu --build-only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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

_DEFAULT_SOURCE = "ref_mscclpp_reducescatter.cu"
_DEFAULT_ARCH = "sm_100a"
_DEFAULT_TIMEOUT_SEC = 900


def get_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


_DATASETS_DIR = os.path.dirname(get_module_dir())
_THIRD_PARTY_DIR = os.path.join(_DATASETS_DIR, "third_party")
_MSCCLPP_SRC_DIR = os.path.join(_THIRD_PARTY_DIR, "mscclpp")
_MSCCLPP_BUILD_PREFIX = os.path.join(_DATASETS_DIR, "build_mscclpp")
_MSCCLPP_BUILD_DIR = os.path.join(_DATASETS_DIR, ".build", "mscclpp")
_MSCCLPP_BUILD_SCRIPT = os.path.join(_MSCCLPP_SRC_DIR, "build_local.sh")
_MSCCLPP_INCLUDE_DIR = os.path.join(_MSCCLPP_BUILD_PREFIX, "include")
_MSCCLPP_TEST_INCLUDE_DIR = os.path.join(_MSCCLPP_SRC_DIR, "test", "mscclpp-test")
_MSCCLPP_LIB_DIR = os.path.join(_MSCCLPP_BUILD_PREFIX, "lib")
_MSCCLPP_STAMP = os.path.join(_MSCCLPP_BUILD_PREFIX, ".mscclpp_build_stamp.json")


@dataclass
class BuildResult:
    success: bool
    source_file: str
    resolved_source_file: str
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


def _existing_file(path: str) -> Optional[str]:
    return path if os.path.isfile(path) else None


def _mscclpp_stamp_payload() -> Dict[str, Any]:
    head = "unknown"
    git_dir = os.path.join(_MSCCLPP_SRC_DIR, ".git")
    if os.path.exists(git_dir):
        try:
            result = subprocess.run(
                ["git", "-C", _MSCCLPP_SRC_DIR, "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                head = result.stdout.strip()
        except OSError:
            pass
    tracked_files = [
        os.path.join(_MSCCLPP_SRC_DIR, "build_local.sh"),
        os.path.join(_MSCCLPP_SRC_DIR, "include", "mscclpp", "gpu.hpp"),
        os.path.join(_MSCCLPP_SRC_DIR, "CMakeLists.txt"),
    ]
    tracked_mtimes = {}
    for path in tracked_files:
        try:
            tracked_mtimes[path] = os.path.getmtime(path)
        except OSError:
            tracked_mtimes[path] = None

    return {
        "source_dir": _MSCCLPP_SRC_DIR,
        "git_head": head,
        "cuda_compiler": _detect_compiler(None),
        "tracked_mtimes": tracked_mtimes,
    }


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _mscclpp_ready() -> bool:
    required_files = [
        os.path.join(_MSCCLPP_INCLUDE_DIR, "mscclpp", "core.hpp"),
        os.path.join(_MSCCLPP_LIB_DIR, "libmscclpp.so"),
        _existing_file(_MSCCLPP_BUILD_SCRIPT),
    ]
    if not all(required_files):
        return False
    current = _mscclpp_stamp_payload()
    previous = _read_json(_MSCCLPP_STAMP)
    return (
        previous.get("git_head") == current.get("git_head")
        and previous.get("cuda_compiler") == current.get("cuda_compiler")
        and previous.get("tracked_mtimes") == current.get("tracked_mtimes")
    )


def _ensure_mscclpp_source() -> None:
    if os.path.isfile(os.path.join(_MSCCLPP_SRC_DIR, "CMakeLists.txt")):
        return
    raise RuntimeError(
        "MSCCL++ source tree is missing. Expected vendored source at "
        f"'{_MSCCLPP_SRC_DIR}'. Clone or copy https://github.com/microsoft/mscclpp "
        "there before building example37."
    )


def _ensure_mscclpp_artifacts(verbose: bool = True) -> None:
    _ensure_mscclpp_source()
    if _mscclpp_ready():
        return

    build_script = _existing_file(_MSCCLPP_BUILD_SCRIPT)
    if not build_script:
        raise RuntimeError(
            f"MSCCL++ build wrapper not found: '{_MSCCLPP_BUILD_SCRIPT}'."
        )

    os.makedirs(_MSCCLPP_BUILD_PREFIX, exist_ok=True)
    if os.path.isdir(_MSCCLPP_BUILD_DIR):
        shutil.rmtree(_MSCCLPP_BUILD_DIR)
    os.makedirs(_MSCCLPP_BUILD_DIR, exist_ok=True)

    cmd = [build_script, _MSCCLPP_BUILD_PREFIX, _MSCCLPP_BUILD_DIR]
    if verbose:
        print(f"[deps] building MSCCL++ into {_MSCCLPP_BUILD_PREFIX}")
        print(f"[deps] {' '.join(cmd)}")

    env = _sanitize_cuda_env(os.environ.copy())
    cuda_lib_dirs = _cuda_runtime_lib_dirs()
    if cuda_lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(cuda_lib_dirs + ([existing] if existing else []))
    env["CMAKE"] = _detect_cmake()
    env["CUDACXX"] = _detect_compiler(None)
    env["MSCCLPP_FORCE_DISABLE_NVLS"] = os.environ.get("MSCCLPP_FORCE_DISABLE_NVLS", "1")
    mscclpp_arch = _mscclpp_gpu_arch()
    if mscclpp_arch:
        env["MSCCLPP_GPU_ARCHS"] = mscclpp_arch
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if verbose and result.stdout:
        print(result.stdout, end="")
    if verbose and result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build vendored MSCCL++. "
            f"See stderr above. Build script: '{build_script}'."
        )

    stamp = _mscclpp_stamp_payload()
    stamp["built_at_utc"] = datetime.now(timezone.utc).isoformat()
    stamp["install_prefix"] = _MSCCLPP_BUILD_PREFIX
    stamp["build_dir"] = _MSCCLPP_BUILD_DIR
    with open(_MSCCLPP_STAMP, "w") as f:
        json.dump(stamp, f, indent=2)


def _resolve_source_path(path: str) -> Tuple[str, Optional[str]]:
    """
    Resolve source path with .cpp <-> .cu fallback.
    Returns (absolute_path, optional_note).
    """
    module_dir = get_module_dir()

    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.append(os.path.join(module_dir, path))

    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand), None

    root, ext = os.path.splitext(path)
    alt = None
    if ext == ".cpp":
        alt = root + ".cu"
    elif ext == ".cu":
        alt = root + ".cpp"

    if alt is not None:
        alt_path = alt if os.path.isabs(alt) else os.path.join(module_dir, alt)
        if os.path.exists(alt_path):
            note = f"'{path}' not found; using '{os.path.relpath(alt_path, module_dir)}' instead"
            return os.path.abspath(alt_path), note

    raise FileNotFoundError(f"Source file '{path}' not found")


def _resolve_output_path(output: Optional[str], source_abs: str) -> str:
    if output:
        if os.path.isabs(output):
            return output
        return os.path.join(get_module_dir(), output)

    stem, _ = os.path.splitext(os.path.basename(source_abs))
    return os.path.join(get_module_dir(), stem)


def _detect_compiler(compiler: Optional[str]) -> str:
    if compiler:
        return compiler

    # Override via env var; otherwise walk a portable search list.  The list
    # intentionally prefers conda-installed nvcc because a system CUDA at
    # /usr/local/cuda may be a different major version than the toolchain the
    # dependencies (e.g. mscclpp) were built against.
    env_nvcc = os.environ.get("NVCC", "").strip()
    if env_nvcc and os.path.isfile(env_nvcc) and os.access(env_nvcc, os.X_OK):
        return env_nvcc

    preferred: List[str] = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        preferred.append(os.path.join(conda_prefix, "bin", "nvcc"))
        # If CONDA_PREFIX is an env (.../envs/<name>), also try the base install.
        parent = os.path.dirname(os.path.dirname(conda_prefix))
        if os.path.basename(os.path.dirname(conda_prefix)) == "envs":
            preferred.append(os.path.join(parent, "bin", "nvcc"))

    home = os.path.expanduser("~")
    preferred += [
        os.path.join(home, "miniconda3", "bin", "nvcc"),
        os.path.join(home, "anaconda3", "bin", "nvcc"),
        "/opt/conda/bin/nvcc",
    ]

    for env_base in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if env_base:
            preferred.append(os.path.join(env_base, "bin", "nvcc"))
    preferred += ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"]

    for candidate in preferred:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc

    raise RuntimeError(
        "nvcc not found. Set $NVCC, pass --compiler, or install CUDA toolkit."
    )


def _cuda_runtime_lib_dirs() -> List[str]:
    # Candidates in preferred order.  A dir is only kept if it actually contains
    # a libcudart.so* — otherwise we'd rpath into an empty conda env and fail to
    # load libcudart at runtime (CONDA_PREFIX often points at an env that only
    # has python packages, while the CUDA runtime lives in the base install).
    candidates: List[str] = []

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates += [
            os.path.join(conda_prefix, "targets", "x86_64-linux", "lib"),
            os.path.join(conda_prefix, "lib"),
        ]

    # Fall back to the toolkit that provides nvcc (often a base conda install
    # with `cudatoolkit`, where libcudart lives in <prefix>/lib while nvcc
    # lives in <prefix>/bin).  Use the same resolution as _detect_compiler so
    # the libcudart we rpath matches the toolkit we compile with.
    try:
        nvcc = _detect_compiler(None)
    except RuntimeError:
        nvcc = None
    if nvcc and os.path.isfile(nvcc):
        base = os.path.dirname(os.path.dirname(nvcc))  # <prefix>/bin/nvcc -> <prefix>
        candidates += [
            os.path.join(base, "lib"),
            os.path.join(base, "targets", "x86_64-linux", "lib"),
            os.path.join(base, "lib64"),
        ]

    for base in (os.environ.get("CUDA_HOME", ""),
                 os.environ.get("CUDA_PATH", ""),
                 "/usr/local/cuda"):
        if base:
            candidates += [os.path.join(base, "lib64"),
                           os.path.join(base, "lib")]

    dirs: List[str] = []
    for path in candidates:
        if not path or not os.path.isdir(path):
            continue
        try:
            if any(name.startswith("libcudart.so") for name in os.listdir(path)):
                dirs.append(path)
        except OSError:
            continue
    return list(dict.fromkeys(dirs))


def _default_host_compiler() -> Optional[str]:
    candidate = "/usr/bin/g++"
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _detect_cmake() -> str:
    for candidate in ("cmake", "cmake3"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    cached = sorted(glob.glob(os.path.expanduser("~/.cache/uv/archive-v0/*/cmake/data/bin/cmake")))
    if cached:
        return cached[-1]

    raise RuntimeError(
        "cmake not found. Install CMake 3.25+ or export CMAKE=/path/to/cmake before running build_and_run.py."
    )


def _mscclpp_gpu_arch() -> str:
    arch = os.environ.get("MSCCLPP_GPU_ARCHS", "").strip()
    if arch:
        return arch

    nvcc_arch = os.environ.get("MSCCLPP_NVCC_ARCH", _DEFAULT_ARCH).strip()
    if not nvcc_arch:
        return ""

    if nvcc_arch.startswith("sm_"):
        nvcc_arch = nvcc_arch[3:]
    elif nvcc_arch.startswith("compute_"):
        nvcc_arch = nvcc_arch[8:]

    digits = "".join(ch for ch in nvcc_arch if ch.isdigit())
    return digits


def _existing_dir(path: str) -> Optional[str]:
    return path if os.path.isdir(path) else None


def _build_env() -> Dict[str, str]:
    _ensure_mscclpp_artifacts(verbose=False)
    env = os.environ.copy()
    lib_dirs = [_existing_dir(_MSCCLPP_LIB_DIR)] + _cuda_runtime_lib_dirs()
    lib_dirs = [path for path in lib_dirs if path]
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    return _sanitize_cuda_env(env)


def _sanitize_cuda_env(env: Dict[str, str]) -> Dict[str, str]:
    # In Conda shells, cross-toolchain vars may force nvcc to use Conda linker/sysroot.
    # Keep Python in Conda, but sanitize compiler/linker selection for native CUDA builds.
    if env.get("CONDA_PREFIX"):
        for key in ("LD", "LDFLAGS", "CC", "CXX", "NVCC_PREPEND_FLAGS"):
            env.pop(key, None)
        existing_path = env.get("PATH", "")
        env["PATH"] = ":".join(["/usr/bin", "/bin", existing_path]) if existing_path else "/usr/bin:/bin"
    return env


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
                snippet = text[start : i + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
    return {}


def _save_metrics_csv(metrics: List[Dict[str, Any]], csv_path: str) -> None:
    if not metrics:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    keys = list(metrics[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "count": float(n),
        "latency_avg": sum(float(m.get("latency_avg", 0.0)) for m in metrics) / n,
        "throughput_avg": sum(float(m.get("throughput_avg", 0.0)) for m in metrics) / n,
    }


def build(
    source_file: str,
    output_file: Optional[str] = None,
    compiler: Optional[str] = None,
    arch: Optional[str] = _DEFAULT_ARCH,
    debug: bool = False,
    verbose: bool = True,
) -> BuildResult:
    if arch:
        os.environ["MSCCLPP_NVCC_ARCH"] = arch
    try:
        _ensure_mscclpp_artifacts(verbose=verbose)
    except RuntimeError as exc:
        return BuildResult(
            success=False,
            source_file=source_file,
            resolved_source_file="",
            output_file=output_file or "",
            return_code=-1,
            stdout="",
            stderr="",
            command=[],
            error_message=str(exc),
        )

    try:
        source_abs, note = _resolve_source_path(source_file)
    except FileNotFoundError as exc:
        return BuildResult(
            success=False,
            source_file=source_file,
            resolved_source_file="",
            output_file=output_file or "",
            return_code=-1,
            stdout="",
            stderr="",
            command=[],
            error_message=str(exc),
        )

    out_abs = _resolve_output_path(output_file, source_abs)

    try:
        compiler_bin = _detect_compiler(compiler)
    except RuntimeError as exc:
        return BuildResult(
            success=False,
            source_file=source_file,
            resolved_source_file=source_abs,
            output_file=out_abs,
            return_code=-1,
            stdout="",
            stderr="",
            command=[],
            error_message=str(exc),
        )

    cmd = [compiler_bin]
    cmd += ["-std=c++17", "-x", "cu"]
    cmd += ["-DMSCCLPP_FORCE_DISABLE_NVLS=1"]
    host_compiler = _default_host_compiler()
    if host_compiler:
        cmd += ["-ccbin", host_compiler]
        if os.path.isdir("/usr/bin"):
            cmd += ["--compiler-options", "-B/usr/bin"]
    cmd += ["-O0", "-g"] if debug else ["-O3"]
    if arch:
        cmd.append(f"-arch={arch}")
    include_dirs = [
        get_module_dir(),
        _existing_dir(_MSCCLPP_INCLUDE_DIR),
        _existing_dir(_MSCCLPP_TEST_INCLUDE_DIR),
    ]
    for include_dir in include_dirs:
        if include_dir:
            cmd += ["-I", include_dir]
    lib_dir = _existing_dir(_MSCCLPP_LIB_DIR)
    if lib_dir:
        cmd += ["-L", lib_dir, "-Xlinker", f"-rpath={lib_dir}"]
        for cuda_lib_dir in _cuda_runtime_lib_dirs():
            cmd += ["-L", cuda_lib_dir, "-Xlinker", f"-rpath={cuda_lib_dir}"]
        cmd += ["-lmscclpp", "-lcudart", "-lcuda", "-lnuma"]
    cmd += [source_abs, "-o", out_abs]

    if verbose:
        if note:
            print(f"[build] NOTE: {note}")
        print(f"[build] {' '.join(cmd)}")

    env = _build_env()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if verbose and result.stdout:
        print(result.stdout, end="")
    if verbose and result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    return BuildResult(
        success=result.returncode == 0,
        source_file=source_file,
        resolved_source_file=source_abs,
        output_file=out_abs,
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=cmd,
        error_message=None if result.returncode == 0 else "Compilation failed",
    )


def run(
    executable: str,
    run_args: Optional[List[str]] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    verbose: bool = True,
) -> RunResult:
    exe_abs = executable if os.path.isabs(executable) else os.path.join(get_module_dir(), executable)
    if not os.path.exists(exe_abs):
        return RunResult(
            success=False,
            executable=executable,
            return_code=-1,
            stdout="",
            stderr="",
            command=[],
            parsed_output={},
            error_message=f"Executable '{executable}' not found",
        )

    cmd = [exe_abs] + (run_args or [])

    if verbose:
        print(f"[run] {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, env=_build_env())
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            success=False,
            executable=executable,
            return_code=-1,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            command=cmd,
            parsed_output={},
            error_message=f"Run timed out after {timeout_sec}s",
        )

    if verbose and result.stdout:
        print(result.stdout, end="")
    if verbose and result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    parsed = _parse_json_output(result.stdout) if result.returncode == 0 else {}

    return RunResult(
        success=result.returncode == 0,
        executable=executable,
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=cmd,
        parsed_output=parsed,
        error_message=None if result.returncode == 0 else "Execution failed",
    )


def build_and_run(
    source_file: str,
    output_file: Optional[str] = None,
    compiler: Optional[str] = None,
    arch: Optional[str] = _DEFAULT_ARCH,
    debug: bool = False,
    build_only: bool = False,
    run_only: bool = False,
    run_args: Optional[List[str]] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    verbose: bool = True,
) -> BuildAndRunResult:
    build_result: Optional[BuildResult] = None
    run_result: Optional[RunResult] = None

    if run_only and not output_file:
        output_file = os.path.splitext(os.path.basename(source_file))[0]

    if not run_only:
        build_result = build(
            source_file=source_file,
            output_file=output_file,
            compiler=compiler,
            arch=arch,
            debug=debug,
            verbose=verbose,
        )
        if not build_result.success:
            return BuildAndRunResult(build_result=build_result, run_result=None)

    if not build_only:
        exe_to_run = build_result.output_file if build_result else output_file
        if not exe_to_run:
            exe_to_run = os.path.splitext(os.path.basename(source_file))[0]
        run_result = run(
            executable=exe_to_run,
            run_args=run_args,
            timeout_sec=timeout_sec,
            verbose=verbose,
        )

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str,
    compiler: Optional[str] = None,
    arch: Optional[str] = _DEFAULT_ARCH,
    debug: bool = False,
    run_args: Optional[List[str]] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    verbose: bool = True,
) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)
    bin_dir = os.path.join(results_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    def _label(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    ref_label = _label(src_ref)
    gen_label = _label(src_gen)

    ref_out = os.path.join(bin_dir, ref_label)
    gen_out = os.path.join(bin_dir, gen_label)
    if os.path.abspath(ref_out) == os.path.abspath(gen_out):
        gen_out = gen_out + "_gen"

    if verbose:
        print(f"[compare] reference: {src_ref}")
        print(f"[compare] generated: {src_gen}")

    ref_res = build_and_run(
        source_file=src_ref,
        output_file=ref_out,
        compiler=compiler,
        arch=arch,
        debug=debug,
        run_args=run_args,
        timeout_sec=timeout_sec,
        verbose=verbose,
    )
    gen_res = build_and_run(
        source_file=src_gen,
        output_file=gen_out,
        compiler=compiler,
        arch=arch,
        debug=debug,
        run_args=run_args,
        timeout_sec=timeout_sec,
        verbose=verbose,
    )

    ref_ok = ref_res.run_result is not None and ref_res.run_result.success
    gen_ok = gen_res.run_result is not None and gen_res.run_result.success

    ref_parsed = ref_res.run_result.parsed_output if ref_ok else {}
    gen_parsed = gen_res.run_result.parsed_output if gen_ok else {}

    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])

    if ref_metrics:
        _save_metrics_csv(ref_metrics, os.path.join(results_dir, f"{ref_label}_metrics.csv"))
    if gen_metrics:
        _save_metrics_csv(gen_metrics, os.path.join(results_dir, f"{gen_label}_metrics.csv"))

    ref_avg = _metrics_avg(ref_metrics)
    gen_avg = _metrics_avg(gen_metrics)

    ref_by_size = {m.get("data_size"): m for m in ref_metrics}
    gen_by_size = {m.get("data_size"): m for m in gen_metrics}
    common_sizes = sorted(set(ref_by_size.keys()) & set(gen_by_size.keys()))

    per_size: List[Dict[str, Any]] = []
    for size in common_sizes:
        r = ref_by_size[size]
        g = gen_by_size[size]
        r_lat = float(r.get("latency_avg", 0.0))
        g_lat = float(g.get("latency_avg", 0.0))
        r_thr = float(r.get("throughput_avg", 0.0))
        g_thr = float(g.get("throughput_avg", 0.0))
        lat_impr = ((r_lat - g_lat) / r_lat * 100.0) if r_lat > 0 else 0.0
        thr_impr = ((g_thr - r_thr) / r_thr * 100.0) if r_thr > 0 else 0.0
        per_size.append(
            {
                "data_size": size,
                "ref_latency_avg": r_lat,
                "gen_latency_avg": g_lat,
                "latency_improvement_pct": round(lat_impr, 3),
                "ref_throughput_avg": r_thr,
                "gen_throughput_avg": g_thr,
                "throughput_improvement_pct": round(thr_impr, 3),
            }
        )

    ref_compile_ok = ref_res.build_result.success if ref_res.build_result else False
    gen_compile_ok = gen_res.build_result.success if gen_res.build_result else False
    summary: Dict[str, Any] = {
        "ref_source": os.path.basename(src_ref),
        "generated_source": os.path.basename(src_gen),
        "model": "",
        "pass_iteration": 1,
        "improvement_iteration": 1,
        "data_size_unit": ref_parsed.get("data_size_unit", gen_parsed.get("data_size_unit", "MiB")),
        "latency_unit": ref_parsed.get("latency_unit", gen_parsed.get("latency_unit", "us")),
        "throughput_unit": ref_parsed.get("aggregate_throughput", gen_parsed.get("aggregate_throughput", "GB/s")),
        "metrics_comparison": {
            "ref":       {"compile_success": ref_compile_ok, "run_success": ref_ok, **ref_avg},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_ok, **gen_avg},
        },
        "common_points": len(common_sizes),
        "per_size_comparison": per_size,
    }

    if ref_avg and gen_avg:
        r_lat = ref_avg.get("latency_avg", 0.0)
        g_lat = gen_avg.get("latency_avg", 0.0)
        r_thr = ref_avg.get("throughput_avg", 0.0)
        g_thr = gen_avg.get("throughput_avg", 0.0)
        summary["latency_improvement_pct"] = round(((r_lat - g_lat) / r_lat * 100.0) if r_lat > 0 else 0.0, 3)
        summary["throughput_improvement_pct"] = round(((g_thr - r_thr) / r_thr * 100.0) if r_thr > 0 else 0.0, 3)

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"[compare] summary saved to: {summary_path}")
        mc = summary["metrics_comparison"]
        print(f"[compare] ref run success: {mc['ref']['run_success']}")
        print(f"[compare] gen run success: {mc['generated']['run_success']}")
        if "throughput_improvement_pct" in summary and "latency_improvement_pct" in summary:
            print(f"[compare] throughput improvement: {summary['throughput_improvement_pct']:+.2f}%")
            print(f"[compare] latency improvement: {summary['latency_improvement_pct']:+.2f}%")

    return summary


def _build_run_args(mode: Optional[str], gpus: Optional[int], program_args: List[str]) -> List[str]:
    args: List[str] = []
    if mode:
        args += ["--mode", mode]
    if gpus is not None:
        args += ["--gpus", str(gpus)]
    args += program_args
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run MSCCL++ reduce-scatter benchmark sources")
    parser.add_argument("--source", "-s", default=_DEFAULT_SOURCE, help=f"Source file (default: {_DEFAULT_SOURCE})")
    parser.add_argument("--compare", nargs=2, metavar=("SRC_REF", "SRC_GEN"), help="Compare two source files")
    parser.add_argument("--output", "-o", default=None, help="Output executable path")
    parser.add_argument("--build-only", "-b", action="store_true", help="Compile only")
    parser.add_argument("--run-only", "-r", action="store_true", help="Run only (expects existing executable)")
    parser.add_argument("--compiler", "-c", default=None, help="Compiler path (default: auto-detect nvcc)")
    parser.add_argument("--arch", "-a", default=_DEFAULT_ARCH, help=f"GPU arch for nvcc (default: {_DEFAULT_ARCH})")
    parser.add_argument("--debug", action="store_true", help="Build with debug flags")
    parser.add_argument("--mode", default=None, help="Optional runtime --mode value")
    parser.add_argument("--gpus", type=int, default=None, help="Optional runtime --gpus value")
    parser.add_argument("--program-args", nargs="*", default=[], help="Extra args forwarded to the executable")
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT_SEC, help="Run timeout seconds")
    parser.add_argument("--results-dir", default=None, help="Results directory (default: ./results)")
    parser.add_argument("--quiet", action="store_true", help="Less console output")
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

    run_args = _build_run_args(args.mode, args.gpus, args.program_args)
    results_dir = args.results_dir or os.path.join(get_module_dir(), "results")

    if args.compare:
        src_ref, src_gen = args.compare
        summary = compare(
            src_ref=src_ref,
            src_gen=src_gen,
            results_dir=results_dir,
            compiler=args.compiler,
            arch=args.arch,
            debug=args.debug,
            run_args=run_args,
            timeout_sec=args.timeout,
            verbose=verbose,
        )
        mc = summary.get("metrics_comparison", {})
        ref_mc = mc.get("ref", {}); gen_mc = mc.get("generated", {})
        ok = ref_mc.get("compile_success") and ref_mc.get("run_success")
        ok = ok and gen_mc.get("compile_success") and gen_mc.get("run_success")
        sys.exit(0 if ok else 1)

    result = build_and_run(
        source_file=args.source,
        output_file=args.output,
        compiler=args.compiler,
        arch=args.arch,
        debug=args.debug,
        build_only=args.build_only,
        run_only=args.run_only,
        run_args=run_args,
        timeout_sec=args.timeout,
        verbose=verbose,
    )

    if result.run_result and result.run_result.success and result.run_result.parsed_output:
        metrics = result.run_result.parsed_output.get("metrics", [])
        if metrics:
            label = os.path.splitext(os.path.basename(args.source))[0]
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            _save_metrics_csv(metrics, csv_path)
            if verbose:
                print(f"[run] metrics csv saved to: {csv_path}")

    if not result.success:
        if result.build_result and result.build_result.error_message:
            print(f"[build] ERROR: {result.build_result.error_message}", file=sys.stderr)
        if result.run_result and result.run_result.error_message:
            print(f"[run] ERROR: {result.run_result.error_message}", file=sys.stderr)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()