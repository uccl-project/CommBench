#!/usr/bin/env python3
"""
Build/run/compare helper for example105_sglang_custom_allreduce.

Intended usage:

# Build and run reference
python build_and_run.py --source ref_sglang_custom_allreduce_pull.cu

# Compare reference vs generated
python build_and_run.py --compare ref_sglang_custom_allreduce_pull.cu generated.cu

# Build only
python build_and_run.py --source ref_sglang_custom_allreduce_pull.cu --build-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── unified perf-verdict bootstrap ────────────────────────────────────────
# Walk up from this file to locate run_eval/perf_verdict.py, add the repo
# root to sys.path, and import override_summary_verdict. Defaults to None
# if perf_verdict.py is unavailable so this build_and_run.py still works
# with the legacy verdict.
import os  # noqa: E402,F811
import sys  # noqa: E402,F811
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

_DEFAULT_SOURCE = "ref_sglang_custom_allreduce_pull.cu"
_DEFAULT_ARCH = "sm_100a"
_DEFAULT_SGL_CUDA_ARCH = "1000"
_DEFAULT_TIMEOUT_SEC = 900


def get_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


_DATASETS_DIR = os.path.dirname(get_module_dir())
_THIRD_PARTY_DIR = os.path.join(_DATASETS_DIR, "third_party")
_SGLANG_INCLUDE_DIR = os.path.join(_THIRD_PARTY_DIR, "sglang", "python", "sglang", "jit_kernel", "include")


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


def _resolve_source_path(path: str) -> Tuple[str, Optional[str]]:
    module_dir = get_module_dir()
    candidates = [path] if os.path.isabs(path) else [os.path.join(module_dir, path)]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate), None

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
        return output if os.path.isabs(output) else os.path.join(get_module_dir(), output)
    stem, _ = os.path.splitext(os.path.basename(source_abs))
    return os.path.join(get_module_dir(), stem)


def _detect_compiler(compiler: Optional[str]) -> str:
    if compiler:
        return compiler

    env_nvcc = os.environ.get("NVCC", "").strip()
    if env_nvcc and os.path.isfile(env_nvcc) and os.access(env_nvcc, os.X_OK):
        return env_nvcc

    preferred: List[str] = []
    for env_base in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if env_base:
            preferred.append(os.path.join(env_base, "bin", "nvcc"))
    preferred += [
        "/usr/local/cuda-13.2/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        preferred.append(os.path.join(conda_prefix, "bin", "nvcc"))

    found = shutil.which("nvcc")
    if found:
        preferred.append(found)

    for candidate in preferred:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise RuntimeError("nvcc not found. Set $NVCC, pass --compiler, or install CUDA toolkit.")


def _default_host_compiler() -> Optional[str]:
    candidate = "/usr/bin/g++"
    return candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None


def _cuda_runtime_lib_dirs(compiler: Optional[str] = None) -> List[str]:
    candidates: List[str] = []
    try:
        nvcc = _detect_compiler(compiler)
        cuda_home = os.path.dirname(os.path.dirname(nvcc))
        candidates += [os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")]
    except RuntimeError:
        pass
    for base in (os.environ.get("CUDA_HOME", ""), os.environ.get("CUDA_PATH", ""), "/usr/local/cuda"):
        if base:
            candidates += [os.path.join(base, "lib64"), os.path.join(base, "lib")]

    dirs: List[str] = []
    for path in candidates:
        if path and os.path.isdir(path):
            try:
                if any(name.startswith("libcudart.so") for name in os.listdir(path)):
                    dirs.append(path)
            except OSError:
                pass
    return list(dict.fromkeys(dirs))


def _sanitize_cuda_env(env: Dict[str, str], compiler: Optional[str] = None) -> Dict[str, str]:
    # Conda can inject cross-linker settings that break native nvcc links.
    for key in ("LD", "LDFLAGS", "CC", "CXX"):
        env.pop(key, None)
    host_compiler = _default_host_compiler()
    if host_compiler:
        env["NVCC_PREPEND_FLAGS"] = f"-ccbin {host_compiler} -Xcompiler -B/usr/bin"

    lib_dirs = _cuda_runtime_lib_dirs(compiler)
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    return env


def _tvm_ffi_include_dir() -> str:
    code = "import pathlib, tvm_ffi; print(pathlib.Path(tvm_ffi.__file__).parent / 'include')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Could not import tvm_ffi to locate dlpack headers. "
            "Activate the SGLang Python environment first.\n"
            + result.stderr
        )
    include_dir = result.stdout.strip()
    if not os.path.isfile(os.path.join(include_dir, "dlpack", "dlpack.h")):
        raise RuntimeError(f"tvm_ffi include path does not contain dlpack/dlpack.h: {include_dir}")
    return include_dir


def _ensure_sglang_headers() -> None:
    header = os.path.join(_SGLANG_INCLUDE_DIR, "sgl_kernel", "distributed", "common.cuh")
    if not os.path.isfile(header):
        raise RuntimeError(
            "SGLang JIT kernel headers are missing. Expected vendored source at "
            f"'{_SGLANG_INCLUDE_DIR}'. Initialize the sglang submodule first."
        )


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
    sgl_cuda_arch: str = _DEFAULT_SGL_CUDA_ARCH,
    debug: bool = False,
    verbose: bool = True,
) -> BuildResult:
    try:
        _ensure_sglang_headers()
        ffi_include = _tvm_ffi_include_dir()
        source_abs, note = _resolve_source_path(source_file)
        out_abs = _resolve_output_path(output_file, source_abs)
        compiler_bin = _detect_compiler(compiler)
    except (RuntimeError, FileNotFoundError) as exc:
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

    cmd = [compiler_bin]
    cmd += ["--std=c++20", "--expt-relaxed-constexpr", "-x", "cu"]
    cmd += ["-O0", "-g"] if debug else ["-O3"]
    if arch:
        cmd.append(f"-arch={arch}")
    cmd.append(f"-DSGL_CUDA_ARCH={sgl_cuda_arch}")
    cmd += ["-I", _SGLANG_INCLUDE_DIR, "-I", ffi_include]
    cmd += [source_abs, "-o", out_abs]

    if verbose:
        if note:
            print(f"[build] NOTE: {note}")
        print(f"[build] {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=_sanitize_cuda_env(os.environ.copy(), compiler))

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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec,
                                env=_sanitize_cuda_env(os.environ.copy()))
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
    sgl_cuda_arch: str = _DEFAULT_SGL_CUDA_ARCH,
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
            sgl_cuda_arch=sgl_cuda_arch,
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
    sgl_cuda_arch: str = _DEFAULT_SGL_CUDA_ARCH,
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
        gen_out += "_gen"

    if verbose:
        print(f"[compare] reference: {src_ref}")
        print(f"[compare] generated: {src_gen}")

    ref_res = build_and_run(src_ref, ref_out, compiler, arch, sgl_cuda_arch, debug, False, False,
                            run_args, timeout_sec, verbose)
    gen_res = build_and_run(src_gen, gen_out, compiler, arch, sgl_cuda_arch, debug, False, False,
                            run_args, timeout_sec, verbose)

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
    ref_compile_ok = ref_res.build_result.success if ref_res.build_result else False
    gen_compile_ok = gen_res.build_result.success if gen_res.build_result else False

    # `metrics_comparison.{ref,generated}` is the flat-dict contract
    # consumed by run_eval/perf_verdict.py (see datasets/readme.md). Keys
    # `count`, `latency_avg`, `throughput_avg` match the mscclpp variant
    # registered in PERF_METRICS — primary=throughput_avg, higher=better.
    summary: Dict[str, Any] = {
        "ref_source": os.path.basename(src_ref),
        "generated_source": os.path.basename(src_gen),
        "model": "",
        "pass_iteration": 1,
        "improvement_iteration": 1,
        "data_size_unit": ref_parsed.get("data_size_unit", gen_parsed.get("data_size_unit", "bytes")),
        "latency_unit": ref_parsed.get("latency_unit", gen_parsed.get("latency_unit", "us")),
        "throughput_unit": ref_parsed.get("throughput_unit", gen_parsed.get("throughput_unit", "GB/s")),
        "metrics_comparison": {
            "ref":       {"compile_success": ref_compile_ok, "run_success": ref_ok, **ref_avg},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_ok, **gen_avg},
        },
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        _print_perf_comparison(summary, summary_path)

    return summary


def _print_perf_comparison(summary: Dict[str, Any], summary_path: str) -> None:
    mc = summary["metrics_comparison"]
    ref_mc = mc.get("ref", {})
    gen_mc = mc.get("generated", {})
    print()
    print(f"PERFORMANCE COMPARISON ({summary_path})")
    print()
    for key in ("count", "latency_avg", "throughput_avg"):
        if key in ref_mc and key in gen_mc:
            print(f"[+] {key}: ref={ref_mc[key]} gen={gen_mc[key]}")
    ref_t = ref_mc.get("throughput_avg")
    gen_t = gen_mc.get("throughput_avg")
    if isinstance(ref_t, (int, float)) and isinstance(gen_t, (int, float)) and ref_t > 0:
        imp = (gen_t - ref_t) / ref_t * 100.0
        print(f"Performance: {imp:+.2f}% (gen vs ref, throughput_avg higher-is-better)")
    print()
    print("=" * 60)
    print(f"ref       compile_success: {ref_mc.get('compile_success')}")
    print(f"ref       run_success:     {ref_mc.get('run_success')}")
    print(f"generated compile_success: {gen_mc.get('compile_success')}")
    print(f"generated run_success:     {gen_mc.get('run_success')}")
    print("=" * 60)


def _build_run_args(mode: Optional[str], gpus: Optional[int], program_args: List[str]) -> List[str]:
    args: List[str] = []
    if mode:
        args += ["--mode", mode]
    if gpus is not None:
        args += ["--gpus", str(gpus)]
    args += program_args
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run SGLang custom all-reduce benchmark sources")
    parser.add_argument("--source", "-s", default=_DEFAULT_SOURCE, help=f"Source file (default: {_DEFAULT_SOURCE})")
    parser.add_argument("--compare", nargs=2, metavar=("SRC_REF", "SRC_GEN"), help="Compare two source files")
    parser.add_argument("--output", "-o", default=None, help="Output executable path")
    parser.add_argument("--build-only", "-b", action="store_true", help="Compile only")
    parser.add_argument("--run-only", "-r", action="store_true", help="Run only (expects existing executable)")
    parser.add_argument("--compiler", "-c", default=None, help="Compiler path (default: auto-detect nvcc)")
    parser.add_argument("--arch", "-a", default=_DEFAULT_ARCH, help=f"GPU arch for nvcc (default: {_DEFAULT_ARCH})")
    parser.add_argument("--sgl-cuda-arch", default=_DEFAULT_SGL_CUDA_ARCH,
                        help=f"SGL_CUDA_ARCH define (default: {_DEFAULT_SGL_CUDA_ARCH})")
    parser.add_argument("--debug", action="store_true", help="Build with debug flags")
    parser.add_argument("--mode", default=None, help="Optional runtime --mode value")
    parser.add_argument("--gpus", type=int, default=None, help="Optional runtime --gpus value")
    parser.add_argument("--program-args", nargs="*", default=[], help="Extra args forwarded to the executable")
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT_SEC, help="Run timeout seconds")
    parser.add_argument("--results-dir", default=None, help="Results directory (default: ./results)")
    parser.add_argument("--platform", default="cuda", choices=["cuda"],
                        help="Forces the compilation platform (only 'cuda' supported here)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots from collected metrics (latency/throughput)")
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
            sgl_cuda_arch=args.sgl_cuda_arch,
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
        sgl_cuda_arch=args.sgl_cuda_arch,
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

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
