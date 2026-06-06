#!/usr/bin/env python3
"""Build and run the standalone SGLang TP QK RMSNorm example.

Examples:
  python build_and_run.py
  python build_and_run.py --gpus 8
  python build_and_run.py --source empty_sglang_qknorm_fused_easy.cu --output candidate --gpus 8
  python build_and_run.py --compare ref_sglang_qknorm_fused.cu empty_sglang_qknorm_fused_easy.cu --gpus 8
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

DEFAULT_SOURCE = "ref_sglang_qknorm_fused.cu"
DEFAULT_ARCH = "sm_100a"
DEFAULT_SGL_CUDA_ARCH = "1000"
DEFAULT_TIMEOUT_SEC = 900


def module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def datasets_dir() -> str:
    return os.path.dirname(module_dir())


def repo_root() -> str:
    return os.path.dirname(datasets_dir())


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


def resolve_source(path: str) -> str:
    candidates = [path] if os.path.isabs(path) else [
        os.path.join(module_dir(), path),
        os.path.join(repo_root(), path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"Source file not found: {path}")


def resolve_output(output: Optional[str], source_abs: str) -> str:
    if output:
        return output if os.path.isabs(output) else os.path.join(module_dir(), output)
    stem, _ = os.path.splitext(os.path.basename(source_abs))
    return os.path.join(module_dir(), stem)


def detect_nvcc(compiler: Optional[str]) -> str:
    if compiler:
        return compiler
    env_nvcc = os.environ.get("NVCC")
    if env_nvcc:
        return env_nvcc

    candidates: List[str] = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin", "nvcc"))
    candidates.extend([
        "/usr/local/cuda-13.2/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
    ])
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "bin", "nvcc"))
    which_nvcc = shutil.which("nvcc")
    if which_nvcc:
        candidates.append(which_nvcc)

    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("nvcc not found. Set NVCC or pass --compiler.")


def sglang_include_dir() -> str:
    path = os.path.join(datasets_dir(), "third_party", "sglang", "python", "sglang", "jit_kernel", "include")
    header = os.path.join(path, "sgl_kernel", "distributed", "common.cuh")
    if not os.path.isfile(header):
        raise RuntimeError(f"SGLang JIT kernel headers not found at {path}")
    return path


def tvm_ffi_include_dir() -> str:
    code = "import pathlib, tvm_ffi; print(pathlib.Path(tvm_ffi.__file__).parent / 'include')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Could not import tvm_ffi to find dlpack headers. "
            "Activate the SGLang Python environment first.\n"
            + result.stderr
        )
    path = result.stdout.strip()
    if not os.path.isfile(os.path.join(path, "dlpack", "dlpack.h")):
        raise RuntimeError(f"tvm_ffi include path does not contain dlpack/dlpack.h: {path}")
    return path


def build_env() -> Dict[str, str]:
    env = os.environ.copy()
    cuda_home = os.path.dirname(os.path.dirname(detect_nvcc(None)))
    lib64 = os.path.join(cuda_home, "lib64")
    if os.path.isdir(lib64):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib64 + (":" + existing if existing else "")

    # Avoid conda's cross-linker/sysroot when compiling a native CUDA binary.
    env["NVCC_PREPEND_FLAGS"] = "-ccbin /usr/bin/g++ -Xcompiler -B/usr/bin"
    return env


def parse_json_output(text: str) -> Dict[str, Any]:
    depth = 0
    start: Optional[int] = None
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
                    pass
    return {}


def save_metrics_csv(metrics: List[Dict[str, Any]], csv_path: str) -> None:
    if not metrics:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def build(
    source_file: str,
    output_file: Optional[str],
    compiler: Optional[str],
    arch: str,
    sgl_cuda_arch: str,
    debug: bool,
    verbose: bool,
) -> BuildResult:
    try:
        source_abs = resolve_source(source_file)
        out_abs = resolve_output(output_file, source_abs)
        nvcc = detect_nvcc(compiler)
        sgl_include = sglang_include_dir()
        ffi_include = tvm_ffi_include_dir()
    except (FileNotFoundError, RuntimeError) as exc:
        return BuildResult(False, source_file, "", output_file or "", -1, "", "", [], str(exc))

    cmd = [
        nvcc,
        "--std=c++20",
        "--expt-relaxed-constexpr",
        "-x",
        "cu",
        "-O0" if debug else "-O3",
        f"-arch={arch}",
        f"-DSGL_CUDA_ARCH={sgl_cuda_arch}",
        "-I",
        sgl_include,
        "-I",
        ffi_include,
        source_abs,
        "-o",
        out_abs,
    ]

    if verbose:
        print("[build] " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=build_env())
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
    run_args: List[str],
    timeout_sec: int,
    verbose: bool,
) -> RunResult:
    exe_abs = executable if os.path.isabs(executable) else os.path.join(module_dir(), executable)
    if not os.path.isfile(exe_abs):
        return RunResult(False, executable, -1, "", "", [], {}, f"Executable not found: {executable}")

    cmd = [exe_abs] + run_args
    if verbose:
        print("[run] " + " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, env=build_env())
    except subprocess.TimeoutExpired as exc:
        return RunResult(False, executable, -1, exc.stdout or "", exc.stderr or "", cmd, {}, "Run timed out")

    if verbose and result.stdout:
        print(result.stdout, end="")
    if verbose and result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    parsed = parse_json_output(result.stdout) if result.returncode == 0 else {}
    return RunResult(
        success=result.returncode == 0,
        executable=exe_abs,
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=cmd,
        parsed_output=parsed,
        error_message=None if result.returncode == 0 else "Execution failed",
    )


def build_and_run(
    source_file: str,
    output_file: Optional[str],
    compiler: Optional[str],
    arch: str,
    sgl_cuda_arch: str,
    debug: bool,
    build_only: bool,
    run_only: bool,
    run_args: List[str],
    timeout_sec: int,
    verbose: bool,
) -> BuildAndRunResult:
    build_result: Optional[BuildResult] = None
    run_result: Optional[RunResult] = None

    if not run_only:
        build_result = build(source_file, output_file, compiler, arch, sgl_cuda_arch, debug, verbose)
        if not build_result.success:
            return BuildAndRunResult(build_result, None)
        executable = build_result.output_file
    else:
        source_abs = resolve_source(source_file)
        executable = resolve_output(output_file, source_abs)

    if not build_only:
        run_result = run(executable, run_args, timeout_sec, verbose)
    return BuildAndRunResult(build_result, run_result)


def compare(
    ref_source: str,
    gen_source: str,
    args: argparse.Namespace,
    run_args: List[str],
    verbose: bool,
) -> Dict[str, Any]:
    results_dir = args.results_dir
    bin_dir = os.path.join(results_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    ref_out = os.path.join(bin_dir, os.path.splitext(os.path.basename(ref_source))[0])
    gen_out = os.path.join(bin_dir, os.path.splitext(os.path.basename(gen_source))[0] + "_gen")

    ref = build_and_run(ref_source, ref_out, args.compiler, args.arch, args.sgl_cuda_arch, args.debug,
                        False, False, run_args, args.timeout, verbose)
    gen = build_and_run(gen_source, gen_out, args.compiler, args.arch, args.sgl_cuda_arch, args.debug,
                        False, False, run_args, args.timeout, verbose)

    ref_metrics = ref.run_result.parsed_output.get("metrics", []) if ref.run_result and ref.run_result.success else []
    gen_metrics = gen.run_result.parsed_output.get("metrics", []) if gen.run_result and gen.run_result.success else []
    save_metrics_csv(ref_metrics, os.path.join(results_dir, "ref_metrics.csv"))
    save_metrics_csv(gen_metrics, os.path.join(results_dir, "generated_metrics.csv"))

    def _avg(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        if not rows:
            return {}
        n = len(rows)
        return {
            "count": float(n),
            "latency_avg": sum(float(r.get("latency_avg", 0.0)) for r in rows) / n,
            "throughput_avg": sum(float(r.get("throughput_avg", 0.0)) for r in rows) / n,
        }

    ref_avg = _avg(ref_metrics)
    gen_avg = _avg(gen_metrics)
    ref_compile_ok = bool(ref.build_result and ref.build_result.success)
    gen_compile_ok = bool(gen.build_result and gen.build_result.success)
    ref_run_ok = bool(ref.run_result and ref.run_result.success)
    gen_run_ok = bool(gen.run_result and gen.run_result.success)
    ref_parsed = (ref.run_result.parsed_output or {}) if ref.run_result else {}
    gen_parsed = (gen.run_result.parsed_output or {}) if gen.run_result else {}

    # `metrics_comparison.{ref,generated}` is the flat-dict contract consumed
    # by run_eval/perf_verdict.py (see datasets/readme.md). Primary metric is
    # `throughput_avg` (direction=higher) — matches the mscclpp variant entry.
    summary = {
        "ref_source": os.path.basename(ref_source),
        "generated_source": os.path.basename(gen_source),
        "model": "",
        "pass_iteration": 1,
        "improvement_iteration": 1,
        "data_size_unit": ref_parsed.get("data_size_unit", gen_parsed.get("data_size_unit", "tokens")),
        "latency_unit":   ref_parsed.get("latency_unit",   gen_parsed.get("latency_unit", "us")),
        "throughput_unit":ref_parsed.get("throughput_unit",gen_parsed.get("throughput_unit", "tokens/s")),
        "ref_correctness":      ref_parsed.get("Correctness"),
        "generated_correctness":gen_parsed.get("Correctness"),
        "metrics_comparison": {
            "ref":       {"compile_success": ref_compile_ok, "run_success": ref_run_ok, **ref_avg},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok, **gen_avg},
        },
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary


def make_run_args(mode: Optional[str], gpus: Optional[int], program_args: List[str]) -> List[str]:
    out: List[str] = []
    if mode:
        out += ["--mode", mode]
    if gpus is not None:
        out += ["--gpus", str(gpus)]
    out += program_args
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "-s", default=DEFAULT_SOURCE)
    parser.add_argument("--compare", nargs=2, metavar=("REF", "GEN"))
    parser.add_argument("--output", "-o")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--compiler", "-c")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--sgl-cuda-arch", default=DEFAULT_SGL_CUDA_ARCH)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--program-args", nargs="*", default=[])
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--results-dir", default=os.path.join(module_dir(), "results"))
    parser.add_argument("--quiet", action="store_true")
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
    run_args = make_run_args(args.mode, args.gpus, args.program_args)

    if args.compare:
        summary = compare(args.compare[0], args.compare[1], args, run_args, verbose)
        mc = summary.get("metrics_comparison", {})
        ref_mc = mc.get("ref", {}); gen_mc = mc.get("generated", {})
        ok = all([ref_mc.get("compile_success"), ref_mc.get("run_success"),
                  gen_mc.get("compile_success"), gen_mc.get("run_success")])
        sys.exit(0 if ok else 1)

    result = build_and_run(
        args.source,
        args.output,
        args.compiler,
        args.arch,
        args.sgl_cuda_arch,
        args.debug,
        args.build_only,
        args.run_only,
        run_args,
        args.timeout,
        verbose,
    )

    if result.run_result and result.run_result.parsed_output:
        metrics = result.run_result.parsed_output.get("metrics", [])
        save_metrics_csv(metrics, os.path.join(args.results_dir, "metrics.csv"))
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
