#!/usr/bin/env python3
"""
DeepEP JIT dataset helper for combine_impl.

The source file in this dataset is a copy of DeepEP's
`deep_ep/include/deep_ep/impls/combine.cuh`, stored
with a `.cu` suffix so existing dataset tooling can discover it.  This helper
does not compile that file directly.  Instead, it creates a temporary DeepEP
JIT include overlay, replaces only the target implementation header, and then
runs a small single-node elastic dispatch/combine test.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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



TARGET_IMPL = "combine.cuh"
TARGET_KERNEL = "combine_impl"
DEFAULT_SOURCE = "ref_deepep_jit_combine.cu"
DEFAULT_TIMEOUT_SEC = 900


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
    cached: bool = False


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


def get_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


THIS_DIR = Path(get_module_dir())
EXAMPLE_DIR = THIS_DIR.parent
DATASETS_DIR = EXAMPLE_DIR.parent
BUILD_ROOT = DATASETS_DIR / ".build" / "deepep_jit"
DEEPEP_ROOT = DATASETS_DIR / "third_party" / "DeepEP"


def _source_abs(source_file: str) -> Path:
    path = Path(source_file)
    return path if path.is_absolute() else THIS_DIR / path


def _find_deepep_root() -> Optional[Path]:
    root = DEEPEP_ROOT
    header = root / "deep_ep" / "include" / "deep_ep" / "impls" / TARGET_IMPL
    if header.is_file():
        return root.resolve()
    return None


def _parse_json_output(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    depth = 0
    start = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:idx + 1])
                except json.JSONDecodeError:
                    continue
    return {}


def _copy_or_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _prepare_overlay(source_file: Path, deepep_root: Path) -> Path:
    source_stem = source_file.stem
    overlay_root = BUILD_ROOT / "overlays" / EXAMPLE_DIR.name / source_stem
    overlay_include = overlay_root / "include" / "deep_ep"
    overlay_impls = overlay_include / "impls"
    upstream_include = deepep_root / "deep_ep" / "include" / "deep_ep"
    upstream_impls = upstream_include / "impls"

    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    overlay_impls.mkdir(parents=True, exist_ok=True)

    _copy_or_link(upstream_include / "common", overlay_include / "common")
    for header in sorted(upstream_impls.glob("*.cuh")):
        if header.name == TARGET_IMPL:
            continue
        _copy_or_link(header, overlay_impls / header.name)
    shutil.copy2(source_file, overlay_impls / TARGET_IMPL)
    return overlay_root


def _site_library_dirs() -> List[Path]:
    code = (
        "import json, site, pathlib; "
        "roots = site.getsitepackages() + [site.getusersitepackages()]; "
        "paths = []\n"
        "for root in roots:\n"
        "    p = pathlib.Path(root)\n"
        "    for rel in ('nvidia/nccl/lib', 'nvidia/nvshmem/lib', 'torch/lib'):\n"
        "        q = p / rel\n"
        "        if q.exists(): paths.append(str(q))\n"
        "print(json.dumps(paths))"
    )
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return [Path(p) for p in json.loads(result.stdout)]
    except Exception:
        pass
    return []


def _detect_cuda_home(env: Dict[str, str]) -> Optional[Path]:
    if env.get("CUDA_HOME"):
        return Path(env["CUDA_HOME"])
    for candidate in ("/usr/local/cuda-13.2", "/usr/local/cuda", "/usr/local/cuda-13.0"):
        path = Path(candidate)
        if (path / "bin" / "nvcc").is_file():
            return path
    code = "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or '')"
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
        cuda_home = result.stdout.strip()
        if result.returncode == 0 and cuda_home:
            return Path(cuda_home)
    except Exception:
        pass
    return None


def _detect_torch_cuda_arch() -> Optional[str]:
    code = (
        "import torch; "
        "print('.'.join(map(str, torch.cuda.get_device_capability(0))) "
        "if torch.cuda.is_available() else '')"
    )
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
        arch = result.stdout.strip()
        if result.returncode == 0 and arch:
            return arch
    except Exception:
        pass
    return None


def _runtime_env(deepep_root: Path, source_stem: str, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    # Anaconda's "anaconda standard" env activation exports
    #   TORCH_CUDA_ARCH_LIST="...;10.0;10.1;12.0+PTX"
    #   CUDAARCHS="...;100-real;101-real;120"
    # torch.utils.cpp_extension._get_cuda_arch_flags rejects the unknown
    # "10.1" with "ValueError: Unknown CUDA arch (10.1)". Drop both vars so
    # the device-detected arch (set via setdefault below) is what wins.
    for _badvar in ("TORCH_CUDA_ARCH_LIST", "CUDAARCHS"):
        _val = env.get(_badvar, "")
        if "10.1" in _val or "101" in _val:
            env.pop(_badvar, None)

    build_libs = sorted((deepep_root / "build").glob("lib.*"))
    python_paths = [str(p) for p in build_libs]
    python_paths.append(str(deepep_root))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    cuda_home = _detect_cuda_home(env)
    if cuda_home is not None:
        env.setdefault("CUDA_HOME", str(cuda_home))
    detected_arch = _detect_torch_cuda_arch()
    if detected_arch is not None:
        env.setdefault("TORCH_CUDA_ARCH_LIST", detected_arch)

    path_parts = []
    if cuda_home is not None:
        path_parts.append(str(cuda_home / "bin"))
    path_parts.append(str(Path(sys.executable).parent))
    if env.get("PATH"):
        path_parts.append(env["PATH"])
    env["PATH"] = os.pathsep.join(path_parts)

    lib_parts = []
    if cuda_home is not None:
        lib_parts.append(str(cuda_home / "lib64"))
    site_libs = _site_library_dirs()
    lib_parts.extend(str(p) for p in site_libs)
    if env.get("LD_LIBRARY_PATH"):
        lib_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_parts)

    # Build-time include/link paths, matching batch_scripts_b300/install_deps.sh
    # step 6. The DeepEP setup.py rebuild needs system Mellanox IB headers
    # (infiniband/mlx5dv.h, pulled in by nvshmem_common_ibgda.h) and the
    # nccl/nvshmem wheel lib dirs so `-l:libnccl.so` / `-l:libnvshmem_host.so`
    # link flags resolve. Without these, build fails with "fatal error:
    # infiniband/mlx5dv.h: No such file or directory".
    cpath_parts = ["/usr/include/x86_64-linux-gnu", "/usr/include"]
    if env.get("CPATH"):
        cpath_parts.append(env["CPATH"])
    env["CPATH"] = os.pathsep.join(cpath_parts)
    libpath_parts = [str(p) for p in site_libs]
    if env.get("LIBRARY_PATH"):
        libpath_parts.append(env["LIBRARY_PATH"])
    env["LIBRARY_PATH"] = os.pathsep.join(libpath_parts)

    for lib_dir in site_libs:
        lib_str = str(lib_dir)
        if lib_str.endswith("nvidia/nccl/lib") and not env.get("EP_NCCL_ROOT_DIR"):
            env["EP_NCCL_ROOT_DIR"] = str(lib_dir.parent)
        if lib_str.endswith("nvidia/nvshmem/lib") and not env.get("EP_NVSHMEM_ROOT_DIR"):
            env["EP_NVSHMEM_ROOT_DIR"] = str(lib_dir.parent)

    env["EP_JIT_CACHE_DIR"] = str(BUILD_ROOT / "jit_cache" / EXAMPLE_DIR.name / source_stem)
    env.setdefault("EP_DISABLE_GIN", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(random.randint(25000, 45000)))
    return env


def _runner_path() -> Path:
    runner = BUILD_ROOT / "runners" / f"{EXAMPLE_DIR.name}_{TARGET_KERNEL}.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(_RUNNER_CODE, encoding="utf-8")
    return runner


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: Path) -> None:
    if not metrics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(metrics[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "data_size_avg": sum(float(m.get("data_size", 0)) for m in metrics) / n,
        "latency_avg": sum(float(m.get("latency_avg", 0)) for m in metrics) / n,
        "throughput": sum(float(m.get("throughput_avg", 0)) for m in metrics) / n,
    }


def build(
    source_file: str = DEFAULT_SOURCE,
    output_file: Optional[str] = None,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    verbose: bool = True,
) -> BuildResult:
    src = _source_abs(source_file)
    if not src.is_file():
        return BuildResult(False, source_file, output_file or "", -1, "", "", [],
                           error_message=f"Source file not found: {src}")
    deepep_root = _find_deepep_root()
    if deepep_root is None:
        return BuildResult(False, source_file, output_file or "", -1, "", "", [],
                           error_message="DeepEP V2 source tree with deep_ep/include/deep_ep/impls was not found")
    env = _runtime_env(deepep_root, src.stem)
    if arch:
        env["TORCH_CUDA_ARCH_LIST"] = arch
    cmd = [sys.executable, "setup.py", "build"]
    if verbose:
        print(f"[build] DeepEP JIT overlay source: {src}", file=sys.stderr)
        print(f"[build] Command: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=deepep_root, capture_output=True, text=True, env=env, timeout=DEFAULT_TIMEOUT_SEC)
    return BuildResult(
        result.returncode == 0,
        str(src),
        output_file or str(src),
        result.returncode,
        result.stdout,
        result.stderr,
        cmd,
        None if result.returncode == 0 else "DeepEP build failed",
    )


def run(
    executable: str,
    verbose: bool = True,
    num_processes: int = 2,
    num_tokens: int = 256,
    hidden: int = 128,
    num_topk: int = 2,
    num_experts: int = 8,
    num_sms: int = 0,
    skip_perf_test: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SEC,
) -> RunResult:
    src = _source_abs(executable)
    if not src.is_file():
        return RunResult(False, executable, -1, "", "", [], error_message=f"Source file not found: {src}")

    deepep_root = _find_deepep_root()
    if deepep_root is None:
        payload = {
            "Correctness": "SKIP",
            "reason": "DeepEP V2 source tree with deep_ep/include/deep_ep/impls was not found",
        }
        stdout = json.dumps(payload)
        return RunResult(True, executable, 0, stdout, "", [], parsed_output=payload)

    overlay_root = _prepare_overlay(src, deepep_root)
    runner = _runner_path()
    env = _runtime_env(deepep_root, src.stem)
    cmd = [
        sys.executable, str(runner),
        "--overlay-root", str(overlay_root),
        "--num-processes", str(num_processes),
        "--num-tokens", str(num_tokens),
        "--hidden", str(hidden),
        "--num-topk", str(num_topk),
        "--num-experts", str(num_experts),
        "--num-sms", str(num_sms),
    ]
    if skip_perf_test:
        cmd.append("--skip-perf-test")

    if verbose:
        print(f"[run] DeepEP root: {deepep_root}", file=sys.stderr)
        print(f"[run] Overlay root: {overlay_root}", file=sys.stderr)
        print(f"[run] Command: {' '.join(cmd)}", file=sys.stderr)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return RunResult(False, executable, -1, stdout, stderr, cmd, _parse_json_output(stdout),
                         error_message=f"Timed out after {timeout} seconds")

    parsed = _parse_json_output(result.stdout)
    success = result.returncode == 0 and bool(parsed)
    return RunResult(success, executable, result.returncode, result.stdout, result.stderr, cmd, parsed,
                     None if success else "DeepEP JIT run failed or produced no JSON")


def build_and_run(
    source_file: str = DEFAULT_SOURCE,
    output_file: Optional[str] = None,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    verbose: bool = True,
    **run_kwargs: Any,
) -> BuildAndRunResult:
    build_result = build(source_file, output_file, compiler, platform, debug, arch, verbose)
    if not build_result.success:
        return BuildAndRunResult(build_result, None)
    run_result = run(build_result.output_file, verbose=verbose, **run_kwargs)
    return BuildAndRunResult(build_result, run_result)


def compare(
    ref_source: str,
    gen_source: str,
    results_dir: str = "results",
    show_raw_output: bool = False,
    **run_kwargs: Any,
) -> Dict[str, Any]:
    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = THIS_DIR / results_path
    results_path.mkdir(parents=True, exist_ok=True)

    ref = build_and_run(ref_source, verbose=False, **run_kwargs)
    gen = build_and_run(gen_source, verbose=False, **run_kwargs)
    ref_out = ref.run_result.parsed_output if ref.run_result else {}
    gen_out = gen.run_result.parsed_output if gen.run_result else {}

    if show_raw_output:
        if ref.run_result:
            print("[ref stdout]\n" + ref.run_result.stdout)
            print("[ref stderr]\n" + ref.run_result.stderr, file=sys.stderr)
        if gen.run_result:
            print("[gen stdout]\n" + gen.run_result.stdout)
            print("[gen stderr]\n" + gen.run_result.stderr, file=sys.stderr)

    _save_metrics_csv(ref_out.get("metrics", []), results_path / "ref_metrics.csv")
    _save_metrics_csv(gen_out.get("metrics", []), results_path / "generated_metrics.csv")

    ref_avg = _metrics_avg(ref_out.get("metrics", []))
    gen_avg = _metrics_avg(gen_out.get("metrics", []))
    ref_compile_ok = bool(ref.build_result and ref.build_result.success)
    gen_compile_ok = bool(gen.build_result and gen.build_result.success)
    ref_run_ok = bool(ref.run_result and ref.run_result.success)
    gen_run_ok = bool(gen.run_result and gen.run_result.success)
    # `metrics_comparison.{ref,generated}` is the flat-dict contract consumed
    # by run_eval/perf_verdict.py (see datasets/readme.md). Primary metric is
    # `throughput` (direction=higher) — matches _metrics_avg() output key.
    summary = {
        "ref_source": "",
        "generated_source": "",
        "model": "",
        "pass_iteration": 1,
        "improvement_iteration": 1,
        "ref_correctness": ref_out.get("Correctness", "FAIL"),
        "generated_correctness": gen_out.get("Correctness", "FAIL"),
        "ref_return_code": ref.run_result.return_code if ref.run_result else None,
        "generated_return_code": gen.run_result.return_code if gen.run_result else None,
        "metrics_comparison": {
            "ref":       {"compile_success": ref_compile_ok, "run_success": ref_run_ok, **ref_avg},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok, **gen_avg},
        },
    }
    (results_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _print_single_json(result: BuildAndRunResult) -> int:
    if result.run_result is not None and result.run_result.parsed_output:
        print(json.dumps(result.run_result.parsed_output))
        return 0 if result.success else 1
    if result.build_result and not result.build_result.success:
        br = result.build_result
        err_msg = br.error_message or "build failed"
        tail = "\n".join(((br.stderr or "") + "\n" + (br.stdout or "")).splitlines()[-60:])
    elif result.run_result:
        rr = result.run_result
        err_msg = rr.error_message or "run failed"
        tail = "\n".join(((rr.stderr or "") + "\n" + (rr.stdout or "")).splitlines()[-60:])
    else:
        err_msg = "unknown error"
        tail = ""
    payload = {
        "Correctness": "FAIL",
        "error": err_msg + (f"\n---\n{tail}" if tail else ""),
    }
    print(json.dumps(payload))
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run DeepEP JIT combine_impl dataset")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--compiler", default=None)
    parser.add_argument("--platform", default="nv")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "GENERATED"))
    parser.add_argument("--show-raw-output", action="store_true")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument("--skip-perf-test", action="store_true")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument(
        "--legacy-perf-verdict",
        action="store_true",
        help="Use this example's local verdict logic instead of the shared "
             "4-tier scheme in run_eval/perf_verdict.py."
    )
    args = parser.parse_args(argv)

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






    run_kwargs = dict(
        num_processes=args.num_processes,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        num_experts=args.num_experts,
        num_sms=args.num_sms,
        skip_perf_test=args.skip_perf_test,
        timeout=args.timeout,
    )

    if args.compare:
        summary = compare(args.compare[0], args.compare[1], args.results_dir, args.show_raw_output, **run_kwargs)
        print(json.dumps(summary))
        return 0 if summary.get("generated_correctness") in ("PASS", "SKIP") else 1

    if args.build_only:
        build_result = build(args.source, args.output, args.compiler, args.platform, False, args.arch, True)
        err = build_result.error_message or ""
        if not build_result.success:
            tail = "\n".join(
                ((build_result.stderr or "") + "\n" + (build_result.stdout or "")).splitlines()[-60:]
            )
            if tail:
                err = (err + ("\n---\n" if err else "") + tail)
        print(json.dumps({"Correctness": "PASS" if build_result.success else "FAIL",
                          "error": err}))
        return 0 if build_result.success else 1

    if args.run_only:
        run_result = run(args.output or args.source, verbose=True, **run_kwargs)
        print(json.dumps(run_result.parsed_output or {"Correctness": "FAIL", "error": run_result.error_message}))
        return 0 if run_result.success else 1

    result = build_and_run(args.source, args.output, args.compiler, args.platform, False, args.arch, True, **run_kwargs)
    return _print_single_json(result)


_RUNNER_CODE = r'''
import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path



def emit(payload, code=0):
    print(json.dumps(payload), flush=True)
    raise SystemExit(code)


def preflight(num_processes):
    try:
        import torch
    except Exception as exc:
        emit({"Correctness": "SKIP", "reason": f"PyTorch is not importable: {exc}"})
    if not torch.cuda.is_available():
        emit({"Correctness": "SKIP", "reason": "CUDA is not available to PyTorch"})
    if torch.cuda.device_count() < num_processes:
        emit({
            "Correctness": "SKIP",
            "reason": f"Need {num_processes} CUDA devices, found {torch.cuda.device_count()}",
        })
    try:
        import deep_ep
        import deep_ep._C as _C
        from deep_ep.utils.find_pkgs import find_nccl_root
        _C.init_jit(os.environ["DEEPEP_JIT_OVERLAY_ROOT"], deep_ep.find_cuda_home(), find_nccl_root())
    except Exception as exc:
        emit({"Correctness": "SKIP", "reason": f"DeepEP runtime is not importable or not built: {exc}"})


def worker(local_rank, num_local_ranks, args, work_dir):
    import torch
    import torch.distributed as dist
    import deep_ep
    import deep_ep._C as _C
    from deep_ep.utils.find_pkgs import find_nccl_root
    from deep_ep.utils.envs import init_dist
    from deep_ep.utils.gate import get_unbalanced_scores
    from deep_ep.utils.refs import combine as ref_combine
    from deep_ep.utils.refs import generate_pre_combine_data, ordered_accumulate
    from deep_ep.utils.testing import bench_kineto

    _C.init_jit(os.environ["DEEPEP_JIT_OVERLAY_ROOT"], deep_ep.find_cuda_home(), find_nccl_root())
    rank_idx, _, group = init_dist(local_rank, num_local_ranks, seed=0)
    torch.cuda.set_device(local_rank)
    buffer = None
    try:
        buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=args.num_tokens,
            hidden=args.hidden,
            deterministic=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
            explicitly_destroy=True,
            num_gpu_timeout_secs=30,
            num_cpu_timeout_secs=30,
        )
        num_scaleout_ranks, num_scaleup_ranks = buffer.get_logical_domain_size()
        assert num_scaleout_ranks == 1, (num_scaleout_ranks, num_scaleup_ranks)
        assert args.num_experts % buffer.num_ranks == 0

        num_tokens = max(1, args.num_tokens - rank_idx)
        scores = get_unbalanced_scores(num_tokens, args.num_experts, buffer.num_ranks, args.num_topk, 1.0, False)
        topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
        topk_idx = topk_idx.to(deep_ep.topk_idx_t)
        x = torch.randn((num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")

        ref_y = generate_pre_combine_data(
            rank_idx * args.num_tokens + torch.arange(num_tokens, device="cuda"),
            args.num_tokens,
            args.num_topk,
            args.hidden,
        )
        ref_y[topk_idx == -1] = 0
        ref_combined_x = ref_combine(
            ref_y,
            topk_idx,
            num_scaleout_ranks,
            num_scaleup_ranks,
            args.num_experts,
            None,
            True,
            False,
        )
        torch.cuda.synchronize()

        num_sms = buffer.get_theoretical_num_sms(args.num_experts, args.num_topk) if args.num_sms == 0 else args.num_sms
        num_qps = buffer.get_theoretical_num_qps(num_sms)
        dispatch_args = dict(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_sms=num_sms,
            num_qps=num_qps,
            num_max_tokens_per_rank=args.num_tokens,
            num_experts=args.num_experts,
            expert_alignment=128,
            async_with_compute_stream=False,
            allocate_on_comm_stream=False,
            do_handle_copy=True,
            do_cpu_sync=True,
        )

        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(**dispatch_args)
        torch.cuda.synchronize()

        num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
        src_token_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]
        local_y = generate_pre_combine_data(src_token_global_idx, args.num_tokens, args.num_topk, args.hidden)
        local_y[recv_topk_idx[:num_recv_tokens] == -1] = 0
        input_for_combine = torch.empty_like(recv_x, dtype=torch.bfloat16, device="cuda")
        input_for_combine[:num_recv_tokens] = ordered_accumulate(local_y)

        combine_args = dict(
            x=input_for_combine,
            topk_weights=recv_topk_weights,
            bias=None,
            handle=handle,
            num_sms=num_sms,
            num_qps=num_qps,
            async_with_compute_stream=False,
            allocate_on_comm_stream=False,
        )
        combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
        torch.cuda.synchronize()

        assert torch.equal(combined_x, ref_combined_x), (combined_x, ref_combined_x)
        assert torch.equal(combined_topk_weights, topk_weights), (combined_topk_weights, topk_weights)

        latency_s = 0.0
        if not args.skip_perf_test:
            latency_s = bench_kineto(
                lambda: buffer.combine(**combine_args),
                kernel_names="combine_impl",
                num_tests=10,
                suppress_kineto_output=True,
                barrier_comm_profiling=True,
                barrier=buffer.barrier,
            )

        send_bytes = num_recv_tokens * args.hidden * 2
        recv_bytes = num_tokens * args.hidden * 2
        weight_bytes = topk_weights.numel() * topk_weights.element_size()
        data_size_mb = (send_bytes + recv_bytes + weight_bytes) / 1e6
        throughput_gbs = 0.0 if latency_s == 0 else (data_size_mb / 1e3) / latency_s
        metric = {
            "name": "combine_impl",
            "rank": rank_idx,
            "data_size": data_size_mb,
            "latency_avg": latency_s * 1e6,
            "throughput_avg": throughput_gbs,
        }
        Path(work_dir, f"rank_{rank_idx}.json").write_text(json.dumps({"metric": metric}), encoding="utf-8")
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay-root", required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--num-topk", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--num-sms", type=int, required=True)
    parser.add_argument("--skip-perf-test", action="store_true")
    args = parser.parse_args()

    if args.num_experts % args.num_processes != 0:
        emit({"Correctness": "SKIP", "reason": "num_experts must be divisible by num_processes"})

    os.environ["DEEPEP_JIT_OVERLAY_ROOT"] = args.overlay_root
    preflight(args.num_processes)

    import torch
    with tempfile.TemporaryDirectory(prefix="deepep_dataset82_") as work_dir:
        try:
            torch.multiprocessing.spawn(worker, args=(args.num_processes, args, work_dir), nprocs=args.num_processes)
        except Exception:
            traceback.print_exc()
            emit({"Correctness": "FAIL", "kernel": "combine_impl"}, code=1)

        metrics = []
        for path in sorted(Path(work_dir).glob("rank_*.json")):
            metrics.append(json.loads(path.read_text())["metric"])
        emit({
            "Correctness": "PASS",
            "kernel": "combine_impl",
            "data_size_unit": "MB",
            "latency_unit": "us",
            "throughput_unit": "GB/s",
            "metrics": metrics,
        })


if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    raise SystemExit(main())
