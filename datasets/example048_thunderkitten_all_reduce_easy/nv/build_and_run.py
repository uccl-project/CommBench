#!/usr/bin/env python3
"""
Build and run the standalone ThunderKittens BF16 multi-GPU all_reduce
benchmark.

The .cu file is now a self-contained C++/CUDA executable: it forks one
child process per GPU, brings up NVLink multicast through ThunderKittens'
KittensBroker + driver-API helpers, runs a correctness check + a perf
benchmark, and prints exactly one JSON object on rank 0's stdout. There
is no PyTorch / pybind / torchrun involved here.

Build artifacts (binary `<source_stem>_<gpu>`) go to the same `nv/` folder
as the .cu source. The cache decision is made purely from file mtimes: the
binary is reused only when its mtime is newer than the .cu source AND
newer than every file under `third_party/ThunderKittens/include` and
`prototype`. Different GPU arches go to different binaries (suffix is the
GPU label), so they never collide. Results (`results/` with CSV / plots /
summary.json) also live under `nv/`.
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


# ----------------------------------------------------------------------
# Project layout
# ----------------------------------------------------------------------

def get_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


_THIS_DIR = get_module_dir()                                          # .../example48_.../nv
_EXAMPLE_DIR = os.path.dirname(_THIS_DIR)                             # .../example48_...
_DATASETS_DIR = os.path.dirname(_EXAMPLE_DIR)                         # .../datasets
_BUILD_DIR = _THIS_DIR                                                # binary + stamp live next to the .cu source
_TK_ROOT = os.path.join(_DATASETS_DIR, "third_party", "ThunderKittens")
_TK_INCLUDE = os.path.join(_TK_ROOT, "include")
_TK_PROTOTYPE = os.path.join(_TK_ROOT, "prototype")

_FALLBACK_ARCH = "sm_103a"         # used only if nvidia-smi auto-detect fails
_DEFAULT_TIMEOUT_SEC = 900


# ----------------------------------------------------------------------
# Result types (signatures pinned by datasets/readme.md)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Compiler / arch resolution
# ----------------------------------------------------------------------

def _detect_compiler(compiler: Optional[str], platform: Optional[str] = None) -> tuple:
    if compiler:
        return compiler, "cuda"
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc, "cuda"
    for cand in ("/usr/local/cuda/bin/nvcc",
                 "/usr/local/cuda-13/bin/nvcc",
                 "/usr/local/cuda-12/bin/nvcc"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand, "cuda"
    raise RuntimeError("nvcc not found. Set --compiler or add nvcc to PATH.")


_ARCH_TO_GPU = {
    "sm_103a": ("B300", "-DKITTENS_BLACKWELL"),
    "sm_100a": ("B200", "-DKITTENS_BLACKWELL"),
    "sm_90a":  ("H100", "-DKITTENS_HOPPER"),
    "sm_80":   ("A100", "-DKITTENS_AMPERE"),
}

# compute_cap (e.g. "10.3") → kittens arch key. Hopper+ needs the `a` suffix
# to enable arch-specific PTX (multimem, wgmma, …) used in this kernel.
_COMPUTE_CAP_TO_ARCH = {
    "10.3": "sm_103a",   # B300
    "10.0": "sm_100a",   # B200
    "9.0":  "sm_90a",    # H100 / H200
    "8.0":  "sm_80",     # A100
}


def _autodetect_arch() -> Optional[str]:
    """Query nvidia-smi for the compute capability of GPU 0 and map it to one
    of the keys in `_ARCH_TO_GPU`. Returns None if detection fails or the
    GPU isn't supported by ThunderKittens here."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    for line in out.splitlines():
        cap = line.strip()
        if cap in _COMPUTE_CAP_TO_ARCH:
            return _COMPUTE_CAP_TO_ARCH[cap]
    return None


def _resolve_arch(arch: Optional[str]) -> tuple:
    if arch is None or not arch.strip() or arch.strip().lower() == "auto":
        a = _autodetect_arch() or _FALLBACK_ARCH
    else:
        a = arch.strip()
    if a not in _ARCH_TO_GPU:
        raise RuntimeError(
            f"Unsupported --arch {a!r}. Supported: {sorted(_ARCH_TO_GPU.keys())}"
        )
    gpu, define = _ARCH_TO_GPU[a]
    compute = a.replace("sm_", "compute_")
    return gpu, define, f"-gencode=arch={compute},code={a}"


# ----------------------------------------------------------------------
# Build flags & cache
# ----------------------------------------------------------------------

def _nvccflags(gencode: str, kittens_define: str) -> List[str]:
    """Same backbone as TK common.mk's standalone config (no PyTorch/pybind)."""
    return [
        "-std=c++20", "-O3", "--use_fast_math",
        "-lrt", "-lpthread", "-ldl", "-lcuda", "-lcudart",
        "--expt-extended-lambda", "--expt-relaxed-constexpr",
        "-forward-unknown-to-host-compiler",
        "-Xcompiler=-Wno-psabi", "-Xcompiler=-fno-strict-aliasing",
        "-Xnvlink=--verbose", "-Xptxas=--verbose", "-Xptxas=--warn-on-spills",
        f"-I{_TK_INCLUDE}", f"-I{_TK_PROTOTYPE}",
        "-DNDEBUG", "-lineinfo", "-ftemplate-backtrace-limit=0",
        kittens_define, gencode,
    ]


def _binary_path_for(source_abs: str, gpu_label: str) -> str:
    stem = os.path.splitext(os.path.basename(source_abs))[0]
    os.makedirs(_BUILD_DIR, exist_ok=True)
    return os.path.join(_BUILD_DIR, f"{stem}_{gpu_label}")


def _latest_mtime_in(*dirs: str) -> float:
    """Most recent mtime among every regular file under each `dirs` entry."""
    latest = 0.0
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for name in files:
                try:
                    m = os.path.getmtime(os.path.join(root, name))
                except OSError:
                    continue
                if m > latest:
                    latest = m
    return latest


def _cache_is_fresh(source_abs: str, binary_path: str) -> bool:
    """Reuse the cached binary iff it exists and is newer than both the .cu
    source and every file under the ThunderKittens include/prototype dirs."""
    if not os.path.isfile(binary_path):
        return False
    bin_mtime = os.path.getmtime(binary_path)
    if bin_mtime < os.path.getmtime(source_abs):
        return False
    if bin_mtime < _latest_mtime_in(_TK_INCLUDE, _TK_PROTOTYPE):
        return False
    return True


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------

def build(
    source_file: str,
    output_file: Optional[str] = None,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    verbose: bool = True,
) -> BuildResult:
    src_abs = source_file if os.path.isabs(source_file) else os.path.join(_THIS_DIR, source_file)
    if not os.path.isfile(src_abs):
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file or "",
            return_code=-1, stdout="", stderr="", command=[],
            error_message=f"Source file '{source_file}' not found",
        )

    try:
        nvcc, _ = _detect_compiler(compiler, platform)
    except RuntimeError as e:
        return BuildResult(
            success=False, source_file=source_file, output_file=output_file or "",
            return_code=-1, stdout="", stderr="", command=[],
            error_message=str(e),
        )

    gpu_label, kittens_define, gencode = _resolve_arch(arch)
    if verbose and (arch is None or not arch.strip() or arch.strip().lower() == "auto"):
        print(f"[build] auto-detected GPU: {gpu_label}")
    binary_path = _binary_path_for(src_abs, gpu_label)
    if output_file:
        out_name = os.path.basename(output_file)
        binary_path = os.path.join(_BUILD_DIR, out_name)
        os.makedirs(_BUILD_DIR, exist_ok=True)

    if _cache_is_fresh(src_abs, binary_path):
        if verbose:
            print(f"[build] cache hit: {binary_path} (source unchanged for {gpu_label})")
        return BuildResult(
            success=True, source_file=source_file, output_file=binary_path,
            return_code=0, stdout="", stderr="", command=[], cached=True,
        )

    flags = _nvccflags(gencode, kittens_define)
    if debug:
        flags = [f for f in flags if f != "-O3"] + ["-O0", "-g"]
    cmd = [nvcc, src_abs] + flags + ["-o", binary_path]

    if verbose:
        print(f"[build] {' '.join(cmd)}")

    env = os.environ.copy()
    # Don't let conda's cross-toolchain wrappers (CC/CXX/LD/...) capture
    # the native CUDA compile.
    for k in ("LD", "LDFLAGS", "CC", "CXX", "NVCC_PREPEND_FLAGS"):
        env.pop(k, None)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if verbose and proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    return BuildResult(
        success=proc.returncode == 0,
        source_file=source_file,
        output_file=binary_path,
        return_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        command=cmd,
        cached=False,
        error_message=None if proc.returncode == 0 else "Compilation failed",
    )


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def run(executable: str, verbose: bool = True) -> RunResult:
    exe_abs = executable if os.path.isabs(executable) else os.path.join(_THIS_DIR, executable)
    if not os.path.isfile(exe_abs):
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=[],
            error_message=f"Executable '{executable}' not found",
        )

    cmd = [exe_abs]
    if verbose:
        print(f"[run] {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=_DEFAULT_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as e:
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout=e.stdout or "", stderr=e.stderr or "", command=cmd,
            error_message=f"Run timed out after {_DEFAULT_TIMEOUT_SEC}s",
        )

    if verbose and proc.stderr and proc.returncode != 0:
        print(proc.stderr, end="", file=sys.stderr)

    parsed = _parse_json_output(proc.stdout) if proc.returncode == 0 else {}
    return RunResult(
        success=proc.returncode == 0,
        executable=executable,
        return_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        command=cmd,
        parsed_output=parsed,
        error_message=None if proc.returncode == 0 else "Execution failed",
    )


def _parse_json_output(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    continue
    return {}


# ----------------------------------------------------------------------
# build_and_run
# ----------------------------------------------------------------------

def build_and_run(
    source_file: str,
    output_file: Optional[str] = None,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    build_only: bool = False,
    run_only: bool = False,
    verbose: bool = True,
) -> BuildAndRunResult:
    build_result: Optional[BuildResult] = None
    run_result: Optional[RunResult] = None

    if not run_only:
        build_result = build(source_file=source_file, output_file=output_file,
                             compiler=compiler, platform=platform,
                             debug=debug, arch=arch, verbose=verbose)
        if not build_result.success:
            return BuildAndRunResult(build_result=build_result, run_result=None)

    if not build_only:
        if build_result is not None:
            binary_path = build_result.output_file
        else:
            gpu_label, _, _ = _resolve_arch(arch)
            src_abs = source_file if os.path.isabs(source_file) else os.path.join(_THIS_DIR, source_file)
            binary_path = _binary_path_for(src_abs, gpu_label)
        run_result = run(executable=binary_path, verbose=verbose)

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


# ----------------------------------------------------------------------
# Comparison + plotting
# ----------------------------------------------------------------------

def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str) -> None:
    if not metrics:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(metrics)


def _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label, results_dir,
                     data_size_unit="MB", latency_unit="ms", throughput_unit="GB/s") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots.")
        return

    os.makedirs(results_dir, exist_ok=True)

    def _series(metrics, key):
        return [m["data_size"] for m in metrics], [m[key] for m in metrics]

    rs, rl = _series(ref_metrics, "latency_avg")
    gs, gl = _series(gen_metrics, "latency_avg")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, rl, marker="o", label=ref_label, linewidth=2)
    ax.plot(gs, gl, marker="s", label=gen_label, linewidth=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"Data Size ({data_size_unit})")
    ax.set_ylabel(f"Latency ({latency_unit})")
    ax.set_title("Latency vs Data Size")
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(results_dir, "latency_comparison.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved {p}")

    rs, rt = _series(ref_metrics, "throughput_avg")
    gs, gt = _series(gen_metrics, "throughput_avg")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, rt, marker="o", label=ref_label, linewidth=2)
    ax.plot(gs, gt, marker="s", label=gen_label, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(f"Data Size ({data_size_unit})")
    ax.set_ylabel(f"Throughput ({throughput_unit})")
    ax.set_title("Throughput vs Data Size")
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(results_dir, "throughput_comparison.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved {p}")


def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "data_size_avg": sum(m.get("data_size", 0.0) for m in metrics) / n,
        "latency_avg": sum(m.get("latency_avg", 0.0) for m in metrics) / n,
        "throughput": sum(m.get("throughput_avg", 0.0) for m in metrics) / n,
    }


def compare(
    src_ref: str,
    src_gen: str,
    results_dir: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    no_plot: bool = False,
    verbose: bool = True,
    show_raw_output: bool = False,
) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)
    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]
    inner_verbose = show_raw_output

    if verbose:
        print(f"\n{'='*60}\n[compare] Building & running: {src_ref} ({ref_label})\n{'='*60}")
    ref_res = build_and_run(source_file=src_ref, compiler=compiler,
                            platform=platform, debug=debug, arch=arch,
                            verbose=inner_verbose)
    if verbose:
        print(f"\n{'='*60}\n[compare] Building & running: {src_gen} ({gen_label})\n{'='*60}")
    gen_res = build_and_run(source_file=src_gen, compiler=compiler,
                            platform=platform, debug=debug, arch=arch,
                            verbose=inner_verbose)

    ref_compile_ok = ref_res.build_result is not None and ref_res.build_result.success
    ref_run_ok = ref_res.run_result is not None and ref_res.run_result.success
    gen_compile_ok = gen_res.build_result is not None and gen_res.build_result.success
    gen_run_ok = gen_res.run_result is not None and gen_res.run_result.success

    ref_parsed = ref_res.run_result.parsed_output if ref_run_ok else {}
    gen_parsed = gen_res.run_result.parsed_output if gen_run_ok else {}
    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])

    data_size_unit = ref_parsed.get("data_size_unit", gen_parsed.get("data_size_unit", "MB"))
    latency_unit = ref_parsed.get("latency_unit", gen_parsed.get("latency_unit", "ms"))
    throughput_unit = ref_parsed.get("throughput_unit", gen_parsed.get("throughput_unit", "GB/s"))

    if ref_metrics:
        _save_metrics_csv(ref_metrics, os.path.join(results_dir, f"{ref_label}_metrics.csv"))
    if gen_metrics:
        _save_metrics_csv(gen_metrics, os.path.join(results_dir, f"{gen_label}_metrics.csv"))

    summary: Dict[str, Any] = {
        "generated_source": os.path.basename(src_gen),
        "ref_source": os.path.basename(src_ref),
        "model": "",
        "pass_iteration": 1,
    }

    has_perf = bool(ref_metrics) and bool(gen_metrics)
    if has_perf:
        summary["improvement_iteration"] = 1
        summary["data_size_unit"] = data_size_unit
        summary["latency_unit"] = latency_unit
        summary["throughput_unit"] = throughput_unit
        ref_avg = _metrics_avg(ref_metrics)
        gen_avg = _metrics_avg(gen_metrics)
        summary["metrics_comparison"] = {
            "ref": {"compile_success": ref_compile_ok, "run_success": ref_run_ok, **ref_avg},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok, **gen_avg},
        }
        if ref_avg.get("latency_avg", 0) != 0:
            summary["latency_improvement_pct"] = round(
                (ref_avg["latency_avg"] - gen_avg["latency_avg"]) / ref_avg["latency_avg"] * 100, 2)
        else:
            summary["latency_improvement_pct"] = 0.0
        if ref_avg.get("throughput", 0) != 0:
            summary["throughput_improvement_pct"] = round(
                (gen_avg["throughput"] - ref_avg["throughput"]) / ref_avg["throughput"] * 100, 2)
        else:
            summary["throughput_improvement_pct"] = 0.0

        lat_imp = summary["latency_improvement_pct"]
        thr_imp = summary["throughput_improvement_pct"]
        if abs(lat_imp) < 5 and abs(thr_imp) < 5:
            summary["performance"] = "same"
        elif lat_imp > 0 and thr_imp > 0:
            summary["performance"] = "better"
        elif lat_imp < -5 or thr_imp < -5:
            summary["performance"] = "worse"
        else:
            summary["performance"] = "same"

        if not no_plot:
            _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label,
                             results_dir, data_size_unit, latency_unit, throughput_unit)

        if verbose:
            print(f"\nPERFORMANCE COMPARISON (averaged over "
                  f"{len(ref_metrics)} {ref_label} / {len(gen_metrics)} {gen_label} records)")
            for metric, ref_val, gen_val, unit, lower_is_better in (
                ("data_size_avg", ref_avg["data_size_avg"], gen_avg["data_size_avg"], data_size_unit, False),
                ("throughput", ref_avg["throughput"], gen_avg["throughput"], throughput_unit, False),
                ("latency_avg", ref_avg["latency_avg"], gen_avg["latency_avg"], latency_unit, True),
            ):
                if ref_val == 0:
                    continue
                imp = ((ref_val - gen_val) / ref_val * 100) if lower_is_better \
                    else ((gen_val - ref_val) / ref_val * 100)
                ratio = gen_val / ref_val
                better = (gen_val <= ref_val) if lower_is_better else (gen_val >= ref_val)
                flag = "+" if better else "-"
                print(f"  [{flag}] {metric}: {gen_val:.4f} {unit} vs ref {ref_val:.4f} {unit} "
                      f"(ratio: {ratio:.2f}, {imp:+.1f}%)")
            print(f"  Performance: {summary['performance']}")
    else:
        summary["metrics_comparison"] = {
            "ref": {"compile_success": ref_compile_ok, "run_success": ref_run_ok},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok},
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


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build and run the standalone ThunderKittens BF16 multi-GPU all_reduce benchmark.")
    p.add_argument("--source", "-s", default="ref_thunderkitten_all_reduce.cu",
                   help="Source .cu file (default: ref_thunderkitten_all_reduce.cu)")
    p.add_argument("--output", "-o", default=None,
                   help="Output binary name placed alongside the .cu source "
                        "in the nv/ folder. Defaults to <source_stem>_<gpu>")
    p.add_argument("--arch", "-a", default=None,
                   help=f"GPU arch (default: auto-detect via nvidia-smi, "
                        f"falling back to {_FALLBACK_ARCH}; supported: "
                        f"{sorted(_ARCH_TO_GPU.keys())})")
    p.add_argument("--build-only", "-b", action="store_true", help="Compile only")
    p.add_argument("--run-only", "-r", action="store_true",
                   help="Run only (expects existing binary in the nv/ folder)")
    p.add_argument("--compiler", "-c", default=None,
                   help="Path to nvcc (default: auto-detect)")
    p.add_argument("--platform", default="cuda", choices=["cuda"],
                   help="Force platform; only `cuda` is supported here")
    p.add_argument("--plot", action="store_true",
                   help="Save latency/throughput plots after a single-source run")
    p.add_argument("--results-dir", default=None,
                   help="Directory for CSV / plots / summary.json (default: ./results)")
    p.add_argument("--compare", nargs=2, metavar=("SRC_REF", "SRC_GEN"),
                   help="Compare reference vs. generated source")
    p.add_argument("--debug", action="store_true", help="Enable debug build")
    p.add_argument("--show-raw-output", action="store_true",
                   help="In compare mode, also echo raw stdout/stderr from each run")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    p.add_argument(
        "--legacy-perf-verdict",
        action="store_true",
        help="Use this example's local verdict logic instead of the shared "
             "4-tier scheme in run_eval/perf_verdict.py."
    )
    args = p.parse_args()

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
    results_dir = args.results_dir or os.path.join(_THIS_DIR, "results")

    if args.compare:
        src_ref, src_gen = args.compare
        compare(src_ref=src_ref, src_gen=src_gen, results_dir=results_dir,
                compiler=args.compiler, platform=args.platform,
                debug=args.debug, arch=args.arch,
                verbose=verbose, show_raw_output=args.show_raw_output)
        sys.exit(0)

    result = build_and_run(
        source_file=args.source, output_file=args.output,
        compiler=args.compiler, platform=args.platform,
        debug=args.debug, arch=args.arch,
        build_only=args.build_only, run_only=args.run_only,
        verbose=verbose,
    )

    if args.plot and result.run_result and result.run_result.parsed_output:
        parsed = result.run_result.parsed_output
        metrics = parsed.get("metrics", [])
        if metrics:
            label = os.path.splitext(os.path.basename(args.source))[0]
            os.makedirs(results_dir, exist_ok=True)
            _save_metrics_csv(metrics, os.path.join(results_dir, f"{label}_metrics.csv"))
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                sizes = [m["data_size"] for m in metrics]
                lat = [m["latency_avg"] for m in metrics]
                thr = [m["throughput_avg"] for m in metrics]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(sizes, lat, marker="o", label=label, linewidth=2)
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlabel(f"Data Size ({parsed.get('data_size_unit', 'MB')})")
                ax.set_ylabel(f"Latency ({parsed.get('latency_unit', 'ms')})")
                ax.set_title("Latency vs Data Size"); ax.legend()
                ax.grid(True, ls="--", alpha=0.5); fig.tight_layout()
                fig.savefig(os.path.join(results_dir, "latency.png"), dpi=150); plt.close(fig)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(sizes, thr, marker="o", label=label, linewidth=2)
                ax.set_xscale("log")
                ax.set_xlabel(f"Data Size ({parsed.get('data_size_unit', 'MB')})")
                ax.set_ylabel(f"Throughput ({parsed.get('throughput_unit', 'GB/s')})")
                ax.set_title("Throughput vs Data Size"); ax.legend()
                ax.grid(True, ls="--", alpha=0.5); fig.tight_layout()
                fig.savefig(os.path.join(results_dir, "throughput.png"), dpi=150); plt.close(fig)
            except ImportError:
                pass

    if result.run_result is not None and result.run_result.success:
        if not verbose:
            print(json.dumps(result.run_result.parsed_output))
        else:
            print(json.dumps(result.run_result.parsed_output, indent=2))

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
