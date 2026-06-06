#!/usr/bin/env python3
"""
Build/run/compare helper for example39_mscclpp_mc_get_easy.

Intended usage:

# Build and run reference
python build_and_run.py --source ref.cu

# Compare reference vs generated
python build_and_run.py --compare ref.cu generated.cu

# Build only
python build_and_run.py --source ref.cu --build-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


DEFAULT_SOURCE = "ref_mscclpp_memorychannel_get.cu"
DEFAULT_ARCH = "sm_100a"
DEFAULT_GPUS = 2
DEFAULT_TIMEOUT_SEC = 900


@dataclass
class BuildResult:
    success: bool
    source: Path
    output: Path
    command: list[str]
    returncode: int


@dataclass
class RunResult:
    success: bool
    executable: Path
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    parsed_json: dict[str, Any] = field(default_factory=dict)


def module_dir() -> Path:
    return Path(__file__).resolve().parent


def datasets_dir() -> Path:
    return module_dir().parent


def build_dir() -> Path:
    path = module_dir() / ".build"
    path.mkdir(exist_ok=True)
    return path


def default_nvcc() -> str:
    conda_nvcc = Path("/home/uccl/miniconda3/bin/nvcc")
    if conda_nvcc.exists():
        return str(conda_nvcc)
    return "nvcc"


def resolve_source(source: str) -> Path:
    path = Path(source)
    if not path.is_absolute():
        path = module_dir() / path
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    return path.resolve()


def default_output_for(source: Path, suffix: str = "") -> Path:
    name = source.stem + suffix
    return build_dir() / name


def output_path(args: argparse.Namespace, source: Path) -> Path:
    if args.output:
        path = Path(args.output)
        if not path.is_absolute():
            path = module_dir() / path
        return path.resolve()
    return (module_dir() / source.stem).resolve()


def compile_command(source: Path, output: Path, args: argparse.Namespace) -> list[str]:
    root = module_dir()
    ds = datasets_dir()
    mscclpp_prefix = ds / "build_mscclpp"
    conda_root = Path("/home/uccl/miniconda3")

    return [
        args.nvcc,
        "-std=c++17",
        "-x",
        "cu",
        "-O3",
        "-DMSCCLPP_FORCE_DISABLE_NVLS=1",
        "-ccbin",
        "/usr/bin/g++",
        "--compiler-options",
        "-B/usr/bin",
        f"-arch={args.arch}",
        "-I",
        str(root),
        "-I",
        str(mscclpp_prefix / "include"),
        "-I",
        str(ds / "third_party" / "mscclpp" / "test" / "mscclpp-test"),
        "-L",
        str(mscclpp_prefix / "lib"),
        "-Xlinker",
        f"-rpath={mscclpp_prefix / 'lib'}",
        "-L",
        str(conda_root / "targets" / "x86_64-linux" / "lib"),
        "-Xlinker",
        f"-rpath={conda_root / 'targets' / 'x86_64-linux' / 'lib'}",
        "-L",
        str(conda_root / "lib"),
        "-Xlinker",
        f"-rpath={conda_root / 'lib'}",
        "-lmscclpp",
        "-lcudart",
        "-lcuda",
        "-lnuma",
        str(source),
        "-o",
        str(output),
    ]


def print_cmd(label: str, cmd: list[str]) -> None:
    print(f"[{label}] " + " ".join(cmd), flush=True)


def build_source(source: Path, output: Path, args: argparse.Namespace) -> BuildResult:
    cmd = compile_command(source, output, args)
    print_cmd("build", cmd)
    result = subprocess.run(cmd, cwd=module_dir(), text=True)
    return BuildResult(result.returncode == 0, source, output, cmd, result.returncode)


def parse_json_output(text: str) -> dict[str, Any]:
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
                    start = None
    return {}


def run_executable(executable: Path, args: argparse.Namespace) -> RunResult:
    cmd = [str(executable), "--gpus", str(args.gpus)]
    if args.mode:
        cmd += ["--mode", args.mode]
    print_cmd("run", cmd)
    try:
        result = subprocess.run(
            cmd,
            cwd=module_dir(),
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        print(f"[run] timed out after {args.timeout}s", file=sys.stderr)
        return RunResult(False, executable, cmd, 124, stdout, stderr)

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    parsed = parse_json_output(result.stdout)
    return RunResult(result.returncode == 0, executable, cmd, result.returncode, result.stdout, result.stderr, parsed)


def build_and_run(source_name: str, output: Path, args: argparse.Namespace) -> tuple[BuildResult, RunResult | None]:
    source = resolve_source(source_name)
    build_result = build_source(source, output, args)
    if not build_result.success or args.build_only:
        return build_result, None
    return build_result, run_executable(output, args)


def metrics_by_size(run: RunResult) -> dict[int, dict[str, Any]]:
    metrics = run.parsed_json.get("metrics", [])
    out: dict[int, dict[str, Any]] = {}
    for row in metrics:
        try:
            out[int(row["data_size"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _metrics_avg(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average per-size metric records into the flat dict that
    `metrics_comparison.{ref,generated}` expects. Keys match the mscclpp
    variant in run_eval/perf_verdict.py PERF_METRICS — primary=throughput_avg."""
    if not rows:
        return {}
    n = len(rows)
    return {
        "count": float(n),
        "latency_avg": sum(float(r.get("latency_avg", 0.0)) for r in rows) / n,
        "throughput_avg": sum(float(r.get("throughput_avg", 0.0)) for r in rows) / n,
    }


def _write_summary(
    src_ref: Path,
    src_gen: Path,
    ref: RunResult,
    gen: RunResult,
    results_dir: Path,
) -> None:
    """Write summary.json with the flat `metrics_comparison.{ref,generated}`
    dict the unified perf-verdict (run_eval/perf_verdict.py) consumes.
    Verdict fields (`performance`, `performance_detail`, `verdict_scheme`)
    are filled in on-exit by the atexit hook installed in main()."""
    ref_run_ok = ref.success
    gen_run_ok = gen.success
    ref_json = ref.parsed_json if ref_run_ok else {}
    gen_json = gen.parsed_json if gen_run_ok else {}
    ref_rows = list(metrics_by_size(ref).values()) if ref_run_ok else []
    gen_rows = list(metrics_by_size(gen).values()) if gen_run_ok else []
    ref_avg = _metrics_avg(ref_rows)
    gen_avg = _metrics_avg(gen_rows)
    summary = {
        "generated_source": src_gen.name,
        "ref_source": src_ref.name,
        "model": "",
        "pass_iteration": 1,
        "improvement_iteration": 1,
        "data_size_unit": ref_json.get("data_size_unit", gen_json.get("data_size_unit", "MiB")),
        "latency_unit": ref_json.get("latency_unit", gen_json.get("latency_unit", "us")),
        "throughput_unit": ref_json.get("throughput_unit", gen_json.get("throughput_unit", "GB/s")),
        "metrics_comparison": {
            "ref": {"compile_success": True, "run_success": ref_run_ok, **ref_avg},
            "generated": {"compile_success": True, "run_success": gen_run_ok, **gen_avg},
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def compare_runs(
    ref: RunResult, gen: RunResult, args: argparse.Namespace,
    src_ref: Path, src_gen: Path, results_dir: Path,
) -> bool:
    ok = ref.success and gen.success
    ref_json = ref.parsed_json
    gen_json = gen.parsed_json

    if not ref_json or not gen_json:
        print("[compare] missing JSON output from reference or generated run", file=sys.stderr)
        _write_summary(src_ref, src_gen, ref, gen, results_dir)
        return False

    ref_correct = ref_json.get("Correctness") == "PASS"
    gen_correct = gen_json.get("Correctness") == "PASS"
    ok = ok and ref_correct and gen_correct

    ref_metrics = metrics_by_size(ref)
    gen_metrics = metrics_by_size(gen)
    common_sizes = sorted(set(ref_metrics) & set(gen_metrics))
    if not common_sizes:
        print("[compare] no common metric sizes found", file=sys.stderr)
        _write_summary(src_ref, src_gen, ref, gen, results_dir)
        return False

    print("[compare] data_size MiB | ref GB/s | gen GB/s | ratio")
    min_ratio = float("inf")
    for size in common_sizes:
        ref_tput = float(ref_metrics[size].get("throughput_avg", 0.0))
        gen_tput = float(gen_metrics[size].get("throughput_avg", 0.0))
        ratio = gen_tput / ref_tput if ref_tput > 0.0 else 0.0
        min_ratio = min(min_ratio, ratio)
        print(f"[compare] {size:13d} | {ref_tput:8.3f} | {gen_tput:8.3f} | {ratio:5.3f}")

    if args.min_throughput_ratio is not None and min_ratio < args.min_throughput_ratio:
        print(
            f"[compare] generated throughput ratio {min_ratio:.3f} below "
            f"threshold {args.min_throughput_ratio:.3f}",
            file=sys.stderr,
        )
        ok = False

    _write_summary(src_ref, src_gen, ref, gen, results_dir)
    print(f"[compare] {'PASS' if ok else 'FAIL'}")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source", default=None, help="CUDA source file to build/run")
    mode.add_argument("--compare", nargs=2, metavar=("REF", "GENERATED"), help="Build/run and compare two sources")
    parser.add_argument("--output", default=None, help="Output executable for --source mode")
    parser.add_argument("--arch", default=DEFAULT_ARCH, help="CUDA GPU architecture, e.g. sm_100a")
    parser.add_argument("--gpus", type=int, default=DEFAULT_GPUS, help="Number of GPUs passed to executable")
    parser.add_argument("--nvcc", default=default_nvcc(), help="nvcc executable")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC, help="Run timeout in seconds")
    parser.add_argument("--mode", default=None, help="Optional --mode argument forwarded to executable")
    parser.add_argument("--build-only", action="store_true", help="Compile but do not run")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for summary.json (defaults to <example>/results).",
    )
    parser.add_argument(
        "--min-throughput-ratio",
        type=float,
        default=None,
        help="Optional generated/reference throughput ratio threshold for --compare",
    )
    parser.add_argument(
        "--legacy-perf-verdict",
        action="store_true",
        help="Use this example's local verdict logic instead of the shared "
             "4-tier scheme in run_eval/perf_verdict.py."
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()

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


    try:
        if args.compare:
            if args.output:
                print("--output is only valid with --source mode", file=sys.stderr)
                return 2
            ref_source = resolve_source(args.compare[0])
            gen_source = resolve_source(args.compare[1])
            ref_output = default_output_for(ref_source, "_ref")
            gen_output = default_output_for(gen_source, "_gen")

            ref_build, ref_run = build_and_run(str(ref_source), ref_output, args)
            if not ref_build.success:
                return ref_build.returncode or 1
            gen_build, gen_run = build_and_run(str(gen_source), gen_output, args)
            if not gen_build.success:
                return gen_build.returncode or 1
            if args.build_only:
                return 0
            if ref_run is None or gen_run is None:
                return 1
            results_dir = Path(args.results_dir) if args.results_dir else module_dir() / "results"
            return 0 if compare_runs(ref_run, gen_run, args, ref_source, gen_source, results_dir) else 1

        source_name = args.source or DEFAULT_SOURCE
        source = resolve_source(source_name)
        output = output_path(args, source)
        build_result, run_result = build_and_run(str(source), output, args)
        if not build_result.success:
            return build_result.returncode or 1
        if args.build_only:
            return 0
        if run_result is None:
            return 1
        return 0 if run_result.success else (run_result.returncode or 1)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
