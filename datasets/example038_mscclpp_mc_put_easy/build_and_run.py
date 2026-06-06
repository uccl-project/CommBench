#!/usr/bin/env python3
"""
Build/run/compare helper for example38_mscclpp_mc_put_easy.

Examples:
  # Build and run reference
  python build_and_run.py --source ref_mscclpp_memorychannel_put.cu

  # Compare reference vs generated
  python build_and_run.py --compare ref_mscclpp_memorychannel_put.cu generated.cu

  # Build only
  python build_and_run.py --source ref_mscclpp_memorychannel_put.cu --build-only
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


DEFAULT_SOURCE = "ref_mscclpp_memorychannel_put.cu"
DEFAULT_OUTPUT = "ref_mscclpp_memorychannel_put"
DEFAULT_ARCH = "sm_100a"
DEFAULT_GPUS = 2
DEFAULT_TIMEOUT_SEC = 900


def module_dir() -> Path:
    return Path(__file__).resolve().parent


def datasets_dir() -> Path:
    return module_dir().parent


def build_dir() -> Path:
    d = module_dir() / ".build"
    d.mkdir(exist_ok=True)
    return d


def default_nvcc() -> str:
    conda_nvcc = Path("/home/uccl/miniconda3/bin/nvcc")
    if conda_nvcc.exists():
        return str(conda_nvcc)
    return "nvcc"


def resolve_source(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = module_dir() / p
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    return p.resolve()


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


def build(source: Path, output: Path, args: argparse.Namespace) -> BuildResult:
    cmd = compile_command(source, output, args)
    print("+ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=module_dir(), text=True)
    return BuildResult(completed.returncode == 0, source, output, cmd, completed.returncode)


def parse_json_output(text: str) -> dict[str, Any]:
    """Pull the first balanced top-level JSON object out of `text`."""
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


def run(executable: Path, args: argparse.Namespace) -> RunResult:
    cmd = [str(executable), "--gpus", str(args.gpus)]
    print("+ " + " ".join(cmd), flush=True)
    try:
        completed = subprocess.run(
            cmd, cwd=module_dir(), text=True,
            capture_output=True, timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        out, err = exc.stdout or "", exc.stderr or ""
        if out: print(out, end="")
        if err: print(err, end="", file=sys.stderr)
        print(f"[run] timed out after {args.timeout}s", file=sys.stderr)
        return RunResult(False, executable, cmd, 124, out, err)

    if completed.stdout: print(completed.stdout, end="")
    if completed.stderr: print(completed.stderr, end="", file=sys.stderr)
    parsed = parse_json_output(completed.stdout)
    return RunResult(completed.returncode == 0, executable, cmd,
                     completed.returncode, completed.stdout, completed.stderr, parsed)


def build_and_run(
    source_name: str, output: Path, args: argparse.Namespace,
) -> tuple[BuildResult | None, RunResult | None]:
    """Build and (if not --build-only) run a single source."""
    source = resolve_source(source_name)
    if getattr(args, "run_only", False):
        return None, run(output, args)
    build_result = build(source, output, args)
    if not build_result.success or args.build_only:
        return build_result, None
    return build_result, run(output, args)


def metrics_by_size(rr: RunResult) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rr.parsed_json.get("metrics", []):
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
    src_ref: Path, src_gen: Path,
    ref: RunResult, gen: RunResult,
    ref_build: BuildResult, gen_build: BuildResult,
    results_dir: Path,
) -> dict[str, Any]:
    """Write summary.json with `metrics_comparison.{ref,generated}` per
    datasets/readme.md. Verdict fields are filled in on-exit by the atexit
    hook installed in main()."""
    ref_run_ok = ref.success
    gen_run_ok = gen.success
    ref_compile_ok = ref_build.success
    gen_compile_ok = gen_build.success
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
            "ref":       {"compile_success": ref_compile_ok, "run_success": ref_run_ok, **ref_avg},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok, **gen_avg},
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def compare(
    src_ref: str, src_gen: str, args: argparse.Namespace, results_dir: Path,
) -> bool:
    """Build/run both sources, write summary.json, return overall pass/fail."""
    ref_source = resolve_source(src_ref)
    gen_source = resolve_source(src_gen)
    ref_output = build_dir() / (ref_source.stem + "_ref")
    gen_output = build_dir() / (gen_source.stem + "_gen")

    ref_build = build(ref_source, ref_output, args)
    if not ref_build.success:
        # Still write a (mostly-empty) summary so the unified verdict has
        # something to look at and the harness can read structured failure.
        empty = RunResult(False, ref_output, [], -1, "", "")
        _write_summary(ref_source, gen_source, empty, empty, ref_build,
                       BuildResult(False, gen_source, gen_output, [], -1),
                       results_dir)
        return False
    gen_build = build(gen_source, gen_output, args)
    if not gen_build.success:
        empty = RunResult(False, gen_output, [], -1, "", "")
        _write_summary(ref_source, gen_source, empty, empty, ref_build, gen_build, results_dir)
        return False
    if args.build_only:
        return True

    ref_run = run(ref_output, args)
    gen_run = run(gen_output, args)

    ref_correct = ref_run.parsed_json.get("Correctness") == "PASS"
    gen_correct = gen_run.parsed_json.get("Correctness") == "PASS"
    ok = ref_run.success and gen_run.success and ref_correct and gen_correct

    _write_summary(ref_source, gen_source, ref_run, gen_run, ref_build, gen_build, results_dir)
    print(f"[compare] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="CUDA source file to compile")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output executable path/name")
    parser.add_argument("--arch", default=DEFAULT_ARCH, help="CUDA GPU architecture, e.g. sm_100a")
    parser.add_argument("--gpus", type=int, default=DEFAULT_GPUS, help="Number of GPUs passed to the executable")
    parser.add_argument("--nvcc", default=default_nvcc(), help="nvcc executable")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC, help="Run timeout in seconds")
    parser.add_argument("--build-only", action="store_true", help="Compile but do not run")
    parser.add_argument("--run-only", action="store_true", help="Run existing executable without compiling")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "GENERATED"),
                        help="Build/run and compare two sources; writes summary.json with metrics_comparison")
    parser.add_argument("--results-dir", default=None,
                        help="Directory for summary.json/CSVs (defaults to <example>/results)")
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

    if args.build_only and args.run_only:
        print("--build-only and --run-only cannot be used together", file=sys.stderr)
        return 2

    try:
        if args.compare:
            results_dir = Path(args.results_dir) if args.results_dir else module_dir() / "results"
            return 0 if compare(args.compare[0], args.compare[1], args, results_dir) else 1

        # Single-source mode (compatible with original behaviour)
        source = resolve_source(args.source)
        output = Path(args.output)
        if not output.is_absolute():
            output = module_dir() / output

        if not args.run_only:
            br = build(source, output, args)
            if not br.success:
                return br.returncode or 1
        if args.build_only:
            return 0
        rr = run(output, args)
        return 0 if rr.success else (rr.returncode or 1)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
