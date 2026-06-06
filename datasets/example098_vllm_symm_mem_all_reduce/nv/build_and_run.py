#!/usr/bin/env python3
"""
vLLM SymmMemAllReduce Build and Run Module

Runs the vLLM SymmMemCommunicator reference / generated Python program
via `torchrun --nproc_per_node=N`.  Python scripts require no compilation;
build() is a no-op.

Pre-flight:
    The interpreter that runs the example must have vLLM importable.  Use
    the vendored installer once per environment:

        bash datasets/third_party/vllm_setup.sh

    `build_and_run.py` itself does NOT install or compile anything on the
    fast path.

Hardware floor:
    SymmMemCommunicator only initializes on CUDA hosts whose device
    capability is in SYMM_MEM_ALL_REDUCE_MAX_SIZES (currently 9.0, 10.0,
    10.3) AND whose torch build exposes torch.distributed._symmetric_memory
    with a multicast-capable rendezvous handle. On any other platform the
    reference exits cleanly with `Correctness: SKIPPED`.

Environment variables:
    LLM_GPU_COMM_NPROC   number of ranks to launch (default: 2; valid
                         counts depend on capability — see
                         vllm/distributed/device_communicators/
                         all_reduce_utils.py SYMM_MEM_ALL_REDUCE_MAX_SIZES).
    LLM_GPU_COMM_PYTHON  python interpreter to launch torchrun from
                         (default: the interpreter running this script).

Usage as module:
    from build_and_run import build, run, build_and_run, compare
"""

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


_DEFAULT_SOURCE = "ref_vllm_symm_mem_all_reduce.py"
_NPROC = int(os.environ.get("LLM_GPU_COMM_NPROC", "2"))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

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
    parsed_output: Dict[str, Any] = field(default_factory=dict)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_module_dir() -> str:
    """Return directory of this script."""
    return os.path.dirname(os.path.abspath(__file__))


def _python_interpreter() -> str:
    """Resolve the python used to launch torchrun.

    Priority:
        $LLM_GPU_COMM_PYTHON  >  sys.executable
    """
    return os.environ.get("LLM_GPU_COMM_PYTHON") or sys.executable


def _torchrun_cmd(python: str) -> List[str]:
    """Build a torchrun-equivalent command bound to the given python.

    We deliberately do NOT use `shutil.which("torchrun")`: it may resolve to
    an interpreter that doesn't have vLLM installed.  Going through
    `<python> -m torch.distributed.run` keeps the launcher and the workers
    in the same env.
    """
    return [python, "-m", "torch.distributed.run"]


def _check_vllm_importable(python: str, verbose: bool) -> Optional[str]:
    """Return None if vLLM is importable from `python`, else an error string."""
    try:
        # Run from /tmp so a local `vllm/` directory cannot shadow the install
        # as a namespace package (this can happen on developer checkouts).
        proc = subprocess.run(
            [python, "-c", "import vllm.distributed.device_communicators."
                           "symm_mem as m; "
                           "assert hasattr(m, 'SymmMemCommunicator')"],
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return f"vLLM import probe failed: {exc}"
    if proc.returncode == 0:
        return None
    msg = (proc.stderr or proc.stdout or "").strip()
    return (
        "vLLM is not importable from the chosen python interpreter "
        f"({python}). Run `bash datasets/third_party/vllm_setup.sh` once to "
        f"install it, then retry. Underlying error:\n{msg}"
    )


def _parse_json_output(text: str) -> Dict[str, Any]:
    """Extract the first complete JSON object from stdout."""
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


def _save_metrics_csv(metrics: List[Dict[str, Any]], path: str) -> None:
    if not metrics:
        return
    keys = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def _metrics_avg(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "data_size_avg": sum(m.get("data_size", 0) for m in metrics) / n,
        "latency_avg": sum(m.get("latency_avg", 0) for m in metrics) / n,
        "throughput": sum(m.get("throughput_avg", 0) for m in metrics) / n,
    }


def _plot_comparison(ref_metrics, gen_metrics, ref_label, gen_label, results_dir,
                     data_size_unit="MB", latency_unit="us", throughput_unit="Gbps"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    os.makedirs(results_dir, exist_ok=True)

    ref_sizes = [m["data_size"] for m in ref_metrics]
    ref_lat = [m["latency_avg"] for m in ref_metrics]
    ref_thr = [m["throughput_avg"] for m in ref_metrics]
    gen_sizes = [m["data_size"] for m in gen_metrics]
    gen_lat = [m["latency_avg"] for m in gen_metrics]
    gen_thr = [m["throughput_avg"] for m in gen_metrics]

    for (y_ref, y_gen, ylabel, unit, fname) in [
        (ref_lat, gen_lat, "Latency", latency_unit, "latency_comparison.png"),
        (ref_thr, gen_thr, "Throughput", throughput_unit, "throughput_comparison.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ref_sizes, y_ref, marker="o", label=ref_label, linewidth=2)
        ax.plot(gen_sizes, y_gen, marker="s", label=gen_label, linewidth=2)
        ax.set_xlabel(f"Data Size ({data_size_unit})")
        ax.set_ylabel(f"{ylabel} ({unit})")
        ax.set_title(f"{ylabel} vs Data Size")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout()
        out = os.path.join(results_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


def _plot_single(metrics, label, results_dir,
                 data_size_unit="MB", latency_unit="us", throughput_unit="Gbps"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    os.makedirs(results_dir, exist_ok=True)
    sizes = [m["data_size"] for m in metrics]

    for (values, ylabel, unit, fname) in [
        ([m["latency_avg"] for m in metrics], "Latency", latency_unit, "latency.png"),
        ([m["throughput_avg"] for m in metrics], "Throughput", throughput_unit, "throughput.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sizes, values, marker="o", label=label, linewidth=2)
        ax.set_xlabel(f"Data Size ({data_size_unit})")
        ax.set_ylabel(f"{ylabel} ({unit})")
        ax.set_title(f"{ylabel} vs Data Size")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout()
        out = os.path.join(results_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def build(
    source_file: str,
    output_file: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    verbose: bool = True,
) -> BuildResult:
    """No-op for Python scripts — always returns success.

    The vLLM C++/CUDA artifacts are built once via
    `datasets/third_party/vllm_setup.sh` and installed editable; nothing is
    rebuilt here on the fast path.
    """
    if verbose:
        print(f"[build] Python script '{source_file}' — no compilation required.")
    return BuildResult(
        success=True,
        source_file=source_file,
        output_file=source_file,
        return_code=0,
        stdout="",
        stderr="",
        command=[],
    )


def run(executable: str, verbose: bool = True) -> RunResult:
    """
    Run a Python script via `<python> -m torch.distributed.run --nproc_per_node=N`.
    """
    wd = get_module_dir()
    script = executable if os.path.isabs(executable) else os.path.join(wd, executable)

    if not os.path.exists(script):
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=[],
            error_message=f"Script '{executable}' not found",
        )

    python = _python_interpreter()
    err = _check_vllm_importable(python, verbose=verbose)
    if err is not None:
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr=err, command=[python, "-c", "import vllm"],
            error_message=err,
        )

    cmd = _torchrun_cmd(python) + [f"--nproc_per_node={_NPROC}", script]

    if verbose:
        print(f"[run] Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except FileNotFoundError as exc:
        return RunResult(
            success=False, executable=executable, return_code=-1,
            stdout="", stderr="", command=cmd,
            error_message=f"torchrun not available: {exc}",
        )

    if verbose:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

    success = result.returncode == 0
    parsed = _parse_json_output(result.stdout) if success else {}

    return RunResult(
        success=success,
        executable=executable,
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=cmd,
        parsed_output=parsed,
        error_message=None if success else "Execution failed",
    )


def build_and_run(
    source_file: str,
    output_file: str,
    compiler: Optional[str] = None,
    platform: Optional[str] = None,
    debug: bool = False,
    arch: Optional[str] = None,
    build_only: bool = False,
    run_only: bool = False,
    verbose: bool = True,
) -> BuildAndRunResult:
    """Build (no-op) and run a Python distributed script."""
    build_result = None
    run_result = None

    if not run_only:
        build_result = build(
            source_file=source_file, output_file=output_file,
            compiler=compiler, platform=platform,
            debug=debug, arch=arch, verbose=verbose,
        )
        if not build_result.success:
            return BuildAndRunResult(build_result=build_result, run_result=None)

    if not build_only:
        run_result = run(executable=source_file, verbose=verbose)

    return BuildAndRunResult(build_result=build_result, run_result=run_result)


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
    """
    Build and run both reference and generated scripts, then produce
    CSV, plots, and a summary JSON in results_dir.
    """
    os.makedirs(results_dir, exist_ok=True)

    ref_label = os.path.splitext(os.path.basename(src_ref))[0]
    gen_label = os.path.splitext(os.path.basename(src_gen))[0]

    def _run_one(src, label):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"[compare] Running: {src}  ({label})")
            print(f"{'=' * 60}")
        res = build_and_run(
            source_file=src, output_file=src,
            compiler=compiler, platform=platform,
            debug=debug, arch=arch,
            verbose=show_raw_output,
        )
        compile_ok = res.build_result is None or res.build_result.success
        run_ok = res.run_result is not None and res.run_result.success
        parsed = res.run_result.parsed_output if run_ok else {}
        if verbose:
            if not compile_ok:
                print(f"[compare] BUILD FAILED for {src}")
            elif not run_ok:
                print(f"[compare] RUN FAILED for {src}")
                if res.run_result:
                    print(res.run_result.stderr[-2000:], file=sys.stderr)
        return compile_ok, run_ok, parsed

    ref_compile_ok, ref_run_ok, ref_parsed = _run_one(src_ref, ref_label)
    gen_compile_ok, gen_run_ok, gen_parsed = _run_one(src_gen, gen_label)

    ref_metrics = ref_parsed.get("metrics", [])
    gen_metrics = gen_parsed.get("metrics", [])

    data_size_unit = ref_parsed.get("data_size_unit", gen_parsed.get("data_size_unit", "MB"))
    latency_unit = ref_parsed.get("latency_unit", gen_parsed.get("latency_unit", "us"))
    throughput_unit = ref_parsed.get("throughput_unit", gen_parsed.get("throughput_unit", "Gbps"))

    # Save CSVs
    for metrics, label in [(ref_metrics, ref_label), (gen_metrics, gen_label)]:
        if metrics:
            path = os.path.join(results_dir, f"{label}_metrics.csv")
            _save_metrics_csv(metrics, path)
            if verbose:
                print(f"Raw metrics saved to {path}")

    has_perf = bool(ref_metrics) and bool(gen_metrics)

    summary: Dict[str, Any] = {
        "generated_source": os.path.basename(src_gen),
        "ref_source": os.path.basename(src_ref),
        "model": "",
        "pass_iteration": 1,
    }

    if has_perf:
        ref_avg = _metrics_avg(ref_metrics)
        gen_avg = _metrics_avg(gen_metrics)

        summary["improvement_iteration"] = 1
        summary["data_size_unit"] = data_size_unit
        summary["latency_unit"] = latency_unit
        summary["throughput_unit"] = throughput_unit
        summary["metrics_comparison"] = {
            "ref": {"compile_success": ref_compile_ok, "run_success": ref_run_ok, **ref_avg},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok, **gen_avg},
        }

        lat_imp = (
            (ref_avg["latency_avg"] - gen_avg["latency_avg"]) / ref_avg["latency_avg"] * 100
            if ref_avg.get("latency_avg") else 0.0
        )
        thr_imp = (
            (gen_avg["throughput"] - ref_avg["throughput"]) / ref_avg["throughput"] * 100
            if ref_avg.get("throughput") else 0.0
        )
        summary["latency_improvement_pct"] = round(lat_imp, 2)
        summary["throughput_improvement_pct"] = round(thr_imp, 2)

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
            print(f"  [+] data_size_avg: {ref_avg['data_size_avg']:.1f} {data_size_unit}")
            flag_thr = "+" if thr_imp >= 0 else "-"
            flag_lat = "+" if lat_imp >= 0 else "-"
            print(f"  [{flag_thr}] throughput: {gen_avg['throughput']:.3f} vs ref "
                  f"{ref_avg['throughput']:.3f} {throughput_unit} ({thr_imp:+.1f}%)")
            print(f"  [{flag_lat}] latency_avg: {gen_avg['latency_avg']:.3f} vs ref "
                  f"{ref_avg['latency_avg']:.3f} {latency_unit} ({lat_imp:+.1f}%)")
            print(f"  Performance: {summary['performance']}")
    else:
        summary["metrics_comparison"] = {
            "ref": {"compile_success": ref_compile_ok, "run_success": ref_run_ok},
            "generated": {"compile_success": gen_compile_ok, "run_success": gen_run_ok},
        }
        # Surface SKIPPED state from the program output if present.
        # SymmMemCommunicator falls back to SKIPPED when the capability /
        # world size combo is unsupported or multicast isn't available.
        ref_skipped = ref_parsed.get("Correctness") == "SKIPPED"
        gen_skipped = gen_parsed.get("Correctness") == "SKIPPED"
        if ref_skipped or gen_skipped:
            if verbose:
                print(f"\nSKIPPED — SymmMemCommunicator did not initialize.")
                if ref_skipped:
                    print(f"  ref       skip_reason: {ref_parsed.get('skip_reason', '')}")
                if gen_skipped:
                    print(f"  generated skip_reason: {gen_parsed.get('skip_reason', '')}")

    # Save summary JSON
    json_path = os.path.join(results_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"Summary saved to {json_path}")
        mc = summary["metrics_comparison"]
        print(f"\n{'=' * 60}")
        print(f"  ref       compile_success: {mc['ref']['compile_success']}")
        print(f"  ref       run_success:     {mc['ref']['run_success']}")
        print(f"  generated compile_success: {mc['generated']['compile_success']}")
        print(f"  generated run_success:     {mc['generated']['run_success']}")
        print(f"  performance:               {summary.get('performance', 'N/A')}")
        print(f"{'=' * 60}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build and run the vLLM SymmMemCommunicator Python "
                    "distributed script via torchrun"
    )
    parser.add_argument("--source", "-s", default=_DEFAULT_SOURCE,
                        help=f"Python script to run (default: {_DEFAULT_SOURCE})")
    parser.add_argument("--output", "-o", default=None,
                        help="Ignored for Python scripts (kept for interface compatibility)")
    parser.add_argument("--arch", "-a", default=None,
                        help="GPU architecture — ignored for Python scripts")
    parser.add_argument("--build-only", "-b", action="store_true",
                        help="Only run the build step (no-op for Python)")
    parser.add_argument("--run-only", "-r", action="store_true",
                        help="Only run without building")
    parser.add_argument("--compiler", "-c", default=None,
                        help="Compiler path — ignored for Python scripts")
    parser.add_argument("--platform", "-p", choices=["hip", "cuda"], default=None,
                        help="Platform — ignored for Python scripts")
    parser.add_argument("--plot", action="store_true",
                        help="Generate benchmark plots after running")
    parser.add_argument("--results-dir", default=None,
                        help="Directory to save results (default: ./results)")
    parser.add_argument("--compare", nargs=2, metavar=("SRC_REF", "SRC_GEN"), default=None,
                        help="Compare reference and generated scripts")
    parser.add_argument("--debug", action="store_true", help="Debug build (no-op here)")
    parser.add_argument("--show-raw-output", action="store_true",
                        help="Print raw stdout/stderr from each run in compare mode")
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
    results_dir = args.results_dir or os.path.join(get_module_dir(), "results")

    if args.compare:
        src_ref, src_gen = args.compare
        wd = get_module_dir()
        for src in (src_ref, src_gen):
            p = src if os.path.isabs(src) else os.path.join(wd, src)
            if not os.path.exists(p):
                print(f"Error: '{src}' not found.")
                sys.exit(1)
        compare(
            src_ref=src_ref, src_gen=src_gen,
            results_dir=results_dir,
            compiler=args.compiler, platform=args.platform,
            debug=args.debug, arch=args.arch,
            verbose=verbose,
            show_raw_output=args.show_raw_output,
        )
        sys.exit(0)

    output = args.output or args.source
    result = build_and_run(
        source_file=args.source,
        output_file=output,
        compiler=args.compiler,
        platform=args.platform,
        debug=args.debug,
        arch=args.arch,
        build_only=args.build_only,
        run_only=args.run_only,
        verbose=verbose,
    )

    if args.plot and result.run_result and result.run_result.parsed_output:
        parsed = result.run_result.parsed_output
        metrics = parsed.get("metrics", [])
        if metrics:
            label = os.path.splitext(os.path.basename(args.source))[0]
            os.makedirs(results_dir, exist_ok=True)
            _plot_single(metrics, label, results_dir,
                         parsed.get("data_size_unit", "MB"),
                         parsed.get("latency_unit", "us"),
                         parsed.get("throughput_unit", "Gbps"))
            csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
            _save_metrics_csv(metrics, csv_path)
            if verbose:
                print(f"Saved {csv_path}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
