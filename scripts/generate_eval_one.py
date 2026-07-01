#!/usr/bin/env python3
"""
Generate and Evaluate One Dataset
Automates the process of generating code from prompts and evaluating the results.
Supports both .cpp and .cu files.
Supports multi-round generation with error feedback.
"""

import os
import sys
import json
import shutil
import random
import time
import argparse
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, ContextManager

# Auto-detect project root (llm-for-gpu-comm/) from this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # scripts/ -> llm-for-gpu-comm/
_DEFAULT_DATASETS_DIR = os.path.join(_PROJECT_ROOT, "datasets")

# Add parent directory to path to import modules
sys.path.insert(0, _PROJECT_ROOT)

from prompt_llm.prompt import (
    generate_completion_prompt,
    get_file_type_description,
    generate_fix_prompt_from_code,
    generate_perf_fix_prompt_from_code,
    extract_code_from_response
)
from prompt_llm.llm_factory import generate_code, get_provider_for_model, list_supported_models
from run_eval.compile_run import (
    compile_and_run,
    compare_save_results,
    CompileRunResult,
    find_empty_file,
    find_ref_file,
    analyze_result,
    get_error_details,
    get_platform_subdir
)
from run_eval.platform_detect import detect_platform, get_platform_string, PlatformInfo
from run_eval.cheat_detect import (
    check_non_todo_modified,
    format_modified_functions,
    run_all_checks,
)


# Substrings (lower-case) that mark a transient/quota-related LLM error
# worth retrying with exponential backoff. Anything else is treated as a
# hard failure and re-raised on the first attempt.
_LLM_RETRYABLE_MARKERS = (
    "rate limit",
    "ratelimit",
    "rate_limit",
    "too many requests",
    "429",
    "quota",
    "resource exhausted",
    "resourceexhausted",
    "overloaded",
    "server is busy",
    "service unavailable",
    "503",
    "502",
    "504",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "deadline exceeded",
    "connection reset",
    "connection aborted",
    "remote disconnected",
)


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """Best-effort: classify exceptions raised by the various provider SDKs
    as rate-limit / transient (retryable) vs. hard failures."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in _LLM_RETRYABLE_MARKERS)


def _call_llm_with_retry(
    *,
    prompt: str,
    model: str,
    language: str,
    temperature: float,
    api: Optional[str],
    max_retries: int,
    backoff_base: float,
    verbose: bool,
    tag: str = "",
) -> str:
    """Call generate_code with bounded exponential backoff on rate-limit /
    transient errors. Non-retryable errors are re-raised immediately."""
    attempt = 0
    while True:
        try:
            return generate_code(
                prompt=prompt,
                model=model,
                language=language,
                temperature=temperature,
                provider=api,
            )
        except Exception as exc:  # noqa: BLE001 - SDKs raise heterogeneous types
            if attempt >= max_retries or not _is_retryable_llm_error(exc):
                raise
            sleep_s = backoff_base * (2 ** attempt) + random.uniform(0.0, 1.0)
            attempt += 1
            if verbose:
                print(f"{tag}[llm-retry] attempt {attempt}/{max_retries} "
                      f"after {type(exc).__name__}: {exc!s:.180} "
                      f"— sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)


def generate_eval_one(
    dataset_name: str,
    datasets_dir: str = _DEFAULT_DATASETS_DIR,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    save_generated: bool = True,
    verbose: bool = True,
    api: Optional[str] = None,
    max_rounds: int = 1,
    platform_info: Optional[PlatformInfo] = None,
    execution_lock: Optional[ContextManager] = None,
    llm_max_retries: int = 6,
    llm_backoff_base: float = 4.0,
    no_ref: bool = False,
) -> Tuple[bool, str, Optional[CompileRunResult], int, Optional[dict]]:
    """
    Generate code for one dataset and evaluate it with multi-round retry.

    Args:
        dataset_name: Name of the dataset (e.g., "example1_ipc_gpu_comm")
        datasets_dir: Base directory containing datasets
        model: LLM model to use (e.g., "gpt-4o", "gemini-1.5-pro")
        temperature: Temperature for code generation
        save_generated: Whether to save the generated code
        verbose: Whether to print detailed progress
        api: Optional API provider ("openai" or "gemini"). Auto-detected from model if not specified.
        max_rounds: Maximum number of generation rounds (default: 1)
        platform_info: Optional PlatformInfo object (will detect if not provided)
        execution_lock: Optional context manager (e.g. threading.Lock) that is
            acquired around every compile-and-run / compare subprocess. Used
            by the batch driver to serialize all on-GPU/binary execution
            across concurrently-running examples so performance measurements
            are not perturbed by neighbours. LLM API calls are NOT held under
            this lock so they can still overlap.
        llm_max_retries: Max retries for transient/rate-limit LLM errors.
        llm_backoff_base: Base seconds for exponential backoff (the n-th
            retry sleeps `base * 2**n + jitter` seconds).
        no_ref: If True (or if no ref_* file exists), skip the gen-vs-ref
            performance comparison entirely. Correctness is still checked from
            the generated program's own test harness, but there is no perf
            verdict and no performance-driven retry.

    When a reference is available, the framework asks each example's
    build_and_run.py to produce gen-vs-ref comparison plots via --compare.
    When no reference is present, this comparison step is skipped.

    Returns:
        Tuple of (passed: bool, reason: str, result: CompileRunResult, rounds_used: int, performance_comparison: dict or None)
    """
    # Construct dataset directory path
    dataset_dir = os.path.join(datasets_dir, dataset_name)

    if not os.path.isdir(dataset_dir):
        error_msg = f"Dataset directory not found: {dataset_dir}"
        if verbose:
            print(f"ERROR: {error_msg}")
        return False, error_msg, None, 0, None

    # Get platform info if not provided
    if platform_info is None:
        platform_info = detect_platform()

    # Check for platform subdirectory
    platform_subdir = get_platform_subdir(dataset_dir, platform_info)
    working_dir = platform_subdir if platform_subdir else dataset_dir

    if verbose:
        print("=" * 80)
        print(f"Generating and Evaluating: {dataset_name}")
        print(f"Max rounds: {max_rounds}")
        print("=" * 80)
        print(f"Dataset directory: {dataset_dir}")
        if platform_subdir:
            print(f"Platform subdirectory: {platform_subdir}")
            print(f"Platform: {get_platform_string(platform_info)}")

    # Step 1: Find empty_* source file
    if verbose:
        print("\n[Step 1] Finding empty_* source file...")

    empty_file = find_empty_file(dataset_dir, platform_info)
    if not empty_file:
        error_msg = f"No empty_* source file found in {dataset_dir}"
        if verbose:
            print(f"ERROR: {error_msg}")
        return False, error_msg, None, 0, None

    # Get file extension for later use
    file_ext = os.path.splitext(empty_file)[1]
    file_type = get_file_type_description(empty_file)

    if verbose:
        print(f"Found: {empty_file}")
        print(f"File type: {file_type}")

    # Step 2: Generate initial prompt
    if verbose:
        print("\n[Step 2] Generating initial prompt...")

    try:
        initial_prompt = generate_completion_prompt(empty_file)
        if verbose:
            print(f"Prompt generated ({len(initial_prompt)} characters)")
            print(f"Preview: {initial_prompt[:200]}...")
    except Exception as e:
        error_msg = f"Failed to generate prompt: {str(e)}"
        if verbose:
            print(f"ERROR: {error_msg}")
        return False, error_msg, None, 0, None

    # Check for build_and_run.py before starting rounds
    build_and_run_script = os.path.join(working_dir, "build_and_run.py")
    if not os.path.isfile(build_and_run_script):
        error_msg = f"build_and_run.py not found: {build_and_run_script}"
        if verbose:
            print(f"ERROR: {error_msg}")
        return False, error_msg, None, 0, None

    # Step 2.5: Locate reference file (used only for the gen-vs-ref performance
    # comparison). A missing reference is NOT fatal: we can still generate,
    # compile, run and verify correctness from the generated program's own test
    # harness. In that case we simply skip the perf comparison and any
    # performance-driven retry. Pass --no-ref to force this path even when a
    # reference exists.
    ref_file = None if no_ref else find_ref_file(
        dataset_dir, platform_info, empty_file=empty_file
    )
    ref_filename = os.path.basename(ref_file) if ref_file else None
    compare_enabled = ref_filename is not None
    if verbose:
        if compare_enabled:
            print(f"Found reference file: {ref_filename}")
        elif no_ref:
            print("Reference comparison disabled (--no-ref): "
                  "correctness only, no performance comparison.")
        else:
            print(f"No ref_* source file found in {dataset_dir} — "
                  "running without reference: correctness only, "
                  "no performance comparison.")

    # Base name and extension for generated file (full name built per-round)
    empty_basename = os.path.basename(empty_file)
    base_generated_name = empty_basename.replace("empty_", "generated_")
    base_name_no_ext = os.path.splitext(base_generated_name)[0]
    model_safe = model.replace("/", "_").replace("\\", "_")

    # Determine language based on file extension
    language = "CUDA" if file_ext == '.cu' else "C++"
    provider_name = api if api else get_provider_for_model(model).value

    # If no external lock was supplied, use a no-op context manager so the
    # rest of the code can `with execution_lock:` unconditionally.
    if execution_lock is None:
        execution_lock = contextlib.nullcontext()

    # Multi-round generation loop
    current_prompt = initial_prompt
    generated_code = None
    result = None
    passed = False
    reason = ""
    # Track each failed round's (code, error) so the next retry prompt can
    # show the most recent 1-2 failed attempts as diffs vs. the current
    # attempt. This lets the model see which alternative paths it has
    # already explored and avoid oscillating between them. Newest-first
    # ordering matches what generate_fix_prompt_from_code expects.
    failed_history: list = []
    HISTORY_DEPTH = 2
    # Each round writes into its OWN results_<model>_round<N>_<ts>/ folder
    # and that round's generated source file is moved into that same folder.
    # current_round_dir tracks the most recent round; last_compared_dir
    # tracks the most recent round that actually ran the inline compare
    # (i.e. compile+run passed) and therefore has a summary.json.
    current_round_dir: Optional[str] = None
    last_compared_dir: Optional[str] = None
    last_summary: Optional[dict] = None

    for round_num in range(1, max_rounds + 1):
        if verbose:
            print("\n" + "=" * 80)
            print(f"ROUND {round_num}/{max_rounds}")
            print("=" * 80)

        # Step 3: Get LLM generated code
        if verbose:
            print(f"\n[Step 3] Requesting code from {provider_name.upper()} model ({model})...")

        try:
            generated_response = _call_llm_with_retry(
                prompt=current_prompt,
                model=model,
                language=language,
                temperature=temperature,
                api=api,
                max_retries=llm_max_retries,
                backoff_base=llm_backoff_base,
                verbose=verbose,
            )

            # Extract code from response
            generated_code = extract_code_from_response(generated_response, file_ext)

            if verbose:
                print(f"Code generated ({len(generated_code)} characters)")
        except Exception as e:
            error_msg = f"Failed to generate code: {str(e)}"
            if verbose:
                print(f"ERROR: {error_msg}")
            return False, error_msg, None, round_num, None

        # Step 4: Save as generated_* with model, round, timestamp in filename
        # e.g. generated_fifo_test_unified_gpt-4o_round1_20260202_153012.cu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_filename = f"{base_name_no_ext}_{model_safe}_round{round_num}_{timestamp}{file_ext}"
        generated_filepath = os.path.join(working_dir, generated_filename)

        if verbose:
            print(f"\n[Step 4] Saving generated code...")

        if save_generated:
            try:
                with open(generated_filepath, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
                if verbose:
                    print(f"Saved to: {generated_filepath}")
            except Exception as e:
                error_msg = f"Failed to save generated code: {str(e)}"
                if verbose:
                    print(f"ERROR: {error_msg}")
                return False, error_msg, None, round_num, None
        else:
            if verbose:
                print("(Skipped saving - save_generated=False)")

        # Always create a per-round results dir so this round's generated
        # source ends up co-located with whatever artifacts (summary.json,
        # CSVs, plots) the round produces. Successful rounds run the
        # inline compare into this dir; failed rounds get just the source.
        # Layout: results/<model>/round<N>_<ts>/  — one subdir per model so
        # different models' outputs never collide.
        results_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dirname = f"round{round_num}_{results_timestamp}"
        current_round_dir = os.path.join(working_dir, "results", model_safe, results_dirname)
        os.makedirs(current_round_dir, exist_ok=True)

        # Save the exact prompt sent to the LLM in this round, alongside the
        # generated source and any artifacts. This is what the LLM saw — for
        # round 1 it's the initial empty_* completion prompt, for round 2+
        # it's the retry prompt (with previous error, current code, and any
        # tried-and-failed history). Useful for diffing prompts across
        # rounds and explaining why a given attempt looked the way it did.
        prompt_save_path = os.path.join(current_round_dir, "prompt.txt")
        try:
            with open(prompt_save_path, "w", encoding="utf-8") as f:
                f.write(current_prompt)
            if verbose:
                print(f"Saved round {round_num} prompt to {prompt_save_path} "
                      f"({len(current_prompt)} chars)")
        except Exception as e:
            if verbose:
                print(f"WARN: failed to save prompt to {prompt_save_path}: {e}")

        # Step 5: Check generated code for template cheating before
        # compiling. A cheat is terminal for this dataset: do not retry.
        if verbose:
            print("\n[Step 5] Running cheat detection...")

        try:
            cheat_result = run_all_checks(Path(generated_filepath), Path(empty_file))
        except Exception as e:
            error_msg = f"Failed during cheat detection: {str(e)}"
            if verbose:
                print(f"ERROR: {error_msg}")
            return False, error_msg, None, round_num, None

        if cheat_result.cheat_detected:
            reason = f"CHEAT: {'; '.join(cheat_result.reasons)}"
            passed = False

            summary_path = os.path.join(current_round_dir, "summary.json")
            summary_data = {
                "status": "cheat",
                "cheat_detected": True,
                "cheat_reasons": cheat_result.reasons,
                "model": model,
                "pass_iteration": None,
                "improvement_iteration": None,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)

            if verbose:
                print("\nCHEAT DETECTED")
                for cheat_reason in cheat_result.reasons:
                    print(f"  - {cheat_reason}")
                _, modified = check_non_todo_modified(Path(generated_filepath), Path(empty_file))
                if modified:
                    print(format_modified_functions(modified))
                print(f"Summary saved to {summary_path}")

            # Move THIS round's generated source into THIS round's results dir
            # so each round's code stays coupled with its results.
            if save_generated and os.path.isfile(generated_filepath):
                try:
                    shutil.move(generated_filepath, os.path.join(current_round_dir, generated_filename))
                    if verbose:
                        print(f"Moved generated source to {current_round_dir}/{generated_filename}")
                except Exception as e:
                    if verbose:
                        print(f"WARN: failed to move {generated_filepath} into {current_round_dir}: {e}")

            last_compared_dir = None
            break

        if verbose:
            print("NO CHEAT DETECTED")
        
        # Step 6: Compile and run using build_and_run.py
        # Held under execution_lock so that the GPU/NIC isn't simultaneously
        # being driven by another example's binary while we measure perf.
        if verbose:
            print("\n[Step 6] Compiling and running (serialized)...")

        try:
            with execution_lock:
                result = compile_and_run(
                    folder_path=working_dir,
                    source_file=generated_filename,
                    timeout=180
                )

            if verbose:
                print("\nOutput:")
                if result.compile_stdout:
                    print(result.compile_stdout)
                if result.compile_stderr:
                    print("Errors/Warnings:")
                    print(result.compile_stderr)
                if result.run_stdout:
                    print("Run output:")
                    print(result.run_stdout)
                if result.run_stderr:
                    print("Run errors:")
                    print(result.run_stderr)

        except Exception as e:
            error_msg = f"Failed during compile/run: {str(e)}"
            if verbose:
                print(f"ERROR: {error_msg}")
            return False, error_msg, None, round_num, None

        # Step 7: Analyze results
        if verbose:
            print("\n[Step 7] Analyzing results...")

        passed, reason = analyze_result(result)

        if verbose:
            print("\n" + "-" * 40)
            if passed:
                print(f"ROUND {round_num} RESULT: PASS")
            else:
                print(f"ROUND {round_num} RESULT: FAIL")
            print(f"Reason: {reason}")
            print("-" * 40)

        # For FAILED rounds, save the full build/run output so we can audit
        # what the LLM was supposed to fix. PASS rounds produce summary.json
        # + plots/CSVs via compare_save_results below and don't need this.
        # Errors are persisted exactly as they were fed back into the retry
        # prompt (via get_error_details), so this file is also a 1:1 record
        # of the error context the next round will see.
        if not passed and result is not None:
            err_save_path = os.path.join(current_round_dir, "build_run_output.txt")
            try:
                parts = [
                    f"# Round {round_num} build/run output",
                    f"# reason: {reason}",
                    f"# compile_success={result.compile_success} "
                    f"run_success={result.run_success}",
                    "",
                    f"=== compile_stdout (rc={result.compile_returncode}) ===",
                    result.compile_stdout or "",
                    "",
                    f"=== compile_stderr ===",
                    result.compile_stderr or "",
                    "",
                    f"=== run_stdout (rc={result.run_returncode}) ===",
                    result.run_stdout or "",
                    "",
                    f"=== run_stderr ===",
                    result.run_stderr or "",
                ]
                with open(err_save_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(parts))
                if verbose:
                    print(f"Saved round {round_num} build/run output to "
                          f"{err_save_path}")
            except Exception as e:
                if verbose:
                    print(f"WARN: failed to save build/run output to "
                          f"{err_save_path}: {e}")

        last_summary = None
        acceptable = False
        performance: Optional[str] = None

        if passed and compare_enabled:
            if verbose:
                print(f"\n[Step 8] Inline comparison for round {round_num} → {current_round_dir} (serialized)")

            # compare_save_results rebuilds and re-runs BOTH ref and
            # generated to produce perf metrics — hold the execution lock so
            # other examples' runs don't perturb these measurements.
            with execution_lock:
                compare_save_results(
                    working_dir=working_dir,
                    ref_filename=ref_filename,
                    generated_filename=generated_filename,
                    result_dir=current_round_dir,
                    verbose=verbose,
                    no_plot=False,
                )

            last_compared_dir = current_round_dir
            summary_path = os.path.join(current_round_dir, "summary.json")
            if os.path.isfile(summary_path):
                try:
                    with open(summary_path, "r") as f:
                        last_summary = json.load(f)
                except Exception:
                    last_summary = None

            performance = (last_summary or {}).get("performance")
            # Detect "info-only" examples (e.g. RDMA NIC info dump) whose
            # metrics_comparison only carries compile_success/run_success —
            # there are no real perf numbers, so any successful run is fine.
            mc = (last_summary or {}).get("metrics_comparison") or {}
            ref_block = mc.get("ref") or {}
            info_only = bool(ref_block) and set(ref_block.keys()).issubset(
                {"compile_success", "run_success"}
            )

            # Acceptable verdicts (no perf retry):
            #   - on_par/better/improved/same     ← real comparison, model is fine
            #   - info_only/no_gen_metrics/no_ref_metrics/unknown
            #     ← no real comparison happened; retrying won't conjure metrics
            #   - None                            ← summary missing entirely
            # Anything else (degraded, severely_degraded, worse) triggers
            # the perf-fix retry branch below.
            acceptable = (
                performance in (
                    None, "same", "better", "improved", "on_par",
                    "info_only", "no_gen_metrics", "no_ref_metrics", "unknown",
                )
                or info_only
            )

            if verbose:
                print(f"Round {round_num} performance verdict: {performance!r}"
                      f" (info_only={info_only}, acceptable={acceptable})")

        elif passed:
            # No reference available (or --no-ref): the generated program's own
            # test harness already verified correctness. There is nothing to
            # compare against, so accept this round as-is — no perf verdict and
            # no performance-driven retry.
            acceptable = True
            performance = None
            if verbose:
                print(f"Round {round_num} passed — no reference, "
                      f"skipping performance comparison.")

        # Move THIS round's generated source into THIS round's results dir
        # so each round's code stays coupled with its results.
        if save_generated and os.path.isfile(generated_filepath):
            try:
                shutil.move(generated_filepath, os.path.join(current_round_dir, generated_filename))
                if verbose:
                    print(f"Moved generated source to {current_round_dir}/{generated_filename}")
            except Exception as e:
                if verbose:
                    print(f"WARN: failed to move {generated_filepath} into {current_round_dir}: {e}")

        # Decide what (if anything) to do for the next round. The same
        # max_rounds budget covers BOTH compile/run failures and
        # performance-degraded retries.
        if not passed:
            # compile/run failed → next round gets the error feedback
            error_details = get_error_details(result)
            if round_num < max_rounds:
                if verbose:
                    print(f"\n[Retry] Preparing error fix prompt for round {round_num + 1}...")
                # Pass the most-recent HISTORY_DEPTH prior failures so the
                # model sees the diff-and-error of each, in newest-first order.
                prior_attempts = list(failed_history[:HISTORY_DEPTH])
                current_prompt = generate_fix_prompt_from_code(
                    code=generated_code,
                    file_type=file_type,
                    error_message=error_details,
                    round_num=round_num,
                    prior_attempts=prior_attempts or None,
                )
                if verbose:
                    print(f"Fix prompt generated ({len(current_prompt)} characters,"
                          f" with {len(prior_attempts)} prior attempt(s) in history)")
            else:
                if verbose:
                    print(f"\n[End] Maximum rounds ({max_rounds}) reached without a successful run.")
            # Record THIS round's failure into the history *after* we built
            # the next-round prompt — so this round becomes round N-1 from
            # the perspective of the round after the next one.
            failed_history.insert(0, (generated_code, error_details))
            del failed_history[HISTORY_DEPTH:]
            continue

        if acceptable:
            break

        # Performance is degraded → ask the model to optimize, if rounds remain.
        if round_num < max_rounds:
            if verbose:
                print(f"\n[Retry] Preparing performance optimization prompt"
                      f" for round {round_num + 1}...")
            try:
                summary_str = json.dumps(last_summary, indent=2) if last_summary else "{}"
            except Exception:
                summary_str = "{}"
            current_prompt = generate_perf_fix_prompt_from_code(
                code=generated_code,
                file_type=file_type,
                summary_json=summary_str,
                round_num=round_num,
            )
            if verbose:
                print(f"Perf-fix prompt generated ({len(current_prompt)} characters)")
        else:
            if verbose:
                print(f"\n[End] Maximum rounds ({max_rounds}) reached;"
                      f" perf still {performance!r}.")
            break

    performance_comparison = None

    # If at least one round produced an inline-compare summary, decorate
    # the most recent such summary with run-level metadata. Otherwise we
    # still leave each round's results dir on disk (with the generated
    # source) — there's just no summary to load.
    summary_results_dir = last_compared_dir
    if summary_results_dir:
        summary_path = os.path.join(summary_results_dir, "summary.json")
        if os.path.isfile(summary_path):
            try:
                with open(summary_path, "r") as f:
                    summary_data = json.load(f)

                # Fill in model, pass_iteration, improvement_iteration
                summary_data["model"] = model
                summary_data["pass_iteration"] = round_num if passed else None
                perf = summary_data.get("performance")
                summary_data["improvement_iteration"] = (
                    round_num if perf in ("better", "improved") else None
                )

                with open(summary_path, "w") as f:
                    json.dump(summary_data, f, indent=2)

                performance_comparison = summary_data.get("metrics_comparison")
                if performance_comparison is None:
                    performance_comparison = {}
                performance_comparison["_summary_file"] = summary_path
            except Exception:
                pass

    # Final result
    if verbose:
        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)
        if passed:
            print(f"RESULT: PASS (completed in round {round_num})")
        else:
            print(f"RESULT: FAIL (after {round_num} rounds)")
        print(f"Reason: {reason}")
        print("=" * 80)

    return passed, reason, result, round_num, performance_comparison


def main():
    """Command-line interface for generate_eval_one."""
    parser = argparse.ArgumentParser(
        description="Generate code from prompts and evaluate results for a dataset. "
                    "Supports multi-round generation with error feedback."
    )

    parser.add_argument(
        "dataset_name",
        type=str,
        nargs='?',
        default=None,
        help="Name of the dataset (e.g., example1_ipc_gpu_comm)"
    )

    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=_DEFAULT_DATASETS_DIR,
        help=f"Base directory containing datasets (default: {_DEFAULT_DATASETS_DIR})"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use (e.g., gpt-4o, gemini-1.5-pro, grok-3, claude-opus-4-20250514, qwen3.6-27B). Default: gpt-4o"
    )

    parser.add_argument(
        "--api",
        type=str,
        choices=["openai", "gemini", "grok", "anthropic", "deepseek", "glm", "qwen", "moonshot"],
        default=None,
        help="API provider to use (openai, gemini, grok, anthropic, deepseek, glm, qwen, or moonshot). Auto-detected from model name if not specified."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for code generation (default: 0.3)"
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        help="Maximum number of generation rounds. If code fails to compile/run, "
             "the error is fed back to the LLM for retry (default: 1, no retry)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the generated code file"
    )

    parser.add_argument(
        "--no-ref",
        action="store_true",
        help="Run without a reference solution: only check correctness from the "
             "generated program's own test harness, skipping the gen-vs-ref "
             "performance comparison and any performance-driven retry. This is "
             "also applied automatically when no ref_* file exists."
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all supported models and exit"
    )

    parser.add_argument(
        "--skip-platform-detect",
        action="store_true",
        help="Skip platform detection at startup"
    )

    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=6,
        help="Max retries when LLM call fails with a rate-limit / transient "
             "error (default: 6, exponential backoff)."
    )

    parser.add_argument(
        "--llm-backoff-base",
        type=float,
        default=4.0,
        help="Base seconds for LLM-retry exponential backoff "
             "(attempt n sleeps base * 2**n + jitter). Default 4.0."
    )

    args = parser.parse_args()

    if not args.skip_platform_detect and not args.list_models:
        if not args.quiet:
            print("\n" + "=" * 80)
            print("PLATFORM DETECTION")
            print("=" * 80)
        platform_info = detect_platform(verbose=not args.quiet)
        if not args.quiet:
            print(platform_info.summary())
            print("")
        else:
            print(f"Platform: {get_platform_string(platform_info)}")

    # Handle --list-models
    if args.list_models:
        print("Supported Models:")
        print("-" * 60)
        models = list_supported_models()
        for provider, model_list in models.items():
            print(f"\n{provider.upper()}:")
            for model in model_list:
                print(f"  - {model}")
        print("\nNote: Model names starting with 'gpt', 'o1', 'o3' are auto-detected as OpenAI.")
        print("      Model names starting with 'gemini' are auto-detected as Google Gemini.")
        print("      Model names starting with 'grok' are auto-detected as xAI Grok.")
        print("      Model names starting with 'claude' are auto-detected as Anthropic.")
        print("      Model names starting with 'deepseek' are auto-detected as DeepSeek.")
        print("      Model names starting with 'glm' or 'z-ai/glm' are auto-detected as GLM.")
        print("      Model names starting with 'qwen' are auto-detected as Qwen.")
        sys.exit(0)

    # Validate dataset_name is provided when not using --list-models
    if not args.dataset_name:
        parser.error("dataset_name is required (unless using --list-models)")

    # Validate max_rounds
    if args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1")

    # Run the evaluation
    passed, _reason, _result, rounds_used, perf_comparison = generate_eval_one(
        dataset_name=args.dataset_name,
        datasets_dir=args.datasets_dir,
        model=args.model,
        temperature=args.temperature,
        save_generated=not args.no_save,
        verbose=not args.quiet,
        api=args.api,
        max_rounds=args.max_rounds,
        llm_max_retries=args.llm_max_retries,
        llm_backoff_base=args.llm_backoff_base,
        no_ref=args.no_ref,
    )

    # Print summary for quiet mode
    if args.quiet:
        status = "PASS" if passed else "FAIL"
        perf_status = ""
        if perf_comparison and perf_comparison.get("summary"):
            perf_status = f" perf: {perf_comparison['summary'].get('status', 'N/A')}"
        print(f"{args.dataset_name}: {status} (rounds: {rounds_used}/{args.max_rounds}){perf_status}")

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
