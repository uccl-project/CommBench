#!/usr/bin/env python3
"""
Prompt Generation Module
Provides functions to generate prompts for GPU code completion.
Supports both .cpp and .cu files.
"""

import difflib
import os
import re
from typing import List, Optional, Tuple
from pathlib import Path


# Per-section size caps for the "tried & failed" history block in retry
# prompts. We need the prior code/error to be informative without blowing
# the prompt out from ~25k to ~75k chars on every retry.
_HISTORY_DIFF_CHAR_CAP = 3000
_HISTORY_ERROR_CHAR_CAP = 1500

ANTI_CHEAT_RULES = """IMPORTANT — Anti-cheat rules (violations will be detected and flagged):
- Only implement functions marked with //TODO; do not modify any other function.
- You may add new helper functions, but do not change existing ones that lack //TODO.
- Every TODO section must be genuinely implemented with real logic.
- Do NOT output hardcoded or fake metric values to pass tests."""


def _truncate_middle(text: str, cap: int) -> str:
    """Trim text to `cap` chars by keeping head and tail and eliding the middle."""
    if text is None:
        return ""
    if len(text) <= cap:
        return text
    head = cap // 2
    tail = cap - head
    return f"{text[:head]}\n... [truncated {len(text) - cap} chars] ...\n{text[-tail:]}"


def _summarize_diff(prior_code: str, current_code: str, label_prior: str,
                    label_current: str) -> str:
    """Build a unified diff from `prior_code` to `current_code`, capped in size.

    Direction is prior → current so the diff reads as "what the model changed
    in response to the prior error". If the prior version is wildly different
    (e.g., a full rewrite), the diff is truncated middle so the prompt stays
    bounded.
    """
    diff_iter = difflib.unified_diff(
        prior_code.splitlines(keepends=True),
        current_code.splitlines(keepends=True),
        fromfile=label_prior,
        tofile=label_current,
        n=2,
    )
    diff_text = "".join(diff_iter)
    if not diff_text.strip():
        return "(no textual diff — prior attempt had identical code)"
    return _truncate_middle(diff_text, _HISTORY_DIFF_CHAR_CAP)


def get_file_extension(file_path: str) -> str:
    """Get the file extension (e.g., '.cpp', '.cu')."""
    return os.path.splitext(file_path)[1].lower()


def get_file_type_description(file_path: str) -> str:
    """Get a description of the file type based on extension."""
    ext = get_file_extension(file_path)
    if ext == '.cu':
        return '.cu'
    elif ext in ['.cpp', '.cxx', '.cc']:
        return '.cpp'
    elif ext in ['.hip', '.hip.cpp']:
        return '.hip'
    elif ext == '.py':
        return '.py'
    else:
        return ext if ext else '.cpp'


def read_source_file(file_path: str) -> str:
    """
    Read content from a source file (.cpp or .cu).

    Args:
        file_path: Path to the source file.

    Returns:
        str: Content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


# Keep old function name for backward compatibility
read_cpp_file = read_source_file


def extract_code_from_response(response: str, file_ext: str = '.cpp') -> str:
    """
    Extract source code from LLM response.

    The response might contain markdown code blocks or other text.
    This function tries to extract just the code.

    Args:
        response: Raw response from LLM
        file_ext: File extension to help identify code type

    Returns:
        Extracted source code

    Example:
        >>> code = extract_code_from_response("```cpp\\n#include <iostream>\\n```")
        >>> print(code)
        #include <iostream>
    """
    # Determine language patterns based on extension
    if file_ext == '.cu':
        lang_patterns = ['cuda', 'cu', 'cpp', 'c\\+\\+', 'C\\+\\+']
    elif file_ext == '.py':
        lang_patterns = ['python', 'py']
    else:
        lang_patterns = ['cpp', 'c\\+\\+', 'C\\+\\+', 'hip']

    for lang in lang_patterns:
        pattern = rf'```{lang}?\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

    # Generic fallback (closed block)
    generic_pattern = r'```\n(.*?)```'
    matches = re.findall(generic_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Handle truncated responses: opening fence present but no closing fence
    for lang in lang_patterns:
        open_pattern = rf'```{lang}?\n(.*)'
        m = re.search(open_pattern, response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

    generic_open = r'```\n(.*)'
    m = re.search(generic_open, response, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Python-specific fallback
    if file_ext == '.py':
        if response.strip().startswith(('import ', 'from ', 'def ', 'class ')):
            return response.strip()

    # C/C++ fallback
    if response.strip().startswith('#include') or response.strip().startswith('//'):
        return response.strip()

    return response.strip()


def generate_completion_prompt(source_file: str) -> str:
    source_content = read_source_file(source_file)
    file_type = get_file_type_description(source_file)

    basename = os.path.basename(source_file)
    name_without_ext = os.path.splitext(basename)[0]
    if name_without_ext.startswith("empty_"):
        name_without_ext = name_without_ext[6:]
    task_name = name_without_ext

    # -------- Python prompt --------
    if file_type == '.py':
        prompt = f"""You are a Python code generator and patch author.

Specification:
Name: {task_name}
Type: python_runtime_orchestration

System goals:
- Functional correctness
- Clear and maintainable Python code
- Correct use of PyTorch / torch.distributed APIs
- Minimal invasive changes; respect existing APIs

Inputs:
- You will be given an existing Python file (possibly incomplete).
- You may be given tests that must pass.

Requirements:
- Keep public APIs stable (function signatures, class names).
- Ensure the code runs correctly.
- Follow idiomatic Python and PyTorch practices.
- Do not introduce unnecessary dependencies.

{ANTI_CHEAT_RULES}

The code to be completed is as follows:

{source_content}

Do not include any other text.

Please return only the complete Python (.py) code, without any additional text, explanations, or markdown formatting.

Now generate/complete the implementation."""
        return prompt

    # -------- C++ / CUDA prompt (original behavior) --------
    prompt = f"""You are a code generator and patch author for GPU communication systems. Generate or complete implementations for the following specification.

Specification:
Name: {task_name}
Type: gpu_communication

System goals:
- Functional correctness under multi-GPU concurrency
- High performance on modern GPUs (NVIDIA or AMD depending on the repo)
- Minimal invasive changes; respect existing APIs

Inputs:
- You will be given one or more existing files (possibly incomplete).
- You will be given tests that must pass.

Requirements:
- Keep public APIs stable (function signatures, exported symbols).
- Ensure compilation and all tests pass.
- Use existing project utilities for:
  - device selection / streams
  - error handling
  - RDMA endpoints / transport
  - collective algorithms (ring/tree)
- If you introduce a new helper, place it in the specified file(s) only.

{ANTI_CHEAT_RULES}

The code to be completed is as follows:

{source_content}

Do not include any other text.

Please return only the complete {file_type} code, without any additional text, explanations, or markdown formatting.

Now generate/complete the implementation."""
    return prompt


def generate_completion_prompt_detailed(
    source_file: str,
    additional_instructions: Optional[str] = None
) -> str:
    """
    Generate a detailed prompt for completing GPU code.

    Args:
        source_file: Path to the source file containing incomplete code.
        additional_instructions: Optional additional instructions to include in the prompt.

    Returns:
        str: The generated prompt with detailed instructions.

    Example:
        >>> prompt = generate_completion_prompt_detailed(
        ...     "empty_gpu_p2p_comm.cpp",
        ...     additional_instructions="Use HIP API for all GPU operations."
        ... )
    """
    source_content = read_source_file(source_file)
    file_type = get_file_type_description(source_file)

    # Extract name from filename
    basename = os.path.basename(source_file)
    name_without_ext = os.path.splitext(basename)[0]
    if name_without_ext.startswith("empty_"):
        name_without_ext = name_without_ext[6:]
    task_name = name_without_ext

    # Build additional requirements section
    extra_requirements = ""
    if additional_instructions:
        extra_requirements = f"\nAdditional requirements:\n{additional_instructions}\n"

    prompt = f"""You are a code generator and patch author for GPU communication systems. Generate or complete implementations for the following specification.

Specification:
Name: {task_name}
Type: gpu_communication

System goals:
- Functional correctness under multi-GPU concurrency
- High performance on modern GPUs (NVIDIA or AMD depending on the repo)
- Minimal invasive changes; respect existing APIs

Inputs:
- You will be given one or more existing files (possibly incomplete).
- You will be given tests that must pass.

Requirements:
- Keep public APIs stable (function signatures, exported symbols).
- Ensure compilation and all tests pass.
- Use existing project utilities for:
  - device selection / streams
  - error handling
  - RDMA endpoints / transport
  - collective algorithms (ring/tree)
- If you introduce a new helper, place it in the specified file(s) only.
- Complete all TODO sections.
- Implement proper error handling.
- Include proper memory allocation and cleanup.
{extra_requirements}
The code to be completed is as follows:

{source_content}

Do not include any other text.

Please return only the complete {file_type} code, without any additional text, explanations, or markdown formatting.

Now generate/complete the implementation."""

    return prompt


def generate_fix_prompt(source_file: str, error_message: Optional[str] = None) -> str:
    """
    Generate a prompt for fixing errors in GPU code (from file).

    Args:
        source_file: Path to the source file containing code with errors.
        error_message: Optional error message to include.

    Returns:
        str: The generated prompt for fixing the code.

    Example:
        >>> prompt = generate_fix_prompt(
        ...     "buggy_code.cpp",
        ...     error_message="Compilation error: undefined reference to hipDeviceEnablePeerAccess"
        ... )
    """
    source_content = read_source_file(source_file)
    file_type = get_file_type_description(source_file)

    return generate_fix_prompt_from_code(
        code=source_content,
        file_type=file_type,
        error_message=error_message
    )


def generate_fix_prompt_from_code(
    code: str,
    file_type: str,
    error_message: Optional[str] = None,
    round_num: Optional[int] = None,
    prior_attempts: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Generate a prompt for fixing errors in GPU code (from code string).

    This is used for multi-round generation where error feedback is provided.

    Args:
        code: The source code that just failed (most-recent attempt).
        file_type: File type description (e.g., '.cpp', '.cu').
        error_message: Error message from compilation/execution of `code`.
        round_num: Round number that produced `code` (so the next round is N+1).
        prior_attempts: Optional list of (prior_code, prior_error) for earlier
            failed rounds, ordered most-recent-first. Each entry is rendered as
            a unified diff vs `code` plus the truncated error it produced, so
            the model can see which alternative paths have already been
            explored and avoid bouncing between them. Diffs and errors are
            individually capped (see _HISTORY_DIFF_CHAR_CAP /
            _HISTORY_ERROR_CHAR_CAP) to keep prompt size bounded.

    Returns:
        str: The generated prompt for fixing the code.

    Example:
        >>> prompt = generate_fix_prompt_from_code(
        ...     code="#include <hip/hip_runtime.h>\\n...",
        ...     file_type=".cpp",
        ...     error_message="undefined reference to hipDeviceEnablePeerAccess",
        ...     round_num=1
        ... )
    """
    round_info = ""
    if round_num is not None:
        round_info = f" (round {round_num})"

    error_section = ""
    if error_message:
        error_section = f"""
The previous code generation attempt{round_info} failed with the following error:

{error_message}

"""

    history_section = ""
    if prior_attempts:
        # `prior_attempts` is most-recent-first. Number them so the model
        # sees absolute round labels (round N-1, round N-2, ...) when
        # round_num is known; otherwise fall back to relative offsets.
        rendered = []
        for offset, (prior_code, prior_error) in enumerate(prior_attempts, start=1):
            if round_num is not None:
                prior_round_num = round_num - offset
                current_label = f"round {round_num} (failed; shown above)"
                prior_label = f"round {prior_round_num} (also failed)"
            else:
                prior_round_num = None
                current_label = "current attempt (failed; shown above)"
                prior_label = f"prior attempt -{offset} (also failed)"
            diff_text = _summarize_diff(
                prior_code=prior_code,
                current_code=code,
                label_prior=prior_label,
                label_current=current_label,
            )
            err_text = _truncate_middle(prior_error or "(no error captured)",
                                        _HISTORY_ERROR_CHAR_CAP)
            label = (f"### {prior_label}"
                     if prior_round_num is None
                     else f"### Round {prior_round_num} (earlier attempt, also failed)")
            rendered.append(
                f"{label}\n"
                f"Diff vs the current attempt (changes the current attempt made on top of "
                f"this earlier version):\n"
                f"```diff\n{diff_text}\n```\n"
                f"Error this earlier attempt produced:\n"
                f"```\n{err_text}\n```\n"
            )

        history_section = (
            "Earlier attempts in this same task ALSO failed. They are shown "
            "below so you can avoid re-trying the same dead ends. Do NOT just "
            "revert to one of these earlier versions — they failed too, with "
            "the errors shown.\n\n"
            + "\n".join(rendered)
            + "\n"
        )

    prompt = f"""You are a code generator and patch author for GPU communication systems.
{error_section}The code that needs to be fixed:

{code}

{history_section}Please fix the code to resolve the above error. Make sure the code:
- Compiles without errors
- Executes correctly
- Passes all tests

Requirements:
- Keep public APIs stable (function signatures, exported symbols).
- Ensure compilation and all tests pass.
- Use existing project utilities for device selection, streams, error handling.
- Fix only what is necessary to resolve the error.
- Take the earlier failed attempts (if any) into account: do not propose a
  patch that is essentially equivalent to one already shown to fail.

{ANTI_CHEAT_RULES}

Do not include any other text.

Please return only the complete {file_type} code, without any additional text, explanations, or markdown formatting.

Now generate the fixed implementation."""

    return prompt


def generate_perf_fix_prompt_from_code(
    code: str,
    file_type: str,
    summary_json: str,
    round_num: Optional[int] = None,
) -> str:
    """
    Build a prompt asking the model to optimize generated code whose
    performance is worse than the reference implementation.

    The prompt includes both the generated and reference per-metric
    averages (as captured in summary.json's `metrics_comparison`) so the
    model can target the metrics that actually regressed.

    Args:
        code: The generated source that compiled and ran but underperformed.
        file_type: File extension (e.g. ".cu", ".cpp").
        summary_json: The full summary.json content as a string.
        round_num: Optional round number for multi-round generation.

    Returns:
        str: The optimization prompt.
    """
    round_info = f" (round {round_num})" if round_num is not None else ""

    prompt = f"""You are a GPU code optimizer.

The previous code generation attempt{round_info} compiled and ran correctly,
but its measured performance is worse than the reference implementation.

Below is a JSON dump comparing the reference run against the generated run.
The `metrics_comparison.ref` block holds the reference's per-metric averages
and the `metrics_comparison.generated` block holds the generated code's
per-metric averages. The `performance` field summarizes the overall verdict
(e.g. "degraded", "severely_degraded", "worse"). Numerical fields whose name
contains "latency" / "lat_" / "wall_" / "time" are LOWER-IS-BETTER. Other
numerical fields (e.g. "throughput", "items_per_sec", "bandwidth") are
HIGHER-IS-BETTER. Use this to identify which metrics regressed.

Performance comparison:
{summary_json}

The current generated code that needs to be optimized:

{code}

Optimize this code so its performance MATCHES OR EXCEEDS the reference on the
slow metrics, while keeping all of these unchanged:
- correctness (the program must still print the same correctness verdict and pass all assertions),
- public APIs (function signatures, exported symbols, JSON output schema),
- the build/run command line accepted by build_and_run.py.

Concrete things to consider:
- memory access patterns (coalescing, alignment, vectorization width),
- launch geometry (block size, grid size, occupancy),
- redundant memory traffic, host↔device synchronization, unnecessary copies,
- compiler/PTX hints (__restrict__, __launch_bounds__, pragma unroll),
- algorithmic changes that preserve the JSON output schema.

Please return ONLY the complete {file_type} code. No markdown fences, no
explanations, no extra text. Now generate the optimized implementation."""
    return prompt


def generate_optimization_prompt(source_file: str, optimization_target: str = "performance") -> str:
    """
    Generate a prompt for optimizing GPU code.

    Args:
        source_file: Path to the source file containing code to optimize.
        optimization_target: Target for optimization (e.g., "performance", "memory", "readability").

    Returns:
        str: The generated prompt for optimization.

    Example:
        >>> prompt = generate_optimization_prompt(
        ...     "gpu_p2p_comm.cpp",
        ...     optimization_target="performance"
        ... )
    """
    source_content = read_source_file(source_file)
    file_type = get_file_type_description(source_file)

    prompt = (
        f"Optimize the following GPU code for {optimization_target}. "
        f"The code is:\n\n{source_content}\n\n"
        f"Please return only the optimized {file_type} code, without any additional text or explanations."
    )

    return prompt


def generate_performance_optimization_prompt(
    code: str,
    file_type: str,
    performance_metrics: dict,
    ref_metrics: Optional[dict] = None,
    optimization_round: int = 1
) -> str:
    """
    Generate a prompt for performance optimization based on current metrics.

    This is used for iterative performance optimization after the code passes
    functional tests.

    Args:
        code: The source code that passed functional tests.
        file_type: File type description (e.g., '.cpp', '.cu').
        performance_metrics: Dictionary of performance metrics from running the code.
        ref_metrics: Optional dictionary of reference implementation metrics for comparison.
        optimization_round: Current optimization round number.

    Returns:
        str: The generated prompt for performance optimization.

    Example:
        >>> prompt = generate_performance_optimization_prompt(
        ...     code="#include <cuda.h>\\n...",
        ...     file_type=".cu",
        ...     performance_metrics={"latency_us": 150.5, "throughput_gbps": 10.2},
        ...     ref_metrics={"latency_us": 100.0, "throughput_gbps": 15.0},
        ...     optimization_round=1
        ... )
    """
    # Format performance metrics
    metrics_str = "\n".join([f"  - {k}: {v}" for k, v in performance_metrics.items()])

    # Build reference comparison section if available
    ref_section = ""
    if ref_metrics:
        ref_str = "\n".join([f"  - {k}: {v}" for k, v in ref_metrics.items()])
        ref_section = f"""
Reference implementation metrics (target to match or exceed):
{ref_str}

"""

    prompt = f"""You are a code optimizer for GPU communication systems. Your task is to optimize the given code for better performance.

Performance Optimization Round: {optimization_round}

Current code performance metrics:
{metrics_str}
{ref_section}The code that needs to be optimized:

{code}

Please optimize the code to improve performance. Focus on:
- Reducing latency
- Increasing throughput/bandwidth
- Better GPU utilization
- Memory access patterns optimization
- Kernel launch configuration tuning
- Reducing synchronization overhead
- Overlapping computation and communication

Requirements:
- Keep public APIs stable (function signatures, exported symbols).
- Ensure the code still compiles and passes all tests.
- Use existing project utilities for device selection, streams, error handling.
- Make meaningful optimizations that can improve the metrics above.

Do not include any other text.

Please return only the complete {file_type} code, without any additional text, explanations, or markdown formatting.

Now generate the optimized implementation."""

    return prompt


def generate_explanation_prompt(source_file: str) -> str:
    """
    Generate a prompt for explaining GPU code.

    Args:
        source_file: Path to the source file containing code to explain.

    Returns:
        str: The generated prompt for code explanation.

    Example:
        >>> prompt = generate_explanation_prompt("gpu_p2p_comm.cpp")
    """
    source_content = read_source_file(source_file)

    prompt = (
        f"Explain the following GPU code in detail:\n\n{source_content}\n\n"
        "Please provide a clear explanation of:\n"
        "1. What the code does\n"
        "2. How GPU communication is implemented\n"
        "3. Key HIP/CUDA API calls and their purposes\n"
        "4. Potential issues or limitations"
    )

    return prompt


def save_prompt_to_file(prompt: str, output_file: str) -> None:
    """
    Save generated prompt to a file.

    Args:
        prompt: The prompt text to save.
        output_file: Path to the output file.

    Example:
        >>> prompt = generate_completion_prompt("empty_gpu_p2p_comm.cpp")
        >>> save_prompt_to_file(prompt, "prompt.txt")
    """
    output_path = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(prompt)

    print(f"Prompt saved to: {output_path}")


def main():
    """Example usage of the prompt generation module."""
    import sys

    # Example file path
    example_cpp = "/home/yangzhou/shuangma/llm-for-gpu-comm/datasets/example1_ipc_gpu_comm/empty_gpu_p2p_comm.cpp"

    # Check if the example file exists
    if not os.path.isfile(example_cpp):
        print(f"Example file not found: {example_cpp}")
        print("Please provide a valid source file path as an argument.")
        if len(sys.argv) > 1:
            example_cpp = sys.argv[1]
        else:
            return

    print("=" * 80)
    print("Example 1: Basic Completion Prompt")
    print("=" * 80)
    prompt1 = generate_completion_prompt(example_cpp)
    print(prompt1[:500] + "..." if len(prompt1) > 500 else prompt1)
    print()


if __name__ == "__main__":
    main()
