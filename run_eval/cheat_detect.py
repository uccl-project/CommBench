#!/usr/bin/env python3
"""Detect potentially invalid generated source files.

This module compares a generated source file with its empty version.
It flags generated files that add extra TODO markers or modify non-TODO
function bodies.

Example:
    python run_eval/cheat_detect.py --empty-file empty.cpp --generated-file generated.cpp
    python run_eval/cheat_detect.py -e empty.cpp -g generated.cpp -le -lg
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ParsedFunction:
    """A function extracted from a source file.

    ParsedFunction represents one comparable function body found by
    tree-sitter, including its fully qualified name, source line span,
    normalized non-comment body text, and whether TODO markers were present.

    Attributes:
        name: Fully qualified function name, including any class or struct
            scope, e.g., ``"GpuDeleter::operator()"``.
        start_line: 1-based line number where the function starts.
        end_line: 1-based line number where the function ends.
        code: Normalized function body with comments and docstrings removed.
        todo_included: Whether the function body contains a TODO marker.
    """

    name: str
    start_line: int
    end_line: int
    code: str
    todo_included: bool


@dataclass(frozen=True)
class CheatResult:
    """The aggregate result of running cheat detection checks.

    CheatResult represents whether any check flagged the generated file and
    stores the human-readable reasons for those findings.

    Attributes:
        cheat_detected: Whether at least one check detected suspicious output.
        reasons: Descriptions of the checks that were flagged.
    """

    cheat_detected: bool
    reasons: List[str]


# Match TODO markers (// TODO in C/C++-style comments and # TODO in Python-style comments, case-insensitively and with optional whitespace) in source text.
TODO_MARKER_PATTERN = re.compile(r"\bTODO\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Preprocess before tree-sitter parsing
# ---------------------------------------------------------------------------


def replace_non_ascii_with_dash(text: str) -> str:
    """Replace non-ASCII code points with an ASCII dash.

    Args:
        text: Input text that may contain non-ASCII characters.

    Returns:
        A string where each non-ASCII character is replaced by ``-``.
    """
    return "".join(c if ord(c) < 128 else "-" for c in text)


def normalize_source_for_tree_sitter(source: str) -> str:
    """Normalize source text before tree-sitter parsing.

    The normalization keeps byte offsets stable enough for parsing while
    replacing non-ASCII comment characters and removing C/C++ digit separators
    that can confuse the parser.

    Args:
        source: Raw source text.

    Returns:
        Source text normalized for tree-sitter parsing.
    """
    parts: List[str] = []
    i = 0
    n = len(source)

    while i < n:
        next = source[i + 1] if i + 1 < n else ""

        # Replace non-ASCII characters with '-' in multi-line comments.
        # For example, some common non-ASCII dash-like characters are: – (en dash), — (em dash), − (minus sign), ‐ (Unicode hyphen), ‑ (non-breaking hyphen). They will be replaced with the plain ASCII - character.
        if source[i] == "/" and next == "*":
            j = i + 2
            while j < n:
                if source[j] == "*" and j + 1 < n and source[j + 1] == "/":
                    j += 2
                    break
                j += 1
            parts.append(replace_non_ascii_with_dash(source[i:j]))
            i = j
            continue

        # Replace non-ASCII characters with '-' in single-line comments.
        if source[i] == "/" and next == "/":
            j = i
            while j < n and source[j] != "\n":
                j += 1
            parts.append(replace_non_ascii_with_dash(source[i:j]))
            i = j
            continue

        # Skip over digit separator boundaries.
        # For example, 1'000'000 will become 1000000 after processing.
        if (
            source[i] == "'"
            and i > 0
            and i + 1 < n
            and source[i - 1].isdigit()
            and source[i + 1].isdigit()
        ):
            i += 1
            continue

        parts.append(source[i])
        i += 1

    return "".join(parts)


# ---------------------------------------------------------------------------
# Parse functions
# ---------------------------------------------------------------------------


def parse_functions(path: Path) -> dict[str, ParsedFunction]:
    """Parse comparable functions from a source file.

    Args:
        path: Path to a supported source file.

    Returns:
        A mapping from fully qualified function name to parsed function data.

    Raises:
        ImportError: If the required tree-sitter package for the detected
            language is not installed.
        ValueError: If the file language is unsupported.
    """
    # Detect the language of the source code and select the appropriate parser.
    language = detect_language(path)
    if language == "cpp":
        try:
            import tree_sitter_cpp as tscpp
            from tree_sitter import Language, Parser
        except ImportError:
            raise ImportError(
                "tree-sitter-cpp requires extra packages:\n"
                "  pip install tree-sitter tree-sitter-cpp"
            )
        parser = Parser(Language(tscpp.language()))

    elif language == "python":
        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser
        except ImportError:
            raise ImportError(
                "tree-sitter-python requires extra packages:\n"
                "  pip install tree-sitter tree-sitter-python"
            )
        parser = Parser(Language(tspython.language()))

    else:
        raise ValueError(f"Unsupported language: {language}")

    # Read the source code and normalize it for tree-sitter parsing.
    source = path.read_text(encoding="utf-8")
    parse_source = normalize_source_for_tree_sitter(source)
    parse_bytes = parse_source.encode("utf-8")

    # Parse the source.
    root = parser.parse(parse_bytes).root_node

    # Initialize the dictionary of parsed functions.
    functions: dict[str, ParsedFunction] = {}

    # Walk the tree and parse the functions.
    def walk(node, scope_stack: List[str]) -> None:
        """Traverse syntax nodes and collect function definitions.

        Args:
            node: Current tree-sitter node.
            scope_stack: Enclosing class or struct names for the current node.

        Returns:
            None.
        """
        # Keep the full class/struct nesting path in source order.
        scope_name = parse_scope_name(node, parse_bytes, language)
        if scope_name is not None:
            scope_stack = scope_stack + [scope_name]

        if node.type == "function_definition":
            # Parse the function name.
            name = parse_function_name(node, parse_bytes, language)
            delimiter = "." if language == "python" else "::"
            full_name = delimiter.join(scope_stack + [name])

            # Get the start and end line numbers of the function.
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            if language == "python":
                # Adjust the start and end line numbers of the function for Python.
                start_line, end_line = adjust_python_function_span(node, source)

            # Parse the function body.
            code, todo_included = parse_function_body(node, parse_bytes, language)

            # Add the function to the list of parsed functions.
            functions[full_name] = ParsedFunction(
                name=full_name,
                start_line=start_line,
                end_line=end_line,
                code=code,
                todo_included=todo_included,
            )

            # Stop descending so nested function definitions are not treated as top-level comparable functions.
            return

        for child in node.children:
            walk(child, scope_stack)

    walk(root, [])
    return functions


def get_file_extension(file_path: str) -> str:
    """Get a lowercase file extension.

    Args:
        file_path: Path-like string whose extension should be inspected.

    Returns:
        The lowercase extension, including the leading dot.
    """
    return os.path.splitext(file_path)[1].lower()


def detect_language(file_path: Path) -> Optional[str]:
    """Detect the parser language for a source file.

    Args:
        file_path: Source file path.

    Returns:
        ``"python"`` for Python files or ``"cpp"`` for C/C++/CUDA/HIP files.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    ext = get_file_extension(file_path)
    if ext in frozenset({".py"}):
        return "python"
    elif ext in frozenset(
        {".c", ".cc", ".cpp", ".cxx", ".h", ".cu", ".hip", ".hip.cpp"}
    ):
        return "cpp"
    else:
        raise ValueError(f"Error: unsupported file type '{ext}'.")


def parse_scope_name(node, source_bytes: bytes, language: str) -> Optional[str]:
    """Parse a class or struct scope name from a syntax node.

    Args:
        node: Tree-sitter node to inspect.
        source_bytes: UTF-8 encoded source text used by the parser.
        language: Detected language name, such as ``"cpp"`` or ``"python"``.

    Returns:
        The scope name if the node represents a class or struct scope;
        otherwise, ``None``.
    """
    # If the node is a class/struct specifier or definition, return the name node.
    if node.type in {"class_specifier", "struct_specifier", "class_definition"}:
        name_node = node.child_by_field_name("name")
        return (
            source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
            if name_node is not None
            else None
        )

    # If the node is an ERROR node and the language is C/C++, try to recover the scope name.
    if language == "cpp" and node.type == "ERROR":
        return parse_error_scope_name(node, source_bytes)

    return None


def parse_error_scope_name(node, source_bytes: bytes) -> Optional[str]:
    """Recover a C/C++ class or struct name from an ERROR node.

    Args:
        node: Tree-sitter ERROR node to inspect.
        source_bytes: UTF-8 encoded source text used by the parser.

    Returns:
        The recovered scope name if the node looks like an incomplete class or
        struct body; otherwise, ``None``.
    """
    saw_class_keyword = False
    name_node = None

    # Traverse the children of the ERROR node
    for child in node.children:
        # If the class/struct keyword is found, set the saw_class_keyword flag to True and move to the next child.
        if child.type in {"class", "struct"}:
            saw_class_keyword = True
            continue
        # If class/struct keyword has not been seen yet, skip the node.
        # This prevents the function from accidentally treating some unrelated identifiers before the class/struct keyword as the scope name.
        if not saw_class_keyword:
            continue
        # If the type/identifier node is found, set the name node to the current child.
        if child.type in {"type_identifier", "identifier"} and name_node is None:
            name_node = child
            continue
        # If the opening brace is found and the name node is set, it treats the identifier as a real class/struct name and returns it.
        if child.type == "{":
            if name_node is None:
                return None
            return source_bytes[name_node.start_byte : name_node.end_byte].decode(
                "utf-8"
            )
        # If it sees ; first, it gives up since that looks more like a declaration than a class body.
        if child.type == ";":
            return None

    # If it cannot confidently recover a scope name, returns None.
    return None


def parse_function_name(func_node, source_bytes: bytes, language: str) -> str:
    """Parse a function name from a function definition node.

    Args:
        func_node: Tree-sitter ``function_definition`` node.
        source_bytes: UTF-8 encoded source text used by the parser.
        language: Detected language name, such as ``"cpp"`` or ``"python"``.

    Returns:
        The parsed function name, or an empty string if no name can be found.

    Raises:
        ValueError: If the language is unsupported.
    """
    # Parse the function name for Python.
    if language == "python":
        # Get the name node.
        name_node = func_node.child_by_field_name("name")
        # Get the text of the name node.
        name = (
            source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
            if name_node is not None
            else ""
        )
        return name

    # Parse the function name for C/C++.
    elif language == "cpp":
        # Get the declarator node.
        decl = func_node.child_by_field_name("declarator")
        if decl is None:
            return ""

        # Traverse the children of the declarator node to find the name node.
        node = decl
        name_node = None
        while node is not None:
            if node.type in {
                "identifier",
                "field_identifier",
                "type_identifier",
                "qualified_identifier",
                "destructor_name",
                "operator_name",
            }:
                name_node = node
                break
            if node.type in {
                "function_declarator",
                "pointer_declarator",
                "reference_declarator",
            }:
                node = node.child_by_field_name("declarator")
                continue
            break

        # Get the text of the name node.
        name = (
            source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
            if name_node is not None
            else ""
        )
        return name

    else:
        raise ValueError(f"Unsupported language: {language}")


def adjust_python_function_span(func_node, source: str) -> Tuple[int, int]:
    """Return the full 1-based line span for a Python function.

    Tree-sitter usually reports the whole function span, but for header-only or
    parse-error stubs it may stop at the `def ...:` line. In that case, expand
    the span manually using Python indentation rules.

    Args:
        func_node: Tree-sitter node for a Python function definition.
        source: Raw Python source text.

    Returns:
        A tuple of ``(start_line, end_line)`` using 1-based line numbers.
    """
    lines = source.splitlines()
    # Get the start and end line numbers of the function.
    start_line = func_node.start_point[0] + 1
    end_line = func_node.end_point[0] + 1
    # If tree-sitter ended on a body line, its span is already complete.
    if end_line > len(lines):
        return start_line, end_line

    header = lines[end_line - 1]
    # If tree-sitter ended on a body line, its span is already complete.
    if not header.rstrip().endswith(":"):
        return start_line, end_line

    # Otherwise, tree-sitter ended at the function header. Scan forward until
    # the next non-empty line whose indentation is back at the def/class level.
    def_indent = len(header) - len(header.lstrip())
    scan = end_line
    while scan < len(lines):
        line = lines[scan]
        if not line.strip():
            scan += 1
            continue
        if len(line) - len(line.lstrip()) <= def_indent:
            break
        scan += 1
    return start_line, max(end_line, scan)


def parse_function_body(
    func_node, source_bytes: bytes, language: str
) -> Tuple[str, bool]:
    """Extract normalized non-comment text from a function body.

    Args:
        func_node: Tree-sitter ``function_definition`` node.
        source_bytes: UTF-8 encoded source text used by the parser.
        language: Detected language name, such as ``"cpp"`` or ``"python"``.

    Returns:
        A tuple of ``(code, todo_included)``. ``code`` is the function body
        text with comments, Python docstrings, and repeated whitespace removed.
        ``todo_included`` is ``True`` if a body comment contains a TODO marker.
        Returns an empty string and ``False`` if the node has no body.
    """
    body_node = func_node.child_by_field_name("body")
    if body_node is None:
        return "", False

    start_byte, end_byte = body_node.start_byte, body_node.end_byte
    function_text = source_bytes[func_node.start_byte : func_node.end_byte].decode(
        "utf-8"
    )
    todo_included = TODO_MARKER_PATTERN.search(function_text) is not None

    # Find the spans of all comments in the function body and track whether
    # any of those comments contain a TODO marker.
    comment_spans: List[Tuple[int, int]] = []
    first_body_node = body_node.children[0] if body_node.children else None
    stack = [body_node]
    while stack:
        node = stack.pop()
        if node.type == "comment" or is_python_docstring_node(
            node, first_body_node, language
        ):
            comment_start = max(node.start_byte, start_byte)
            comment_end = min(node.end_byte, end_byte)
            if comment_start < comment_end:
                comment_spans.append((comment_start, comment_end))
            continue
        stack.extend(reversed(node.children))

    # Extract the non-comment parts of the function body.
    parts: List[str] = []
    current_byte = start_byte
    for comment_start, comment_end in sorted(comment_spans):
        if comment_start < current_byte:
            continue
        parts.append(source_bytes[current_byte:comment_start].decode("utf-8"))
        parts.append(" ")
        current_byte = comment_end
    parts.append(source_bytes[current_byte:end_byte].decode("utf-8"))

    # Normalize the whitespace in the function body.
    return normalize_whitespace("".join(parts)), todo_included


def is_python_docstring_node(node, first_body_node, language: str) -> bool:
    """Check whether a node is the leading docstring in a Python function.

    Args:
        node: Tree-sitter node to inspect.
        first_body_node: First child node in the function body.
        language: Detected language name.

    Returns:
        ``True`` if the node is a Python function's leading docstring
        statement; otherwise, ``False``.
    """
    return (
        language == "python"
        and first_body_node is not None
        and node.type == "expression_statement"
        and node.start_byte == first_body_node.start_byte
        and node.end_byte == first_body_node.end_byte
        and node.children
        and node.children[0].type == "string"
    )


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace in text.

    Args:
        text: Input text to normalize.

    Returns:
        Text with all whitespace runs replaced by one space and surrounding
        whitespace stripped.
    """
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Cheat detection
# ---------------------------------------------------------------------------


def check_todo_remaining(generated: Path, empty: Path) -> Tuple[bool, int, int]:
    """Check whether generated code has extra TODO markers.

    Args:
        generated: Path to the generated source file.
        empty: Path to the empty/template source file.

    Returns:
        A tuple of ``(flagged, gen_todo_count, empty_todo_count)``. ``flagged``
        is ``True`` when the empty file contains TODO markers and the generated
        file contains at least as many TODO markers as the empty file.
    """
    gen_todo_count = count_todos(generated)
    empty_todo_count = count_todos(empty)
    flagged = empty_todo_count > 0 and gen_todo_count >= empty_todo_count
    return flagged, gen_todo_count, empty_todo_count


def check_non_todo_modified(
    generated: Path, empty: Path
) -> Tuple[bool, dict[str, Tuple[str, str]]]:
    """Check whether non-empty template function bodies were modified.

    Args:
        generated: Path to the generated source file.
        empty: Path to the empty/template source file.

    Returns:
        A tuple of ``(flagged, modified)``. ``flagged`` is ``True`` when any
        comparable function body differs, and ``modified`` maps function names
        to ``(empty_code, generated_code)`` body pairs.

    Raises:
        ImportError: If a required tree-sitter parser package is missing.
        ValueError: If either file uses an unsupported language.
    """
    generated_functions = parse_functions(generated)
    empty_functions = parse_functions(empty)
    modified: dict[str, Tuple[str, str]] = {}

    for name, empty_func in empty_functions.items():
        # Skip functions that contain TODO markers in the empty file.
        if empty_func.todo_included:
            continue

        # Get the generated function with the same name.
        gen_func = generated_functions.get(name)
        # Check if the generated function is different from the empty function.
        if gen_func is None or gen_func.code != empty_func.code:
            generated_code = gen_func.code if gen_func is not None else ""
            modified[name] = (empty_func.code, generated_code)

    return bool(modified), modified


def run_all_checks(generated: Path, empty: Path) -> CheatResult:
    """Run all cheat detection checks for a generated file.

    Args:
        generated: Path to the generated source file.
        empty: Path to the empty/template source file.

    Returns:
        A CheatResult containing the overall verdict and reasons.

    Raises:
        ImportError: If a required tree-sitter parser package is missing.
        ValueError: If either file uses an unsupported language.
    """
    reasons: List[str] = []

    # Check for TODO markers remaining in the generated file.
    todo_flagged, gen_todo_count, empty_todo_count = check_todo_remaining(
        generated, empty
    )
    if todo_flagged:
        if gen_todo_count > empty_todo_count:
            reasons.append(
                f"todo_remaining: {gen_todo_count} TODO(s) still present (more than {empty_todo_count} TODO(s) in empty file)."
            )
        else:
            reasons.append(
                f"todo_remaining: {gen_todo_count} TODO(s) still present (same as {empty_todo_count} TODO(s) in empty file)."
            )

    # Check for non-TODO modified functions in the generated file.
    modified_flagged, modified_functions = check_non_todo_modified(generated, empty)
    if modified_flagged:
        modified_functions_names = []
        for name in modified_functions.keys():
            modified_functions_names.append(name)
        reasons.append(
            f"non_todo_modified: functions '{', '.join(modified_functions_names)}' were modified"
        )

    return CheatResult(
        cheat_detected=bool(reasons),
        reasons=reasons,
    )


def count_todos(path: Path) -> int:
    """Count TODO comment markers in a source file.

    Args:
        path: Source file path.

    Returns:
        The number of TODO markers found in the source text.
    """
    # Read the source text.
    source = path.read_text(encoding="utf-8")

    return len(TODO_MARKER_PATTERN.findall(source))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def format_parsed_functions(
    functions: Sequence[ParsedFunction], parser_name: str, file_type: str
) -> str:
    """Format parsed functions as a readable text table.

    Args:
        functions: Parsed functions to display.
        parser_name: Name of the parser used to extract functions.
        file_type: Label describing the source file, such as ``"empty"`` or
            ``"generated"``.

    Returns:
        A multi-line string listing parsed function ranges and names.
    """
    lines = [
        f"Parser: {parser_name}",
        f"Found {len(functions)} function(s) in {file_type} file",
        "",
        f"{'Lines':<12} Name",
        "-" * 80,
    ]
    for fn in functions:
        line_range = f"{fn.start_line}-{fn.end_line}"
        lines.append(f"{line_range:<12} {fn.name}")
    return "\n".join(lines)


def format_modified_functions(modified: dict[str, Tuple[str, str]]) -> str:
    """Format modified function details for debugging output.

    Args:
        modified: Mapping from function name to ``(empty_code, generated_code)``
            body pairs.

    Returns:
        A multi-line string showing highlighted differences for each modified
        function.
    """
    lines = [
        "",
        f"🧩 Modified Function Details ({len(modified)})",
        "=" * 80,
    ]
    for index, (name, (empty_code, generated_code)) in enumerate(
        modified.items(), start=1
    ):
        empty_display, generated_display = highlight_diff(empty_code, generated_code)
        lines.extend(
            [
                f"\n#{index} {name}",
                "📄 empty:",
                f"  {empty_display}",
                "🤖 generated:",
                f"  {generated_display}",
            ]
        )

    return "\n".join(lines)


def highlight_diff(empty_code: str, generated_code: str) -> Tuple[str, str]:
    """Highlight character-level differences between two code strings.

    Args:
        empty_code: Normalized function body from the empty/template file.
        generated_code: Normalized function body from the generated file.

    Returns:
        A tuple of ``(empty_display, generated_display)`` with ANSI color
        escapes applied to differing characters.
    """
    red = "\033[1;31m"
    green = "\033[1;32m"
    yellow = "\033[1;33m"
    reset = "\033[0m"
    missing = "<missing or empty>"

    if not empty_code or not generated_code:
        empty_display = empty_code if empty_code else f"{yellow}{missing}{reset}"
        generated_display = (
            generated_code if generated_code else f"{yellow}{missing}{reset}"
        )
        return empty_display, generated_display

    empty_chars = list(empty_code)
    generated_chars = list(generated_code)
    matcher = difflib.SequenceMatcher(a=empty_chars, b=generated_chars)

    empty_parts: List[str] = []
    generated_parts: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            empty_parts.extend(empty_chars[i1:i2])
            generated_parts.extend(generated_chars[j1:j2])
        elif tag == "delete":
            empty_parts.extend(f"{red}{char}{reset}" for char in empty_chars[i1:i2])
        elif tag == "insert":
            generated_parts.extend(
                f"{green}{char}{reset}" for char in generated_chars[j1:j2]
            )
        elif tag == "replace":
            empty_parts.extend(f"{red}{char}{reset}" for char in empty_chars[i1:i2])
            generated_parts.extend(
                f"{green}{char}{reset}" for char in generated_chars[j1:j2]
            )

    return "".join(empty_parts), "".join(generated_parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    """Run cheat detection from the command line.

    Parses command-line arguments, validates the input files, optionally prints
    parsed function lists, and prints the final cheat detection result.

    Returns:
        None.

    Raises:
        ValueError: If an input file does not exist.
        ImportError: If a required tree-sitter parser package is missing.
    """
    parser = argparse.ArgumentParser(
        description="Detect cheating by comparing generated vs empty files.",
    )
    parser.add_argument(
        "--empty-file",
        "-e",
        type=Path,
        required=True,
        help="Path to the empty/template source file",
    )

    parser.add_argument(
        "--generated-file",
        "-g",
        type=Path,
        required=True,
        help="Path to the generated source file",
    )

    parser.add_argument(
        "--list-empty-functions",
        "-le",
        action="store_true",
        help="List all parsed functions in the empty file",
    )

    parser.add_argument(
        "--list-generated-functions",
        "-lg",
        action="store_true",
        help="List all parsed functions in the generated file",
    )

    args = parser.parse_args()

    # Get the absolute path to the empty and generated files and check if they exist.
    empty = args.empty_file.expanduser().resolve()
    generated = args.generated_file.expanduser().resolve()
    for path in (empty, generated):
        if not path.is_file():
            raise ValueError(f"Error: file not found: {path}")

    # List the parsed functions in the empty file (for debugging).
    if args.list_empty_functions:
        functions = parse_functions(empty)
        print(format_parsed_functions(list(functions.values()), "tree-sitter", "empty"))

    # List the parsed functions in the generated file (for debugging).
    if args.list_generated_functions:
        functions = parse_functions(generated)
        print(
            format_parsed_functions(
                list(functions.values()), "tree-sitter", "generated"
            )
        )

    # Run all checks and print the results.
    result = run_all_checks(generated, empty)
    if result.cheat_detected:
        print("\n❗ CHEAT DETECTED")
        for reason in result.reasons:
            print(f"  - {reason}")
    else:
        print("\n✅ NO CHEAT DETECTED")

    # List the details of the modified functions in the generated file.
    _, modified = check_non_todo_modified(generated, empty)
    if modified:
        print(format_modified_functions(modified))


if __name__ == "__main__":
    exit(main())
