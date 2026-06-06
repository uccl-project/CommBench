#!/usr/bin/env bash
# Idempotent installer for the vendored vLLM checkout under third_party/vllm.
#
# We install vLLM in editable mode using its precompiled CUDA artifacts so the
# dataset examples can `import vllm` without rebuilding C++/CUDA code on every
# run.  Subsequent invocations short-circuit when the installed version
# already matches the checkout.
#
# Usage:
#   bash datasets/third_party/vllm_setup.sh           # use auto-detected python
#   PYTHON=/path/to/python bash .../vllm_setup.sh     # pin a specific interpreter
#
# Environment:
#   PYTHON          Python interpreter to install into (default: `which python`).
#   VLLM_REPO_DIR   Override the source checkout (default: third_party/vllm).
#   FORCE_REINSTALL If set to 1, reinstall even if already present.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_REPO_DIR="${VLLM_REPO_DIR:-$THIS_DIR/vllm}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

if [[ ! -d "$VLLM_REPO_DIR" ]]; then
    echo "[vllm_setup] ERROR: vLLM checkout not found at $VLLM_REPO_DIR" >&2
    echo "[vllm_setup] Place the repo there (or symlink) and re-run." >&2
    exit 1
fi

if [[ -z "$PYTHON" ]]; then
    echo "[vllm_setup] ERROR: no python interpreter found" >&2
    exit 1
fi

# Resolve symlink to canonical path so editable install survives renames.
VLLM_REPO_REAL="$(readlink -f "$VLLM_REPO_DIR")"

echo "[vllm_setup] python    : $PYTHON ($($PYTHON --version 2>&1))"
echo "[vllm_setup] vllm repo : $VLLM_REPO_REAL"

# Skip if vllm is already importable AND points at our checkout AND forced reinstall not requested.
if [[ "${FORCE_REINSTALL:-0}" != "1" ]]; then
    # Run the check from /tmp so that any local `vllm/` directory in cwd
    # cannot shadow the installed package as a namespace package.
    if (cd /tmp && "$PYTHON" - <<PY 2>/dev/null
import importlib.util, os, sys
expected = os.path.realpath(os.path.join("$VLLM_REPO_REAL", "vllm"))
spec = importlib.util.find_spec("vllm")
if spec is None:
    sys.exit(1)
candidates = []
if spec.origin:
    candidates.append(os.path.dirname(os.path.realpath(spec.origin)))
for loc in (spec.submodule_search_locations or []):
    candidates.append(os.path.realpath(loc))
sys.exit(0 if expected in candidates else 1)
PY
    )
    then
        echo "[vllm_setup] vllm already installed editable from $VLLM_REPO_REAL — nothing to do."
        echo "[vllm_setup] (set FORCE_REINSTALL=1 to override.)"
        exit 0
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[vllm_setup] ERROR: 'uv' is required (https://github.com/astral-sh/uv)" >&2
    exit 1
fi

# Install into the env owning $PYTHON; uv pip honours $VIRTUAL_ENV / current python.
PY_PREFIX="$("$PYTHON" -c 'import sys; print(sys.prefix)')"

echo "[vllm_setup] installing vllm (precompiled, editable) into $PY_PREFIX ..."
VLLM_USE_PRECOMPILED=1 VIRTUAL_ENV="$PY_PREFIX" \
    uv pip install -e "$VLLM_REPO_REAL" --torch-backend=auto

echo "[vllm_setup] done."
