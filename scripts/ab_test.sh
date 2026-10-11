#!/usr/bin/env bash
# ==============================================================================
# Independent A/B Testing Runner for Bookeeper Branches & VS Mode
#
# Usage:
#   # 1. Standard Branch vs Branch:
#   ./ab_test.sh <branch1> <branch2> [num_books=5] [percent=1.0] [extra_args...]
#
#   # 2. VS Mode (Existing Baseline Report vs Candidate Branch):
#   ./ab_test.sh vs <baseline_report_path> <branch2> [num_books=5] [percent=1.0] [extra_args...]
#   ./ab_test.sh --vs <baseline_report_path> <branch2> [num_books=5] [percent=1.0] [extra_args...]
#
#   # 3. Re-verify Mode (Reuse Knowledge Graphs & Re-run Verification with New Percent):
#   ./ab_test.sh reverify <path_to_abtest_result> [percent=5.0] [extra_args...]
#   ./ab_test.sh --reverify <path_to_abtest_result> [percent=5.0] [extra_args...]
#
#   # 4. Config Customization (Auto-applied or explicit):
#   #   Places config-a.yaml and config-b.yaml in repo root to automatically overlay
#   #   over config.yaml for Branch A and Branch B respectively.
#   #   Or pass explicitly:
#   ./ab_test.sh <branch1> <branch2> 5 1.0 --config-a ./custom-a.yaml --config-b ./custom-b.yaml
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || echo "/home/ubuntu/repos/bookeeper")"
PYTHON_EXEC="${REPO_DIR}/.venv/bin/python"

if [ ! -x "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="${SCRIPT_DIR}/repos/bookeeper/.venv/bin/python"
fi

if [ ! -x "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="$(command -v python3 || command -v python)"
fi

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    exec "$PYTHON_EXEC" "${SCRIPT_DIR}/ab_test.py" --help
fi

# Check for cross-check mode keyword or flag
if [ "${1:-}" = "cross-check" ] || [ "${1:-}" = "--cross-check" ]; then
    if [ $# -lt 2 ]; then
        echo "Error: In 'cross-check' mode, specify path to previous A/B test run directory."
        echo "Usage: $0 cross-check <path_to_abtest_result> [number_blocks=20] [extra_args...]"
        exit 1
    fi
    RUN_PATH="$2"
    BLOCKS="${3:-20}"
    shift 2
    if [ $# -ge 1 ]; then shift; fi
    exec "$PYTHON_EXEC" "${SCRIPT_DIR}/ab_test.py" cross-check "$RUN_PATH" --blocks "$BLOCKS" "$@"
fi

# Check for reverify mode keyword or flag
if [ "${1:-}" = "reverify" ] || [ "${1:-}" = "--reverify" ]; then
    if [ $# -lt 2 ]; then
        echo "Error: In 'reverify' mode, specify path to previous A/B test run directory."
        echo "Usage: $0 reverify <path_to_abtest_result> [percent] [extra_args...]"
        exit 1
    fi
    RUN_PATH="$2"
    PERCENT="${3:-}"
    shift 2
    if [ $# -ge 1 ]; then shift; fi
    ARGS=()
    if [ -n "$PERCENT" ]; then ARGS+=(--percent "$PERCENT"); fi
    exec "$PYTHON_EXEC" "${SCRIPT_DIR}/ab_test.py" reverify "$RUN_PATH" "${ARGS[@]}" "$@"
fi

# Check for VS mode keyword or flag
if [ "${1:-}" = "vs" ] || [ "${1:-}" = "--vs" ]; then
    if [ $# -lt 3 ]; then
        echo "Error: In 'vs' mode, specify baseline report path and candidate branch."
        echo "Usage: $0 vs <baseline_report.json> <candidate_branch> [num_books] [percent] [extra_args...]"
        exit 1
    fi
    REPORT="$2"
    BRANCH2="$3"
    BOOKS="${4:-}"
    PERCENT="${5:-}"
    shift 3
    if [ $# -ge 1 ]; then shift; fi
    if [ $# -ge 1 ]; then shift; fi
    ARGS=()
    if [ -n "$BOOKS" ]; then ARGS+=(--books "$BOOKS"); fi
    if [ -n "$PERCENT" ]; then ARGS+=(--percent "$PERCENT"); fi
    exec "$PYTHON_EXEC" "${SCRIPT_DIR}/ab_test.py" vs "$REPORT" "$BRANCH2" "${ARGS[@]}" "$@"
fi

# Standard 2-branch mode
BRANCH1="${1:-}"
BRANCH2="${2:-}"
BOOKS="${3:-}"
PERCENT="${4:-}"

if [ -z "$BRANCH1" ] || [ -z "$BRANCH2" ]; then
    echo "Usage: $0 <branch1> <branch2> [num_books] [percent] [extra_args...]"
    echo "   or: $0 vs <baseline_report.json> <candidate_branch> [num_books] [percent] [extra_args...]"
    exit 1
fi

shift 2
if [ $# -ge 1 ]; then shift; fi
if [ $# -ge 1 ]; then shift; fi

ARGS=()
if [ -n "$BOOKS" ]; then ARGS+=(--books "$BOOKS"); fi
if [ -n "$PERCENT" ]; then ARGS+=(--percent "$PERCENT"); fi

exec "$PYTHON_EXEC" "${SCRIPT_DIR}/ab_test.py" "$BRANCH1" "$BRANCH2" "${ARGS[@]}" "$@"
