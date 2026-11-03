#!/usr/bin/env bash
# ==============================================================================
# Chunk Filtering & Semantic Redundancy Benchmark Runner
#
# Runs benchmark test sweeps across chunk similarity thresholds (0.92 - 0.96 and 1.00 baseline),
# builds Knowledge Graphs from scratch while preserving atomic chunk and state caches (--keep-book-cache),
# evaluates graph space coverage via `bookeeper verify -m kg`, saves reports into a dedicated
# reports directory, and outputs an analysis prompt to generate an empirical report matching docs/filters.md.
#
# Usage:
#   ./scripts/run_filter_benchmark.sh [OPTIONS]
#
# Options:
#   -o, --output-dir DIR      Custom directory to store reports (default: reports/filter_benchmark_<timestamp>)
#   -c, --config FILE         Path to base config.yaml (default: config.yaml)
#   -n, --limit INT           Maximum number of books to index in build-graph (default: 60)
#   -t, --thresholds "LIST"   Threshold values to sweep (default: "0.92 0.93 0.94 0.95 0.96 1.00")
#   --tau FLOAT               Semantic proximity radius for verify -m kg (default: 0.75)
#   --skip-build              Skip build-graph and run only verify / generate report prompt from existing data
#   --dry-run                 Print scheduled commands without executing
#   -h, --help                Show this help message
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

BASE_CONFIG="config.yaml"
OUTPUT_DIR="reports/filter_benchmark_$(date +%Y%m%d_%H%M%S)"
THRESHOLDS=("0.92" "0.93" "0.94" "0.95" "0.96" "1.00")
VERIFY_TAU="0.75"
BOOK_LIMIT=60
SKIP_BUILD=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -c|--config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    -n|--limit)
      BOOK_LIMIT="$2"
      shift 2
      ;;
    -t|--thresholds)
      read -ra THRESHOLDS <<< "$2"
      shift 2
      ;;
    --tau)
      VERIFY_TAU="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      sed -ne '/^#/!q;s/^# //;2,$p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

DIFFS_DIR="${OUTPUT_DIR}/diffs"
REPORTS_DIR="${OUTPUT_DIR}/reports"

mkdir -p "${DIFFS_DIR}" "${REPORTS_DIR}"

LIMIT_ARGS=()
if [ -n "${BOOK_LIMIT}" ] && [ "${BOOK_LIMIT}" -gt 0 ] 2>/dev/null; then
  LIMIT_ARGS=("--limit" "${BOOK_LIMIT}")
  LIMIT_DESC="${BOOK_LIMIT}"
else
  LIMIT_DESC="Unlimited (all books)"
fi

echo "=============================================================================="
echo "🧹 Bookeeper Chunk Similarity Pruning Benchmark Sweep"
echo "=============================================================================="
echo "• Base Config:       ${BASE_CONFIG}"
echo "• Output Directory:  ${OUTPUT_DIR}"
echo "• Book Limit:        ${LIMIT_DESC}"
echo "• Threshold Sweep:   ${THRESHOLDS[*]}"
echo "• Verification Tau:  ${VERIFY_TAU}"
echo "• Mode:              Build from scratch (preserving chunk & state cache: --keep-book-cache)"
echo "=============================================================================="

REPORT_FILES=()

for TAU in "${THRESHOLDS[@]}"; do
  DIFF_FILE="${DIFFS_DIR}/diff_tau_${TAU}.yaml"
  REPORT_FILE="${REPORTS_DIR}/graph_space_report_tau_${TAU}.json"
  REPORT_FILES+=("${REPORT_FILE}")

  echo ""
  echo "------------------------------------------------------------------------------"
  echo "▶ Processing Threshold Tau = ${TAU}"
  echo "------------------------------------------------------------------------------"

  # 1. Generate YAML config diff
  if [ "${TAU}" = "1.00" ]; then
    cat <<EOF > "${DIFF_FILE}"
filter:
  chunk_similarity:
    enabled: false # Disable intra-section chunk similarity pruning (Baseline)
EOF
  else
    cat <<EOF > "${DIFF_FILE}"
filter:
  chunk_similarity:
    enabled: true  # Enable drop by chunk similarity (Least priority: formulas are never dropped)
    threshold: ${TAU} # Cosine similarity threshold (>=) within section to drop chunk as redundant
EOF
  fi

  echo "✓ Generated config diff: ${DIFF_FILE}"

  if [ "${DRY_RUN}" = true ]; then
    echo "[DRY-RUN] uv run bookeeper build-graph --from-scratch --keep-book-cache ${LIMIT_ARGS[*]} --config \"${BASE_CONFIG}\" --config-diff \"${DIFF_FILE}\""
    echo "[DRY-RUN] uv run bookeeper verify -m kg --tau \"${VERIFY_TAU}\" --config \"${BASE_CONFIG}\" --config-diff \"${DIFF_FILE}\" -o \"${REPORT_FILE}\""
    continue
  fi

  # 2. Build graph from scratch preserving chunk and state cache
  if [ "${SKIP_BUILD}" = false ]; then
    echo "🔨 Building Knowledge Graph from scratch (--keep-book-cache${LIMIT_ARGS:+ ${LIMIT_ARGS[*]}})..."
    uv run bookeeper build-graph \
      --from-scratch \
      --keep-book-cache \
      "${LIMIT_ARGS[@]}" \
      --config "${BASE_CONFIG}" \
      --config-diff "${DIFF_FILE}"
  else
    echo "⏭ Skipping build-graph (--skip-build specified)."
  fi

  # 3. Verify Knowledge Graph space coverage
  echo "🔍 Evaluating Graph Embedding Space (verify -m kg)..."
  uv run bookeeper verify \
    -m kg \
    --tau "${VERIFY_TAU}" \
    --config "${BASE_CONFIG}" \
    --config-diff "${DIFF_FILE}" \
    -o "${REPORT_FILE}"

  echo "✓ Saved verification report: ${REPORT_FILE}"
done

echo ""
echo "=============================================================================="
echo "📊 Generating Comparison Table & Analysis Prompt"
echo "=============================================================================="

PROMPT_FILE="${OUTPUT_DIR}/analysis_prompt.txt"

# Run Python helper to aggregate JSON reports, render markdown table, and synthesize the prompt
REPORTS_DIR="${REPORTS_DIR}" THRESHOLDS="${THRESHOLDS[*]}" uv run python3 - <<'EOF' > "${PROMPT_FILE}"
import json
import os
import sys
from pathlib import Path

reports_dir = Path(os.environ.get("REPORTS_DIR", "."))
thresholds = os.environ.get("THRESHOLDS", "").split()
report_map = {}

for tau in thresholds:
    fpath = reports_dir / f"graph_space_report_tau_{tau}.json"
    if fpath.is_file():
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                report_map[tau] = (fpath, json.load(f))
        except Exception as e:
            pass

print("# ==============================================================================")
print("# PROMPT: Generating Empirical Verification & Benchmark Analysis (docs/filters.md)")
print("# ==============================================================================")
print("")
print("Please prepare an empirical analysis report similar to docs/filters.md based on the")
print("following benchmark results across chunk similarity thresholds:")
print("")
print("## Available Reports:")
for tau, (path, data) in report_map.items():
    label = "Aggressive" if tau == "0.92" else ("Recommended" if tau == "0.95" else ("Conservative" if tau == "0.96" else ("Baseline" if tau == "1.00" else "")))
    label_str = f" ({label})" if label else ""
    print(f"- tau = {tau}{label_str}: {path}")

print("")
print("## Master Comparison Matrix (Extracted from JSON reports):")
print("")

# Header
cols = [f"tau = {t}" for t in thresholds if t in report_map]
header_row = "| Metric | " + " | ".join(cols) + " |"
separator_row = "| :--- | " + " | ".join([":---:"] * len(cols)) + " |"
print(header_row)
print(separator_row)

def get_val(tau, key, default="N/A", fmt=None):
    if tau not in report_map:
        return default
    data = report_map[tau][1]
    keys = key.split(".")
    val = data
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    if val is None:
        return default
    if fmt:
        return fmt(val)
    return str(val)

# Extract metrics
u_chunks = [get_val(t, "universe_chunks_count", fmt=lambda x: f"{x:,}") for t in cols]
p_chunks = [get_val(t.split()[-1], "parent_chunks_count", fmt=lambda x: f"{x:,}") for t in cols]
s_chunks = [get_val(t.split()[-1], "skipped_chunks_count", fmt=lambda x: f"{x:,}") for t in cols]

def calc_skip_pct(t):
    tau_clean = t.split()[-1]
    if tau_clean not in report_map:
        return "N/A"
    d = report_map[tau_clean][1]
    u = d.get("universe_chunks_count") or 0
    s = d.get("skipped_chunks_count") or 0
    return f"{(s / u * 100):.2f}%" if u > 0 else "N/A"

skip_pcts = [calc_skip_pct(t) for t in cols]

base_skipped = report_map.get("1.00", [None, {}])[1].get("skipped_chunks_count") or 0
def calc_extra_pruned(t):
    tau_clean = t.split()[-1]
    if tau_clean not in report_map:
        return "N/A"
    s = report_map[tau_clean][1].get("skipped_chunks_count") or 0
    diff = s - base_skipped
    return f"+{diff:,}" if diff > 0 else ("0" if diff == 0 else f"{diff:,}")

extra_pruned = [calc_extra_pruned(t) for t in cols]
g_nodes = [get_val(t.split()[-1], "nodes_count", fmt=lambda x: f"{x:,}") for t in cols]
raw_ideas = [get_val(t.split()[-1], "postprocessing_drift.raw_ideas_count", fmt=lambda x: f"{x:,}") for t in cols]
chamfer = [get_val(t.split()[-1], "chamfer_mean", fmt=lambda x: f"{x:.4f}") for t in cols]
p10 = [get_val(t.split()[-1], "worst_decile_p10", fmt=lambda x: f"{x:.4f}") for t in cols]
cov = [get_val(t.split()[-1], "coverage_at_tau_pct", fmt=lambda x: f"{x:.2f}%") for t in cols]
vendi = [get_val(t.split()[-1], "vendi_diversity_score", fmt=lambda x: f"{x:.2f}") for t in cols]
skipped_ret = [get_val(t.split()[-1], "skipped_retention_pct", fmt=lambda x: f"{x:.2f}%") for t in cols]

print("| **Universe Chunks** | " + " | ".join(u_chunks) + " |")
print("| **Parent Chunks Extracted** | " + " | ".join(p_chunks) + " |")
print("| **Skipped Chunks** | " + " | ".join(s_chunks) + " |")
print("| **Total Skipped %** | " + " | ".join(skip_pcts) + " |")
print("| **Extra Chunks Pruned vs Base** | " + " | ".join(extra_pruned) + " |")
print("| **Concept Graph Nodes** | " + " | ".join(g_nodes) + " |")
print("| **Raw Ideas Extracted** | " + " | ".join(raw_ideas) + " |")
print("| **Chamfer Mean Alignment** | " + " | ".join(chamfer) + " |")
print("| **Worst Decile ($p_{10}$)** | " + " | ".join(p10) + " |")
print("| **Coverage at $\\\\tau = 0.75$** | " + " | ".join(cov) + " |")
print("| **Vendi Diversity Score** | " + " | ".join(vendi) + " |")
print("| **Skipped Retention ($R_{\\\\text{skipped}}$)** | " + " | ".join(skipped_ret) + " |")

print("")
print("## Required Report Structure & Topics to Cover:")
print("1. **Master Comparison Matrix**: Integrate the markdown table above.")
print("2. **Nominal Pruning Stat vs. Real Skip Rate in Production**:")
print("   - Explain the difference between gross theoretical chunk pruning rate and live execution skip rate.")
print("   - Highlight why math formulas (Priority 1 Math Shield) are unconditionally protected with FilterDecision.RETAIN.")
print("   - Highlight section anchor preservation and substantive expansion guards.")
print("3. **Key Benchmark Insights & ASCII Curve**:")
print("   - Draw the Chamfer Mean vs Pruning Rate ASCII curve highlighting the Pareto optimal knee.")
print("   - Detail why tau = 0.95 is optimal (pruning rate, compute savings, Chamfer retention, peak skipped retention).")
print("   - Detail the near-lossless boundary at tau = 0.96.")
print("   - Discuss why Vendi Diversity Score remains invariant across pruning thresholds.")
print("4. **Recommended Configuration Profiles**:")
print("   - Profile 1: Recommended Production Profile (tau = 0.95)")
print("   - Profile 2: High-Fidelity / Conservative Profile (tau = 0.96)")
print("   - Profile 3: Aggressive Compression Profile (tau = 0.92)")
EOF

# Output to terminal
cat "${PROMPT_FILE}"

echo ""
echo "=============================================================================="
echo "✓ Benchmark sweep complete!"
echo "• Report JSONs:  ${REPORTS_DIR}"
echo "• Prompt file:   ${PROMPT_FILE}"
echo "=============================================================================="
