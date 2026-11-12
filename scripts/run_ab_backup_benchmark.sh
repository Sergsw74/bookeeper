#!/usr/bin/env bash
# ==============================================================================
# Headless A/B Benchmark Runner with Knowledge Graph Backups (Option 1)
#
# Builds Knowledge Graphs once per branch from cached book extractions, archives
# each into an isolated backup archive (.zip), and runs an exhaustive suite of
# instant offline comparative validations without any graph regenerations.
#
# Workflow:
#   1. Checkout Branch A (master) -> build-graph (limit N) -> backup to master_kg.zip
#   2. Checkout Branch B (jepa)   -> build-graph (limit N) -> backup to jepa_kg.zip
#   3. Offline Validation Suite:
#      • Topological & Overlap Comparison (ab-compare --mode stats)
#      • Semantic Space Coverage & Diversity (ab-compare --mode graph --tau 0.75)
#      • NLI Atomic Cross-Check Audit (ab-compare --mode cross-check --blocks 20)
#      • Factual Grounding Recall Audit (verify --mode ideas --percent 5.0)
#      • Idea Synthesis Drift Audit (verify --mode drift)
#   4. Restores original branch and writes consolidated markdown report.
#
# Usage:
#   ./scripts/run_ab_backup_benchmark.sh [OPTIONS]
#
# Options:
#   -n, --limit INT           Number of books to index (default: 60)
#   -a, --branch-a BRANCH     Baseline branch name (default: master)
#   -b, --branch-b BRANCH     Candidate branch name (default: feature/jepa-encoder-test)
#   --config-a FILE           Overlay config for Branch A (default: config-a.yaml)
#   --config-b FILE           Overlay config for Branch B (default: config-b.yaml)
#   --backup-a FILE           Path to pre-existing Branch A backup zip (auto-detected if omitted with --skip-a)
#   --backup-b FILE           Path to pre-existing Branch B backup zip (auto-detected if omitted with --skip-b)
#   -o, --output-dir DIR      Results directory (default: reports/ab_backup_YYYYMMDD_HHMMSS)
#   --tau FLOAT               Proximity radius for graph space coverage (default: 0.75)
#   --blocks INT              Contiguous chunk blocks for cross-check (default: 20)
#   --percent FLOAT           Sampling percentage for ideas audit (default: 5.0)
#   --skip-build              Skip both graph builds and run audits on existing backups
#   --skip-a                  Skip Branch A build (reuse existing backup A)
#   --skip-b                  Skip Branch B build (reuse existing backup B)
#   --dry-run                 Show commands without executing
#   -h, --help                Show this help message
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# Defaults
BOOK_LIMIT=60
BRANCH_A="master"
BRANCH_B="feature/jepa-encoder-test"
CONFIG_A="config-a.yaml"
CONFIG_B="config-b.yaml"
CUSTOM_BACKUP_A=""
CUSTOM_BACKUP_B=""
TAU="0.75"
BLOCKS=20
PERCENT="5.0"
SKIP_BUILD=false
SKIP_A=false
SKIP_B=false
DRY_RUN=false
OUTPUT_DIR="reports/ab_backup_$(date +%Y%m%d_%H%M%S)"

# Parse CLI options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--limit)
      BOOK_LIMIT="$2"
      shift 2
      ;;
    -a|--branch-a)
      BRANCH_A="$2"
      shift 2
      ;;
    -b|--branch-b)
      BRANCH_B="$2"
      shift 2
      ;;
    --config-a)
      CONFIG_A="$2"
      shift 2
      ;;
    --config-b)
      CONFIG_B="$2"
      shift 2
      ;;
    --backup-a)
      CUSTOM_BACKUP_A="$2"
      shift 2
      ;;
    --backup-b)
      CUSTOM_BACKUP_B="$2"
      shift 2
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --tau)
      TAU="$2"
      shift 2
      ;;
    --blocks)
      BLOCKS="$2"
      shift 2
      ;;
    --percent)
      PERCENT="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --skip-a)
      SKIP_A=true
      shift
      ;;
    --skip-b)
      SKIP_B=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      sed -n '2,/^# ==============================================================================/p' "$0" | sed 's/^# //' | sed 's/^#//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# Ensure safe environment
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

BACKUP_DIR="${OUTPUT_DIR}/backups"
LOG_FILE="${OUTPUT_DIR}/session.log"

mkdir -p "${OUTPUT_DIR}" "${BACKUP_DIR}"

# Logging helper
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo -e "${msg}" | tee -a "${LOG_FILE}"
}

# Error handler for unexpected failures
error_handler() {
  local exit_code=$1
  local line_no=$2
  echo "" >&2
  log "❌ Error: Command failed at line ${line_no} with exit code ${exit_code}."
  log "   Full execution log saved at: ${LOG_FILE}"
}
trap 'error_handler $? $LINENO' ERR

# Safe git checkout that clears transient uv.lock modifications
safe_checkout() {
  local target_branch="$1"
  local cur
  cur="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "${cur}" == "${target_branch}" ]]; then
    log "✓ Already on branch: ${target_branch}"
    return 0
  fi
  log "Switching to git branch: ${target_branch}..."
  # Clean ephemeral uv lockfile modifications before switching
  git checkout -- uv.lock 2>/dev/null || git restore uv.lock 2>/dev/null || true
  if ! git checkout "${target_branch}" 2>&1 | tee -a "${LOG_FILE}"; then
    log "❌ Failed to switch to branch '${target_branch}'! See above for git error."
    return 1
  fi
  log "✓ Successfully checked out: ${target_branch}"
}

# Resolve Backup A path
if [[ -n "${CUSTOM_BACKUP_A}" ]]; then
  BACKUP_A="${CUSTOM_BACKUP_A}"
elif [[ "${SKIP_BUILD}" == "true" || "${SKIP_A}" == "true" ]]; then
  # Auto-discover most recent existing backup for Branch A
  LATEST_FOUND=$(find reports/ -name "*${BRANCH_A//\//_}*kg_${BOOK_LIMIT}.zip" -o -name "master_kg_${BOOK_LIMIT}.zip" 2>/dev/null | sort -r | head -n 1 || true)
  if [[ -n "${LATEST_FOUND}" && -f "${LATEST_FOUND}" ]]; then
    BACKUP_A="${LATEST_FOUND}"
    log "🔍 Auto-detected existing Branch A backup archive: ${BACKUP_A}"
  else
    BACKUP_A="${BACKUP_DIR}/${BRANCH_A//\//_}_kg_${BOOK_LIMIT}.zip"
  fi
else
  BACKUP_A="${BACKUP_DIR}/${BRANCH_A//\//_}_kg_${BOOK_LIMIT}.zip"
fi

# Resolve Backup B path
if [[ -n "${CUSTOM_BACKUP_B}" ]]; then
  BACKUP_B="${CUSTOM_BACKUP_B}"
elif [[ "${SKIP_BUILD}" == "true" || "${SKIP_B}" == "true" ]]; then
  # Auto-discover most recent existing backup for Branch B
  LATEST_FOUND=$(find reports/ -name "*${BRANCH_B//\//_}*kg_${BOOK_LIMIT}.zip" -o -name "*jepa*kg_${BOOK_LIMIT}.zip" 2>/dev/null | sort -r | head -n 1 || true)
  if [[ -n "${LATEST_FOUND}" && -f "${LATEST_FOUND}" ]]; then
    BACKUP_B="${LATEST_FOUND}"
    log "🔍 Auto-detected existing Branch B backup archive: ${BACKUP_B}"
  else
    BACKUP_B="${BACKUP_DIR}/${BRANCH_B//\//_}_kg_${BOOK_LIMIT}.zip"
  fi
else
  BACKUP_B="${BACKUP_DIR}/${BRANCH_B//\//_}_kg_${BOOK_LIMIT}.zip"
fi

# Remember starting git branch for graceful rollback
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
cleanup_branch() {
  local exit_code=$?
  # Discard ephemeral uv lockfile changes
  git checkout -- uv.lock 2>/dev/null || git restore uv.lock 2>/dev/null || true
  if [[ "${CURRENT_BRANCH}" != "$(git rev-parse --abbrev-ref HEAD)" ]]; then
    echo ""
    echo "↩️ Restoring starting git branch: ${CURRENT_BRANCH}..."
    git checkout "${CURRENT_BRANCH}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup_branch EXIT INT TERM

log "=============================================================================="
log "🚀 Headless A/B Benchmark (Option 1: Backup & Offline Multi-Audit)"
log "=============================================================================="
log "• Target Books:      ${BOOK_LIMIT} (reusing data/book_states_cache)"
log "• Branch A (Base):   ${BRANCH_A} (config: ${CONFIG_A})"
log "• Branch B (Cand):   ${BRANCH_B} (config: ${CONFIG_B})"
log "• Output Directory:  ${OUTPUT_DIR}"
log "• Backup A Archive:  ${BACKUP_A}"
log "• Backup B Archive:  ${BACKUP_B}"
log "• Evaluation Params: tau=${TAU}, blocks=${BLOCKS}, percent=${PERCENT}%"
log "=============================================================================="

# Verify state cache exists
if [[ ! -d "data/book_states_cache" ]]; then
  log "❌ Error: 'data/book_states_cache' directory not found!"
  log "Please ensure book states are placed in data/book_states_cache before running."
  exit 1
fi

CACHED_COUNT=$(ls -1 data/book_states_cache/book_*_state.json 2>/dev/null | wc -l || echo 0)
log "✓ Protected state cache verified: ${CACHED_COUNT} book state files available."

# ------------------------------------------------------------------------------
# Phase 1: Build & Archive Branch A (master)
# ------------------------------------------------------------------------------
TIME_A=0
if [[ "${SKIP_BUILD}" == "true" || "${SKIP_A}" == "true" ]]; then
  if [[ ! -f "${BACKUP_A}" ]]; then
    log "❌ Error: Branch A backup archive '${BACKUP_A}' does not exist!"
    log "Cannot skip Branch A without a valid existing backup. Pass --backup-a <path> or run without --skip-a."
    exit 1
  fi
  log "⏭️ Skipping Branch A build (reusing existing backup: ${BACKUP_A})"
else
  log ""
  log "──────────────────────────────────────────────────────────────────────────────"
  log "🔨 Phase 1/4: Building Knowledge Graph on Branch A (${BRANCH_A})..."
  log "──────────────────────────────────────────────────────────────────────────────"
  
  if [[ "${DRY_RUN}" == "true" ]]; then
    log "[DRY-RUN] git checkout ${BRANCH_A}"
    log "[DRY-RUN] uv run bookeeper build-graph --limit ${BOOK_LIMIT} --from-scratch --state-cache-dir data/book_states_cache --config-diff ${CONFIG_A} --no-lightrag"
    log "[DRY-RUN] uv run bookeeper backup -t knowledge -o ${BACKUP_A}"
  else
    safe_checkout "${BRANCH_A}"
    
    START_A=$(date +%s)
    uv run bookeeper build-graph \
      --limit "${BOOK_LIMIT}" \
      --from-scratch \
      --state-cache-dir data/book_states_cache \
      --config-diff "${CONFIG_A}" \
      --no-lightrag 2>&1 | tee -a "${LOG_FILE}"
    END_A=$(date +%s)
    TIME_A=$((END_A - START_A))
    
    log "📦 Archiving Branch A knowledge graph to ${BACKUP_A}..."
    uv run bookeeper backup -t knowledge -o "${BACKUP_A}" 2>&1 | tee -a "${LOG_FILE}"
    log "✓ Branch A completed in ${TIME_A}s. Backup size: $(du -sh "${BACKUP_A}" | awk '{print $1}')"
  fi
fi

# ------------------------------------------------------------------------------
# Phase 2: Build & Archive Branch B (feature/jepa-encoder-test)
# ------------------------------------------------------------------------------
TIME_B=0
if [[ "${SKIP_BUILD}" == "true" || "${SKIP_B}" == "true" ]]; then
  if [[ ! -f "${BACKUP_B}" ]]; then
    log "❌ Error: Branch B backup archive '${BACKUP_B}' does not exist!"
    log "Cannot skip Branch B without a valid existing backup. Pass --backup-b <path> or run without --skip-b."
    exit 1
  fi
  log "⏭️ Skipping Branch B build (reusing existing backup: ${BACKUP_B})"
else
  log ""
  log "──────────────────────────────────────────────────────────────────────────────"
  log "🔨 Phase 2/4: Building Knowledge Graph on Branch B (${BRANCH_B})..."
  log "──────────────────────────────────────────────────────────────────────────────"
  
  if [[ "${DRY_RUN}" == "true" ]]; then
    log "[DRY-RUN] git checkout ${BRANCH_B}"
    log "[DRY-RUN] uv run bookeeper build-graph --limit ${BOOK_LIMIT} --from-scratch --state-cache-dir data/book_states_cache --config-diff ${CONFIG_B} --no-lightrag"
    log "[DRY-RUN] uv run bookeeper backup -t knowledge -o ${BACKUP_B}"
  else
    safe_checkout "${BRANCH_B}"
    
    START_B=$(date +%s)
    uv run bookeeper build-graph \
      --limit "${BOOK_LIMIT}" \
      --from-scratch \
      --state-cache-dir data/book_states_cache \
      --config-diff "${CONFIG_B}" \
      --no-lightrag 2>&1 | tee -a "${LOG_FILE}"
    END_B=$(date +%s)
    TIME_B=$((END_B - START_B))
    
    log "📦 Archiving Branch B knowledge graph to ${BACKUP_B}..."
    uv run bookeeper backup -t knowledge -o "${BACKUP_B}" 2>&1 | tee -a "${LOG_FILE}"
    log "✓ Branch B completed in ${TIME_B}s. Backup size: $(du -sh "${BACKUP_B}" | awk '{print $1}')"
  fi
fi

# ------------------------------------------------------------------------------
# Phase 3: Instant Offline Comparative Validation Suite (0 Rebuilds)
# ------------------------------------------------------------------------------
log ""
log "──────────────────────────────────────────────────────────────────────────────"
log "🔬 Phase 3/4: Executing Offline Validation Suite Against Backups..."
log "──────────────────────────────────────────────────────────────────────────────"

# Pre-extract lightweight targets (knowledge_graph.json) to avoid unpacking 60,000 files repeatedly into /tmp
TARGET_A="${BACKUP_A}"
TARGET_B="${BACKUP_B}"
DIR_A=""
DIR_B=""

if [[ "${BACKUP_A}" =~ \.zip$ || "${BACKUP_A}" =~ \.7z$ ]]; then
  DIR_A="${OUTPUT_DIR}/target_a"
  mkdir -p "${DIR_A}"
  if [[ ! -f "${DIR_A}/knowledge_graph.json" ]]; then
    log "📦 Extracting knowledge graph from ${BACKUP_A} into ${DIR_A}..."
    unzip -q -o "${BACKUP_A}" "knowledge_graph.json" -d "${DIR_A}" 2>/dev/null || true
  fi
  if [[ -f "${DIR_A}/knowledge_graph.json" ]]; then
    TARGET_A="${DIR_A}"
  fi
fi

if [[ "${BACKUP_B}" =~ \.zip$ || "${BACKUP_B}" =~ \.7z$ ]]; then
  DIR_B="${OUTPUT_DIR}/target_b"
  mkdir -p "${DIR_B}"
  if [[ ! -f "${DIR_B}/knowledge_graph.json" ]]; then
    log "📦 Extracting knowledge graph from ${BACKUP_B} into ${DIR_B}..."
    unzip -q -o "${BACKUP_B}" "knowledge_graph.json" -d "${DIR_B}" 2>/dev/null || true
  fi
  if [[ -f "${DIR_B}/knowledge_graph.json" ]]; then
    TARGET_B="${DIR_B}"
  fi
fi

# Audit 1: Topological Statistics & Overlap
log "▶ [1/5] Running Topological & Jaccard Overlap Audit (--mode stats)..."
if [[ "${DRY_RUN}" == "false" ]]; then
  uv run bookeeper ab-compare "${TARGET_A}" "${TARGET_B}" --mode stats 2>&1 | tee "${OUTPUT_DIR}/audit_stats.txt" | tee -a "${LOG_FILE}"
fi

# Audit 2: Semantic Space Coverage & Vendi Diversity Suite
log "▶ [2/5] Running Semantic Space Coverage Audit (--mode graph, tau=${TAU})..."
if [[ "${DRY_RUN}" == "false" ]]; then
  uv run bookeeper ab-compare "${TARGET_A}" "${TARGET_B}" --mode graph --tau "${TAU}" 2>&1 | tee "${OUTPUT_DIR}/audit_graph.txt" | tee -a "${LOG_FILE}"
fi

# Audit 3: Deep NLI Atomic Assertion Cross-Check
log "▶ [3/5] Running NLI Atomic Assertion Cross-Check Audit (--mode cross-check, blocks=${BLOCKS})..."
if [[ "${DRY_RUN}" == "false" ]]; then
  uv run bookeeper ab-compare "${TARGET_A}" "${TARGET_B}" --mode cross-check --blocks "${BLOCKS}" --model "qwen2.5:7b-instruct-q4_K_M" --embedding-model bge-m3:latest 2>&1 | tee "${OUTPUT_DIR}/audit_cross_check.txt" | tee -a "${LOG_FILE}"
fi

# Audit 4: Candidate Factual Grounding Recall
log "▶ [4/5] Running Candidate Factual Grounding Audit (verify --mode ideas, percent=${PERCENT}%)..."
if [[ "${DRY_RUN}" == "false" ]]; then
  KG_B_PATH="${TARGET_B}"
  if [[ -d "${TARGET_B}" && -f "${TARGET_B}/knowledge_graph.json" ]]; then
    KG_B_PATH="${TARGET_B}/knowledge_graph.json"
  fi
  uv run bookeeper verify --graph-file "${KG_B_PATH}" --mode ideas --percent "${PERCENT}" --model "qwen2.5:7b-instruct-q4_K_M" 2>&1 | tee "${OUTPUT_DIR}/audit_grounding_b.txt" | tee -a "${LOG_FILE}"
fi

# Audit 5: Candidate Idea Synthesis Drift
log "▶ [5/5] Running Candidate Synthesis Drift Audit (verify --mode drift)..."
if [[ "${DRY_RUN}" == "false" ]]; then
  KG_B_PATH="${TARGET_B}"
  if [[ -d "${TARGET_B}" && -f "${TARGET_B}/knowledge_graph.json" ]]; then
    KG_B_PATH="${TARGET_B}/knowledge_graph.json"
  fi
  uv run bookeeper verify --graph-file "${KG_B_PATH}" --mode drift --states-dir data/book_states_cache --embedding-model bge-m3:latest 2>&1 | tee "${OUTPUT_DIR}/audit_drift_b.txt" | tee -a "${LOG_FILE}"
fi

# Clean up lightweight extracted targets
if [[ -n "${DIR_A}" && -d "${DIR_A}" ]]; then
  rm -rf "${DIR_A}"
fi
if [[ -n "${DIR_B}" && -d "${DIR_B}" ]]; then
  rm -rf "${DIR_B}"
fi

# ------------------------------------------------------------------------------
# Phase 4: Generate Consolidated Markdown Report
# ------------------------------------------------------------------------------
log ""
log "──────────────────────────────────────────────────────────────────────────────"
log "📊 Phase 4/4: Generating Consolidated Summary Report..."
log "──────────────────────────────────────────────────────────────────────────────"

REPORT_FILE="${OUTPUT_DIR}/ab_benchmark_summary.md"

cat <<EOF > "${REPORT_FILE}"
# 🔬 Headless A/B Benchmark Summary: Master vs. JEPA

**Execution Date:** $(date '+%Y-%m-%d %H:%M:%S')  
**Session Directory:** \`${OUTPUT_DIR}\`  
**Target Books:** ${BOOK_LIMIT} (reusing cached book states from \`data/book_states_cache\`)  

---

## ⏱️ Execution & Throughput Performance

| Metric | Branch A (\`${BRANCH_A}\` / Classical) | Branch B (\`${BRANCH_B}\` / JEPA) | Delta / Speedup |
| :--- | :--- | :--- | :--- |
| **Stage 1 Duration** | ${TIME_A}s | ${TIME_B}s | $(if [[ ${TIME_A} -gt 0 && ${TIME_B} -gt 0 ]]; then awk -v a="${TIME_A}" -v b="${TIME_B}" 'BEGIN {printf "%.1fx faster", a/b}'; else echo "N/A"; fi) |
| **Backup Archive** | \`${BACKUP_A}\` | \`${BACKUP_B}\` | - |
| **Archive Size** | $(ls -lh "${BACKUP_A}" 2>/dev/null | awk '{print $5}' || echo "N/A") | $(ls -lh "${BACKUP_B}" 2>/dev/null | awk '{print $5}' || echo "N/A") | - |

---

## 📊 Summary of Comparative Audits

### 1. Topology & Entity Metrics (\`--mode stats\`)
\`\`\`text
$(cat "${OUTPUT_DIR}/audit_stats.txt" 2>/dev/null || echo "No stats audit output available.")
\`\`\`

### 2. Semantic Space Coverage & Diversity Suite (\`--mode graph\`, $\tau=${TAU}$)
\`\`\`text
$(cat "${OUTPUT_DIR}/audit_graph.txt" 2>/dev/null || echo "No graph audit output available.")
\`\`\`

### 3. NLI Atomic Cross-Check Audit (\`--mode cross-check\`, blocks=${BLOCKS})
\`\`\`text
$(cat "${OUTPUT_DIR}/audit_cross_check.txt" 2>/dev/null || echo "No cross-check audit output available.")
\`\`\`

### 4. Candidate Factual Grounding Audit (\`--mode ideas\`, percent=${PERCENT}%)
\`\`\`text
$(cat "${OUTPUT_DIR}/audit_grounding_b.txt" 2>/dev/null || echo "No grounding audit output available.")
\`\`\`

### 5. Candidate Synthesis Drift Audit (\`--mode drift\`)
\`\`\`text
$(cat "${OUTPUT_DIR}/audit_drift_b.txt" 2>/dev/null || echo "No drift audit output available.")
\`\`\`

---
*Report generated automatically by \`scripts/run_ab_backup_benchmark.sh\`.*
EOF

log "✓ Consolidated summary report saved to:"
log "  📄 ${REPORT_FILE}"
log "=============================================================================="
log "🎉 Benchmark completed successfully."
log "=============================================================================="
