# 🔬 Automated A/B Testing & Benchmarking

Bookkeeper includes an automated test harness (`scripts/ab_test.sh` and `scripts/ab_test.py`) designed to benchmark, compare, and validate pipeline changes—such as alternative chunking strategies, prompt refinements, model swaps, or graph post-processors—across extraction speed, idea yields, and factual grounding recall.

---

## 📋 Table of Contents

- [1. Overview & Motivation](#1-overview--motivation)
- [2. Execution Modes](#2-execution-modes)
  - [Mode 1: Same-Branch Mode (Config A vs Config B)](#mode-1-same-branch-mode-config-a-vs-config-b)
  - [Mode 2: Branch vs Branch Mode](#mode-2-branch-vs-branch-mode)
  - [Mode 3: Baseline Comparison Mode](#mode-3-baseline-comparison-mode)
- [3. Configuration Overlays (`config-a.yaml` & `config-b.yaml`)](#3-configuration-overlays-config-a-yaml--config-b-yaml)
  - [Deep-Merge Semantics](#deep-merge-semantics)
  - [Overlay Example](#overlay-example)
- [4. Session Directory & Reports](#4-session-directory--reports)
  - [Directory Structure](#directory-structure)
  - [Markdown Summary Report](#markdown-summary-report)
- [5. Automated Post-Test Cross-Check Verification](#5-automated-post-test-cross-check-verification)
- [6. Full CLI Reference](#6-full-cli-reference)
- [7. Next Steps](#7-next-steps)

---

## 1. Overview & Motivation

When optimizing a Knowledge Graph extraction pipeline, evaluating changes manually is error-prone:
- How does changing the chunk size or switching to a new LLM affect concept recall?
- Does a new prompt produce more ideas, or does it introduce hallucinations?
- How much additional inference time does an architectural change require?

The A/B testing harness automates this workflow:
1. Provisions isolated runs for **Case A** and **Case B**.
2. Builds Knowledge Graphs from scratch for a fixed number of books under each configuration.
3. Automatically launches factual grounding audits (`bookeeper verify`) on both results.
4. Executes NLI Atomic Assertion Cross-Check verification between Graph A and Graph B.
5. Emits a comprehensive Markdown report summarizing runtime differences, idea count deltas, and grounding recall.

---

## 2. Execution Modes

### Mode 1: Same-Branch Mode (Config A vs Config B)

Evaluates two configuration profiles on your **currently active Git branch**. Does not switch branches or require stashing uncommitted code:

```bash
# Bash wrapper: Test 5 books at 1.0% verification sample:
./scripts/ab_test.sh 5 1.0

# Direct Python invocation:
python3 scripts/ab_test.py --books 5 --percent 1.0
```

### Mode 2: Branch vs Branch Mode

Compares two Git branches (e.g. `main` vs a feature branch). The harness checks out each branch, runs the ingestion pipeline, and switches back to your original branch when finished:

```bash
# Ingest 5 books per branch from scratch and compare:
./scripts/ab_test.sh main feature-dual-pass 5

# Direct Python invocation:
python3 scripts/ab_test.py main feature-dual-pass --books 5 --percent 2.0
```

### Mode 3: Baseline Comparison Mode

If you already have a completed verification run for Branch A, you can compare a new candidate branch against the pre-computed baseline **without re-indexing Branch A**:

```bash
./scripts/ab_test.sh vs ./ab_test_runs/run_20261010_120000/branch_A/verification_report.json feature-dual-pass 5
```

---

## 3. Configuration Overlays (`config-a.yaml` & `config-b.yaml`)

To test architectural, prompt, or model variations without modifying your working `config.yaml` or dirtying your Git tree, Bookkeeper supports **Configuration Overlays**:

- **Base Configuration**: The runner reads your base `config.yaml`.
- **Case A Overlay (`config-a.yaml`)**: Deep-merged on top of `config.yaml` for Case A / Branch A.
- **Case B Overlay (`config-b.yaml`)**: Deep-merged on top of `config.yaml` for Case B / Branch B.

### Deep-Merge Semantics
Overlays recursively overwrite specific keys while preserving all unmentioned base settings (such as Ollama cluster URLs, Calibre library paths, or timeouts).

### Overlay Example

Suppose you want to compare `qwen2.5:3b` with single-pass extraction against `llama3.1:8b` with dual-pass post-processing:

**`config-a.yaml`**:
```yaml
llm_model: "qwen2.5:3b"
dual_pass:
  enabled: false
```

**`config-b.yaml`**:
```yaml
llm_model: "llama3.1:8b"
dual_pass:
  enabled: true
  model: "llama3.1:8b"
```

Run the comparison:
```bash
python3 scripts/ab_test.py \
  --books 5 \
  --config config.yaml \
  --config-a config-a.yaml \
  --config-b config-b.yaml
```

---

## 4. Session Directory & Reports

### Directory Structure

Each A/B test run produces a self-contained, timestamped session folder under `ab_test_runs/`:

```text
ab_test_runs/
└── run_20261012_143000/
    ├── ab_test_summary.md          # High-level comparison report with deltas
    ├── branch_A/
    │   ├── config.yaml             # Merged configuration used for Case A
    │   ├── knowledge_graph.json    # Resulting knowledge graph for Case A
    │   ├── verification_report.json# Grounding audit results
    │   └── build_graph.log         # Complete execution log
    ├── branch_B/
    │   ├── config.yaml             # Merged configuration used for Case B
    │   ├── knowledge_graph.json    # Resulting knowledge graph for Case B
    │   ├── verification_report.json# Grounding audit results
    │   └── build_graph.log
    └── cross_check_report.json     # NLI assertion comparison between A and B
```

### Markdown Summary Report

The generated `ab_test_summary.md` provides an instant side-by-side comparison:

| Metric | Case A (`master`) | Case B (`feature-dual-pass`) | Delta |
| :--- | :--- | :--- | :--- |
| **Ingestion Time** | 4m 12s | 9m 45s | +5m 33s (+132%) |
| **Total Chunks** | 240 | 240 | 0 |
| **Extracted Concepts** | 312 | 284 | -28 (-8.9%) |
| **Idea Relationships** | 520 | 598 | +78 (+15.0%) |
| **Factual Grounding %** | 94.2% | 95.1% | +0.9% |
| **Cross-Check NLI Recall** | 51.4% | 52.8% | +1.4% |
| **Seam Integrity (SIMINT)**| 68.2% | 69.0% | +0.8% |

---

## 5. Automated Post-Test Cross-Check Verification

At the conclusion of the verification phase, the A/B test harness automatically runs a **Cross-Check Audit** comparing Graph A against Graph B:
- Selects contiguous chunk blocks clamped to exact raw document spans.
- Decomposes the source passages into atomic factual assertions.
- Evaluates whether assertions preserved in Graph A are also entailed by Graph B, and vice versa.
- Emits atomic recall and seam integrity metrics into `cross_check_report.json`.

---

## 6. Full CLI Reference

```text
usage: ab_test.py [-h] [--books N] [--percent P] [--timeout SEC]
                  [--config PATH] [--config-a PATH] [--config-b PATH]
                  [--blocks N] [--output-dir DIR]
                  [branch_a] [branch_b]

positional arguments:
  branch_a             Base Git branch or report path (omit for same-branch mode)
  branch_b             Candidate Git branch

options:
  --books N            Number of books to index from scratch (default: 5)
  --percent P          Sample percentage for factual verification (default: 1.0)
  --timeout SEC        HTTP timeout in seconds per LLM request (default: 60)
  --config PATH        Base config.yaml path (default: config.yaml)
  --config-a PATH      Optional overlay configuration for Case A
  --config-b PATH      Optional overlay configuration for Case B
  --blocks N           Number of chunk blocks to evaluate in cross-check (default: 10)
  --output-dir DIR     Custom root destination for test runs
```

---

## 7. Next Steps

- 🛡️ **[Factual Verification & Auditing](verification.md)** — Detailed breakdown of verification modes and NLI cross-checking.
- 🏗️ **[Building the Knowledge Graph](building-graph.md)** — Core graph extraction options and rebuilds.
- 🧪 **[Two-Pass Idea Post-Processing (Experimental POC)](post-processing.md)** — Experimental synthesis details.
- 🏠 **[Return to Documentation Index](../README.md)**
