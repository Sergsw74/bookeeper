# 🔬 Automated A/B Testing & Benchmarking

Bookkeeper includes an automated test harness (`scripts/ab_test.sh` and `scripts/ab_test.py`) designed to benchmark, compare, and validate pipeline changes—such as alternative chunking strategies, prompt refinements, model swaps, or graph post-processors—across extraction speed, idea yields, and factual grounding recall.

---

## 📋 Table of Contents

- [1. Overview & Motivation](#1-overview--motivation)
- [2. Execution Modes](#2-execution-modes)
  - [Mode 1: Same-Branch Mode (Config A vs Config B)](#mode-1-same-branch-mode-config-a-vs-config-b)
  - [Mode 2: Branch vs Branch Mode](#mode-2-branch-vs-branch-mode)
  - [Mode 3: Baseline Comparison Mode](#mode-3-baseline-comparison-mode)
  - [Mode 4: Bookkeeper Knowledge Graph vs. LightRAG Benchmark](#mode-4-bookkeeper-knowledge-graph-vs-lightrag-benchmark)
  - [Mode 5: A/B Testing vs Archived Backups (`ab-compare`)](#mode-5-ab-testing-vs-archived-backups-ab-compare)
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

#### Mode 4: Bookkeeper Knowledge Graph vs. LightRAG Benchmark

Directly benchmarks Bookkeeper's structured Knowledge Graph against LightRAG's native raw text chunking engine on the exact same $N$ books. The benchmark is split into **3 distinct phases** with **safe points** between runs:

1. **Phase 1: Bookkeeper Analysis & Knowledge Graph Generation (Safe Point 1)**:
   - Indexes $N$ books into Bookkeeper (hierarchical chunking, idea extraction, dual-pass post-processing from `config.yaml`).
   - Generates `output/knowledge_graph.json` and updates Neo4j.
   - **Safe Point 1** is recorded in `benchmark_checkpoint.json`. If a pre-computed knowledge graph already exists, Phase 1 can be skipped automatically without re-indexing.
2. **Phase 2: LightRAG Data Ingestion & Native Chunking (Safe Point 2)**:
   - Ingests the exact same $N$ books into LightRAG directly (`LightRAGRawIndexer`), using raw full-text input and LightRAG's native token-based chunker.
   - Operates on a dedicated, persistent event loop thread to prevent `PriorityQueue is bound to a different event loop` worker issues.
   - Persists per-book progress in `indexed_books.json` and records **Safe Point 2** in `benchmark_checkpoint.json`. If indexing is interrupted or if Phase 2 was already completed, it resumes without re-processing.
3. **Phase 3: Dual Retrieval & Objective Verification Queries (Safe Point 3)**:
   - **Subject-Matter Queries**: Queries name concrete mechanisms, algorithms, models, and failure modes directly. Meta-references (*"in this book"*, *"the author"*, *"in chapter X"*, *"according to the paper"*) are strictly sanitized because queries run against a global aggregated knowledge base.
   - **Binary Invariant Verification Checklist**: Each query is paired with 3–4 boolean criteria ([`ChecklistItem`](file:///Users/serg/repos/bookeeper/src/bookeeper/evaluation/question_generator.py)) including positive mechanisms (`expected: true`) and negative anti-hallucination distractor checks (`expected: false`).
   - **Dual Retrieval & Objective LLM Judge**: Submits queries to Bookkeeper (`SearchEngineBackend` with Neo4j or local graph fallback) and LightRAG (`LightRAGService`). The LLM Judge verifies retrieved context against the invariant checklist.
   - **Reporting**: Generates `ab_test_summary.md` with pass rates and deltas, and saves `benchmark_report.json`.

```bash
# 1. Run complete end-to-end benchmark (Phases 1-3):
bookeeper ab-benchmark 3

# 2. Continue from Phase 2 (reuse pre-computed Bookkeeper graph, index LightRAG, and run queries):
bookeeper ab-benchmark 3 --from-phase 2

# 3. Continue from Phase 3 (run queries across pre-computed Bookkeeper & LightRAG storages):
bookeeper ab-benchmark 3 --from-phase 3

# 4. Execute ONLY a specific phase:
bookeeper ab-benchmark 3 --phase bookkeeper  # Run only Phase 1
bookeeper ab-benchmark 3 --phase lightrag    # Run only Phase 2
bookeeper ab-benchmark 3 --phase benchmark   # Run only Phase 3

# 5. Using the ab_test.py automation runner:
./scripts/ab_test.py --vs-lightrag --books 3 --from-phase 2
./scripts/ab_test.py --vs-lightrag --books 3 --phase lightrag

# 6. Discard previous safe points and rebuild completely from scratch:
bookeeper ab-benchmark 3 --clean

# 7. Use an existing or curated question library:
bookeeper ab-benchmark 3 --questions questions.json

# 8. Launch the interactive LightRAG WebUI & Graph Visualizer on test results:
bookeeper lightrag-ui
# Or point directly to a specific A/B benchmark run directory:
bookeeper lightrag-ui --lightrag-dir ab_test_runs/run_20261014_144807
```

### Mode 5: A/B Testing vs Archived Backups (`ab-compare`)

Re-indexing books from scratch to run comparative evaluations can take tens of minutes of LLM inference time. If you maintain regular compressed backups (`.zip` or `.7z` generated via `bookeeper backup`), you can run instant A/B comparative evaluations directly against archived runs.

The automated backup resolver unpacks archives into isolated temporal directories (`tempfile.TemporaryDirectory(prefix="bk_ab_")`), loads the graph topologies, runs audits, and guarantees clean purging of temporal directories on exit:

```bash
# 1. Topological & overlap metrics between two historical backups (instant, offline):
bookeeper ab-compare backups/backup_run_oct10.zip backups/backup_run_oct20.zip --mode stats

# 2. Compare a historical backup against your currently active knowledge graph:
bookeeper ab-compare backups/baseline_prod.zip --mode stats

# 3. Deep NLI Cross-Check audit (evaluates recall, seam integrity, and contradictions between backups):
bookeeper ab-compare backups/baseline.zip backups/candidate.zip --mode cross-check --blocks 10

# 4. Using the ab_test.py automation runner:
python3 scripts/ab_test.py vs-backup backups/baseline.zip backups/candidate.zip --blocks 10
python3 scripts/ab_test.py --vs-backup backups/baseline.zip

# 5. Direct verification command accepting backup archives:
bookeeper verify --graph-file backups/baseline.zip --compare-graph backups/candidate.zip --mode cross-check
```

#### What is Evaluated:
- **Topological & Extraction Metrics**: Side-by-side comparison of Book counts, Section counts, hierarchical Chunks, extracted Ideas, total Edges, and Edge-to-Node Relational Density.
- **Idea Set Overlap & Jaccard Similarity**: Exact intersection of extracted concepts, count and proportion of common ideas, ideas unique to Run A vs Run B, and Jaccard similarity coefficient.
- **NLI Cross-Check Recall & Seam Integrity**: Evaluates atomic assertions from contiguous text spans against both graphs to compute comparative recall ($\Delta \text{Recall}$), seam integrity, and net semantic integrity $\Delta S_{net}$.
- **Zero Disk Footprint**: Unpacked temporary files (`knowledge_graph.json`, `.book_states/`, `chunks/`) are strictly removed even if audits fail or are interrupted.

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

## 6. Phase 3 Dual-Level Query Execution & Keyword Extraction

During Phase 3, both Bookkeeper (hierarchical ideas/deep/exact search) and LightRAG (hybrid/local/global search) execute against identical evaluation questions:
- **LightRAG Dual Retrieval**: LightRAG combines local subgraphs (entity-level) and global summaries (relationship-level).
- **Keyword Extraction Resilience**: LightRAG automatically extracts `high_level_keywords` and `low_level_keywords` using the configured LLM. Bookkeeper ensures structured JSON enforcement with Ollama, normalizes `keep_alive` values (numeric `-1` duration vs string duration), and applies an automated fallback extractor directly from the query terms if either keyword tier is missing or unparseable.
- **Fail-Safe Retrieval**: If keyword extraction produces an empty list, terms are balanced between tiers so graph traversal and vector chunk extraction never abort or trigger default empty responses.

---

## 7. Knowledge Graph Structural Topology Comparison

Both engines maintain an underlying graph structure that is compared directly in `ab_test_summary.md` and `benchmark_report.json`:
- **Nodes Count**: Total distinct entities, concepts, sections, and ideas extracted into the graph.
- **Edges Count**: Total cross-link relationships, structural links, and thematic edges.
- **Relational Density**: Edge-to-node connectivity ratio ($|E| / |V|$), measuring relational cross-referencing depth.
- **Chunk Breakdown**: Number of hierarchical source chunks vs. flat token chunks.

| Graph Structural Metric | Bookkeeper (A) | LightRAG (B) | Comparison / Delta |
| :--- | :---: | :---: | :---: |
| **Total Graph Nodes** | Extracted ideas, concepts, sections | Named entities and concepts | Node density ratio |
| **Total Graph Edges** | Multi-hop thematic & semantic links | Relationship co-occurrences | Relational link ratio |
| **Connectivity Density** | Edges per node | Edges per node | Graph density delta |
| **Indexed Chunks** | Hierarchical chunks | Flat token chunks | Context granularity |

---

## 8. Full CLI Reference

### 8.1 `scripts/ab_test.py`
```text
usage: ab_test.py [-h] [--books N] [--percent P] [--timeout SEC]
                  [--config PATH] [--config-a PATH] [--config-b PATH]
                  [--blocks N] [--output-dir DIR]
                  [--vs-lightrag] [--questions PATH] [--questions-per-book N]
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
  --vs-lightrag        Run A/B benchmark against LightRAG engine
  --questions PATH     Path to pre-generated QuestionLibrary JSON
  --questions-per-book Questions to generate per book if synthesizing library (default: 5)
```

### 8.2 `bookeeper ab-compare` (alias `bookeeper ab-backup`)
```text
Usage: bookeeper ab-compare [OPTIONS] TARGET_A [TARGET_B]

  Perform automated A/B comparative evaluation between archived backups (.zip, .7z)
  or against the active knowledge graph. Automatically recovers targets to temporal
  directories and guarantees full cleanup on exit.

Arguments:
  TARGET_A             First backup archive (.zip, .7z, .tar.gz) or graph JSON
  [TARGET_B]           Second backup archive or graph JSON (defaults to active output/knowledge_graph.json)

Options:
  -m, --mode [stats|cross-check|ideas]
                       Comparison mode: 'stats' (fast topological & Jaccard metrics),
                       'cross-check' (NLI assertion audit), or 'ideas' (factual grounding) [default: stats]
  -b, --blocks INTEGER Number of contiguous chunk blocks to audit in cross-check mode [default: 10]
  -o, --output PATH    Destination path to save JSON comparison report
  -c, --config PATH    Path to config.yaml [default: config.yaml]
```

### 8.3 `bookeeper ab-benchmark`
```text
Usage: bookeeper ab-benchmark [OPTIONS] [NUM_BOOKS]

  Run an end-to-end comparative A/B benchmark between Bookkeeper and LightRAG.

Arguments:
  [NUM_BOOKS]            Number of books to benchmark [default: 3]

Options:
  -q, --questions PATH   Path to existing question library JSON
  -k, --questions-per-book INTEGER
                         Number of questions per book if generating [default: 5]
  -m, --lightrag-mode TEXT
                         LightRAG query mode (hybrid/local/global/naive) [default: hybrid]
  -b, --bookkeeper-mode TEXT
                         Bookkeeper search mode (ideas/deep/exact) [default: ideas]
  -j, --judge-model TEXT Judge / verification model name
  -o, --output-dir PATH  Custom output directory for benchmark reports
  -s, --skip-lightrag-indexing
                         Skip LightRAG ingestion and reuse existing storage
```

### 8.3 `bookeeper generate-questions`
```text
Usage: bookeeper generate-questions [OPTIONS] [NUM_BOOKS]

  Generate an evaluation question library for N books using the verification
  model pool with subject-matter queries and binary invariant checklists.

Arguments:
  [NUM_BOOKS]            Number of books to include [default: 3]

Options:
  -k, --questions-per-book INTEGER
                         Questions to generate per book [default: 5]
  -o, --output PATH      Destination path for generated question library JSON
  -m, --model TEXT       LLM model to use (defaults to verification model)
  -t, --timeout INTEGER  Ollama request timeout in seconds [default: 180]
```

### 8.4 `bookeeper lightrag-index`
```text
Usage: bookeeper lightrag-index [OPTIONS] [NUM_BOOKS]

  Index N books into LightRAG directly using raw book text (natural chunking)
  for direct A/B baseline comparison.

Arguments:
  [NUM_BOOKS]            Number of books to index [default: 3]

Options:
  -w, --working-dir PATH Custom LightRAG storage directory
  -c, --clean            Clear existing LightRAG working directory before indexing
  -f, --file PATH        Index a specific book file directly
```

---

## 9. Next Steps

- 🛡️ **[Factual Verification & Auditing](verification.md)** — Detailed breakdown of verification modes and NLI cross-checking.
- 🏗️ **[Building the Knowledge Graph](building-graph.md)** — Core graph extraction options and rebuilds.
- 🧪 **[Two-Pass Idea Post-Processing (Experimental POC)](post-processing.md)** — Experimental synthesis details.
- 🏠 **[Return to Documentation Index](../README.md)**
