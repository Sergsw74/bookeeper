# 🧪 Two-Pass Idea Post-Processing (Dual-Pass Synthesis)

> [!WARNING]
> ### ⚠️ Experimental Proof-of-Concept (POC) — Very Low Performance Improvement
> 
> **Two-Pass Idea Post-Processing** is an experimental research feature. In empirical evaluations, this secondary synthesis pass yielded **very low to marginal improvements** in factual grounding and knowledge graph query quality, while **significantly increasing compute time and token consumption** (doubling or tripling total LLM inference runtime).
> 
> For practical production pipelines and standard Knowledge Graph builds, the standard single-pass pipeline (`bookeeper build-graph`) is **strongly recommended**.

---

## 📋 Table of Contents

- [1. Motivation & Overview](#1-motivation--overview)
- [2. Algorithmic Pipeline](#2-algorithmic-pipeline)
  - [Step 1: Deterministic Grouping & Semantic Clustering](#step-1-deterministic-grouping--semantic-clustering)
  - [Step 2: Provenance Aggregation & Chronological Linearization](#step-2-provenance-aggregation--chronological-linearization)
  - [Step 3: Branching & Structured LLM Synthesis](#step-3-branching--structured-llm-synthesis)
  - [Step 4: Graph Ingestion & Edge Rewiring](#step-4-graph-ingestion--edge-rewiring)
- [3. CLI Usage](#3-cli-usage)
  - [Standalone Post-Processing (`postprocess-ideas`)](#standalone-post-processing-postprocess-ideas)
  - [Integrated Ingestion (`build-graph --dual-pass`)](#integrated-ingestion-build-graph---dual-pass)
- [4. Persistent State Checkpointing & Auto-Continue](#4-persistent-state-checkpointing--auto-continue)
  - [Auto-Save Every 100 Ideas](#auto-save-every-100-ideas)
  - [Auto-Continue on Resume](#auto-continue-on-resume)
  - [Resetting State](#resetting-state)
- [5. Preparation Progress Indicators](#5-preparation-progress-indicators)
- [6. Configuration Options](#6-configuration-options)
- [7. Next Steps](#7-next-steps)

---

## 1. Motivation & Overview

During standard chunk-by-chunk ingestion, concepts that appear across multiple chapters (e.g. *Event Sourcing*, *Write-Ahead Logging*, *Consensus Protocol*) accumulate independent, localized descriptions and quotes.

The **Two-Pass Post-Processor** executes a secondary reconciliation pass over the finished Knowledge Graph:
- Aggregates all localized observations and supporting quotes for an entity across the corpus.
- Groups syntactic and semantic duplicates.
- Prompts a consolidation model to synthesize a single, cohesive macro-definition combining cause-and-effect chains without dropping earlier facts.

---

## 2. Algorithmic Pipeline

```mermaid
flowchart TD
    A["Raw Concept Nodes in Graph"] --> B["1a. Normalization & Grouping\nCase folding, punctuation, determiner strip"]
    B --> C["1b. Dense Vector Embeddings\nnomic-embed-text"]
    C --> D["1c. Cosine Clustering\nMerge if similarity >= 0.92"]
    D --> E["2. Chronological Linearization\nOrder observations by (book, chapter, chunk)\nRetain top 3-5 quotes"]
    E --> F{"Mention Count Branching"}
    F -->|N = 1 (Single-Occurrence)| G["Bypass Second LLM Pass\nPreserve local observation as-is"]
    F -->|N >= 2 (Multi-Occurrence)| H["3. Structured LLM Synthesis\nConsolidate macro-definition + relations"]
    G --> I["4. Graph Ingestion & Overwrite\nUpsert Canonical Node\nAdd :MENTIONED_IN & :RELATES_TO"]
    H --> I
    I --> J[("Persistent Checkpoint\nSaved every 100 ideas")]
```

### Step 1: Deterministic Grouping & Semantic Clustering
1. **Deterministic Normalization**: Strips leading determiners (*The*, *A*, *An*), removes trailing punctuation, and case-folds concept names into unified clusters.
2. **Semantic Similarity Clustering**: Generates dense vector embeddings for canonical names and brief descriptions via `nomic-embed-text`. Unmerged clusters with pairwise cosine similarity $\ge 0.92$ are automatically merged (resolving direct aliases, plurals, and minor spelling variants).

### Step 2: Provenance Aggregation & Chronological Linearization
For each resolved entity cluster:
- Aggregates all source `chunk_id` anchors.
- Filters and deduplicates quotes, retaining the top 3–5 most representative supporting quotes.
- Orders observations chronologically by `(book_id, chapter_idx, chunk_idx)` to maintain the logical progression of ideas across the text.

### Step 3: Branching & Structured LLM Synthesis
- **Single-Occurrence Clusters ($N = 1$)**: Pass through untouched. Bypasses the secondary LLM invocation to conserve compute.
- **Multi-Occurrence Clusters ($N \ge 2$)**: Submits the aggregated chronological payload to the consolidation model (e.g. `llama3.1:8b`). Produces structured JSON:
  - `brief_description`: Cohesive definition capturing the entity's macro-identity.
  - `detailed_explanation`: Comprehensive explanation integrating distinct mechanisms, cause-and-effect chains, and conditions across all chunks.
  - `related_entities`: Explicit co-occurring architectural patterns and concepts.
  - `category` & `weight`: Refined domain classification and significance score (0–10).

### Step 4: Graph Ingestion & Edge Rewiring
- **Upsert Canonical Node**: Overwrites the canonical `:Concept` node with the unified definitions and aggregate weight.
- **Redirect Aliases**: Redirects incoming and outgoing edges from redundant alias nodes to the canonical node, then removes the alias nodes.
- **Provenance Edges**: Connects the canonical node to all source chunks via `[:SUPPORTED_BY]` and `[:MENTIONED_IN]` edges.
- **Relational Edges**: Emits directed `[:RELATES_TO]` edges between the consolidated idea and newly identified co-occurring entities.

---

## 3. CLI Usage

### Standalone Post-Processing (`postprocess-ideas`)

Run the post-processor directly against an existing `knowledge_graph.json`:

```bash
# Run on default knowledge graph (output/knowledge_graph.json):
bookeeper postprocess-ideas

# Run with custom model, threshold, and minimum occurrence count:
bookeeper postprocess-ideas \
  --graph-file ./output/knowledge_graph.json \
  --model llama3.1:8b \
  --similarity-threshold 0.92 \
  --min-occurrences 2
```

### Integrated Ingestion (`build-graph --dual-pass`)

Automatically trigger the second-pass post-processor immediately upon completing book ingestion:

```bash
bookeeper build-graph --limit 5 --dual-pass
```

---

## 4. Persistent State Checkpointing & Auto-Continue

### Auto-Save Every 100 Ideas
Because synthesizing multi-occurrence ideas across large graphs can take significant time, Bookkeeper persists progress checkpoints inside `.bookeeper_dual_pass_state.json`:
- **Periodic Checkpoints**: Automatically saved to disk **every 100 ideas** (`checkpoint_interval = 100` by default), upon final completion, and if interrupted via `Ctrl+C`.
- **Atomic File Writes**: Writes state to a temporary file before renaming to prevent corruption on sudden termination.

Customize the save interval:
```bash
bookeeper postprocess-ideas --checkpoint-interval 50
```

### Auto-Continue on Resume
When Bookkeeper detects an existing `.bookeeper_dual_pass_state.json`, it **auto-continues** by default:
- Detects already synthesized ideas and instantly applies their cached definitions to the graph in milliseconds.
- **Zero redundant LLM calls**: Calls the consolidation model only for remaining or newly discovered ideas.

```bash
# Automatically resumes from existing checkpoint:
bookeeper postprocess-ideas
```

### Resetting State
To discard saved progress and force a fresh re-synthesis of all ideas:

```bash
# Delete existing state checkpoint and start from scratch:
bookeeper postprocess-ideas --reset-state
```

---

## 5. Preparation Progress Indicators

To eliminate dead time while loading large graphs and generating dense embeddings, Bookkeeper displays real-time visual progress across all preparation stages:

1. **Graph Loading**: `Loading Knowledge Graph from knowledge_graph.json...` spinner with node/edge totals.
2. **Concept Scanning**: `Scanning Knowledge Graph & normalizing idea mentions...` spinner with candidate counts.
3. **Batched Semantic Embeddings**: Live Rich progress bar showing batch progress, percentage, model name, and elapsed time:
   ```text
   Generating Semantic Embeddings: ━━━━━━━━━━━━━━━━━━━━ 100% 154/154 • nomic-embed-text  0:12
   ✓ Generated semantic embeddings for 154 clusters (nomic-embed-text)
   ```
4. **Cosine Clustering**: `Clustering concepts via cosine similarity (>= 0.92)...` spinner with merge statistics.
5. **Live Consolidation**: Displays current idea, mention count, and synthesis model before invoking Ollama.

---

## 6. Configuration Options

Configure two-pass post-processing defaults in `config.yaml`:

```yaml
dual_pass:
  enabled: false                      # Automatically run after build-graph
  model: "llama3.1:8b"                # Dedicated LLM consolidation model
  similarity_threshold: 0.92          # Cosine similarity merge threshold
  max_quotes: 5                       # Max supporting quotes per cluster
  min_occurrences: 2                  # Minimum mentions (N >= 2) to trigger LLM synthesis
  temperature: 0.1                    # LLM temperature
  state_file: null                    # Custom path for state checkpoint JSON
  checkpoint_interval: 100            # Save checkpoint every N ideas
```

---

## 7. Next Steps

- 🏗️ **[Building the Knowledge Graph](building-graph.md)** — Core single-pass knowledge graph extraction.
- 🔬 **[A/B Testing & Benchmarking](ab-tests.md)** — Benchmark single-pass vs dual-pass quality and runtime.
- 🛡️ **[Factual Verification & Auditing](verification.md)** — Audit factual grounding across idea nodes.
- 🏠 **[Return to Documentation Index](../README.md)**
