# 🚀 Getting Started with Bookkeeper

Welcome to **Bookkeeper**! This guide walks you through installation, configuration, initial cluster setup, and essential day-to-day commands to get you up and running in minutes.

---

## 📋 Table of Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Installation](#2-installation)
- [3. Configuration](#3-configuration)
- [4. Basic CLI Commands Walkthrough](#4-basic-cli-commands-walkthrough)
  - [Discovering Books (`list-books`)](#discovering-books-list-books)
  - [Inspecting System Status & Configuration (`status`, `stats`, `config`)](#inspecting-system-status--configuration-status-stats-config)
  - [Cleaning Calibre Metadata (`clean-metadata`)](#cleaning-calibre-metadata-clean-metadata)
  - [Creating Backups & Restoring Knowledge Bases (`backup`, `restore`)](#creating-backups--restoring-knowledge-bases-backup-restore)
- [5. Your First Knowledge Graph Run](#5-your-first-knowledge-graph-run)
- [6. Next Steps & Detailed Guides](#6-next-steps--detailed-guides)

---

## 1. Prerequisites

Before installing Bookkeeper, ensure your system meets the following requirements:

1. **Python 3.10+** (Python 3.12 recommended).
2. **Ollama Instance(s)** running locally or distributed across your local network:
   ```bash
   # Pull primary concept extraction model (small & fast):
   ollama pull qwen2.5:3b

   # Pull fallback and verification model (higher reasoning capability):
   ollama pull llama3.1:8b

   # Pull embedding model for chunking and clustering:
   ollama pull nomic-embed-text
   ```
3. **Digital Book Collection**:
   - A local or network-mounted **Calibre Library** containing EPUB, PDF, or FB2 files.
   - **OR** a direct folder of downloaded books/papers (e.g. `archive_downloader/downloads`).

---

## 2. Installation

### Option A: Using `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) provides rapid, deterministic environment management:

```bash
git clone https://github.com/your-org/bookeeper.git
cd bookeeper

# Create virtual environment and install in editable mode with development tools:
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Option B: Using Standard `pip`

```bash
git clone https://github.com/your-org/bookeeper.git
cd bookeeper

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify your installation:
```bash
bookeeper --help
```

---

## 3. Configuration

Bookkeeper uses a centralized `config.yaml` file for all runtime options. Copy the provided template to create your active configuration:

```bash
cp config.yaml.example config.yaml
```

### Key Configuration Sections

Open `config.yaml` and adjust the paths and endpoints for your environment:

```yaml
# 1. Book Source: Calibre Library OR Direct Folder
# Direct download directory / arXiv papers:
books_dir: "archive_downloader/downloads"

# Calibre Library path (local folder or mounted SMB share):
calibre_library_path: "~/Calibre Library"
stage_metadata_db: true        # Stages SQLite DB locally on SSD for microsecond queries
backup_metadata_db: true       # Creates remote metadata.db.bak backup before syncing

# 2. Primary Ollama LLM Configuration
ollama_base_url: "http://192.168.50.15:11434"
llm_model: "qwen2.5:3b"
llm_model_fallback: "llama3.1:8b"   # Switched to after 50% chunk retry failures
request_timeout: 30                 # HTTP timeout in seconds per LLM call
max_retries: 1                     # Retries per attempt on an Ollama server
max_chunk_attempts: 6              # Maximum total retry attempts across pool for a stalled chunk

# 3. Embeddings Configuration
embedding_base_url: "http://localhost:11434"  # Dedicated local endpoint or "pool"
embedding_model: "nomic-embed-text"

# 4. Multi-Node Server Pool with Task Capabilities
ollama_servers:
  - url: "http://192.168.50.118:11434"
    priority: 1
    name: "primary-gpu-node"
    capability: ["llm", "embedding", "verification"]
  - url: "http://localhost:11434"
    priority: 2
    name: "mac-local"
    capability: ["llm", "embedding"]

# Failover Cooldowns
analysis_cooldown_seconds: 600       # Cooldown for extraction failovers (large text batches)
verification_cooldown_seconds: 60    # Cooldown for verification audits (fast note retries)
```

---

## 4. Basic CLI Commands Walkthrough

### Discovering Books (`list-books`)

List, filter, and inspect books available in your library or download folder:

```bash
# List first 20 books from Calibre:
bookeeper list-books --limit 20

# Search by title or author:
bookeeper list-books --search "Distributed Systems"

# Discover books in a direct download or arXiv folder:
bookeeper list-books --books-dir archive_downloader/downloads --limit 15
```

### Inspecting System Status & Configuration (`status`, `stats`, `config`)

```bash
# View active checkpoint state, completed books, assigned fast-path chunks, and skipped limbo chunks:
bookeeper status

# Inspect existing Knowledge Graph metrics (nodes, edges, categories, relationships):
bookeeper stats

# Print active settings, resolved server pool, and node capabilities:
bookeeper config
```

### Cleaning Calibre Metadata (`clean-metadata`)

Prompt Ollama to clean, standardize, and repair titles, authors, and summaries in your Calibre library:

```bash
# Preview metadata changes without modifying the database (dry-run):
bookeeper clean-metadata --limit 10 --dry-run

# Apply cleaned metadata to the first 50 books:
bookeeper clean-metadata --limit 50
```

### Creating Backups & Restoring Knowledge Bases (`backup`, `restore`)

Bookkeeper includes a standalone backup engine that packages your Knowledge Graph, Obsidian vault, per-book state checkpoints, pre-computed chunks, and verification reports into portable archives:

```bash
# 1. Create a timestamped ZIP backup in current directory:
bookeeper backup

# 2. Create a high-compression 7-Zip backup (.7z):
bookeeper backup --format 7z -o ./backups/my_knowledge.7z

# 3. Dry-run inspect a backup archive:
bookeeper restore ./backups/my_knowledge.7z --dry-run

# 4. Clean restore into a new directory:
bookeeper restore ./backups/my_knowledge.7z --output-dir ./restored_output --clean
```

---

## 5. Your First Knowledge Graph Run

Once configured, index a small batch of books to verify your pipeline end-to-end:

```bash
# Ingest 3 books from scratch and export to Obsidian:
bookeeper build-graph --limit 3 --from-scratch --no-lightrag
```

You can monitor real-time extraction progress in the terminal:
- Per-chapter and per-chunk processing indicators.
- Live Ollama node latency, throughput, and failover statistics.
- Instant Obsidian vault export in `output/obsidian_vault/`.

---

## 6. Next Steps & Detailed Guides

Dive deeper into specific subsystems:

- 🔄 **[Recommended Workflow](workflow.md)** — Start here: Calibration via small A/B tests, full graph build, exports, and optional POC post-processing.
- 🏗️ **[Building the Knowledge Graph](building-graph.md)** — Comprehensive options for `build-graph`, instant offline rebuilds via `rebuild-graph`, concept significance weighting, and Neo4j exports.
- 🔍 **[Decoupled Hybrid RAG](rag.md)** — Indexing cached chunks with LightRAG and interactive querying across local and global modes.
- 🧪 **[Two-Pass Idea Post-Processing (Experimental POC)](post-processing.md)** — Secondary synthesis pass over multi-occurrence ideas, clustering, and state checkpointing.
- 🔬 **[A/B Testing & Benchmarking](ab-tests.md)** — Automated regression test harness, same-branch vs branch-to-branch comparisons, and config overlays.
- 🛡️ **[Factual Verification & Auditing](verification.md)** — Multi-mode verification audits (Ideas Grounding, Chunking Coherence, and Dual-KG Cross-Checking).
- 🏠 **[Return to Documentation Index](../README.md)**
