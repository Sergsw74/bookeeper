# 📥 Archive Downloader

> **Automated arXiv & scientific article crawler with pagination, deduplication, persistent indexing, and Calibre-compatible metadata.**

`archive_downloader` is a standalone project within the `bookeeper` repository designed to crawl, index, and download papers and books from arXiv list feeds (*e.g.* `https://arxiv.org/list/cs.AI/recent`).

---

## ✨ Features

- **Automated Pagination**: Seamlessly navigates across multi-page listings (`skip=0`, `skip=50`, `skip=100`, etc.) until the target article quota is fulfilled.
- **Smart Indexing & Deduplication**:
  - Maintains a local persistent `index.json` catalog with article metadata, local file paths, file sizes, and SHA-256 hashes.
  - Verifies both index registration and physical presence on disk. If a paper was already downloaded, it is instantly skipped without redundant network calls.
- **Calibre Integration & Formats**:
  - Prefers standard **PDF** format (natively supported by Calibre and PDF viewers).
  - Automatically generates an accompanying Open Packaging Format (`.opf`) XML metadata file alongside each `.pdf`. When importing into Calibre (`calibredb add`), Calibre reads the `.opf` to populate titles, authors, categories/tags, and arXiv IDs with 100% fidelity.
- **Flexible Accounting Modes**:
  - **Count Existing (Default)**: Inspects the top `N` most recent articles (*e.g.* 100). Any already downloaded articles among them count towards the 100 target, downloading only new articles published since your last run.
  - **New-Only (`--new-only`)**: Continues crawling through older pagination pages until exactly `N` *brand-new* papers are downloaded.
- **Rate-Limiting & Politeness**: Configurable request delay (`--delay 0.5s`) to respect arXiv guidelines and avoid firewall throttling.
- **Rich Terminal UI**: Displays real-time progress indicators, ETA, and download summaries.

---

## 🚀 Quick Start

### 1. Basic Invocation

Using the convenience launcher:

```bash
# Download up to 100 most recent articles from cs.AI into ./downloads:
python archive_downloader/run.py download --limit 100
```

Or via module invocation:

```bash
PYTHONPATH=archive_downloader/src python -m archive_downloader download --limit 100
```

### 2. High-Throughput Batch URLs (`show=500`)

arXiv supports fetching up to 500 (or up to 2000) articles per listing page using `?skip=0&show=500`, drastically minimizing HTML scraping round-trips:

```bash
# Pass the full show=500 URL directly:
python archive_downloader/run.py download \
  --url "https://arxiv.org/list/cs.AI/recent?skip=0&show=500" \
  --limit 500

# Or configure page size with --show / --page-size (default: 500):
python archive_downloader/run.py download --limit 500 --page-size 500
```

### 3. Custom Output Directory & Category URL

```bash
# Download 50 articles from a specific arXiv category to a custom folder:
python archive_downloader/run.py download \
  --url "https://arxiv.org/list/cs.LG/recent?skip=0&show=100" \
  --limit 50 \
  --output-dir "./my_ai_library"
```

### 4. Downloading Strictly New Articles (`--new-only`)

```bash
# Keep paginating until 20 brand-new articles are downloaded:
python archive_downloader/run.py download --limit 20 --new-only
```

### 4. Viewing Index Status & Disk Usage

```bash
# Display summary of all downloaded articles, sizes, and categories:
python archive_downloader/run.py status

# Inspect a custom download directory:
python archive_downloader/run.py status --output-dir "./my_ai_library"
```

### 5. Inspecting arXiv Pages Without Downloading

```bash
# Preview the top 10 articles and pagination link from arXiv:
python archive_downloader/run.py inspect --limit 10
```

---

## 📁 Directory Structure

```text
archive_downloader/
├── pyproject.toml               # Standalone package definition
├── README.md                    # Project documentation
├── run.py                       # Executable CLI launcher
├── .gitignore                   # Ignores downloaded PDFs and index caches
├── src/
│   └── archive_downloader/
│       ├── __init__.py
│       ├── __main__.py          # Entry point for python -m archive_downloader
│       ├── cli.py               # Typer / Rich CLI commands
│       ├── models.py            # ArticleMetadata, IndexEntry, DownloadStats
│       ├── parser.py            # arXiv HTML page parser & pagination extractor
│       ├── indexer.py           # Persistent JSON index & deduplication
│       ├── downloader.py        # Core download engine with stream hashing
│       └── calibre_metadata.py  # Calibre-compatible .opf XML generator
├── tests/                       # Pytest test suite
│   ├── test_parser.py
│   ├── test_indexer.py
│   └── test_downloader.py
└── downloads/                   # Default download destination (auto-created)
    ├── 2610.12466 - On the estimation and validity of AI time horizons.pdf
    ├── 2610.12466 - On the estimation and validity of AI time horizons.opf
    └── index.json               # Catalog of all downloaded files
```

---

## 📖 Integrating with Calibre

Once articles are downloaded, you can import them into your Calibre library in one command:

```bash
calibredb add archive_downloader/downloads/*.pdf
```

Because `archive_downloader` generates matching `.opf` files (*e.g.* `2610.12466 - Title.opf`), Calibre automatically imports:
- **Title**: Normalized article title
- **Authors**: Clean list of authors
- **Tags / Subjects**: arXiv primary and secondary categories (*e.g.* `Artificial Intelligence (cs.AI)`)
- **Identifiers**: `arxiv:2610.12466`

---

## 🧪 Running Tests

```bash
PYTHONPATH=archive_downloader/src .venv/bin/pytest archive_downloader/tests/ -v
```
