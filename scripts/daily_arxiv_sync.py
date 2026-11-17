#!/usr/bin/env python3
"""
Daily arXiv Article Processing & Remote Synchronization Pipeline

End-to-end automated pipeline for daily arXiv updates:
  1. Downloader: Crawl and download the most recent articles from arXiv (via archive_downloader).
  2. Build Graph: Extract concepts, chunks, and relations into the Knowledge Graph (via bookeeper build-graph).
  3. Prepare Incremental Update: Calculate conceptual diff and package an incremental archive (.zip).
  4. Upload via REST API: Stream the incremental archive to a remote Bookkeeper server (/api/backup/restore)
     to merge updates into remote storage and Neo4j without wiping existing data.

Usage:
    # Full end-to-end run: download 10 recent cs.AI papers, build graph, package diff, upload to remote server:
    python scripts/daily_arxiv_sync.py --server http://192.168.50.20:8855 --limit 10

    # Custom category (e.g. cs.LG, cs.CL, stat.ML):
    python scripts/daily_arxiv_sync.py --server http://192.168.50.20:8855 --category cs.LG --limit 5

    # Keyword search query on arXiv:
    python scripts/daily_arxiv_sync.py --server http://192.168.50.20:8855 --query "world model" --limit 5

    # Local preparation only (download + build + package without uploading):
    python scripts/daily_arxiv_sync.py --skip-upload --limit 5

    # Dry-run mode:
    python scripts/daily_arxiv_sync.py --server http://192.168.50.20:8855 --dry-run
"""

import sys
import os
import time
import json
import re
import argparse
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure archive_downloader source is on sys.path
ARCHIVE_SRC = REPO_ROOT / "archive_downloader" / "src"
if str(ARCHIVE_SRC) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_SRC))

from archive_downloader.downloader import ArxivDownloader
from archive_downloader.indexer import DownloadIndex
from bookeeper.backup import create_incremental_backup

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None


def log_step(title: str, subtitle: str = ""):
    if HAS_RICH:
        console.print(f"\n[bold cyan]▶ {title}[/bold cyan] [dim]{subtitle}[/dim]")
    else:
        print(f"\n▶ {title} {subtitle}")


def log_success(msg: str):
    if HAS_RICH:
        console.print(f"[bold green]✓[/bold green] {msg}")
    else:
        print(f"✓ {msg}")


def log_warning(msg: str):
    if HAS_RICH:
        console.print(f"[bold yellow]⚠ {msg}[/bold yellow]")
    else:
        print(f"⚠ {msg}")


def log_error(msg: str):
    if HAS_RICH:
        console.print(f"[bold red]✗ {msg}[/bold red]")
    else:
        print(f"✗ {msg}")


def resolve_category_url(category_or_url: str) -> str:
    """Normalize arXiv category abbreviation (e.g. cs.AI) into list URL."""
    val = category_or_url.strip()
    if val.startswith("http://") or val.startswith("https://"):
        return val
    # Clean e.g. "cs.AI" or "cs.LG"
    cat_clean = val.replace("/", ".")
    return f"https://arxiv.org/list/{cat_clean}/recent"


# ==============================================================================
# Phase 1: Downloader (arXiv Crawl)
# ==============================================================================

def run_download_phase(
    downloads_dir: Path,
    limit: int = 10,
    category: str = "cs.AI",
    query: Optional[str] = None,
    new_only: bool = True,
    delay: float = 0.5,
) -> Tuple[List[Path], Dict[str, Any]]:
    """
    Crawl arXiv and download recent articles.
    Returns (list of newly downloaded PDF file paths, stats_dict).
    """
    downloads_dir.mkdir(parents=True, exist_ok=True)
    index_obj = DownloadIndex(downloads_dir)
    existing_arxiv_ids = set(index_obj.entries.keys())

    cat_url = resolve_category_url(category) if not query else None

    log_step(
        "Phase 1: Downloading Recent Articles from arXiv",
        f"Category: {category if not query else query} (target: {limit} articles)"
    )

    downloader = ArxivDownloader(
        base_url=cat_url,
        query=query,
        output_dir=downloads_dir,
        limit=limit,
        count_existing=not new_only,
        request_delay=delay,
        write_opf=True,
    )

    t0 = time.perf_counter()
    stats = downloader.run()
    elapsed = time.perf_counter() - t0

    # Reload index to detect newly registered articles
    index_obj.load()
    newly_added_ids = set(index_obj.entries.keys()) - existing_arxiv_ids

    new_pdf_paths: List[Path] = []
    for aid in newly_added_ids:
        entry = index_obj.get(aid)
        if entry:
            rel_file = entry.file_path or entry.file_name
            if rel_file:
                p = downloads_dir / rel_file
                if p.is_file():
                    new_pdf_paths.append(p)

    # Sort deterministically
    new_pdf_paths.sort(key=lambda p: p.name)

    stats_summary = {
        "new_downloaded": len(new_pdf_paths),
        "total_inspected": stats.total_inspected,
        "already_existing": stats.already_existing,
        "failed": stats.failed,
        "duration_seconds": round(elapsed, 2),
    }

    log_success(
        f"Download complete: [bold white]{len(new_pdf_paths)}[/bold white] new articles downloaded "
        f"([dim]{stats.already_existing} already existed, {stats.failed} failed[/dim]) in {elapsed:.1f}s"
    )

    return new_pdf_paths, stats_summary


# ==============================================================================
# Phase 2: Knowledge Graph Ingestion (Build Graph)
# ==============================================================================

def run_build_graph_phase(
    new_files: List[Path],
    downloads_dir: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    config_diff: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    postprocess: bool = True,
) -> Dict[str, Any]:
    """
    Ingest newly downloaded articles into Bookkeeper knowledge graph.
    """
    import subprocess

    if not manifest_path:
        manifest_path = output_dir / "daily_new_articles.txt"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for fp in new_files:
            f.write(f"{fp.resolve()}\n")

    log_step(
        "Phase 2: Ingesting Articles into Knowledge Graph",
        f"Processing {len(new_files)} new book(s) via bookeeper build-graph"
    )

    cmd = [
        "uv", "run", "bookeeper", "build-graph",
        "--books-dir", str(downloads_dir.resolve()),
        "--file-list", str(manifest_path.resolve()),
    ]
    if config_path:
        cmd.extend(["--config", str(config_path.resolve())])
    if config_diff:
        cmd.extend(["--config-diff", str(config_diff.resolve())])

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=False)
    if proc.returncode != 0:
        log_error(f"build-graph failed with exit code {proc.returncode}")
        raise RuntimeError(f"build-graph failed with exit code {proc.returncode}")

    # Optional post-processing
    pp_duration = 0.0
    if postprocess:
        log_step("Phase 2b: Running Idea Post-Processing", "Consolidating clusters via postprocess-ideas")
        pp_cmd = [
            "uv", "run", "bookeeper", "postprocess-ideas",
            "--graph-file", str((output_dir / "knowledge_graph.json").resolve()),
            "--min-occurrences", "9999",  # Fast deterministic JEPA cluster resolution
        ]
        if config_diff:
            pp_cmd.extend(["--config-diff", str(config_diff.resolve())])
        t_pp = time.perf_counter()
        subprocess.run(pp_cmd, cwd=str(REPO_ROOT), check=False)
        pp_duration = round(time.perf_counter() - t_pp, 2)

    total_duration = round(time.perf_counter() - t0, 2)
    log_success(f"Graph build completed in {total_duration:.1f}s (post-processing: {pp_duration:.1f}s)")

    return {
        "success": True,
        "articles_processed": len(new_files),
        "manifest_path": str(manifest_path),
        "duration_seconds": total_duration,
    }


# ==============================================================================
# Phase 3: Incremental Update Preparation
# ==============================================================================

def run_incremental_backup_phase(
    output_dir: Path,
    downloads_dir: Path,
    baseline: str = "neo4j",
    format_type: str = "zip",
) -> Tuple[Path, Dict[str, Any]]:
    """
    Calculate conceptual diff and create an incremental update package (.zip).
    """
    log_step(
        "Phase 3: Preparing Incremental Update Archive",
        f"Calculating diff against baseline: '{baseline}'"
    )

    t0 = time.perf_counter()
    now_str = time.strftime("%Y%m%d_%H%M%S")
    backup_dest = output_dir / "backups" / f"incremental_update_{now_str}.{format_type}"
    backup_dest.parent.mkdir(parents=True, exist_ok=True)

    res = create_incremental_backup(
        source_dir=output_dir,
        baseline=baseline,
        destination=backup_dest,
        format_type=format_type,
        books_dir=downloads_dir,
        include_books=True,
        include_chunks=True,
        include_obsidian=True,
        include_states=True,
    )

    elapsed = round(time.perf_counter() - t0, 2)
    size_mb = res["compressed_bytes"] / (1024 * 1024)

    log_success(
        f"Incremental archive created: [bold white]{backup_dest.name}[/bold white] "
        f"([bold white]{size_mb:.2f} MB[/bold white], {res['total_files']} files) in {elapsed:.1f}s"
    )

    # Save reference link to last synced backup for offline chaining
    last_backup_ptr = output_dir / "last_synced_backup.zip"
    try:
        if last_backup_ptr.exists() or last_backup_ptr.is_symlink():
            last_backup_ptr.unlink()
        last_backup_ptr.symlink_to(backup_dest.resolve())
    except Exception:
        pass

    return backup_dest, res


# ==============================================================================
# Phase 4: Remote API Upload & Restore
# ==============================================================================

def upload_incremental_backup(
    server_url: str,
    archive_path: Path,
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    Upload the incremental backup to the remote Bookkeeper server via REST API:
    POST /api/backup/restore?incremental=true&clean=false
    """
    log_step(
        "Phase 4: Uploading Incremental Update to Remote Server",
        f"Destination: {server_url}/api/backup/restore"
    )

    target_url = server_url.strip()
    if not target_url.startswith("http://") and not target_url.startswith("https://"):
        target_url = f"http://{target_url}"
    target_url = target_url.rstrip("/")
    restore_endpoint = f"{target_url}/api/backup/restore"

    archive_bytes = archive_path.read_bytes()
    arch_size_mb = len(archive_bytes) / (1024 * 1024)
    boundary = f"----BookkeeperDailySyncBoundary{int(time.time())}"

    body_parts = [
        f"--{boundary}\r\n".encode("latin1"),
        b'Content-Disposition: form-data; name="clean"\r\n\r\n',
        b"false\r\n",
        f"--{boundary}\r\n".encode("latin1"),
        b'Content-Disposition: form-data; name="incremental"\r\n\r\n',
        b"true\r\n",
        f"--{boundary}\r\n".encode("latin1"),
        f'Content-Disposition: form-data; name="file"; filename="{archive_path.name}"\r\n'.encode("latin1"),
        b"Content-Type: application/zip\r\n\r\n",
        archive_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("latin1"),
    ]
    payload = b"".join(body_parts)

    req = urllib.request.Request(
        restore_endpoint,
        data=payload,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(payload)),
        },
        method="POST",
    )

    t0 = time.perf_counter()
    if HAS_RICH:
        with console.status(f"[bold cyan]Uploading {arch_size_mb:.2f} MB to {target_url} and merging into Neo4j...[/bold cyan]"):
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
    else:
        print(f"Uploading {arch_size_mb:.2f} MB to {target_url}...")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))

    elapsed = round(time.perf_counter() - t0, 2)

    if not resp_data.get("success"):
        err = resp_data.get("error", "Unknown remote restore error")
        log_error(f"Remote server rejected backup: {err}")
        raise RuntimeError(f"Remote upload failed: {err}")

    neo4j_stats = resp_data.get("neo4j_export") or {}
    merge_stats = resp_data.get("merge_stats") or {}
    restored_files = resp_data.get("restored_files", 0)

    log_success(
        f"Successfully synchronized with remote server in {elapsed:.1f}s!\n"
        f"  • Files Restored: [bold white]{restored_files}[/bold white]\n"
        f"  • Neo4j Nodes Upserted: [bold green]{neo4j_stats.get('nodes_upserted', merge_stats.get('nodes_added', 0))}[/bold green]\n"
        f"  • Neo4j Relationships: [bold green]{neo4j_stats.get('edges_upserted', merge_stats.get('edges_added', 0))}[/bold green]"
    )

    return resp_data


# ==============================================================================
# Main Orchestration Loop
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for daily sync pipeline."""
    parser = argparse.ArgumentParser(
        description="End-to-End Daily arXiv Article Crawler, Knowledge Graph Builder, and Incremental Server Sync."
    )
    parser.add_argument(
        "--server", "-s",
        type=str,
        default=None,
        help="Remote Bookkeeper server URL (e.g. http://192.168.50.20:8855). If omitted, prepares package locally without uploading.",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Target number of articles to download / process (default: 10).",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default="cs.AI",
        help="arXiv category abbreviation (e.g. cs.AI, cs.LG, cs.CL, stat.ML) or full list URL (default: cs.AI).",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Optional search query to match on arXiv instead of category feed.",
    )
    parser.add_argument(
        "--books-dir", "-b",
        type=Path,
        default=REPO_ROOT / "archive_downloader" / "downloads",
        help="Directory to store downloaded PDFs (default: archive_downloader/downloads).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=REPO_ROOT / "output",
        help="Knowledge artifacts directory (default: output).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--config-diff",
        type=Path,
        default=None,
        help="Path to YAML config overlay.",
    )
    parser.add_argument(
        "--diff-against",
        type=str,
        default="neo4j",
        help="Baseline for incremental diff: 'neo4j' (default), path to prior backup archive, or 'auto'.",
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        default=True,
        help="Crawl pages until exactly N brand-new articles are downloaded (default: True).",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip idea post-processing clustering before packaging.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force graph build and packaging even if 0 new articles were downloaded.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip arXiv download phase and process already downloaded literature.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip graph build phase and package/sync current graph state.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip remote API upload and keep incremental package on local disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview all execution steps without downloading, modifying, or uploading.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between arXiv requests (default: 0.5s).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP upload timeout in seconds (default: 600s).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    downloads_dir = args.books_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    t_global_start = time.perf_counter()

    if HAS_RICH:
        console.print(
            Panel.fit(
                f"[bold cyan]Daily arXiv Knowledge Ingestion & Sync Pipeline[/bold cyan]\n"
                f"• Target Category / Query: [yellow]{args.category if not args.query else args.query}[/yellow]\n"
                f"• Target Article Quota: [bold white]{args.limit}[/bold white] articles\n"
                f"• Downloads Storage: [dim]{downloads_dir}[/dim]\n"
                f"• Knowledge Storage: [dim]{output_dir}[/dim]\n"
                f"• Incremental Baseline: [cyan]{args.diff_against}[/cyan]\n"
                f"• Remote Destination: [bold green]{args.server or '(Local Packaging Only)'}[/bold green]",
                title="Bookkeeper Daily Automation",
                border_style="cyan",
            )
        )

    if args.dry_run:
        log_warning("Dry-run requested: No articles will be downloaded, processed, or uploaded.")
        return 0

    # Auto-resolve baseline if requested
    diff_baseline = args.diff_against
    if diff_baseline == "auto":
        last_ptr = output_dir / "last_synced_backup.zip"
        if last_ptr.is_file():
            diff_baseline = str(last_ptr.resolve())
        else:
            diff_baseline = "neo4j"

    new_pdf_files: List[Path] = []
    download_stats = {}

    # Phase 1: Downloader
    if not args.skip_download:
        new_pdf_files, download_stats = run_download_phase(
            downloads_dir=downloads_dir,
            limit=args.limit,
            category=args.category,
            query=args.query,
            new_only=args.new_only,
            delay=args.delay,
        )

        if not new_pdf_files and not args.force:
            log_success("No new articles published since last run. Everything is up to date!")
            return 0
    else:
        log_step("Phase 1: Downloader Skipped", "Using existing PDFs on disk")
        # In skip-download mode, locate all PDFs in downloads_dir
        new_pdf_files = sorted(list(downloads_dir.glob("*.pdf")))[:args.limit]

    # Phase 2: Build Graph
    if not args.skip_build and new_pdf_files:
        run_build_graph_phase(
            new_files=new_pdf_files,
            downloads_dir=downloads_dir,
            output_dir=output_dir,
            config_path=args.config,
            config_diff=args.config_diff,
            postprocess=not args.no_postprocess,
        )
    elif args.skip_build:
        log_step("Phase 2: Build Graph Skipped", "Packaging existing graph state")

    # Phase 3: Prepare Incremental Update
    inc_archive, backup_meta = run_incremental_backup_phase(
        output_dir=output_dir,
        downloads_dir=downloads_dir,
        baseline=diff_baseline,
        format_type="zip",
    )

    # Phase 4: Upload to Remote Server
    upload_res = None
    if args.server and not args.skip_upload:
        upload_res = upload_incremental_backup(
            server_url=args.server,
            archive_path=inc_archive,
            timeout=args.timeout,
        )
    elif not args.server:
        log_step("Phase 4: Remote Upload Skipped", "No remote --server specified.")
        console.print(
            f"\n[dim]To synchronize this package to your remote server, run:\n"
            f"  bookeeper sync http://<server-ip>:<port> --incremental --diff-against {diff_baseline}[/dim]\n"
        )

    total_time = round(time.perf_counter() - t_global_start, 1)

    if HAS_RICH:
        summary_table = Table(title="Daily Pipeline Execution Summary", border_style="green")
        summary_table.add_column("Pipeline Phase", style="bold cyan")
        summary_table.add_column("Details & Output", style="white")
        summary_table.add_column("Status", style="bold green")

        summary_table.add_row(
            "1. Downloader",
            f"{len(new_pdf_files)} new articles fetched from arXiv",
            "Completed" if not args.skip_download else "Skipped",
        )
        summary_table.add_row(
            "2. Build Graph",
            f"Concepts & chunks extracted into {output_dir.name}",
            "Completed" if not args.skip_build else "Skipped",
        )
        summary_table.add_row(
            "3. Incremental Diff",
            f"Package: {inc_archive.name} ({backup_meta.get('compressed_bytes', 0)/(1024*1024):.2f} MB)",
            "Packaged",
        )
        summary_table.add_row(
            "4. Remote Server Sync",
            f"Server: {args.server}" if args.server and not args.skip_upload else "Local only",
            "Restored & Merged" if upload_res else "Bypassed",
        )
        console.print(summary_table)

    log_success(f"Pipeline finished successfully in {total_time}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
