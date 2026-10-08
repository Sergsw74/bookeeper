"""
Production-quality Typer CLI interface for bookeeper with rich progress bars and configurable Ollama / Calibre paths.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from langchain_ollama import OllamaEmbeddings
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from bookeeper.calibre.client import CalibreClient
from bookeeper.calibre.parser import BookParser
from bookeeper.config import Settings, get_settings
from bookeeper.graph.exporters import GraphMLExporter, ObsidianExporter
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import ChunkStore, HierarchicalChunker
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import KnowledgeExtractor
from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings
from bookeeper.processing.state import ProgressTracker

app = typer.Typer(
    name="bookeeper",
    help="Connects Calibre libraries to local/remote Ollama LLMs and extracts Concept Knowledge Graphs.",
    add_completion=False,
)
console = Console()


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that creates daemon threads so cancellation terminates immediately without hanging."""

    def _adjust_thread_count(self):
        orig_thread = threading.Thread

        def _daemon_thread(*args, **kwargs):
            kwargs["daemon"] = True
            return orig_thread(*args, **kwargs)

        threading.Thread = _daemon_thread
        try:
            super()._adjust_thread_count()
        finally:
            threading.Thread = orig_thread


def _get_effective_settings(
    config_path: Optional[str] = None,
    calibre_path: Optional[str] = None,
    ollama_url: Optional[str] = None,
    model: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Settings:
    """Retrieve settings and merge explicit CLI arguments with highest precedence."""
    cfg = get_settings(config_path)
    overrides = {}
    if calibre_path:
        overrides["calibre_library_path"] = calibre_path
    if ollama_url:
        overrides["ollama_base_url"] = ollama_url
    if model:
        overrides["llm_model"] = model
    if embedding_model:
        overrides["embedding_model"] = embedding_model

    if overrides:
        data = cfg.model_dump()
        data.update(overrides)
        return Settings(**data)
    return cfg


def _print_pool_configuration(cfg: Settings, console: Console, show_embeddings: bool = False) -> None:
    """Print clean summary of configured Ollama pool nodes and models."""
    servers = cfg.resolved_ollama_servers
    emb_suffix = f" | Embeddings: [bold cyan]{cfg.embedding_model}[/bold cyan]" if show_embeddings else ""
    if servers and len(servers) > 1:
        nodes_str = ", ".join(f"{s.name or s.url} ({s.url})" for s in servers)
        console.print(
            f"[dim]Configured Ollama Multi-Server Pool ({len(servers)} nodes): [bold cyan]{nodes_str}[/bold cyan] | "
            f"LLM: [bold cyan]{cfg.llm_model}[/bold cyan]{emb_suffix}[/dim]"
        )
    else:
        console.print(
            f"[dim]Configured Ollama: [bold cyan]{cfg.ollama_base_url}[/bold cyan] | "
            f"LLM: [bold cyan]{cfg.llm_model}[/bold cyan]{emb_suffix}[/dim]"
        )


def _perform_ollama_warmup(extractor: KnowledgeExtractor, console: Console) -> Dict[str, Any]:
    """Execute warmup ping to load model and report CPU vs GPU acceleration status across server pool."""
    pool_nodes = extractor.pool.nodes
    with console.status(
        f"[bold blue]Checking Ollama acceleration across server pool ({len(pool_nodes)} node(s))...[/bold blue]"
    ):
        status = extractor.warmup_and_check_device()

    servers = status.get("servers", [])
    primary = status.get("primary", {})

    # Display multi-server pool table if more than one server configured
    if len(servers) > 1:
        table = Table(
            title=f"Ollama Multi-Server Pool ({len(servers)} servers, {extractor.cooldown_seconds // 60}m failover cooldown)",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Priority", justify="center", width=8)
        table.add_column("Endpoint", style="bold white", min_width=25)
        table.add_column("Device / Mode", style="magenta")
        table.add_column("VRAM Offload", justify="right")
        table.add_column("Pool State", style="green")

        for s in sorted(servers, key=lambda x: x.get("priority", 99)):
            pri = str(s.get("priority", 1))
            url = s.get("url", "")
            dev = s.get("device", "Unknown")
            st = s.get("status", "ok")
            vram_mb = s.get("size_vram", 0) / (1024 * 1024)
            size_mb = s.get("size", 0) / (1024 * 1024)
            pct = s.get("vram_pct", 0.0)
            vram_str = f"{vram_mb:.0f} / {size_mb:.0f} MB ({pct}%)" if size_mb > 0 else "N/A"
            pool_state = "[bold green]Primary / Active[/bold green]" if s["url"] == primary.get("url") else "[cyan]Standby / Backup[/cyan]"
            if st != "ok":
                pool_state = f"[red]{st}[/red]"
            table.add_row(pri, url, dev, vram_str, pool_state)

        console.print(table)

    is_gpu = primary.get("is_gpu", False)
    size_mb = primary.get("size", 0) / (1024 * 1024)
    vram_mb = primary.get("size_vram", 0) / (1024 * 1024)
    dev = primary.get("device", "Unknown")

    if not is_gpu:
        warning_msg = (
            f"[bold yellow]⚠️ Active Ollama server '[cyan]{primary.get('url')}[/cyan]' is executing model '[cyan]{primary.get('model', extractor.model_name)}[/cyan]' on [bold red]CPU[/bold red][/bold yellow]\n\n"
            f"• [bold]VRAM Allocated:[/bold] 0 MB / {size_mb:.0f} MB (0% offloaded)\n"
            f"• [bold]Inference Runner:[/bold] {primary.get('runner', 'llamacpp')}\n"
            f"• [bold]Multi-Server Pool:[/bold] {len(servers)} server(s) configured (auto-retry cooldown: {extractor.cooldown_seconds // 60}m)\n\n"
            f"[dim]Note: Extraction will be slower on CPU. If this server fails, requests will automatically fail over to next server.[/dim]"
        )
        console.print(
            Panel(
                warning_msg,
                title="[bold yellow]Hardware Acceleration Notice[/bold yellow]",
                border_style="yellow",
            )
        )
    else:
        vram_pct = primary.get("vram_pct", 100.0)
        console.print(
            f"[bold green]✓ Active Ollama GPU acceleration:[/bold green] [bold cyan]{dev}[/bold cyan] "
            f"({vram_mb:.0f} MB / {size_mb:.0f} MB VRAM, {vram_pct}% offloaded) at {primary.get('url')}"
        )

    return status


@app.command()
def config(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to custom config.yaml file."
    ),
    calibre_path: Optional[str] = typer.Option(
        None, "--calibre-path", help="Override Calibre library path or SMB share mount."
    ),
    ollama_url: Optional[str] = typer.Option(
        None, "--ollama-url", "-u", help="Override Ollama base URL (e.g. http://192.168.50.15:11434)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override Ollama LLM model name (e.g. llama3.1:8b)."
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", help="Override Ollama embedding model (e.g. nomic-embed-text)."
    ),
):
    """Display the active bookeeper configuration settings."""
    cfg = _get_effective_settings(config_path, calibre_path, ollama_url, model, embedding_model)
    servers_desc = []
    for s in cfg.resolved_ollama_servers:
        name_tag = f" ({s.name})" if s.name else ""
        servers_desc.append(f"    • [cyan]{s.url}[/cyan]{name_tag} [dim](priority: {s.priority})[/dim]")
    servers_block = "\n".join(servers_desc)

    console.print(
        Panel.fit(
            f"[bold green]Calibre Library / SMB Share:[/bold green] {cfg.calibre_library_path}\n"
            f"[bold green]Calibre Auth:[/bold green] user={cfg.calibre_user or '[dim]none[/dim]'}\n"
            f"[bold green]Ollama Failover Pool ({len(cfg.resolved_ollama_servers)} server(s)):[/bold green]\n{servers_block}\n"
            f"[bold green]Failover Cooldown:[/bold green] {cfg.failover_cooldown_seconds}s\n"
            f"[bold green]LLM Model:[/bold green] {cfg.llm_model}\n"
            f"[bold green]Embedding Model:[/bold green] {cfg.embedding_model}\n"
            f"[bold green]Similarity Threshold:[/bold green] {cfg.similarity_threshold}\n"
            f"[bold green]Output Directory:[/bold green] {cfg.resolved_output_dir}",
            title="Active Configuration",
        )
    )


@app.command("list-books")
def list_books(
    limit: Optional[int] = typer.Option(
        25, "--limit", "-n", help="Maximum number of books to display (use 0 for all)."
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s", help="Filter books by title or author query."
    ),
    calibre_path: Optional[str] = typer.Option(
        None, "--calibre-path", help="Calibre library path or SMB mount (e.g. /Volumes/share/calibre)."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
):
    """
    List and search books in the configured Calibre library or mounted SMB share.
    Useful for testing database connectivity and discovering book IDs.
    """
    cfg = _get_effective_settings(config_path, calibre_path=calibre_path)
    client = CalibreClient(
        library_path=cfg.calibre_library_path,
        user=cfg.calibre_user,
        password=cfg.calibre_password,
    )

    if not client.is_available():
        console.print(
            f"[bold red]Error:[/bold red] Cannot access Calibre library at '[bold cyan]{cfg.calibre_library_path}[/bold cyan]'.\n"
            f"If this is on a remote SMB share, ensure the share is mounted (e.g. /Volumes/...) or that calibredb is installed."
        )
        raise typer.Exit(1)

    with console.status(f"[bold blue]Connecting to Calibre library at {cfg.calibre_library_path}...[/bold blue]"):
        try:
            books = client.list_books(fields=["id", "title", "authors", "formats"])
        except Exception as e:
            console.print(f"[bold red]Failed to read Calibre library:[/bold red] {e}")
            raise typer.Exit(1)

    if search:
        q = search.lower()
        books = [
            b
            for b in books
            if q in b.get("title", "").lower()
            or any(q in a.lower() for a in b.get("authors", []))
        ]

    total_count = len(books)
    if limit and limit > 0:
        books = books[:limit]

    table = Table(
        title=f"Calibre Library: {cfg.calibre_library_path} (Showing {len(books)} of {total_count})",
        show_lines=False,
    )
    table.add_column("ID", justify="right", style="cyan", width=6)
    table.add_column("Title", style="bold white", min_width=30)
    table.add_column("Authors", style="green", width=24)
    table.add_column("Formats", style="magenta", width=16)

    for b in books:
        authors = b.get("authors", [])
        authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
        formats = b.get("formats", [])
        formats_str = ", ".join(formats) if isinstance(formats, list) else str(formats)

        title = b.get("title", "Untitled")
        if len(title) > 50:
            title = title[:47] + "..."

        table.add_row(
            str(b.get("id", "")),
            title,
            authors_str[:24],
            formats_str or "[dim]None[/dim]",
        )

    console.print(table)
    if limit and total_count > limit:
        console.print(
            f"[dim]Showing first {limit} books. Use --limit 0 to view all {total_count} books, or --search <term>.[/dim]"
        )


@app.command("clean-metadata")
def clean_metadata(
    book_id: Optional[int] = typer.Option(
        None, "--book-id", "-b", help="Specific Calibre book ID to clean."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Inspect and display cleaned metadata without writing back."
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from previous checkpoint, skipping already cleaned books."
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Only retry books that previously failed in the checkpoint state."
    ),
    reset_progress: bool = typer.Option(
        False, "--reset-progress", help="Reset saved progress checkpoint and start from scratch."
    ),
    start_from_id: Optional[int] = typer.Option(
        None, "--start-from-id", help="Only process books with ID >= this value (useful for manual restart)."
    ),
    state_file: Optional[str] = typer.Option(
        None, "--state-file", help="Custom path for progress checkpoint JSON file."
    ),
    calibre_path: Optional[str] = typer.Option(
        None, "--calibre-path", help="Calibre library path or SMB mount (e.g. /Volumes/share/calibre)."
    ),
    ollama_url: Optional[str] = typer.Option(
        None, "--ollama-url", "-u", help="Ollama server URL (e.g. http://192.168.50.15:11434)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Ollama LLM model name (e.g. llama3.1:8b, qwen2.5:7b-instruct)."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
    skip_warmup: bool = typer.Option(
        False, "--skip-warmup", help="Skip Ollama warmup and GPU acceleration check."
    ),
    stage_db: Optional[bool] = typer.Option(
        None, "--stage-db/--no-stage-db", help="Stage metadata.db locally on SSD for fast transactions and sync back on completion (default: true)."
    ),
    backup_db: Optional[bool] = typer.Option(
        None, "--backup-db/--no-backup-db", help="Create metadata.db.bak on remote library before uploading updated staged database (default: true)."
    ),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-t", help="Max concurrent active tasks across Ollama servers (default: auto: 3x alive servers, up to 10)."
    ),
    fresh_db: bool = typer.Option(
        False, "--fresh-db", help="Force downloading a fresh copy of metadata.db from Calibre, discarding any existing local staged database."
    ),
):
    """
    Query Calibre (local, SMB share, or server), inspect titles/authors/summaries,
    prompt Ollama to normalize, and write clean metadata back.
    Supports persistent checkpointing to resume from interrupted points or retry only failed books.
    """
    cfg = _get_effective_settings(config_path, calibre_path, ollama_url, model)
    effective_stage_db = cfg.stage_metadata_db if stage_db is None else stage_db
    effective_backup_db = cfg.backup_metadata_db if backup_db is None else backup_db

    client = CalibreClient(
        library_path=cfg.calibre_library_path,
        user=cfg.calibre_user,
        password=cfg.calibre_password,
    )

    if not client.is_available():
        console.print(
            f"[bold red]Error:[/bold red] Cannot access Calibre library at '{cfg.calibre_library_path}'.\n"
            f"Ensure the SMB share is mounted or calibredb is in PATH."
        )
        raise typer.Exit(1)

    # Local DB Staging: Copy metadata.db locally to bypass SMB network roundtrips and locking
    should_stage = (
        effective_stage_db
        and not client.is_remote_url
        and client.db_path is not None
        and client.db_path.is_file()
    )

    reused_staged = False
    if should_stage:
        staged_target = cfg.resolved_staged_db_path

        if reset_progress or fresh_db:
            if staged_target.is_file():
                try:
                    staged_target.unlink()
                except Exception:
                    pass

        # Check if an existing staged database already exists locally from a previous session
        if resume and not reset_progress and not fresh_db and staged_target.is_file() and staged_target.stat().st_size > 0:
            try:
                CalibreClient._verify_sqlite_integrity(staged_target, allow_index_warnings=True)
                client.staged_db_path = staged_target
                client.is_staged = True
                reused_staged = True
                console.print(
                    f"[bold cyan]Found existing local staged database at {staged_target}.[/bold cyan] "
                    f"[dim]Reusing local SSD database to preserve previous progress and avoid re-downloading.[/dim]"
                )
            except Exception as e:
                console.print(
                    f"[dim yellow]Existing local staged database could not be reused ({e}). Re-staging fresh copy from Calibre share...[/dim yellow]"
                )
                client.cleanup_staged(delete_file=True)

        if not client.is_staged:
            try:
                total_bytes = client.db_path.stat().st_size
                size_mb = total_bytes / (1024 * 1024)
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TextColumn("• {task.completed:.1f}/{task.total:.1f} MB"),
                    console=console,
                ) as stage_progress:
                    stage_task = stage_progress.add_task(
                        f"Downloading metadata.db ({size_mb:.1f} MB) locally for fast SSD transactions...",
                        total=size_mb,
                    )

                    def _on_stage_progress(copied_bytes: int, total_b: int):
                        stage_progress.update(stage_task, completed=copied_bytes / (1024 * 1024))

                    staged_file = client.stage_database(
                        staged_target,
                        progress_callback=_on_stage_progress,
                    )

                console.print(f"[dim green]✓ Staged database at {staged_file} (integrity check ok).[/dim green]")
            except Exception as e:
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Failed to stage metadata.db locally ({e}). "
                    f"Falling back to direct database access."
                )
                client.cleanup_staged(delete_file=False)
                should_stage = False

    tracker_path = Path(state_file) if state_file else cfg.resolved_state_file
    tracker = ProgressTracker(tracker_path)

    if reset_progress:
        tracker.clear("clean_metadata")
        console.print("[dim yellow]Reset progress checkpoint for clean_metadata.[/dim yellow]")

    extractor = KnowledgeExtractor.from_settings(cfg)
    _print_pool_configuration(cfg, console, show_embeddings=False)

    if not skip_warmup:
        _perform_ollama_warmup(extractor, console)

    query_msg = "Reading book catalog from staged local SQLite..." if client.is_staged else "Querying Calibre library over network/calibredb..."
    with console.status(f"[bold blue]{query_msg}[/bold blue]"):
        try:
            books = client.list_books(fields=["id", "title", "authors", "comments", "formats"])
        except Exception as e:
            if should_stage and client.is_staged:
                client.cleanup_staged(delete_file=True)
            console.print(f"[bold red]Failed to query Calibre library:[/bold red] {e}")
            raise typer.Exit(1)

    if book_id is not None:
        books = [b for b in books if b.get("id") == book_id]
        if not books:
            console.print(f"[bold yellow]Book ID {book_id} not found in Calibre library.[/bold yellow]")
            raise typer.Exit(1)
    elif retry_failed:
        failed_ids = tracker.get_failed_ids("clean_metadata")
        if not failed_ids:
            console.print("[bold green]No failed books found in checkpoint to retry.[/bold green]")
            raise typer.Exit(0)
        books = [b for b in books if b.get("id") in failed_ids]
        console.print(
            f"[bold cyan]Retrying {len(books)} previously failed book(s) (IDs: {sorted(failed_ids)})...[/bold cyan]"
        )
    else:
        # Filter by start_from_id if requested
        if start_from_id is not None:
            books = [b for b in books if b.get("id", 0) >= start_from_id]

        # Resume from checkpoint
        if resume:
            completed_ids = tracker.get_completed_ids("clean_metadata")
            orig_len = len(books)
            books = [b for b in books if b.get("id") not in completed_ids]
            skipped = orig_len - len(books)
            if skipped > 0:
                console.print(
                    f"[dim cyan]Checkpoint Resume: Skipped {skipped} already processed or skipped book(s). "
                    f"({len(books)} remaining. Use --no-resume to reprocess all).[/dim cyan]"
                )

    if not books:
        if should_stage and client.is_staged:
            client.cleanup_staged(delete_file=True)
        console.print("[bold green]All books are already cleaned! Nothing to process.[/bold green]")
        raise typer.Exit(0)

    num_servers = len(extractor.pool.alive_nodes) or len(extractor.pool.nodes)
    pool_concurrency = max_tasks if max_tasks is not None else cfg.calculate_pool_concurrency(num_servers)

    console.print(f"[bold green]Found {len(books)} book(s) to process.[/bold green]")
    console.print(
        f"[bold cyan]Parallel Task Pool:[/bold cyan] Running up to {pool_concurrency} active task(s) "
        f"across {num_servers} Ollama server(s) (low: {num_servers}, max default: {min(cfg.max_active_tasks_cap, 3 * num_servers)})."
    )

    success_count = 0
    failed_count = 0
    skipped_graphical_count = 0

    db_lock = threading.Lock()
    display_lock = threading.Lock()
    stats_lock = threading.Lock()

    active_tasks_count = 0
    durations: List[float] = []
    abort_event = threading.Event()
    interrupted = False

    def _calc_stats() -> Tuple[float, float, float]:
        """Return (avg, p80, fastest) in seconds."""
        with stats_lock:
            if not durations:
                return (0.0, 0.0, 0.0)
            avg_val = sum(durations) / len(durations)
            fastest_val = min(durations)
            sorted_d = sorted(durations)
            p80_idx = int(0.80 * (len(sorted_d) - 1))
            p80_val = sorted_d[p80_idx]
            return (avg_val, p80_val, fastest_val)

    def _process_single_book(b_item: Dict[str, Any]) -> Dict[str, Any]:
        bid = b_item["id"]
        if abort_event.is_set():
            return {"id": bid, "status": "aborted"}

        raw_title = BookParser.repair_mojibake(b_item.get("title", "Untitled"))
        raw_authors = [BookParser.repair_mojibake(a) for a in b_item.get("authors", [])]
        raw_comments = BookParser.repair_mojibake(b_item.get("comments", ""))
        formats = b_item.get("formats", [])
        book_path_str = b_item.get("path")
        book_dir = Path(book_path_str) if book_path_str else None

        # Check if book only contains graphical formats (CBR, CBZ, DJVU)
        is_graphical = False
        if formats and BookParser.only_has_graphical_formats(formats):
            is_graphical = True
        elif book_dir and book_dir.exists() and BookParser.directory_only_has_graphical_formats(book_dir):
            is_graphical = True

        if is_graphical:
            fmts_label = ", ".join(formats) if formats else "graphical"
            return {
                "id": bid,
                "status": "skipped_graphical",
                "raw_title": raw_title,
                "reason": f"graphical format ({fmts_label})",
            }

        # Sample book content and file hint from filesystem / SMB share
        content_sample = None
        file_hint = None
        if book_path_str:
            if book_dir and book_dir.exists():
                content_sample, file_hint = BookParser.sample_content(book_dir)

        if abort_event.is_set():
            return {"id": bid, "status": "aborted"}

        # Track active in-flight task and timing
        nonlocal active_tasks_count
        with stats_lock:
            active_tasks_count += 1

        t0 = time.time()
        duration = 0.0
        server_used = "ollama"
        try:
            # Run Ollama structured normalization with content sample (load-balanced across servers)
            cleaned = extractor.clean_metadata(
                raw_title=raw_title,
                raw_authors=raw_authors,
                raw_comments=raw_comments,
                content_sample=content_sample,
                file_hint=file_hint,
            )
            duration = time.time() - t0
            server_used = extractor.pool.get_last_used_server() or "ollama"
            with stats_lock:
                durations.append(duration)
        finally:
            with stats_lock:
                active_tasks_count = max(0, active_tasks_count - 1)

        return {
            "id": bid,
            "status": "cleaned",
            "raw_title": raw_title,
            "raw_authors": raw_authors,
            "raw_comments": raw_comments,
            "cleaned": cleaned,
            "content_sample": content_sample,
            "file_hint": file_hint,
            "duration": duration,
            "server_used": server_used,
        }

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Normalizing metadata across {num_servers} Ollama server(s)...",
                total=len(books),
            )

            def _update_progress_description():
                avg_s, p80_s, fastest_s = _calc_stats()
                with stats_lock:
                    cur_active = active_tasks_count
                desc = (
                    f"Normalizing across {num_servers} server(s) | "
                    f"Active: [bold cyan]{cur_active}[/bold cyan]"
                )
                if durations:
                    desc += (
                        f" | avg: [bold green]{avg_s:.1f}s[/bold green] | "
                        f"p80: [bold yellow]{p80_s:.1f}s[/bold yellow] | "
                        f"fastest: [bold magenta]{fastest_s:.1f}s[/bold magenta]"
                    )
                progress.update(task, description=desc)

            _update_progress_description()

            executor = DaemonThreadPoolExecutor(max_workers=pool_concurrency)
            try:
                future_to_book = {
                    executor.submit(_process_single_book, b): b for b in books
                }

                try:
                    for future in as_completed(future_to_book):
                        if abort_event.is_set():
                            break

                        b_orig = future_to_book[future]
                        bid = b_orig["id"]
                        raw_title = BookParser.repair_mojibake(b_orig.get("title", "Untitled"))

                        try:
                            res = future.result()
                            if res.get("status") == "aborted":
                                continue
                            elif res["status"] == "skipped_graphical":
                                with display_lock:
                                    console.print(
                                        f"[dim yellow]⚡ Skipping book #{bid} ('{raw_title}'): {res['reason']}.[/dim yellow]"
                                    )
                                with db_lock:
                                    tracker.mark_skipped(
                                        "clean_metadata", bid, title=raw_title, reason=res["reason"]
                                    )
                                skipped_graphical_count += 1

                            elif res["status"] == "cleaned":
                                cleaned = res["cleaned"]
                                raw_authors = res["raw_authors"]
                                raw_comments = res["raw_comments"]
                                content_sample = res["content_sample"]
                                file_hint = res["file_hint"]
                                dur = res.get("duration", 0.0)
                                srv = res.get("server_used", "ollama")

                                # Display diff panel
                                table = Table(show_header=True, header_style="bold magenta", expand=True)
                                table.add_column("Field", style="dim", width=12)
                                table.add_column("Original Calibre Value")
                                table.add_column("Cleaned LLM Value", style="bold green")

                                table.add_row("Title", raw_title, cleaned.title)
                                table.add_row("Authors", ", ".join(raw_authors), cleaned.author)
                                table.add_row(
                                    "Summary",
                                    (raw_comments[:120] + "...") if raw_comments else "[dim]None[/dim]",
                                    cleaned.summary,
                                )
                                if file_hint:
                                    table.add_row(
                                        "Source File",
                                        file_hint,
                                        "[dim green]Content sampled[/dim green]" if content_sample else "[dim]Inspected[/dim]",
                                    )

                                with display_lock:
                                    console.print(Panel(table, title=f"Book #{bid} Metadata Diff"))

                                    if not dry_run:
                                        with db_lock:
                                            client.update_metadata(
                                                book_id=bid,
                                                title=cleaned.title,
                                                authors=[cleaned.author],
                                                comments=cleaned.summary,
                                            )
                                            tracker.mark_completed(
                                                "clean_metadata",
                                                bid,
                                                title=cleaned.title,
                                                metadata={"author": cleaned.author, "dry_run": False},
                                            )
                                        console.print(
                                            f"[green]✓ Successfully updated book #{bid} in Calibre[/green] "
                                            f"[dim]({dur:.1f}s via {srv})[/dim]"
                                        )
                                        success_count += 1
                                    else:
                                        console.print(f"[yellow]⚡ [Dry-Run] Skipped writing back to Calibre.[/yellow]")
                                        with db_lock:
                                            tracker.mark_completed(
                                                "clean_metadata",
                                                bid,
                                                title=cleaned.title,
                                                metadata={"author": cleaned.author, "dry_run": True},
                                            )
                                        success_count += 1

                        except Exception as e:
                            with display_lock:
                                console.print(f"[bold red]✗ Failed to process book #{bid} ('{raw_title}'):[/bold red] {e}")
                            with db_lock:
                                tracker.mark_failed("clean_metadata", bid, title=raw_title, error=str(e))
                            failed_count += 1
                        finally:
                            progress.advance(task)
                            _update_progress_description()

                except KeyboardInterrupt:
                    interrupted = True
                    abort_event.set()
                    progress.stop()
                    console.print("\n[bold yellow]Cancelled by user. Terminating pending pool tasks...[/bold yellow]")
                    for f in future_to_book:
                        f.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    # Unregister daemon threads from Python's atexit handler so exit is instant
                    import concurrent.futures.thread
                    with concurrent.futures.thread._global_shutdown_lock:
                        for t in list(executor._threads):
                            concurrent.futures.thread._threads_queues.pop(t, None)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        if interrupted:
            console.print(
                f"[bold yellow]Metadata cleaning stopped by user.[/bold yellow] "
                f"Successfully completed before cancel: [bold green]{success_count}[/bold green] book(s)."
            )
        else:
            summary_text = f"[bold green]Metadata cleaning finished.[/bold green] Completed: [bold green]{success_count}[/bold green]"
            if skipped_graphical_count > 0:
                summary_text += f" | Skipped (graphical): [yellow]{skipped_graphical_count}[/yellow]"
            if failed_count > 0:
                summary_text += f" | Failed: [bold red]{failed_count}[/bold red] [dim](run with --retry-failed to re-attempt)[/dim]"
            if durations:
                avg_s, p80_s, fastest_s = _calc_stats()
                summary_text += (
                    f"\n[bold cyan]Processing Performance:[/bold cyan] "
                    f"Avg: [bold green]{avg_s:.2f}s[/bold green] | "
                    f"80th Percentile (p80): [bold yellow]{p80_s:.2f}s[/bold yellow] | "
                    f"Fastest: [bold magenta]{fastest_s:.2f}s[/bold magenta] "
                    f"[dim](across {len(durations)} normalized books)[/dim]"
                )
            console.print(summary_text)

    finally:
        if should_stage and client.is_staged:
            sync_succeeded = False
            has_modifications = (success_count > 0) or reused_staged
            if not dry_run and has_modifications:
                try:
                    total_bytes = client.staged_db_path.stat().st_size
                    size_mb = total_bytes / (1024 * 1024)
                    console.print(
                        f"[bold blue]Syncing updated database ({success_count} newly modified book(s)) back to Calibre library...[/bold blue]"
                    )
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TextColumn("• {task.completed:.1f}/{task.total:.1f} MB"),
                        console=console,
                    ) as sync_progress:
                        sync_task = sync_progress.add_task(
                            "Uploading updated metadata.db to Calibre share...",
                            total=size_mb,
                        )

                        def _on_sync_progress(copied_bytes: int, total_b: int):
                            sync_progress.update(sync_task, completed=copied_bytes / (1024 * 1024))

                        client.sync_database(
                            create_backup=effective_backup_db,
                            progress_callback=_on_sync_progress,
                        )
                        sync_succeeded = True

                    backup_info = " (remote backup created: metadata.db.bak)" if effective_backup_db else ""
                    console.print(
                        f"[bold green]✓ Successfully synced staged metadata.db to Calibre library{backup_info}.[/bold green]"
                    )
                except Exception as sync_err:
                    console.print(
                        f"[bold red]Error syncing database back to Calibre:[/bold red] {sync_err}\n"
                        f"[yellow]Your local staged database with updates is preserved at: {client.staged_db_path}[/yellow]"
                    )
            elif dry_run:
                sync_succeeded = True
                if success_count > 0:
                    console.print("[dim yellow]⚡ [Dry-Run] Discarded staged database changes (no remote modifications made).[/dim yellow]")
            else:
                # 0 modifications and not reused
                sync_succeeded = True

            if client.is_staged:
                # Only delete local staged file if sync succeeded or was a discardable run
                client.cleanup_staged(delete_file=sync_succeeded)

        if interrupted:
            raise typer.Exit(code=130)


@app.command("build-graph")
def build_graph(
    book_id: Optional[int] = typer.Option(
        None, "--book-id", "-b", help="Process specific Calibre book ID."
    ),
    file_path: Optional[str] = typer.Option(
        None, "--file", "-f", help="Process a standalone EPUB or PDF file directly."
    ),
    all_books: bool = typer.Option(
        False, "--all", "-a", help="Process all books in the Calibre library."
    ),
    calibre_path: Optional[str] = typer.Option(
        None, "--calibre-path", help="Calibre library path or SMB mount (e.g. /Volumes/share/calibre)."
    ),
    ollama_url: Optional[str] = typer.Option(
        None, "--ollama-url", "-u", help="Ollama server URL (e.g. http://192.168.50.15:11434)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Ollama LLM model name (e.g. llama3.1:8b, qwen2.5:7b-instruct)."
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", help="Ollama embedding model name (e.g. nomic-embed-text)."
    ),
    export_obsidian: Optional[str] = typer.Option(
        None, "--export-obsidian", help="Custom destination directory for Obsidian Markdown vault."
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from previous checkpoint, skipping already processed books."
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Only retry books that previously failed in the checkpoint state."
    ),
    reset_progress: bool = typer.Option(
        False, "--reset-progress", help="Reset saved progress checkpoint and start from scratch."
    ),
    start_from_id: Optional[int] = typer.Option(
        None, "--start-from-id", help="Only process books with ID >= this value."
    ),
    state_file: Optional[str] = typer.Option(
        None, "--state-file", help="Custom path for progress checkpoint JSON file."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
    skip_warmup: bool = typer.Option(
        False, "--skip-warmup", help="Skip Ollama warmup and GPU acceleration check."
    ),
    chunks_dir: Optional[str] = typer.Option(
        None, "--chunks-dir", help="Custom directory for persistent chunks storage (default: output_dir/chunks)."
    ),
    rechunk: bool = typer.Option(
        False, "--rechunk", help="Force re-chunking from EPUB source even if chunks already exist in local storage."
    ),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-t", help="Max concurrent active tasks across Ollama servers (default: auto: 3x alive servers, up to 10)."
    ),
    stage_db: Optional[bool] = typer.Option(
        None, "--stage-db/--no-stage-db", help="Stage metadata.db locally on SSD for fast queries (default: true)."
    ),
    fresh_db: bool = typer.Option(
        False, "--fresh-db", help="Force downloading a fresh copy of metadata.db from Calibre, discarding any existing local staged database."
    ),
):
    """
    Ingest sections, perform semantic chunking, extract concepts via Ollama,
    deduplicate entities, and export the Concept Knowledge Graph.
    Supports persistent checkpointing to resume from interrupted runs or retry failed books.
    """
    cfg = _get_effective_settings(config_path, calibre_path, ollama_url, model, embedding_model)
    output_dir = cfg.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_file = output_dir / "knowledge_graph.json"

    store = ConceptGraphStore()
    if graph_file.is_file():
        try:
            store.load(graph_file)
            console.print(f"[dim]Loaded existing graph with {store.graph.number_of_nodes()} nodes.[/dim]")
        except Exception:
            pass

    client = CalibreClient(
        library_path=cfg.calibre_library_path,
        user=cfg.calibre_user,
        password=cfg.calibre_password,
    )

    # Local DB Staging: Copy metadata.db locally to bypass SMB network roundtrips and latency
    effective_stage_db = cfg.stage_metadata_db if stage_db is None else stage_db
    should_stage = (
        not file_path
        and effective_stage_db
        and not client.is_remote_url
        and client.db_path is not None
        and client.db_path.is_file()
    )

    if should_stage:
        staged_target = cfg.resolved_staged_db_path

        if reset_progress or fresh_db:
            if staged_target.is_file():
                try:
                    staged_target.unlink()
                except Exception:
                    pass

        # Check if an existing staged database already exists locally from a previous session
        if resume and not reset_progress and not fresh_db and staged_target.is_file() and staged_target.stat().st_size > 0:
            try:
                CalibreClient._verify_sqlite_integrity(staged_target, allow_index_warnings=True)
                client.staged_db_path = staged_target
                client.is_staged = True
                console.print(
                    f"[bold cyan]Found existing local staged database at {staged_target}.[/bold cyan] "
                    f"[dim]Reusing local SSD database for instant queries and avoiding re-downloading.[/dim]"
                )
            except Exception as e:
                console.print(
                    f"[dim yellow]Existing local staged database could not be reused ({e}). Re-staging fresh copy from Calibre share...[/dim yellow]"
                )
                client.cleanup_staged(delete_file=True)

        if not client.is_staged:
            try:
                total_bytes = client.db_path.stat().st_size
                size_mb = total_bytes / (1024 * 1024)
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TextColumn("• {task.completed:.1f}/{task.total:.1f} MB"),
                    console=console,
                ) as stage_progress:
                    stage_task = stage_progress.add_task(
                        f"Downloading metadata.db ({size_mb:.1f} MB) locally for fast SSD transactions...",
                        total=size_mb,
                    )

                    def _on_stage_progress(copied_bytes: int, total_b: int):
                        stage_progress.update(stage_task, completed=copied_bytes / (1024 * 1024))

                    staged_file = client.stage_database(
                        staged_target,
                        progress_callback=_on_stage_progress,
                    )

                console.print(f"[dim green]✓ Staged database at {staged_file} (integrity check ok).[/dim green]")
            except Exception as e:
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Failed to stage metadata.db locally ({e}). "
                    f"Falling back to direct database access."
                )
                client.cleanup_staged(delete_file=False)
                should_stage = False

    extractor = KnowledgeExtractor.from_settings(cfg)
    _print_pool_configuration(cfg, console, show_embeddings=True)

    if not skip_warmup:
        _perform_ollama_warmup(extractor, console)

    deduplicator = EntityDeduplicator.from_settings(
        cfg,
        embedding_model=cfg.embedding_model,
        similarity_threshold=cfg.similarity_threshold,
    )

    # Initialize ChunkStore for local persistence and fast reuse of book chunks
    resolved_chunks_path = Path(chunks_dir).expanduser().resolve() if chunks_dir else cfg.resolved_chunks_dir
    chunk_store = ChunkStore(resolved_chunks_path)
    console.print(f"[dim]Persistent Chunks Directory: [bold cyan]{chunk_store.storage_dir}[/bold cyan][/dim]")

    # Initialize HierarchicalChunker with Ollama embeddings using failover pool
    try:
        embeddings = FailoverOllamaEmbeddings(
            pool=extractor.pool,
            model=cfg.embedding_model,
        )
        chunker = HierarchicalChunker(embeddings=embeddings)
    except Exception as e:
        console.print(f"[dim yellow]Warning: Remote Ollama embeddings init skipped ({e}); using paragraph chunker.[/dim yellow]")
        chunker = HierarchicalChunker()

    tracker_path = Path(state_file) if state_file else cfg.resolved_state_file
    tracker = ProgressTracker(tracker_path)

    if reset_progress:
        tracker.clear("build_graph")
        console.print("[dim yellow]Reset progress checkpoint for build_graph.[/dim yellow]")

    books_to_process = []

    # Case 1: Standalone file passed
    if file_path:
        p = Path(file_path).expanduser().resolve()
        if not p.is_file():
            console.print(f"[bold red]File not found:[/bold red] {p}")
            raise typer.Exit(1)
        if BookParser.is_graphical_format(p):
            console.print(f"[yellow]Skipping graphical file '{p.name}': contains image-based content ({p.suffix}).[/yellow]")
            raise typer.Exit(0)
        books_to_process.append({"id": 1, "title": p.stem, "author": "Unknown", "path": p})

    # Case 2: From Calibre / SMB share
    elif client.is_available():
        query_msg = (
            "Reading book catalog from staged local SQLite..."
            if client.is_staged
            else "Querying Calibre library over network/calibredb..."
        )
        console.print(f"[dim]{query_msg}[/dim]")
        with console.status(f"[bold blue]{query_msg}[/bold blue]"):
            try:
                all_calibre_books = client.list_books(fields=["id", "title", "authors", "formats"])
            except Exception as e:
                if should_stage and client.is_staged:
                    client.cleanup_staged(delete_file=True)
                console.print(f"[bold red]Failed to query Calibre library:[/bold red] {e}")
                raise typer.Exit(1)

        if book_id is not None:
            matches = [b for b in all_calibre_books if b["id"] == book_id]
            if not matches:
                console.print(f"[bold red]Book ID {book_id} not found in Calibre library.[/bold red]")
                raise typer.Exit(1)
            target_list = matches
        elif retry_failed:
            failed_ids = tracker.get_failed_ids("build_graph")
            if not failed_ids:
                console.print("[bold green]No failed books found in checkpoint to retry.[/bold green]")
                raise typer.Exit(0)
            target_list = [b for b in all_calibre_books if b["id"] in failed_ids]
            console.print(
                f"[bold cyan]Retrying {len(target_list)} previously failed book(s) (IDs: {sorted(failed_ids)})...[/bold cyan]"
            )
        elif all_books:
            target_list = all_calibre_books
        else:
            console.print(
                "[bold yellow]Please specify --book-id <ID>, --file <path>, --retry-failed, or --all to build graph.[/bold yellow]"
            )
            raise typer.Exit(1)

        # Pre-filter by start_from_id
        if start_from_id is not None:
            target_list = [b for b in target_list if b.get("id", 0) >= start_from_id]

        # Pre-filter by resume checkpoint BEFORE touching network or exporting EPUBs
        if resume and not retry_failed:
            completed_ids = tracker.get_completed_ids("build_graph")
            orig_len = len(target_list)
            target_list = [b for b in target_list if b.get("id") not in completed_ids]
            skipped = orig_len - len(target_list)
            if skipped > 0:
                console.print(
                    f"[dim cyan]Checkpoint Resume: Skipped {skipped} already indexed or skipped book(s). "
                    f"({len(target_list)} remaining. Use --no-resume to reprocess all).[/dim cyan]"
                )

        export_dir = output_dir / "calibre_ingest"
        for b in target_list:
            bid = b["id"]
            formats = b.get("formats", [])
            # Check if book only has graphical formats (CBR, CBZ, DJVU)
            if formats and BookParser.only_has_graphical_formats(formats):
                fmts_label = ", ".join(formats)
                console.print(f"[dim yellow]⚡ Skipping book #{bid}: graphical format ({fmts_label}).[/dim yellow]")
                tracker.mark_skipped("build_graph", bid, title=b.get("title", f"Book {bid}"), reason=f"graphical: {fmts_label}")
                continue

            # If cached chunks exist locally, skip exporting or downloading EPUB from SMB share
            if chunk_store.has_chunks(bid) and not rechunk:
                books_to_process.append(
                    {
                        "id": bid,
                        "title": b.get("title", f"Book {bid}"),
                        "author": ", ".join(b.get("authors", [])) or "Unknown",
                        "path": None,
                    }
                )
            else:
                epub_path = client.export_book(bid, target_dir=export_dir, fmt="EPUB")
                if epub_path and epub_path.is_file():
                    books_to_process.append(
                        {
                            "id": bid,
                            "title": b.get("title", f"Book {bid}"),
                            "author": ", ".join(b.get("authors", [])) or "Unknown",
                            "path": epub_path,
                        }
                    )
                else:
                    console.print(f"[yellow]Skipping book #{bid}: No EPUB format available.[/yellow]")
                    tracker.mark_skipped("build_graph", bid, title=b.get("title", f"Book {bid}"), reason="No EPUB format available")
    else:
        console.print(
            "[bold red]Calibre not available. Pass an explicit book file via --file <path.epub> "
            "or mount the SMB share and pass --calibre-path <path>.[/bold red]"
        )
        raise typer.Exit(1)

    if not books_to_process:
        console.print("[bold yellow]No books available to process.[/bold yellow]")
        raise typer.Exit(0)

    console.print(f"[bold green]Starting pipeline for {len(books_to_process)} book(s)...[/bold green]")

    num_servers = len(extractor.pool.alive_nodes) or len(extractor.pool.nodes)
    pool_concurrency = max_tasks if max_tasks is not None else cfg.calculate_pool_concurrency(num_servers)

    console.print(
        f"[bold cyan]Parallel Task Pool:[/bold cyan] Running up to {pool_concurrency} active chunk task(s) "
        f"across {num_servers} Ollama server(s) (low: {num_servers}, max default: {min(cfg.max_active_tasks_cap, 3 * num_servers)})."
    )

    completed_in_session = 0
    interrupted = False

    stats_lock = threading.Lock()
    display_lock = threading.Lock()
    active_tasks_count = 0
    durations: List[float] = []
    abort_event = threading.Event()

    def _calc_stats() -> Tuple[float, float, float, float]:
        """Return (avg, p80, p18, fastest) in seconds."""
        with stats_lock:
            if not durations:
                return (0.0, 0.0, 0.0, 0.0)
            avg_val = sum(durations) / len(durations)
            fastest_val = min(durations)
            sorted_d = sorted(durations)
            p80_idx = int(0.80 * (len(sorted_d) - 1))
            p80_val = sorted_d[p80_idx]
            p18_idx = int(0.18 * (len(sorted_d) - 1))
            p18_val = sorted_d[p18_idx]
            return (avg_val, p80_val, p18_val, fastest_val)

    try:
        for binfo in books_to_process:
            bid = binfo["id"]
            btitle = binfo["title"]
            bauthor = binfo["author"]
            bpath: Optional[Path] = binfo.get("path")

            console.print(f"\n[bold blue]► Ingesting Book #{bid}: {btitle}[/bold blue]")

            try:
                store.add_book(bid, title=btitle, author=bauthor)

                # Check if chunks already exist in local ChunkStore
                if chunk_store.has_chunks(bid) and not rechunk:
                    chunks = chunk_store.load_chunks(bid)
                    console.print(f"  Loaded [bold green]{len(chunks)} cached atomic chunks[/bold green] from local storage.")
                else:
                    if not bpath or not bpath.is_file():
                        raise FileNotFoundError(f"Book file not available for #{bid}")
                    # Parse sections
                    sections = BookParser.parse(bpath)
                    console.print(f"  Extracted [green]{len(sections)} sections/chapters[/green].")

                    # Chunk sections
                    chunks = chunker.chunk_book(sections, book_id=bid, book_title=btitle)
                    console.print(f"  Created [green]{len(chunks)} atomic thematic chunks[/green].")

                    # Persist chunks to local storage immediately
                    chunk_store.save_chunks(bid, btitle, chunks)
                    console.print(f"  [dim]Saved chunks to local storage: {chunk_store._chunk_file(bid).name}[/dim]")

                def _process_single_chunk(chk_item: HierarchicalChunk) -> Dict[str, Any]:
                    nonlocal active_tasks_count
                    if abort_event.is_set():
                        return {"status": "aborted", "chunk": chk_item}

                    with stats_lock:
                        active_tasks_count += 1
                    t0 = time.time()
                    dur = 0.0
                    srv = "ollama"
                    try:
                        if abort_event.is_set():
                            return {"status": "aborted", "chunk": chk_item}

                        extraction = extractor.extract_section(
                            text=chk_item.text,
                            book_title=btitle,
                            section_title=chk_item.section_title,
                            subtitle=chk_item.subtitle,
                            parent_context=chk_item.parent_text,
                        )
                        dur = time.time() - t0
                        srv = extractor.pool.get_last_used_server() or "ollama"
                        with stats_lock:
                            durations.append(dur)

                        return {
                            "status": "extracted",
                            "chunk": chk_item,
                            "extraction": extraction,
                            "duration": dur,
                            "server_used": srv,
                        }
                    finally:
                        with stats_lock:
                            active_tasks_count = max(0, active_tasks_count - 1)

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Extracting '{btitle[:22]}'...", total=len(chunks))

                    def _update_progress_description():
                        avg_s, p80_s, p18_s, fastest_s = _calc_stats()
                        with stats_lock:
                            cur_active = active_tasks_count
                        desc = (
                            f"Extracting '{btitle[:18]}' across {num_servers} server(s) | "
                            f"Active: [bold cyan]{cur_active}[/bold cyan]"
                        )
                        if durations:
                            desc += (
                                f" | avg: [bold green]{avg_s:.1f}s[/bold green] | "
                                f"p80: [bold yellow]{p80_s:.1f}s[/bold yellow] | "
                                f"p18: [bold blue]{p18_s:.1f}s[/bold blue] | "
                                f"fastest: [bold magenta]{fastest_s:.1f}s[/bold magenta]"
                            )
                        progress.update(task, description=desc)

                    _update_progress_description()

                    chunk_executor = DaemonThreadPoolExecutor(max_workers=pool_concurrency)
                    try:
                        future_to_chunk = {
                            chunk_executor.submit(_process_single_chunk, chk): chk for chk in chunks
                        }

                        for future in as_completed(future_to_chunk):
                            if abort_event.is_set():
                                break

                            res = future.result()
                            if res.get("status") == "aborted":
                                continue

                            chk = res["chunk"]
                            extraction = res["extraction"]
                            dur = res["duration"]
                            srv = res["server_used"]

                            # Ensure section exists in graph
                            sec_node_id = store.add_section(
                                book_id=bid,
                                chapter_idx=chk.chapter_idx,
                                title=chk.section_title,
                                text=chk.text,
                            )

                            # Register chunk node in graph
                            store.add_chunk(chk)

                            # Deduplicate and register ideas/concepts
                            for concept in extraction.concepts:
                                canonical_concept = deduplicator.resolve_concept(concept)
                                store.add_concept(canonical_concept)

                                # Provenance link: (:Idea) -[:SUPPORTED_BY]-> (:Chunk)
                                store.add_idea_support_link(
                                    concept_name=canonical_concept.name,
                                    chunk=chk,
                                    quote=canonical_concept.supporting_quote or "",
                                    brief_description=canonical_concept.brief_description,
                                    detailed_explanation=canonical_concept.detailed_explanation,
                                )

                                # Link Section -> DISCUSSES -> Concept
                                store.add_section_concept_link(
                                    section_node_id=sec_node_id,
                                    concept_name=canonical_concept.name,
                                    summary=canonical_concept.summary,
                                    quote=canonical_concept.supporting_quote or "",
                                )

                                # Link related concepts
                                for rel_name in canonical_concept.related_concepts:
                                    store.add_concept_relation(
                                        src_concept_name=canonical_concept.name,
                                        tgt_concept_name=rel_name,
                                    )

                            with display_lock:
                                console.print(
                                    f"  [dim green]✓ [{chk.section_title[:24]} p{chk.chunk_idx}][/dim green] "
                                    f"Extracted [bold cyan]{len(extraction.concepts)} concept(s)[/bold cyan] "
                                    f"[dim]({dur:.1f}s via [bold green]{srv}[/bold green])[/dim]"
                                )

                            progress.advance(task)
                            _update_progress_description()

                    except KeyboardInterrupt:
                        abort_event.set()
                        progress.stop()
                        for f in future_to_chunk:
                            f.cancel()
                        chunk_executor.shutdown(wait=False, cancel_futures=True)
                        import concurrent.futures.thread
                        with concurrent.futures.thread._global_shutdown_lock:
                            for t in list(chunk_executor._threads):
                                concurrent.futures.thread._threads_queues.pop(t, None)
                        raise
                    finally:
                        chunk_executor.shutdown(wait=False, cancel_futures=True)

                tracker.mark_completed("build_graph", bid, btitle)
                # Incremental persistence: save graph checkpoint after every processed book
                store.save(graph_file)
                completed_in_session += 1
                console.print(f"  [dim green]✓ Book #{bid} concepts integrated and graph checkpoint saved.[/dim green]")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                console.print(f"[bold red]✗ Failed to build graph for book #{bid} ('{btitle}'):[/bold red] {e}")
                tracker.mark_failed("build_graph", bid, title=btitle, error=str(e))

    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[bold yellow]Cancelled by user. Saving current Knowledge Graph state...[/bold yellow]")

    finally:
        vault_dest = Path(export_obsidian) if export_obsidian else output_dir / "obsidian_vault"
        graphml_dest = output_dir / "knowledge_graph.graphml"

        if store.graph.number_of_nodes() > 0:
            store.save(graph_file)
            obs = ObsidianExporter(vault_dest)
            obs.export(store)
            gml = GraphMLExporter(graphml_dest)
            gml.export(store)

        if durations:
            avg_s, p80_s, p18_s, fastest_s = _calc_stats()
            console.print(
                f"\n[bold cyan]Concept Extraction Performance:[/bold cyan] "
                f"Avg: [bold green]{avg_s:.2f}s[/bold green] | "
                f"80th Percentile (p80): [bold yellow]{p80_s:.2f}s[/bold yellow] | "
                f"18th Percentile (p18): [bold blue]{p18_s:.2f}s[/bold blue] | "
                f"Fastest: [bold magenta]{fastest_s:.2f}s[/bold magenta] "
                f"[dim](across {len(durations)} extracted chunk tasks)[/dim]"
            )

        if interrupted:
            console.print(
                f"[bold yellow]Graph build stopped by user.[/bold yellow] "
                f"Completed in this session: [bold green]{completed_in_session}[/bold green] book(s). "
                f"Knowledge graph checkpoint and Obsidian vault preserved."
            )
            raise typer.Exit(code=130)

    # Persist graph to JSON and exports
    console.print(f"\n[bold green]✓ Knowledge Graph persisted to:[/bold green] {graph_file}")
    console.print(f"[bold green]✓ Obsidian Markdown Vault exported to:[/bold green] {vault_dest}")
    console.print(f"[bold green]✓ GraphML exported to:[/bold green] {graphml_dest}")

    # Display final graph statistics
    st = store.stats()
    console.print(
        Panel.fit(
            f"[bold]Total Nodes:[/bold] {st['total_nodes']}\n"
            f"[bold]Total Edges:[/bold] {st['total_edges']}\n\n"
            f"[bold green]Node Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["node_types"].items())
            + "\n\n[bold green]Relationship Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["edge_types"].items()),
            title="Knowledge Graph Build Complete",
        )
    )


@app.command("status")
def show_status(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml."),
    state_file: Optional[str] = typer.Option(None, "--state-file", help="Custom path for progress checkpoint JSON file."),
):
    """Display progress and checkpoint summary for batch operations."""
    cfg = get_settings(config_path)
    tracker_path = Path(state_file) if state_file else cfg.resolved_state_file
    tracker = ProgressTracker(tracker_path)
    clean_summary = tracker.summary("clean_metadata")
    graph_summary = tracker.summary("build_graph")

    table = Table(title=f"Bookeeper Checkpoint State ({tracker_path.name})", show_header=True)
    table.add_column("Operation", style="bold cyan")
    table.add_column("Completed", style="bold green", justify="right")
    table.add_column("Skipped (Graphical)", style="yellow", justify="right")
    table.add_column("Failed", style="bold red", justify="right")
    table.add_column("Total Recorded", style="white", justify="right")

    table.add_row(
        "clean_metadata",
        str(clean_summary["completed"]),
        str(clean_summary.get("skipped", 0)),
        str(clean_summary["failed"]),
        str(clean_summary["total_recorded"]),
    )
    table.add_row(
        "build_graph",
        str(graph_summary["completed"]),
        str(graph_summary.get("skipped", 0)),
        str(graph_summary["failed"]),
        str(graph_summary["total_recorded"]),
    )
    console.print(table)
    console.print(f"[dim]Full state location: {tracker_path}[/dim]")


@app.command()
def stats(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml.")
):
    """Display metrics for the current Knowledge Graph."""
    cfg = get_settings(config_path)
    store = ConceptGraphStore()
    graph_file = cfg.resolved_output_dir / "knowledge_graph.json"
    if not graph_file.is_file():
        console.print(f"[yellow]No graph found at {graph_file}[/yellow]")
        return

    store.load(graph_file)
    st = store.stats()
    console.print(
        Panel.fit(
            f"[bold]Total Nodes:[/bold] {st['total_nodes']}\n"
            f"[bold]Total Edges:[/bold] {st['total_edges']}\n\n"
            f"[bold green]Node Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["node_types"].items())
            + "\n\n[bold green]Relationship Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["edge_types"].items()),
            title="Knowledge Graph Metrics",
        )
    )


if __name__ == "__main__":
    app()
