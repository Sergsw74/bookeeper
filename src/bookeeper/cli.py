"""
Production-quality Typer CLI interface for bookeeper with rich progress bars and configurable Ollama / Calibre paths.
"""

import json
import logging
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import typer
from langchain_ollama import OllamaEmbeddings
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)

from bookeeper.calibre.client import CalibreClient
from bookeeper.calibre.parser import BookParser
from bookeeper.config import Settings, get_settings
from bookeeper.graph.exporters import GraphMLExporter, Neo4jExporter, ObsidianExporter
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import ChunkStore, HierarchicalChunk, HierarchicalChunker
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import (
    Concept,
    KnowledgeExtractor,
    SectionExtraction,
    validate_concept_chunk_grounding,
)
from bookeeper.processing.ollama_pool import (
    FailoverOllamaEmbeddings,
    OllamaPool,
    aggregate_server_statistics,
    format_server_stats_table,
)
from bookeeper.processing.state import BookProcessingState, ProgressTracker
from bookeeper.processing.verifier import (
    IdeaVerifier,
    VerificationItem,
    VerificationReport,
    verify_graph,
    verify_chunking,
    verify_chunks,
    ChunkingVerificationReport,
    ChunkVerificationReport,
    ChunkAuditItem,
    run_cross_check,
    CrossCheckReport,
    CrossCheckBlockResult,
)
from bookeeper.rag.lightrag_engine import LightRAGEngine

app = typer.Typer(
    name="bookeeper",
    help="Connects Calibre libraries to local/remote Ollama LLMs and extracts Concept Knowledge Graphs.",
    add_completion=False,
)
console = Console()
logger = logging.getLogger(__name__)


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


def format_eta_min_sec(seconds: Optional[float]) -> str:
    """Format remaining time as (min, sec), e.g. '2m 15s' or '0m 45s'."""
    if seconds is None:
        return "--m --s"
    if seconds <= 0:
        return "0m 00s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def format_eta_h_min_sec(seconds: Optional[float]) -> str:
    """Format remaining time as (h, min, sec), e.g. '1h 12m 30s' or '0h 05m 12s'."""
    if seconds is None:
        return "--h --m --s"
    if seconds <= 0:
        return "0h 00m 00s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def format_eta_days_h_m(seconds: Optional[float]) -> str:
    """Format remaining time as (days, h, m), e.g. '2d 5h 42m' or '0d 14h 05m'."""
    if seconds is None:
        return "--d --h --m"
    if seconds <= 0:
        return "0d 0h 00m"
    d, rem = divmod(int(seconds), 86400)
    h, rem2 = divmod(rem, 3600)
    m = int(rem2 // 60)
    return f"{d}d {h}h {m:02d}m"


class ChunkRemainingColumn(ProgressColumn):
    """Renders estimated chunking time remaining formatted as (min, sec)."""

    def render(self, task: "Task") -> Text:
        if task.finished:
            return Text("ETA: 0m 00s", style="dim")
        eta_str = task.fields.get("eta_str")
        if not eta_str:
            rem = task.time_remaining
            eta_str = format_eta_min_sec(rem)
        style = "dim" if eta_str.startswith("--") else "bold yellow"
        return Text(f"ETA: {eta_str}", style=style)


class IngestionRemainingColumn(ProgressColumn):
    """Renders custom ETA based on task type:
    - 'book': ETA in (h, min, sec)
    - 'overall': ETA in (days, h, m)
    """

    def render(self, task: "Task") -> Text:
        eta_type = task.fields.get("eta_type", "book")
        if task.finished:
            zero_str = "0h 00m 00s" if eta_type == "book" else "0d 0h 00m"
            return Text(f"ETA: {zero_str}", style="dim")

        eta_str = task.fields.get("eta_str")
        if not eta_str:
            rem = task.time_remaining
            if eta_type == "overall":
                eta_str = format_eta_days_h_m(rem)
            else:
                eta_str = format_eta_h_min_sec(rem)

        style = "dim" if eta_str.startswith("--") else "bold yellow"
        return Text(f"ETA: {eta_str}", style=style)


def _resolve_opt(val: Any, default: Any = None) -> Any:
    """Unwrap Typer OptionInfo when commands are called directly within Python."""
    if hasattr(val, "default"):
        return val.default
    return val if val is not None else default


def _get_effective_settings(
    config_path: Optional[str] = None,
    calibre_path: Optional[str] = None,
    ollama_url: Optional[str] = None,
    model: Optional[str] = None,
    model_fallback: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_url: Optional[str] = None,
    request_timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    max_chunk_attempts: Optional[int] = None,
) -> Settings:
    """Retrieve settings and merge explicit CLI arguments with highest precedence."""
    config_path = _resolve_opt(config_path)
    calibre_path = _resolve_opt(calibre_path)
    ollama_url = _resolve_opt(ollama_url)
    model = _resolve_opt(model)
    model_fallback = _resolve_opt(model_fallback)
    embedding_model = _resolve_opt(embedding_model)
    embedding_url = _resolve_opt(embedding_url)
    request_timeout = _resolve_opt(request_timeout)
    max_retries = _resolve_opt(max_retries)
    max_chunk_attempts = _resolve_opt(max_chunk_attempts)

    cfg = get_settings(config_path)
    overrides = {}
    if calibre_path:
        overrides["calibre_library_path"] = calibre_path
    if ollama_url:
        overrides["ollama_base_url"] = ollama_url
    if model:
        overrides["llm_model"] = model
    if model_fallback:
        overrides["llm_model_fallback"] = model_fallback
    if embedding_model:
        overrides["embedding_model"] = embedding_model
    if embedding_url:
        if embedding_url.strip().lower() == "pool":
            overrides["embedding_base_url"] = "pool"
            overrides["embedding_servers"] = []
        else:
            overrides["embedding_base_url"] = embedding_url
            overrides["embedding_servers"] = [embedding_url]
    if request_timeout is not None:
        overrides["request_timeout"] = request_timeout
    if max_retries is not None:
        overrides["max_retries"] = max_retries
    if max_chunk_attempts is not None:
        overrides["max_chunk_attempts"] = max_chunk_attempts

    if overrides:
        data = cfg.model_dump()
        data.update(overrides)
        return Settings(**data)
    return cfg


def _print_pool_configuration(cfg: Settings, console: Console, show_embeddings: bool = False) -> None:
    """Print clean summary of configured Ollama pool nodes and models."""
    servers = cfg.resolved_ollama_servers
    if cfg.uses_llm_pool_for_embeddings:
        emb_loc = " [dim](shares LLM pool)[/dim]"
    else:
        emb_loc = f" @ {cfg.embedding_base_url}"
    fallback_str = f" [dim](fallback: [bold cyan]{cfg.llm_model_fallback}[/bold cyan])[/dim]" if cfg.llm_model_fallback else ""
    emb_suffix = f" | Embeddings: [bold cyan]{cfg.embedding_model}[/bold cyan]{emb_loc}" if show_embeddings else ""
    req_meta = f" | Timeout: [bold cyan]{cfg.request_timeout}s[/bold cyan] | Retries: [bold cyan]{cfg.max_retries}[/bold cyan]"
    if servers and len(servers) > 1:
        nodes_str = ", ".join(f"{s.name or s.url} ({s.url}) [{s.capability_str}]" for s in servers)
        console.print(
            f"[dim]Configured Ollama Multi-Server Pool ({len(servers)} nodes): [bold cyan]{nodes_str}[/bold cyan] | "
            f"LLM: [bold cyan]{cfg.llm_model}[/bold cyan]{fallback_str}{emb_suffix}{req_meta}[/dim]"
        )
    else:
        single_cap = f" [{servers[0].capability_str}]" if servers else ""
        console.print(
            f"[dim]Configured Ollama: [bold cyan]{cfg.ollama_base_url}[/bold cyan]{single_cap} | "
            f"LLM: [bold cyan]{cfg.llm_model}[/bold cyan]{fallback_str}{emb_suffix}{req_meta}[/dim]"
        )


def _perform_ollama_warmup(extractor: KnowledgeExtractor, console: Console) -> Dict[str, Any]:
    """Execute warmup ping to load model and report CPU vs GPU acceleration status across server pool."""
    pool_nodes = extractor.pool.nodes
    node_cap_map = {n.url.rstrip("/"): ", ".join(n.capabilities) for n in pool_nodes}
    node_mac_map = {n.url.rstrip("/"): n.mac for n in pool_nodes}

    def _status_cb(msg: str):
        console.print(f"[bold yellow]⚡ {msg}[/bold yellow]")

    with console.status(
        f"[bold blue]Checking Ollama acceleration across server pool ({len(pool_nodes)} node(s))...[/bold blue]"
    ):
        status = extractor.warmup_and_check_device(on_status_callback=_status_cb)

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
        table.add_column("Capabilities", style="yellow")
        table.add_column("WOL MAC", style="dim cyan")
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
            caps = node_cap_map.get(url.rstrip("/"), "llm, embedding, verification")
            wol_mac = node_mac_map.get(url.rstrip("/")) or "[dim]-[/dim]"
            table.add_row(pri, url, caps, wol_mac, dev, vram_str, pool_state)

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


def _print_ollama_server_stats(
    pools: Sequence[OllamaPool],
    console: Console,
    title: str = "Ollama Server Workload & Effort Statistics",
) -> None:
    """Print clean summary table of Ollama server efforts (chars processed, tasks, avg/total time)."""
    valid_pools = [p for p in pools if p is not None]
    if not valid_pools:
        return
    stats = aggregate_server_statistics(valid_pools)
    if not stats:
        return
    table = format_server_stats_table(stats, title=title)
    console.print()
    console.print(table)


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
    embedding_url: Optional[str] = typer.Option(
        None, "--embedding-url", help="Override Ollama embedding endpoint (e.g. http://localhost:11434 for local embeddings)."
    ),
):
    """Display the active bookeeper configuration settings."""
    cfg = _get_effective_settings(config_path, calibre_path, ollama_url, model, embedding_model, embedding_url)
    servers_desc = []
    for s in cfg.resolved_ollama_servers:
        name_tag = f" ({s.name})" if s.name else ""
        cap_tag = f" [{s.capability_str}]"
        mac_tag = f" [yellow](WOL MAC: {s.mac}{' + secret' if s.wol_secret else ''})[/yellow]" if s.mac else ""
        servers_desc.append(f"    • [cyan]{s.url}[/cyan]{name_tag}{cap_tag}{mac_tag} [dim](priority: {s.priority})[/dim]")
    servers_block = "\n".join(servers_desc)

    emb_endpoint_str = "[dim]shares LLM pool[/dim]" if cfg.uses_llm_pool_for_embeddings else f"[cyan]{cfg.embedding_base_url}[/cyan] [bold green](dedicated local)[/bold green]"

    wol_status_str = f"[bold green]Enabled[/bold green] (timeout: {cfg.wol_wait_seconds}s, interval: {cfg.wol_probe_interval}s)" if cfg.wol_enabled else "[dim]Disabled[/dim]"

    console.print(
        Panel.fit(
            f"[bold green]Calibre Library / SMB Share:[/bold green] {cfg.calibre_library_path}\n"
            f"[bold green]Calibre Auth:[/bold green] user={cfg.calibre_user or '[dim]none[/dim]'}\n"
            f"[bold green]Ollama Failover Pool ({len(cfg.resolved_ollama_servers)} server(s)):[/bold green]\n{servers_block}\n"
            f"[bold green]Wake-on-LAN (WOL):[/bold green] {wol_status_str}\n"
            f"[bold green]Failover Cooldown:[/bold green] {cfg.failover_cooldown_seconds}s\n"
            f"[bold green]LLM Model:[/bold green] {cfg.llm_model}\n"
            f"[bold green]Embedding Model:[/bold green] {cfg.embedding_model}\n"
            f"[bold green]Embedding Endpoint:[/bold green] {emb_endpoint_str}\n"
            f"[bold green]Similarity Thresholds:[/bold green] {cfg.similarity_threshold} (LLM check) / {getattr(cfg, 'high_similarity_threshold', 0.95)} (Direct merge)\n"
            f"[bold green]Output Directory:[/bold green] {cfg.resolved_output_dir}\n"
            f"[bold green]LightRAG Engine:[/bold green] {'[bold green]Enabled[/bold green]' if cfg.enable_lightrag else '[dim]Disabled[/dim]'} (dir: {cfg.resolved_lightrag_dir}, mode: [cyan]{cfg.lightrag_mode}[/cyan])\n"
            f"[bold green]Neo4j Export:[/bold green] {'[bold green]Enabled[/bold green]' if cfg.neo4j.enabled else '[dim]Disabled[/dim]'} (uri: [cyan]{cfg.neo4j.uri}[/cyan], db: {cfg.neo4j.database}, user: {cfg.neo4j.user})",
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
    model_fallback: Optional[str] = typer.Option(
        None, "--model-fallback", help="Ollama LLM fallback model (e.g. llama3.1:8b)."
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
        None, "--max-tasks", "-t", help="Max concurrent active tasks across Ollama servers (default: auto: 1x alive servers, sequential per server)."
    ),
    fresh_db: bool = typer.Option(
        False, "--fresh-db", help="Force downloading a fresh copy of metadata.db from Calibre, discarding any existing local staged database."
    ),
    request_timeout: Optional[int] = typer.Option(
        None, "--request-timeout", "--timeout", help="HTTP timeout in seconds for Ollama LLM requests (default: 60s)."
    ),
    retries: Optional[int] = typer.Option(
        None, "--retries", help="Maximum retries per attempt on an Ollama server before failover (default: 1)."
    ),
):
    """
    Query Calibre (local, SMB share, or server), inspect titles/authors/summaries,
    prompt Ollama to normalize, and write clean metadata back.
    Supports persistent checkpointing to resume from interrupted points or retry only failed books.
    """
    cfg = _get_effective_settings(
        config_path,
        calibre_path,
        ollama_url,
        model,
        model_fallback=model_fallback,
        request_timeout=request_timeout,
        max_retries=retries,
    )
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
        f"across {num_servers} Ollama server(s) (sequential 1 task per server)..."
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

        _print_ollama_server_stats([extractor.pool], console=console, title="Ollama Server Metadata Cleaning Effort")

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
    model_fallback: Optional[str] = typer.Option(
        None, "--model-fallback", help="Ollama LLM fallback model (used when a chunk fails >= 50% of retry attempts)."
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", help="Ollama embedding model name (e.g. nomic-embed-text)."
    ),
    embedding_url: Optional[str] = typer.Option(
        None, "--embedding-url", help="Dedicated Ollama embedding endpoint (e.g. http://localhost:11434 for local embeddings)."
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
    retry_skipped: bool = typer.Option(
        False, "--retry-skipped", help="Re-check and retry books previously recorded as skipped in checkpoint state."
    ),
    reset_progress: bool = typer.Option(
        False, "--reset-progress", help="Reset saved progress checkpoint and start from scratch."
    ),
    from_scratch: bool = typer.Option(
        False,
        "--from-scratch",
        help="Start build entirely from scratch: wipes progress checkpoint state, deletes all per-book states (.book_states), purges cached chunks, deletes existing knowledge graph files, and starts indexing from the first book.",
    ),
    clean_chunks: bool = typer.Option(
        False,
        "--clean-chunks",
        help="Purge cached book chunks in storage directory and force fresh re-chunking.",
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
        None, "--max-tasks", "-t", help="Max concurrent active tasks across Ollama servers (default: auto: 1x alive servers, sequential per server)."
    ),
    stage_db: Optional[bool] = typer.Option(
        None, "--stage-db/--no-stage-db", help="Stage metadata.db locally on SSD for fast queries (default: true)."
    ),
    fresh_db: bool = typer.Option(
        False, "--fresh-db", help="Force downloading a fresh copy of metadata.db from Calibre, discarding any existing local staged database."
    ),
    enable_lightrag: Optional[bool] = typer.Option(
        None, "--lightrag/--no-lightrag", help="Also index chunks into LightRAG engine for fast dual-level retrieval."
    ),
    lightrag_dir: Optional[str] = typer.Option(
        None, "--lightrag-dir", help="Working directory for LightRAG graph and vector indices (default: output_dir/lightrag)."
    ),
    export_neo4j: Optional[bool] = typer.Option(
        None, "--export-neo4j/--no-export-neo4j", help="Upsert knowledge graph directly into Neo4j (default: config.yaml neo4j.enabled)."
    ),
    clean_export: Optional[bool] = typer.Option(
        None, "--clean-export/--no-clean-export", "--clean/--no-clean", help="Perform clean start on export targets (delete existing Obsidian vault notes and/or wipe Neo4j database before export)."
    ),
    continue_run: bool = typer.Option(
        False, "--continue", help="Continue processing from the next unprocessed book in Calibre (relative to books already in knowledge graph)."
    ),
    request_timeout: Optional[int] = typer.Option(
        None, "--request-timeout", "--timeout", help="HTTP timeout in seconds for Ollama LLM requests (default: 60s)."
    ),
    retries: Optional[int] = typer.Option(
        None, "--retries", help="Maximum retries per attempt on an Ollama server before failover/skipping (default: 1)."
    ),
    max_chunk_attempts: Optional[int] = typer.Option(
        None, "--max-chunk-attempts", help="Maximum total retry attempts across pool for a stalled chunk before marking failed (default: 6)."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Stop build-graph after successfully processing N books."
    ),
):
    """
    Ingest sections, perform semantic chunking, extract concepts via Ollama,
    deduplicate entities, and export the Concept Knowledge Graph.
    Supports persistent checkpointing to resume from interrupted runs or retry failed books.
    """
    book_id = _resolve_opt(book_id)
    file_path = _resolve_opt(file_path)
    all_books = _resolve_opt(all_books, False)
    export_obsidian = _resolve_opt(export_obsidian)
    resume = _resolve_opt(resume, True)
    retry_failed = _resolve_opt(retry_failed, False)
    retry_skipped = _resolve_opt(retry_skipped, False)
    reset_progress = _resolve_opt(reset_progress, False)
    from_scratch = _resolve_opt(from_scratch, False)
    clean_chunks = _resolve_opt(clean_chunks, False)
    start_from_id = _resolve_opt(start_from_id)
    state_file = _resolve_opt(state_file)
    skip_warmup = _resolve_opt(skip_warmup, False)
    chunks_dir = _resolve_opt(chunks_dir)
    rechunk = _resolve_opt(rechunk, False)
    max_tasks = _resolve_opt(max_tasks)
    stage_db = _resolve_opt(stage_db)
    fresh_db = _resolve_opt(fresh_db, False)
    enable_lightrag = _resolve_opt(enable_lightrag)
    lightrag_dir = _resolve_opt(lightrag_dir)
    export_neo4j = _resolve_opt(export_neo4j)
    clean_export = _resolve_opt(clean_export)
    continue_run = _resolve_opt(continue_run, False)
    limit = _resolve_opt(limit)

    cfg = _get_effective_settings(
        config_path,
        calibre_path,
        ollama_url,
        model,
        model_fallback=model_fallback,
        embedding_model=embedding_model,
        embedding_url=embedding_url,
        request_timeout=request_timeout,
        max_retries=retries,
        max_chunk_attempts=max_chunk_attempts,
    )
    output_dir = cfg.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_file = output_dir / "knowledge_graph.json"
    vault_dest = Path(export_obsidian) if export_obsidian else output_dir / "obsidian_vault"
    graphml_dest = output_dir / "knowledge_graph.graphml"
    should_clean = cfg.clean_export if clean_export is None else clean_export
    should_export_neo4j = cfg.neo4j.enabled if export_neo4j is None else export_neo4j
    has_cleaned = False

    tracker_path = Path(state_file) if state_file else cfg.resolved_state_file
    tracker = ProgressTracker(tracker_path)

    if from_scratch:
        tracker.clear("build_graph", preserve_skipped=not retry_skipped)
        preserved_cnt = len(tracker.get_skipped_ids("build_graph")) if not retry_skipped else 0
        preserved_str = f" (preserving {preserved_cnt} skipped books in skip list)" if preserved_cnt > 0 else ""
        console.print(
            f"[bold yellow]Restarting build entirely from scratch: clearing progress tracker checkpoint, resetting knowledge graph, deleting all book states and cached chunks{preserved_str}...[/bold yellow]"
        )
        reset_progress = True
        resume = False
        continue_run = False
        rechunk = True
        if clean_export is None:
            should_clean = True
        if graph_file.is_file():
            try:
                graph_file.unlink()
            except Exception:
                pass
        if graphml_dest.is_file():
            try:
                graphml_dest.unlink()
            except Exception:
                pass

        # 1. Mandatory cleanup: delete all per-book state checkpoints (.book_states)
        candidate_state_dirs = {
            output_dir / ".book_states",
            getattr(cfg, "resolved_output_dir", output_dir) / ".book_states",
        }
        total_purged_states = 0
        for b_sdir in candidate_state_dirs:
            total_purged_states += BookProcessingState.clean_all(b_sdir)
        if total_purged_states > 0:
            console.print(
                f"[bold yellow]Deleted {total_purged_states} per-book state checkpoint file(s) from .book_states.[/bold yellow]"
            )

        # 2. Chunks cleanup: delete all persistent cached chunks
        target_chunks_path = Path(chunks_dir).expanduser().resolve() if chunks_dir else cfg.resolved_chunks_dir
        cs = ChunkStore(target_chunks_path)
        purged_chunks = cs.clean_all()
        if purged_chunks > 0:
            console.print(
                f"[bold yellow]Deleted {purged_chunks} cached chunk file(s) from {target_chunks_path}.[/bold yellow]"
            )

    store = ConceptGraphStore()
    if not from_scratch and graph_file.is_file():
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
        high_similarity_threshold=getattr(cfg, "high_similarity_threshold", 0.95),
        extractor=extractor,
    )

    # Initialize ChunkStore for local persistence and fast reuse of book chunks
    resolved_chunks_path = Path(chunks_dir).expanduser().resolve() if chunks_dir else cfg.resolved_chunks_dir
    chunk_store = ChunkStore(resolved_chunks_path)
    if clean_chunks:
        purged_count = chunk_store.clean_all()
        console.print(f"[dim yellow]Purged {purged_count} cached chunk file(s) from {chunk_store.storage_dir}.[/dim yellow]")
        rechunk = True
    console.print(f"[dim]Persistent Chunks Directory: [bold cyan]{chunk_store.storage_dir}[/bold cyan][/dim]")

    # Initialize HierarchicalChunker with Ollama embeddings using failover pool
    try:
        embeddings = FailoverOllamaEmbeddings.from_settings(cfg)
        chunker = HierarchicalChunker(embeddings=embeddings)
    except Exception as e:
        console.print(f"[dim yellow]Warning: Embeddings init skipped ({e}); using paragraph chunker.[/dim yellow]")
        chunker = HierarchicalChunker()

    effective_lightrag = cfg.enable_lightrag if enable_lightrag is None else enable_lightrag
    lightrag_engine: Optional[LightRAGEngine] = None
    if effective_lightrag:
        try:
            target_lrag_dir = lightrag_dir or cfg.resolved_lightrag_dir
            lightrag_engine = LightRAGEngine.from_settings(cfg, pool=extractor.pool, working_dir=target_lrag_dir)
            if from_scratch:
                lightrag_engine.clean()
            lightrag_engine.initialize()
            console.print(f"[dim]LightRAG Engine active at: [bold cyan]{lightrag_engine.working_dir}[/bold cyan][/dim]")
        except Exception as e:
            console.print(f"[bold yellow]Warning: Could not initialize LightRAG ({e}). Proceeding without LightRAG.[/bold yellow]")
            lightrag_engine = None

    if reset_progress and not from_scratch:
        tracker.clear("build_graph", preserve_skipped=True)
        console.print("[dim yellow]Reset progress checkpoint for build_graph (preserving skipped books in skip list).[/dim yellow]")

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
        elif retry_skipped:
            skipped_ids = tracker.get_skipped_ids("build_graph")
            if not skipped_ids:
                console.print("[bold green]No skipped books found in checkpoint to retry.[/bold green]")
                raise typer.Exit(0)
            target_list = [b for b in all_calibre_books if b["id"] in skipped_ids]
            sample_ids = sorted(skipped_ids)[:10]
            sample_str = f" (IDs: {sample_ids}...)" if len(skipped_ids) > 10 else f" (IDs: {sample_ids})"
            console.print(
                f"[bold cyan]Retrying {len(target_list)} previously skipped book(s){sample_str}...[/bold cyan]"
            )
        elif all_books or continue_run or from_scratch or (book_id is None and not file_path and not retry_failed and not retry_skipped):
            target_list = all_calibre_books
            if from_scratch:
                console.print("[bold cyan]Starting build entirely from scratch across Calibre catalog...[/bold cyan]")
            elif continue_run or not all_books:
                console.print("[dim cyan]Continuing build from next book (relative to knowledge graph)...[/dim cyan]")
        else:
            console.print(
                "[bold yellow]Please specify --book-id <ID>, --file <path>, --retry-failed, --continue, --from-scratch, or --all to build graph.[/bold yellow]"
            )
            raise typer.Exit(1)

        # Pre-filter by start_from_id
        if start_from_id is not None:
            target_list = [b for b in target_list if b.get("id", 0) >= start_from_id]

        # Pre-filter by resume checkpoint and existing knowledge_graph BEFORE touching network or exporting EPUBs
        if resume and not retry_failed and not retry_skipped and not from_scratch:
            tracker_completed = tracker.get_completed_ids("build_graph")
            tracker_skipped = tracker.get_skipped_ids("build_graph")
            graph_books = {
                attrs["book_id"]: attrs.get("title", f"Book {attrs['book_id']}")
                for _, attrs in store.graph.nodes(data=True)
                if attrs.get("type") == "Book" and "book_id" in attrs
            }
            graph_book_ids = set(graph_books.keys())

            # Detect unprocessed / partially ingested books in knowledge graph from previous interrupted runs!
            # If tracker has recorded completions, any book in the graph not marked completed was interrupted mid-ingest.
            if tracker_completed and graph_book_ids:
                incomplete_graph_books = graph_book_ids - tracker_completed
                if incomplete_graph_books:
                    console.print(
                        f"[bold yellow]Notice: Detected {len(incomplete_graph_books)} unprocessed/partially ingested book(s) "
                        f"in knowledge graph from previous interrupted runs. Purging them so only fully processed books remain...[/bold yellow]"
                    )
                    for inc_bid in sorted(incomplete_graph_books):
                        inc_title = graph_books.get(inc_bid, f"Book {inc_bid}")
                        stats = store.remove_book(inc_bid)
                        console.print(
                            f"  [dim yellow]✓ Purged unprocessed book #{inc_bid} ('{inc_title}'): "
                            f"{stats['chunks_removed']} chunks, {stats['orphan_concepts_removed']} orphan concepts removed.[/dim yellow]"
                        )
                    store.save(graph_file)
                    # Refresh graph_book_ids after purging incomplete books
                    graph_book_ids = {
                        attrs["book_id"]
                        for _, attrs in store.graph.nodes(data=True)
                        if attrs.get("type") == "Book" and "book_id" in attrs
                    }
                completed_ids = tracker_completed & graph_book_ids
            elif graph_book_ids:
                # If tracker is empty but graph has books (e.g. external/seeded graph), sync existing graph books into tracker
                completed_ids = set(graph_book_ids)
                for g_bid in graph_book_ids:
                    tracker.mark_completed("build_graph", g_bid, graph_books.get(g_bid, f"Book {g_bid}"))
            else:
                completed_ids = set()

            # Check if knowledge graph was manually wiped / deleted externally while checkpoint remains
            pure_completed = tracker_completed - tracker_skipped
            if (not graph_file.is_file() or len(graph_book_ids) == 0) and pure_completed:
                console.print(
                    f"[bold yellow]Notice: Knowledge graph is empty (0 books in graph), but progress tracker recorded "
                    f"{len(pure_completed)} completed book(s). Resetting checkpoint to match empty graph.[/bold yellow]"
                )
                for c_bid in pure_completed:
                    tracker.remove_book("build_graph", c_bid)
                completed_ids = set()

            # Bypass both already indexed books AND memorized skipped books upfront!
            orig_len = len(target_list)
            bypass_ids = completed_ids | tracker_skipped
            target_list = [b for b in target_list if b.get("id") not in bypass_ids]
            bypassed = orig_len - len(target_list)
            if bypassed > 0:
                next_bid_str = f"Next book: #{target_list[0]['id']} ('{target_list[0].get('title', '')}')" if target_list else "None (all books indexed or skipped)"
                if tracker_skipped:
                    msg = (
                        f"Knowledge Graph Resume: Skipped {bypassed} book(s) "
                        f"({len(completed_ids)} already indexed in graph, {len(tracker_skipped)} memorized as skipped). "
                        f"{len(target_list)} remaining to process. {next_bid_str}"
                    )
                else:
                    msg = (
                        f"Knowledge Graph Resume: Skipped {bypassed} already indexed book(s) "
                        f"({len(completed_ids)} in graph). {len(target_list)} remaining to process. {next_bid_str}"
                    )
                console.print(f"[dim cyan]{msg}[/dim cyan]")
        elif not retry_skipped and tracker.get_skipped_ids("build_graph"):
            tracker_skipped = tracker.get_skipped_ids("build_graph")
            orig_len = len(target_list)
            target_list = [b for b in target_list if b.get("id") not in tracker_skipped]
            bypassed = orig_len - len(target_list)
            if bypassed > 0:
                next_bid_str = f"Next book: #{target_list[0]['id']} ('{target_list[0].get('title', '')}')" if target_list else "None (all books skipped)"
                msg = (
                    f"Preserved Skip List: Bypassed {bypassed} previously skipped book(s) "
                    f"(graphical or unsupported formats). {len(target_list)} remaining to process. {next_bid_str}"
                )
                console.print(f"[dim cyan]{msg}[/dim cyan]")

        export_dir = output_dir / "calibre_ingest"
        for b in target_list:
            if limit is not None and limit > 0 and len(books_to_process) >= limit:
                break
            bid = b["id"]
            formats = b.get("formats", [])
            # Check if book only has graphical formats (CBR, CBZ, DJVU)
            if formats and BookParser.only_has_graphical_formats(formats):
                fmts_label = ", ".join(formats)
                console.print(f"[dim yellow]⚡ Memorized skipped book #{bid}: graphical format ({fmts_label}).[/dim yellow]")
                tracker.mark_skipped("build_graph", bid, title=b.get("title", f"Book {bid}"), reason=f"graphical: {fmts_label}")
                continue

            # If cached chunks exist locally, skip exporting or downloading EPUB from SMB share
            if chunk_store.has_chunks(bid) and not rechunk:
                cached_files = [
                    f for f in export_dir.glob(f"{bid}_*.*")
                    if f.is_file() and f.stat().st_size > 0 and not f.name.endswith((".json", ".part", ".tmp"))
                ]
                books_to_process.append(
                    {
                        "id": bid,
                        "title": b.get("title", f"Book {bid}"),
                        "author": ", ".join(b.get("authors", [])) or "Unknown",
                        "path": cached_files[0] if cached_files else None,
                    }
                )
            else:
                try:
                    book_file = client.export_book(bid, target_dir=export_dir)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not export book #{bid} ({e}).[/yellow]")
                    book_file = None

                if book_file and book_file.is_file():
                    books_to_process.append(
                        {
                            "id": bid,
                            "title": b.get("title", f"Book {bid}"),
                            "author": ", ".join(b.get("authors", [])) or "Unknown",
                            "path": book_file,
                        }
                    )
                else:
                    console.print(f"[yellow]Skipping book #{bid}: No supported text format available (EPUB, FB2, PDF, RTF, TXT, ZIP).[/yellow]")
                    tracker.mark_skipped("build_graph", bid, title=b.get("title", f"Book {bid}"), reason="No supported text format available")
    else:
        console.print(
            "[bold red]Calibre not available. Pass an explicit book file via --file <path.epub> "
            "or mount the SMB share and pass --calibre-path <path>.[/bold red]"
        )
        raise typer.Exit(1)

    if should_clean:
        indexed_books_count = len([n for n, d in store.graph.nodes(data=True) if d.get("type") == "Book"])
        export_reindex_str = f" and re-exporting {indexed_books_count} book(s) already in knowledge graph" if indexed_books_count > 0 else ""
        console.print(
            f"[bold cyan]Clean export requested: clearing old export destinations (Obsidian / Neo4j){export_reindex_str}...[/bold cyan]"
        )
        obs = ObsidianExporter(vault_dest)
        obs.export(store, clean=True)
        gml = GraphMLExporter(graphml_dest)
        gml.export(store)
        if should_export_neo4j:
            try:
                neo_exp = Neo4jExporter.from_config(cfg.neo4j)
                res = neo_exp.export(store, clean=True, show_progress=True, console=console)
                clean_msg = f" (clean start: purged {res['cleaned_nodes']} previous nodes)" if res.get("clean_start") else ""
                console.print(
                    f"[bold green]✓ Neo4j Clean Export Complete{clean_msg}:[/bold green] "
                    f"{res['nodes_upserted']} nodes, {res['edges_upserted']} relationships "
                    f"in database '{res['database']}' at {res['uri']}."
                )
            except Exception as e:
                console.print(f"[bold red]✗ Failed to export to Neo4j:[/bold red] {e}")
        has_cleaned = True

    if not books_to_process:
        if store.graph.number_of_nodes() > 0:
            console.print("[bold green]All books are already indexed in knowledge graph. Knowledge graph and export are up to date.[/bold green]")
        else:
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
    pipeline_start_time = time.perf_counter()

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

    total_books = len(books_to_process)
    try:
        for book_idx, binfo in enumerate(books_to_process, start=1):
            bid = binfo["id"]
            btitle = BookParser.repair_mojibake(binfo["title"])
            bauthor = BookParser.repair_mojibake(binfo["author"])
            bpath: Optional[Path] = binfo.get("path")

            model_banner = f"LLM: [bold green]{cfg.llm_model}[/bold green]"
            if cfg.llm_model_fallback:
                model_banner += f" (fallback: [bold green]{cfg.llm_model_fallback}[/bold green])"
            console.print(
                f"\n[bold blue]► Ingesting Book #{bid} ({book_idx}/{total_books}): {btitle}[/bold blue] "
                f"[dim]| {model_banner} | Embeddings: [bold cyan]{cfg.embedding_model}[/bold cyan][/dim]"
            )

            try:
                # Check if chunks already exist in local ChunkStore
                chunks = None
                if chunk_store.has_chunks(bid) and not rechunk:
                    chunks = chunk_store.load_chunks(bid)
                    if chunks:
                        console.print(f"  Loaded [bold green]{len(chunks)} cached atomic chunks[/bold green] from local storage.")

                if not chunks:
                    # Resolve book file path lazily if bpath is missing or not a file
                    if not bpath or not bpath.is_file():
                        # 1. Check if already exported in export_dir
                        candidates = [
                            f for f in export_dir.glob(f"{bid}_*.*")
                            if f.is_file() and f.stat().st_size > 0 and not f.name.endswith((".json", ".part", ".tmp"))
                        ]
                        if candidates:
                            bpath = candidates[0]
                        elif client and client.is_available():
                            try:
                                bpath = client.export_book(bid, target_dir=export_dir)
                            except Exception as e:
                                bpath = None

                    if not bpath or not bpath.is_file():
                        console.print(f"  [yellow]Skipping book #{bid} ('{btitle}'): Book source file not available or unreadable.[/yellow]")
                        tracker.mark_skipped("build_graph", bid, title=btitle, reason="Book source file not available")
                        continue

                    # Parse sections
                    sections = BookParser.parse(bpath)
                    if not sections:
                        console.print(f"  [yellow]Skipping book #{bid} ('{btitle}'): No extractable text content found in {bpath.name}.[/yellow]")
                        tracker.mark_skipped("build_graph", bid, title=btitle, reason="No extractable text content found")
                        continue

                    console.print(f"  Extracted [green]{len(sections)} sections/chapters[/green].")

                    if hasattr(chunker, "reset_embedding_stats"):
                        chunker.reset_embedding_stats()
                    elif hasattr(embeddings, "reset_stats"):
                        embeddings.reset_stats()

                    t_chunk_start = time.perf_counter()

                    # Chunk sections with live visual progress
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                        BarColumn(),
                        TaskProgressColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                        ChunkRemainingColumn(),
                        console=console,
                    ) as chunk_progress:
                        chunk_task = chunk_progress.add_task(
                            f"Smart chunking '{btitle[:25]}'...",
                            total=len(sections),
                            completed=0,
                            eta_str="--m --s",
                        )

                        def _on_chunk_progress(completed: int, total: int, sec_title: str, num_chunks: int):
                            clean_sec = (sec_title or f"Section #{completed + 1}").strip().replace("\n", " ")
                            if len(clean_sec) > 30:
                                clean_sec = clean_sec[:27] + "..."
                            st = chunker.embedding_stats or getattr(embeddings, "embedding_stats", {})
                            spd_txt = st.get("speed_texts_per_sec", 0.0)
                            spd_kb = st.get("speed_chars_per_sec", 0.0) / 1024.0
                            if spd_txt > 0:
                                speed_badge = f" | [bold green]{spd_txt:.1f} sent/s[/bold green] [dim]({spd_kb:.1f} KB/s)[/dim]"
                            else:
                                speed_badge = ""
                            elapsed = time.perf_counter() - t_chunk_start
                            if completed > 0 and total > completed:
                                rate = elapsed / completed
                                rem_sec = rate * (total - completed)
                                eta_val = format_eta_min_sec(rem_sec)
                            elif completed >= total:
                                eta_val = "0m 00s"
                            else:
                                eta_val = "--m --s"

                            chunk_progress.update(
                                chunk_task,
                                completed=completed,
                                total=total,
                                eta_str=eta_val,
                                description=f"Chunking [yellow]{clean_sec}[/yellow] ({num_chunks} chunks{speed_badge})",
                            )

                        chunks = chunker.chunk_book(
                            sections,
                            book_id=bid,
                            book_title=btitle,
                            progress_callback=_on_chunk_progress,
                        )

                        st = chunker.embedding_stats or getattr(embeddings, "embedding_stats", {})
                        spd_txt = st.get("speed_texts_per_sec", 0.0)
                        spd_kb = st.get("speed_chars_per_sec", 0.0) / 1024.0
                        total_sent = st.get("total_texts", 0)
                        total_emb_time = st.get("total_seconds", 0.0)
                        final_speed_str = f" | {spd_txt:.1f} sent/s" if spd_txt > 0 else ""
                        chunk_progress.update(
                            chunk_task,
                            completed=len(sections),
                            eta_str="0m 00s",
                            description=f"Chunked {len(sections)} sections ({len(chunks)} chunks{final_speed_str})",
                        )

                    chunk_duration = time.perf_counter() - t_chunk_start
                    if total_sent > 0 and total_emb_time > 0:
                        speed_detail = (
                            f" [dim](Embedding speed: [bold cyan]{spd_txt:.1f} sent/s[/bold cyan] / "
                            f"[bold cyan]{spd_kb:.1f} KB/s[/bold cyan], "
                            f"{total_sent} sentences in {total_emb_time:.2f}s)[/dim]"
                        )
                    else:
                        speed_detail = f" [dim]({chunk_duration:.2f}s)[/dim]"

                    console.print(f"  Created [bold green]{len(chunks)} atomic thematic chunks[/bold green]{speed_detail}.")

                    # Persist chunks to local storage immediately
                    chunk_store.save_chunks(bid, btitle, chunks)
                    console.print(f"  [dim]Saved chunks to local storage: {chunk_store._chunk_file(bid).name}[/dim]")

                t_book_extract_start = time.perf_counter()

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    IngestionRemainingColumn(),
                    console=console,
                ) as progress:
                    # Line 1: Current book chunk extraction
                    book_task = progress.add_task(
                        f"Current Book #{bid} ('{btitle[:18]}')",
                        total=len(chunks),
                        completed=0,
                        eta_type="book",
                        eta_str="--h --m --s",
                    )
                    # Line 2: Overall catalog progress
                    overall_task = progress.add_task(
                        f"Overall Catalog Progress",
                        total=total_books,
                        completed=book_idx - 1,
                        eta_type="overall",
                        eta_str="--d --h --m",
                    )

                    extracted_results: List[Dict[str, Any]] = []

                    def _update_progress_description():
                        avg_s, p80_s, p18_s, fastest_s = _calc_stats()
                        with stats_lock:
                            cur_active = active_tasks_count
                            num_extracted = len(extracted_results)

                        # Calculate book analyses ETA formatted as (h, min, sec)
                        rem_chunks = max(0, len(chunks) - num_extracted)
                        book_elapsed = time.perf_counter() - t_book_extract_start
                        if rem_chunks == 0:
                            book_eta_str = "0h 00m 00s"
                        elif num_extracted > 0 and book_elapsed > 0:
                            rate = book_elapsed / num_extracted
                            book_rem_sec = rate * rem_chunks
                            book_eta_str = format_eta_h_min_sec(book_rem_sec)
                        elif durations:
                            avg_chunk_dur = sum(durations) / len(durations)
                            concurrency = max(1, min(pool_concurrency, rem_chunks))
                            book_rem_sec = (rem_chunks * avg_chunk_dur) / concurrency
                            book_eta_str = format_eta_h_min_sec(book_rem_sec)
                        else:
                            book_eta_str = "--h --m --s"

                        desc1 = (
                            f"[bold cyan]Current Book #{bid}[/bold cyan] ('{btitle[:18]}') | "
                            f"Active: [bold cyan]{cur_active}[/bold cyan] tasks"
                        )
                        if durations:
                            desc1 += (
                                f" | avg: [bold green]{avg_s:.1f}s[/bold green] | "
                                f"p80: [bold yellow]{p80_s:.1f}s[/bold yellow]"
                            )
                        progress.update(book_task, description=desc1, eta_str=book_eta_str)

                        # Calculate overall library processing ETA formatted as (days, h, m)
                        curr_fraction = (num_extracted / len(chunks)) if chunks else 0.0
                        overall_completed_fraction = (book_idx - 1) + curr_fraction
                        overall_remaining_fraction = max(0.0, total_books - overall_completed_fraction)
                        overall_elapsed = time.perf_counter() - pipeline_start_time

                        if overall_remaining_fraction == 0:
                            overall_eta_str = "0d 0h 00m"
                        elif overall_completed_fraction > 0 and overall_elapsed > 0:
                            overall_rate = overall_elapsed / overall_completed_fraction
                            overall_rem_sec = overall_rate * overall_remaining_fraction
                            overall_eta_str = format_eta_days_h_m(overall_rem_sec)
                        else:
                            overall_eta_str = "--d --h --m"

                        desc2 = (
                            f"[bold magenta]Overall Progress[/bold magenta] ({book_idx}/{total_books} books) | "
                            f"LLM: [bold green]{cfg.llm_model}[/bold green] | "
                            f"Embeddings: [bold cyan]{cfg.embedding_model}[/bold cyan]"
                        )
                        progress.update(
                            overall_task,
                            completed=book_idx - 1 + curr_fraction,
                            description=desc2,
                            eta_str=overall_eta_str,
                        )

                    _update_progress_description()

                    book_state_dir = output_dir / ".book_states"
                    book_state = BookProcessingState(
                        book_id=bid,
                        book_title=btitle,
                        state_dir=book_state_dir,
                        max_chunk_attempts=6,
                    )
                    book_state.set_total_chunks(len(chunks))

                    # 1. Restore any already processed or failed chunks from previous run
                    restored_from_state = 0
                    restored_failed = 0
                    for chk in chunks:
                        if book_state.is_chunk_processed(chk.chunk_id):
                            cached_rec = book_state.get_processed_chunk(chk.chunk_id)
                            raw_extraction = cached_rec.get("extraction", {"concepts": []})
                            extraction = SectionExtraction(**raw_extraction)
                            dur = cached_rec.get("duration", 0.0)
                            srv = cached_rec.get("server_used", "cached")
                            with stats_lock:
                                durations.append(dur)
                                extracted_results.append({
                                    "status": "extracted",
                                    "chunk": chk,
                                    "extraction": extraction,
                                    "duration": dur,
                                    "server_used": srv,
                                })
                            restored_from_state += 1
                        elif book_state.is_chunk_failed(chk.chunk_id):
                            failed_rec = book_state.get_failed_chunk(chk.chunk_id)
                            with stats_lock:
                                extracted_results.append({
                                    "status": "failed_chunk",
                                    "chunk": chk,
                                    "extraction": SectionExtraction(concepts=[]),
                                    "duration": 0.0,
                                    "server_used": failed_rec.get("server_used", "failed_cache"),
                                    "error": failed_rec.get("error", "Poison chunk failed 6 attempts"),
                                })
                            restored_failed += 1

                    if restored_from_state > 0 or restored_failed > 0:
                        progress.advance(book_task, advance=restored_from_state + restored_failed)
                        console.print(
                            f"  [bold cyan]↺ Resuming Book #{bid}: {restored_from_state} processed chunk(s) restored from state"
                            f"{f', {restored_failed} previously failed poison chunk(s)' if restored_failed else ''}.[/bold cyan]"
                        )
                        _update_progress_description()

                    fresh_queue: queue.Queue[Tuple[HierarchicalChunk, int, Set[str]]] = queue.Queue()
                    deferred_retries: List[Tuple[HierarchicalChunk, int, Set[str]]] = []
                    retry_queue: queue.Queue[Tuple[HierarchicalChunk, int, Set[str]]] = queue.Queue()

                    for chk in chunks:
                        if not book_state.is_chunk_processed(chk.chunk_id) and not book_state.is_chunk_failed(chk.chunk_id):
                            init_attempt = book_state.get_initial_attempt(chk.chunk_id)
                            if init_attempt == 0:
                                fresh_queue.put((chk, init_attempt, set()))
                            else:
                                deferred_retries.append((chk, init_attempt, set()))

                    max_chunk_attempts = cfg.max_chunk_attempts
                    phase: str = "fresh"
                    fresh_in_flight: int = 0
                    retry_in_flight: int = 0

                    def _chunk_worker():
                        nonlocal active_tasks_count, phase, fresh_in_flight, retry_in_flight
                        while not abort_event.is_set():
                            with stats_lock:
                                if len(extracted_results) >= len(chunks):
                                    break

                                # Transition from initial pass to retry phase once ALL fresh chunks are finished
                                if phase == "fresh" and fresh_queue.empty() and fresh_in_flight == 0:
                                    if deferred_retries:
                                        phase = "retry"
                                        with display_lock:
                                            console.print(
                                                f"  [bold yellow]↺ Completed initial pass on all unprocessed chunks. "
                                                f"Starting retry phase for {len(deferred_retries)} skipped/failed chunk(s)...[/bold yellow]"
                                            )
                                        for item in deferred_retries:
                                            retry_queue.put(item)
                                        deferred_retries.clear()
                                    else:
                                        # No retries needed, all chunks resolved!
                                        break

                                cur_phase = phase

                            # Fetch next chunk according to current phase
                            task_phase: Optional[str] = None
                            if cur_phase == "fresh":
                                try:
                                    chk_item, attempt, tried_urls = fresh_queue.get(timeout=0.2)
                                except queue.Empty:
                                    continue
                                task_phase = "fresh"
                                with stats_lock:
                                    fresh_in_flight += 1
                                    active_tasks_count += 1
                                _update_progress_description()
                            else:  # cur_phase == "retry"
                                try:
                                    chk_item, attempt, tried_urls = retry_queue.get(timeout=0.2)
                                except queue.Empty:
                                    with stats_lock:
                                        if len(extracted_results) >= len(chunks):
                                            break
                                    continue
                                task_phase = "retry"
                                with stats_lock:
                                    retry_in_flight += 1
                                    active_tasks_count += 1
                                _update_progress_description()

                            t0 = time.time()
                            srv = "ollama"
                            book_state.mark_chunk_stacked(
                                chunk_id=chk_item.chunk_id,
                                chunk_idx=chk_item.chunk_idx,
                                section_title=chk_item.section_title,
                                attempt=attempt,
                            )
                            is_fallback_attempt = bool(
                                cfg.llm_model_fallback and attempt >= (max_chunk_attempts / 2.0)
                            )
                            target_model = cfg.llm_model_fallback if is_fallback_attempt else cfg.llm_model

                            try:
                                if abort_event.is_set():
                                    book_state.unstack_chunk(chk_item.chunk_id)
                                    if task_phase == "fresh":
                                        fresh_queue.task_done()
                                    else:
                                        retry_queue.task_done()
                                    break

                                extraction = extractor.extract_section(
                                    text=chk_item.text,
                                    book_title=btitle,
                                    section_title=chk_item.section_title,
                                    subtitle=chk_item.subtitle,
                                    parent_context=chk_item.parent_text,
                                    model=target_model,
                                    retries=cfg.max_retries,
                                    quarantine_server=False,
                                    exclude_urls=tried_urls,
                                    raise_on_error=True,
                                )
                                dur = time.time() - t0
                                srv = extractor.pool.get_last_used_server() or "ollama"
                                with stats_lock:
                                    durations.append(dur)
                                    extracted_results.append({
                                        "status": "extracted",
                                        "chunk": chk_item,
                                        "extraction": extraction,
                                        "duration": dur,
                                        "server_used": srv,
                                        "model_used": target_model,
                                    })

                                book_state.mark_chunk_processed(
                                    chunk_id=chk_item.chunk_id,
                                    chunk_idx=chk_item.chunk_idx,
                                    section_title=chk_item.section_title,
                                    extraction_dict=extraction.model_dump(),
                                    duration=dur,
                                    server_used=srv,
                                )

                                model_tag = f" [{target_model}]" if is_fallback_attempt else ""
                                with display_lock:
                                    console.print(
                                        f"  [dim green]✓ [{chk_item.section_title[:24]} p{chk_item.chunk_idx}][/dim green] "
                                        f"Extracted [bold cyan]{len(extraction.concepts)} concept(s)[/bold cyan] "
                                        f"[dim]({dur:.1f}s via [bold green]{srv}{model_tag}[/bold green])[/dim]"
                                    )

                                progress.advance(book_task)
                                _update_progress_description()
                                if task_phase == "fresh":
                                    fresh_queue.task_done()
                                else:
                                    retry_queue.task_done()

                            except Exception as exc:
                                dur = time.time() - t0
                                srv = extractor.pool.get_last_used_server() or "ollama"
                                srv_url = extractor.pool.get_last_used_server_url()
                                if srv_url:
                                    tried_urls.add(srv_url)
                                attempt += 1

                                next_is_fallback = bool(
                                    cfg.llm_model_fallback and attempt >= (max_chunk_attempts / 2.0)
                                )
                                fallback_note = (
                                    f" (switching to fallback model '{cfg.llm_model_fallback}')"
                                    if (next_is_fallback and not is_fallback_attempt)
                                    else ""
                                )

                                if task_phase == "fresh":
                                    # At first failed attempt, skip it and continue with rest of unprocessed chunks.
                                    # Collect in deferred_retries to be retried only after all fresh chunks finish.
                                    book_state.mark_chunk_retried(
                                        chunk_id=chk_item.chunk_id,
                                        chunk_idx=chk_item.chunk_idx,
                                        section_title=chk_item.section_title,
                                        attempt=attempt,
                                        error=str(exc),
                                        server_used=srv,
                                    )
                                    with display_lock:
                                        console.print(
                                            f"  [yellow]⚠ [{chk_item.section_title[:24]} p{chk_item.chunk_idx}] "
                                            f"Attempt {attempt}/{max_chunk_attempts} failed on {srv} ({exc}). "
                                            f"Skipping for now; will retry after all unprocessed chunks are completed{fallback_note}.[/yellow]"
                                        )
                                    with stats_lock:
                                        deferred_retries.append((chk_item, attempt, tried_urls))
                                    fresh_queue.task_done()
                                else:
                                    # Retry phase: retry across pool until max_chunk_attempts
                                    if attempt < max_chunk_attempts and not abort_event.is_set():
                                        book_state.mark_chunk_retried(
                                            chunk_id=chk_item.chunk_id,
                                            chunk_idx=chk_item.chunk_idx,
                                            section_title=chk_item.section_title,
                                            attempt=attempt,
                                            error=str(exc),
                                            server_used=srv,
                                        )
                                        with display_lock:
                                            console.print(
                                                f"  [yellow]⚠ [{chk_item.section_title[:24]} p{chk_item.chunk_idx}] "
                                                f"Attempt {attempt}/{max_chunk_attempts} stalled/timed out on {srv} ({exc}). "
                                                f"Moving to end of retry queue for another server{fallback_note}...[/yellow]"
                                            )
                                        retry_queue.put((chk_item, attempt, tried_urls))
                                        retry_queue.task_done()
                                    else:
                                        book_state.mark_chunk_failed(
                                            chunk_id=chk_item.chunk_id,
                                            chunk_idx=chk_item.chunk_idx,
                                            section_title=chk_item.section_title,
                                            attempts=attempt,
                                            error=str(exc),
                                            server_used=srv,
                                        )
                                        with stats_lock:
                                            extracted_results.append({
                                                "status": "failed_chunk",
                                                "chunk": chk_item,
                                                "extraction": SectionExtraction(concepts=[]),
                                                "duration": dur,
                                                "server_used": srv,
                                                "error": str(exc),
                                            })
                                        with display_lock:
                                            console.print(
                                                f"  [bold red]✗ [{chk_item.section_title[:24]} p{chk_item.chunk_idx}] "
                                                f"Failed after {max_chunk_attempts} attempts across pool ({exc}). "
                                                f"Marking chunk as poison pill and continuing book processing.[/bold red]"
                                            )
                                        progress.advance(book_task)
                                        _update_progress_description()
                                        retry_queue.task_done()
                            finally:
                                if task_phase == "fresh":
                                    with stats_lock:
                                        fresh_in_flight = max(0, fresh_in_flight - 1)
                                        active_tasks_count = max(0, active_tasks_count - 1)
                                elif task_phase == "retry":
                                    with stats_lock:
                                        retry_in_flight = max(0, retry_in_flight - 1)
                                        active_tasks_count = max(0, active_tasks_count - 1)
                                _update_progress_description()

                    chunk_executor = DaemonThreadPoolExecutor(max_workers=pool_concurrency)
                    worker_futures = []
                    try:
                        worker_futures = [
                            chunk_executor.submit(_chunk_worker) for _ in range(pool_concurrency)
                        ]
                        for f in worker_futures:
                            f.result()

                    except KeyboardInterrupt:
                        abort_event.set()
                        progress.stop()
                        for f in worker_futures:
                            f.cancel()
                        chunk_executor.shutdown(wait=False, cancel_futures=True)
                        import concurrent.futures.thread
                        with concurrent.futures.thread._global_shutdown_lock:
                            for t in list(chunk_executor._threads):
                                concurrent.futures.thread._threads_queues.pop(t, None)
                        store.remove_book(bid)
                        raise
                    finally:
                        chunk_executor.shutdown(wait=False, cancel_futures=True)

                    if abort_event.is_set():
                        store.remove_book(bid)
                        console.print(
                            f"  [yellow]Book #{bid} processing was cancelled by user. Partial chunk state retained for resume.[/yellow]"
                        )
                        continue

                    successful_results = [r for r in extracted_results if r.get("status") == "extracted"]
                    failed_poison_results = [r for r in extracted_results if r.get("status") == "failed_chunk"]

                    # Atomically integrate successful extractions into knowledge graph
                    store.add_book(bid, title=btitle, author=bauthor)
                    for res in successful_results:
                        chk = res["chunk"]
                        extraction = res["extraction"]

                        sec_node_id = store.add_section(
                            book_id=bid,
                            chapter_idx=chk.chapter_idx,
                            title=chk.section_title,
                            text=chk.text,
                        )
                        store.add_chunk(chk)

                        for concept in extraction.concepts:
                            canonical_concept = deduplicator.resolve_concept(concept)
                            store.add_concept(canonical_concept)

                            store.add_book_idea_link(
                                book_id=bid,
                                concept_name=canonical_concept.name,
                            )

                            # Validate textual grounding in this specific chunk
                            local_brief = (concept.brief_description or "").strip()
                            local_detailed = (concept.detailed_explanation or "").strip()
                            is_grounded, valid_quote = validate_concept_chunk_grounding(
                                concept_name=canonical_concept.name,
                                supporting_quote=concept.supporting_quote,
                                chunk_text=chk.text,
                                brief_description=local_brief or local_detailed,
                            )

                            if is_grounded:
                                store.add_idea_support_link(
                                    concept_name=canonical_concept.name,
                                    chunk=chk,
                                    quote=valid_quote,
                                    brief_description=local_brief,
                                    detailed_explanation=local_detailed,
                                )
                                store.add_section_concept_link(
                                    section_node_id=sec_node_id,
                                    concept_name=canonical_concept.name,
                                    summary=concept.summary,
                                    quote=valid_quote,
                                )
                            else:
                                logger.debug(
                                    f"Skipping ungrounded concept link '{canonical_concept.name}' on chunk {chk.chunk_id} "
                                    f"(neither quote nor concept name present in chunk text)"
                                )
                            for rel_name in canonical_concept.related_concepts:
                                store.add_concept_relation(
                                    src_concept_name=canonical_concept.name,
                                    tgt_concept_name=rel_name,
                                )

                    # Ensure sequential NEXT and PREV relationships between chunks for this book
                    store.link_sequential_chunks(book_id=bid)

                    overall_completed_fraction = float(book_idx)
                    overall_remaining_fraction = max(0.0, total_books - overall_completed_fraction)
                    overall_elapsed = time.perf_counter() - pipeline_start_time
                    if overall_remaining_fraction == 0:
                        overall_eta_str = "0d 0h 00m"
                    elif overall_completed_fraction > 0 and overall_elapsed > 0:
                        overall_rate = overall_elapsed / overall_completed_fraction
                        overall_rem_sec = overall_rate * overall_remaining_fraction
                        overall_eta_str = format_eta_days_h_m(overall_rem_sec)
                    else:
                        overall_eta_str = "--d --h --m"

                    progress.update(overall_task, completed=book_idx, eta_str=overall_eta_str)

                if lightrag_engine is not None and chunks:
                    try:
                        valid_chunks = [r["chunk"] for r in successful_results]
                        if valid_chunks:
                            console.print(f"  [dim cyan]Indexing {len(valid_chunks)} chunk(s) into LightRAG engine...[/dim cyan]")
                            lightrag_engine.insert_chunks(valid_chunks, book_title=btitle, book_id=bid)
                            console.print(f"  [dim green]✓ Indexed into LightRAG.[/dim green]")
                    except Exception as e:
                        console.print(f"  [yellow]Warning: LightRAG indexing failed for '{btitle}': {e}[/yellow]")

                # Finalize book state:
                # If there are poison pill failed chunks -> partially indexed, state retained.
                # If no errors -> fully indexed, state file deleted.
                if failed_poison_results:
                    book_state.finalize(has_errors=True)
                    tracker.mark_partially_indexed(
                        "build_graph",
                        bid,
                        title=btitle,
                        failed_chunks=len(failed_poison_results),
                        total_chunks=len(chunks),
                    )
                    console.print(
                        f"  [bold yellow]⚠ Book #{bid} partially indexed: {len(successful_results)}/{len(chunks)} chunks succeeded, "
                        f"{len(failed_poison_results)} failed poison chunk(s) skipped. State retained at {book_state.file_path}.[/bold yellow]"
                    )
                else:
                    book_state.finalize(has_errors=False)
                    tracker.mark_completed(
                        "build_graph",
                        bid,
                        title=btitle,
                        metadata={"indexing_status": "fully_indexed", "total_chunks": len(chunks)},
                    )
                    console.print(
                        f"  [dim green]✓ Book #{bid} fully indexed (all {len(chunks)} chunks succeeded). State retained at {book_state.file_path}.[/dim green]"
                    )

                # Incremental persistence: save graph checkpoint after every processed book
                store.save(graph_file)
                completed_in_session += 1
                console.print(f"  [dim green]✓ Book #{bid} concepts integrated and graph checkpoint saved.[/dim green]")

                if limit is not None and limit > 0 and completed_in_session >= limit:
                    console.print(
                        f"\n[bold green]✓ Reached target limit of {limit} successfully processed book(s). Halting build-graph as requested.[/bold green]"
                    )
                    break

            except KeyboardInterrupt:
                store.remove_book(bid)
                raise
            except Exception as e:
                console.print(f"[bold red]✗ Failed to build graph for book #{bid} ('{btitle}'):[/bold red] {e}")
                tracker.mark_failed("build_graph", bid, title=btitle, error=str(e))
                store.remove_book(bid)

    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[bold yellow]Cancelled by user. Saving current Knowledge Graph state...[/bold yellow]")

    finally:
        # Ensure only 100% fully processed books remain in graph before final save and export
        tracker_completed = tracker.get_completed_ids("build_graph")
        graph_bids = {
            attrs["book_id"]
            for _, attrs in store.graph.nodes(data=True)
            if attrs.get("type") == "Book" and "book_id" in attrs
        }
        unprocessed = graph_bids - tracker_completed
        for unproc_bid in unprocessed:
            store.remove_book(unproc_bid)

        vault_dest = Path(export_obsidian) if export_obsidian else output_dir / "obsidian_vault"
        graphml_dest = output_dir / "knowledge_graph.graphml"

        if store.graph.number_of_nodes() > 0:
            store.save(graph_file)
            final_clean = should_clean and not has_cleaned
            obs = ObsidianExporter(vault_dest)
            obs.export(store, clean=final_clean)
            gml = GraphMLExporter(graphml_dest)
            gml.export(store)

            should_export_neo4j = cfg.neo4j.enabled if export_neo4j is None else export_neo4j
            if should_export_neo4j:
                try:
                    console.print("\n[bold cyan]Exporting Knowledge Graph to Neo4j...[/bold cyan]")
                    neo_clean = final_clean or (cfg.neo4j.clean_export and not has_cleaned)
                    neo_exp = Neo4jExporter.from_config(cfg.neo4j)
                    res = neo_exp.export(store, clean=neo_clean, show_progress=True, console=console)
                    clean_msg = f" (clean start: purged {res['cleaned_nodes']} previous nodes)" if res.get("clean_start") else ""
                    console.print(
                        f"[bold green]✓ Neo4j Export Complete{clean_msg}:[/bold green] "
                        f"{res['nodes_upserted']} nodes, {res['edges_upserted']} relationships "
                        f"in database '{res['database']}' at {res['uri']}."
                    )
                except Exception as e:
                    console.print(f"[bold red]✗ Failed to export to Neo4j:[/bold red] {e}")

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

        pools_to_report = [extractor.pool]
        if hasattr(embeddings, "pool") and getattr(embeddings, "pool", None) is not extractor.pool:
            pools_to_report.append(embeddings.pool)
        _print_ollama_server_stats(pools_to_report, console=console)

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
    table.add_column("Partially Indexed", style="cyan", justify="right")
    table.add_column("Skipped (Graphical)", style="yellow", justify="right")
    table.add_column("Failed", style="bold red", justify="right")
    table.add_column("Total Recorded", style="white", justify="right")

    table.add_row(
        "clean_metadata",
        str(clean_summary["completed"]),
        str(clean_summary.get("partially_indexed", 0)),
        str(clean_summary.get("skipped", 0)),
        str(clean_summary["failed"]),
        str(clean_summary["total_recorded"]),
    )
    table.add_row(
        "build_graph",
        str(graph_summary["completed"]),
        str(graph_summary.get("partially_indexed", 0)),
        str(graph_summary.get("skipped", 0)),
        str(graph_summary["failed"]),
        str(graph_summary["total_recorded"]),
    )
    console.print(table)

    book_states_dir = cfg.resolved_output_dir / ".book_states"
    if book_states_dir.is_dir():
        state_files = list(book_states_dir.glob("book_*_state.json"))
        if state_files:
            console.print(f"[dim cyan]Per-book state checkpoints ({len(state_files)} found in {book_states_dir}):[/dim cyan]")
            for sf in state_files[:5]:
                try:
                    with open(sf, "r", encoding="utf-8") as f:
                        sdata = json.load(f)
                    b_status = sdata.get("status", "unknown")
                    b_proc = len(sdata.get("processed_chunks", {}))
                    b_fail = len(sdata.get("failed_chunks", {}))
                    b_tot = sdata.get("total_chunks", 0)
                    b_title = sdata.get("book_title", sf.stem)
                    console.print(
                        f"  • [bold]{b_title}[/bold] (ID: {sdata.get('book_id')}): status=[cyan]{b_status}[/cyan], "
                        f"processed={b_proc}/{b_tot}, failed={b_fail}"
                    )
                except Exception:
                    pass
            if len(state_files) > 5:
                console.print(f"  [dim]... and {len(state_files) - 5} more.[/dim]")

    console.print(
        f"[dim]Active Models: LLM: [bold green]{cfg.llm_model}[/bold green] | "
        f"Embedding Model: [bold cyan]{cfg.embedding_model}[/bold cyan][/dim]"
    )
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
            f"[bold]Total Edges:[/bold] {st['total_edges']}\n"
            + (f"[bold]Avg Concept Weight:[/bold] {st.get('average_concept_weight', 0.0)}/10\n\n" if "average_concept_weight" in st else "\n")
            + "[bold green]Node Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["node_types"].items())
            + "\n\n[bold green]Relationship Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["edge_types"].items()),
            title="Knowledge Graph Metrics",
        )
    )


@app.command("query")
def query_command(
    query_text: str = typer.Argument(..., help="Question or research query to synthesize from ingested books."),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        "-m",
        help="LightRAG query mode: 'hybrid', 'local' (entities), 'global' (themes), 'naive' (vector), or 'mix' (default: config).",
    ),
    top_k: int = typer.Option(40, "--top-k", "-k", help="Maximum entities/chunks to retrieve."),
    chunk_top_k: int = typer.Option(20, "--chunk-top-k", help="Maximum chunks to retrieve for context."),
    lightrag_dir: Optional[str] = typer.Option(
        None, "--lightrag-dir", help="Path to LightRAG storage directory (default: output_dir/lightrag)."
    ),
    ollama_url: Optional[str] = typer.Option(None, "--ollama-url", help="Ollama server URL."),
    model: Optional[str] = typer.Option(None, "--model", help="LLM model name."),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Embedding model name."),
    embedding_url: Optional[str] = typer.Option(None, "--embedding-url", help="Ollama server URL dedicated for embeddings."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
):
    """
    Query the ingested library using the LightRAG dual-level graph retrieval engine.
    Fast synthesis combining local entity graphs with global thematic summaries.
    """
    cfg = _get_effective_settings(
        config_path,
        ollama_url=ollama_url,
        model=model,
        embedding_model=embedding_model,
        embedding_url=embedding_url,
    )
    target_dir = Path(lightrag_dir).expanduser().resolve() if lightrag_dir else cfg.resolved_lightrag_dir
    effective_mode = (mode or cfg.lightrag_mode).lower().strip()

    if not target_dir.exists() or not any(target_dir.iterdir()):
        console.print(
            f"[bold red]LightRAG index directory '{target_dir}' is empty or does not exist.[/bold red]\n"
            f"[yellow]Please run 'bookeeper build-graph --lightrag' first to index your books.[/yellow]"
        )
        raise typer.Exit(1)

    _print_pool_configuration(cfg, console, show_embeddings=True)

    with console.status(f"[bold cyan]Querying LightRAG (mode: {effective_mode})...[/bold cyan]"):
        try:
            engine = LightRAGEngine.from_settings(cfg, working_dir=target_dir)
            engine.initialize()
            response = engine.query(query_text, mode=effective_mode, top_k=top_k, chunk_top_k=chunk_top_k)
        except Exception as e:
            console.print(f"[bold red]LightRAG query failed:[/bold red] {e}")
            raise typer.Exit(1)

    console.print(
        Panel(
            response,
            title=f"[bold green]LightRAG Answer ({effective_mode.upper()} mode)[/bold green]",
            border_style="green",
        )
    )


@app.command("export-obsidian")
def export_obsidian_command(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    graph_file: Optional[str] = typer.Option(
        None, "--graph-file", help="Path to knowledge_graph.json (default: output_dir/knowledge_graph.json)."
    ),
    vault_dir: Optional[str] = typer.Option(
        None, "--vault-dir", "-o", help="Target Obsidian vault directory (default: output_dir/obsidian_vault)."
    ),
    clean: Optional[bool] = typer.Option(
        None,
        "--clean/--no-clean",
        help="Perform clean start: delete all existing notes in target vault before export (default: config.yaml clean_export).",
    ),
):
    """
    Export current Knowledge Graph to an Obsidian Markdown vault.
    Generates notes with Wikilinks, concept significance weights, and an index.
    Exports strictly based on existing knowledge_graph.json without processing books.
    """
    cfg = get_settings(config_path)
    target_graph_file = Path(graph_file).expanduser().resolve() if graph_file else cfg.resolved_output_dir / "knowledge_graph.json"
    if not target_graph_file.is_file():
        console.print(
            f"[bold red]Knowledge graph file not found at {target_graph_file}[/bold red]\n"
            f"[yellow]Please run 'bookeeper build-graph' first to generate knowledge_graph.json.[/yellow]"
        )
        raise typer.Exit(1)

    store = ConceptGraphStore()
    with console.status(f"[bold cyan]Loading graph from {target_graph_file}...[/bold cyan]"):
        try:
            store.load(target_graph_file)
        except Exception as e:
            console.print(f"[bold red]Failed to load knowledge graph:[/bold red] {e}")
            raise typer.Exit(1)

    target_vault = Path(vault_dir).expanduser().resolve() if vault_dir else cfg.resolved_output_dir / "obsidian_vault"
    effective_clean = cfg.clean_export if clean is None else clean

    obs = ObsidianExporter(target_vault)
    obs.export(store, clean=effective_clean)
    clean_note = " (clean start)" if effective_clean else ""
    console.print(f"[bold green]✓ Successfully exported Knowledge Graph to Obsidian vault{clean_note}:[/bold green] {target_vault}")


@app.command("export")
def export_command(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    graph_file: Optional[str] = typer.Option(
        None, "--graph-file", help="Path to knowledge_graph.json (default: output_dir/knowledge_graph.json)."
    ),
    vault_dir: Optional[str] = typer.Option(
        None, "--vault-dir", "-o", help="Target Obsidian vault directory (default: output_dir/obsidian_vault)."
    ),
    export_neo4j: Optional[bool] = typer.Option(
        None, "--neo4j/--no-neo4j", help="Also export to Neo4j (default: config.yaml neo4j.enabled)."
    ),
    neo4j_uri: Optional[str] = typer.Option(
        None, "--neo4j-uri", help="Override Neo4j connection URI (e.g. bolt://192.168.50.20:7687)."
    ),
    clean: Optional[bool] = typer.Option(
        None,
        "--clean/--no-clean",
        help="Perform clean start on export destinations before exporting (default: config.yaml clean_export).",
    ),
):
    """
    Export current Knowledge Graph to all destinations (Obsidian, GraphML, and Neo4j)
    based strictly on existing knowledge_graph.json without continuing book processing.
    """
    cfg = get_settings(config_path)
    target_graph_file = Path(graph_file).expanduser().resolve() if graph_file else cfg.resolved_output_dir / "knowledge_graph.json"
    if not target_graph_file.is_file():
        console.print(
            f"[bold red]Knowledge graph file not found at {target_graph_file}[/bold red]\n"
            f"[yellow]Please run 'bookeeper build-graph' first to generate knowledge_graph.json.[/yellow]"
        )
        raise typer.Exit(1)

    store = ConceptGraphStore()
    with console.status(f"[bold cyan]Loading graph from {target_graph_file}...[/bold cyan]"):
        try:
            store.load(target_graph_file)
        except Exception as e:
            console.print(f"[bold red]Failed to load knowledge graph:[/bold red] {e}")
            raise typer.Exit(1)

    node_count = store.graph.number_of_nodes()
    if node_count == 0:
        console.print("[yellow]Knowledge graph contains 0 nodes. Nothing to export.[/yellow]")
        return

    effective_clean = cfg.clean_export if clean is None else clean
    target_vault = Path(vault_dir).expanduser().resolve() if vault_dir else cfg.resolved_output_dir / "obsidian_vault"
    graphml_dest = cfg.resolved_output_dir / "knowledge_graph.graphml"

    clean_note = " (clean start)" if effective_clean else ""
    console.print(f"[bold cyan]Exporting Knowledge Graph ({node_count} nodes)...[/bold cyan]")

    obs = ObsidianExporter(target_vault)
    obs.export(store, clean=effective_clean)
    console.print(f"[bold green]✓ Obsidian Vault exported to{clean_note}:[/bold green] {target_vault}")

    gml = GraphMLExporter(graphml_dest)
    gml.export(store)
    console.print(f"[bold green]✓ GraphML exported to:[/bold green] {graphml_dest}")

    should_neo4j = cfg.neo4j.enabled if export_neo4j is None else export_neo4j
    if should_neo4j:
        try:
            neo_clean = effective_clean or cfg.neo4j.clean_export
            neo_config = cfg.neo4j.model_copy()
            if neo4j_uri:
                neo_config.uri = neo4j_uri
            neo_exp = Neo4jExporter.from_config(neo_config)
            res = neo_exp.export(store, clean=neo_clean, show_progress=True, console=console)
            cmsg = f" (clean start: purged {res['cleaned_nodes']} previous nodes)" if res.get("clean_start") else ""
            console.print(
                f"[bold green]✓ Neo4j Export Complete{cmsg}:[/bold green] "
                f"{res['nodes_upserted']} nodes, {res['edges_upserted']} relationships "
                f"in database '{res['database']}' at {res['uri']}."
            )
        except Exception as e:
            console.print(f"[bold red]✗ Failed to export to Neo4j:[/bold red] {e}")


@app.command("export-neo4j")
def export_neo4j_command(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    graph_file: Optional[str] = typer.Option(
        None, "--graph-file", help="Path to knowledge_graph.json (default: output_dir/knowledge_graph.json)."
    ),
    uri: Optional[str] = typer.Option(None, "--uri", help="Neo4j connection URI (e.g. bolt://localhost:7687)."),
    user: Optional[str] = typer.Option(None, "--user", "-u", help="Neo4j username."),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Neo4j password."),
    database: Optional[str] = typer.Option(None, "--database", "-d", help="Neo4j target database name."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size for Cypher UNWIND transactions."),
    clean: Optional[bool] = typer.Option(
        None,
        "--clean/--no-clean",
        help="Perform clean start: delete all existing content in target Neo4j database before export (default: config.yaml neo4j.clean_export).",
    ),
):
    """
    Export and upsert the current Knowledge Graph into Neo4j.
    Performs high-throughput Cypher MERGE operations (inserting new nodes and relationships,
    and updating existing ones with latest attributes) based on current knowledge_graph.json data.
    Supports clean start to wipe the database before exporting.
    """
    cfg = get_settings(config_path)

    # Resolve overrides
    target_uri = uri or cfg.neo4j.uri
    target_user = user or cfg.neo4j.user
    target_password = password or cfg.neo4j.password
    target_database = database or cfg.neo4j.database
    target_batch = batch_size or cfg.neo4j.batch_size
    effective_clean = cfg.neo4j.clean_export if clean is None else clean

    # Load Knowledge Graph
    target_graph_file = Path(graph_file).expanduser().resolve() if graph_file else cfg.resolved_output_dir / "knowledge_graph.json"
    if not target_graph_file.is_file():
        console.print(
            f"[bold red]Knowledge graph file not found at {target_graph_file}[/bold red]\n"
            f"[yellow]Please run 'bookeeper build-graph' first to generate knowledge_graph.json.[/yellow]"
        )
        raise typer.Exit(1)

    store = ConceptGraphStore()
    with console.status(f"[bold cyan]Loading graph from {target_graph_file}...[/bold cyan]"):
        try:
            store.load(target_graph_file)
        except Exception as e:
            console.print(f"[bold red]Failed to load knowledge graph:[/bold red] {e}")
            raise typer.Exit(1)

    node_count = store.graph.number_of_nodes()
    edge_count = store.graph.number_of_edges()
    if node_count == 0:
        console.print("[yellow]Knowledge graph contains 0 nodes. Nothing to export.[/yellow]")
        return

    export_type_str = (
        "[bold red]clean start + upsert[/bold red] (delete all existing content, then insert fresh)"
        if effective_clean
        else "[bold green]upsert[/bold green] (insert new, update existing)"
    )

    console.print(
        Panel.fit(
            f"[bold]Knowledge Graph File:[/bold] {target_graph_file}\n"
            f"[bold]Total Nodes:[/bold] {node_count}\n"
            f"[bold]Total Edges:[/bold] {edge_count}\n\n"
            f"[bold cyan]Neo4j Target:[/bold cyan] {target_uri} (db: {target_database})\n"
            f"[bold cyan]Export Type:[/bold cyan] {export_type_str}",
            title="Neo4j Export Plan",
        )
    )

    try:
        exporter = Neo4jExporter(
            uri=target_uri,
            user=target_user,
            password=target_password,
            database=target_database,
            batch_size=target_batch,
            connection_timeout=cfg.neo4j.connection_timeout,
        )
        result = exporter.export(store, clean=effective_clean, show_progress=True, console=console)
    except Exception as e:
        console.print(f"[bold red]✗ Neo4j export failed:[/bold red] {e}")
        raise typer.Exit(1)

    # Display results table
    table = Table(title="Neo4j Export Summary", border_style="green")
    table.add_column("Category", style="cyan")
    table.add_column("Type / Label", style="bold")
    table.add_column("Count", justify="right", style="green")

    if result.get("clean_start"):
        table.add_row("[bold red]Clean Start[/bold red]", "Purged Previous Nodes", f"[yellow]{result['cleaned_nodes']}[/yellow]")
        table.add_section()

    for lbl, count in sorted(result["node_breakdown"].items()):
        table.add_row("Node", f":{lbl}", str(count))

    for rel, count in sorted(result["edge_breakdown"].items()):
        table.add_row("Relationship", f":{rel}", str(count))

    table.add_section()
    table.add_row("[bold]Total Nodes[/bold]", "", f"[bold green]{result['nodes_upserted']}[/bold green]")
    table.add_row("[bold]Total Relationships[/bold]", "", f"[bold green]{result['edges_upserted']}[/bold green]")

    console.print(table)
    clean_note = f" (clean start: purged {result['cleaned_nodes']} previous nodes)" if result.get("clean_start") else ""
    console.print(
        f"[bold green]✓ Successfully exported Knowledge Graph to Neo4j{clean_note}[/bold green] "
        f"[dim]({target_uri}, db: {target_database})[/dim]\n"
    )


@app.command("build-rag")
def build_rag(
    clean: bool = typer.Option(
        False, "--clean", help="Clean start: wipe existing LightRAG database and indices before rebuilding."
    ),
    book_id: Optional[int] = typer.Option(
        None, "--book-id", "-b", help="Only index chunks for a specific Calibre book ID."
    ),
    chunks_dir: Optional[str] = typer.Option(
        None, "--chunks-dir", help="Path to chunks storage directory (default: output_dir/chunks)."
    ),
    lightrag_dir: Optional[str] = typer.Option(
        None, "--lightrag-dir", help="Working directory for LightRAG database (default: output_dir/lightrag)."
    ),
    embedding_url: Optional[str] = typer.Option(
        None, "--embedding-url", help="Ollama server URL dedicated for embeddings."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
):
    """
    Index locally cached chunks from disk into LightRAG database.
    Allows decoupling fast concept graph extraction from heavy LightRAG vector/graph indexing.
    """
    cfg = _get_effective_settings(config_path, embedding_url=embedding_url)
    resolved_chunks_path = Path(chunks_dir).expanduser().resolve() if chunks_dir else cfg.resolved_chunks_dir
    chunk_store = ChunkStore(resolved_chunks_path)

    stored_ids = chunk_store.list_stored_book_ids()
    if not stored_ids:
        console.print(f"[yellow]No cached chunks found in {chunk_store.storage_dir}. Run 'build-graph' first.[/yellow]")
        raise typer.Exit(0)

    if book_id is not None:
        if book_id not in stored_ids:
            console.print(f"[red]No cached chunks found for book #{book_id} in {chunk_store.storage_dir}.[/red]")
            raise typer.Exit(1)
        target_ids = [book_id]
    else:
        target_ids = stored_ids

    target_lrag_dir = Path(lightrag_dir).expanduser().resolve() if lightrag_dir else cfg.resolved_lightrag_dir
    extractor = KnowledgeExtractor.from_settings(cfg)
    engine = LightRAGEngine.from_settings(cfg, pool=extractor.pool, working_dir=target_lrag_dir)

    if clean:
        console.print(f"[bold cyan]Cleaning existing LightRAG database at {engine.working_dir}...[/bold cyan]")
        engine.clean()

    engine.initialize()

    total_chunks_indexed = 0
    console.print(f"[bold green]Starting LightRAG indexing for {len(target_ids)} book(s)...[/bold green]")
    for bid in target_ids:
        chunks = chunk_store.load_chunks(bid)
        btitle = chunk_store.get_book_title(bid) or f"Book {bid}"
        if not chunks:
            continue
        console.print(f"  [dim cyan]Indexing {len(chunks)} chunk(s) for Book #{bid} ('{btitle}')...[/dim cyan]")
        try:
            indexed = engine.insert_chunks(chunks, book_title=btitle, book_id=bid)
            total_chunks_indexed += indexed
            console.print(f"  [bold green]✓ Indexed {indexed} chunk(s) into LightRAG.[/bold green]")
        except Exception as e:
            console.print(f"  [bold red]✗ Failed to index Book #{bid}:[/bold red] {e}")

    console.print(
        f"\n[bold green]✓ LightRAG build complete:[/bold green] "
        f"{total_chunks_indexed} total chunks indexed across {len(target_ids)} book(s) "
        f"at [cyan]{engine.working_dir}[/cyan]."
    )


@app.command("verify")
def verify_command(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    graph_file: Optional[str] = typer.Option(
        None, "--graph-file", help="Path to knowledge_graph.json (default: output_dir/knowledge_graph.json)."
    ),
    percent: Optional[float] = typer.Option(
        None, "--percent", "-p", help="Percentage of ideas to randomly verify (default: 1.0 for 1%)."
    ),
    mode: Optional[str] = typer.Option(
        None, "--mode", "-m", help="Verification mode: 'ideas', 'chunking', 'chunk', or 'cross-check' (default: 'ideas')."
    ),
    compare_graph: Optional[str] = typer.Option(
        None,
        "--compare-graph",
        help="Path to A/B test result directory OR candidate knowledge_graph.json to compare against in cross-check mode.",
    ),
    blocks: Optional[int] = typer.Option(
        None, "--blocks", help="Number of contiguous chunk blocks to audit in cross-check mode (default: 20)."
    ),
    book_id: Optional[int] = typer.Option(
        None, "--book-id", "-b", help="Specific book ID to verify (in chunking mode)."
    ),
    all_books: bool = typer.Option(
        False, "--all", help="Verify chunking across all stored books."
    ),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", help="Cosine distance threshold for semantic chunk expansion (default: 0.20)."
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", help="Embedding model name for chunking verification (default: nomic-embed-text)."
    ),
    embedding_url: Optional[str] = typer.Option(
        None, "--embedding-url", help="Ollama server URL dedicated for embeddings (e.g. http://localhost:11434)."
    ),
    chunks_dir: Optional[str] = typer.Option(
        None, "--chunks-dir", help="Directory containing stored book chunks (default: output_dir/chunks)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Dedicated LLM model for ideas verification (default: config.yaml verification.model or llm_model)."
    ),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", help="Sampling temperature for verifier model (default: 0.0)."
    ),
    max_examples: Optional[int] = typer.Option(
        None, "--max-examples", help="Maximum discrepancy examples to display (default: 20)."
    ),
    output_report: Optional[str] = typer.Option(
        None, "--output", "-o", help="Optional path to save JSON verification report (default: output_dir/verification_report.json)."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for reproducible idea/book sampling."
    ),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-t", help="Max concurrent verification worker tasks across Ollama servers."
    ),
):
    """
    Verify Knowledge Graph factual integrity, chunking correctness, or chunk Oracle alignment.
    - 'ideas' mode: randomly samples X% (default: 1%) of ideas from knowledge_graph.json,
      retrieves their assigned chunks, audits whether each chunk contains or supports the idea,
      and reports stats and discrepancy examples.
    - 'chunking' mode: verifies semantic chunk coherence using incremental sentence expansion
      distances (smart chunking). Tests whether adding each subsequent sentence maintains distance
      within accepted thresholds, shows distance distribution across chunks within the book,
      and verifies that actual stored chunks match smart semantic boundaries.
    - 'chunk' mode: randomly samples X% of chunks from knowledge_graph.json, re-extracts ideas
      using the Oracle verifier model, aligns ideas against the original run extractions via
      the EntityDeduplicator, and computes Success Rate (matched / Oracle) and Fail Rate (missed / original).
    """
    graph_file = _resolve_opt(graph_file)
    compare_graph = _resolve_opt(compare_graph)
    blocks = _resolve_opt(blocks)
    percent = _resolve_opt(percent)
    mode = _resolve_opt(mode)
    book_id = _resolve_opt(book_id)
    all_books = _resolve_opt(all_books, False)
    threshold = _resolve_opt(threshold)
    chunks_dir = _resolve_opt(chunks_dir)
    model = _resolve_opt(model)
    temperature = _resolve_opt(temperature)
    max_examples = _resolve_opt(max_examples)
    output_report = _resolve_opt(output_report)
    seed = _resolve_opt(seed)
    max_tasks = _resolve_opt(max_tasks)

    cfg = _get_effective_settings(
        config_path=config_path,
        embedding_model=embedding_model,
        embedding_url=embedding_url,
    )
    effective_mode = (mode or cfg.verification.mode).strip().lower()

    if effective_mode in ("chunking", "chunks"):
        target_chunks_dir = Path(chunks_dir).expanduser().resolve() if chunks_dir else cfg.resolved_chunks_dir
        chunk_store = ChunkStore(target_chunks_dir)
        stored_ids = chunk_store.list_stored_book_ids()
        if not stored_ids:
            console.print(
                f"[bold red]No stored book chunks found in {target_chunks_dir}[/bold red]\n"
                f"[yellow]Please run 'bookeeper build-graph' first to generate chunks.[/yellow]"
            )
            raise typer.Exit(1)

        effective_threshold = threshold if threshold is not None else cfg.verification.distance_threshold
        effective_emb_model = embedding_model or cfg.embedding_model
        effective_max_ex = cfg.verification.max_examples if max_examples is None else max_examples

        extractor = KnowledgeExtractor.from_settings(cfg)
        pool = extractor.pool
        num_servers = len(pool.alive_nodes) or len(pool.nodes)

        # Determine target book(s)
        if book_id is not None:
            if book_id not in stored_ids:
                console.print(f"[bold red]Book #{book_id} not found in chunks directory. Available books: {stored_ids}[/bold red]")
                raise typer.Exit(1)
            selected_desc = f"Book #{book_id} (Specified via CLI)"
            target_bid = book_id
        elif all_books:
            selected_desc = f"All {len(stored_ids)} Stored Books"
            target_bid = None
        else:
            rng = random.Random(seed) if seed is not None else random.Random()
            picked_id = rng.choice(stored_ids)
            picked_title = chunk_store.get_book_title(picked_id) or f"Book #{picked_id}"
            selected_desc = f"Book #{picked_id} - '{picked_title}' (Random Selection from {len(stored_ids)} books)"
            target_bid = picked_id

        console.print(
            Panel.fit(
                f"[bold]Chunks Directory:[/bold] {target_chunks_dir}\n"
                f"[bold]Total Stored Books:[/bold] {len(stored_ids)}\n\n"
                f"[bold cyan]Verification Mode:[/bold cyan] CHUNKING (Smart Incremental Expansion)\n"
                f"[bold cyan]Target Selection:[/bold cyan] {selected_desc}\n"
                f"[bold cyan]Distance Threshold:[/bold cyan] [bold green]{effective_threshold:.3f}[/bold green]\n"
                f"[bold cyan]Embedding Model:[/bold cyan] [bold magenta]{effective_emb_model}[/bold magenta]",
                title="Smart Chunking Verification Plan",
            )
        )

        embeddings = FailoverOllamaEmbeddings.from_settings(cfg, model=effective_emb_model)

        t_chunk_start = time.perf_counter()
        coherent_cnt = 0
        divergent_cnt = 0

        chunk_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Total Progress:[/bold cyan]"),
            BarColumn(bar_width=25),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[stats_line]}"),
            console=console,
        )
        tot_chunk_task = chunk_progress.add_task("chunk_total", total=100, completed=0, stats_line="")
        cur_chunk_line = Text.from_markup("[bold blue]Current Chunk:[/bold blue] Initializing chunk auditor...")

        with Live(Group(cur_chunk_line, chunk_progress), console=console, refresh_per_second=10) as live:
            def _on_chunk_progress(completed: int, total: int, chunk_id: str, is_coh: bool = True, chk: Any = None):
                nonlocal coherent_cnt, divergent_cnt
                if is_coh:
                    coherent_cnt += 1
                    status_badge = "[bold green]✓ Coherent[/bold green]"
                else:
                    divergent_cnt += 1
                    status_badge = "[bold red]✗ Divergent[/bold red]"

                book_part = f" | '{chk.book_title[:20]}'" if chk and getattr(chk, "book_title", None) else ""
                sec_part = f" ({chk.section_title[:15]})" if chk and getattr(chk, "section_title", None) else ""
                cur_text = f"[bold blue]Current Chunk:[/bold blue] '{chunk_id[:24]}'{book_part}{sec_part} | {status_badge}"

                elapsed = time.perf_counter() - t_chunk_start
                rate = (completed / elapsed) if elapsed > 0 else 0.0
                if completed > 0 and total > completed and rate > 0:
                    rem_sec = (total - completed) / rate
                    eta_str = format_eta_min_sec(rem_sec)
                elif completed >= total:
                    eta_str = "0m 00s"
                else:
                    eta_str = "--m --s"

                rate_str = f"{rate:.2f} chunk/s" if rate > 0 else "-- chunk/s"
                stats_str = (
                    f" | [bold yellow]{rate_str}[/bold yellow] | ETA: {eta_str} | "
                    f"[bold green]✓ {coherent_cnt} coherent[/bold green] | "
                    f"[bold red]✗ {divergent_cnt} divergent[/bold red]"
                )

                chunk_progress.update(tot_chunk_task, completed=completed, total=total, stats_line=stats_str)
                live.update(Group(Text.from_markup(cur_text), chunk_progress))

            report = verify_chunking(
                chunk_store=chunk_store,
                embeddings=embeddings,
                distance_threshold=effective_threshold,
                book_id=target_bid,
                all_books=all_books,
                seed=seed,
                max_examples=effective_max_ex,
                progress_callback=_on_chunk_progress,
            )

        # 1. Summary Metrics Table
        st = report.stats
        summary_table = Table(title=f"Chunking Verification Results - {st.book_title} (Book #{st.book_id})", border_style="cyan")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", justify="right", style="cyan")

        summary_table.add_row("Total Chunks Audited", str(st.total_chunks))
        summary_table.add_row("Total Sentences Evaluated", str(st.total_sentences))
        summary_table.add_row("Sentence Expansion Transitions", str(st.total_transitions))
        summary_table.add_row("Distance Threshold", f"{st.distance_threshold:.3f}")
        summary_table.add_section()

        coh_color = "green" if st.coherence_rate >= 80.0 else "yellow"
        summary_table.add_row(
            "Semantically Coherent Chunks (Good)",
            f"[bold {coh_color}]{st.coherent_chunks} ({st.coherence_rate}%)[/bold {coh_color}]",
        )
        div_color = "red" if st.divergent_chunks > 0 else "green"
        summary_table.add_row(
            "Internal Drift Discrepancies (Failed)",
            f"[bold {div_color}]{st.divergent_chunks} ({round(100.0 - st.coherence_rate, 2)}%)[/bold {div_color}]",
        )
        summary_table.add_row("Smart Boundary Match Rate", f"{st.boundary_match_rate}%")
        summary_table.add_section()

        summary_table.add_row("Min Distance", f"{st.min_distance:.4f}")
        summary_table.add_row("Mean Distance", f"{st.mean_distance:.4f}")
        summary_table.add_row("Median (P50) Distance", f"{st.median_distance:.4f}")
        summary_table.add_row("P80 Distance", f"{st.p80_distance:.4f}")
        summary_table.add_row("P95 Distance", f"{st.p95_distance:.4f}")
        summary_table.add_row("Max Distance", f"{st.max_distance:.4f}")
        summary_table.add_row("Distance Std Dev", f"{st.std_dev_distance:.4f}")
        summary_table.add_row("Duration", f"{st.total_duration_seconds:.2f}s")
        emb_st = getattr(embeddings, "embedding_stats", {})
        if emb_st.get("total_texts", 0) > 0:
            summary_table.add_row(
                "Embedding Speed",
                f"{emb_st['speed_texts_per_sec']:.1f} sent/s ({emb_st['speed_chars_per_sec'] / 1024:.1f} KB/s)",
            )

        console.print(summary_table)

        # 2. Distance Distribution Table (Histogram)
        dist_table = Table(title="Semantic Distance Distribution (Incremental Sentence Expansions)", border_style="magenta")
        dist_table.add_column("Distance Range", style="bold", justify="center")
        dist_table.add_column("Coherence Level", style="cyan")
        dist_table.add_column("Transitions", justify="right", style="white")
        dist_table.add_column("% Share", justify="right", style="yellow")
        dist_table.add_column("Distribution", style="green")

        for b in st.distribution_buckets:
            dist_table.add_row(
                b.range_label,
                b.coherence_level,
                str(b.count),
                f"{b.percentage:.1f}%",
                b.bar,
            )
        console.print(dist_table)

        # 3. Discrepancy Examples Table (Up to max_examples)
        if report.discrepancies:
            disc_table = Table(
                title=f"Chunking Discrepancies ({len(report.discrepancies)} shown, max {effective_max_ex})",
                border_style="red",
            )
            disc_table.add_column("#", justify="center", width=4)
            disc_table.add_column("Chunk ID", style="bold cyan", min_width=18)
            disc_table.add_column("Type", style="yellow")
            disc_table.add_column("Max Dist / Thresh", justify="center", style="white")
            disc_table.add_column("Drift Sentence / Reason", style="white")

            for idx, d in enumerate(report.discrepancies, 1):
                dist_str = f"{d.max_distance:.3f} / {d.threshold:.3f}" if d.max_distance > 0 else "-"
                reason = d.explanation
                if d.drift_sentence:
                    reason += f"\n[dim]Drift Sentence: \"{d.drift_sentence}\"[/dim]"
                disc_table.add_row(
                    str(idx),
                    f"{d.chunk_id}\n[dim]{d.section_title}[/dim]",
                    d.discrepancy_type,
                    dist_str,
                    reason,
                )
            console.print(disc_table)
        else:
            console.print("\n[bold green]✓ Zero chunking discrepancies found! All chunks are semantically coherent and aligned.[/bold green]\n")

        # 4. Save JSON Report
        report_dest = Path(output_report).expanduser().resolve() if output_report else cfg.resolved_output_dir / "verification_report.json"
        report_dest.parent.mkdir(parents=True, exist_ok=True)
        with open(report_dest, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
        console.print(f"[dim green]✓ Full chunking verification report saved to: [bold]{report_dest}[/bold][/dim green]\n")
        _print_ollama_server_stats([pool], console=console, title="Ollama Server Chunking Verification Effort Statistics")
        return

    # Mode: 'chunk' (Oracle Idea Extraction & Alignment)
    if effective_mode in ("chunk", "oracle_chunk"):
        target_graph_file = (
            Path(graph_file).expanduser().resolve()
            if graph_file
            else cfg.resolved_output_dir / "knowledge_graph.json"
        )
        if not target_graph_file.is_file():
            console.print(
                f"[bold red]Knowledge graph file not found at {target_graph_file}[/bold red]\n"
                f"[yellow]Please run 'bookeeper build-graph' first to generate knowledge_graph.json.[/yellow]"
            )
            raise typer.Exit(1)

        store = ConceptGraphStore()
        with console.status(f"[bold cyan]Loading graph from {target_graph_file}...[/bold cyan]"):
            try:
                store.load(target_graph_file)
            except Exception as e:
                console.print(f"[bold red]Failed to load knowledge graph:[/bold red] {e}")
                raise typer.Exit(1)

        effective_percent = cfg.verification.percent if percent is None else percent
        effective_model = model or cfg.resolved_verifier_model
        effective_max_ex = cfg.verification.max_examples if max_examples is None else max_examples

        verifier_servers = cfg.resolved_verification_servers
        pool = OllamaPool(
            servers=verifier_servers,
            cooldown_seconds=cfg.failover_cooldown_seconds,
            max_tasks_per_server=1,
        )
        num_servers = len(pool.alive_nodes) or len(pool.nodes)
        pool_concurrency = max_tasks if max_tasks is not None else cfg.calculate_pool_concurrency(num_servers)

        raw_cdir = (
            chunks_dir
            or getattr(cfg, "chunks_dir", None)
            or (cfg.resolved_output_dir / "chunks" if hasattr(cfg, "resolved_output_dir") else None)
        )
        chunks_storage_dir = Path(raw_cdir).expanduser().resolve() if raw_cdir else None
        chunk_store = ChunkStore(chunks_storage_dir) if chunks_storage_dir and chunks_storage_dir.is_dir() else None

        effective_emb_model = embedding_model or cfg.embedding_model
        embeddings = FailoverOllamaEmbeddings.from_settings(cfg, model=effective_emb_model)

        all_chunks_in_graph = len([n for n, d in store.graph.nodes(data=True) if d.get("type") == "Chunk"])

        console.print(
            Panel.fit(
                f"[bold]Knowledge Graph:[/bold] {target_graph_file}\n"
                f"[bold]Total Graph Chunks:[/bold] {all_chunks_in_graph}\n\n"
                f"[bold cyan]Verification Mode:[/bold cyan] CHUNK (Oracle Idea Extraction & Alignment)\n"
                f"[bold cyan]Sample Rate:[/bold cyan] {effective_percent}% of chunks\n"
                f"[bold cyan]Oracle Model:[/bold cyan] [bold magenta]{effective_model}[/bold magenta]\n"
                f"[bold cyan]Embedding Model (Dedup):[/bold cyan] [bold magenta]{effective_emb_model}[/bold magenta]\n"
                f"[bold cyan]Concurrency:[/bold cyan] {pool_concurrency} tasks across {num_servers} Ollama server(s)",
                title="Chunk Oracle Verification Plan",
            )
        )

        with console.status(
            f"[bold blue]Preloading and warming up verifier model '{effective_model}' across Ollama pool...[/bold blue]"
        ):
            warmup_verifier = IdeaVerifier(pool=pool, model_name=effective_model)
            warmup_verifier.warmup(timeout=90)

        t_start = time.perf_counter()
        tot_matched = 0
        tot_missed = 0

        chunk_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Total Progress:[/bold cyan]"),
            BarColumn(bar_width=25),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[stats_line]}"),
            console=console,
        )
        tot_chunk_task = chunk_progress.add_task("chunk_total", total=100, completed=0, stats_line="")
        cur_chunk_line = Text.from_markup("[bold blue]Current Chunk:[/bold blue] Initializing chunk auditor...")

        with Live(Group(cur_chunk_line, chunk_progress), console=console, refresh_per_second=10) as live:

            def _on_chunk_progress(completed: int, total: int, chunk_id: str, item: Optional[ChunkAuditItem] = None):
                nonlocal tot_matched, tot_missed
                if item is not None:
                    tot_matched += item.matched_count
                    tot_missed += item.missed_count
                    if item.missed_count == 0:
                        badge = "[bold green]✓ Full Match[/bold green]"
                    elif item.matched_count > 0:
                        badge = f"[bold yellow]⚠ Partial ({item.matched_count}/{item.oracle_count})[/bold yellow]"
                    else:
                        badge = "[bold red]✗ Omission[/bold red]"

                    book_str = f" | '{item.book_title[:18]}'" if item.book_title else ""
                    sec_str = f" ({item.section_title[:16]})" if item.section_title else ""
                    cur_text = (
                        f"[bold blue]Current Chunk:[/bold blue] '{chunk_id[:24]}'{book_str}{sec_str} | {badge}"
                    )
                else:
                    cur_text = f"[bold blue]Current Chunk:[/bold blue] '{chunk_id[:28]}' | [bold yellow]Auditing...[/bold yellow]"

                elapsed = time.perf_counter() - t_start
                rate = (completed / elapsed) if elapsed > 0 else 0.0
                if completed > 0 and total > completed and rate > 0:
                    rem_sec = (total - completed) / rate
                    eta_str = format_eta_min_sec(rem_sec)
                elif completed >= total:
                    eta_str = "0m 00s"
                else:
                    eta_str = "--"

                stats_str = (
                    f"[cyan]{rate:.1f} chk/s[/cyan] | [yellow]ETA: {eta_str}[/yellow] | "
                    f"[bold green]✓ {tot_matched} matched[/bold green] | "
                    f"[bold red]✗ {tot_missed} missed[/bold red]"
                )
                chunk_progress.update(tot_chunk_task, completed=completed, total=total, stats_line=stats_str)
                live.update(Group(Text.from_markup(cur_text), chunk_progress))

            report = verify_chunks(
                store=store,
                pool=pool,
                model_name=effective_model,
                percent=effective_percent,
                chunk_store=chunk_store,
                embeddings=embeddings,
                max_examples=effective_max_ex,
                seed=seed,
                concurrency=pool_concurrency,
                progress_callback=_on_chunk_progress,
            )

        # 1. Summary Metrics Table
        st = report.stats
        summary_table = Table(
            title="Chunk Verification Results Summary (Oracle Alignment)",
            border_style="cyan",
        )
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", justify="right", style="cyan")

        summary_table.add_row("Total Chunks in Graph", str(st.total_chunks_in_graph))
        summary_table.add_row("Audited Chunks Sampled", f"{st.sampled_chunks_count} ({st.sample_percentage:.1f}%)")
        summary_table.add_row("Original Model Extracted Ideas", str(st.total_original_ideas))
        summary_table.add_row("Oracle Model Discovered Ideas", str(st.total_oracle_ideas))
        summary_table.add_section()

        succ_color = "green" if st.overall_success_rate >= 80.0 else "yellow"
        summary_table.add_row(
            "Matched Ideas (Verified in Both)",
            f"[bold {succ_color}]{st.total_matched_ideas} ({st.overall_success_rate}%)[/bold {succ_color}]",
        )
        fail_color = "red" if st.overall_fail_rate > 15.0 else "green"
        summary_table.add_row(
            "Missed Ideas (Omitted by Candidate)",
            f"[bold {fail_color}]{st.total_missed_ideas} ({st.overall_fail_rate}%)[/bold {fail_color}]",
        )
        summary_table.add_row("Extra Original Ideas (Candidate Exclusive)", str(st.total_extra_original_ideas))
        summary_table.add_section()

        summary_table.add_row("Chunks with 100% Idea Match", f"{st.chunks_with_perfect_match} / {st.sampled_chunks_count}")
        summary_table.add_row("Chunks with Omissions", f"{st.chunks_with_omissions} / {st.sampled_chunks_count}")
        summary_table.add_row("Total Audit Duration", f"{st.total_duration_seconds:.2f}s")
        console.print(summary_table)

        # 2. Omissions Discrepancy Table
        if report.omission_examples:
            disc_table = Table(
                title=f"Top Idea Omission Discrepancies (Missed by Candidate, Found by Oracle '{st.model_name}')",
                border_style="red",
            )
            disc_table.add_column("#", justify="center", style="dim")
            disc_table.add_column("Chunk ID / Section", style="cyan")
            disc_table.add_column("Missed Oracle Ideas", style="bold red")
            disc_table.add_column("Original Ideas", style="dim")
            disc_table.add_column("Fail Rate", justify="right", style="yellow")

            for idx, item in enumerate(report.omission_examples, 1):
                missed_str = "\n".join(f"• {name}" for name in item.missed_oracle_ideas[:3])
                if len(item.missed_oracle_ideas) > 3:
                    missed_str += f"\n... +{len(item.missed_oracle_ideas) - 3} more"

                orig_str = "\n".join(f"• {name}" for name in item.original_ideas[:3]) if item.original_ideas else "(None)"
                if len(item.original_ideas) > 3:
                    orig_str += f"\n... +{len(item.original_ideas) - 3} more"

                disc_table.add_row(
                    str(idx),
                    f"{item.chunk_id}\n[dim]{item.book_title} ({item.section_title})[/dim]",
                    missed_str,
                    orig_str,
                    f"{item.fail_rate:.1f}%",
                )
            console.print(disc_table)
        else:
            console.print(
                "\n[bold green]✓ Zero chunk omissions found! All Oracle ideas were successfully matched in the candidate run.[/bold green]\n"
            )

        # 3. Save JSON Report
        report_dest = (
            Path(output_report).expanduser().resolve()
            if output_report
            else cfg.resolved_output_dir / "verification_report.json"
        )
        report_dest.parent.mkdir(parents=True, exist_ok=True)
        with open(report_dest, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
        console.print(
            f"[dim green]✓ Full chunk verification report saved to: [bold]{report_dest}[/bold][/dim green]\n"
        )
        _print_ollama_server_stats([pool], console=console, title="Ollama Server Chunk Verification Effort Statistics")
        return

    # Mode: 'cross-check' (Dual Knowledge Base Seam & Recall Comparative Audit)
    if effective_mode in ("cross-check", "cross_check"):
        # Resolve candidate path helper: check direct, CWD/ab_test_runs, repo/ab_test_runs
        def _resolve_candidate_path(p_str: Optional[str]) -> Optional[Path]:
            if not p_str:
                return None
            cand = Path(p_str).expanduser()
            if cand.exists():
                return cand.resolve()
            alt1 = Path.cwd() / "ab_test_runs" / p_str
            if alt1.exists():
                return alt1.resolve()
            if cfg and getattr(cfg, "repo_dir", None):
                alt2 = Path(cfg.repo_dir) / "ab_test_runs" / p_str
                if alt2.exists():
                    return alt2.resolve()
            return cand.resolve()

        def _discover_graphs_from_run_dir(d: Path) -> Optional[Tuple[Path, Path, str, str]]:
            """Locate branch A and branch B knowledge_graph.json within an A/B test run directory."""
            if not d.is_dir():
                return None
            subdirs = sorted([p for p in d.iterdir() if p.is_dir()])
            cand_a = [p for p in subdirs if p.name.startswith("branch_A")]
            cand_b = [p for p in subdirs if p.name.startswith("branch_B")]
            if cand_a and cand_b:
                kg_a = cand_a[0] / "knowledge_graph.json"
                kg_b = cand_b[0] / "knowledge_graph.json"
                if kg_a.is_file() and kg_b.is_file():
                    name_a = cand_a[0].name.replace("branch_A_", "").replace("branch_A", "").strip("_") or cand_a[0].name
                    name_b = cand_b[0].name.replace("branch_B_", "").replace("branch_B", "").strip("_") or cand_b[0].name
                    return kg_a, kg_b, name_a, name_b

            # Fallback: check any two subdirectories containing knowledge_graph.json
            with_kg = [p for p in subdirs if (p / "knowledge_graph.json").is_file()]
            if len(with_kg) >= 2:
                kg_a = with_kg[0] / "knowledge_graph.json"
                kg_b = with_kg[1] / "knowledge_graph.json"
                name_a = with_kg[0].name.strip("_")
                name_b = with_kg[1].name.strip("_")
                return kg_a, kg_b, name_a, name_b

            return None

        cand_path_a = _resolve_candidate_path(graph_file)
        cand_path_b = _resolve_candidate_path(compare_graph)

        target_graph_a: Optional[Path] = None
        target_graph_b: Optional[Path] = None
        branch_a_label = "Baseline (A)"
        branch_b_label = "Candidate (B)"

        # 1. If compare_graph points to an A/B test result directory containing both branches:
        if cand_path_b and cand_path_b.is_dir():
            run_graphs = _discover_graphs_from_run_dir(cand_path_b)
            if run_graphs:
                target_graph_a, target_graph_b, branch_a_label, branch_b_label = run_graphs

        # 2. If graph_file points to an A/B test result directory containing both branches:
        if not target_graph_a and cand_path_a and cand_path_a.is_dir():
            run_graphs = _discover_graphs_from_run_dir(cand_path_a)
            if run_graphs:
                target_graph_a, target_graph_b, branch_a_label, branch_b_label = run_graphs

        # 3. If individual branch folders or files were passed:
        if not target_graph_a:
            if cand_path_a:
                if cand_path_a.is_file():
                    target_graph_a = cand_path_a
                elif cand_path_a.is_dir() and (cand_path_a / "knowledge_graph.json").is_file():
                    target_graph_a = cand_path_a / "knowledge_graph.json"
            else:
                default_kg = cfg.resolved_output_dir / "knowledge_graph.json"
                if default_kg.is_file():
                    target_graph_a = default_kg

        if not target_graph_b:
            if cand_path_b:
                if cand_path_b.is_file():
                    target_graph_b = cand_path_b
                elif cand_path_b.is_dir() and (cand_path_b / "knowledge_graph.json").is_file():
                    target_graph_b = cand_path_b / "knowledge_graph.json"

        if not target_graph_a or not target_graph_a.is_file():
            console.print(
                f"[bold red]Primary knowledge graph file not found at '{graph_file or cfg.resolved_output_dir}'.[/bold red]\n"
                f"[yellow]Specify --graph-file <path_to_baseline_kg.json> or an A/B test result directory via --compare-graph <path_to_ab_test_result>.[/yellow]"
            )
            raise typer.Exit(1)

        if not target_graph_b or not target_graph_b.is_file():
            console.print(
                f"[bold red]Comparison knowledge graph or A/B test run directory (--compare-graph) not found at '{compare_graph}'.[/bold red]\n"
                f"[yellow]Specify --compare-graph <path_to_ab_test_result_or_candidate_kg.json> to run cross-check mode.[/yellow]"
            )
            raise typer.Exit(1)

        if branch_a_label == "Baseline (A)":
            branch_a_label = target_graph_a.parent.name.replace("branch_A_", "").strip("_") or target_graph_a.name
        if branch_b_label == "Candidate (B)":
            branch_b_label = target_graph_b.parent.name.replace("branch_B_", "").strip("_") or target_graph_b.name

        store_a = ConceptGraphStore()
        store_b = ConceptGraphStore()
        with console.status(f"[bold cyan]Loading Graph A from {target_graph_a}...[/bold cyan]"):
            store_a.load(target_graph_a)
        with console.status(f"[bold cyan]Loading Graph B from {target_graph_b}...[/bold cyan]"):
            store_b.load(target_graph_b)

        effective_blocks = blocks if blocks is not None else 20
        effective_model = model or cfg.resolved_verifier_model
        effective_embedding_model = embedding_model or getattr(cfg, "embedding_model", "nomic-embed-text")

        verifier_servers = cfg.resolved_verification_servers
        pool = OllamaPool(
            servers=verifier_servers,
            cooldown_seconds=cfg.failover_cooldown_seconds,
            max_tasks_per_server=1,
        )
        num_servers = len(pool.alive_nodes) or len(pool.nodes)
        pool_concurrency = max_tasks if max_tasks is not None else cfg.calculate_pool_concurrency(num_servers)

        chunks_a_cnt = len([n for n, d in store_a.graph.nodes(data=True) if d.get("type") == "Chunk"])
        chunks_b_cnt = len([n for n, d in store_b.graph.nodes(data=True) if d.get("type") == "Chunk"])

        console.print(
            Panel.fit(
                f"[bold]Baseline Graph A ({branch_a_label}):[/bold] {target_graph_a} ({chunks_a_cnt} chunks)\n"
                f"[bold]Candidate Graph B ({branch_b_label}):[/bold] {target_graph_b} ({chunks_b_cnt} chunks)\n\n"
                f"[bold cyan]Verification Mode:[/bold cyan] CROSS-CHECK (Dual Seam & Recall Comparative Audit)\n"
                f"[bold cyan]Contiguous Blocks to Audit:[/bold cyan] {effective_blocks}\n"
                f"[bold cyan]Oracle Model:[/bold cyan] [bold magenta]{effective_model}[/bold magenta]\n"
                f"[bold cyan]Disproportion Metric:[/bold cyan] sum(block_A.len) / sum(block_A ∪ block_B.len)\n"
                f"[bold cyan]Concurrency:[/bold cyan] {pool_concurrency} tasks across {num_servers} Ollama server(s)",
                title="Cross-Check Verification Plan",
            )
        )

        with console.status(
            f"[bold blue]Preloading and warming up verifier model '{effective_model}' across Ollama pool...[/bold blue]"
        ):
            warmup_verifier = IdeaVerifier(pool=pool, model_name=effective_model)
            warmup_verifier.warmup(timeout=90)

        t_start = time.perf_counter()
        tot_delta = 0.0
        tot_trunc = 0.0

        cc_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Cross-Check Progress:[/bold cyan]"),
            BarColumn(bar_width=25),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[stats_line]}"),
            console=console,
        )
        tot_cc_task = cc_progress.add_task("cc_total", total=effective_blocks, completed=0, stats_line="")
        cur_cc_line = Text.from_markup("[bold blue]Current Block:[/bold blue] Initializing cross-check auditor...")

        with Live(Group(cur_cc_line, cc_progress), console=console, refresh_per_second=10) as live:
            def _on_cc_progress(completed: int, total: int, block_info: str, item: Optional[CrossCheckBlockResult] = None):
                nonlocal tot_delta, tot_trunc
                if item is not None:
                    tot_delta += item.delta_recall
                    tot_trunc += item.new_system.truncation_rate
                    avg_d = tot_delta / completed if completed > 0 else 0.0
                    cur_text = (
                        f"[bold blue]Current Block:[/bold blue] {block_info} | "
                        f"ΔRecall: [{'green' if avg_d >= 0 else 'red'}]{avg_d * 100:+.1f}% pts[/] | "
                        f"Disprop: [cyan]{item.chunk_disproportion:.2f}[/cyan]"
                    )
                else:
                    cur_text = f"[bold blue]Current Block:[/bold blue] {block_info} | [bold yellow]Auditing...[/bold yellow]"

                elapsed = time.perf_counter() - t_start
                rate = (completed / elapsed) if elapsed > 0 else 0.0
                if completed > 0 and total > completed and rate > 0:
                    rem_sec = (total - completed) / rate
                    eta_str = f"{int(rem_sec // 60)}m {int(rem_sec % 60):02d}s" if rem_sec >= 60 else f"{rem_sec:.0f}s"
                else:
                    eta_str = "--"

                stats_str = f"[cyan]{rate:.1f} blk/s[/cyan] | [yellow]ETA: {eta_str}[/yellow]"
                cc_progress.update(tot_cc_task, completed=completed, total=total, stats_line=stats_str)
                live.update(Group(Text.from_markup(cur_text), cc_progress))

            report = run_cross_check(
                store_a=store_a,
                store_b=store_b,
                num_blocks=effective_blocks,
                pool=pool,
                model_name=effective_model,
                embedding_model=effective_embedding_model,
                branch_a_name=branch_a_label,
                branch_b_name=branch_b_label,
                seed=seed,
                progress_callback=_on_cc_progress,
            )

        # 1. Summary Metrics Table
        sm = report.summary
        summary_table = Table(title="Cross-Check Verification Summary & Decision Gates", border_style="cyan")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Baseline (Old / A)", justify="right", style="cyan")
        summary_table.add_column("Candidate (New / B)", justify="right", style="magenta")
        summary_table.add_column("Delta (B - A)", justify="right")

        summary_table.add_row("Total Contiguous Blocks Audited", str(sm.total_samples), str(sm.total_samples), "-")
        summary_table.add_row(
            "Mean Oracle Recall (%)",
            f"{sm.mean_old_recall * 100:.2f}%",
            f"{sm.mean_new_recall * 100:.2f}%",
            f"[{'green' if sm.mean_delta_recall >= -0.03 else 'red'}]{sm.mean_delta_recall * 100:+.2f}% pts[/]",
        )
        summary_table.add_row(
            "Aggregate Fail Ratio [sum(Failed)/sum(Total Ideas)] (%)",
            f"{sm.mean_old_fail_ratio * 100:.2f}%",
            f"{sm.mean_new_fail_ratio * 100:.2f}%",
            f"[{'green' if sm.mean_delta_fail_ratio <= 0 else 'red'}]{sm.mean_delta_fail_ratio * 100:+.2f}% pts[/]",
        )
        summary_table.add_row(
            "Mean Grounded Precision (%)",
            f"{sm.mean_old_precision * 100:.2f}%",
            f"{sm.mean_new_precision * 100:.2f}%",
            f"{(sm.mean_new_precision - sm.mean_old_precision) * 100:+.2f}% pts",
        )
        summary_table.add_row(
            "Boundary Truncation Artifact Rate (%)",
            f"{sm.mean_old_truncation_rate * 100:.2f}%",
            f"[{'green' if sm.mean_new_truncation_rate < 0.02 else 'red'}]{sm.mean_new_truncation_rate * 100:.2f}%[/]",
            f"{(sm.mean_new_truncation_rate - sm.mean_old_truncation_rate) * 100:+.2f}% pts",
        )
        summary_table.add_section()
        summary_table.add_row(
            "Mean Chunk Disproportion [sum(A)/sum(A∪B)]",
            "-",
            f"[bold cyan]{sm.mean_chunk_disproportion:.4f}[/bold cyan]",
            "-",
        )
        summary_table.add_row(
            "Decision Gating Checklist",
            "-",
            "[bold green]PASS[/bold green]" if sm.decision_pass else "[bold red]FAIL[/bold red]",
            f"[dim]{sm.pass_reason}[/dim]",
        )
        console.print(summary_table)

        # 2. Block Sample Detail Breakdown Table
        if report.samples:
            sample_table = Table(title="Audited Block Samples Breakdown (Top 10)", border_style="yellow")
            sample_table.add_column("#", justify="right", style="bold")
            sample_table.add_column("Book / Section", style="dim")
            sample_table.add_column("Passage Len", justify="right")
            sample_table.add_column("Disproportion", justify="right", style="cyan")
            sample_table.add_column("Recall A", justify="right")
            sample_table.add_column("Recall B", justify="right")
            sample_table.add_column("ΔRecall", justify="right")
            sample_table.add_column("Fail A", justify="right")
            sample_table.add_column("Fail B", justify="right")
            sample_table.add_column("ΔFail", justify="right")
            sample_table.add_column("B Truncation", justify="right")

            for s in report.samples[:10]:
                d_color = "green" if s.delta_recall >= 0 else ("yellow" if s.delta_recall >= -0.03 else "red")
                f_color = "green" if s.delta_fail_ratio <= 0 else "red"
                t_color = "green" if s.new_system.truncation_rate == 0 else "red"
                sample_table.add_row(
                    str(s.sample_index),
                    f"{s.book_title[:16]} ({s.section_title[:14]})",
                    f"{s.passage_length_chars} chars",
                    f"{s.chunk_disproportion:.3f}",
                    f"{s.old_system.oracle_recall * 100:.1f}%",
                    f"{s.new_system.oracle_recall * 100:.1f}%",
                    f"[{d_color}]{s.delta_recall * 100:+.1f}%[/]",
                    f"{s.old_system.fail_ratio * 100:.1f}%",
                    f"{s.new_system.fail_ratio * 100:.1f}%",
                    f"[{f_color}]{s.delta_fail_ratio * 100:+.1f}%[/]",
                    f"[{t_color}]{s.new_system.truncation_rate * 100:.1f}%[/]",
                )
            console.print(sample_table)

        # 3. Save JSON Report
        report_dest = (
            Path(output_report).expanduser().resolve()
            if output_report
            else cfg.resolved_output_dir / "cross_check_report.json"
        )
        report_dest.parent.mkdir(parents=True, exist_ok=True)
        with open(report_dest, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
        console.print(
            f"[dim green]✓ Full cross-check verification report saved to: [bold]{report_dest}[/bold][/dim green]\n"
        )
        _print_ollama_server_stats([pool], console=console, title="Ollama Server Cross-Check Verification Effort Statistics")
        return

    # Mode: 'ideas'
    target_graph_file = Path(graph_file).expanduser().resolve() if graph_file else cfg.resolved_output_dir / "knowledge_graph.json"
    if not target_graph_file.is_file():
        console.print(
            f"[bold red]Knowledge graph file not found at {target_graph_file}[/bold red]\n"
            f"[yellow]Please run 'bookeeper build-graph' first to generate knowledge_graph.json.[/yellow]"
        )
        raise typer.Exit(1)

    store = ConceptGraphStore()
    with console.status(f"[bold cyan]Loading graph from {target_graph_file}...[/bold cyan]"):
        try:
            store.load(target_graph_file)
        except Exception as e:
            console.print(f"[bold red]Failed to load knowledge graph:[/bold red] {e}")
            raise typer.Exit(1)

    effective_percent = cfg.verification.percent if percent is None else percent
    effective_model = model or cfg.resolved_verifier_model
    effective_temp = cfg.verification.temperature if temperature is None else temperature
    effective_max_ex = cfg.verification.max_examples if max_examples is None else max_examples

    verifier_servers = cfg.resolved_verification_servers
    pool = OllamaPool(
        servers=verifier_servers,
        cooldown_seconds=cfg.failover_cooldown_seconds,
        max_tasks_per_server=1,
    )
    num_servers = len(pool.alive_nodes) or len(pool.nodes)
    pool_concurrency = max_tasks if max_tasks is not None else cfg.calculate_pool_concurrency(num_servers)

    total_ideas = len([n for n, d in store.graph.nodes(data=True) if d.get("type") == "Concept"])
    console.print(
        Panel.fit(
            f"[bold]Knowledge Graph File:[/bold] {target_graph_file}\n"
            f"[bold]Total Ideas in Graph:[/bold] {total_ideas}\n\n"
            f"[bold cyan]Verification Mode:[/bold cyan] {effective_mode.upper()}\n"
            f"[bold cyan]Sample Percentage:[/bold cyan] {effective_percent}%\n"
            f"[bold cyan]Dedicated Verifier Model:[/bold cyan] [bold green]{effective_model}[/bold green]\n"
            f"[bold cyan]Concurrency:[/bold cyan] {pool_concurrency} tasks across {num_servers} Ollama server(s)",
            title="Knowledge Graph Verification Plan",
        )
    )

    with console.status(f"[bold blue]Preloading and warming up verifier model '{effective_model}' across Ollama pool...[/bold blue]"):
        warmup_verifier = IdeaVerifier(pool=pool, model_name=effective_model)
        warmup_verifier.warmup(timeout=90)

    t_verify_start = time.perf_counter()
    verified_cnt = 0
    discrepancy_cnt = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Total Progress:[/bold cyan]"),
        BarColumn(bar_width=25),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[stats_line]}"),
        console=console,
    )
    tot_task = progress.add_task("total", total=100, completed=0, stats_line="")
    cur_line = Text.from_markup("[bold blue]Current Idea:[/bold blue] Initializing verifier pool...")

    with Live(Group(cur_line, progress), console=console, refresh_per_second=10) as live:
        def _on_progress(completed: int, total: int, idea_name: str, item: Optional[VerificationItem] = None):
            nonlocal verified_cnt, discrepancy_cnt
            if item is not None:
                if item.is_supported:
                    verified_cnt += 1
                    status_badge = "[bold green]✓ Supported[/bold green]"
                else:
                    discrepancy_cnt += 1
                    status_badge = "[bold red]✗ Discrepancy[/bold red]"

                server_str = f" [dim]via {item.server_node}[/dim]" if item.server_node else ""
                short_idea = (item.idea_name[:24] + "...") if len(item.idea_name) > 27 else item.idea_name
                short_book = (item.book_title[:20] + "...") if len(item.book_title) > 23 else item.book_title
                cur_text = (
                    f"[bold blue]Current Idea:[/bold blue] '{short_idea}' | '{short_book}' "
                    f"({item.chunk_id}) | {status_badge}{server_str}"
                )
            else:
                short_idea = (idea_name[:28] + "...") if len(idea_name) > 31 else idea_name
                cur_text = f"[bold blue]Current Idea:[/bold blue] '{short_idea}' | [bold yellow]Auditing...[/bold yellow]"

            elapsed = time.perf_counter() - t_verify_start
            rate = (completed / elapsed) if elapsed > 0 else 0.0
            if completed > 0 and total > completed and rate > 0:
                rem_sec = (total - completed) / rate
                eta_str = format_eta_min_sec(rem_sec)
            elif completed >= total:
                eta_str = "0m 00s"
            else:
                eta_str = "--m --s"

            rate_str = f"{rate:.2f} task/s" if rate > 0 else "-- task/s"
            stats_str = (
                f" | [bold yellow]{rate_str}[/bold yellow] | ETA: {eta_str} | "
                f"[bold green]✓ {verified_cnt} passed[/bold green] | "
                f"[bold red]✗ {discrepancy_cnt} failed[/bold red]"
            )

            progress.update(tot_task, completed=completed, total=total, stats_line=stats_str)
            live.update(Group(Text.from_markup(cur_text), progress))

        raw_cdir = chunks_dir or getattr(cfg, "chunks_dir", None) or (cfg.resolved_output_dir / "chunks" if hasattr(cfg, "resolved_output_dir") else None)
        chunks_storage_dir = Path(raw_cdir).expanduser().resolve() if raw_cdir else None
        chunk_store = ChunkStore(chunks_storage_dir) if chunks_storage_dir and chunks_storage_dir.is_dir() else None

        report = verify_graph(
            store=store,
            pool=pool,
            model_name=effective_model,
            percent=effective_percent,
            mode=effective_mode,
            max_examples=effective_max_ex,
            temperature=effective_temp,
            seed=seed,
            concurrency=pool_concurrency,
            progress_callback=_on_progress,
            chunk_store=chunk_store,
        )

    # 1. Summary Metrics Table
    st = report.stats
    summary_table = Table(title="Verification Results Summary", border_style="cyan")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right", style="cyan")

    summary_table.add_row("Total Ideas in Graph", str(st.total_ideas_in_graph))
    summary_table.add_row("Ideas with Supporting Chunks", str(st.candidate_ideas_with_chunks))
    summary_table.add_row("Sampled Ideas Verified", f"{st.sampled_ideas} ({st.sample_percentage}%)")
    summary_table.add_row("Total Idea-Chunk Pairs Evaluated", str(st.total_evaluations))
    summary_table.add_section()
    summary_table.add_row(
        "Verified / Supported (Good)",
        f"[bold green]{st.verified_count} ({st.verified_percentage}%)[/bold green]",
    )
    disc_color = "red" if st.discrepancy_count > 0 else "green"
    summary_table.add_row(
        "Discrepancies / Unsupported (Failed)",
        f"[bold {disc_color}]{st.discrepancy_count} ({st.discrepancy_percentage}%)[/bold {disc_color}]",
    )
    summary_table.add_row("Verification Duration", f"{st.total_duration_seconds:.2f}s")

    console.print(summary_table)

    # 2. Discrepancy Examples Table (Up to 20 examples)
    if report.discrepancies:
        disc_table = Table(
            title=f"Discrepancy Examples ({len(report.discrepancies)} shown, max {effective_max_ex})",
            border_style="red",
        )
        disc_table.add_column("#", justify="center", width=4)
        disc_table.add_column("Idea (Weight)", style="bold cyan", min_width=18)
        disc_table.add_column("Source Location", style="white", min_width=20)
        disc_table.add_column("Explanation / Reason", style="yellow")

        for idx, d in enumerate(report.discrepancies, 1):
            loc_str = f"{d.book_title}\n[dim]{d.section_title} ({d.chunk_id})[/dim]"
            disc_table.add_row(
                str(idx),
                f"{d.idea_name} (W:{d.idea_weight})",
                loc_str,
                d.explanation,
            )
        console.print(disc_table)
    else:
        console.print("\n[bold green]✓ Zero discrepancies found! All sampled idea-chunk pairs are factual and supported.[/bold green]\n")

    # 3. Save JSON report
    report_dest = Path(output_report).expanduser().resolve() if output_report else cfg.resolved_output_dir / "verification_report.json"
    report_dest.parent.mkdir(parents=True, exist_ok=True)
    with open(report_dest, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
    console.print(f"[dim green]✓ Full verification report saved to: [bold]{report_dest}[/bold][/dim green]\n")
    _print_ollama_server_stats([pool], console=console, title="Ollama Server Verification Effort Statistics")


@app.command("remove-book")
def remove_book_cmd(
    book_id: int = typer.Argument(..., help="Calibre ID of the book to remove from knowledge graph."),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    export_obsidian: Optional[str] = typer.Option(
        None,
        "--export-obsidian",
        help="Custom export directory for Obsidian Markdown vault to refresh after removal.",
    ),
):
    """
    Remove an unprocessed or unwanted book from the Knowledge Graph.
    Purges the book node, its sections, chunks, and any orphan concepts that only belonged to this book.
    Also clears its completed state from the progress checkpoint so it can be re-indexed cleanly.
    """
    cfg = get_settings(config)
    output_dir = cfg.resolved_output_dir
    graph_file = output_dir / "knowledge_graph.json"
    tracker = ProgressTracker(cfg.resolved_state_file)

    if not graph_file.is_file():
        console.print(f"[bold yellow]Knowledge graph file not found at {graph_file}.[/bold yellow]")
        raise typer.Exit(1)

    store = ConceptGraphStore()
    store.load(graph_file)

    stats = store.remove_book(book_id)
    if stats["total_nodes_removed"] > 0:
        store.save(graph_file)
        tracker.remove_book("build_graph", book_id)

        vault_dest = Path(export_obsidian) if export_obsidian else output_dir / "obsidian_vault"
        graphml_dest = output_dir / "knowledge_graph.graphml"
        obs = ObsidianExporter(vault_dest)
        obs.export(store, clean=True)
        gml = GraphMLExporter(graphml_dest)
        gml.export(store)

        console.print(
            f"[bold green]✓ Successfully removed Book #{book_id} from Knowledge Graph:[/bold green] "
            f"{stats['chunks_removed']} chunks, {stats['sections_removed']} sections, "
            f"{stats['orphan_concepts_removed']} orphan concepts purged. Graph and exports updated."
        )
    else:
        tracker.remove_book("build_graph", book_id)
        console.print(f"[bold yellow]Book #{book_id} was not found in the Knowledge Graph (checkpoint cleared).[/bold yellow]")


@app.command("test-run")
def test_run_command(
    book_cnt: int = typer.Argument(
        ...,
        metavar="BOOK-CNT",
        help="Number of books to successfully process from scratch before running verification.",
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
    calibre_path: Optional[str] = typer.Option(
        None, "--calibre-path", help="Calibre library path or SMB mount."
    ),
    timeout: int = typer.Option(
        60, "--request-timeout", "--timeout", help="HTTP timeout in seconds for Ollama LLM requests (default: 60s)."
    ),
    percent: float = typer.Option(
        1.0, "--percent", "-p", help="Percentage of ideas to randomly verify (default: 1.0 for 1%)."
    ),
    mode: str = typer.Option(
        "ideas", "--mode", "-m", help="Verification mode (default: 'ideas')."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Optional override LLM model for build-graph."
    ),
    verifier_model: Optional[str] = typer.Option(
        None, "--verifier-model", help="Optional dedicated verifier LLM model for verify."
    ),
    export_neo4j: Optional[bool] = typer.Option(
        None, "--export-neo4j/--no-export-neo4j", help="Upsert knowledge graph directly into Neo4j (default: config.yaml neo4j.enabled)."
    ),
    skip_warmup: bool = typer.Option(
        False, "--skip-warmup", help="Skip Ollama warmup and GPU acceleration check."
    ),
):
    """
    Execute an end-to-end test run across {book-cnt} books:
    1. Rebuilds knowledge graph from scratch for the first {book-cnt} books:
       (--from-scratch --no-lightrag --clean-export --timeout 60)
    2. Halts immediately once {book-cnt} books have completed successfully.
    3. Automatically starts factual verification on the resulting graph (--mode ideas --percent 1%).
    """
    if book_cnt < 1:
        console.print("[bold red]book-cnt must be at least 1[/bold red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold cyan]🧪 Test Run Pipeline Initiated[/bold cyan]\n"
            f"• Target Books: [bold green]{book_cnt}[/bold green] (from scratch)\n"
            f"• Build Parameters: [dim]--from-scratch --no-lightrag --clean-export --timeout {timeout}[/dim]\n"
            f"• Verification: [bold green]{percent}%[/bold green] (mode: [cyan]{mode}[/cyan])",
            title="bookeeper test-run",
            border_style="cyan",
        )
    )

    # 1. Build Graph from Scratch for first book_cnt books
    console.print(
        f"\n[bold yellow]═══ Stage 1/2: Building Knowledge Graph from scratch (limit: {book_cnt} books) ═══[/bold yellow]\n"
    )
    try:
        build_graph(
            calibre_path=calibre_path,
            config_path=config_path,
            from_scratch=True,
            enable_lightrag=False,
            clean_export=True,
            request_timeout=timeout,
            limit=book_cnt,
            model=model,
            export_neo4j=export_neo4j,
            skip_warmup=skip_warmup,
        )
    except typer.Exit as e:
        if e.exit_code != 0:
            console.print(f"[bold red]Stage 1 build-graph exited with code {e.exit_code}. Aborting test-run.[/bold red]")
            raise

    # Verify that Stage 1 successfully populated books in the knowledge graph
    eff_cfg = _get_effective_settings(config_path, calibre_path, model=model)
    kg_file = eff_cfg.resolved_output_dir / "knowledge_graph.json"
    if kg_file.is_file():
        chk_store = ConceptGraphStore()
        chk_store.load(kg_file)
        indexed_books = [n for n, d in chk_store.graph.nodes(data=True) if d.get("type") == "Book"]
        if not indexed_books:
            console.print(
                "[bold red]Stage 1 build-graph completed but 0 books were successfully indexed into the Knowledge Graph. "
                "Aborting test-run verification.[/bold red]"
            )
            raise typer.Exit(1)

    # 2. Automatically Run Verification
    console.print(
        f"\n[bold yellow]═══ Stage 2/2: Verifying Knowledge Graph Integrity ({percent}%, mode: {mode}) ═══[/bold yellow]\n"
    )
    try:
        verify_command(
            config_path=config_path,
            percent=percent,
            mode=mode,
            model=verifier_model,
        )
    except typer.Exit as e:
        if e.exit_code != 0:
            console.print(f"[bold red]Stage 2 verification exited with code {e.exit_code}.[/bold red]")
            raise

    console.print(
        f"\n[bold green]✓ Test-run completed successfully ({book_cnt} book(s) processed & verified).[/bold green]\n"
    )


@app.command("rebuild-graph")
def rebuild_graph_command(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml configuration file."
    ),
    calibre_path: Optional[str] = typer.Option(
        None, "--calibre-path", help="Calibre library root path or SMB mount."
    ),
    states_dir: Optional[str] = typer.Option(
        None, "--states-dir", help="Directory containing .book_states JSON files (default: output_dir/.book_states)."
    ),
    chunks_dir: Optional[str] = typer.Option(
        None, "--chunks-dir", help="Directory containing cached chunk JSON files (default: output_dir/chunks)."
    ),
    state_file: Optional[str] = typer.Option(
        None, "--state-file", help="Path to progress tracker state file (default: output_dir/.bookeeper_state.json)."
    ),
    book_id: Optional[int] = typer.Option(
        None, "--book-id", "-b", help="Rebuild only a specific book ID."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Maximum number of books to rebuild."
    ),
    export_obsidian: Optional[str] = typer.Option(
        None, "--export-obsidian", help="Path to Obsidian vault export directory."
    ),
    export_neo4j: Optional[bool] = typer.Option(
        None, "--export-neo4j/--no-export-neo4j", help="Upsert rebuilt knowledge graph directly into Neo4j (default: config neo4j.enabled)."
    ),
    clean_export: bool = typer.Option(
        True, "--clean-export/--no-clean-export", "--clean/--no-clean", help="Wipe existing knowledge graph and export targets before rebuilding."
    ),
    dedup: bool = typer.Option(
        True, "--dedup/--no-dedup", help="Apply entity deduplication (default: True)."
    ),
):
    """
    Fast, offline rebuild of the Knowledge Graph directly from existing per-book state checkpoints (.book_states).
    Bypasses all LLM extraction and sentence chunking, reconstructing all books, sections, chunks,
    grounded concepts, and graph relationships in seconds.
    Wipes existing graph content before rebuild to ensure a clean, authoritative state.
    """
    config_path = _resolve_opt(config_path)
    calibre_path = _resolve_opt(calibre_path)
    states_dir = _resolve_opt(states_dir)
    chunks_dir = _resolve_opt(chunks_dir)
    state_file = _resolve_opt(state_file)
    book_id = _resolve_opt(book_id)
    limit = _resolve_opt(limit)
    export_obsidian = _resolve_opt(export_obsidian)
    export_neo4j = _resolve_opt(export_neo4j)
    clean_export = _resolve_opt(clean_export, True)
    dedup = _resolve_opt(dedup, True)

    cfg = _get_effective_settings(config_path, calibre_path)
    output_dir = cfg.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_file = output_dir / "knowledge_graph.json"
    graphml_dest = output_dir / "knowledge_graph.graphml"
    vault_dest = Path(export_obsidian) if export_obsidian else output_dir / "obsidian_vault"

    resolved_states_dir = Path(states_dir) if states_dir else output_dir / ".book_states"
    resolved_chunks_dir = Path(chunks_dir) if chunks_dir else output_dir / "chunks"
    tracker_path = Path(state_file) if state_file else cfg.resolved_state_file
    tracker = ProgressTracker(tracker_path)
    should_export_neo4j = cfg.neo4j.enabled if export_neo4j is None else export_neo4j

    if not resolved_states_dir.is_dir():
        console.print(f"[bold red]Book states directory not found at:[/bold red] {resolved_states_dir}")
        raise typer.Exit(1)

    # 1. Clean existing knowledge graph and exports if requested
    if clean_export:
        console.print(
            "[bold yellow]Swiping existing Knowledge Graph, GraphML, Obsidian notes, and checkpoint completions...[/bold yellow]"
        )
        tracker.clear("build_graph", keep_skipped=True)
        if graph_file.is_file():
            try:
                graph_file.unlink()
            except Exception:
                pass
        if graphml_dest.is_file():
            try:
                graphml_dest.unlink()
            except Exception:
                pass
        obs = ObsidianExporter(vault_dest)
        obs.export(ConceptGraphStore(), clean=True)
        if should_export_neo4j:
            try:
                neo_exp = Neo4jExporter.from_config(cfg.neo4j)
                neo_exp.export(ConceptGraphStore(), clean=True, show_progress=False, console=console)
                console.print(f"[dim yellow]✓ Cleared existing Neo4j graph nodes and relationships.[/dim yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not wipe Neo4j database ({e})[/yellow]")

    # 2. Discover book state files
    state_files = sorted(resolved_states_dir.glob("book_*_state.json"))
    candidate_states = []
    for sf in state_files:
        try:
            with open(sf, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            bid = s_data.get("book_id")
            if bid is None:
                continue
            if book_id is not None and bid != book_id:
                continue
            processed = s_data.get("processed_chunks", {})
            extracted_count = sum(1 for v in processed.values() if v.get("status") == "extracted")
            if extracted_count > 0:
                candidate_states.append((bid, sf, s_data, extracted_count))
        except Exception as e:
            logger.warning(f"Could not read state file {sf}: {e}")

    candidate_states.sort(key=lambda x: x[0])
    if limit is not None and limit > 0:
        candidate_states = candidate_states[:limit]

    if not candidate_states:
        console.print(f"[bold yellow]No extracted book state files found in {resolved_states_dir}.[/bold yellow]")
        raise typer.Exit(0)

    console.print(
        Panel.fit(
            f"[bold cyan]Knowledge Graph Rebuild Initiated[/bold cyan]\n"
            f"• Source States: [bold green]{len(candidate_states)} book(s)[/bold green] (from {resolved_states_dir.name})\n"
            f"• Chunks Source: [dim]{resolved_chunks_dir}[/dim]\n"
            f"• Entity Deduplication: [bold]{'Enabled' if dedup else 'Disabled'}[/bold]\n"
            f"• Neo4j Export: [bold]{'Enabled' if should_export_neo4j else 'Disabled'}[/bold]",
            title="bookeeper rebuild-graph",
            border_style="cyan",
        )
    )

    # 3. Setup store and deduplicator
    store = ConceptGraphStore()
    chunk_store = ChunkStore(resolved_chunks_dir)
    deduplicator = EntityDeduplicator.from_settings(cfg) if dedup else None

    # Fetch Calibre authors if available
    authors_by_id = {}
    client = CalibreClient(cfg.calibre_library_path)
    if client.is_available():
        try:
            b_meta = client.list_books(fields=["id", "authors"])
            authors_by_id = {b["id"]: ", ".join(b.get("authors", [])) for b in b_meta}
        except Exception:
            pass

    t0 = time.perf_counter()
    rebuilt_books = 0
    total_rebuilt_chunks = 0
    total_rebuilt_concepts = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        rebuild_task = progress.add_task(
            "Rebuilding Knowledge Graph...",
            total=len(candidate_states),
            completed=0,
        )

        for bid, sf, s_data, extracted_count in candidate_states:
            btitle = BookParser.repair_mojibake(s_data.get("book_title", f"Book {bid}"))
            bauthor = authors_by_id.get(bid) or "Unknown"
            progress.update(rebuild_task, description=f"Rebuilding Book #{bid} ('{btitle[:20]}')...")

            # Load chunks
            chunks = chunk_store.load_chunks(bid) if chunk_store.has_chunks(bid) else []
            chunks_map = {c.chunk_id: c for c in chunks}

            processed = s_data.get("processed_chunks", {})
            extracted_items = [v for v in processed.values() if v.get("status") == "extracted"]
            extracted_items.sort(key=lambda x: x.get("chunk_idx", 0))

            store.add_book(bid, title=btitle, author=bauthor)

            for res in extracted_items:
                chk_id = res["chunk_id"]
                chk = chunks_map.get(chk_id)
                if not chk:
                    chk = HierarchicalChunk(
                        chunk_id=chk_id,
                        book_id=bid,
                        book_title=btitle,
                        section_title=res.get("section_title", "Section"),
                        chapter_idx=1,
                        chunk_idx=res.get("chunk_idx", 1),
                        text="",
                    )
                total_rebuilt_chunks += 1
                sec_node_id = store.add_section(
                    book_id=bid,
                    chapter_idx=chk.chapter_idx,
                    title=chk.section_title,
                    text=chk.text,
                )
                store.add_chunk(chk)

                extraction = res.get("extraction", {})
                for c_dict in extraction.get("concepts", []):
                    try:
                        concept = Concept(**c_dict)
                    except Exception:
                        continue
                    canonical_concept = deduplicator.resolve_concept(concept) if deduplicator else concept
                    store.add_concept(canonical_concept)
                    store.add_book_idea_link(book_id=bid, concept_name=canonical_concept.name)
                    total_rebuilt_concepts += 1

                    local_brief = (concept.brief_description or "").strip()
                    local_detailed = (concept.detailed_explanation or "").strip()
                    is_grounded, valid_quote = validate_concept_chunk_grounding(
                        concept_name=canonical_concept.name,
                        supporting_quote=concept.supporting_quote,
                        chunk_text=chk.text,
                        brief_description=local_brief or local_detailed,
                    )
                    if is_grounded:
                        store.add_idea_support_link(
                            concept_name=canonical_concept.name,
                            chunk=chk,
                            quote=valid_quote,
                            brief_description=local_brief,
                            detailed_explanation=local_detailed,
                        )
                        store.add_section_concept_link(
                            section_node_id=sec_node_id,
                            concept_name=canonical_concept.name,
                            summary=concept.summary,
                            quote=valid_quote,
                        )
                    for rel_name in canonical_concept.related_concepts:
                        store.add_concept_relation(
                            src_concept_name=canonical_concept.name,
                            tgt_concept_name=rel_name,
                        )

            # Link sequential chunks
            store.link_sequential_chunks(book_id=bid)

            # Mark state in tracker
            total_expected = s_data.get("total_chunks", len(chunks))
            failed_chunks = len(s_data.get("failed_chunks", []))
            if failed_chunks == 0 and len(extracted_items) >= total_expected and total_expected > 0:
                tracker.mark_completed(
                    "build_graph",
                    bid,
                    title=btitle,
                    metadata={"indexing_status": "fully_indexed", "total_chunks": total_expected},
                )
            else:
                tracker.mark_partially_indexed(
                    "build_graph",
                    bid,
                    title=btitle,
                    failed_chunks=failed_chunks,
                    total_chunks=total_expected,
                )

            rebuilt_books += 1
            progress.advance(rebuild_task)

    elapsed = time.perf_counter() - t0

    # Save rebuilt graph
    store.save(graph_file)
    obs = ObsidianExporter(vault_dest)
    obs.export(store, clean=False)
    gml = GraphMLExporter(graphml_dest)
    gml.export(store)

    if should_export_neo4j:
        try:
            console.print("\n[bold cyan]Exporting Rebuilt Knowledge Graph to Neo4j...[/bold cyan]")
            neo_exp = Neo4jExporter.from_config(cfg.neo4j)
            res = neo_exp.export(store, clean=False, show_progress=True, console=console)
            clean_msg = f" (clean start: purged {res['cleaned_nodes']} previous nodes)" if res.get("clean_start") else ""
            console.print(
                f"[bold green]✓ Neo4j Export Complete{clean_msg}:[/bold green] "
                f"{res['nodes_upserted']} nodes, {res['edges_upserted']} relationships "
                f"in database '{res['database']}' at {res['uri']}."
            )
        except Exception as e:
            console.print(f"[bold red]✗ Failed to export to Neo4j:[/bold red] {e}")

    # Summary Table
    st = store.stats()
    console.print(f"\n[bold green]✓ Knowledge Graph rebuilt successfully in {elapsed:.2f}s![/bold green]")
    console.print(
        Panel.fit(
            f"[bold]Rebuilt Books:[/bold] {rebuilt_books}\n"
            f"[bold]Rebuilt Chunks:[/bold] {total_rebuilt_chunks}\n"
            f"[bold]Rebuilt Concepts:[/bold] {total_rebuilt_concepts}\n"
            f"[bold]Total Nodes in Graph:[/bold] {st['total_nodes']}\n"
            f"[bold]Total Relationships:[/bold] {st['total_edges']}\n\n"
            f"[bold green]Node Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["node_types"].items())
            + "\n\n[bold green]Relationship Types:[/bold green]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in st["edge_types"].items()),
            title="Knowledge Graph Rebuild Complete",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
