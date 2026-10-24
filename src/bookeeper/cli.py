"""
Production-quality Typer CLI interface for bookeeper with rich progress bars and configurable Ollama / Calibre paths.
"""

import datetime
import json
import logging
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import typer
import numpy as np
from langchain_ollama import OllamaEmbeddings
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.markup import escape
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

from bookeeper.calibre import (
    BookParser,
    CalibreClient,
    FolderBookClient,
    Section,
    get_book_client,
)
from bookeeper.config import Settings, get_settings
from bookeeper.graph.exporters import GraphMLExporter, Neo4jExporter, ObsidianExporter
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import ChunkStore, HierarchicalChunk, HierarchicalChunker
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import (
    Concept,
    KnowledgeExtractor,
    SectionExtraction,
    batch_chunks_by_context_size,
    validate_concept_chunk_grounding,
)
from bookeeper.processing.idea_postprocessor import (
    ConsolidatedIdeaOutput,
    IdeaCluster,
    IdeaPostProcessor,
    IdeaPostProcessorReport,
)
from bookeeper.processing.recent_ideas import (
    BatchRoutingResult,
    ChunkPriority,
    IdeaEmbeddingIndex,
    RecentIdeaChunkRouter,
    RoutedChunk,
    get_chunk_size,
    normalize_sort_date,
    partition_books_into_batches,
    sort_books_chronologically,
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
    export_verification_stats,
    export_verification_debug,
)
from bookeeper.backup import (
    clean_knowledge_artifacts,
    compare_knowledge_graphs,
    create_backup,
    detect_archive_format,
    inspect_backup,
    open_backup_target,
    resolve_backup_target,
    resolve_target_books_dir,
    restore_backup,
    ResolvedBackupTarget,
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


def compute_book_chunk_eta(
    total_chunks: int,
    completed_chunks_count: int,
    durations: Sequence[float],
    pool_concurrency: int = 1,
) -> Tuple[Optional[float], str]:
    """
    Compute estimated remaining time for current book chunk extraction strictly based
    on real LLM-processed chunk durations, ignoring zero-duration fastpath/skipped chunks.

    Parameters:
        total_chunks: Total number of chunks in the book.
        completed_chunks_count: Chunks accounted for (including assigned, skipped, and extracted).
        durations: Measured durations (in seconds) of chunks actually processed by LLM.
        pool_concurrency: Parallel worker concurrency across the server pool.

    Returns:
        (rem_seconds, formatted_eta_str)
    """
    rem_chunks = max(0, total_chunks - completed_chunks_count)
    if rem_chunks == 0:
        return 0.0, "0h 00m 00s"

    if not durations:
        return None, "--h --m --s"

    # Use sliding window of the last 50 LLM chunks to adapt to current cluster throughput
    recent = durations[-50:] if len(durations) > 50 else durations
    avg_chunk_dur = sum(recent) / len(recent)
    if avg_chunk_dur <= 0:
        return None, "--h --m --s"

    concurrency = max(1, min(pool_concurrency, rem_chunks))
    rem_sec = (rem_chunks * avg_chunk_dur) / concurrency
    return rem_sec, format_eta_h_min_sec(rem_sec)


def compute_overall_library_eta(
    total_books: int,
    current_book_idx: int,
    current_book_rem_sec: Optional[float],
    current_book_elapsed: float,
    completed_book_records: Sequence[Dict[str, Any]],
    batch_window: int = 10,
) -> Tuple[Optional[float], Optional[float], str, str]:
    """
    Compute overall library processing ETA based on average book processing time
    windowed across at most the last book batch (including extraction, indexing,
    and batch post-processing).

    Parameters:
        total_books: Total books to process in the catalog.
        current_book_idx: 1-based index of the currently processing book.
        current_book_rem_sec: Estimated remaining seconds for current book (from compute_book_chunk_eta).
        current_book_elapsed: Elapsed seconds so far for the current book.
        completed_book_records: List of dicts with 'effective_total' (processing + indexing + batch overhead).
        batch_window: Maximum books in the sliding window (typically batch_size_books).

    Returns:
        (avg_book_time_sec, overall_rem_sec, formatted_eta_str, formatted_avg_book_str)
    """
    recent_books = completed_book_records[-batch_window:] if completed_book_records else []
    avg_book_time: Optional[float] = None

    if recent_books:
        avg_book_time = sum(r.get("effective_total", r.get("book_duration", 0.0)) for r in recent_books) / len(recent_books)
    else:
        # First batch in progress: estimate based on available time of current book
        if current_book_rem_sec is not None:
            avg_book_time = max(1.0, current_book_elapsed + current_book_rem_sec)
        elif current_book_elapsed > 0:
            avg_book_time = max(1.0, current_book_elapsed)

    remaining_books_after = max(0, total_books - current_book_idx)

    if total_books == 0 or (remaining_books_after == 0 and (current_book_rem_sec is None or current_book_rem_sec <= 0)):
        overall_rem_sec: Optional[float] = 0.0
        overall_eta_str = "0d 0h 00m"
    elif avg_book_time is not None:
        cur_part = current_book_rem_sec if current_book_rem_sec is not None else avg_book_time
        overall_rem_sec = cur_part + (remaining_books_after * avg_book_time)
        overall_eta_str = format_eta_days_h_m(overall_rem_sec)
    else:
        overall_rem_sec = None
        overall_eta_str = "--d --h --m"

    avg_book_fmt = format_eta_min_sec(avg_book_time) if avg_book_time is not None else "--m --s"
    return avg_book_time, overall_rem_sec, overall_eta_str, avg_book_fmt


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


class CrossCheckProgressTracker:
    """
    Manages live status display and metrics tracking for Cross-Check A/B testing.
    Renders two dynamic lines:
      Line 1: Current comparative performance (Base vs Candidate) with Fidelity, Score (S_net), Recall, and deltas.
      Line 2: Progress bar, block counter, elapsed time, block rate (s/blk), and ETA.
    """

    def __init__(
        self,
        total_blocks: int,
        branch_a_label: str = "Baseline (A)",
        branch_b_label: str = "Candidate (B)",
        console: Optional[Console] = None,
    ):
        self.total_blocks = total_blocks
        self.branch_a_label = str(branch_a_label)
        self.branch_b_label = str(branch_b_label)
        self.escaped_a = escape(self.branch_a_label)
        self.escaped_b = escape(self.branch_b_label)
        self.console = console or Console()
        self.completed = 0
        self.completed_samples: List[CrossCheckBlockResult] = []
        self.block_durations: List[float] = []
        self.t_start = time.perf_counter()
        self.current_block_info = "Auditing block #1..."
        self._live: Optional[Live] = None
        self._progress: Optional[Progress] = None
        self._task_id: Optional[Any] = None
        self._log_handler: Optional[logging.Handler] = None

    def get_performance_line(self) -> str:
        """
        Build Line 1:
        Current performance, base model versus candidate model, with fidelity number and current score.
        """
        if not self.completed_samples:
            status_desc = escape(str(self.current_block_info))
            return (
                f"[bold cyan]{self.escaped_a}[/] vs [bold magenta]{self.escaped_b}[/] | "
                f"Current Performance: [dim]{status_desc}[/dim]"
            )

        mean_old_fid = float(np.mean([s.old_system.calibrated_fidelity for s in self.completed_samples]))
        mean_new_fid = float(np.mean([s.new_system.calibrated_fidelity for s in self.completed_samples]))
        delta_fid = mean_new_fid - mean_old_fid

        mean_old_net = float(np.mean([s.old_system.net_integrity_score for s in self.completed_samples]))
        mean_new_net = float(np.mean([s.new_system.net_integrity_score for s in self.completed_samples]))
        delta_net = mean_new_net - mean_old_net

        mean_old_rec = float(np.mean([s.old_system.qa_recall for s in self.completed_samples]))
        mean_new_rec = float(np.mean([s.new_system.qa_recall for s in self.completed_samples]))
        delta_rec = mean_new_rec - mean_old_rec

        d_fid_col = "green" if delta_fid >= 0 else "red"
        d_net_col = "green" if delta_net >= 0 else "red"
        d_rec_col = "green" if delta_rec >= 0 else "red"

        return (
            f"[bold cyan]{self.escaped_a}[/] vs [bold magenta]{self.escaped_b}[/] | "
            f"Fidelity: [cyan]{mean_old_fid * 100:.1f}%[/] vs [magenta]{mean_new_fid * 100:.1f}%[/] ([{d_fid_col}]{delta_fid * 100:+.1f}%[/]) | "
            f"Score: [cyan]{mean_old_net:+.2f}[/] vs [magenta]{mean_new_net:+.2f}[/] ([{d_net_col}]{delta_net:+.2f}[/]) | "
            f"Recall: [cyan]{mean_old_rec * 100:.1f}%[/] vs [magenta]{mean_new_rec * 100:.1f}%[/] ([{d_rec_col}]{delta_rec * 100:+.1f}% pts[/])"
        )

    def compute_eta_and_speed(self) -> Tuple[str, str, str]:
        """
        Compute (speed_str, elapsed_str, eta_str) for Line 2.
        """
        elapsed = time.perf_counter() - self.t_start
        elapsed_str = format_eta_min_sec(elapsed)

        if self.block_durations:
            recent = self.block_durations[-10:]
            avg_dur = float(np.mean(recent))
            if avg_dur >= 1.0:
                speed_str = f"{avg_dur:.1f} s/blk"
            elif avg_dur > 0:
                speed_str = f"{1.0 / avg_dur:.2f} blk/s"
            else:
                speed_str = "-- s/blk"

            remaining = max(0, self.total_blocks - self.completed)
            if self.completed >= self.total_blocks:
                eta_str = "0m 00s"
            else:
                rem_sec = remaining * avg_dur
                eta_str = format_eta_h_min_sec(rem_sec) if rem_sec >= 3600 else format_eta_min_sec(rem_sec)
        else:
            speed_str = "-- s/blk"
            eta_str = "--m --s"

        return speed_str, elapsed_str, eta_str

    def get_progress_renderable(self) -> Group:
        """Combine Line 1 and Line 2 into a single Group for Live rendering."""
        line1 = Text.from_markup(self.get_performance_line())
        return Group(line1, self._progress) if self._progress else Group(line1)

    def __enter__(self) -> "CrossCheckProgressTracker":
        # Attach RichHandler so warning logs (like chunk length ratio) scroll above the live area cleanly
        target_logger = logging.getLogger("bookeeper.processing.cross_checker")
        self._log_handler = RichHandler(console=self.console, show_time=False, show_path=False)
        target_logger.addHandler(self._log_handler)

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Cross-Check Progress:[/bold cyan]"),
            BarColumn(bar_width=25),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[stats_line]}"),
            console=self.console,
        )
        self._task_id = self._progress.add_task(
            "cc_total",
            total=self.total_blocks,
            completed=0,
            stats_line="[dim]Elapsed: 0m 00s[/dim] | [cyan]-- s/blk[/cyan] | [bold yellow]ETA: --m --s[/bold yellow]",
        )
        self._live = Live(self.get_progress_renderable(), console=self.console, refresh_per_second=10)
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live:
            self._live.stop()
        if self._log_handler:
            logging.getLogger("bookeeper.processing.cross_checker").removeHandler(self._log_handler)

    def on_progress(
        self,
        completed: int,
        total: int,
        block_info: str,
        item: Optional[CrossCheckBlockResult] = None,
    ):
        """
        Progress callback invoked by run_cross_check.
        """
        self.completed = completed
        self.total_blocks = total
        self.current_block_info = block_info

        if item is not None:
            self.completed_samples.append(item)
            dur = getattr(item, "duration_seconds", None)
            if dur and dur > 0:
                self.block_durations.append(dur)
            else:
                elapsed = time.perf_counter() - self.t_start
                self.block_durations.append(max(0.1, elapsed / max(1, completed)))

        speed_str, elapsed_str, eta_str = self.compute_eta_and_speed()
        stats_str = f"[dim]Elapsed: {elapsed_str}[/dim] | [cyan]{speed_str}[/cyan] | [bold yellow]ETA: {eta_str}[/bold yellow]"

        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, completed=completed, total=total, stats_line=stats_str)

        if self._live:
            self._live.update(self.get_progress_renderable())

        # Milestone periodic output (every 5 blocks or when finished)
        if item is not None and (completed % 5 == 0 or completed == total):
            pct = (completed / total * 100) if total > 0 else 0.0
            mean_old_fid = float(np.mean([s.old_system.calibrated_fidelity for s in self.completed_samples]))
            mean_new_fid = float(np.mean([s.new_system.calibrated_fidelity for s in self.completed_samples]))
            delta_fid = mean_new_fid - mean_old_fid
            d_fid_col = "green" if delta_fid >= 0 else "red"

            mean_old_net = float(np.mean([s.old_system.net_integrity_score for s in self.completed_samples]))
            mean_new_net = float(np.mean([s.new_system.net_integrity_score for s in self.completed_samples]))
            delta_net = mean_new_net - mean_old_net
            d_net_col = "green" if delta_net >= 0 else "red"

            self.console.print(
                f"[bold cyan][Cross-Check][/bold cyan] {completed}/{total} blocks ({pct:.1f}%) | "
                f"Fid: {mean_old_fid * 100:.1f}% vs {mean_new_fid * 100:.1f}% ([{d_fid_col}]{delta_fid * 100:+.1f}%[/]) | "
                f"Score: {mean_old_net:+.2f} vs {mean_new_net:+.2f} ([{d_net_col}]{delta_net:+.2f}[/]) | "
                f"{speed_str} | ETA: {eta_str}"
            )
            sys.stdout.flush()


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
    books_dir: Optional[str] = None,
    max_context_size: Optional[int] = None,
    max_chunk_batch: Optional[int] = None,
    batch_books: Optional[int] = None,
    recent_ideas: Optional[bool] = None,
    similarity_unassigned: Optional[float] = None,
    similarity_high_confidence: Optional[float] = None,
    skip_limbo_chunks: Optional[bool] = None,
    max_batch_chunk_size: Optional[int] = None,
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
    books_dir = _resolve_opt(books_dir)
    max_context_size = _resolve_opt(max_context_size)
    max_chunk_batch = _resolve_opt(max_chunk_batch)
    batch_books = _resolve_opt(batch_books)
    recent_ideas = _resolve_opt(recent_ideas)
    similarity_unassigned = _resolve_opt(similarity_unassigned)
    similarity_high_confidence = _resolve_opt(similarity_high_confidence)
    skip_limbo_chunks = _resolve_opt(skip_limbo_chunks)
    max_batch_chunk_size = _resolve_opt(max_batch_chunk_size)

    cfg = get_settings(config_path)
    overrides = {}
    if calibre_path:
        overrides["calibre_library_path"] = str(calibre_path)
    if books_dir:
        overrides["books_dir"] = str(books_dir)
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
    if max_context_size is not None:
        overrides["max_context_size"] = max_context_size
    if max_chunk_batch is not None:
        overrides["max_chunk_batch"] = max_chunk_batch
    if batch_books is not None:
        overrides["batch_books"] = batch_books
    if similarity_unassigned is not None:
        overrides["similarity_unassigned_threshold"] = similarity_unassigned
    if similarity_high_confidence is not None:
        overrides["similarity_high_confidence_threshold"] = similarity_high_confidence
    if skip_limbo_chunks is not None:
        overrides["skip_limbo_chunks"] = skip_limbo_chunks
    if max_batch_chunk_size is not None:
        overrides["max_batch_chunk_size"] = max_batch_chunk_size

    has_recent_override = (
        recent_ideas is not None
        or skip_limbo_chunks is not None
        or max_batch_chunk_size is not None
    )
    if has_recent_override:
        rc = cfg.recent_ideas.model_copy()
        if recent_ideas is not None:
            rc.enabled = recent_ideas
        if skip_limbo_chunks is not None:
            rc.skip_limbo_chunks = skip_limbo_chunks
        if max_batch_chunk_size is not None:
            rc.max_batch_chunk_size = max_batch_chunk_size
        overrides["recent_ideas"] = rc

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
    cd_label = f"{extractor.cooldown_seconds // 60}m" if extractor.cooldown_seconds >= 60 else f"{extractor.cooldown_seconds}s"
    if len(servers) > 1:
        table = Table(
            title=f"Ollama Multi-Server Pool ({len(servers)} servers, {cd_label} failover cooldown)",
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
            f"• [bold]Multi-Server Pool:[/bold] {len(servers)} server(s) configured (auto-retry cooldown: {cd_label})\n\n"
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
            f"[bold green]Analysis Failover Cooldown:[/bold green] {cfg.resolved_analysis_cooldown_seconds}s\n"
            f"[bold green]Verification Failover Cooldown:[/bold green] {cfg.resolved_verification_cooldown_seconds}s\n"
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
    books_dir: Optional[str] = typer.Option(
        None,
        "--books-dir",
        "--download-dir",
        "--source-dir",
        help="Path to folder containing downloaded books/papers (e.g. archive_downloader/downloads) to use as source instead of Calibre database.",
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
):
    """
    List and search books in the configured Calibre library, download folder, or mounted SMB share.
    Useful for testing database connectivity and discovering book IDs.
    """
    cfg = _get_effective_settings(config_path, calibre_path=calibre_path, books_dir=books_dir)
    client = get_book_client(cfg, calibre_path=calibre_path, books_dir=books_dir)
    source_display = getattr(client, "display_name", f"Calibre Library: {cfg.calibre_library_path}")

    if not client.is_available():
        console.print(
            f"[bold red]Error:[/bold red] Cannot access book source at '[bold cyan]{source_display}[/bold cyan]'.\n"
            f"If using Calibre, ensure the share is mounted or that calibredb is installed.\n"
            f"If using a download folder, ensure --books-dir points to a valid folder containing downloaded books."
        )
        raise typer.Exit(1)

    with console.status(f"[bold blue]Connecting to book source ({source_display})...[/bold blue]"):
        try:
            books = client.list_books(fields=["id", "title", "authors", "formats"])
        except Exception as e:
            console.print(f"[bold red]Failed to read book source:[/bold red] {e}")
            raise typer.Exit(1)

    if search:
        q = search.lower()
        books = [
            b
            for b in books
            if q in b.get("title", "").lower()
            or any(q in a.lower() for a in b.get("authors", []))
            or q in b.get("arxiv_id", "").lower()
        ]

    total_count = len(books)
    if limit and limit > 0:
        books = books[:limit]

    has_arxiv = any(b.get("arxiv_id") for b in books)
    table = Table(
        title=f"{source_display} (Showing {len(books)} of {total_count})",
        show_lines=False,
    )
    table.add_column("ID", justify="right", style="cyan", width=6)
    if has_arxiv:
        table.add_column("arXiv ID", style="yellow", width=12)
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

        row_vals = [str(b.get("id", ""))]
        if has_arxiv:
            row_vals.append(str(b.get("arxiv_id", "")))
        row_vals.extend([title, authors_str[:24], formats_str or "[dim]None[/dim]"])
        table.add_row(*row_vals)

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
    books_dir: Optional[str] = typer.Option(
        None,
        "--books-dir",
        "--download-dir",
        "--source-dir",
        help="Path to folder containing downloaded books/papers (e.g. archive_downloader/downloads) to use as source instead of Calibre database.",
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
        books_dir=books_dir,
    )
    effective_stage_db = cfg.stage_metadata_db if stage_db is None else stage_db
    effective_backup_db = cfg.backup_metadata_db if backup_db is None else backup_db

    client = get_book_client(cfg, calibre_path=calibre_path, books_dir=books_dir)

    if not client.is_available():
        source_display = getattr(client, "display_name", cfg.calibre_library_path)
        console.print(
            f"[bold red]Error:[/bold red] Cannot access book source at '{source_display}'.\n"
            f"Ensure the SMB share is mounted, calibredb is in PATH, or --books-dir contains downloaded books."
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

    if client.is_staged:
        query_msg = "Reading book catalog from staged local SQLite..."
    elif getattr(client, "source_type", None) == "folder":
        query_msg = f"Reading book catalog from {getattr(client, 'display_name', 'download folder')}..."
    else:
        query_msg = "Querying Calibre library over network/calibredb..."
    with console.status(f"[bold blue]{query_msg}[/bold blue]"):
        try:
            books = client.list_books(fields=["id", "title", "authors", "comments", "formats"])
        except Exception as e:
            if should_stage and client.is_staged:
                client.cleanup_staged(delete_file=True)
            console.print(f"[bold red]Failed to query book source:[/bold red] {e}")
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
    books_dir: Optional[str] = typer.Option(
        None,
        "--books-dir",
        "--download-dir",
        "--source-dir",
        help="Path to folder containing downloaded books/papers (e.g. archive_downloader/downloads) to use as source instead of Calibre database.",
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
    dual_pass: Optional[bool] = typer.Option(
        None,
        "--dual-pass/--no-dual-pass",
        "--postprocess-ideas/--no-postprocess-ideas",
        help="Perform second-pass idea consolidation and synthesis for multi-occurrence ideas (default: config.yaml dual_pass.enabled).",
    ),
    max_context_size: Optional[int] = typer.Option(
        None,
        "--max-context-size",
        help="Maximum combined chunk context size in bytes (default: 5120, i.e. 5KB) for simultaneous LLM extraction.",
    ),
    max_chunk_batch: Optional[int] = typer.Option(
        None,
        "--max-chunk-batch",
        help="Top limit on the number of chunks combined in a single extraction task (default: 3).",
    ),
    batch_books: Optional[int] = typer.Option(
        None,
        "--batch-books",
        help="Number of books grouped per chronological ingestion batch (default: 10).",
    ),
    recent_ideas: Optional[bool] = typer.Option(
        None,
        "--recent-ideas/--no-recent-ideas",
        help="Enable chronological batch ingestion with semantic chunk routing (default: True).",
    ),
    similarity_unassigned: Optional[float] = typer.Option(
        None,
        "--similarity-unassigned",
        help="Cosine similarity threshold below which chunks are classified as unassigned (brand new ideas, default: 0.65).",
    ),
    similarity_high_confidence: Optional[float] = typer.Option(
        None,
        "--similarity-high-confidence",
        help="Cosine similarity threshold above which chunks are directly assigned to an existing idea without LLM extraction (default: 0.90).",
    ),
    skip_limbo_chunks: Optional[bool] = typer.Option(
        None,
        "--skip-limbo-chunks/--no-skip-limbo-chunks",
        "--skip-limbo/--no-skip-limbo",
        help="Skip limbo chunks (chunks between unassigned and high-confidence similarity thresholds) completely from LLM processing (default: False).",
    ),
    max_batch_chunk_size: Optional[int] = typer.Option(
        None,
        "--max-batch-chunk-size",
        help="Maximum batch chunk size threshold (in characters): if a limbo chunk is this size or bigger, recover it to unprocessed LLM queue.",
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
    dual_pass = _resolve_opt(dual_pass)
    books_dir = _resolve_opt(books_dir)
    max_context_size = _resolve_opt(max_context_size)
    max_chunk_batch = _resolve_opt(max_chunk_batch)
    batch_books = _resolve_opt(batch_books)
    recent_ideas = _resolve_opt(recent_ideas)
    similarity_unassigned = _resolve_opt(similarity_unassigned)
    similarity_high_confidence = _resolve_opt(similarity_high_confidence)
    skip_limbo_chunks = _resolve_opt(skip_limbo_chunks)
    max_batch_chunk_size = _resolve_opt(max_batch_chunk_size)

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
        books_dir=books_dir,
        max_context_size=max_context_size,
        max_chunk_batch=max_chunk_batch,
        batch_books=batch_books,
        recent_ideas=recent_ideas,
        similarity_unassigned=similarity_unassigned,
        similarity_high_confidence=similarity_high_confidence,
        skip_limbo_chunks=skip_limbo_chunks,
        max_batch_chunk_size=max_batch_chunk_size,
    )
    output_dir = cfg.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_file = output_dir / "knowledge_graph.json"
    vault_dest = Path(export_obsidian) if export_obsidian else output_dir / "obsidian_vault"
    graphml_dest = output_dir / "knowledge_graph.graphml"
    should_clean = cfg.clean_export if clean_export is None else clean_export
    should_export_neo4j = cfg.neo4j.enabled if export_neo4j is None else export_neo4j
    has_cleaned = False
    has_cleaned_neo4j = False

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

        # 3. Neo4j cleanup: ensure Neo4j database is completely wiped for a clean start synchronized with KG
        if should_export_neo4j:
            try:
                neo_exp = Neo4jExporter.from_config(cfg.neo4j)
                deleted_nodes = neo_exp.clean_database(console=console)
                has_cleaned_neo4j = True
            except Exception as e:
                console.print(
                    f"[bold red]✗ Failed to clean Neo4j database on start from scratch:[/bold red] {e}. "
                    "Will attempt clean export during subsequent sync."
                )
                has_cleaned_neo4j = False

    store = ConceptGraphStore()
    if not from_scratch and graph_file.is_file():
        try:
            store.load(graph_file)
            console.print(f"[dim]Loaded existing graph with {store.graph.number_of_nodes()} nodes.[/dim]")
        except Exception:
            pass

    client = get_book_client(cfg, calibre_path=calibre_path, books_dir=books_dir)

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

    # Initialize shared FailoverOllamaEmbeddings pool once so all pipeline stages
    # (chunker, semantic router, entity deduplicator, and dual-pass postprocessor)
    # distribute embedding requests across all available embedding nodes in the pool.
    embeddings: Optional[FailoverOllamaEmbeddings] = None
    try:
        embeddings = FailoverOllamaEmbeddings.from_settings(cfg)
        chunker = HierarchicalChunker(embeddings=embeddings)
    except Exception as e:
        console.print(f"[dim yellow]Warning: Embeddings init skipped ({e}); using paragraph chunker.[/dim yellow]")
        chunker = HierarchicalChunker()

    deduplicator = EntityDeduplicator.from_settings(
        cfg,
        embedding_model=cfg.embedding_model,
        similarity_threshold=cfg.similarity_threshold,
        high_similarity_threshold=getattr(cfg, "high_similarity_threshold", 0.95),
        extractor=extractor,
        embeddings=embeddings,
    )
    deduplicator.seed_from_graph(store)

    active_neo4j_exporter: Optional[Neo4jExporter] = None
    if cfg.neo4j.enabled if export_neo4j is None else export_neo4j:
        try:
            active_neo4j_exporter = Neo4jExporter.from_config(cfg.neo4j)
        except Exception as e:
            logger.debug(f"Could not initialize active Neo4jExporter for HNSW idea index: {e}")

    # Initialize ChunkStore for local persistence and fast reuse of book chunks
    resolved_chunks_path = Path(chunks_dir).expanduser().resolve() if chunks_dir else cfg.resolved_chunks_dir
    chunk_store = ChunkStore(resolved_chunks_path)
    if clean_chunks:
        purged_count = chunk_store.clean_all()
        console.print(f"[dim yellow]Purged {purged_count} cached chunk file(s) from {chunk_store.storage_dir}.[/dim yellow]")
        rechunk = True
    console.print(f"[dim]Persistent Chunks Directory: [bold cyan]{chunk_store.storage_dir}[/bold cyan][/dim]")

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
        p_pub = BookParser.extract_publication_date(p)
        books_to_process.append({
            "id": 1,
            "title": p.stem,
            "author": "Unknown",
            "path": p,
            "publication_date": p_pub,
            "pubdate": p_pub,
        })

    # Case 2: From Calibre / Download Folder / SMB share
    elif client.is_available():
        if client.is_staged:
            query_msg = "Reading book catalog from staged local SQLite..."
        elif getattr(client, "source_type", None) == "folder":
            query_msg = f"Reading book catalog from {getattr(client, 'display_name', 'download folder')}..."
        else:
            query_msg = "Querying Calibre library over network/calibredb..."
        console.print(f"[dim]{query_msg}[/dim]")
        with console.status(f"[bold blue]{query_msg}[/bold blue]"):
            try:
                all_calibre_books = client.list_books(fields=["id", "title", "authors", "formats"])
            except Exception as e:
                if should_stage and client.is_staged:
                    client.cleanup_staged(delete_file=True)
                console.print(f"[bold red]Failed to query book source:[/bold red] {e}")
                raise typer.Exit(1)

        if book_id is not None:
            matches = [b for b in all_calibre_books if b["id"] == book_id]
            if not matches:
                console.print(f"[bold red]Book ID {book_id} not found in book catalog.[/bold red]")
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

            b_pub = b.get("publication_date") or b.get("pubdate")
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
                        "pubdate": b_pub,
                        "publication_date": b_pub,
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
                            "pubdate": b_pub,
                            "publication_date": b_pub,
                        }
                    )
                else:
                    console.print(f"[yellow]Skipping book #{bid}: No supported text format available (EPUB, FB2, PDF, RTF, TXT, ZIP).[/yellow]")
                    tracker.mark_skipped("build_graph", bid, title=b.get("title", f"Book {bid}"), reason="No supported text format available")
    else:
        source_display = getattr(client, "display_name", "Calibre library")
        console.print(
            f"[bold red]Book source not available ({source_display}). Pass an explicit book file via --file <path.epub>, "
            "a download folder via --books-dir <dir>, or mount the Calibre library SMB share via --calibre-path <path>.[/bold red]"
        )
        raise typer.Exit(1)

    if should_clean and not has_cleaned:
        indexed_books_count = len([n for n, d in store.graph.nodes(data=True) if d.get("type") == "Book"])
        export_reindex_str = f" and re-exporting {indexed_books_count} book(s) already in knowledge graph" if indexed_books_count > 0 else ""
        console.print(
            f"[bold cyan]Clean export requested: clearing old export destinations (Obsidian / Neo4j){export_reindex_str}...[/bold cyan]"
        )
        obs = ObsidianExporter(vault_dest)
        obs.export(store, clean=True)
        gml = GraphMLExporter(graphml_dest)
        gml.export(store)
        if should_export_neo4j and not has_cleaned_neo4j:
            try:
                neo_exp = Neo4jExporter.from_config(cfg.neo4j)
                res = neo_exp.export(store, clean=True, show_progress=True, console=console)
                clean_msg = f" (clean start: purged {res['cleaned_nodes']} previous nodes)" if res.get("clean_start") else ""
                console.print(
                    f"[bold green]✓ Neo4j Clean Export Complete{clean_msg}:[/bold green] "
                    f"{res['nodes_upserted']} nodes, {res['edges_upserted']} relationships "
                    f"in database '{res['database']}' at {res['uri']}."
                )
                has_cleaned_neo4j = True
            except Exception as e:
                console.print(f"[bold red]✗ Failed to export to Neo4j:[/bold red] {e}")
        has_cleaned = True

    if not books_to_process:
        if store.graph.number_of_nodes() > 0:
            console.print("[bold green]All books are already indexed in knowledge graph. Knowledge graph and export are up to date.[/bold green]")
        else:
            console.print("[bold yellow]No books available to process.[/bold yellow]")
        raise typer.Exit(0)

    if cfg.recent_ideas.enabled:
        books_to_process = sort_books_chronologically(books_to_process)
        dated_count = sum(1 for b in books_to_process if b.get("publication_date") or b.get("pubdate"))
        console.print(
            f"[bold cyan]Chronological Ingestion:[/bold cyan] Sorted {len(books_to_process)} book(s) "
            f"by publication date ({dated_count} dated, {len(books_to_process) - dated_count} undated)."
        )

    console.print(f"[bold green]Starting pipeline for {len(books_to_process)} book(s)...[/bold green]")

    num_servers = len(extractor.pool.alive_nodes) or len(extractor.pool.nodes)
    pool_concurrency = max_tasks if max_tasks is not None else cfg.calculate_pool_concurrency(num_servers)

    console.print(
        f"[bold cyan]Parallel Task Pool:[/bold cyan] Running up to {pool_concurrency} active chunk task(s) "
        f"across {num_servers} Ollama server(s) (low: {num_servers}, max default: {min(cfg.max_active_tasks_cap, 3 * num_servers)})."
    )

    completed_in_session = 0
    session_assigned_chunks = 0
    session_skipped_chunks = 0
    interrupted = False
    pipeline_start_time = time.perf_counter()

    stats_lock = threading.Lock()
    display_lock = threading.Lock()
    active_tasks_count = 0
    durations: List[float] = []
    completed_book_records: List[Dict[str, Any]] = []
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
    book_state = None
    ran_batch_postproc_on_last_book = False
    has_cleaned_neo4j = bool(has_cleaned_neo4j)
    batch_size_books = cfg.batch_books if cfg.recent_ideas.enabled else 10
    try:
        for book_idx, binfo in enumerate(books_to_process, start=1):
            # Reload knowledge graph at the start of each new batch to synchronize with previous batch state
            if completed_in_session > 0 and (completed_in_session % batch_size_books == 0) and graph_file.is_file():
                preserved_dirty_nodes = set(store.dirty_nodes)
                preserved_dirty_edges = set(store.dirty_edges)
                preserved_removed_nodes = set(store.removed_nodes)
                store.load(graph_file)
                store.dirty_nodes.update(preserved_dirty_nodes)
                store.dirty_edges.update(preserved_dirty_edges)
                store.removed_nodes.update(preserved_removed_nodes)
                deduplicator.seed_from_graph(store)
            bid = binfo["id"]
            btitle = BookParser.repair_mojibake(binfo["title"])
            bauthor = BookParser.repair_mojibake(binfo["author"])
            bpath: Optional[Path] = binfo.get("path")
            bpubdate = binfo.get("publication_date") or binfo.get("pubdate")
            if not bpubdate and bpath and bpath.is_file():
                bpubdate = BookParser.extract_publication_date(bpath)

            book_state_dir = output_dir / ".book_states"
            book_state = BookProcessingState(
                book_id=bid,
                book_title=btitle,
                state_dir=book_state_dir,
                max_chunk_attempts=6,
            )
            if bpubdate:
                book_state.data["publication_date"] = bpubdate

            model_banner = f"LLM: [bold green]{cfg.llm_model}[/bold green]"
            if cfg.llm_model_fallback:
                model_banner += f" (fallback: [bold green]{cfg.llm_model_fallback}[/bold green])"
            console.print(
                f"\n[bold blue]► Ingesting Book #{bid} ({book_idx}/{total_books}): {btitle}[/bold blue] "
                f"[dim]| {model_banner} | Embeddings: [bold cyan]{cfg.embedding_model}[/bold cyan][/dim]"
            )

            t_book_start = time.perf_counter()
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
                            raw_st = getattr(chunker, "embedding_stats", None) or getattr(embeddings, "embedding_stats", {})
                            st = raw_st if isinstance(raw_st, dict) else {}
                            spd_txt = st.get("speed_texts_per_sec", 0.0)
                            if not isinstance(spd_txt, (int, float)):
                                spd_txt = 0.0
                            raw_kb = st.get("speed_chars_per_sec", 0.0)
                            spd_kb = (raw_kb / 1024.0) if isinstance(raw_kb, (int, float)) else 0.0
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
                    book_assigned_chunks = 0
                    book_skipped_chunks = 0
                    book_llm_durations: List[float] = []

                    def _update_progress_description():
                        avg_s, p80_s, p18_s, fastest_s = _calc_stats()
                        with stats_lock:
                            cur_active = active_tasks_count
                            num_extracted = len(extracted_results)
                            cur_book_assigned = book_assigned_chunks
                            cur_book_skipped = book_skipped_chunks
                            cur_session_assigned = session_assigned_chunks
                            cur_session_skipped = session_skipped_chunks
                            # Prefer current book's measured LLM durations; fallback to session durations
                            cur_durations = list(book_llm_durations) if len(book_llm_durations) >= 2 else list(durations)

                        # 1. Calculate book analyses ETA strictly from real LLM-processed chunk durations
                        book_rem_sec, book_eta_str = compute_book_chunk_eta(
                            total_chunks=len(chunks),
                            completed_chunks_count=num_extracted,
                            durations=cur_durations,
                            pool_concurrency=pool_concurrency,
                        )

                        desc1 = (
                            f"[bold cyan]Current Book #{bid}[/bold cyan] ('{btitle[:18]}') | "
                            f"Active: [bold cyan]{cur_active}[/bold cyan] tasks | "
                            f"Assigned: [bold green]{cur_book_assigned}[/bold green] | "
                            f"Skipped: [bold yellow]{cur_book_skipped}[/bold yellow]"
                        )
                        if durations:
                            desc1 += (
                                f" | avg: [bold green]{avg_s:.1f}s[/bold green] | "
                                f"p80: [bold yellow]{p80_s:.1f}s[/bold yellow]"
                            )
                        progress.update(book_task, description=desc1, eta_str=book_eta_str)

                        # 2. Calculate overall library processing ETA based on average book processing time
                        # Windowed across at most the last book batch (including extraction, indexing, and batch post-processing)
                        book_elapsed_so_far = time.perf_counter() - t_book_start
                        avg_book_time, overall_rem_sec, overall_eta_str, avg_book_fmt = compute_overall_library_eta(
                            total_books=total_books,
                            current_book_idx=book_idx,
                            current_book_rem_sec=book_rem_sec,
                            current_book_elapsed=book_elapsed_so_far,
                            completed_book_records=completed_book_records,
                            batch_window=batch_size_books,
                        )

                        desc2 = (
                            f"[bold magenta]Overall Progress[/bold magenta] ({book_idx}/{total_books} books) | "
                            f"avg book: [bold yellow]{avg_book_fmt}[/bold yellow] | "
                            f"Assigned: [bold green]{cur_session_assigned}[/bold green] | "
                            f"Skipped: [bold yellow]{cur_session_skipped}[/bold yellow] | "
                            f"LLM: [bold green]{cfg.llm_model}[/bold green] | "
                            f"Embeddings: [bold cyan]{cfg.embedding_model}[/bold cyan]"
                        )
                        curr_fraction = (num_extracted / len(chunks)) if chunks else 0.0
                        progress.update(
                            overall_task,
                            completed=book_idx - 1 + curr_fraction,
                            description=desc2,
                            eta_str=overall_eta_str,
                        )

                    _update_progress_description()

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
                            restored_status = "skipped_limbo" if srv == "skipped_limbo" else "extracted"
                            with stats_lock:
                                if srv not in ("semantic_fastpath", "skipped_limbo"):
                                    durations.append(dur)
                                if srv == "semantic_fastpath":
                                    book_assigned_chunks += 1
                                    session_assigned_chunks += 1
                                elif srv == "skipped_limbo":
                                    book_skipped_chunks += 1
                                    session_skipped_chunks += 1
                                extracted_results.append({
                                    "status": restored_status,
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

                    fresh_queue: queue.Queue[Tuple[List[HierarchicalChunk], int, Set[str]]] = queue.Queue()
                    deferred_retries: List[Tuple[List[HierarchicalChunk], int, Set[str]]] = []
                    retry_queue: queue.Queue[Tuple[List[HierarchicalChunk], int, Set[str]]] = queue.Queue()

                    # Check if any previously skipped limbo chunks should be recovered from state by size
                    recovery_th = cfg.effective_limbo_recovery_size
                    if recovery_th is not None and recovery_th > 0:
                        recovered_prev = book_state.recover_skipped_limbo_chunks(chunks, threshold=recovery_th)
                        if recovered_prev:
                            console.print(
                                f"  [bold cyan]↺ Recovered {len(recovered_prev)} previously skipped limbo chunk(s) "
                                f"(size >= {recovery_th}) -> restored to unprocessed queue.[/bold cyan]"
                            )

                    unprocessed_fresh: List[HierarchicalChunk] = []
                    unprocessed_retries: List[Tuple[HierarchicalChunk, int]] = []

                    for chk in chunks:
                        if not book_state.is_chunk_processed(chk.chunk_id) and not book_state.is_chunk_failed(chk.chunk_id):
                            init_attempt = book_state.get_initial_attempt(chk.chunk_id)
                            if init_attempt == 0:
                                unprocessed_fresh.append(chk)
                            else:
                                unprocessed_retries.append((chk, init_attempt))

                    # Semantic chunk routing if recent ideas are enabled
                    limbo_candidate_map: Dict[str, Tuple[str, str]] = {}
                    should_skip_limbo = bool(cfg.skip_limbo_chunks or cfg.recent_ideas.skip_limbo_chunks)
                    if cfg.recent_ideas.enabled and unprocessed_fresh and embeddings is not None:
                        idea_index = IdeaEmbeddingIndex.build_from_graph(store, embeddings=embeddings)
                        if (idea_index.vectors is not None and len(idea_index.idea_names) > 0) or active_neo4j_exporter is not None:
                            router = RecentIdeaChunkRouter(
                                idea_index=idea_index,
                                embeddings=embeddings,
                                unassigned_threshold=cfg.similarity_unassigned_threshold,
                                high_confidence_threshold=cfg.similarity_high_confidence_threshold,
                                neo4j_exporter=active_neo4j_exporter,
                                max_batch_chunk_size=recovery_th,
                            )
                            routing = router.route_chunks(unprocessed_fresh)
                            if routing.embeddings_updated:
                                try:
                                    chunk_store.save_chunks(bid, btitle, chunks)
                                except Exception as e:
                                    logger.debug(f"Could not save updated chunk embeddings to ChunkStore: {e}")

                            # Priority 2: High confidence chunks (Fast-path: direct assignment, skip LLM extraction completely!)
                            for rc in routing.high_confidence:
                                h_chk = rc.chunk
                                h_idea = rc.matched_idea
                                if h_idea:
                                    store.add_idea_support_link(
                                        concept_name=h_idea,
                                        chunk=h_chk,
                                        quote=h_chk.text[:200] if h_chk.text else "",
                                        brief_description=rc.matched_idea_summary,
                                        publication_date=bpubdate,
                                    )
                                    store.add_book_idea_link(
                                        book_id=bid,
                                        concept_name=h_idea,
                                        publication_date=bpubdate,
                                    )
                                    sec_node_id = store.add_section(
                                        book_id=bid,
                                        chapter_idx=h_chk.chapter_idx,
                                        title=h_chk.section_title,
                                        text=h_chk.text,
                                    )
                                    store.add_chunk(h_chk)
                                    store.add_section_concept_link(
                                        section_node_id=sec_node_id,
                                        concept_name=h_idea,
                                        summary=rc.matched_idea_summary,
                                        quote=h_chk.text[:200] if h_chk.text else "",
                                    )
                                    book_state.mark_chunk_processed(
                                        chunk_id=h_chk.chunk_id,
                                        chunk_idx=h_chk.chunk_idx,
                                        section_title=h_chk.section_title,
                                        extraction_dict={"concepts": [{"name": h_idea, "summary": rc.matched_idea_summary, "category": "Idea"}]},
                                        duration=0.0,
                                        server_used="semantic_fastpath",
                                    )
                                    with stats_lock:
                                        book_assigned_chunks += 1
                                        session_assigned_chunks += 1
                                        extracted_results.append({
                                            "status": "extracted",
                                            "chunk": h_chk,
                                            "extraction": SectionExtraction(
                                                concepts=[Concept(name=h_idea, summary=rc.matched_idea_summary, category="Idea")],
                                                extracted_ideas=[],
                                            ),
                                            "duration": 0.0,
                                            "server_used": "semantic_fastpath",
                                        })
                                    progress.advance(book_task, advance=1)
                                    with display_lock:
                                        console.print(
                                            f"  [bold green]⚡ [{h_chk.section_title[:24]} p{h_chk.chunk_idx}][/bold green] "
                                            f"High-confidence match ({rc.similarity:.2f}) -> directly assigned to '[bold cyan]{h_idea}[/bold cyan]' (skipped LLM)."
                                        )

                            # Priority 3: Limbo chunks (either skip completely if --skip-limbo-chunks enabled, or validate via LLM)
                            if should_skip_limbo:
                                for rc in routing.limbo:
                                    l_chk = rc.chunk
                                    book_state.mark_chunk_processed(
                                        chunk_id=l_chk.chunk_id,
                                        chunk_idx=l_chk.chunk_idx,
                                        section_title=l_chk.section_title,
                                        extraction_dict={"concepts": [], "extracted_ideas": []},
                                        duration=0.0,
                                        server_used="skipped_limbo",
                                    )
                                    with stats_lock:
                                        book_skipped_chunks += 1
                                        session_skipped_chunks += 1
                                        extracted_results.append({
                                            "status": "skipped_limbo",
                                            "chunk": l_chk,
                                            "extraction": SectionExtraction(concepts=[], extracted_ideas=[]),
                                            "duration": 0.0,
                                            "server_used": "skipped_limbo",
                                        })
                                    progress.advance(book_task, advance=1)
                                    with display_lock:
                                        console.print(
                                            f"  [dim yellow]⏭ [{l_chk.section_title[:24]} p{l_chk.chunk_idx}][/dim yellow] "
                                            f"Limbo chunk ({rc.similarity:.2f}) skipped (--skip-limbo-chunks enabled)."
                                        )
                            else:
                                for rc in routing.limbo:
                                    if rc.matched_idea:
                                        limbo_candidate_map[rc.chunk.chunk_id] = (rc.matched_idea, rc.matched_idea_summary)

                            # Order LLM extraction queue: Priority 1 (Unassigned, including size-recovered limbo chunks) first
                            unprocessed_fresh = routing.get_llm_queue_chunks(skip_limbo=should_skip_limbo)
                            _update_progress_description()
                            if routing.recovered_from_limbo_count > 0:
                                with display_lock:
                                    for rc in routing.recovered_from_limbo:
                                        r_chk = rc.chunk
                                        r_sec = getattr(r_chk, "section_title", "")[:24]
                                        r_idx = getattr(r_chk, "chunk_idx", 0)
                                        r_sz = get_chunk_size(r_chk)
                                        console.print(
                                            f"  [bold cyan]↺ [{r_sec} p{r_idx}][/bold cyan] "
                                            f"Chunk size {r_sz} >= {recovery_th}: recovered from limbo ({rc.similarity:.2f}) -> restored to unprocessed (Priority 1 queue)."
                                        )
                            if routing.high_confidence_count > 0 or routing.limbo_count > 0 or routing.recovered_from_limbo_count > 0:
                                limbo_desc = "skipped" if should_skip_limbo else "Priority 3 validation"
                                recov_desc = f" ({routing.recovered_from_limbo_count} recovered from limbo by size)" if routing.recovered_from_limbo_count > 0 else ""
                                console.print(
                                    f"  [dim cyan]Semantic Chunk Routing: {routing.unassigned_count} unassigned{recov_desc} (Priority 1 brand-new ideas), "
                                    f"{routing.high_confidence_count} high-confidence (Priority 2 direct assignment), "
                                    f"{routing.limbo_count} limbo ({limbo_desc}).[/dim cyan]"
                                )

                    fresh_batches = batch_chunks_by_context_size(
                        unprocessed_fresh,
                        max_context_size=cfg.max_context_size,
                        max_chunk_batch=cfg.max_chunk_batch,
                    )
                    for f_batch in fresh_batches:
                        fresh_queue.put((f_batch, 0, set()))

                    for chk, init_att in unprocessed_retries:
                        deferred_retries.append(([chk], init_att, set()))

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
                            chunk_batch: List[HierarchicalChunk] = []
                            attempt: int = 0
                            tried_urls: Set[str] = set()

                            if cur_phase == "fresh":
                                try:
                                    chunk_batch, attempt, tried_urls = fresh_queue.get(timeout=0.2)
                                except queue.Empty:
                                    continue
                                task_phase = "fresh"
                                with stats_lock:
                                    fresh_in_flight += len(chunk_batch)
                                    active_tasks_count += len(chunk_batch)
                                _update_progress_description()
                            else:  # cur_phase == "retry"
                                try:
                                    chunk_batch, attempt, tried_urls = retry_queue.get(timeout=0.2)
                                except queue.Empty:
                                    with stats_lock:
                                        if len(extracted_results) >= len(chunks):
                                            break
                                    continue
                                task_phase = "retry"
                                with stats_lock:
                                    retry_in_flight += len(chunk_batch)
                                    active_tasks_count += len(chunk_batch)
                                _update_progress_description()

                            t0 = time.time()
                            srv = "ollama"
                            for chk_item in chunk_batch:
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
                                    for chk_item in chunk_batch:
                                        book_state.unstack_chunk(chk_item.chunk_id)
                                    if task_phase == "fresh":
                                        fresh_queue.task_done()
                                    else:
                                        retry_queue.task_done()
                                    break

                                extractions = extractor.extract_chunks(
                                    chunks=chunk_batch,
                                    book_title=btitle,
                                    model=target_model,
                                    retries=cfg.max_retries,
                                    quarantine_server=False,
                                    exclude_urls=tried_urls,
                                    raise_on_error=True,
                                )
                                dur = time.time() - t0
                                srv = extractor.pool.get_last_used_server() or "ollama"
                                chunk_dur = dur / len(chunk_batch) if chunk_batch else dur

                                with stats_lock:
                                    for chk_item in chunk_batch:
                                        durations.append(chunk_dur)
                                        book_llm_durations.append(chunk_dur)
                                        extraction = extractions.get(chk_item.chunk_id) or SectionExtraction(concepts=[], extracted_ideas=[])
                                        extracted_results.append({
                                            "status": "extracted",
                                            "chunk": chk_item,
                                            "extraction": extraction,
                                            "duration": chunk_dur,
                                            "server_used": srv,
                                            "model_used": target_model,
                                        })

                                for chk_item in chunk_batch:
                                    extraction = extractions.get(chk_item.chunk_id) or SectionExtraction(concepts=[], extracted_ideas=[])
                                    book_state.mark_chunk_processed(
                                        chunk_id=chk_item.chunk_id,
                                        chunk_idx=chk_item.chunk_idx,
                                        section_title=chk_item.section_title,
                                        extraction_dict=extraction.model_dump(),
                                        duration=chunk_dur,
                                        server_used=srv,
                                    )

                                model_tag = f" [{target_model}]" if is_fallback_attempt else ""
                                with display_lock:
                                    if len(chunk_batch) == 1:
                                        chk_item = chunk_batch[0]
                                        ext = extractions.get(chk_item.chunk_id) or SectionExtraction(concepts=[], extracted_ideas=[])
                                        console.print(
                                            f"  [dim green]✓ [{chk_item.section_title[:24]} p{chk_item.chunk_idx}][/dim green] "
                                            f"Extracted [bold cyan]{len(ext.concepts)} concept(s)[/bold cyan] "
                                            f"[dim]({dur:.1f}s via [bold green]{srv}{model_tag}[/bold green])[/dim]"
                                        )
                                    else:
                                        total_concepts = sum(len(extractions.get(c.chunk_id, SectionExtraction()).concepts) for c in chunk_batch)
                                        total_bytes = sum(len((c.text or '').encode('utf-8')) for c in chunk_batch)
                                        total_kb = total_bytes / 1024.0
                                        console.print(
                                            f"  [dim green]✓ [{chunk_batch[0].section_title[:18]} p{chunk_batch[0].chunk_idx}-{chunk_batch[-1].chunk_idx} "
                                            f"({len(chunk_batch)} chunks, {total_kb:.1f}KB)][/dim green] "
                                            f"Extracted [bold cyan]{total_concepts} concept(s)[/bold cyan] "
                                            f"[dim]({dur:.1f}s via [bold green]{srv}{model_tag}[/bold green])[/dim]"
                                        )

                                progress.advance(book_task, advance=len(chunk_batch))
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
                                    # At first failed attempt, unpack batch into individual chunks to isolate poison chunks
                                    for chk_item in chunk_batch:
                                        book_state.mark_chunk_retried(
                                            chunk_id=chk_item.chunk_id,
                                            chunk_idx=chk_item.chunk_idx,
                                            section_title=chk_item.section_title,
                                            attempt=attempt,
                                            error=str(exc),
                                            server_used=srv,
                                        )
                                        with stats_lock:
                                            deferred_retries.append(([chk_item], attempt, tried_urls.copy()))

                                    with display_lock:
                                        if len(chunk_batch) > 1:
                                            console.print(
                                                f"  [yellow]⚠ Combined batch ({len(chunk_batch)} chunks: p{chunk_batch[0].chunk_idx}-{chunk_batch[-1].chunk_idx}) "
                                                f"Attempt {attempt}/{max_chunk_attempts} failed on {srv} ({exc}). "
                                                f"Splitting into individual chunks for deferred retry{fallback_note}.[/yellow]"
                                            )
                                        else:
                                            chk_item = chunk_batch[0]
                                            console.print(
                                                f"  [yellow]⚠ [{chk_item.section_title[:24]} p{chk_item.chunk_idx}] "
                                                f"Attempt {attempt}/{max_chunk_attempts} failed on {srv} ({exc}). "
                                                f"Skipping for now; will retry after all unprocessed chunks are completed{fallback_note}.[/yellow]"
                                            )
                                    fresh_queue.task_done()
                                else:
                                    # Retry phase: retry across pool until max_chunk_attempts
                                    if attempt < max_chunk_attempts and not abort_event.is_set():
                                        for chk_item in chunk_batch:
                                            book_state.mark_chunk_retried(
                                                chunk_id=chk_item.chunk_id,
                                                chunk_idx=chk_item.chunk_idx,
                                                section_title=chk_item.section_title,
                                                attempt=attempt,
                                                error=str(exc),
                                                server_used=srv,
                                            )
                                            retry_queue.put(([chk_item], attempt, tried_urls.copy()))
                                        with display_lock:
                                            console.print(
                                                f"  [yellow]⚠ [p{chunk_batch[0].chunk_idx}] "
                                                f"Attempt {attempt}/{max_chunk_attempts} stalled/timed out on {srv} ({exc}). "
                                                f"Moving to end of retry queue for another server{fallback_note}...[/yellow]"
                                            )
                                    else:
                                        for chk_item in chunk_batch:
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
                                                    "duration": dur / len(chunk_batch),
                                                    "server_used": srv,
                                                    "error": str(exc),
                                                })
                                        with display_lock:
                                            console.print(
                                                f"  [bold red]✗ [{chunk_batch[0].section_title[:24]} p{chunk_batch[0].chunk_idx}] "
                                                f"Failed after {max_chunk_attempts} attempts across pool ({exc}). "
                                                f"Marking chunk as poison pill and continuing book processing.[/bold red]"
                                            )
                                        progress.advance(book_task, advance=len(chunk_batch))
                                    _update_progress_description()
                                    retry_queue.task_done()
                            finally:
                                batch_count = len(chunk_batch) if chunk_batch else 1
                                if task_phase == "fresh":
                                    with stats_lock:
                                        fresh_in_flight = max(0, fresh_in_flight - batch_count)
                                        active_tasks_count = max(0, active_tasks_count - batch_count)
                                elif task_phase == "retry":
                                    with stats_lock:
                                        retry_in_flight = max(0, retry_in_flight - batch_count)
                                        active_tasks_count = max(0, active_tasks_count - batch_count)
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
                        if book_state is not None:
                            book_state.data["stacked_chunks"] = {}
                            if book_state.data.get("status") != "fully_indexed":
                                book_state.data["status"] = "partially_indexed"
                            book_state.save()
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

                    t_finalize_start = time.perf_counter()

                    # Batch-resolve all extracted concepts from this book/article in a single HNSW + embedding batch
                    all_book_concepts: List[Concept] = []
                    for res in successful_results:
                        ext = res["extraction"]
                        if ext and ext.concepts:
                            all_book_concepts.extend(ext.concepts)

                    # Update overall task before exiting chunk extraction progress bar
                    book_elapsed_so_far = time.perf_counter() - t_book_start
                    avg_book_time, overall_rem_sec, overall_eta_str, avg_book_fmt = compute_overall_library_eta(
                        total_books=total_books,
                        current_book_idx=book_idx,
                        current_book_rem_sec=0.0,
                        current_book_elapsed=book_elapsed_so_far,
                        completed_book_records=completed_book_records,
                        batch_window=batch_size_books,
                    )
                    progress.update(overall_task, completed=book_idx, eta_str=overall_eta_str)

                # Transparent 4-step Book Finalization Progress Display
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                    BarColumn(bar_width=24),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console,
                    transient=False,
                ) as fin_progress:
                    fin_task = fin_progress.add_task(
                        f"  Finalizing Book #{bid} [1/4]: Resolving {len(all_book_concepts)} extracted idea(s)...",
                        total=4,
                        completed=0,
                    )

                    # Step 1/4: Resolve & deduplicate extracted concepts (embedding + HNSW + parallel LLM disambiguation)
                    t_step1 = time.perf_counter()

                    def _on_resolve_progress(stage: str, done_cnt: int, tot_cnt: int, detail: str) -> None:
                        fin_progress.update(
                            fin_task,
                            description=f"  Finalizing Book #{bid} [1/4]: {detail}...",
                        )

                    resolved_book_concepts = deduplicator.resolve_concepts_batch(
                        all_book_concepts,
                        neo4j_exporter=active_neo4j_exporter,
                        top_k=3,
                        on_progress=_on_resolve_progress,
                    )
                    dur_resolve = time.perf_counter() - t_step1
                    fin_progress.update(
                        fin_task,
                        completed=1,
                        description=f"  Finalizing Book #{bid} [2/4]: Integrating {len(successful_results)} chunk(s) & grounded quotes...",
                    )

                    # Step 2/4: Integrate sections, chunks, concepts, and grounded quotes into Knowledge Graph
                    t_step2 = time.perf_counter()
                    store.add_book(bid, title=btitle, author=bauthor, publication_date=bpubdate)
                    resolved_concept_iter = iter(resolved_book_concepts)
                    upserted_ideas_for_neo4j: Dict[str, Dict[str, Any]] = {}

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
                            canonical_concept = next(resolved_concept_iter, None) or deduplicator.resolve_concept(concept)
                            store.add_concept(canonical_concept, embedding=canonical_concept.embedding)
                            if canonical_concept.embedding and len(canonical_concept.embedding) > 1:
                                upserted_ideas_for_neo4j[canonical_concept.name] = {
                                    "node_id": f"concept:{canonical_concept.name}",
                                    "name": canonical_concept.name,
                                    "brief_description": canonical_concept.brief_description or canonical_concept.summary,
                                    "detailed_explanation": canonical_concept.detailed_explanation,
                                    "summary": canonical_concept.summary,
                                    "category": canonical_concept.category,
                                    "weight": canonical_concept.weight,
                                    "embedding": canonical_concept.embedding,
                                }

                            store.add_book_idea_link(
                                book_id=bid,
                                concept_name=canonical_concept.name,
                                publication_date=bpubdate,
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
                                    publication_date=bpubdate,
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

                        if chk.chunk_id in limbo_candidate_map:
                            matched_cand_idea, matched_cand_summary = limbo_candidate_map[chk.chunk_id]
                            store.add_idea_support_link(
                                concept_name=matched_cand_idea,
                                chunk=chk,
                                quote=chk.text[:200] if chk.text else "",
                                brief_description=matched_cand_summary,
                                publication_date=bpubdate,
                            )
                            store.add_book_idea_link(
                                book_id=bid,
                                concept_name=matched_cand_idea,
                                publication_date=bpubdate,
                            )

                    dur_integrate = time.perf_counter() - t_step2
                    fin_progress.update(
                        fin_task,
                        completed=2,
                        description=f"  Finalizing Book #{bid} [3/4]: Linking sequential chunks & syncing HNSW vectors...",
                    )

                    # Step 3/4: Link sequential chunks for this book and upsert new idea vectors into Neo4j HNSW index
                    t_step3 = time.perf_counter()
                    store.link_sequential_chunks(book_id=bid)
                    if active_neo4j_exporter is not None and upserted_ideas_for_neo4j:
                        try:
                            active_neo4j_exporter.upsert_idea_vectors_batch(list(upserted_ideas_for_neo4j.values()))
                        except Exception as e:
                            logger.debug(f"Could not upsert idea vectors batch to Neo4j: {e}")
                    dur_link = time.perf_counter() - t_step3
                    fin_progress.update(
                        fin_task,
                        completed=3,
                        description=f"  Finalizing Book #{bid} [4/4]: Saving Knowledge Graph checkpoint...",
                    )

                    # Step 4/4: Finalize book state & persist graph checkpoint (without redundant full-graph chunk re-linking)
                    t_step4 = time.perf_counter()
                    if failed_poison_results:
                        book_state.finalize(has_errors=True)
                        tracker.mark_partially_indexed(
                            "build_graph",
                            bid,
                            title=btitle,
                            failed_chunks=len(failed_poison_results),
                            total_chunks=len(chunks),
                            metadata={
                                "assigned_chunks": book_assigned_chunks,
                                "skipped_chunks": book_skipped_chunks,
                                "publication_date": bpubdate,
                            },
                        )
                    else:
                        book_state.finalize(has_errors=False)
                        tracker.mark_completed(
                            "build_graph",
                            bid,
                            title=btitle,
                            metadata={
                                "indexing_status": "fully_indexed",
                                "total_chunks": len(chunks),
                                "assigned_chunks": book_assigned_chunks,
                                "skipped_chunks": book_skipped_chunks,
                                "publication_date": bpubdate,
                            },
                        )

                    store.save(graph_file, link_chunks=False)
                    dur_save = time.perf_counter() - t_step4
                    total_fin_dur = time.perf_counter() - t_finalize_start
                    fin_progress.update(
                        fin_task,
                        completed=4,
                        description=(
                            f"  [bold green]✓ Book #{bid} finalized in {total_fin_dur:.2f}s[/bold green] "
                            f"[dim](resolve={dur_resolve:.2f}s, integrate={dur_integrate:.2f}s, link={dur_link:.2f}s, save={dur_save:.2f}s)[/dim]"
                        ),
                    )

                if lightrag_engine is not None and chunks:
                    try:
                        valid_chunks = [r["chunk"] for r in successful_results]
                        if valid_chunks:
                            console.print(f"  [dim cyan]Indexing {len(valid_chunks)} chunk(s) into LightRAG engine...[/dim cyan]")
                            with console.status(f"[cyan]Indexing LightRAG ({len(valid_chunks)} chunks)...[/cyan]") as lrag_status:
                                def _on_inline_progress(done: int, tot: int):
                                    lrag_status.update(f"[cyan]Indexing LightRAG: {done}/{tot} chunks ({int(done/tot*100)}%)...[/cyan]")

                                lightrag_engine.insert_chunks(
                                    valid_chunks,
                                    book_title=btitle,
                                    book_id=bid,
                                    batch_size=getattr(cfg, "lightrag_max_parallel_insert", 6) * 2,
                                    on_progress=_on_inline_progress,
                                )
                            console.print(f"  [dim green]✓ Indexed into LightRAG.[/dim green]")
                    except Exception as e:
                        console.print(f"  [yellow]Warning: LightRAG indexing failed for '{btitle}': {e}[/yellow]")

                if failed_poison_results:
                    console.print(
                        f"  [bold yellow]⚠ Book #{bid} partially indexed: {len(successful_results)}/{len(chunks)} chunks succeeded "
                        f"(assigned={book_assigned_chunks}, skipped={book_skipped_chunks}), "
                        f"{len(failed_poison_results)} failed poison chunk(s) skipped. State retained at {book_state.file_path}.[/bold yellow]"
                    )
                else:
                    console.print(
                        f"  [dim green]✓ Book #{bid} fully indexed (all {len(chunks)} chunks succeeded; "
                        f"assigned={book_assigned_chunks}, skipped={book_skipped_chunks}). State retained at {book_state.file_path}.[/dim green]"
                    )

                completed_in_session += 1
                t_book_end = time.perf_counter()
                book_total_dur = t_book_end - t_book_start
                completed_book_records.append({
                    "book_id": bid,
                    "book_duration": book_total_dur,
                    "batch_overhead_share": 0.0,
                    "effective_total": book_total_dur,
                })
                console.print(f"  [dim green]✓ Book #{bid} concepts integrated and graph checkpoint saved ({book_total_dur:.1f}s).[/dim green]")

                # Inter-batch post-processing & incremental Neo4j export of affected nodes
                batch_size = cfg.batch_books if cfg.recent_ideas.enabled else 10
                is_batch_end = (completed_in_session % batch_size == 0) or (book_idx == len(books_to_process))
                should_dual_pass = cfg.dual_pass.enabled if dual_pass is None else dual_pass
                if is_batch_end and not interrupted and store.graph.number_of_nodes() > 0:
                    t_batch_start = time.perf_counter()
                    if should_dual_pass:
                        console.print(f"\n  [bold cyan]═══ Running Inter-Batch Dual-Pass Idea Consolidation (Batch size: {batch_size}) ═══[/bold cyan]")
                        try:
                            postprocessor = IdeaPostProcessor.from_settings(
                                cfg,
                                pool=extractor.pool,
                                embeddings=embeddings,
                            )
                            pp_report = postprocessor.process_graph(store, console=console)
                            store.save(graph_file, link_chunks=False)
                            if book_idx == len(books_to_process):
                                ran_batch_postproc_on_last_book = True
                            console.print(
                                f"  [bold green]✓ Inter-Batch Postprocessing Complete ({pp_report.duration_seconds:.2f}s):[/bold green] "
                                f"{pp_report.total_clusters} clusters synthesized/consolidated."
                            )
                        except Exception as e:
                            console.print(f"  [bold yellow]⚠ Inter-Batch Idea Postprocessor failed:[/bold yellow] {e}")

                    # Export ONLY affected nodes (chunks, ideas, books, sections) and relationships to Neo4j after each batch
                    should_export_neo4j = cfg.neo4j.enabled if export_neo4j is None else export_neo4j
                    if should_export_neo4j and store.has_dirty_changes():
                        try:
                            batch_neo_clean = bool((should_clean or cfg.neo4j.clean_export) and not has_cleaned_neo4j)
                            console.print(
                                f"  [bold cyan]Syncing Batch Affected Nodes to Neo4j "
                                f"({len(store.dirty_nodes)} nodes, {len(store.dirty_edges)} relationships)...[/bold cyan]"
                            )
                            neo_exp = Neo4jExporter.from_config(cfg.neo4j)
                            res = neo_exp.export(
                                store,
                                clean=batch_neo_clean,
                                only_affected=not batch_neo_clean,
                                show_progress=True,
                                console=console,
                            )
                            has_cleaned_neo4j = True
                            console.print(
                                f"  [bold green]✓ Batch Neo4j Sync Complete:[/bold green] "
                                f"upserted {res['nodes_upserted']} affected nodes, {res['edges_upserted']} relationships "
                                f"in '{res['database']}'."
                            )
                        except Exception as e:
                            console.print(f"  [bold yellow]⚠ Batch Neo4j sync failed:[/bold yellow] {e}")

                    # Attribute batch postprocessing & sync overhead evenly to the books of this batch
                    batch_overhead = time.perf_counter() - t_batch_start
                    rem_in_batch = completed_in_session % batch_size
                    books_in_this_batch = rem_in_batch if rem_in_batch != 0 else batch_size
                    overhead_per_book = batch_overhead / max(1, books_in_this_batch)
                    for r in completed_book_records[-books_in_this_batch:]:
                        r["batch_overhead_share"] = overhead_per_book
                        r["effective_total"] = r["book_duration"] + overhead_per_book

                if limit is not None and limit > 0 and completed_in_session >= limit:
                    console.print(
                        f"\n[bold green]✓ Reached target limit of {limit} successfully processed book(s). Halting build-graph as requested.[/bold green]"
                    )
                    break

            except KeyboardInterrupt:
                if book_state is not None:
                    book_state.data["stacked_chunks"] = {}
                    if book_state.data.get("status") != "fully_indexed":
                        book_state.data["status"] = "partially_indexed"
                    book_state.save()
                store.remove_book(bid)
                raise
            except Exception as e:
                console.print(f"[bold red]✗ Failed to build graph for book #{bid} ('{btitle}'):[/bold red] {e}")
                tracker.mark_failed("build_graph", bid, title=btitle, error=str(e))
                store.remove_book(bid)

    except KeyboardInterrupt:
        interrupted = True
        if book_state is not None:
            book_state.data["stacked_chunks"] = {}
            if book_state.data.get("status") != "fully_indexed":
                book_state.data["status"] = "partially_indexed"
            book_state.save()
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
            should_dual_pass = cfg.dual_pass.enabled if dual_pass is None else dual_pass
            if should_dual_pass and not interrupted and not ran_batch_postproc_on_last_book:
                console.print("\n[bold cyan]═══ Running Dual-Pass Idea Postprocessor & Synthesis ═══[/bold cyan]")
                try:
                    postprocessor = IdeaPostProcessor.from_settings(
                        cfg,
                        pool=extractor.pool,
                        embeddings=embeddings,
                    )
                    pp_report = postprocessor.process_graph(store, console=console)
                    console.print(
                        f"[bold green]✓ Idea Postprocessing Complete ({pp_report.duration_seconds:.2f}s):[/bold green] "
                        f"{pp_report.total_clusters} clusters ({pp_report.multi_occurrence_clusters_count} synthesized, "
                        f"{pp_report.single_occurrence_clusters_count} bypassed, {pp_report.merged_clusters_count} merged). "
                        f"Added {pp_report.edges_added_count} provenance/relational edges."
                    )
                except Exception as e:
                    console.print(f"[bold red]✗ Idea Postprocessor failed:[/bold red] {e}")

            store.save(graph_file, link_chunks=False)
            final_clean = should_clean and not has_cleaned
            try:
                obs = ObsidianExporter(vault_dest)
                obs.export(store, clean=final_clean)
            except Exception as e:
                console.print(f"[bold yellow]⚠ Obsidian export skipped or incomplete:[/bold yellow] {e}")

            try:
                gml = GraphMLExporter(graphml_dest)
                gml.export(store)
            except Exception as e:
                console.print(f"[bold yellow]⚠ GraphML export skipped or incomplete:[/bold yellow] {e}")

            should_export_neo4j = cfg.neo4j.enabled if export_neo4j is None else export_neo4j
            neo_clean = final_clean or (cfg.neo4j.clean_export and not has_cleaned_neo4j)
            if should_export_neo4j and (neo_clean or store.has_dirty_changes()):
                try:
                    console.print(
                        "\n[bold cyan]Exporting Affected Knowledge Graph Nodes to Neo4j...[/bold cyan]"
                        if not neo_clean
                        else "\n[bold cyan]Exporting Knowledge Graph to Neo4j...[/bold cyan]"
                    )
                    neo_exp = Neo4jExporter.from_config(cfg.neo4j)
                    res = neo_exp.export(
                        store,
                        clean=neo_clean,
                        only_affected=not neo_clean,
                        show_progress=True,
                        console=console,
                    )
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
            raise typer.Exit(code=0)

    # Persist graph to JSON and exports
    console.print(f"\n[bold green]✓ Knowledge Graph persisted to:[/bold green] {graph_file}")
    console.print(f"[bold green]✓ Obsidian Markdown Vault exported to:[/bold green] {vault_dest}")
    console.print(f"[bold green]✓ GraphML exported to:[/bold green] {graphml_dest}")

    # Display final graph statistics
    st = store.stats()
    console.print(
        Panel.fit(
            f"[bold]Total Nodes:[/bold] {st['total_nodes']}\n"
            f"[bold]Total Edges:[/bold] {st['total_edges']}\n"
            f"[bold]Assigned Chunks (Fast-Path):[/bold] {session_assigned_chunks}\n"
            f"[bold]Skipped Chunks (Limbo):[/bold] {session_skipped_chunks}\n\n"
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
    config_path = _resolve_opt(config_path)
    state_file = _resolve_opt(state_file)
    cfg = get_settings(config_path)
    tracker_path = Path(state_file) if state_file else cfg.resolved_state_file
    tracker = ProgressTracker(tracker_path)
    clean_summary = tracker.summary("clean_metadata")
    graph_summary = tracker.summary("build_graph")

    # Aggregate assigned and skipped chunks from per-book state files & tracker records
    book_states_dir = cfg.resolved_output_dir / ".book_states"
    state_files = sorted(book_states_dir.glob("book_*_state.json")) if book_states_dir.is_dir() else []
    per_book_chunk_stats: Dict[str, Tuple[int, int]] = {}
    parsed_state_records: List[Dict[str, Any]] = []
    for sf in state_files:
        try:
            with open(sf, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            bid_str = str(sdata.get("book_id", sf.stem))
            proc_map = sdata.get("processed_chunks", {})
            b_assigned = sdata.get("assigned_chunks_count")
            if b_assigned is None:
                b_assigned = sum(
                    1
                    for rec in proc_map.values()
                    if isinstance(rec, dict) and (rec.get("server_used") == "semantic_fastpath" or rec.get("status") == "assigned")
                )
            b_skipped = sdata.get("skipped_chunks_count")
            if b_skipped is None:
                b_skipped = sum(
                    1
                    for rec in proc_map.values()
                    if isinstance(rec, dict) and (rec.get("server_used") == "skipped_limbo" or rec.get("status") == "skipped_limbo")
                )
            per_book_chunk_stats[bid_str] = (int(b_assigned), int(b_skipped))
            sdata["_computed_assigned"] = int(b_assigned)
            sdata["_computed_skipped"] = int(b_skipped)
            parsed_state_records.append(sdata)
        except Exception:
            pass

    # Merge tracker metadata for books whose state files might not be in per_book_chunk_stats
    graph_records = tracker._get_op_records("build_graph")
    for bid_str, rec in graph_records.items():
        if bid_str not in per_book_chunk_stats:
            meta = rec.get("metadata") or {}
            per_book_chunk_stats[bid_str] = (
                int(meta.get("assigned_chunks", 0)),
                int(meta.get("skipped_chunks", 0)),
            )

    total_assigned_chunks = sum(v[0] for v in per_book_chunk_stats.values())
    total_skipped_chunks = sum(v[1] for v in per_book_chunk_stats.values())

    table = Table(title=f"Bookeeper Checkpoint State ({tracker_path.name})", show_header=True)
    table.add_column("Operation", style="bold cyan")
    table.add_column("Completed", style="bold green", justify="right")
    table.add_column("Partially Indexed", style="cyan", justify="right")
    table.add_column("Skipped (Graphical)", style="yellow", justify="right")
    table.add_column("Failed", style="bold red", justify="right")
    table.add_column("Assigned Chunks", style="bold green", justify="right")
    table.add_column("Skipped Chunks", style="bold yellow", justify="right")
    table.add_column("Total Recorded", style="white", justify="right")

    table.add_row(
        "clean_metadata",
        str(clean_summary["completed"]),
        str(clean_summary.get("partially_indexed", 0)),
        str(clean_summary.get("skipped", 0)),
        str(clean_summary["failed"]),
        "0",
        "0",
        str(clean_summary["total_recorded"]),
    )
    table.add_row(
        "build_graph",
        str(graph_summary["completed"]),
        str(graph_summary.get("partially_indexed", 0)),
        str(graph_summary.get("skipped", 0)),
        str(graph_summary["failed"]),
        str(total_assigned_chunks),
        str(total_skipped_chunks),
        str(graph_summary["total_recorded"]),
    )
    console.print(table)
    console.print(
        f"[bold cyan]Chunk Routing Summary:[/bold cyan] "
        f"Assigned Chunks: [bold green]{total_assigned_chunks}[/bold green] | "
        f"Skipped Chunks (Limbo): [bold yellow]{total_skipped_chunks}[/bold yellow]"
    )

    if parsed_state_records:
        console.print(f"[dim cyan]Per-book state checkpoints ({len(parsed_state_records)} found in {book_states_dir}):[/dim cyan]")
        for sdata in parsed_state_records[:5]:
            b_status = sdata.get("status", "unknown")
            b_proc = len(sdata.get("processed_chunks", {}))
            b_fail = len(sdata.get("failed_chunks", {}))
            b_tot = sdata.get("total_chunks", 0)
            b_assigned = sdata.get("_computed_assigned", 0)
            b_skipped = sdata.get("_computed_skipped", 0)
            b_title = sdata.get("book_title", "Unknown")
            console.print(
                f"  • [bold]{b_title}[/bold] (ID: {sdata.get('book_id')}): status=[cyan]{b_status}[/cyan], "
                f"processed={b_proc}/{b_tot}, assigned={b_assigned}, skipped={b_skipped}, failed={b_fail}"
            )
        if len(parsed_state_records) > 5:
            console.print(f"  [dim]... and {len(parsed_state_records) - 5} more.[/dim]")

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
    backfill_embeddings: bool = typer.Option(
        True,
        "--backfill-embeddings/--no-backfill-embeddings",
        help="Automatically backfill missing embeddings on existing Idea and Chunk nodes before exporting.",
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

    if backfill_embeddings:
        try:
            from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings
            emb_client = FailoverOllamaEmbeddings.from_settings(cfg, model=cfg.embedding_model)
            bf_res = store.backfill_missing_embeddings(
                emb_client,
                console=console,
                chunks_dir=cfg.resolved_chunks_dir,
            )
            if bf_res["ideas_embedded"] > 0 or bf_res["chunks_embedded"] > 0:
                store.save(target_graph_file, link_chunks=False)
                console.print(
                    f"[bold green]✓ Backfilled and saved missing embeddings:[/bold green] "
                    f"{bf_res['ideas_embedded']} ideas, {bf_res['chunks_embedded']} chunks."
                )
        except Exception as e:
            logger.debug(f"Could not backfill embeddings before export: {e}")

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
    backfill_embeddings: bool = typer.Option(
        True,
        "--backfill-embeddings/--no-backfill-embeddings",
        help="Automatically backfill missing embeddings on existing Idea and Chunk nodes before exporting.",
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

    if backfill_embeddings:
        try:
            from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings
            emb_client = FailoverOllamaEmbeddings.from_settings(cfg, model=cfg.embedding_model)
            bf_res = store.backfill_missing_embeddings(
                emb_client,
                console=console,
                chunks_dir=cfg.resolved_chunks_dir,
            )
            if bf_res["ideas_embedded"] > 0 or bf_res["chunks_embedded"] > 0:
                store.save(target_graph_file, link_chunks=False)
                console.print(
                    f"[bold green]✓ Backfilled and saved missing embeddings:[/bold green] "
                    f"{bf_res['ideas_embedded']} ideas, {bf_res['chunks_embedded']} chunks."
                )
        except Exception as e:
            logger.debug(f"Could not backfill embeddings before Neo4j export: {e}")

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
        err_str = str(e).lower()
        if "errno 65" in err_str or "no route to host" in err_str:
            console.print(
                "\n[bold yellow]ℹ macOS Local Network Privacy Restriction Detected (Errno 65: No route to host)[/bold yellow]\n"
                "[dim]macOS Sequoia blocks third-party Python environments from accessing LAN IP addresses until granted permission.[/dim]\n\n"
                "[bold cyan]How to resolve:[/bold cyan]\n"
                "  1. Open [bold]System Settings[/bold] -> [bold]Privacy & Security[/bold] -> [bold]Local Network[/bold].\n"
                "  2. Find your terminal application ([bold]Terminal[/bold], [bold]iTerm[/bold], [bold]VS Code[/bold], etc.) and toggle it [bold green]ON[/bold green].\n"
                "     (If it is already ON, toggle it [bold red]OFF[/bold red] and back [bold green]ON[/bold green], then restart your terminal.)\n"
            )
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
    console.print(
        f"  [dim]Engine: LLM='{engine.llm_model}' | Embed='{engine.embedding_model}' ({engine.embedding_dim}d) "
        f"at {engine.working_dir}[/dim]"
    )

    total_chunks_indexed = 0
    console.print(f"[bold green]Starting LightRAG indexing for {len(target_ids)} book(s)...[/bold green]\n")

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("• ETA:"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        book_task = progress.add_task("[bold cyan]Books[/bold cyan]", total=len(target_ids))
        chunk_task = progress.add_task("[dim cyan]Chunks[/dim cyan]", total=1)

        for bid in target_ids:
            chunks = chunk_store.load_chunks(bid)
            btitle = chunk_store.get_book_title(bid) or f"Book {bid}"
            if not chunks:
                progress.advance(book_task, 1)
                continue

            progress.update(
                chunk_task,
                description=f"[dim cyan]#{bid} {btitle[:30]}[/dim cyan]",
                completed=0,
                total=len(chunks),
            )

            def _on_chunk_progress(completed: int, total: int):
                progress.update(chunk_task, completed=completed, total=total)

            try:
                indexed = engine.insert_chunks(
                    chunks,
                    book_title=btitle,
                    book_id=bid,
                    batch_size=getattr(cfg, "lightrag_max_parallel_insert", 6) * 2,
                    on_progress=_on_chunk_progress,
                )
                total_chunks_indexed += indexed
            except Exception as e:
                console.print(f"  [bold red]✗ Failed to index Book #{bid} ('{btitle}'):[/bold red] {e}")

            progress.advance(book_task, 1)

    console.print(
        f"\n[bold green]✓ LightRAG build complete:[/bold green] "
        f"{total_chunks_indexed} total chunks indexed across {len(target_ids)} book(s) "
        f"at [cyan]{engine.working_dir}[/cyan]."
    )


def _save_verification_artifacts(
    report: Any,
    effective_mode: str,
    effective_model: str,
    output_report: Optional[str],
    stats_output: Optional[str],
    debug_output: Optional[str],
    effective_debug: bool,
    cfg: Settings,
    pool: Optional[OllamaPool] = None,
    mode_label: str = "verification",
) -> Dict[str, Any]:
    """
    Save primary verification report, standalone statistics summary file, and debug trace JSON.
    """
    if output_report:
        report_dest = Path(output_report).expanduser().resolve()
        stats_dest = (
            Path(stats_output).expanduser().resolve()
            if stats_output
            else report_dest.parent / f"{report_dest.stem}_stats{report_dest.suffix}"
        )
        debug_dest = (
            Path(debug_output).expanduser().resolve()
            if debug_output
            else report_dest.parent / f"{report_dest.stem}_debug{report_dest.suffix}"
        )
    else:
        report_dest = cfg.resolved_output_dir / "verification_report.json"
        stats_dest = (
            Path(stats_output).expanduser().resolve()
            if stats_output
            else cfg.resolved_output_dir / "verification_stats.json"
        )
        debug_dest = (
            Path(debug_output).expanduser().resolve()
            if debug_output
            else cfg.resolved_output_dir / "verification_debug.json"
        )

    # 1. Primary verification report
    report_dest.parent.mkdir(parents=True, exist_ok=True)
    report_dict = report.model_dump() if hasattr(report, "model_dump") else report
    with open(report_dest, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    console.print(f"[dim green]✓ Full {mode_label} report saved to: [bold]{report_dest}[/bold][/dim green]")

    # 2. Standalone statistical summary file (for A/B testing and direct comparison)
    export_verification_stats(report, stats_dest, mode=effective_mode, model_name=effective_model)
    console.print(f"[dim green]✓ Standalone statistics file saved to: [bold]{stats_dest}[/bold][/dim green]")

    # 3. Comprehensive debug trace JSON (raw prompts, responses, questions, sentences, context)
    if effective_debug:
        export_verification_debug(report, debug_dest, mode=effective_mode, model_name=effective_model)
        console.print(f"[dim green]✓ Verification debug trace saved to: [bold]{debug_dest}[/bold][/dim green]\n")
    else:
        console.print()

    if pool is not None:
        _print_ollama_server_stats([pool], console=console, title=f"Ollama Server {mode_label.title()} Effort Statistics")

    return report_dict


def _execute_single_verification(
    config_path: Optional[str] = None,
    graph_file: Optional[str] = None,
    percent: Optional[float] = None,
    mode: Optional[str] = None,
    compare_graph: Optional[str] = None,
    blocks: Optional[int] = None,
    book_id: Optional[int] = None,
    all_books: bool = False,
    threshold: Optional[float] = None,
    embedding_model: Optional[str] = None,
    embedding_url: Optional[str] = None,
    chunks_dir: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_examples: Optional[int] = None,
    output_report: Optional[str] = None,
    seed: Optional[int] = None,
    print_chunks: Optional[int] = None,
    max_tasks: Optional[int] = None,
    cooldown: Optional[int] = None,
    debug: Optional[bool] = None,
    debug_output: Optional[str] = None,
    stats_output: Optional[str] = None,
    cfg: Optional[Settings] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute a single verification run for a specific mode and parameters.
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
    print_chunks = _resolve_opt(print_chunks)
    max_tasks = _resolve_opt(max_tasks)
    cooldown = _resolve_opt(cooldown)
    debug = _resolve_opt(debug)
    debug_output = _resolve_opt(debug_output)
    stats_output = _resolve_opt(stats_output)

    if cfg is None:
        cfg = _get_effective_settings(
            config_path=config_path,
            embedding_model=embedding_model,
            embedding_url=embedding_url,
        )
    effective_debug = debug if debug is not None else getattr(cfg.verification, "debug", False)
    effective_cooldown = cooldown if cooldown is not None else cfg.resolved_verification_cooldown_seconds
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
                if completed > 0 and (completed % 10 == 0 or completed == total):
                    pct = (completed / total * 100) if total > 0 else 0.0
                    console.print(
                        f"[bold cyan][Chunking Coherence][/bold cyan] {completed}/{total} chunks ({pct:.1f}%)"
                        f" | {rate_str} | ETA: {eta_str} | [bold green]✓ {coherent_cnt} coherent[/bold green] | [bold red]✗ {divergent_cnt} divergent[/bold red]"
                    )
                    sys.stdout.flush()

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

        # 4. Save JSON Report, Standalone Statistics, and Debug Trace
        return _save_verification_artifacts(
            report=report,
            effective_mode=effective_mode,
            effective_model=effective_emb_model,
            output_report=output_report,
            stats_output=stats_output,
            debug_output=debug_output,
            effective_debug=effective_debug,
            cfg=cfg,
            pool=pool,
            mode_label="chunking verification",
        )

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
            cooldown_seconds=effective_cooldown,
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
        tot_sup = 0
        tot_not = 0
        tot_opp = 0
        tot_recall = 0.0

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
                nonlocal tot_sup, tot_not, tot_opp, tot_recall
                if item is not None:
                    tot_sup += item.supported_count
                    tot_not += item.not_mentioned_count
                    tot_opp += item.against_count
                    tot_recall += item.qa_recall
                    avg_rec = tot_recall / completed if completed > 0 else 0.0

                    if item.qa_recall >= 1.0:
                        badge = f"[bold green]✓ 100% Recall ({item.supported_count}/{item.total_probes})[/bold green]"
                    elif item.supported_count > 0:
                        badge = f"[bold yellow]⚠ {item.qa_recall * 100:.1f}% ({item.supported_count}/{item.total_probes})[/bold yellow]"
                    else:
                        badge = f"[bold red]✗ 0% Recall ({item.failed_probes} failed)[/bold red]"

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

                if rate >= 1.0:
                    speed_str = f"{rate:.2f} chk/s"
                elif rate > 0:
                    sec_per_chk = 1.0 / rate
                    speed_str = f"{sec_per_chk:.1f} s/chk ({rate:.3f} chk/s)"
                else:
                    speed_str = "-- chk/s"

                avg_rec_str = f"{tot_recall / completed * 100:.1f}%" if completed > 0 else "0.0%"
                stats_str = (
                    f"[cyan]{speed_str}[/cyan] | [yellow]ETA: {eta_str}[/yellow] | "
                    f"[bold green]✓ {tot_sup} Sup[/bold green] | [bold yellow]◯ {tot_not} Omit[/bold yellow] | [bold red]✗ {tot_opp} Opp[/bold red] | "
                    f"Avg Rec: [bold]{avg_rec_str}[/bold]"
                )
                chunk_progress.update(tot_chunk_task, completed=completed, total=total, stats_line=stats_str)
                live.update(Group(Text.from_markup(cur_text), chunk_progress))
                if item is not None and (completed % 10 == 0 or completed == total):
                    pct = (completed / total * 100) if total > 0 else 0.0
                    console.print(
                        f"[bold cyan][Chunk Verification][/bold cyan] {completed}/{total} chunks ({pct:.1f}%)"
                        f" | {speed_str} | ETA: {eta_str} | [bold green]✓ {tot_sup} Sup[/bold green] | [bold yellow]◯ {tot_not} Omit[/bold yellow] | [bold red]✗ {tot_opp} Opp[/bold red]"
                    )
                    sys.stdout.flush()

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
            title="Chunk Verification Results Summary (NLI Continuous Entailment)",
            border_style="cyan",
        )
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", justify="right", style="cyan")

        summary_table.add_row("Total Chunks in Graph", str(st.total_chunks_in_graph))
        summary_table.add_row("Audited Chunks Sampled", f"{st.sampled_chunks_count} ({st.sample_percentage:.1f}%)")
        summary_table.add_row("Total Assertions Evaluated", str(st.total_assertions_count))
        summary_table.add_section()

        summary_table.add_row(
            "Supported Assertions Count (+1.0)",
            f"[bold green]{st.total_supported_count}[/bold green]",
        )
        summary_table.add_row(
            "Not Mentioned / Omission Count (-0.25)",
            f"[bold yellow]{st.total_not_mentioned_count}[/bold yellow]",
        )
        summary_table.add_row(
            "Contradicted / Hallucination Count (-1.0)",
            f"[bold red]{st.total_against_count}[/bold red]",
        )
        summary_table.add_section()

        rec_color = "green" if st.mean_qa_recall >= 0.80 else ("yellow" if st.mean_qa_recall >= 0.60 else "red")
        summary_table.add_row(
            "Mean QA Recall (%)",
            f"[bold {rec_color}]{st.mean_qa_recall * 100:.2f}%[/bold {rec_color}]",
        )
        seam_color = "green" if st.mean_seam_integrity_rate >= 0.80 else "yellow"
        summary_table.add_row(
            "Mean Seam Integrity Rate (Cross-Boundary) (%)",
            f"[bold {seam_color}]{st.mean_seam_integrity_rate * 100:.2f}%[/bold {seam_color}]",
        )
        net_color = "green" if st.mean_net_integrity_score >= 0.50 else ("yellow" if st.mean_net_integrity_score >= 0.0 else "red")
        summary_table.add_row(
            "Mean Net Semantic Integrity (S_net [-1 to +1])",
            f"[bold {net_color}]{st.mean_net_integrity_score:+.2f}[/bold {net_color}]",
        )
        fid_color = "green" if st.mean_calibrated_fidelity >= 0.70 else "yellow"
        summary_table.add_row(
            "Mean Calibrated Fidelity (R_calibrated) (%)",
            f"[bold {fid_color}]{st.mean_calibrated_fidelity * 100:.2f}%[/bold {fid_color}]",
        )
        summary_table.add_row(
            "Mean Hallucination / Contradiction Rate (%)",
            f"[bold {'green' if st.mean_hallucination_ratio == 0 else 'red'}]{st.mean_hallucination_ratio * 100:.2f}%[/bold {'green' if st.mean_hallucination_ratio == 0 else 'red'}]",
        )
        summary_table.add_section()

        summary_table.add_row("Chunks with 100% Recall", f"{st.chunks_with_perfect_recall} / {st.sampled_chunks_count}")
        summary_table.add_row("Chunks with Omissions", f"{st.chunks_with_omissions} / {st.sampled_chunks_count}")
        summary_table.add_row("Chunks with Contradictions", f"{st.chunks_with_hallucinations} / {st.sampled_chunks_count}")
        summary_table.add_row("Total Audit Duration", f"{st.total_duration_seconds:.2f}s")
        console.print(summary_table)

        # 2. Discrepancy Table
        if report.discrepancy_examples:
            disc_table = Table(
                title=f"Top Chunk Discrepancies (Unsubstantiated or Contradicted Assertions)",
                border_style="red",
            )
            disc_table.add_column("#", justify="center", style="dim")
            disc_table.add_column("Chunk ID / Section", style="cyan")
            disc_table.add_column("Unsubstantiated / Failed Assertions", style="bold red")
            disc_table.add_column("QA Recall", justify="right", style="yellow")
            disc_table.add_column("Status (S/N/C)", justify="center", style="dim")
            disc_table.add_column("S_net", justify="right", style="magenta")

            for idx, item in enumerate(report.discrepancy_examples, 1):
                failed_evals = [e for e in item.evaluations if e.classification in ("NOT_MENTIONED", "AGAINST", "CONTRADICTED")]
                if failed_evals:
                    disc_lines = []
                    for e in failed_evals[:3]:
                        tag = "[red]CONTRADICTED[/red]" if e.classification in ("AGAINST", "CONTRADICTED") else "[yellow]OMISSION[/yellow]"
                        disc_lines.append(f"• [{tag}] {e.assertion}")
                    if len(failed_evals) > 3:
                        disc_lines.append(f"... +{len(failed_evals) - 3} more")
                    disc_str = "\n".join(disc_lines)
                elif item.missed_oracle_ideas:
                    disc_str = "\n".join(f"• {name}" for name in item.missed_oracle_ideas[:3])
                    if len(item.missed_oracle_ideas) > 3:
                        disc_str += f"\n... +{len(item.missed_oracle_ideas) - 3} more"
                else:
                    disc_str = "(No failed assertions)"

                status_str = f"{item.supported_count}/{item.not_mentioned_count}/{item.against_count}"
                disc_table.add_row(
                    str(idx),
                    f"{item.chunk_id}\n[dim]{item.book_title} ({item.section_title})[/dim]",
                    disc_str,
                    f"{item.qa_recall * 100:.1f}%",
                    status_str,
                    f"{item.net_integrity_score:+.2f}",
                )
            console.print(disc_table)
        else:
            console.print(
                "\n[bold green]✓ Zero chunk discrepancies found! All evaluated atomic assertions were supported by the knowledge base.[/bold green]\n"
            )

        # 3. Metrics Legend Panel
        legend_content = (
            "[bold cyan]• Recall (QA / NLI Recall):[/bold cyan] Proportion of chunk factual assertions verified as SUPPORTED by the knowledge base claims.\n"
            "  [dim]Formula: Recall = (Supported Assertions) / (Total Assertions)[/dim]\n\n"
            "[bold cyan]• Net Semantic Integrity (S_net):[/bold cyan] Continuous ternary score bounded in [-1.0, +1.0].\n"
            "  [dim]Weights: Supported (+1.0), Omission / Not Mentioned (-0.25), Contradiction / Against (-1.0).[/dim]\n"
            "  [dim]Formula: S_net = (1.0 * N_sup - 0.25 * N_not - 1.0 * N_opp) / N[/dim]\n\n"
            "[bold cyan]• Calibrated Fidelity (R_calibrated):[/bold cyan] Normalized [0.0, 1.0] score penalizing omission (-0.25) and contradiction (-1.0), bounded at 0.\n"
            "  [dim]Formula: R_calibrated = max(0.0, (N_sup - 0.25 * N_not - 1.0 * N_opp) / N)[/dim]\n\n"
            "[bold cyan]• Seam Integrity (Seam Int):[/bold cyan] Retention rate of cross-boundary assertions spanning multiple sentences within the chunk.\n"
            "  [dim]Formula: Seam Int = (Supported Cross-Boundary Assertions) / (Total Cross-Boundary Assertions)[/dim]"
        )
        console.print(Panel(legend_content, title="[bold]📖 Metrics Legend[/bold]", border_style="dim", expand=False))

        # 3. Save JSON Report, Standalone Statistics, and Debug Trace
        return _save_verification_artifacts(
            report=report,
            effective_mode=effective_mode,
            effective_model=effective_model,
            output_report=output_report,
            stats_output=stats_output,
            debug_output=debug_output,
            effective_debug=effective_debug,
            cfg=cfg,
            pool=pool,
            mode_label="chunk verification",
        )

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

        temp_backup_targets: List[ResolvedBackupTarget] = []
        archive_suffixes = (".zip", ".7z", ".tar", ".gz", ".tgz")

        target_graph_a: Optional[Path] = None
        target_graph_b: Optional[Path] = None
        branch_a_label = "Baseline (A)"
        branch_b_label = "Candidate (B)"

        # 0. Check if either target is a backup archive (.zip, .7z)
        if cand_path_a and cand_path_a.is_file() and any(cand_path_a.name.lower().endswith(s) for s in archive_suffixes):
            bt_a = resolve_backup_target(cand_path_a, label_fallback="Baseline (A)")
            temp_backup_targets.append(bt_a)
            target_graph_a = bt_a.graph_file
            branch_a_label = bt_a.label

        if cand_path_b and cand_path_b.is_file() and any(cand_path_b.name.lower().endswith(s) for s in archive_suffixes):
            bt_b = resolve_backup_target(cand_path_b, label_fallback="Candidate (B)")
            temp_backup_targets.append(bt_b)
            target_graph_b = bt_b.graph_file
            branch_b_label = bt_b.label

        # 1. If compare_graph points to an A/B test result directory containing both branches:
        if not target_graph_b and cand_path_b and cand_path_b.is_dir():
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
            cooldown_seconds=effective_cooldown,
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
                f"[bold cyan]Disproportion Metric:[/bold cyan] intersect(block_A, block_B) / union(block_A, block_B)\n"
                + (f"[bold cyan]Print Blocks (< 1.0):[/bold cyan] Up to {print_chunks}\n" if print_chunks else "")
                + f"[bold cyan]Concurrency:[/bold cyan] {pool_concurrency} tasks across {num_servers} Ollama server(s)",
                title="Cross-Check Verification Plan",
            )
        )

        with console.status(
            f"[bold blue]Preloading and warming up verifier model '{effective_model}' across Ollama pool...[/bold blue]"
        ):
            warmup_verifier = IdeaVerifier(pool=pool, model_name=effective_model)
            warmup_verifier.warmup(timeout=90)

        with CrossCheckProgressTracker(
            total_blocks=effective_blocks,
            branch_a_label=branch_a_label,
            branch_b_label=branch_b_label,
            console=console,
        ) as tracker:
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
                print_chunks=int(print_chunks or 0),
                console=console,
                progress_callback=tracker.on_progress,
            )

        # 1. Summary Metrics Table
        sm = report.summary
        summary_table = Table(title="Cross-Check Verification Summary & Decision Gates", border_style="cyan")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Baseline (Old / A)", justify="right", style="cyan")
        summary_table.add_column("Candidate (New / B)", justify="right", style="magenta")
        summary_table.add_column("Delta (B - A)", justify="right")

        summary_table.add_row("Total Contiguous Blocks Audited", str(sm.total_samples), str(sm.total_samples), "-")
        summary_table.add_row("Total Assertions Evaluated", str(sm.total_assertions_count), str(sm.total_assertions_count), "-")
        d_sup_color = "green" if sm.delta_supported >= 0 else "red"
        summary_table.add_row(
            "Supported Assertions Count (+1.0)",
            str(sm.total_old_supported),
            str(sm.total_new_supported),
            f"[{d_sup_color}]{sm.delta_supported:+d}[/]",
        )
        d_not_color = "green" if sm.delta_not_mentioned <= 0 else "yellow"
        summary_table.add_row(
            "Not Mentioned / Omission Count (-0.25)",
            str(sm.total_old_not_mentioned),
            str(sm.total_new_not_mentioned),
            f"[{d_not_color}]{sm.delta_not_mentioned:+d}[/]",
        )
        d_opp_color = "green" if sm.delta_against <= 0 else "red"
        summary_table.add_row(
            "Contradicted / Hallucination Count (-1.0)",
            str(sm.total_old_against),
            str(sm.total_new_against),
            f"[{d_opp_color}]{sm.delta_against:+d}[/]",
        )
        summary_table.add_row(
            "Mean QA Recall (%)",
            f"{sm.mean_old_recall * 100:.2f}%",
            f"{sm.mean_new_recall * 100:.2f}%",
            f"[{'green' if sm.mean_delta_recall >= -0.03 else 'red'}]{sm.mean_delta_recall * 100:+.2f}% pts[/]",
        )
        summary_table.add_row(
            "Mean Seam Integrity Rate (Cross-Sentence) (%)",
            f"{sm.mean_old_seam_integrity * 100:.2f}%",
            f"[{'green' if sm.mean_new_seam_integrity >= 0.80 else 'red'}]{sm.mean_new_seam_integrity * 100:.2f}%[/]",
            f"{(sm.mean_new_seam_integrity - sm.mean_old_seam_integrity) * 100:+.2f}% pts",
        )
        summary_table.add_row(
            "Mean Net Semantic Integrity (S_net [-1 to +1])",
            f"{sm.mean_old_net_integrity:+.2f}",
            f"{sm.mean_new_net_integrity:+.2f}",
            f"[{'green' if sm.mean_delta_net_integrity >= 0 else 'red'}]{sm.mean_delta_net_integrity:+.2f}[/]",
        )
        summary_table.add_row(
            "Mean Calibrated Fidelity (R_calibrated) (%)",
            f"{sm.mean_old_calibrated_fidelity * 100:.2f}%",
            f"{sm.mean_new_calibrated_fidelity * 100:.2f}%",
            f"[{'green' if sm.mean_delta_calibrated_fidelity >= 0 else 'red'}]{sm.mean_delta_calibrated_fidelity * 100:+.2f}% pts[/]",
        )
        summary_table.add_row(
            "Mean Hallucination / Contradiction Rate (%)",
            f"{sm.mean_old_hallucination_ratio * 100:.2f}%",
            f"[{'green' if sm.mean_new_hallucination_ratio == 0 else 'red'}]{sm.mean_new_hallucination_ratio * 100:.2f}%[/]",
            f"[{'green' if sm.mean_delta_hallucination_ratio <= 0 else 'red'}]{sm.mean_delta_hallucination_ratio * 100:+.2f}% pts[/]",
        )
        summary_table.add_row(
            "Aggregate Fail Ratio [sum(Failed)/sum(Total Probes)] (%)",
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
            "Mean Chunk Disproportion [intersect(A,B)/union(A,B)]",
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
            sample_table.add_column("QA Rec A", justify="right")
            sample_table.add_column("QA Rec B", justify="right")
            sample_table.add_column("ΔRecall", justify="right")
            sample_table.add_column("Status A (S/N/C)", justify="right", style="cyan")
            sample_table.add_column("Status B (S/N/C)", justify="right", style="magenta")
            sample_table.add_column("S_net A", justify="right")
            sample_table.add_column("S_net B", justify="right")
            sample_table.add_column("ΔS_net", justify="right")
            sample_table.add_column("Seam Int B", justify="right")
            sample_table.add_column("Fail A", justify="right")
            sample_table.add_column("Fail B", justify="right")

            for s in report.samples[:10]:
                d_color = "green" if s.delta_recall >= 0 else ("yellow" if s.delta_recall >= -0.03 else "red")
                d_net_color = "green" if s.delta_net_integrity >= 0 else "red"
                seam_color = "green" if s.new_system.seam_integrity_rate >= 0.80 else "red"
                sample_table.add_row(
                    str(s.sample_index),
                    f"{s.book_title[:16]} ({s.section_title[:14]})",
                    f"{s.passage_length_chars} chars",
                    f"{s.chunk_disproportion:.3f}",
                    f"{s.old_system.qa_recall * 100:.1f}%",
                    f"{s.new_system.qa_recall * 100:.1f}%",
                    f"[{d_color}]{s.delta_recall * 100:+.1f}%[/]",
                    f"{s.old_system.supported_count}/{s.old_system.not_mentioned_count}/{s.old_system.against_count}",
                    f"{s.new_system.supported_count}/{s.new_system.not_mentioned_count}/{s.new_system.against_count}",
                    f"{s.old_system.net_integrity_score:+.2f}",
                    f"{s.new_system.net_integrity_score:+.2f}",
                    f"[{d_net_color}]{s.delta_net_integrity:+.2f}[/]",
                    f"[{seam_color}]{s.new_system.seam_integrity_rate * 100:.1f}%[/]",
                    f"{s.old_system.fail_ratio * 100:.1f}%",
                    f"{s.new_system.fail_ratio * 100:.1f}%",
                )
            console.print(sample_table)

        # 3. Metrics Legend Panel
        legend_content = (
            "[bold cyan]• Recall (QA / NLI Recall):[/bold cyan] Proportion of raw-passage factual assertions verified as SUPPORTED by the knowledge base.\n"
            "  [dim]Formula: Recall = (Supported Assertions) / (Total Assertions)[/dim]\n\n"
            "[bold cyan]• Net Semantic Integrity (S_net):[/bold cyan] Continuous ternary score bounded in [-1.0, +1.0].\n"
            "  [dim]Weights: Supported (+1.0), Omission / Not Mentioned (-0.25), Contradiction / Against (-1.0).[/dim]\n"
            "  [dim]Formula: S_net = (1.0 * N_sup - 0.25 * N_not - 1.0 * N_opp) / N[/dim]\n\n"
            "[bold cyan]• Calibrated Fidelity (R_calibrated):[/bold cyan] Normalized [0.0, 1.0] score penalizing omission (-0.25) and contradiction (-1.0), bounded at 0.\n"
            "  [dim]Formula: R_calibrated = max(0.0, (N_sup - 0.25 * N_not - 1.0 * N_opp) / N)[/dim]\n\n"
            "[bold cyan]• Seam Integrity (Seam Int / SIMINT):[/bold cyan] Retention rate of cross-boundary assertions spanning multiple sentences.\n"
            "  [dim]Formula: Seam Int = (Supported Cross-Boundary Assertions) / (Total Cross-Boundary Assertions)[/dim]\n"
            "  [dim]Detects seam failure: when Sentence A (cause) and Sentence B (effect) land on separate sides of a chunk boundary and the relationship is severed.[/dim]"
        )
        console.print(Panel(legend_content, title="[bold]📖 Metrics Legend[/bold]", border_style="dim", expand=False))

        # 4. Save JSON Report, Standalone Statistics, and Debug Trace
        try:
            return _save_verification_artifacts(
                report=report,
                effective_mode=effective_mode,
                effective_model=effective_model,
                output_report=output_report or str(cfg.resolved_output_dir / "cross_check_report.json"),
                stats_output=stats_output,
                debug_output=debug_output,
                effective_debug=effective_debug,
                cfg=cfg,
                pool=pool,
                mode_label="cross-check verification",
            )
        finally:
            for bt in temp_backup_targets:
                bt.cleanup()

    # Mode: 'ideas'
    target_graph_file = Path(graph_file).expanduser().resolve() if graph_file else cfg.resolved_output_dir / "knowledge_graph.json"
    temp_ideas_target: Optional[ResolvedBackupTarget] = None
    if target_graph_file.is_file() and any(target_graph_file.name.lower().endswith(s) for s in (".zip", ".7z", ".tar", ".gz", ".tgz")):
        temp_ideas_target = resolve_backup_target(target_graph_file, label_fallback="Target")
        target_graph_file = temp_ideas_target.graph_file

    if not target_graph_file.is_file():
        console.print(
            f"[bold red]Knowledge graph file not found at {target_graph_file}[/bold red]\n"
            f"[yellow]Please run 'bookeeper build-graph' first to generate knowledge_graph.json.[/yellow]"
        )
        if temp_ideas_target:
            temp_ideas_target.cleanup()
        raise typer.Exit(1)

    store = ConceptGraphStore()
    with console.status(f"[bold cyan]Loading graph from {target_graph_file}...[/bold cyan]"):
        try:
            store.load(target_graph_file)
        except Exception as e:
            if temp_ideas_target:
                temp_ideas_target.cleanup()
            console.print(f"[bold red]Failed to load knowledge graph:[/bold red] {e}")
            raise typer.Exit(1)

    effective_percent = cfg.verification.percent if percent is None else percent
    effective_model = model or cfg.resolved_verifier_model
    effective_temp = cfg.verification.temperature if temperature is None else temperature
    effective_max_ex = cfg.verification.max_examples if max_examples is None else max_examples

    verifier_servers = cfg.resolved_verification_servers
    pool = OllamaPool(
        servers=verifier_servers,
        cooldown_seconds=effective_cooldown,
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
            if item is not None and (completed % 10 == 0 or completed == total):
                pct = (completed / total * 100) if total > 0 else 0.0
                console.print(
                    f"[bold cyan][Ideas Verification][/bold cyan] {completed}/{total} items ({pct:.1f}%)"
                    f" | {rate_str} | ETA: {eta_str} | [bold green]✓ {verified_cnt} passed[/bold green] | [bold red]✗ {discrepancy_cnt} failed[/bold red]"
                )
                sys.stdout.flush()

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
    # 3. Save JSON Report, Standalone Statistics, and Debug Trace
    try:
        return _save_verification_artifacts(
            report=report,
            effective_mode=effective_mode,
            effective_model=effective_model,
            output_report=output_report,
            stats_output=stats_output,
            debug_output=debug_output,
            effective_debug=effective_debug,
            cfg=cfg,
            pool=pool,
            mode_label="ideas verification",
        )
    finally:
        if temp_ideas_target:
            temp_ideas_target.cleanup()


def _write_combined_multi_verification_report(
    dest: Path, reports: List[Tuple[str, str, Any]]
) -> None:
    """
    Write a combined JSON report summarizing multi-verification while preserving
    backward compatibility with single-verifier consumers (e.g. A/B test scripts).
    """
    if not reports:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    first_mode, first_path, first_data = reports[0]
    combined = dict(first_data) if isinstance(first_data, dict) else {"primary": first_data}
    combined["multi_verification"] = [
        {
            "mode": m,
            "report_path": str(p),
            "stats": d.get("stats") if isinstance(d, dict) else None,
        }
        for m, p, d in reports
    ]
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    console.print(
        f"[dim green]✓ Combined multi-verification report saved to: [bold]{dest}[/bold][/dim green]"
    )

    stats_dest = (
        dest.parent / f"{dest.stem}_stats{dest.suffix}"
        if dest.stem != "verification_report"
        else dest.parent / "verification_stats.json"
    )
    stats_combined = {
        "multi_verification": True,
        "mode": first_mode,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stats": first_data.get("stats") if isinstance(first_data, dict) else getattr(first_data, "stats", None),
        "verifiers": [
            {
                "mode": m,
                "stats": d.get("stats") if isinstance(d, dict) else getattr(d, "stats", None),
            }
            for m, p, d in reports
        ],
    }
    with open(stats_dest, "w", encoding="utf-8") as f:
        json.dump(stats_combined, f, indent=2, ensure_ascii=False)
    console.print(
        f"[dim green]✓ Combined multi-verification statistics saved to: [bold]{stats_dest}[/bold][/dim green]\n"
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
        None, "--mode", "-m", help="Verification mode: 'ideas', 'chunking', 'chunk', or 'cross-check' (default: from config.yaml or 'ideas')."
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
    print_chunks: Optional[int] = typer.Option(
        None,
        "--print-chunks",
        help="Print text & QA probes comparison for up to X blocks (cross-check mode).",
    ),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-t", help="Max concurrent verification worker tasks across Ollama servers."
    ),
    cooldown: Optional[int] = typer.Option(
        None, "--cooldown", help="Failover cooldown in seconds for verification tasks (overrides config.yaml)."
    ),
    debug: Optional[bool] = typer.Option(
        None,
        "--debug/--no-debug",
        help="Generate detailed debug JSON with raw prompts, questions, sentences, and responses (default: False).",
    ),
    debug_output: Optional[str] = typer.Option(
        None,
        "--debug-output",
        help="Optional path to save debug JSON report (default: output_dir/verification_debug.json).",
    ),
    stats_output: Optional[str] = typer.Option(
        None,
        "--stats-output",
        help="Optional path to save standalone statistics JSON file (default: output_dir/verification_stats.json).",
    ),
):
    """
    Verify Knowledge Graph factual integrity, chunking correctness, or chunk Oracle alignment.
    Supports multi-verification when configured in config.yaml or called without parameters.
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
    print_chunks = _resolve_opt(print_chunks)
    max_tasks = _resolve_opt(max_tasks)
    cooldown = _resolve_opt(cooldown)
    debug = _resolve_opt(debug)
    debug_output = _resolve_opt(debug_output)
    stats_output = _resolve_opt(stats_output)

    cfg = _get_effective_settings(
        config_path=config_path,
        embedding_model=embedding_model,
        embedding_url=embedding_url,
    )

    enabled_verifiers = [v for v in cfg.verifiers if v.enabled]

    # If no explicit mode requested and multiple verifiers are configured: execute multi-verification
    if mode is None and len(enabled_verifiers) > 1:
        console.print(
            Panel.fit(
                f"[bold cyan]🔍 Multi-Verification Pipeline[/bold cyan]\n"
                f"• Configured Verifiers: [bold green]{len(enabled_verifiers)}[/bold green]\n"
                f"• Modes: [dim]{', '.join(v.mode for v in enabled_verifiers)}[/dim]",
                title="bookeeper multi-verification",
                border_style="cyan",
            )
        )
        reports = []
        for idx, v in enumerate(enabled_verifiers, 1):
            console.print(
                f"\n[bold yellow]═══ Running Verifier {idx}/{len(enabled_verifiers)}: {v.mode.upper()} ({v.name or v.mode}) ═══[/bold yellow]\n"
            )
            v_mode = v.mode
            v_percent = percent if percent is not None else v.percent
            v_model = model if model is not None else v.model
            v_temp = temperature if temperature is not None else v.temperature
            v_max_ex = max_examples if max_examples is not None else v.max_examples
            v_thresh = threshold if threshold is not None else v.resolved_distance_threshold
            v_book_id = book_id if book_id is not None else v.book_id
            v_all_books = all_books or (v.all_books if v.all_books is not None else False)
            v_emb_model = embedding_model or v.embedding_model
            v_emb_url = embedding_url or v.embedding_url
            v_chunks_dir = chunks_dir or v.chunks_dir
            v_graph_file = graph_file or v.graph_file
            v_compare_graph = compare_graph or v.compare_graph
            v_blocks = blocks if blocks is not None else v.blocks
            v_print_chunks = print_chunks if print_chunks is not None else v.print_chunks
            v_seed = seed if seed is not None else v.seed
            v_max_tasks = max_tasks if max_tasks is not None else v.max_tasks
            v_cooldown = cooldown if cooldown is not None else v.cooldown_seconds
            v_debug = debug if debug is not None else v.debug
            v_debug_out = debug_output if debug_output is not None else v.debug_output
            v_stats_out = stats_output if stats_output is not None else v.stats_output

            if output_report:
                p_out = Path(output_report)
                v_report = str(p_out.parent / f"{p_out.stem}_{v_mode}{p_out.suffix}")
                v_stats = str(p_out.parent / f"{p_out.stem}_{v_mode}_stats{p_out.suffix}") if not v_stats_out else v_stats_out
                v_debug_file = str(p_out.parent / f"{p_out.stem}_{v_mode}_debug{p_out.suffix}") if not v_debug_out else v_debug_out
            elif v.output_report:
                v_report = v.output_report
                v_stats = v_stats_out or str(Path(v_report).parent / f"{Path(v_report).stem}_stats{Path(v_report).suffix}")
                v_debug_file = v_debug_out or str(Path(v_report).parent / f"{Path(v_report).stem}_debug{Path(v_report).suffix}")
            else:
                v_report = str(cfg.resolved_output_dir / f"verification_report_{v_mode}.json")
                v_stats = v_stats_out or str(cfg.resolved_output_dir / f"verification_stats_{v_mode}.json")
                v_debug_file = v_debug_out or str(cfg.resolved_output_dir / f"verification_debug_{v_mode}.json")

            rep = _execute_single_verification(
                cfg=cfg,
                config_path=config_path,
                graph_file=v_graph_file,
                percent=v_percent,
                mode=v_mode,
                compare_graph=v_compare_graph,
                blocks=v_blocks,
                book_id=v_book_id,
                all_books=v_all_books,
                threshold=v_thresh,
                embedding_model=v_emb_model,
                embedding_url=v_emb_url,
                chunks_dir=v_chunks_dir,
                model=v_model,
                temperature=v_temp,
                max_examples=v_max_ex,
                output_report=v_report,
                seed=v_seed,
                print_chunks=v_print_chunks,
                max_tasks=v_max_tasks,
                cooldown=v_cooldown,
                debug=v_debug,
                debug_output=v_debug_file,
                stats_output=v_stats,
            )
            reports.append((v_mode, v_report, rep))

        primary_rep_path = Path(output_report).expanduser().resolve() if output_report else cfg.resolved_output_dir / "verification_report.json"
        _write_combined_multi_verification_report(primary_rep_path, reports)
        return

    # Single verifier execution
    if mode is None and enabled_verifiers:
        first_v = enabled_verifiers[0]
        mode = first_v.mode
        if percent is None:
            percent = first_v.percent
        if model is None:
            model = first_v.model
        if temperature is None:
            temperature = first_v.temperature
        if max_examples is None:
            max_examples = first_v.max_examples
        if threshold is None:
            threshold = first_v.resolved_distance_threshold
        if cooldown is None:
            cooldown = first_v.cooldown_seconds
        if book_id is None:
            book_id = first_v.book_id
        if all_books is False and first_v.all_books:
            all_books = first_v.all_books
        if blocks is None:
            blocks = first_v.blocks
        if print_chunks is None:
            print_chunks = first_v.print_chunks
        if debug is None:
            debug = first_v.debug
        if debug_output is None:
            debug_output = first_v.debug_output
        if stats_output is None:
            stats_output = first_v.stats_output

    return _execute_single_verification(
        cfg=cfg,
        config_path=config_path,
        graph_file=graph_file,
        percent=percent,
        mode=mode,
        compare_graph=compare_graph,
        blocks=blocks,
        book_id=book_id,
        all_books=all_books,
        threshold=threshold,
        embedding_model=embedding_model,
        embedding_url=embedding_url,
        chunks_dir=chunks_dir,
        model=model,
        temperature=temperature,
        max_examples=max_examples,
        output_report=output_report,
        seed=seed,
        print_chunks=print_chunks,
        max_tasks=max_tasks,
        cooldown=cooldown,
        debug=debug,
        debug_output=debug_output,
        stats_output=stats_output,
    )


@app.command("verifiers")
def verifiers_command(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    graph_file: Optional[str] = typer.Option(
        None, "--graph-file", help="Path to knowledge_graph.json (default: output_dir/knowledge_graph.json)."
    ),
    percent: Optional[float] = typer.Option(
        None, "--percent", "-p", help="Percentage of ideas to randomly verify (default: 1.0 for 1%)."
    ),
    mode: Optional[str] = typer.Option(
        None, "--mode", "-m", help="Verification mode: 'ideas', 'chunking', 'chunk', or 'cross-check' (default: from config.yaml or 'ideas')."
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
    print_chunks: Optional[int] = typer.Option(
        None,
        "--print-chunks",
        help="Print text & QA probes comparison for up to X blocks (cross-check mode).",
    ),
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks", "-t", help="Max concurrent verification worker tasks across Ollama servers."
    ),
    cooldown: Optional[int] = typer.Option(
        None, "--cooldown", help="Failover cooldown in seconds for verification tasks (overrides config.yaml)."
    ),
):
    """Execute configured verifiers (multi-verification by default when multiple verifiers configured)."""
    return verify_command(
        config_path=config_path,
        graph_file=graph_file,
        percent=percent,
        mode=mode,
        compare_graph=compare_graph,
        blocks=blocks,
        book_id=book_id,
        all_books=all_books,
        threshold=threshold,
        embedding_model=embedding_model,
        embedding_url=embedding_url,
        chunks_dir=chunks_dir,
        model=model,
        temperature=temperature,
        max_examples=max_examples,
        output_report=output_report,
        seed=seed,
        print_chunks=print_chunks,
        max_tasks=max_tasks,
        cooldown=cooldown,
    )


@app.command("ab-compare")
@app.command("ab-backup")
def ab_compare_command(
    target_a: str = typer.Argument(
        ...,
        help="Path to baseline Knowledge Graph (backup archive .zip/.7z, directory, or knowledge_graph.json).",
    ),
    target_b: Optional[str] = typer.Argument(
        None,
        help="Path to candidate Knowledge Graph or backup archive. Defaults to active output directory if omitted.",
    ),
    mode: str = typer.Option(
        "cross-check",
        "--mode",
        "-m",
        help="Comparison mode: 'cross-check' (NLI adversarial QA probe audit), 'ideas' (factual grounding audit), or 'stats' (fast topological & entity metrics).",
    ),
    blocks: int = typer.Option(
        20,
        "--blocks",
        help="Number of contiguous chunk blocks to audit in cross-check mode (default: 20).",
    ),
    percent: float = typer.Option(
        1.0,
        "--percent",
        "-p",
        help="Percentage of ideas to sample in ideas verification mode (default: 1.0 for 1%).",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Dedicated verifier LLM model (default: from config.yaml).",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml file.",
    ),
    output_report: Optional[str] = typer.Option(
        None,
        "--output",
        "--output-report",
        "-o",
        help="Optional path to save JSON comparative report (default: output_dir/ab_compare_report.json).",
    ),
    print_chunks: Optional[int] = typer.Option(
        None,
        "--print-chunks",
        help="Print text & QA probes comparison for up to X blocks (cross-check mode).",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducible block/sample selection.",
    ),
    max_tasks: Optional[int] = typer.Option(
        None,
        "--max-tasks",
        "-t",
        help="Max concurrent verification worker tasks across Ollama servers.",
    ),
):
    """
    Compare two Knowledge Bases (e.g. archived backup vs active graph, or backup vs backup)
    without re-indexing through LLMs. Unpacks backup archives (.zip, .7z) into isolated temporary
    folders, evaluates topological and idea differences, and executes comparative cross-check audits.
    """
    cfg = get_settings(config_path)
    output_dir = cfg.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_b_path = target_b
    if not target_b_path:
        default_b = output_dir / "knowledge_graph.json"
        if not default_b.is_file():
            console.print(
                f"[bold red]Candidate target B not provided and default Knowledge Graph not found at '{default_b}'.[/bold red]\n"
                f"[yellow]Specify target_b as a backup archive (.zip/.7z) or path to knowledge_graph.json.[/yellow]"
            )
            raise typer.Exit(1)
        target_b_path = str(default_b)

    with open_backup_target(target_a, label_fallback="Baseline (A)") as bt_a, \
         open_backup_target(target_b_path, label_fallback="Candidate (B)") as bt_b:

        if not bt_a.graph_file.is_file():
            console.print(f"[bold red]Knowledge graph file not found in baseline target: '{target_a}'.[/bold red]")
            raise typer.Exit(1)
        if not bt_b.graph_file.is_file():
            console.print(f"[bold red]Knowledge graph file not found in candidate target: '{target_b_path}'.[/bold red]")
            raise typer.Exit(1)

        store_a = ConceptGraphStore()
        store_b = ConceptGraphStore()

        with console.status(f"[bold cyan]Loading Baseline Knowledge Graph ({bt_a.label})...[/bold cyan]"):
            store_a.load(bt_a.graph_file)
        with console.status(f"[bold cyan]Loading Candidate Knowledge Graph ({bt_b.label})...[/bold cyan]"):
            store_b.load(bt_b.graph_file)

        comp_stats = compare_knowledge_graphs(store_a, store_b, label_a=bt_a.label, label_b=bt_b.label)

        # 1. Header Banner
        a_src = f"{Path(target_a).name} (recovered to temp)" if bt_a.is_temp else str(bt_a.graph_file)
        b_src = f"{Path(target_b_path).name} (recovered to temp)" if bt_b.is_temp else str(bt_b.graph_file)
        console.print(
            Panel.fit(
                f"[bold cyan]🔬 Knowledge Base A/B Backup Comparison[/bold cyan]\n"
                f"• [bold]Baseline (A):[/bold] [cyan]{bt_a.label}[/cyan] ({a_src})\n"
                f"• [bold]Candidate (B):[/bold] [magenta]{bt_b.label}[/magenta] ({b_src})\n"
                f"• [bold]Mode:[/bold] [yellow]{mode.upper()}[/yellow]",
                title="bookeeper ab-compare",
                border_style="cyan",
            )
        )

        # 2. Structural & Topological Metrics Table
        s_tbl = Table(title="Knowledge Base Structural Comparison", border_style="cyan")
        s_tbl.add_column("Entity / Dimension", style="bold")
        s_tbl.add_column(f"Baseline A ({bt_a.label})", justify="right", style="cyan")
        s_tbl.add_column(f"Candidate B ({bt_b.label})", justify="right", style="magenta")
        s_tbl.add_column("Delta (B - A)", justify="right")

        def _fmt_delta(val: int) -> str:
            if val > 0:
                return f"[green]+{val:,}[/green]"
            if val < 0:
                return f"[red]{val:,}[/red]"
            return "[dim]0[/dim]"

        b_st = comp_stats["books"]
        s_tbl.add_row("Total Books", f"{b_st['a']:,}", f"{b_st['b']:,}", _fmt_delta(b_st["delta"]))
        sec_st = comp_stats["sections"]
        s_tbl.add_row("Total Sections", f"{sec_st['a']:,}", f"{sec_st['b']:,}", _fmt_delta(sec_st["delta"]))
        chk_st = comp_stats["chunks"]
        s_tbl.add_row("Total Chunks", f"{chk_st['a']:,}", f"{chk_st['b']:,}", _fmt_delta(chk_st["delta"]))
        s_tbl.add_row("Avg Chunk Size (chars)", f"{chk_st['avg_chars_a']:.1f}", f"{chk_st['avg_chars_b']:.1f}", f"{chk_st['avg_chars_b'] - chk_st['avg_chars_a']:+.1f}")
        s_tbl.add_section()

        idea_st = comp_stats["ideas"]
        s_tbl.add_row("Unique Concepts / Ideas", f"{idea_st['a']:,}", f"{idea_st['b']:,}", _fmt_delta(idea_st["delta"]))
        s_tbl.add_row("Shared Ideas (A ∩ B)", f"{idea_st['common_count']:,}", f"{idea_st['common_count']:,}", f"[yellow]{idea_st['jaccard_similarity']*100:.1f}% Jaccard[/yellow]")
        s_tbl.add_row("Unique to Baseline (A \\ B)", f"{idea_st['only_a_count']:,}", "-", "[dim]-[/dim]")
        s_tbl.add_row("Novel to Candidate (B \\ A)", "-", f"{idea_st['only_b_count']:,}", f"[green]+{idea_st['only_b_count']:,}[/green]" if idea_st["only_b_count"] else "[dim]0[/dim]")
        s_tbl.add_section()

        edg_st = comp_stats["edges"]
        s_tbl.add_row("Relationships / Edges", f"{edg_st['a']:,}", f"{edg_st['b']:,}", _fmt_delta(edg_st["delta"]))
        console.print(s_tbl)

        final_report: Dict[str, Any] = {
            "mode": mode,
            "target_a": str(target_a),
            "target_b": str(target_b_path),
            "label_a": bt_a.label,
            "label_b": bt_b.label,
            "structural_comparison": comp_stats,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # 3. Mode execution
        if mode.lower() in ("cross-check", "cross_check"):
            effective_model = model or cfg.resolved_verifier_model
            effective_embedding_model = getattr(cfg, "embedding_model", "nomic-embed-text")
            verifier_servers = cfg.resolved_verification_servers
            pool = OllamaPool(
                servers=verifier_servers,
                cooldown_seconds=cfg.resolved_verification_cooldown_seconds,
                max_tasks_per_server=1,
            )
            pool_concurrency = max_tasks if max_tasks is not None else cfg.calculate_pool_concurrency(len(pool.alive_nodes) or 1)

            with console.status(f"[bold blue]Preloading and warming up verifier model '{effective_model}'...[/bold blue]"):
                warmup_verifier = IdeaVerifier(pool=pool, model_name=effective_model)
                warmup_verifier.warmup(timeout=90)

            with CrossCheckProgressTracker(
                total_blocks=blocks,
                branch_a_label=bt_a.label,
                branch_b_label=bt_b.label,
                console=console,
            ) as tracker:
                cc_report = run_cross_check(
                    store_a=store_a,
                    store_b=store_b,
                    num_blocks=blocks,
                    pool=pool,
                    model_name=effective_model,
                    embedding_model=effective_embedding_model,
                    branch_a_name=bt_a.label,
                    branch_b_name=bt_b.label,
                    seed=seed,
                    print_chunks=int(print_chunks or 0),
                    console=console,
                    progress_callback=tracker.on_progress,
                )

            sm = cc_report.summary
            cc_tbl = Table(title="Cross-Check Verification Summary & Decision Gates", border_style="cyan")
            cc_tbl.add_column("Metric", style="bold")
            cc_tbl.add_column(f"Baseline ({bt_a.label})", justify="right", style="cyan")
            cc_tbl.add_column(f"Candidate ({bt_b.label})", justify="right", style="magenta")
            cc_tbl.add_column("Delta (B - A)", justify="right")

            cc_tbl.add_row("Total Contiguous Blocks Audited", str(sm.total_samples), str(sm.total_samples), "-")
            cc_tbl.add_row("Total Assertions Evaluated", str(sm.total_assertions_count), str(sm.total_assertions_count), "-")
            cc_tbl.add_row("Supported Assertions (+1.0)", str(sm.total_old_supported), str(sm.total_new_supported), f"[{'green' if sm.delta_supported >= 0 else 'red'}]{sm.delta_supported:+d}[/]")
            cc_tbl.add_row("Omission / Not Mentioned (-0.25)", str(sm.total_old_not_mentioned), str(sm.total_new_not_mentioned), f"[{'green' if sm.delta_not_mentioned <= 0 else 'yellow'}]{sm.delta_not_mentioned:+d}[/]")
            cc_tbl.add_row("Contradicted Assertions (-1.0)", str(sm.total_old_against), str(sm.total_new_against), f"[{'green' if sm.delta_against <= 0 else 'red'}]{sm.delta_against:+d}[/]")
            cc_tbl.add_row("Mean QA Recall (%)", f"{sm.mean_old_recall * 100:.2f}%", f"{sm.mean_new_recall * 100:.2f}%", f"[{'green' if sm.mean_delta_recall >= -0.03 else 'red'}]{sm.mean_delta_recall * 100:+.2f}% pts[/]")
            cc_tbl.add_row("Mean Seam Integrity Rate (%)", f"{sm.mean_old_seam_integrity * 100:.2f}%", f"{sm.mean_new_seam_integrity * 100:.2f}%", f"{(sm.mean_new_seam_integrity - sm.mean_old_seam_integrity) * 100:+.2f}% pts")
            cc_tbl.add_row("Mean Net Semantic Integrity (S_net)", f"{sm.mean_old_net_integrity:+.2f}", f"{sm.mean_new_net_integrity:+.2f}", f"[{'green' if sm.mean_delta_net_integrity >= 0 else 'red'}]{sm.mean_delta_net_integrity:+.2f}[/]")
            cc_tbl.add_row("Mean Calibrated Fidelity (R_calib) (%)", f"{sm.mean_old_calibrated_fidelity * 100:.1f}%", f"{sm.mean_new_calibrated_fidelity * 100:.1f}%", f"[{'green' if sm.mean_delta_calibrated_fidelity >= 0 else 'red'}]{sm.mean_delta_calibrated_fidelity * 100:+.1f}% pts[/]")
            cc_tbl.add_section()
            cc_tbl.add_row("Decision Gating Checklist", "-", "[bold green]PASS[/bold green]" if sm.decision_pass else "[bold red]FAIL[/bold red]", f"[dim]{sm.pass_reason}[/dim]")
            console.print(cc_tbl)

            final_report["cross_check"] = cc_report.model_dump()
            final_report["decision_pass"] = sm.decision_pass
            final_report["pass_reason"] = sm.pass_reason

        # 4. Save JSON Report and Markdown
        report_dest = Path(output_report) if output_report else output_dir / "ab_compare_report.json"
        md_dest = report_dest.with_suffix(".md")
        with open(report_dest, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        md_lines = [
            f"# 🔬 Knowledge Base A/B Comparison Report",
            f"",
            f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            f"- **Baseline (A):** `{bt_a.label}` (`{target_a}`)",
            f"- **Candidate (B):** `{bt_b.label}` (`{target_b_path}`)",
            f"- **Mode:** `{mode}`",
            f"",
            f"## 1. Structural Comparison",
            f"",
            f"| Metric | Baseline (A) | Candidate (B) | Delta (B - A) |",
            f"| :--- | :---: | :---: | :---: |",
            f"| Books | {b_st['a']} | {b_st['b']} | {b_st['delta']:+d} |",
            f"| Sections | {sec_st['a']} | {sec_st['b']} | {sec_st['delta']:+d} |",
            f"| Chunks | {chk_st['a']} | {chk_st['b']} | {chk_st['delta']:+d} |",
            f"| Ideas / Concepts | {idea_st['a']} | {idea_st['b']} | {idea_st['delta']:+d} |",
            f"| Shared Ideas (A ∩ B) | {idea_st['common_count']} | {idea_st['common_count']} | Jaccard: {idea_st['jaccard_similarity']*100:.1f}% |",
            f"| Novel Ideas (B \\ A) | - | {idea_st['only_b_count']} | +{idea_st['only_b_count']} |",
            f"| Edges / Relationships | {edg_st['a']} | {edg_st['b']} | {edg_st['delta']:+d} |",
            f"",
        ]
        if "cross_check" in final_report:
            sm_cc = final_report["cross_check"]["summary"]
            gate_badge = "✅ PASS" if sm_cc.get("decision_pass") else "❌ FAIL"
            md_lines.extend([
                f"## 2. Cross-Check Adversarial QA Probe Results",
                f"",
                f"- **Decision Gate Status:** **{gate_badge}** ({sm_cc.get('pass_reason')})",
                f"- **Mean QA Recall Delta:** `{sm_cc.get('mean_delta_recall', 0)*100:+.2f}% pts` (Baseline: `{sm_cc.get('mean_old_recall', 0)*100:.2f}%`, Candidate: `{sm_cc.get('mean_new_recall', 0)*100:.2f}%`)",
                f"- **Mean Seam Integrity Rate:** `{sm_cc.get('mean_new_seam_integrity', 0)*100:.2f}%`",
                f"- **Net Semantic Integrity (S_net):** Baseline: `{sm_cc.get('mean_old_net_integrity', 0):+.2f}`, Candidate: `{sm_cc.get('mean_new_net_integrity', 0):+.2f}` (Delta: `{sm_cc.get('mean_delta_net_integrity', sm_cc.get('delta_net_integrity', 0)):+.2f}`)",
                f"- **Calibrated Fidelity (R_calib):** Baseline: `{sm_cc.get('mean_old_calibrated_fidelity', 0)*100:.1f}%`, Candidate: `{sm_cc.get('mean_new_calibrated_fidelity', 0)*100:.1f}%` (Delta: `{sm_cc.get('mean_delta_calibrated_fidelity', sm_cc.get('delta_calibrated_fidelity', 0))*100:+.1f}% pts`)",
                f"",
            ])
        md_dest.write_text("\n".join(md_lines), encoding="utf-8")

        console.print(f"\n[bold green]✓ Comparative Report saved to:[/bold green] [bold cyan]{report_dest}[/bold cyan]")
        console.print(f"[bold green]✓ Markdown Summary saved to:[/bold green] [bold cyan]{md_dest}[/bold cyan]\n")
        return final_report


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


@app.command("postprocess-ideas")
def postprocess_ideas_command(
    graph_file: Optional[str] = typer.Option(
        None, "--graph-file", "-g", help="Path to knowledge_graph.json (default: output_dir/knowledge_graph.json)."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml file."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Dedicated LLM model for idea consolidation (default: from config.yaml dual_pass.model or llm_model)."
    ),
    similarity_threshold: Optional[float] = typer.Option(
        None, "--similarity-threshold", help="Cosine similarity threshold for merging idea clusters (default: 0.92)."
    ),
    min_occurrences: Optional[int] = typer.Option(
        None, "--min-occurrences", help="Minimum occurrences (N >= 2) to trigger LLM synthesis (default: 2)."
    ),
    output_graph: Optional[str] = typer.Option(
        None, "--output-graph", "-o", help="Optional output path for consolidated graph (defaults to overwriting input graph)."
    ),
    export_obsidian: Optional[str] = typer.Option(
        None, "--export-obsidian", help="Custom destination directory for Obsidian Markdown vault to refresh after consolidation."
    ),
    state_file: Optional[str] = typer.Option(
        None, "--state-file", help="Custom path for postprocessing state checkpoint JSON file (defaults to output_dir/.bookeeper_dual_pass_state.json)."
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Auto-continue and resume from existing state file if present (default: True)."
    ),
    reset_state: bool = typer.Option(
        False, "--reset-state", help="Reset/discard existing postprocessing state file and start from scratch."
    ),
    checkpoint_interval: int = typer.Option(
        100, "--checkpoint-interval", help="Save postprocessor state checkpoint every N ideas (default: 100)."
    ),
):
    """
    Execute Two-Pass Idea Aggregation and Synthesis on a Knowledge Graph:
    1. Deterministic normalization and grouping + semantic cosine clustering (>= 0.92).
    2. Chronological observation linearization and top 3-5 quote selection.
    3. Branching: N=1 passes through; N>=2 synthesized via LLM consolidation.
    4. Graph overwrite with unified brief_description, detailed_explanation, :MENTIONED_IN, and :RELATES_TO edges.
    Supports persistent state checkpoints (every 100 ideas) and auto-continue.
    """
    cfg = get_settings(config_path)
    output_dir = cfg.resolved_output_dir
    target_graph = Path(graph_file).expanduser().resolve() if graph_file else output_dir / "knowledge_graph.json"

    if not target_graph.is_file():
        console.print(f"[bold red]Knowledge graph file not found at {target_graph}.[/bold red]")
        raise typer.Exit(1)

    eff_model = model or cfg.resolved_dual_pass_model
    eff_thresh = similarity_threshold if similarity_threshold is not None else cfg.dual_pass.similarity_threshold
    eff_min_occ = min_occurrences if min_occurrences is not None else cfg.dual_pass.min_occurrences
    eff_state_path = Path(state_file).expanduser().resolve() if state_file else cfg.resolved_dual_pass_state_file

    if reset_state and eff_state_path.is_file():
        eff_state_path.unlink()
        console.print(f"[bold yellow]Reset existing postprocessing state file at {eff_state_path}.[/bold yellow]")

    console.print(
        Panel.fit(
            f"[bold cyan]🧠 Two-Pass Idea Aggregation & Synthesis Initialized[/bold cyan]\n"
            f"• Knowledge Graph: [bold]{target_graph}[/bold]\n"
            f"• Consolidation Model: [bold green]{eff_model}[/bold green]\n"
            f"• Cosine Similarity Threshold: [bold yellow]{eff_thresh}[/bold yellow]\n"
            f"• Multi-Occurrence Threshold: [bold]{eff_min_occ}+ mentions[/bold]\n"
            f"• State Checkpoint File: [bold]{eff_state_path.name}[/bold] (saves every {checkpoint_interval} ideas)\n"
            f"• Auto-Continue (Resume): [bold green]{'Enabled' if resume else 'Disabled'}[/bold green]",
            title="bookeeper postprocess-ideas",
            border_style="cyan",
        )
    )

    store = ConceptGraphStore()
    with console.status(f"[bold cyan]Loading Knowledge Graph from {target_graph.name}...[/bold cyan]", spinner="dots"):
        store.load(target_graph)
    console.print(f"[bold green]✓[/bold green] [dim]Loaded Knowledge Graph: {len(store.graph.nodes)} nodes, {len(store.graph.edges)} edges[/dim]")

    postprocessor = IdeaPostProcessor.from_settings(
        cfg,
        model_name=eff_model,
        similarity_threshold=eff_thresh,
        state_file=eff_state_path,
        checkpoint_interval=checkpoint_interval,
        resume=resume,
    )
    if min_occurrences is not None:
        postprocessor.min_occurrences = eff_min_occ

    report = postprocessor.process_graph(store, console=console)

    dest_graph = Path(output_graph).expanduser().resolve() if output_graph else target_graph
    store.save(dest_graph)

    # Refresh Obsidian vault & GraphML
    vault_dest = Path(export_obsidian) if export_obsidian else output_dir / "obsidian_vault"
    graphml_dest = output_dir / "knowledge_graph.graphml"
    obs = ObsidianExporter(vault_dest)
    obs.export(store, clean=False)
    gml = GraphMLExporter(graphml_dest)
    gml.export(store)

    console.print(
        f"\n[bold green]✓ Dual-Pass Idea Aggregation & Synthesis Complete ({report.duration_seconds:.2f}s):[/bold green]\n"
        f"• Total Initial Concepts: [bold]{report.total_input_concepts}[/bold]\n"
        f"• Resolved Clusters: [bold]{report.total_clusters}[/bold] (Merged: [yellow]{report.merged_clusters_count}[/yellow])\n"
        f"• Synthesized Multi-Occurrence Ideas: [bold green]{report.synthesized_count}[/bold green]\n"
        f"• Bypassed Single-Occurrence Ideas: [dim]{report.single_occurrence_clusters_count}[/dim]\n"
        f"• Resumed from State Checkpoint: [bold cyan]{report.resumed_count}[/bold cyan]\n"
        f"• Provenance & Relational Edges Added: [bold cyan]{report.edges_added_count}[/bold cyan]\n"
        f"• Updated Knowledge Graph saved to: [bold]{dest_graph}[/bold]\n"
        f"• State Checkpoint saved to: [bold]{eff_state_path}[/bold]\n"
    )


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
    books_dir: Optional[str] = typer.Option(
        None,
        "--books-dir",
        "--download-dir",
        "--source-dir",
        help="Path to folder containing downloaded books/papers (e.g. archive_downloader/downloads) to use as source instead of Calibre database.",
    ),
    timeout: int = typer.Option(
        60, "--request-timeout", "--timeout", help="HTTP timeout in seconds for Ollama LLM requests (default: 60s)."
    ),
    percent: Optional[float] = typer.Option(
        None, "--percent", "-p", help="Percentage of ideas to randomly verify (default: from config.yaml or 1.0)."
    ),
    mode: Optional[str] = typer.Option(
        None, "--mode", "-m", help="Verification mode (default: from config.yaml or 'ideas')."
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
    dual_pass: Optional[bool] = typer.Option(
        None,
        "--dual-pass/--no-dual-pass",
        "--postprocess-ideas/--no-postprocess-ideas",
        help="Perform second-pass idea consolidation across multi-occurrence ideas (default: config.yaml dual_pass.enabled).",
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

    eff_cfg = _get_effective_settings(config_path, calibre_path, model=model, books_dir=books_dir)
    enabled_verifiers = [v for v in eff_cfg.verifiers if v.enabled]
    is_multi_verif = (mode is None and len(enabled_verifiers) > 1)

    effective_percent = percent if percent is not None else (eff_cfg.verification.percent if eff_cfg.verification.percent is not None else 1.0)
    effective_mode = mode or eff_cfg.verification.mode or "ideas"

    verif_desc = (
        f"[bold green]Multi-Verification[/bold green] ({len(enabled_verifiers)} verifiers: {', '.join(v.mode for v in enabled_verifiers)})"
        if is_multi_verif
        else f"[bold green]{effective_percent}%[/bold green] (mode: [cyan]{effective_mode}[/cyan])"
    )

    console.print(
        Panel.fit(
            f"[bold cyan]🧪 Test Run Pipeline Initiated[/bold cyan]\n"
            f"• Target Books: [bold green]{book_cnt}[/bold green] (from scratch)\n"
            f"• Build Parameters: [dim]--from-scratch --no-lightrag --clean-export --timeout {timeout}[/dim]\n"
            f"• Verification: {verif_desc}",
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
            books_dir=books_dir,
            config_path=config_path,
            from_scratch=True,
            enable_lightrag=False,
            clean_export=True,
            request_timeout=timeout,
            limit=book_cnt,
            model=model,
            export_neo4j=export_neo4j,
            skip_warmup=skip_warmup,
            dual_pass=dual_pass,
        )
    except typer.Exit as e:
        if e.exit_code != 0:
            console.print(f"[bold red]Stage 1 build-graph exited with code {e.exit_code}. Aborting test-run.[/bold red]")
            raise

    # Verify that Stage 1 successfully populated books in the knowledge graph
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
    stage2_title = (
        f"Multi-Verification: {len(enabled_verifiers)} verifiers ({', '.join(v.mode for v in enabled_verifiers)})"
        if is_multi_verif
        else f"{effective_percent}%, mode: {effective_mode}"
    )
    console.print(
        f"\n[bold yellow]═══ Stage 2/2: Verifying Knowledge Graph Integrity ({stage2_title}) ═══[/bold yellow]\n"
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
    books_dir: Optional[str] = typer.Option(
        None,
        "--books-dir",
        "--download-dir",
        "--source-dir",
        help="Path to folder containing downloaded books/papers (e.g. archive_downloader/downloads) to use as source instead of Calibre database.",
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
    books_dir = _resolve_opt(books_dir)

    cfg = _get_effective_settings(config_path, calibre_path, books_dir=books_dir)
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

    # Fetch authors if available
    authors_by_id = {}
    client = get_book_client(cfg, calibre_path=calibre_path, books_dir=books_dir)
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

            rebuild_pubdate = s_data.get("publication_date")
            store.add_book(bid, title=btitle, author=bauthor, publication_date=rebuild_pubdate)

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


@app.command("backup")
def backup_command(
    destination: Optional[Path] = typer.Argument(
        None,
        help="Optional destination archive file path (e.g. backup.zip, backup.7z) or target directory.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        "--destination",
        help="Alternative option for destination archive path or target directory.",
    ),
    format_type: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Archive compression format: 'zip' or '7z'. Default: auto-detected from filename or 'zip'.",
    ),
    backup_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Backup type: 'knowledge' (default) or 'full' (includes original indexed source books).",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Create a full backup containing knowledge artifacts and original indexed source books.",
    ),
    books_dir: Optional[Path] = typer.Option(
        None,
        "--books-dir",
        "-b",
        help="Source directory containing books/papers (default: books_dir in config or archive_downloader/downloads).",
    ),
    all_books: bool = typer.Option(
        False,
        "--all-books",
        help="For full backup: include all books in books-dir rather than only indexed books.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Source directory containing knowledge artifacts to backup (default: configured output_dir).",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml.",
    ),
    calibre_path: Optional[Path] = typer.Option(
        None,
        "--calibre-path",
        help="Path to Calibre library.",
    ),
    no_chunks: bool = typer.Option(
        False,
        "--no-chunks",
        help="Exclude cached chunks directory from the backup archive.",
    ),
    no_obsidian: bool = typer.Option(
        False,
        "--no-obsidian",
        help="Exclude Obsidian vault notes from the backup archive.",
    ),
    no_states: bool = typer.Option(
        False,
        "--no-states",
        help="Exclude per-book state checkpoints (.book_states) from the backup archive.",
    ),
    no_lightrag: bool = typer.Option(
        False,
        "--no-lightrag",
        help="Exclude LightRAG database directory from the backup archive.",
    ),
):
    """
    Create a compressed backup archive (.zip or .7z) of the current knowledge graph,
    Obsidian vault, book states, chunks, checkpoints, and optionally source books (full backup).
    """
    destination = _resolve_opt(destination)
    output = _resolve_opt(output)
    format_type = _resolve_opt(format_type)
    backup_type = _resolve_opt(backup_type)
    full = _resolve_opt(full, False)
    books_dir = _resolve_opt(books_dir)
    all_books = _resolve_opt(all_books, False)
    output_dir = _resolve_opt(output_dir)
    config_path = _resolve_opt(config_path)
    calibre_path = _resolve_opt(calibre_path)
    no_chunks = _resolve_opt(no_chunks, False)
    no_obsidian = _resolve_opt(no_obsidian, False)
    no_states = _resolve_opt(no_states, False)
    no_lightrag = _resolve_opt(no_lightrag, False)

    target_dest = destination or output

    cfg = _get_effective_settings(config_path, calibre_path)
    source_dir = Path(output_dir).expanduser().resolve() if output_dir else cfg.resolved_output_dir

    if not source_dir.is_dir():
        console.print(f"[bold red]Source output directory not found:[/bold red] {source_dir}")
        raise typer.Exit(1)

    effective_type = "full" if (full or (backup_type and backup_type.lower() == "full")) else "knowledge"
    effective_books_dir = None
    if effective_type == "full":
        if books_dir:
            effective_books_dir = Path(books_dir).expanduser().resolve()
        elif cfg.resolved_books_dir:
            effective_books_dir = cfg.resolved_books_dir
        else:
            default_books = Path("archive_downloader/downloads").resolve()
            effective_books_dir = default_books

        if not effective_books_dir.is_dir():
            console.print(f"[bold red]Source books directory not found:[/bold red] {effective_books_dir}")
            raise typer.Exit(1)

    custom_dirs = {
        "chunks_dir": cfg.resolved_chunks_dir,
        "state_file": cfg.resolved_state_file,
    }

    t0 = time.perf_counter()
    type_desc = "full backup (knowledge + source books)" if effective_type == "full" else "knowledge backup"
    console.print(f"[bold cyan]Creating {type_desc} from [bold white]{source_dir}[/bold white]...[/bold cyan]")
    if effective_type == "full" and effective_books_dir:
        console.print(f"[bold cyan]Source books directory: [bold white]{effective_books_dir}[/bold white][/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        backup_task = progress.add_task("Archiving files...", total=1)

        def _on_progress(completed: int, total: int, fname: str):
            short_name = (fname[:32] + "...") if len(fname) > 35 else fname
            progress.update(
                backup_task,
                total=total,
                completed=completed,
                description=f"Archiving: [cyan]{short_name}[/cyan]",
            )

        try:
            result = create_backup(
                source_dir=source_dir,
                destination=Path(target_dest) if target_dest else None,
                format_type=format_type,
                backup_type=effective_type,
                books_dir=effective_books_dir,
                indexed_only=not all_books,
                include_chunks=not no_chunks,
                include_obsidian=not no_obsidian,
                include_states=not no_states,
                include_lightrag=not no_lightrag,
                custom_dirs=custom_dirs,
                progress_callback=_on_progress,
            )
        except Exception as e:
            console.print(f"[bold red]Failed to create backup:[/bold red] {e}")
            raise typer.Exit(1)

    elapsed = time.perf_counter() - t0
    dest_path = result["destination"]
    comp_bytes = result["compressed_bytes"]
    uncomp_bytes = result["uncompressed_bytes"]
    comp_ratio = ((1.0 - (comp_bytes / uncomp_bytes)) * 100) if uncomp_bytes > 0 else 0.0

    table_title = "Full Backup Summary" if effective_type == "full" else "Knowledge Backup Summary"
    table = Table(title=table_title, border_style="green")
    table.add_column("Property", style="bold")
    table.add_column("Details", style="cyan")

    table.add_row("Backup Archive", str(dest_path))
    table.add_row("Backup Type", result.get("backup_type", effective_type).upper())
    table.add_row("Format", result["format"].upper())
    if effective_type == "full" and effective_books_dir:
        table.add_row("Books Source", str(effective_books_dir))
        table.add_row("Indexed Only", "Yes" if not all_books else "No (All Books)")
        table.add_row("Book Files Archived", f"{result.get('books_count', 0):,}")
    table.add_row("Total Files Archived", f"{result['total_files']:,}")
    table.add_row("Uncompressed Data Size", f"{uncomp_bytes / (1024 * 1024):.2f} MB ({uncomp_bytes:,} bytes)")
    table.add_row("Compressed Archive Size", f"{comp_bytes / (1024 * 1024):.2f} MB ({comp_bytes:,} bytes)")
    table.add_row("Compression Space Saved", f"{comp_ratio:.1f}%")
    table.add_row("Duration", f"{elapsed:.2f}s")
    console.print("\n", table)
    console.print(f"[bold green]✓ Successfully created {type_desc} at:[/bold green] [bold cyan]{dest_path}[/bold cyan]\n")


@app.command("restore")
def restore_command(
    archive_path: Optional[Path] = typer.Argument(
        None,
        help="Path to the backup archive (.zip or .7z) to restore from.",
    ),
    from_archive: Optional[Path] = typer.Option(
        None,
        "--from",
        "-i",
        "--input",
        "--archive",
        help="Alternative option for backup archive path.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Target directory to restore knowledge artifacts into (default: configured output_dir).",
    ),
    books_dir: Optional[Path] = typer.Option(
        None,
        "--books-dir",
        "-b",
        help="Target directory to restore source books into for full backups (default: configured books_dir or archive_downloader/downloads).",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml.",
    ),
    calibre_path: Optional[Path] = typer.Option(
        None,
        "--calibre-path",
        help="Path to Calibre library.",
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Wipe existing knowledge graph, Obsidian vault, chunks, and book states in destination before restoring.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Inspect archive contents and display summary table without writing files.",
    ),
):
    """
    Restore knowledge graph, Obsidian vault, book states, chunks, checkpoints,
    and original source books (for full backups) from a backup archive (.zip or .7z).
    """
    archive_path = _resolve_opt(archive_path)
    from_archive = _resolve_opt(from_archive)
    output_dir = _resolve_opt(output_dir)
    books_dir = _resolve_opt(books_dir)
    config_path = _resolve_opt(config_path)
    calibre_path = _resolve_opt(calibre_path)
    clean = _resolve_opt(clean, False)
    dry_run = _resolve_opt(dry_run, False)

    target_archive = archive_path or from_archive
    if not target_archive:
        console.print("[bold red]Please specify the archive file path to restore from.[/bold red]")
        console.print("[dim]Usage: bookeeper restore <path_to_backup.zip|.7z> [--output-dir <path>] [--books-dir <path>] [--clean][/dim]")
        raise typer.Exit(1)

    archive_file = Path(target_archive).expanduser().resolve()
    if not archive_file.is_file():
        console.print(f"[bold red]Backup archive not found:[/bold red] {archive_file}")
        raise typer.Exit(1)

    cfg = _get_effective_settings(config_path, calibre_path)
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else cfg.resolved_output_dir
    explicit_books_dir = Path(books_dir).expanduser().resolve() if books_dir else None

    if dry_run:
        console.print(f"[bold cyan]Inspecting archive [bold white]{archive_file}[/bold white] (Dry Run)...[/bold cyan]")
        try:
            info = inspect_backup(archive_file)
        except Exception as e:
            console.print(f"[bold red]Failed to inspect archive:[/bold red] {e}")
            raise typer.Exit(1)

        is_full = info.get("backup_type") == "full" or info.get("categories", {}).get("books", 0) > 0
        computed_books_dir = None
        if is_full:
            computed_books_dir = resolve_target_books_dir(
                target_dir=target_dir,
                metadata=info.get("metadata"),
                explicit_books_dir=explicit_books_dir,
            )

        table = Table(title=f"Archive Inspection: {archive_file.name}", border_style="cyan")
        table.add_column("Category", style="bold")
        table.add_column("Count", justify="right", style="cyan")
        table.add_row("Backup Type", info.get("backup_type", "knowledge").upper(), style="bold magenta")
        table.add_row("Target Directory (Knowledge)", str(target_dir))
        if is_full and computed_books_dir:
            table.add_row("Target Directory (Books)", str(computed_books_dir))
        for cat, cnt in info["categories"].items():
            table.add_row(cat.replace("_", " ").title(), str(cnt))
        table.add_row("Total Files", f"{info['total_files']:,}", style="bold green")
        table.add_row("Compressed Size", f"{info['compressed_bytes'] / (1024 * 1024):.2f} MB")
        table.add_row("Uncompressed Size", f"{info['uncompressed_bytes'] / (1024 * 1024):.2f} MB")
        console.print("\n", table)
        return

    t0 = time.perf_counter()
    if clean:
        console.print(f"[bold yellow]Wiping existing artifacts before restore...[/bold yellow]")

    console.print(f"[bold cyan]Restoring backup into [bold white]{target_dir}[/bold white]...[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        restore_task = progress.add_task("Restoring files...", total=1)

        def _on_progress(completed: int, total: int, fname: str):
            short_name = (fname[:32] + "...") if len(fname) > 35 else fname
            progress.update(
                restore_task,
                total=total,
                completed=completed,
                description=f"Restoring: [cyan]{short_name}[/cyan]",
            )

        try:
            result = restore_backup(
                archive_path=archive_file,
                target_dir=target_dir,
                books_dir=explicit_books_dir,
                clean=clean,
                dry_run=False,
                progress_callback=_on_progress,
            )
        except Exception as e:
            console.print(f"[bold red]Failed to restore archive:[/bold red] {e}")
            raise typer.Exit(1)

    elapsed = time.perf_counter() - t0
    res_bytes = result["bytes_restored"]
    res_files = result["files_restored"]
    is_full = result.get("backup_type") == "full"

    table_title = "Full Restoration Summary" if is_full else "Knowledge Restoration Summary"
    table = Table(title=table_title, border_style="green")
    table.add_column("Property", style="bold")
    table.add_column("Details", style="cyan")

    table.add_row("Source Archive", str(archive_file))
    table.add_row("Backup Type", result.get("backup_type", "knowledge").upper())
    table.add_row("Target Directory (Knowledge)", str(target_dir))
    if is_full:
        table.add_row("Target Directory (Books)", str(result.get("target_books_dir") or explicit_books_dir or (target_dir / "archive_downloader" / "downloads")))
    table.add_row("Format", result["format"].upper())
    if clean:
        cl = result.get("cleaned", {})
        table.add_row(
            "Clean Mode",
            f"✓ Purged {cl.get('files_removed', 0)} old file(s), {cl.get('dirs_removed', 0)} old folder(s)",
        )
    table.add_row("Knowledge Files Restored", f"{result.get('knowledge_files_restored', res_files):,}")
    if is_full:
        table.add_row("Book Files Restored", f"{result.get('book_files_restored', 0):,}")
    table.add_row("Total Files Restored", f"{res_files:,}")
    table.add_row("Data Restored", f"{res_bytes / (1024 * 1024):.2f} MB ({res_bytes:,} bytes)")
    table.add_row("Duration", f"{elapsed:.2f}s")

    # If knowledge_graph.json is restored, verify integrity
    kg_file = target_dir / "knowledge_graph.json"
    if kg_file.is_file():
        try:
            temp_store = ConceptGraphStore()
            temp_store.load(kg_file)
            st = temp_store.stats()
            table.add_row(
                "Knowledge Graph Verification",
                f"[bold green]Valid[/bold green] ({st['total_nodes']:,} nodes, {st['total_edges']:,} edges)",
            )
        except Exception as e:
            table.add_row("Knowledge Graph Verification", f"[yellow]Could not parse graph ({e})[/yellow]")

    console.print("\n", table)
    restored_label = "Full backup (knowledge + source books)" if is_full else "Knowledge"
    console.print(f"[bold green]✓ {restored_label} successfully restored to:[/bold green] [bold cyan]{target_dir}[/bold cyan]\n")


@app.command("sync")
def sync_command(
    server: str = typer.Argument(
        ...,
        help="Remote Bookkeeper server address (e.g. 192.168.50.20:8855 or http://192.168.50.20:8855).",
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Perform clean restore on remote server (wipe remote artifacts and Neo4j before importing).",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml configuration file.",
    ),
    books_dir: Optional[Path] = typer.Option(
        None,
        "--books-dir",
        "-b",
        help="Source directory containing books/papers (default: configured books_dir or archive_downloader/downloads).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Source directory containing knowledge artifacts to sync (default: configured output_dir).",
    ),
    calibre_path: Optional[Path] = typer.Option(
        None,
        "--calibre-path",
        help="Path to Calibre library.",
    ),
    format_type: str = typer.Option(
        "zip",
        "--format",
        "-f",
        help="Archive format to package for transmission: 'zip' or '7z' (default: zip).",
    ),
    timeout: int = typer.Option(
        600,
        "--timeout",
        help="HTTP request timeout in seconds for backup transmission and remote restore (default: 600).",
    ),
):
    """
    Synchronize local knowledge graph and books with a remote Bookkeeper server via REST API:
      1. Backup local books and knowledge graph into a compressed archive
      2. Upload archive to remote server REST API (/api/backup/restore) with the --clean flag synchronized,
         which unpacks artifacts, updates Neo4j database, and validates database consistency.
    """
    import tempfile
    import urllib.request
    import urllib.parse
    import urllib.error

    server = _resolve_opt(server)
    clean = _resolve_opt(clean, False)
    config_path = _resolve_opt(config_path)
    books_dir = _resolve_opt(books_dir)
    output_dir = _resolve_opt(output_dir)
    calibre_path = _resolve_opt(calibre_path)
    format_type = _resolve_opt(format_type, "zip")
    timeout = _resolve_opt(timeout, 600)

    cfg = _get_effective_settings(config_path, calibre_path, books_dir=books_dir)

    # Normalize remote server URL
    target_server = server.strip()
    if not target_server.startswith("http://") and not target_server.startswith("https://"):
        target_server = f"http://{target_server}"
    target_server = target_server.rstrip("/")

    console.print(
        Panel.fit(
            f"[bold]Remote Server:[/bold] [bold cyan]{target_server}[/bold cyan]\n"
            f"[bold]Step 1: Local Backup:[/bold] Full Backup (knowledge graph + indexed source books, {format_type.upper()})\n"
            f"[bold]Step 2: Remote Restore:[/bold] POST {target_server}/api/backup/restore (clean={clean})\n"
            f"[bold]Remote Neo4j Update:[/bold] Automatic (server restores graph and updates Neo4j database)\n",
            title="Bookkeeper Sync Plan",
            border_style="cyan",
        )
    )

    t_start = time.perf_counter()

    # -------------------------------------------------------------
    # Step 1: Backup Local Books and Knowledge Graph
    # -------------------------------------------------------------
    console.print("[bold cyan][Step 1/2][/bold cyan] Creating full local backup archive (books + knowledge graph)...")
    src_out_dir = Path(output_dir).expanduser().resolve() if output_dir else cfg.resolved_output_dir
    src_books_dir = Path(books_dir).expanduser().resolve() if books_dir else (cfg.resolved_books_dir or Path("archive_downloader/downloads").resolve())

    if not src_out_dir.is_dir():
        console.print(f"[bold red]Knowledge output directory not found:[/bold red] {src_out_dir}")
        raise typer.Exit(1)

    temp_archive_dir = Path(tempfile.mkdtemp(prefix="bookeeper_sync_"))
    archive_ext = ".7z" if format_type.lower() == "7z" else ".zip"
    temp_archive_file = temp_archive_dir / f"sync_backup{archive_ext}"

    custom_dirs = {
        "chunks_dir": cfg.resolved_chunks_dir,
        "state_file": cfg.resolved_state_file,
    }

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            backup_task = progress.add_task("Archiving local books & graph...", total=1)

            def _on_sync_progress(completed: int, total: int, fname: str):
                short_name = (fname[:32] + "...") if len(fname) > 35 else fname
                progress.update(
                    backup_task,
                    total=total,
                    completed=completed,
                    description=f"Packaging: [cyan]{short_name}[/cyan]",
                )

            backup_res = create_backup(
                source_dir=src_out_dir,
                destination=temp_archive_file,
                format_type=format_type,
                backup_type="full",
                books_dir=src_books_dir if src_books_dir.is_dir() else None,
                indexed_only=True,
                include_chunks=True,
                include_obsidian=True,
                include_states=True,
                include_lightrag=True,
                custom_dirs=custom_dirs,
                progress_callback=_on_sync_progress,
            )

        arch_size_mb = backup_res["compressed_bytes"] / (1024 * 1024)
        console.print(
            f"[bold green]✓ Step 1 Complete: Created local backup archive[/bold green] "
            f"([bold white]{arch_size_mb:.2f} MB[/bold white], {backup_res['total_files']:,} files)\n"
        )
    except Exception as e:
        console.print(f"[bold red]✗ Step 1 Failed (create_backup):[/bold red] {e}")
        if temp_archive_dir.exists():
            import shutil
            shutil.rmtree(temp_archive_dir, ignore_errors=True)
        raise typer.Exit(1)

    # -------------------------------------------------------------
    # Step 2: Transmit and Restore Backup via Remote REST API
    # -------------------------------------------------------------
    restore_url = f"{target_server}/api/backup/restore"
    console.print(f"[bold cyan][Step 2/2][/bold cyan] Uploading backup to remote server and restoring (clean={clean}) at [bold white]{restore_url}[/bold white]...")

    restore_response_data = None
    try:
        archive_bytes = temp_archive_file.read_bytes()
        boundary = f"----BookkeeperSyncBoundary{int(time.time())}"
        
        # Build multipart/form-data payload with synchronized clean flag
        body_parts = []
        # 'clean' field
        body_parts.append(f"--{boundary}\r\n".encode("latin1"))
        body_parts.append(b'Content-Disposition: form-data; name="clean"\r\n\r\n')
        body_parts.append(b"true\r\n" if clean else b"false\r\n")

        # 'file' field
        content_type = "application/x-7z-compressed" if archive_ext == ".7z" else "application/zip"
        body_parts.append(f"--{boundary}\r\n".encode("latin1"))
        body_parts.append(f'Content-Disposition: form-data; name="file"; filename="{temp_archive_file.name}"\r\n'.encode("latin1"))
        body_parts.append(f"Content-Type: {content_type}\r\n\r\n".encode("latin1"))
        body_parts.append(archive_bytes)
        body_parts.append(b"\r\n")
        body_parts.append(f"--{boundary}--\r\n".encode("latin1"))

        multipart_payload = b"".join(body_parts)

        req = urllib.request.Request(
            restore_url,
            data=multipart_payload,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(multipart_payload)),
            },
            method="POST",
        )

        with console.status(f"[bold cyan]Transmitting {arch_size_mb:.2f} MB and executing remote restore & Neo4j sync...[/bold cyan]"):
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_text = resp.read().decode("utf-8")
                restore_response_data = json.loads(resp_text)

        if not restore_response_data.get("success"):
            err_msg = restore_response_data.get("error", "Unknown server error")
            console.print(f"[bold red]✗ Step 2 Failed: Server returned failure:[/bold red] {err_msg}")
            raise typer.Exit(1)

        restored_files = restore_response_data.get("restored_files", 0)
        neo4j_info = restore_response_data.get("neo4j_export") or {}
        nodes_up = neo4j_info.get("nodes_upserted", 0)
        edges_up = neo4j_info.get("edges_upserted", 0)

        console.print(
            f"[bold green]✓ Step 2 Complete: Remote restore and Neo4j synchronization successful![/bold green] "
            f"({restored_files:,} files restored, Neo4j: {nodes_up:,} nodes, {edges_up:,} relationships)\n"
        )
    except urllib.error.HTTPError as e:
        err_detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        console.print(f"[bold red]✗ Step 2 Failed (HTTP {e.code}):[/bold red] {err_detail}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗ Step 2 Failed (REST API):[/bold red] {e}")
        raise typer.Exit(1)
    finally:
        if temp_archive_dir.exists():
            import shutil
            shutil.rmtree(temp_archive_dir, ignore_errors=True)

    elapsed_total = time.perf_counter() - t_start

    # Summary table
    table = Table(title="Bookkeeper Client Sync Summary", border_style="green")
    table.add_column("Phase", style="bold cyan")
    table.add_column("Status / Details", style="green")

    clean_step_txt = "Clean restore (remote database wiped + restored)" if clean else "Additive restore (existing preserved)"
    table.add_row("1. Local Backup Archive", f"{backup_res['total_files']:,} files ({arch_size_mb:.2f} MB)")
    table.add_row("2. Remote REST Restore", f"Restored into {target_server} ({clean_step_txt})")
    table.add_row("Remote Neo4j Sync", f"{neo4j_info.get('nodes_upserted', 0):,} nodes, {neo4j_info.get('edges_upserted', 0):,} edges")
    table.add_row("Total Duration", f"{elapsed_total:.2f}s")

    console.print(table)
    console.print(f"[bold green]✓ Synchronization completed successfully for {target_server}![/bold green]\n")


@app.command("ui")
def ui_command(
    host: str = typer.Option("127.0.0.1", "--host", "-h", envvar=["BOOKEEPER_HOST", "HOST"], help="Host interface to bind the web server (default: 127.0.0.1)."),
    port: int = typer.Option(8000, "--port", "-p", envvar=["BOOKEEPER_PORT", "PORT"], help="HTTP port to listen on (default: 8000)."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", envvar=["BOOKEEPER_CONFIG"], help="Path to config.yaml configuration file."),
):
    """
    Launch the interactive Bookeeper Idea-Based Search Engine Web UI and REST API.
    Provides a central idea search bar, interactive 2-column results pane
    (Cytoscape Neo4j graph on the left, Obsidian-style detail note on the right),
    an observability Debug Tab, and a LightRAG-compatible query API (/query).
    """
    from bookeeper.ui.backend import SearchEngineBackend
    from bookeeper.ui.server import run_server

    cfg = _get_effective_settings(config_path)
    backend = SearchEngineBackend(config=cfg, neo4j_config=cfg.neo4j)

    server = run_server(host=host, port=port, backend=backend, auto_port=True)
    actual_port = server.server_port

    if actual_port != port:
        console.print(f"[yellow]Note: Port {port} was already in use. Switched to available port {actual_port}.[/yellow]")

    console.print(f"\n[bold cyan]Starting Bookeeper Idea-Based Search Engine UI...[/bold cyan]")
    console.print(f"  • [bold white]Web Interface:[/bold white] [bold green]http://{host}:{actual_port}/[/bold green]")
    console.print(f"  • [bold white]Query API:[/bold white] [bold cyan]http://{host}:{actual_port}/query[/bold cyan] [dim](LightRAG compatible)[/dim]")
    console.print(f"  • [bold white]Neo4j URI:[/bold white] {cfg.neo4j.uri} (database: '{cfg.neo4j.database}')")

    # Pre-flight Database Consistency Verification
    try:
        consistency = backend.validate_database_consistency()
        if consistency.get("neo4j_connected"):
            if not consistency.get("is_consistent"):
                discrepancy_lines = "\n".join(f"  • [yellow]{d}[/yellow]" for d in consistency.get("discrepancies", []))
                console.print(Panel(
                    f"[bold yellow]⚠️  DATABASE INCONSISTENCY DETECTED[/bold yellow]\n\n"
                    f"The connected Neo4j database is outdated or out of sync with your local knowledge base:\n"
                    f"{discrepancy_lines}\n\n"
                    f"[bold white]Impact:[/bold white] Search results and graph exploration may return incomplete data.\n"
                    f"[bold white]Resolution:[/bold white] Run [bold cyan]bookeeper rebuild-graph --export-neo4j[/bold cyan] or [bold cyan]bookeeper build-graph --export-neo4j[/bold cyan] to synchronize.",
                    title="[bold yellow]Consistency Warning[/bold yellow]",
                    border_style="yellow",
                ))
            else:
                ls = consistency.get("local_stats", {})
                console.print(f"  • [bold green]Neo4j Status:[/bold green] [green]Synchronized with local KB[/green] ({ls.get('books', 0)} books, {ls.get('chunks', 0):,} chunks, {ls.get('ideas', 0):,} ideas)")
        else:
            console.print(f"  • [bold dim]Neo4j Status:[/bold dim] [dim]Offline / Standalone mode (serving from local knowledge_graph.json)[/dim]")
    except Exception as e:
        console.print(f"  • [dim]Consistency check skipped: {e}[/dim]")

    console.print("[dim]Press Ctrl+C to terminate the web server.[/dim]\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down Bookeeper UI server...[/yellow]")
    finally:
        server.server_close()
        backend.close()


@app.command("search")
def search_command(
    query_text: str = typer.Argument(..., help="Idea name or concept query string to search."),
    mode: str = typer.Option("ideas", "--mode", "-m", help="Search mode: 'ideas', 'deep' (2-hop), or 'exact'."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Maximum ideas to retrieve."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
):
    """
    Search ideas directly in the Bookkeeper Knowledge Graph.
    Retrieves matching ideas and their surrounding neighborhood notes (connected ideas, books, evidence).
    """
    from bookeeper.ui.backend import SearchEngineBackend
    cfg = _get_effective_settings(config_path)
    backend = SearchEngineBackend(config=cfg, neo4j_config=cfg.neo4j)
    try:
        with console.status(f"[bold cyan]Searching ideas for '{query_text}'...[/bold cyan]"):
            result = backend.query_ideas(query=query_text, mode=mode, top_k=top_k)
        console.print(Panel(result["answer"], title=f"[bold green]Bookkeeper Idea Search Results[/bold green]", border_style="green"))
    finally:
        backend.close()


@app.command("lightrag-ui")
def lightrag_ui_command(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host interface to bind the LightRAG server (default: 127.0.0.1)."),
    port: int = typer.Option(9621, "--port", "-p", help="Port for the LightRAG server (default: 9621)."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml configuration file."),
    lightrag_dir: Optional[str] = typer.Option(None, "--lightrag-dir", help="Path to LightRAG storage directory."),
):
    """
    Launch the official LightRAG FastAPI server with embedded WebUI and Workspace query interface.
    Uses Bookeeper's indexed LightRAG storage directory and configured Ollama models.
    """
    from bookeeper.ui.lightrag_server import LightRAGServerManager

    cfg = _get_effective_settings(config_path)
    if lightrag_dir:
        target_lightrag = Path(lightrag_dir).expanduser().resolve()
        if (target_lightrag / "lightrag_storage").is_dir():
            target_lightrag = target_lightrag / "lightrag_storage"
        elif (target_lightrag / "output" / "lightrag_storage").is_dir():
            target_lightrag = target_lightrag / "output" / "lightrag_storage"
    else:
        target_lightrag = cfg.resolved_lightrag_dir
        if not (target_lightrag / "graph_chunk_entity_relation.graphml").is_file():
            # Check for most recent A/B test run storage
            for candidate_root in [Path("ab_test_runs"), Path("output/ab_test_runs")]:
                if candidate_root.is_dir():
                    runs = sorted(candidate_root.glob("run_*/lightrag_storage"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if runs and (runs[0] / "graph_chunk_entity_relation.graphml").is_file():
                        target_lightrag = runs[0].resolve()
                        break

    if not target_lightrag.exists():
        console.print(f"[bold red]LightRAG storage directory not found:[/bold red] {target_lightrag}")
        console.print("[yellow]Please run 'bookeeper ab-benchmark' or 'bookeeper build-graph --lightrag' first to populate the storage.[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Starting Official LightRAG Server & WebUI...[/bold cyan]")
    console.print(f"  • [bold white]Storage Directory:[/bold white] {target_lightrag}")
    console.print(f"  • [bold white]LLM Model:[/bold white] {cfg.llm_model} ({cfg.ollama_base_url})")
    console.print(f"  • [bold white]Embedding Model:[/bold white] {cfg.embedding_model} (dim: {cfg.embedding_dim})")
    console.print(f"  • [bold white]WebUI Interface:[/bold white] [bold green]http://{host}:{port}/webui/[/bold green]")
    console.print(f"  • [bold white]Workspace Query:[/bold white] [bold green]http://{host}:{port}/workspace/[/bold green]")
    console.print(f"  • [bold white]API Documentation:[/bold white] [bold green]http://{host}:{port}/docs[/bold green]")
    console.print("[dim]Press Ctrl+C to terminate the server.[/dim]\n")

    mgr = LightRAGServerManager(config=cfg, working_dir=target_lightrag, host=host, port=port)
    if not mgr.start(wait_ready=True):
        console.print(f"[bold red]Failed to start LightRAG server on port {port}.[/bold red]")
        raise typer.Exit(1)

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down LightRAG server...[/yellow]")
    finally:
        mgr.stop()


@app.command("lightrag-index")
def lightrag_index_command(
    book_cnt: int = typer.Argument(1, metavar="BOOK-CNT", help="Number of books to index with LightRAG native chunking."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    calibre_path: Optional[str] = typer.Option(None, "--calibre-path", help="Calibre library path or SMB mount."),
    books_dir: Optional[str] = typer.Option(None, "--books-dir", help="Path to folder containing books/articles."),
    lightrag_dir: Optional[str] = typer.Option(None, "--lightrag-dir", help="Working directory for LightRAG database."),
    clean: bool = typer.Option(False, "--clean", help="Wipe existing LightRAG database before indexing."),
):
    """
    Index N books into LightRAG directly using raw book text and LightRAG's native chunking.
    Decoupled from Bookkeeper's core concept graph extraction for clean standalone indexing.
    """
    from bookeeper.calibre import get_book_client
    from bookeeper.rag.lightrag_service import LightRAGRawIndexer

    cfg = _get_effective_settings(config_path, calibre_path, books_dir=books_dir)
    client = get_book_client(cfg, calibre_path=calibre_path, books_dir=books_dir)
    all_books = client.list_books()

    if not all_books:
        console.print("[bold red]No books found in the specified book source.[/bold red]")
        raise typer.Exit(1)

    target_books_meta = all_books[:book_cnt]
    export_dir = cfg.resolved_output_dir / "cache" / "books"
    export_dir.mkdir(parents=True, exist_ok=True)

    books_to_index = []
    for b in target_books_meta:
        bid = b.get("id")
        bpath = b.get("path")
        if not bpath or not Path(bpath).is_file():
            try:
                bpath = client.export_book(bid, target_dir=export_dir)
            except Exception as e:
                console.print(f"[yellow]Could not export book #{bid}: {e}[/yellow]")
                continue
        if bpath and Path(bpath).is_file():
            books_to_index.append({"id": bid, "title": b.get("title", f"Book #{bid}"), "path": str(bpath)})

    if not books_to_index:
        console.print("[bold red]No valid book files could be prepared for LightRAG indexing.[/bold red]")
        raise typer.Exit(1)

    target_dir = Path(lightrag_dir).expanduser().resolve() if lightrag_dir else cfg.resolved_lightrag_dir
    indexer = LightRAGRawIndexer(settings=cfg, working_dir=target_dir, console=console)
    indexer.index_books(books=books_to_index, clean=clean, show_progress=True)


@app.command("generate-questions")
def generate_questions_command(
    book_cnt: int = typer.Argument(1, metavar="BOOK-CNT", help="Number of books to generate evaluation questions for."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Destination path for question library JSON (default: output/question_library.json)."),
    questions_per_book: int = typer.Option(5, "--questions-per-book", "-q", help="Number of questions to generate per book."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Verification LLM model for question synthesis (default: from config)."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    calibre_path: Optional[str] = typer.Option(None, "--calibre-path", help="Calibre library path or SMB mount."),
    books_dir: Optional[str] = typer.Option(None, "--books-dir", help="Path to folder containing books/articles."),
    timeout: int = typer.Option(180, "--timeout", "-t", help="Timeout in seconds per LLM request for question synthesis (default: 180s)."),
):
    """
    Generate an evaluation question library for N books using the capable verification model.
    Questions span concept definitions, relational trade-offs, factual mechanics, and thematic synthesis.
    Saved to JSON for human review and revision prior to benchmarking.
    """
    from bookeeper.calibre import get_book_client
    from bookeeper.evaluation.question_generator import QuestionGenerator

    cfg = _get_effective_settings(config_path, calibre_path, books_dir=books_dir)
    client = get_book_client(cfg, calibre_path=calibre_path, books_dir=books_dir)
    all_books = client.list_books()

    if not all_books:
        console.print("[bold red]No books found in the specified book source.[/bold red]")
        raise typer.Exit(1)

    target_books_meta = all_books[:book_cnt]
    export_dir = cfg.resolved_output_dir / "cache" / "books"
    export_dir.mkdir(parents=True, exist_ok=True)

    books_to_eval = []
    for b in target_books_meta:
        bid = b.get("id")
        bpath = b.get("path")
        if not bpath or not Path(bpath).is_file():
            try:
                bpath = client.export_book(bid, target_dir=export_dir)
            except Exception:
                pass
        books_to_eval.append({
            "id": bid,
            "title": b.get("title", f"Book #{bid}"),
            "path": str(bpath) if bpath and Path(bpath).is_file() else None,
        })

    out_file = Path(output).expanduser().resolve() if output else (cfg.resolved_output_dir / "question_library.json")
    generator = QuestionGenerator(settings=cfg, model_name=model, timeout=timeout, console=console)

    library = generator.generate_library(
        books=books_to_eval,
        questions_per_book=questions_per_book,
        output_file=out_file,
        show_progress=True,
    )

    console.print(
        f"\n[bold green]✓ Question library successfully created:[/bold green] "
        f"[bold white]{len(library.questions)}[/bold white] questions saved to [bold cyan]{out_file}[/bold cyan]\n"
    )


@app.command("ab-benchmark")
def ab_benchmark_command(
    book_cnt: int = typer.Argument(1, metavar="BOOK-CNT", help="Number of books to benchmark across Bookkeeper and LightRAG."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    calibre_path: Optional[str] = typer.Option(None, "--calibre-path", help="Calibre library path or SMB mount."),
    books_dir: Optional[str] = typer.Option(None, "--books-dir", help="Path to folder containing books/articles."),
    questions_file: Optional[str] = typer.Option(None, "--questions", "-q", help="Path to existing question library JSON file (if omitted, questions are generated)."),
    questions_per_book: int = typer.Option(5, "--questions-per-book", help="Questions to generate per book if --questions is not provided."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Target output session folder for benchmark artifacts and reports."),
    from_phase: int = typer.Option(1, "--from-phase", help="Starting benchmark phase: 1 (Bookkeeper Analysis), 2 (LightRAG Analysis), 3 (Benchmark Queries). Default: 1."),
    phase: Optional[str] = typer.Option(None, "--phase", help="Execute ONLY a single specific phase: '1'|'bookkeeper', '2'|'lightrag', '3'|'benchmark'."),
    resume: bool = typer.Option(True, "--resume/--no-resume", "--continue/--no-continue", help="Resume from detected safe points (default: true)."),
    clean: bool = typer.Option(False, "--clean", help="Discard previous safe points and rebuild from scratch."),
    lightrag_mode: str = typer.Option("hybrid", "--lightrag-mode", help="Query mode for LightRAG ('hybrid', 'local', 'global')."),
    bookkeeper_mode: str = typer.Option("ideas", "--bookkeeper-mode", help="Query mode for Bookkeeper ('ideas', 'deep')."),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Dedicated LLM model for judging (default: verification model from config)."),
    dual_pass: Optional[bool] = typer.Option(None, "--dual-pass/--no-dual-pass", "--postprocess-ideas/--no-postprocess-ideas", help="Execute two-pass idea postprocessing for Bookkeeper (default: from config.yaml)."),
    skip_lightrag_indexing: bool = typer.Option(False, "--skip-lightrag-indexing", help="Skip re-indexing LightRAG if storage already exists."),
    skip_bookkeeper_build: bool = typer.Option(False, "--skip-bookkeeper-build", help="Skip rebuilding Bookkeeper graph if already built."),
):
    """
    Run an end-to-end comparative A/B benchmark between Bookkeeper and LightRAG:
    - Phase 1: Bookkeeper indexes N books with post-processing (Safe Point 1).
    - Phase 2: LightRAG indexes the exact same N books using raw text and native chunking (Safe Point 2).
    - Phase 3: Synthesizes / loads question library, executes dual queries, and judges responses (Safe Point 3).
    Supports resuming or starting from any phase with safe-point checkpoints.
    """
    from bookeeper.calibre import get_book_client
    from bookeeper.evaluation.benchmark import ABBenchmarkRunner
    from bookeeper.evaluation.question_generator import QuestionLibrary

    cfg = _get_effective_settings(config_path, calibre_path, books_dir=books_dir)
    client = get_book_client(cfg, calibre_path=calibre_path, books_dir=books_dir)
    all_books = client.list_books()

    if not all_books:
        console.print("[bold red]No books found in the specified book source.[/bold red]")
        raise typer.Exit(1)

    target_books_meta = all_books[:book_cnt]
    export_dir = cfg.resolved_output_dir / "cache" / "books"
    export_dir.mkdir(parents=True, exist_ok=True)

    books_to_process = []
    for b in target_books_meta:
        bid = b.get("id")
        bpath = b.get("path")
        if not bpath or not Path(bpath).is_file():
            try:
                bpath = client.export_book(bid, target_dir=export_dir)
            except Exception:
                pass
        if bpath and Path(bpath).is_file():
            books_to_process.append({"id": bid, "title": b.get("title", f"Book #{bid}"), "path": str(bpath)})

    if not books_to_process:
        console.print("[bold red]No valid book files available for benchmarking.[/bold red]")
        raise typer.Exit(1)

    # Resolve target output directory
    if output_dir:
        target_output_dir = Path(output_dir).expanduser().resolve()
    elif not clean and (resume or from_phase >= 2):
        latest_session = ABBenchmarkRunner.find_latest_session_dir()
        if latest_session:
            target_output_dir = latest_session
            console.print(f"[bold cyan]Resuming existing benchmark session:[/bold cyan] [dim]{target_output_dir}[/dim]")
        else:
            target_output_dir = cfg.resolved_output_dir / "ab_test_runs" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        target_output_dir = cfg.resolved_output_dir / "ab_test_runs" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    runner = ABBenchmarkRunner(settings=cfg, output_dir=target_output_dir, console=console)

    # 1. Bookkeeper Graph Build (Phase 1)
    should_run_phase1 = (from_phase <= 1) and (phase in (None, "all", "1", "bookkeeper")) and not skip_bookkeeper_build

    if not should_run_phase1:
        if runner.is_phase1_complete():
            console.print(
                f"[bold green]✓ Phase 1 (Bookkeeper Analysis) Safe Point detected:[/bold green] Reusing existing Knowledge Graph. Skipping Phase 1."
            )
        else:
            console.print(f"[dim]Phase 1 skipped (starting from phase {from_phase}).[/dim]")
    else:
        # Check if pre-computed graph already exists and resume is enabled
        existing_kg = cfg.resolved_output_dir / "knowledge_graph.json"
        can_reuse_existing = resume and not clean and (runner.is_phase1_complete() or existing_kg.is_file())

        if can_reuse_existing:
            console.print(
                f"[bold green]✓ Phase 1 (Bookkeeper Analysis) Safe Point detected:[/bold green] "
                f"Found existing Knowledge Graph at [dim]{existing_kg}[/dim]. Skipping rebuild. (Pass --clean to rebuild from scratch)."
            )
            runner.record_safe_point_1(existing_kg, len(books_to_process), books=books_to_process)
        else:
            console.print(
                f"\n[bold yellow]═══ Phase 1/3: Building Bookkeeper Knowledge Graph for {len(books_to_process)} books ═══[/bold yellow]\n"
            )
            try:
                build_graph(
                    calibre_path=calibre_path,
                    books_dir=books_dir,
                    config_path=config_path,
                    from_scratch=True,
                    enable_lightrag=False,
                    clean_export=True,
                    limit=book_cnt,
                    dual_pass=dual_pass,
                )
            except typer.Exit as e:
                if e.exit_code != 0:
                    console.print(f"[bold red]Bookkeeper build-graph failed with code {e.exit_code}. Aborting benchmark.[/bold red]")
                    raise
            runner.record_safe_point_1(existing_kg, len(books_to_process), books=books_to_process)

        if phase in ("1", "bookkeeper"):
            console.print(f"\n[bold green]✓ Phase 1 (Bookkeeper Analysis) complete. Exiting as requested by --phase {phase}.[/bold green]\n")
            return

    # 2. Load Question Library if provided
    ql = None
    if questions_file:
        qp = Path(questions_file).expanduser().resolve()
        if qp.is_file():
            ql = QuestionLibrary.load(qp)

    # 3. Execute Benchmark Pipeline (Phase 2 & Phase 3)
    report = runner.run_benchmark(
        books=books_to_process,
        question_library=ql,
        questions_per_book=questions_per_book,
        lightrag_mode=lightrag_mode,
        bookkeeper_mode=bookkeeper_mode,
        judge_model=judge_model,
        from_phase=from_phase,
        phase=phase,
        clean=clean,
        resume=resume,
        skip_lightrag_indexing=skip_lightrag_indexing,
    )

    if report.total_questions > 0:
        console.print(
            Panel.fit(
                f"[bold green]✓ A/B Benchmark Complete[/bold green]\n"
                f"• Bookkeeper Wins: [bold cyan]{report.bookkeeper_wins}[/bold cyan] | "
                f"LightRAG Wins: [bold yellow]{report.lightrag_wins}[/bold yellow] | "
                f"Ties: [bold white]{report.ties}[/bold white]\n"
                f"• Bookkeeper Pass Rate: [bold green]{report.bookkeeper_overall_pass_rate}%[/bold green] "
                f"({report.bookkeeper_avg_latency:.4f}s avg latency)\n"
                f"• LightRAG Pass Rate: [bold yellow]{report.lightrag_overall_pass_rate}%[/bold yellow] "
                f"({report.lightrag_avg_latency:.4f}s avg latency)\n"
                f"• Summary Report: [bold cyan]{runner.output_dir / 'ab_test_summary.md'}[/bold cyan]",
                title="Benchmark Results",
                border_style="green",
            )
        )


if __name__ == "__main__":
    app()

