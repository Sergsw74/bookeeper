"""
LightRAG Raw Indexer and Standalone Service Submodule.

Provides:
1. Ingestion of raw book text into LightRAG with LightRAG's native natural token-based chunking.
2. Direct programmatic querying and standalone FastAPI server lifecycle management.
3. Decoupled operation for benchmarking and AB comparisons against Bookkeeper.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from bookeeper.calibre.parser import BookParser
from bookeeper.config import Settings
from bookeeper.rag.lightrag_engine import LightRAGEngine
from bookeeper.ui.lightrag_server import LightRAGServerManager

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
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
except ImportError:
    Console = None
    Progress = None


class LightRAGRawIndexer:
    """
    Ingests books into LightRAG directly from raw text, delegating chunking
    and entity extraction entirely to LightRAG's native pipeline.
    """

    def __init__(
        self,
        settings: Settings,
        working_dir: Optional[Union[Path, str]] = None,
        console: Optional[Any] = None,
    ):
        self.settings = settings
        self.working_dir = Path(working_dir or settings.resolved_lightrag_dir).expanduser().resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.console = console or (Console() if Console is not None else None)
        self.engine: Optional[LightRAGEngine] = None

    def get_engine(self) -> LightRAGEngine:
        """Get or initialize the underlying LightRAGEngine."""
        if self.engine is None:
            self.engine = LightRAGEngine.from_settings(
                self.settings,
                working_dir=self.working_dir,
            )
            self.engine.initialize()
        return self.engine

    def extract_raw_book_text(self, file_path: Union[Path, str]) -> Tuple[str, int]:
        """
        Extract clean full text from a book file using BookParser.
        Returns (raw_text, char_count).
        """
        p = Path(file_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Book file does not exist: {p}")

        ext = p.suffix.lower()
        if ext in (".txt", ".md"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            text = BookParser.repair_mojibake(text)
            return text, len(text)

        sections = BookParser.parse(p)
        content_blocks = []
        for s in sections:
            txt = s.text.strip() if s.text else ""
            if len(txt) < 30:
                continue
            heading = f"\n\n# {s.title}\n" if s.title else "\n\n"
            content_blocks.append(f"{heading}{txt}")

        full_text = "".join(content_blocks).strip()
        return full_text, len(full_text)

    def close(self) -> None:
        """Clean up underlying LightRAGEngine resources."""
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    def index_books(
        self,
        books: List[Dict[str, Any]],
        clean: bool = False,
        show_progress: bool = True,
        on_book_complete: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a list of books into LightRAG with native chunking.
        Supports continuation and safe points: books already indexed are preserved
        unless clean=True is requested.
        """
        import json

        t0 = time.perf_counter()
        engine = self.get_engine()
        indexed_tracker_file = self.working_dir / "indexed_books.json"

        if clean:
            if self.console:
                self.console.print(f"[bold yellow]Purging previous LightRAG storage at {self.working_dir}...[/bold yellow]")
            engine.clean()
            engine.initialize()
            if indexed_tracker_file.exists():
                try:
                    indexed_tracker_file.unlink()
                except Exception:
                    pass

        # Load existing indexed books map to support resumption without re-indexing
        already_indexed_ids: Dict[str, Dict[str, Any]] = {}
        if not clean and indexed_tracker_file.is_file():
            try:
                with open(indexed_tracker_file, "r", encoding="utf-8") as f:
                    saved_entries = json.load(f)
                    for item in saved_entries:
                        did = item.get("doc_id") or f"book_{item.get('id')}"
                        already_indexed_ids[did] = item
            except Exception as e:
                logger.warning(f"Could not load indexed_books.json: {e}")

        # Check LightRAG's internal kv_store_doc_status.json as well
        doc_status_file = self.working_dir / "kv_store_doc_status.json"
        internal_doc_status: Dict[str, Any] = {}
        if not clean and doc_status_file.is_file():
            try:
                with open(doc_status_file, "r", encoding="utf-8") as f:
                    internal_doc_status = json.load(f)
            except Exception:
                pass

        total_books = len(books)
        total_chars = 0
        indexed_books: List[Dict[str, Any]] = []

        use_progress = show_progress and Progress is not None and self.console is not None
        progress_cm = (
            Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                BarColumn(bar_width=28),
                TaskProgressColumn(),
                TextColumn("•"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
            )
            if use_progress
            else None
        )

        with progress_cm if progress_cm else open("/dev/null", "w"):
            task_id = (
                progress_cm.add_task(f"Indexing {total_books} books into LightRAG (native chunking)...", total=total_books)
                if progress_cm
                else None
            )

            for idx, b in enumerate(books, 1):
                bid = b.get("id", idx)
                title = b.get("title", f"Book #{bid}")
                path = b.get("path")
                doc_id = f"book_{bid}"

                if not path:
                    logger.warning(f"Skipping book #{bid} ('{title}'): no valid file path.")
                    if progress_cm and task_id is not None:
                        progress_cm.update(task_id, advance=1)
                    continue

                # Continuation check: Check if book is already successfully indexed
                is_already_done = (
                    doc_id in already_indexed_ids
                    or (doc_id in internal_doc_status and internal_doc_status[doc_id].get("status") == "processed")
                )

                if is_already_done and not clean:
                    existing_meta = already_indexed_ids.get(doc_id, {
                        "id": bid,
                        "title": title,
                        "char_count": internal_doc_status.get(doc_id, {}).get("content_length", 0),
                        "doc_id": doc_id,
                        "status": "already_indexed",
                    })
                    indexed_books.append(existing_meta)
                    total_chars += existing_meta.get("char_count", 0)
                    if self.console and not use_progress:
                        self.console.print(f"[dim green]• Book #{bid} ('{title}') already indexed in LightRAG. Skipping.[/dim green]")
                    if progress_cm and task_id is not None:
                        progress_cm.update(task_id, advance=1, description=f"Reused: [dim]{title[:30]}[/dim]")
                    if on_book_complete:
                        on_book_complete(existing_meta, idx, total_books)
                    continue

                try:
                    if progress_cm and task_id is not None:
                        short_title = (title[:32] + "...") if len(title) > 35 else title
                        progress_cm.update(task_id, description=f"Extracting & Indexing: [bold white]{short_title}[/bold white]")

                    raw_text, char_count = self.extract_raw_book_text(path)
                    if not raw_text:
                        logger.warning(f"No extractable text found in '{title}' ({path}).")
                        continue

                    # Insert raw text directly into LightRAG with native token-based chunking
                    engine.insert_text(raw_text, doc_id=doc_id, file_path=title)

                    total_chars += char_count
                    b_meta = {
                        "id": bid,
                        "title": title,
                        "char_count": char_count,
                        "doc_id": doc_id,
                        "status": "indexed",
                    }
                    indexed_books.append(b_meta)
                    already_indexed_ids[doc_id] = b_meta

                    # Persist indexed books metadata checkpoint
                    try:
                        with open(indexed_tracker_file, "w", encoding="utf-8") as f:
                            json.dump(list(already_indexed_ids.values()), f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

                    if on_book_complete:
                        on_book_complete(b_meta, idx, total_books)

                except Exception as e:
                    logger.error(f"Failed to index book #{bid} ('{title}'): {e}", exc_info=True)
                    if self.console:
                        self.console.print(f"[bold red]Error indexing '{title}':[/bold red] {e}")

                finally:
                    if progress_cm and task_id is not None:
                        progress_cm.update(task_id, advance=1)

        elapsed = time.perf_counter() - t0
        summary = {
            "total_books_requested": total_books,
            "total_books_indexed": len(indexed_books),
            "total_characters": total_chars,
            "working_dir": str(self.working_dir),
            "elapsed_seconds": round(elapsed, 2),
            "books": indexed_books,
        }

        if self.console:
            self.console.print(
                f"\n[bold green]✓ LightRAG native ingestion complete:[/bold green] "
                f"{len(indexed_books)}/{total_books} books ({total_chars:,} chars) indexed in {round(elapsed, 1)}s."
            )

        return summary


class LightRAGService:
    """
    Query interface and standalone FastAPI server manager for LightRAG.
    """

    def __init__(
        self,
        settings: Settings,
        working_dir: Optional[Union[Path, str]] = None,
    ):
        self.settings = settings
        self.working_dir = Path(working_dir or settings.resolved_lightrag_dir).expanduser().resolve()
        self._engine: Optional[LightRAGEngine] = None
        self._server_mgr: Optional[LightRAGServerManager] = None

    def get_engine(self) -> LightRAGEngine:
        """Get or initialize the underlying LightRAGEngine."""
        if self._engine is None:
            self._engine = LightRAGEngine.from_settings(
                self.settings,
                working_dir=self.working_dir,
            )
            self.engine_init_sync()
        return self._engine

    def engine_init_sync(self) -> None:
        if self._engine and not self._engine._initialized:
            self._engine.initialize()

    def query(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 40,
        chunk_top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute query against LightRAG and measure latency.
        """
        t0 = time.perf_counter()
        engine = self.get_engine()
        response_text = engine.query(
            query=query,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
        )
        elapsed = time.perf_counter() - t0

        return {
            "query": query,
            "response": response_text,
            "mode": mode,
            "response_time": round(elapsed, 4),
            "working_dir": str(self.working_dir),
            "engine": "lightrag",
        }

    def start_api_server(
        self,
        host: str = "127.0.0.1",
        port: int = 9621,
        wait_ready: bool = True,
    ) -> LightRAGServerManager:
        """Start the official embedded LightRAG FastAPI server as a daemon."""
        if self._server_mgr is None:
            self._server_mgr = LightRAGServerManager(
                config=self.settings,
                working_dir=self.working_dir,
                host=host,
                port=port,
            )
        self._server_mgr.start(wait_ready=wait_ready)
        return self._server_mgr

    def stop_api_server(self) -> None:
        """Stop running API server if active."""
        if self._server_mgr is not None:
            self._server_mgr.stop()
            self._server_mgr = None

    def close(self) -> None:
        """Clean up LightRAG engine and server manager resources."""
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self.stop_api_server()

