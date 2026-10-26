"""
Persistent progress and checkpoint manager for book processing tasks.
Allows resuming interrupted or failed runs without repeating already processed books.
"""

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Tracks and persists completion and failure status of books across CLI operations.
    Thread-safe and atomic file updates prevent corruption on unexpected termination.
    """

    def __init__(self, state_path: Path | str):
        self.state_path = Path(state_path).expanduser().resolve()
        self.state: Dict[str, Any] = {
            "version": 1,
            "operations": {},
        }
        self.load()

    def load(self) -> None:
        """Load state from disk, gracefully handling non-existent or corrupted files."""
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.state = data
                    if "operations" not in self.state:
                        self.state["operations"] = {}
        except Exception as e:
            logger.warning(f"Could not load state from {self.state_path} ({e}); starting with clean state.")

    def save(self) -> None:
        """Atomically persist state to disk using a temporary file rename."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state["updated_at"] = time.time()
            dir_name = self.state_path.parent
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                json.dump(self.state, tf, indent=2, ensure_ascii=False)
                temp_name = tf.name
            os.replace(temp_name, self.state_path)
        except Exception as e:
            logger.warning(f"Failed to persist state to {self.state_path}: {e}")

    def _get_op_records(self, operation: str) -> Dict[str, Dict[str, Any]]:
        ops = self.state.setdefault("operations", {})
        return ops.setdefault(operation, {})

    def is_completed(self, operation: str, book_id: int | str) -> bool:
        """Check if a specific book has already completed successfully, was partially indexed, or was skipped."""
        records = self._get_op_records(operation)
        entry = records.get(str(book_id))
        return bool(entry and entry.get("status") in ("completed", "partially_indexed", "skipped"))

    def get_completed_ids(self, operation: str) -> Set[int]:
        """Return the set of integer book IDs that completed successfully, were partially indexed, or were skipped."""
        records = self._get_op_records(operation)
        completed = set()
        for bid_str, rec in records.items():
            if rec.get("status") in ("completed", "partially_indexed", "skipped"):
                try:
                    completed.add(int(bid_str))
                except ValueError:
                    pass
        return completed

    def get_partially_indexed_ids(self, operation: str) -> Set[int]:
        """Return the set of integer book IDs that were marked partially indexed."""
        records = self._get_op_records(operation)
        partially = set()
        for bid_str, rec in records.items():
            if rec.get("status") == "partially_indexed":
                try:
                    partially.add(int(bid_str))
                except ValueError:
                    pass
        return partially

    def get_failed_ids(self, operation: str) -> Set[int]:
        """Return the set of integer book IDs that previously failed."""
        records = self._get_op_records(operation)
        failed = set()
        for bid_str, rec in records.items():
            if rec.get("status") == "failed":
                try:
                    failed.add(int(bid_str))
                except ValueError:
                    pass
        return failed

    def get_skipped_ids(self, operation: str) -> Set[int]:
        """Return the set of integer book IDs that were marked skipped (e.g. graphical formats)."""
        records = self._get_op_records(operation)
        skipped = set()
        for bid_str, rec in records.items():
            if rec.get("status") == "skipped":
                try:
                    skipped.add(int(bid_str))
                except ValueError:
                    pass
        return skipped

    def mark_completed(
        self,
        operation: str,
        book_id: int | str,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a book ID as successfully completed and persist to disk."""
        records = self._get_op_records(operation)
        records[str(book_id)] = {
            "status": "completed",
            "title": title,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.save()

    def mark_partially_indexed(
        self,
        operation: str,
        book_id: int | str,
        title: str = "",
        failed_chunks: int = 0,
        total_chunks: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a book ID as partially indexed (some poison chunks failed, but rest indexed) and persist to disk."""
        records = self._get_op_records(operation)
        meta = metadata or {}
        meta.update({
            "indexing_status": "partially_indexed",
            "failed_chunks": failed_chunks,
            "total_chunks": total_chunks,
        })
        records[str(book_id)] = {
            "status": "partially_indexed",
            "title": title,
            "timestamp": time.time(),
            "metadata": meta,
        }
        self.save()

    def mark_skipped(
        self,
        operation: str,
        book_id: int | str,
        title: str = "",
        reason: str = "graphical_format",
    ) -> None:
        """Mark a book ID as skipped (e.g. graphical formats like CBR/CBZ/DJVU)."""
        records = self._get_op_records(operation)
        records[str(book_id)] = {
            "status": "skipped",
            "title": title,
            "reason": reason,
            "timestamp": time.time(),
        }
        self.save()

    def mark_failed(
        self,
        operation: str,
        book_id: int | str,
        error: str,
        title: str = "",
    ) -> None:
        """Mark a book ID as failed with error details and persist to disk."""
        records = self._get_op_records(operation)
        records[str(book_id)] = {
            "status": "failed",
            "title": title,
            "error": error,
            "timestamp": time.time(),
        }
        self.save()

    def remove_book(self, operation: str, book_id: int | str) -> bool:
        """Remove a book record from the given operation in state tracking."""
        records = self._get_op_records(operation)
        bid_str = str(book_id)
        if bid_str in records:
            del records[bid_str]
            self.save()
            return True
        return False

    def clear(
        self,
        operation: Optional[str] = None,
        preserve_skipped: bool = False,
        keep_skipped: Optional[bool] = None,
    ) -> None:
        """Reset state for a specific operation or all operations.

        If preserve_skipped (or keep_skipped) is True, entries marked with status='skipped' are preserved.
        """
        should_preserve = preserve_skipped or (keep_skipped is True)
        if operation:
            ops = self.state.setdefault("operations", {})
            if should_preserve:
                records = self._get_op_records(operation)
                ops[operation] = {
                    bid: rec for bid, rec in records.items()
                    if rec.get("status") == "skipped"
                }
            else:
                ops[operation] = {}
        else:
            if should_preserve:
                ops = self.state.setdefault("operations", {})
                for op, records in list(ops.items()):
                    ops[op] = {
                        bid: rec for bid, rec in records.items()
                        if rec.get("status") == "skipped"
                    }
            else:
                self.state["operations"] = {}
        self.save()

    def summary(self, operation: str) -> Dict[str, Any]:
        """Return counts of completed, partially_indexed, failed, skipped items, and assigned/skipped chunks for an operation."""
        records = self._get_op_records(operation)
        completed = sum(1 for r in records.values() if r.get("status") == "completed")
        partially_indexed = sum(1 for r in records.values() if r.get("status") == "partially_indexed")
        failed = sum(1 for r in records.values() if r.get("status") == "failed")
        skipped = sum(1 for r in records.values() if r.get("status") == "skipped")
        assigned_chunks = sum(int((r.get("metadata") or {}).get("assigned_chunks", 0)) for r in records.values())
        skipped_chunks = sum(int((r.get("metadata") or {}).get("skipped_chunks", 0)) for r in records.values())
        return {
            "operation": operation,
            "total_recorded": len(records),
            "completed": completed,
            "partially_indexed": partially_indexed,
            "failed": failed,
            "skipped": skipped,
            "assigned_chunks": assigned_chunks,
            "skipped_chunks": skipped_chunks,
        }


class BookProcessingState:
    """
    Tracks and persists granular chunk-level processing state for a single book.
    Describes:
      - processed_chunks: successfully extracted chunks with cached concepts
      - stacked_chunks: currently in-flight / queued chunks
      - retried_chunks: chunks that experienced transient errors and their retry attempt count
      - failed_chunks: poison-pill chunks that reached max_chunk_attempts (default 6)
    Supports resuming interrupted book processing without repeating already processed chunks.
    Retains per-book state file on disk with status='fully_indexed' or 'partially_indexed'
    for downstream verification and auditing.
    """

    def __init__(
        self,
        book_id: int | str,
        book_title: str = "",
        state_dir: Optional[Path | str] = None,
        max_chunk_attempts: int = 6,
    ):
        from bookeeper.calibre.parser import BookParser

        self.book_id = str(book_id)
        self.book_title = BookParser.repair_mojibake(book_title)
        self.max_chunk_attempts = max_chunk_attempts
        self.state_dir = Path(state_dir).expanduser().resolve() if state_dir else Path("./output/.book_states").resolve()
        self.file_path = self.state_dir / f"book_{self.book_id}_state.json"
        self._lock = threading.RLock()
        self.data: Dict[str, Any] = {
            "version": 1,
            "book_id": self.book_id,
            "book_title": self.book_title,
            "total_chunks": 0,
            "max_chunk_attempts": self.max_chunk_attempts,
            "status": "in_progress",
            "processed_chunks": {},
            "retried_chunks": {},
            "stacked_chunks": {},
            "failed_chunks": {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self.load()

    def load(self) -> None:
        """Load per-book state from disk if it exists."""
        from bookeeper.calibre.parser import BookParser

        with self._lock:
            if not self.file_path.exists():
                return
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.data.update(loaded)
                        # On restart, any previously stacked/in-flight chunks are no longer running;
                        # clear stacked_chunks so they can be re-queued.
                        self.data["stacked_chunks"] = {}
                        if self.book_title and not self.data.get("book_title"):
                            self.data["book_title"] = self.book_title
                        if self.data.get("book_title"):
                            self.data["book_title"] = BookParser.repair_mojibake(self.data["book_title"])
            except Exception as e:
                logger.warning(f"Could not load book state from {self.file_path} ({e}); starting fresh.")

    def save(self) -> None:
        """Atomically persist state to disk using a temporary file rename."""
        with self._lock:
            try:
                self.state_dir.mkdir(parents=True, exist_ok=True)
                self.data["updated_at"] = time.time()
                with tempfile.NamedTemporaryFile("w", dir=self.state_dir, delete=False, encoding="utf-8") as tf:
                    json.dump(self.data, tf, indent=2, ensure_ascii=False)
                    temp_name = tf.name
                os.replace(temp_name, self.file_path)
            except Exception as e:
                logger.warning(f"Failed to persist book state to {self.file_path}: {e}")

    def delete(self) -> bool:
        """Remove state file from disk when book completes with 0 errors."""
        with self._lock:
            try:
                if self.file_path.exists():
                    self.file_path.unlink()
                    return True
            except Exception as e:
                logger.warning(f"Could not delete book state file {self.file_path}: {e}")
            return False

    def set_total_chunks(self, total: int) -> None:
        with self._lock:
            self.data["total_chunks"] = total
            self.save()

    @property
    def processed_chunks(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self.data.get("processed_chunks", {}))

    @property
    def retried_chunks(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self.data.get("retried_chunks", {}))

    @property
    def stacked_chunks(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self.data.get("stacked_chunks", {}))

    @property
    def failed_chunks(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self.data.get("failed_chunks", {}))

    def is_chunk_processed(self, chunk_id: str) -> bool:
        with self._lock:
            return chunk_id in self.data.get("processed_chunks", {})

    def unmark_chunk_processed(self, chunk_id: str) -> bool:
        """Remove chunk from processed_chunks (e.g. to recover a skipped limbo chunk) and persist."""
        with self._lock:
            if "processed_chunks" in self.data and chunk_id in self.data["processed_chunks"]:
                del self.data["processed_chunks"][chunk_id]
                self.save()
                return True
            return False

    def recover_skipped_limbo_chunks(self, chunks: Sequence[Any], threshold: int) -> List[str]:
        """
        Recover chunks previously marked as skipped_limbo whose size is >= threshold,
        unmarking them from processed_chunks so they are returned to unprocessed status.
        Returns list of recovered chunk IDs.
        """
        recovered = []
        with self._lock:
            processed = self.data.get("processed_chunks", {})
            for chk in chunks:
                cid = getattr(chk, "chunk_id", None) or (chk.get("chunk_id") if isinstance(chk, dict) else None)
                if not cid or cid not in processed:
                    continue
                rec = processed[cid]
                if rec.get("server_used") == "skipped_limbo" or rec.get("status") == "skipped_limbo":
                    c_size = getattr(chk, "char_count", 0) or len(getattr(chk, "text", "") or "")
                    if isinstance(chk, dict) and not c_size:
                        c_size = chk.get("char_count") or len(chk.get("text", "") or "")
                    if c_size >= threshold:
                        del processed[cid]
                        recovered.append(cid)
            if recovered:
                self.save()
        return recovered

    def is_chunk_failed(self, chunk_id: str) -> bool:
        with self._lock:
            return chunk_id in self.data.get("failed_chunks", {})

    def get_processed_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.data.get("processed_chunks", {}).get(chunk_id)

    def get_failed_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.data.get("failed_chunks", {}).get(chunk_id)

    def get_initial_attempt(self, chunk_id: str) -> int:
        """Return previous retry count for a chunk if it was retried in a prior run."""
        with self._lock:
            return int(self.data.get("retried_chunks", {}).get(chunk_id, {}).get("attempts", 0))

    def mark_chunk_stacked(
        self,
        chunk_id: str,
        chunk_idx: int,
        section_title: str,
        attempt: int = 0,
        server: str = "",
    ) -> None:
        """Mark a chunk as actively being processed (stacked)."""
        with self._lock:
            stacked = self.data.setdefault("stacked_chunks", {})
            stacked[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk_idx": chunk_idx,
                "section_title": section_title,
                "attempt": attempt,
                "server": server,
                "started_at": time.time(),
            }

    def unstack_chunk(self, chunk_id: str) -> None:
        """Remove chunk from stacked_chunks."""
        with self._lock:
            stacked = self.data.setdefault("stacked_chunks", {})
            stacked.pop(chunk_id, None)

    @property
    def assigned_chunks_count(self) -> int:
        with self._lock:
            processed = self.data.get("processed_chunks", {})
            return sum(
                1
                for rec in processed.values()
                if rec.get("server_used") == "semantic_fastpath" or rec.get("status") == "assigned"
            )

    @property
    def skipped_chunks_count(self) -> int:
        with self._lock:
            processed = self.data.get("processed_chunks", {})
            return sum(
                1
                for rec in processed.values()
                if rec.get("server_used") == "skipped_limbo" or rec.get("status") == "skipped_limbo"
            )

    def mark_chunk_processed(
        self,
        chunk_id: str,
        chunk_idx: int,
        section_title: str,
        extraction_dict: Dict[str, Any],
        duration: float,
        server_used: str,
    ) -> None:
        """Mark a chunk as successfully extracted and persist."""
        with self._lock:
            self.unstack_chunk(chunk_id)
            self.data.setdefault("retried_chunks", {}).pop(chunk_id, None)

            if server_used == "semantic_fastpath":
                chunk_status = "assigned"
            elif server_used == "skipped_limbo":
                chunk_status = "skipped_limbo"
            else:
                chunk_status = "extracted"

            processed = self.data.setdefault("processed_chunks", {})
            processed[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk_idx": chunk_idx,
                "section_title": section_title,
                "status": chunk_status,
                "duration": duration,
                "server_used": server_used,
                "extraction": extraction_dict,
                "completed_at": time.time(),
            }
            self.data["assigned_chunks_count"] = self.assigned_chunks_count
            self.data["skipped_chunks_count"] = self.skipped_chunks_count
            self.save()

    def mark_chunk_retried(
        self,
        chunk_id: str,
        chunk_idx: int,
        section_title: str,
        attempt: int,
        error: str,
        server_used: str,
    ) -> None:
        """Record a chunk retry attempt and persist."""
        with self._lock:
            self.unstack_chunk(chunk_id)
            retried = self.data.setdefault("retried_chunks", {})
            retried[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk_idx": chunk_idx,
                "section_title": section_title,
                "attempts": attempt,
                "last_error": error,
                "last_server": server_used,
                "updated_at": time.time(),
            }
            self.save()

    def mark_chunk_failed(
        self,
        chunk_id: str,
        chunk_idx: int,
        section_title: str,
        attempts: int,
        error: str,
        server_used: str,
    ) -> None:
        """Mark a chunk as permanently failed (poison pill after max attempts) and persist."""
        with self._lock:
            self.unstack_chunk(chunk_id)
            self.data.setdefault("retried_chunks", {}).pop(chunk_id, None)

            failed = self.data.setdefault("failed_chunks", {})
            failed[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk_idx": chunk_idx,
                "section_title": section_title,
                "attempts": attempts,
                "error": error,
                "server_used": server_used,
                "failed_at": time.time(),
            }
            self.save()

    def finalize(self, has_errors: bool = False) -> str:
        """
        Finalize the book processing state.
        If any chunk failed (or has_errors is True):
          - Marks status as 'partially_indexed'
          - Persists state file to disk for debugging/resuming
          - Returns 'partially_indexed'
        If all chunks succeeded with no errors:
          - Marks status as 'fully_indexed'
          - Persists state file to disk for downstream verification
          - Returns 'fully_indexed'
        """
        with self._lock:
            has_failed = bool(self.data.get("failed_chunks")) or has_errors
            if has_failed:
                self.data["status"] = "partially_indexed"
            else:
                self.data["status"] = "fully_indexed"
            self.save()
            return self.data["status"]

    def set_chunks(self, chunks: Sequence[Any]) -> None:
        """Store chunk records in state file so it serves as a standalone local cache."""
        with self._lock:
            serialized = []
            for c in chunks:
                if hasattr(c, "model_dump"):
                    serialized.append(c.model_dump())
                elif isinstance(c, dict):
                    serialized.append(c)
            self.data["chunks"] = serialized
            self.data["total_chunks"] = len(chunks)
            self.save()

    def get_chunks(self) -> Optional[List[Any]]:
        """Retrieve stored chunks if present in state file."""
        from bookeeper.processing.chunker import HierarchicalChunk

        with self._lock:
            raw = self.data.get("chunks")
            if isinstance(raw, list) and raw:
                try:
                    return [HierarchicalChunk.model_validate(c) for c in raw]
                except Exception:
                    return None
            return None

    @classmethod
    def clean_all(cls, state_dir: Path | str) -> int:
        """
        Purge all per-book checkpoint files (book_*_state.json) in state_dir.
        Returns count of deleted files.
        """
        p = Path(state_dir).expanduser().resolve()
        if not p.is_dir():
            return 0
        count = 0
        for f in p.glob("book_*_state.json"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        return count


class BookStateCache:
    """
    Manages an external read-only directory of historical or backup book state files
    (book_*_state.json) and optional chunk files to serve as local cache.
    Never modified or purged during cleanups or start-from-scratch.
    """

    def __init__(self, cache_dir: Optional[Path | str] = None):
        self.cache_dir: Optional[Path] = (
            Path(cache_dir).expanduser().resolve() if cache_dir else None
        )
        self._id_to_file: Dict[str, Path] = {}
        self._title_to_file: Dict[str, Path] = {}
        self._loaded_cache: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.reused_books_count: int = 0
        self.reused_chunks_count: int = 0
        self.reused_extractions_count: int = 0

        self._index_cache()

    def is_enabled(self) -> bool:
        """Return True if a valid cache directory was configured and exists."""
        return self.cache_dir is not None and self.cache_dir.is_dir()

    def _index_cache(self) -> None:
        """Scan and index all book state files in cache_dir by book_id and book_title."""
        if not self.is_enabled():
            return
        assert self.cache_dir is not None

        candidate_dirs = [self.cache_dir]
        sub_states = self.cache_dir / ".book_states"
        if sub_states.is_dir():
            candidate_dirs.append(sub_states)

        for cdir in candidate_dirs:
            for f in cdir.glob("book_*_state.json"):
                stem = f.stem  # e.g. "book_1_state"
                parts = stem.split("_")
                if len(parts) >= 3 and parts[0] == "book" and parts[-1] == "state":
                    bid = "_".join(parts[1:-1])
                    if bid and bid not in self._id_to_file:
                        self._id_to_file[bid] = f

                try:
                    with open(f, "r", encoding="utf-8") as sf:
                        meta = json.load(sf)
                    f_bid = str(meta.get("book_id") or "")
                    if f_bid and f_bid not in self._id_to_file:
                        self._id_to_file[f_bid] = f
                    title = meta.get("book_title")
                    if title:
                        from bookeeper.calibre.parser import BookParser

                        norm_title = BookParser.repair_mojibake(str(title)).strip().lower()
                        if norm_title and norm_title not in self._title_to_file:
                            self._title_to_file[norm_title] = f
                except Exception as e:
                    logger.debug(f"Failed to pre-index cache file {f}: {e}")

    def get_cached_state(self, book_id: int | str, book_title: str = "") -> Optional[Dict[str, Any]]:
        """Retrieve cached state dictionary for a book by ID or title."""
        if not self.is_enabled():
            return None

        bid_str = str(book_id)
        if bid_str in self._loaded_cache:
            return self._loaded_cache[bid_str]

        f_path = self._id_to_file.get(bid_str)
        if not f_path and book_title:
            from bookeeper.calibre.parser import BookParser

            norm_title = BookParser.repair_mojibake(str(book_title)).strip().lower()
            f_path = self._title_to_file.get(norm_title)

        if not f_path and self.cache_dir:
            direct = self.cache_dir / f"book_{bid_str}_state.json"
            if direct.is_file():
                f_path = direct
            else:
                sub_direct = self.cache_dir / ".book_states" / f"book_{bid_str}_state.json"
                if sub_direct.is_file():
                    f_path = sub_direct

        if f_path and f_path.is_file():
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._loaded_cache[bid_str] = data
                    return data
            except Exception as e:
                logger.warning(f"Failed to read cache state file from {f_path}: {e}")

        return None

    def get_cached_chunks(self, book_id: int | str, book_title: str = "") -> Optional[List[Any]]:
        """
        Check if chunks for this book exist in the cached state file or in companion chunk files.
        Returns list of HierarchicalChunk instances if found, or None if absent.
        """
        if not self.is_enabled():
            return None

        from bookeeper.processing.chunker import HierarchicalChunk

        # 1. Check if cached state file itself contains embedded chunks
        st = self.get_cached_state(book_id, book_title)
        if st and st.get("chunks"):
            try:
                raw_chunks = st["chunks"]
                if isinstance(raw_chunks, list) and len(raw_chunks) > 0:
                    chunks = [HierarchicalChunk.model_validate(c) for c in raw_chunks]
                    self.reused_chunks_count += len(chunks)
                    return chunks
            except Exception as e:
                logger.debug(f"Failed to parse embedded chunks from cache state for book #{book_id}: {e}")

        # 2. Check if companion book_{id}_chunks.json exists in cache_dir or its chunks/ subfolder
        bid_str = str(book_id)
        assert self.cache_dir is not None
        candidate_paths = [
            self.cache_dir / f"book_{bid_str}_chunks.json",
            self.cache_dir / "chunks" / f"book_{bid_str}_chunks.json",
            self.cache_dir.parent / "chunks" / f"book_{bid_str}_chunks.json",
        ]
        sub_states = self.cache_dir / ".book_states"
        if sub_states.is_dir():
            candidate_paths.append(self.cache_dir / "chunks" / f"book_{bid_str}_chunks.json")

        for cp in candidate_paths:
            if cp.is_file() and cp.stat().st_size > 0:
                try:
                    with open(cp, "r", encoding="utf-8") as cf:
                        c_data = json.load(cf)
                    raw_chunks = c_data.get("chunks", [])
                    if isinstance(raw_chunks, list) and len(raw_chunks) > 0:
                        chunks = [HierarchicalChunk.model_validate(c) for c in raw_chunks]
                        self.reused_chunks_count += len(chunks)
                        return chunks
                except Exception as e:
                    logger.debug(f"Failed to read chunks from {cp}: {e}")

        return None

    def get_cached_raw_analysis(
        self, book_id: int | str, chunk_id: str, book_title: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Check if raw extraction / analysis for chunk_id is present in the cached state file.
        Returns the chunk extraction record dict if present, or None.
        """
        st = self.get_cached_state(book_id, book_title)
        if not st:
            return None
        processed = st.get("processed_chunks", {})
        if chunk_id in processed:
            rec = processed[chunk_id]
            if rec.get("status") in ("extracted", "assigned", "skipped_limbo") and "extraction" in rec:
                return rec
        return None

    def populate_state_from_cache(self, active_state: BookProcessingState) -> int:
        """
        Seed active_state with cached processed chunks from cache state file.
        Returns count of chunk records reused.
        """
        st = self.get_cached_state(active_state.book_id, active_state.book_title)
        if not st:
            return 0
        cached_processed = st.get("processed_chunks", {})
        if not cached_processed:
            return 0

        seeded = 0
        with active_state._lock:
            act_processed = active_state.data.setdefault("processed_chunks", {})
            for cid, rec in cached_processed.items():
                if (
                    cid not in act_processed
                    and rec.get("status") in ("extracted", "assigned", "skipped_limbo")
                    and "extraction" in rec
                ):
                    act_processed[cid] = rec
                    seeded += 1
            if seeded > 0:
                self.reused_books_count += 1
                self.reused_extractions_count += seeded
                active_state.save()
        return seeded

    def stats(self) -> Dict[str, Any]:
        """Return cache reuse statistics."""
        return {
            "enabled": self.is_enabled(),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "indexed_state_files": len(self._id_to_file),
            "reused_books_count": self.reused_books_count,
            "reused_chunks_count": self.reused_chunks_count,
            "reused_extractions_count": self.reused_extractions_count,
        }

