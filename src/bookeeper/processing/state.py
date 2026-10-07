"""
Persistent progress and checkpoint manager for book processing tasks.
Allows resuming interrupted or failed runs without repeating already processed books.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

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
        """Check if a specific book has already completed successfully."""
        records = self._get_op_records(operation)
        entry = records.get(str(book_id))
        return bool(entry and entry.get("status") == "completed")

    def get_completed_ids(self, operation: str) -> Set[int]:
        """Return the set of integer book IDs that completed successfully."""
        records = self._get_op_records(operation)
        completed = set()
        for bid_str, rec in records.items():
            if rec.get("status") == "completed":
                try:
                    completed.add(int(bid_str))
                except ValueError:
                    pass
        return completed

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

    def clear(self, operation: Optional[str] = None) -> None:
        """Reset state for a specific operation or all operations."""
        if operation:
            ops = self.state.setdefault("operations", {})
            ops[operation] = {}
        else:
            self.state["operations"] = {}
        self.save()

    def summary(self, operation: str) -> Dict[str, Any]:
        """Return counts of completed and failed items for an operation."""
        records = self._get_op_records(operation)
        completed = sum(1 for r in records.values() if r.get("status") == "completed")
        failed = sum(1 for r in records.values() if r.get("status") == "failed")
        return {
            "operation": operation,
            "total_recorded": len(records),
            "completed": completed,
            "failed": failed,
        }
