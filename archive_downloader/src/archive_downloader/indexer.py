"""
Indexing and deduplication engine for downloaded articles.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from archive_downloader.models import IndexEntry

logger = logging.getLogger(__name__)


class DownloadIndex:
    """
    Persistent JSON index managing downloaded articles and deduplication.
    """

    def __init__(self, storage_dir: Path, index_filename: str = "index.json"):
        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.index_file = self.storage_dir / index_filename
        self.entries: Dict[str, IndexEntry] = {}
        self.load()

    def load(self) -> None:
        """Load index from JSON file on disk."""
        self.entries.clear()
        if self.index_file.is_file():
            try:
                data = json.loads(self.index_file.read_text(encoding="utf-8"))
                for k, v in data.items():
                    try:
                        self.entries[k] = IndexEntry(**v)
                    except Exception as e:
                        logger.warning(f"Error parsing index entry {k}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load index file {self.index_file}: {e}")

    def save(self) -> None:
        """Atomically persist index to disk."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump() for k, v in self.entries.items()}
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=self.storage_dir, delete=False, encoding="utf-8") as tf:
                json.dump(data, tf, indent=2, ensure_ascii=False)
                temp_file = Path(tf.name)
            temp_file.replace(self.index_file)
        except Exception as e:
            logger.error(f"Failed to save index file: {e}")
            if temp_file and temp_file.is_file():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    def is_downloaded(self, arxiv_id: str) -> bool:
        """
        Check if an article has already been downloaded and exists on disk.
        If registered in index but the file was deleted or corrupted, returns False.
        """
        clean_id = arxiv_id.strip()
        if clean_id not in self.entries:
            # Also check if a file matching this arxiv_id exists on disk directly
            matching = list(self.storage_dir.glob(f"{clean_id}*.pdf"))
            if matching and matching[0].is_file() and matching[0].stat().st_size > 0:
                return True
            return False

        entry = self.entries[clean_id]
        full_path = self.storage_dir / entry.file_path
        if full_path.is_file() and full_path.stat().st_size > 0:
            return True

        # File is missing from disk
        return False

    def add(self, entry: IndexEntry) -> None:
        """Add an entry to the index and persist."""
        self.entries[entry.arxiv_id] = entry
        self.save()

    def get(self, arxiv_id: str) -> Optional[IndexEntry]:
        """Get index entry by arXiv ID."""
        return self.entries.get(arxiv_id.strip())

    def count(self) -> int:
        """Total number of indexed entries."""
        return len(self.entries)

    def all_entries(self) -> List[IndexEntry]:
        """List all indexed entries sorted by download date descending."""
        return sorted(self.entries.values(), key=lambda e: e.downloaded_at, reverse=True)
