"""
Hierarchical and semantic text chunking preserving chapter and section metadata.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from bookeeper.calibre.parser import Section

logger = logging.getLogger(__name__)


class HierarchicalChunk(BaseModel):
    """Atomic thematic chunk containing complete parent book and section metadata."""

    chunk_id: str
    book_id: int
    book_title: str
    section_title: str
    chapter_idx: int
    chunk_idx: int
    text: str
    char_count: int = 0
    word_count: int = 0

    def model_post_init(self, __context) -> None:
        if not self.char_count and self.text:
            self.char_count = len(self.text)
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())

    @property
    def breadcrumb(self) -> str:
        return f"{self.book_title} > {self.section_title} [part {self.chapter_idx}.{self.chunk_idx}]"


class HierarchicalChunker:
    """
    Splits document sections into atomic thematic chunks using SemanticChunker
    (via OllamaEmbeddings) with graceful heuristic fallback when embeddings are unavailable.
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 85.0,
        max_chunk_chars: int = 2500,
        min_chunk_chars: int = 150,
    ):
        self.embeddings = embeddings
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

        self._semantic_splitter = None
        if self.embeddings is not None:
            try:
                from langchain_experimental.text_splitter import SemanticChunker

                self._semantic_splitter = SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type=self.breakpoint_threshold_type,
                    breakpoint_threshold_amount=self.breakpoint_threshold_amount,
                )
            except Exception as e:
                logger.warning(f"Could not initialize SemanticChunker with embeddings: {e}")

    def chunk_section(
        self,
        section: Section,
        book_id: int,
        book_title: str,
    ) -> List[HierarchicalChunk]:
        """Split a single section into thematic chunks preserving section metadata."""
        text = section.text.strip()
        if not text or len(text) < self.min_chunk_chars:
            return []

        raw_chunks: List[str] = []

        # 1. Attempt semantic chunking via embeddings
        if self._semantic_splitter is not None:
            try:
                docs = self._semantic_splitter.create_documents([text])
                raw_chunks = [d.page_content.strip() for d in docs if d.page_content.strip()]
            except Exception as e:
                logger.warning(
                    f"Semantic chunking failed on '{section.title}', falling back to paragraph chunker: {e}"
                )
                raw_chunks = []

        # 2. Fallback: Paragraph and sentence boundary chunking
        if not raw_chunks:
            raw_chunks = self._fallback_split(text, self.max_chunk_chars)

        chunks: List[HierarchicalChunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            if len(chunk_text) < self.min_chunk_chars and raw_chunks and len(raw_chunks) > 1:
                continue

            h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:8]
            cid = f"b{book_id}_c{section.chapter_idx}_p{idx}_{h}"

            chunks.append(
                HierarchicalChunk(
                    chunk_id=cid,
                    book_id=book_id,
                    book_title=book_title,
                    section_title=section.title,
                    chapter_idx=section.chapter_idx,
                    chunk_idx=idx + 1,
                    text=chunk_text,
                )
            )

        return chunks

    def chunk_book(
        self,
        sections: List[Section],
        book_id: int,
        book_title: str,
    ) -> List[HierarchicalChunk]:
        """Process all sections of a book sequentially."""
        all_chunks: List[HierarchicalChunk] = []
        for sec in sections:
            all_chunks.extend(self.chunk_section(sec, book_id, book_title))
        return all_chunks

    def _fallback_split(self, text: str, max_chars: int) -> List[str]:
        """Split text along natural paragraph boundaries."""
        if len(text) <= max_chars:
            return [text]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[str] = []
        current: List[str] = []
        curr_len = 0

        for p in paragraphs:
            if curr_len + len(p) + 2 > max_chars and current:
                chunks.append("\n\n".join(current))
                current = []
                curr_len = 0

            # If a single paragraph is longer than max_chars, split by sentence
            if len(p) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", p)
                for s in sentences:
                    if curr_len + len(s) + 1 > max_chars and current:
                        chunks.append(" ".join(current))
                        current = []
                        curr_len = 0
                    current.append(s)
                    curr_len += len(s) + 1
            else:
                current.append(p)
                curr_len += len(p) + 2

        if current:
            chunks.append("\n\n".join(current))

        return chunks


class ChunkStore:
    """
    Manages persistent local storage and retrieval of atomic book chunks.
    Allows re-using pre-computed chunks without re-parsing books or repeating embeddings.
    """

    def __init__(self, storage_dir: Path | str):
        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _chunk_file(self, book_id: int) -> Path:
        return self.storage_dir / f"book_{book_id}_chunks.json"

    def has_chunks(self, book_id: int) -> bool:
        """Check if valid cached chunks exist for a book."""
        p = self._chunk_file(book_id)
        return p.is_file() and p.stat().st_size > 0

    def load_chunks(self, book_id: int) -> Optional[List[HierarchicalChunk]]:
        """Load cached chunks for a book from disk."""
        p = self._chunk_file(book_id)
        if not p.is_file() or p.stat().st_size == 0:
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_chunks = data.get("chunks", [])
            return [HierarchicalChunk.model_validate(c) for c in raw_chunks]
        except Exception as e:
            logger.warning(f"Failed to load cached chunks from {p}: {e}")
            return None

    def save_chunks(self, book_id: int, book_title: str, chunks: List[HierarchicalChunk]) -> Path:
        """Atomically persist chunks for a book to disk."""
        p = self._chunk_file(book_id)
        data = {
            "book_id": book_id,
            "book_title": book_title,
            "total_chunks": len(chunks),
            "created_at": time.time(),
            "chunks": [c.model_dump() for c in chunks],
        }
        with tempfile.NamedTemporaryFile("w", dir=self.storage_dir, delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2, ensure_ascii=False)
            tmp_name = tf.name
        os.replace(tmp_name, p)
        return p

    def delete_chunks(self, book_id: int) -> bool:
        """Remove cached chunks for a book."""
        p = self._chunk_file(book_id)
        if p.is_file():
            try:
                p.unlink()
                return True
            except Exception:
                pass
        return False

    def list_stored_book_ids(self) -> List[int]:
        """Return sorted list of book IDs currently stored."""
        ids = []
        for f in self.storage_dir.glob("book_*_chunks.json"):
            try:
                bid_str = f.stem.split("_")[1]
                ids.append(int(bid_str))
            except Exception:
                pass
        return sorted(ids)

    def stats(self) -> dict:
        """Return summary statistics of the chunk store."""
        book_ids = self.list_stored_book_ids()
        total_size = sum(self._chunk_file(b).stat().st_size for b in book_ids if self._chunk_file(b).is_file())
        return {
            "total_books": len(book_ids),
            "storage_dir": str(self.storage_dir),
            "total_bytes": total_size,
        }

