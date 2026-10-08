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
    """Atomic thematic chunk containing complete parent book, chapter, and section/subtitle metadata."""

    chunk_id: str
    book_id: int
    book_title: str
    section_title: str
    chapter_idx: int
    chunk_idx: int
    text: str
    chapter_title: str = ""
    subtitle: str = ""
    char_count: int = 0
    word_count: int = 0
    hierarchy_level: str = "child"  # "parent" (macro) or "child" (micro)
    parent_chunk_id: Optional[str] = None
    parent_text: Optional[str] = None
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    def model_post_init(self, __context) -> None:
        if not self.char_count and self.text:
            self.char_count = len(self.text)
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())
        if not self.chapter_title and self.section_title:
            self.chapter_title = self.section_title
        if not self.subtitle and self.section_title:
            self.subtitle = self.section_title

    @property
    def breadcrumb(self) -> str:
        parts = [self.book_title]
        ch = self.chapter_title or self.section_title
        if ch:
            parts.append(ch)
        if self.subtitle and self.subtitle != ch and self.subtitle not in parts:
            parts.append(self.subtitle)
        return " > ".join(parts) + f" [part {self.chapter_idx}.{self.chunk_idx}]"

    @property
    def context_header(self) -> str:
        return f"[Context: {self.breadcrumb}]"


class HierarchicalChunker:
    """
    Combines Smart Structural Parsing and Hierarchical Parent-Child RAG:
    - Parses document structure, respecting headers (#, ##, ###), code fences, tables, and lists.
    - Emits micro (child) chunks linked directly to their macro (parent) context passages.
    - Enriches chunks with contextual breadcrumb headers and bidirectional sequential pointers.
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 85.0,
        max_chunk_chars: int = 2500,
        min_chunk_chars: int = 120,
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

    def _extract_subsections(self, text: str, default_title: str) -> List[tuple[str, str]]:
        """
        Extract subtitles and sub-blocks by detecting markdown headings (#, ##, ###).
        Returns list of (subtitle, subsection_text).
        """
        heading_pattern = re.compile(r"(?m)^(#{1,6})\s+(.+)$")
        matches = list(heading_pattern.finditer(text))
        if not matches:
            return [(default_title, text)]

        subsections: List[tuple[str, str]] = []
        # Any text before the first heading
        first_start = matches[0].start()
        if first_start > 0:
            preamble = text[:first_start].strip()
            if preamble:
                subsections.append((default_title, preamble))

        for idx, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                subsections.append((title, body))

        return subsections if subsections else [(default_title, text)]

    def chunk_section(
        self,
        section: Section,
        book_id: int,
        book_title: str,
    ) -> List[HierarchicalChunk]:
        """Split a section into hierarchical smart chunks preserving section and subtitle metadata."""
        text = section.text.strip()
        if not text or len(text) < self.min_chunk_chars:
            return []

        subsections = self._extract_subsections(text, default_title=section.title)
        all_chunks: List[HierarchicalChunk] = []
        global_chunk_idx = 1

        for sub_idx, (subtitle, sub_text) in enumerate(subsections):
            if not sub_text.strip():
                continue

            # Create macro parent chunk ID and text for this subsection
            sub_h = hashlib.sha256(sub_text.encode("utf-8")).hexdigest()[:8]
            parent_chunk_id = f"b{book_id}_c{section.chapter_idx}_sub{sub_idx}_{sub_h}"
            parent_text = sub_text[:3000]

            raw_chunks: List[str] = []

            # 1. Attempt semantic chunking via embeddings if text is sufficiently large
            if self._semantic_splitter is not None and len(sub_text) > self.max_chunk_chars:
                try:
                    docs = self._semantic_splitter.create_documents([sub_text])
                    raw_chunks = [d.page_content.strip() for d in docs if d.page_content.strip()]
                except Exception as e:
                    logger.debug(
                        f"Semantic chunking fallback on '{section.title}' > '{subtitle}': {e}"
                    )
                    raw_chunks = []

            # 2. Smart structural fallback: paragraphs, code blocks, tables, sentences
            if not raw_chunks:
                raw_chunks = self._smart_split(sub_text, self.max_chunk_chars)

            for chunk_text in raw_chunks:
                if len(chunk_text) < self.min_chunk_chars and raw_chunks and len(raw_chunks) > 1:
                    continue

                h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:8]
                cid = f"b{book_id}_c{section.chapter_idx}_p{global_chunk_idx}_{h}"

                chunk = HierarchicalChunk(
                    chunk_id=cid,
                    book_id=book_id,
                    book_title=book_title,
                    chapter_idx=section.chapter_idx,
                    chapter_title=section.title,
                    section_title=section.title,
                    subtitle=subtitle,
                    chunk_idx=global_chunk_idx,
                    text=chunk_text,
                    hierarchy_level="child",
                    parent_chunk_id=parent_chunk_id,
                    parent_text=parent_text,
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        # Link sequential prev_chunk_id and next_chunk_id
        for i in range(len(all_chunks)):
            if i > 0:
                all_chunks[i].prev_chunk_id = all_chunks[i - 1].chunk_id
            if i + 1 < len(all_chunks):
                all_chunks[i].next_chunk_id = all_chunks[i + 1].chunk_id

        return all_chunks

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

        # Re-link across section boundaries if needed
        for i in range(len(all_chunks)):
            if i > 0:
                all_chunks[i].prev_chunk_id = all_chunks[i - 1].chunk_id
            if i + 1 < len(all_chunks):
                all_chunks[i].next_chunk_id = all_chunks[i + 1].chunk_id

        return all_chunks

    def _smart_split(self, text: str, max_chars: int) -> List[str]:
        """
        Smart text splitter that protects code blocks, tables, and lists,
        splitting on paragraph boundaries with sentence-level fallback.
        """
        if len(text) <= max_chars:
            return [text]

        # Break text into blocks, preserving fenced code blocks intact
        blocks: List[str] = []
        in_code_block = False
        current_block: List[str] = []

        for line in text.splitlines(keepends=True):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_block.append(line)
                if not in_code_block:
                    blocks.append("".join(current_block).strip())
                    current_block = []
                continue

            if in_code_block:
                current_block.append(line)
            else:
                if line.strip() == "":
                    if current_block:
                        blocks.append("".join(current_block).strip())
                        current_block = []
                else:
                    current_block.append(line)

        if current_block:
            blocks.append("".join(current_block).strip())

        blocks = [b for b in blocks if b]
        if not blocks:
            return [text]

        chunks: List[str] = []
        current_chunk: List[str] = []
        curr_len = 0

        for b in blocks:
            if curr_len + len(b) + 2 > max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                curr_len = 0

            # If a single block exceeds max_chars and isn't a code fence, split by sentences
            if len(b) > max_chars and not b.startswith("```"):
                sentences = re.split(r"(?<=[.!?])\s+", b)
                for s in sentences:
                    if curr_len + len(s) + 1 > max_chars and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        curr_len = 0
                    current_chunk.append(s)
                    curr_len += len(s) + 1
            else:
                current_chunk.append(b)
                curr_len += len(b) + 2

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

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

