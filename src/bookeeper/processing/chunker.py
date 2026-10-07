"""
Semantic and TOC-based text chunking engine.
"""

import hashlib
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from bookeeper.calibre.parser import ChapterSection
from bookeeper.config import ProcessingSettings


class TextChunk(BaseModel):
    """Normalized text chunk with full chapter and book breadcrumb metadata."""

    chunk_id: str
    book_id: int
    book_title: str
    chapter_title: str
    chapter_sequence: int
    chunk_index: int
    text: str
    character_count: int
    word_count: int

    @property
    def source_ref(self) -> str:
        return f"{self.book_title} > {self.chapter_title} (part {self.chunk_index + 1})"


class SemanticChunker:
    """Chunker that preserves TOC structure and paragraph boundaries."""

    def __init__(self, settings: Optional[ProcessingSettings] = None):
        self.settings = settings or ProcessingSettings()

    def chunk_sections(
        self,
        sections: List[ChapterSection],
        book_id: int,
        book_title: str,
    ) -> List[TextChunk]:
        """Split a list of chapter sections into structured text chunks."""
        all_chunks: List[TextChunk] = []

        for sec in sections:
            sec_text = sec.text.strip()
            if not sec_text or len(sec_text) < self.settings.min_chunk_size:
                continue

            raw_chunks = self._split_text(
                sec_text,
                target_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )

            for idx, chunk_str in enumerate(raw_chunks):
                if len(chunk_str) < self.settings.min_chunk_size and raw_chunks:
                    # Skip tiny residual fragments unless it's the only one
                    if len(raw_chunks) > 1:
                        continue

                # Unique deterministic chunk id based on book, section, and text hash
                h = hashlib.sha256(chunk_str.encode("utf-8")).hexdigest()[:8]
                cid = f"b{book_id}_s{sec.sequence}_c{idx}_{h}"

                all_chunks.append(
                    TextChunk(
                        chunk_id=cid,
                        book_id=book_id,
                        book_title=book_title,
                        chapter_title=sec.title,
                        chapter_sequence=sec.sequence,
                        chunk_index=idx,
                        text=chunk_str,
                        character_count=len(chunk_str),
                        word_count=len(chunk_str.split()),
                    )
                )

        return all_chunks

    def _split_text(self, text: str, target_size: int, overlap: int) -> List[str]:
        """Split text along paragraph or sentence boundaries with overlapping windows."""
        if len(text) <= target_size:
            return [text]

        # Break text into paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)

            # If a single paragraph is enormous, break it by sentences
            if para_len > target_size:
                if current_chunk:
                    joined = "\n\n".join(current_chunk)
                    chunks.append(joined)
                    current_chunk = []
                    current_length = 0

                sentence_chunks = self._split_by_sentences(para, target_size, overlap)
                chunks.extend(sentence_chunks)
                continue

            if current_length + para_len + 2 > target_size and current_chunk:
                joined = "\n\n".join(current_chunk)
                chunks.append(joined)

                # Keep overlap from the end of the previous chunk
                overlap_chunk: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= overlap:
                        overlap_chunk.insert(0, item)
                        overlap_len += len(item) + 2
                    else:
                        break

                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(para)
            current_length += para_len + 2

        if current_chunk:
            joined = "\n\n".join(current_chunk)
            chunks.append(joined)

        return chunks

    @staticmethod
    def _split_by_sentences(text: str, target_size: int, overlap: int) -> List[str]:
        """Fallback split for huge paragraphs using sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current: List[str] = []
        length = 0

        for s in sentences:
            s_len = len(s)
            if length + s_len + 1 > target_size and current:
                chunks.append(" ".join(current))
                # Build overlap
                overlap_items: List[str] = []
                overlap_len = 0
                for item in reversed(current):
                    if overlap_len + len(item) <= overlap:
                        overlap_items.insert(0, item)
                        overlap_len += len(item) + 1
                    else:
                        break
                current = overlap_items
                length = overlap_len

            current.append(s)
            length += s_len + 1

        if current:
            chunks.append(" ".join(current))

        return chunks
