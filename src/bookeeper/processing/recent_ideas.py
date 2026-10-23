"""
Recent Ideas Processing Module.

Implements chronological batching, publication date sorting, embedding-based chunk
pre-assignment against existing ideas, and prioritized chunk extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from bookeeper.calibre.parser import BookParser, format_idea_month

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.extractor import KnowledgeExtractor

logger = logging.getLogger(__name__)


# ==============================================================================
# Publication Date Sorting & Batch Partitioning
# ==============================================================================


def normalize_sort_date(date_val: Any) -> str:
    """
    Convert publication date into a normalized sortable string (YYYY-MM-DD).
    Missing, invalid, or epoch-zero dates sort last ('9999-12-31').
    """
    if not date_val:
        return "9999-12-31"
    s = BookParser.repair_mojibake(str(date_val)).strip()
    if not s or s.startswith("0101-01-01"):
        return "9999-12-31"

    # Match YYYY-MM-DD
    m_full = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m_full:
        y, m, d = int(m_full.group(1)), int(m_full.group(2)), int(m_full.group(3))
        if 1000 <= y <= 9999 and 1 <= m <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{m:02d}-{d:02d}"

    # Match YYYY-MM
    m_ym = re.match(r"^(\d{4})-(\d{2})", s)
    if m_ym:
        y, m = int(m_ym.group(1)), int(m_ym.group(2))
        if 1000 <= y <= 9999 and 1 <= m <= 12:
            return f"{y:04d}-{m:02d}-01"

    # Match YYYY
    m_y = re.match(r"^(\d{4})", s)
    if m_y:
        y = int(m_y.group(1))
        if 1000 <= y <= 9999:
            return f"{y:04d}-01-01"

    return "9999-12-31"


def sort_books_chronologically(books: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort initial documents/books by publication date in ascending order (earliest first).
    Books without publication date are placed at the end.
    Tie-breaking uses book ID or title for deterministic ordering.
    """
    def _sort_key(b: Dict[str, Any]) -> Tuple[str, int, str]:
        date_cand = b.get("publication_date") or b.get("pubdate")
        if not date_cand and b.get("path"):
            try:
                from pathlib import Path
                p = Path(b["path"])
                if p.is_file():
                    date_cand = BookParser.extract_publication_date(p)
            except Exception:
                pass

        sort_d = normalize_sort_date(date_cand)
        bid = int(b.get("id") or 0)
        title = str(b.get("title") or "")
        return (sort_d, bid, title)

    return sorted(books, key=_sort_key)


def partition_books_into_batches(
    books: List[Dict[str, Any]],
    batch_size: int = 10,
) -> List[List[Dict[str, Any]]]:
    """
    Group sorted books into sequential blocks/batches of batch_size (default: 10 books).
    """
    if batch_size <= 0:
        return [list(books)]
    batches = []
    for i in range(0, len(books), batch_size):
        batches.append(books[i : i + batch_size])
    return batches


# ==============================================================================
# Semantic Chunk Routing & Idea Embedding Index
# ==============================================================================


class ChunkPriority(str, Enum):
    """
    Chunk routing priority tiers:
    - UNASSIGNED: Low similarity (<0.65). Expected to contain brand new ideas. Processed FIRST.
    - HIGH_CONFIDENCE: High similarity (>=0.90). Direct assignment to existing idea. Skips LLM.
    - LIMBO: Intermediate similarity (0.65 <= sim < 0.90). Validated vs candidate idea.
    """
    UNASSIGNED = "unassigned"
    HIGH_CONFIDENCE = "high_confidence"
    LIMBO = "limbo"


@dataclass
class RoutedChunk:
    chunk: Any
    priority: ChunkPriority
    similarity: float = 0.0
    matched_idea: Optional[str] = None
    matched_idea_summary: str = ""


@dataclass
class BatchRoutingResult:
    unassigned: List[RoutedChunk] = field(default_factory=list)
    high_confidence: List[RoutedChunk] = field(default_factory=list)
    limbo: List[RoutedChunk] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return len(self.unassigned) + len(self.high_confidence) + len(self.limbo)

    @property
    def unassigned_count(self) -> int:
        return len(self.unassigned)

    @property
    def high_confidence_count(self) -> int:
        return len(self.high_confidence)

    @property
    def limbo_count(self) -> int:
        return len(self.limbo)


@dataclass
class IdeaEmbeddingIndex:
    """
    Vector index of existing knowledge graph ideas for fast cosine similarity routing.
    """
    idea_names: List[str] = field(default_factory=list)
    idea_summaries: List[str] = field(default_factory=list)
    idea_months: Dict[str, Optional[str]] = field(default_factory=dict)
    vectors: Optional[np.ndarray] = None

    @classmethod
    def build_from_graph(
        cls,
        store: ConceptGraphStore,
        embeddings: Optional[Embeddings] = None,
        batch_size: int = 64,
    ) -> "IdeaEmbeddingIndex":
        """
        Extract all Concept/Idea nodes from knowledge graph and compute their embeddings.
        """
        names: List[str] = []
        texts: List[str] = []
        months: Dict[str, Optional[str]] = {}

        if store and store.graph:
            for node_id, attrs in store.graph.nodes(data=True):
                if attrs.get("type") == "Concept" or attrs.get("is_idea"):
                    raw_name = attrs.get("name") or node_id.replace("concept:", "")
                    clean_name = BookParser.repair_mojibake(raw_name).strip()
                    if not clean_name:
                        continue
                    brief = attrs.get("brief_description") or attrs.get("summary") or ""
                    detailed = attrs.get("detailed_explanation") or ""
                    summary = brief or detailed or clean_name
                    month = format_idea_month(attrs.get("first_occurrence_month") or attrs.get("month"))

                    names.append(clean_name)
                    texts.append(f"{clean_name}: {summary}")
                    months[clean_name] = month

        if not names or embeddings is None:
            return cls(idea_names=names, idea_summaries=texts, idea_months=months, vectors=None)

        # Batch embed idea texts
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            sub_batch = texts[i : i + batch_size]
            try:
                v = embeddings.embed_documents(sub_batch)
                all_vecs.extend(v)
            except Exception as e:
                logger.warning(f"Error embedding idea texts batch {i}: {e}")
                for _ in sub_batch:
                    all_vecs.append([0.0])

        arr = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = arr / norms

        return cls(
            idea_names=names,
            idea_summaries=texts,
            idea_months=months,
            vectors=normalized,
        )

    def find_best_match(self, query_vec: np.ndarray) -> Tuple[Optional[str], float, str]:
        """
        Compute cosine similarity between normalized query_vec and all indexed ideas.
        Returns (best_idea_name, max_similarity, best_idea_summary).
        """
        if self.vectors is None or len(self.idea_names) == 0:
            return (None, 0.0, "")

        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0.0:
            return (None, 0.0, "")
        q_unit = query_vec / q_norm

        sims = np.dot(self.vectors, q_unit)
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])
        return (
            self.idea_names[best_idx],
            max_sim,
            self.idea_summaries[best_idx],
        )


class RecentIdeaChunkRouter:
    """
    Routes chunks from a batch into three priority tiers based on semantic similarity:
    1. Unassigned (<0.65): Brand new ideas expected -> Priority 1 (Processed first)
    2. High Confidence (>=0.90): Assign directly to existing idea -> Priority 2 (Skips LLM)
    3. Limbo (0.65 <= sim < 0.90): Ambiguous / related -> Priority 3 (Validated / extracted)
    """

    def __init__(
        self,
        idea_index: IdeaEmbeddingIndex,
        embeddings: Optional[Embeddings] = None,
        unassigned_threshold: float = 0.65,
        high_confidence_threshold: float = 0.90,
    ):
        self.idea_index = idea_index
        self.embeddings = embeddings
        self.unassigned_threshold = unassigned_threshold
        self.high_confidence_threshold = high_confidence_threshold

    def route_chunks(
        self,
        chunks: Sequence[Any],
        batch_size: int = 64,
    ) -> BatchRoutingResult:
        """
        Compute embeddings for chunks and route each into Unassigned, High-Confidence, or Limbo.
        """
        result = BatchRoutingResult()
        if not chunks:
            return result

        # If knowledge base is empty (e.g. Batch 1), all chunks are unassigned brand new ideas
        if self.idea_index.vectors is None or len(self.idea_index.idea_names) == 0 or self.embeddings is None:
            for chk in chunks:
                result.unassigned.append(
                    RoutedChunk(
                        chunk=chk,
                        priority=ChunkPriority.UNASSIGNED,
                        similarity=0.0,
                        matched_idea=None,
                        matched_idea_summary="",
                    )
                )
            return result

        # Extract texts for embedding
        chunk_texts: List[str] = []
        for chk in chunks:
            raw = getattr(chk, "text", "") or ""
            chunk_texts.append(raw)

        all_chunk_vecs: List[List[float]] = []
        for i in range(0, len(chunk_texts), batch_size):
            sub_batch = chunk_texts[i : i + batch_size]
            try:
                v = self.embeddings.embed_documents(sub_batch)
                all_chunk_vecs.extend(v)
            except Exception as e:
                logger.warning(f"Error embedding chunks batch {i}: {e}")
                for _ in sub_batch:
                    all_chunk_vecs.append([0.0])

        chunk_arr = np.array(all_chunk_vecs, dtype=np.float32)

        for idx, chk in enumerate(chunks):
            vec = chunk_arr[idx]
            best_idea, sim, best_summary = self.idea_index.find_best_match(vec)

            if sim < self.unassigned_threshold:
                result.unassigned.append(
                    RoutedChunk(
                        chunk=chk,
                        priority=ChunkPriority.UNASSIGNED,
                        similarity=sim,
                        matched_idea=best_idea,
                        matched_idea_summary=best_summary,
                    )
                )
            elif sim >= self.high_confidence_threshold:
                result.high_confidence.append(
                    RoutedChunk(
                        chunk=chk,
                        priority=ChunkPriority.HIGH_CONFIDENCE,
                        similarity=sim,
                        matched_idea=best_idea,
                        matched_idea_summary=best_summary,
                    )
                )
            else:
                result.limbo.append(
                    RoutedChunk(
                        chunk=chk,
                        priority=ChunkPriority.LIMBO,
                        similarity=sim,
                        matched_idea=best_idea,
                        matched_idea_summary=best_summary,
                    )
                )

        return result
