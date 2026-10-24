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
from bookeeper.processing.ollama_pool import (
    format_idea_for_embedding,
    partition_texts_for_embedding,
)

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


def get_chunk_size(chunk: Any) -> int:
    """
    Return the size of a chunk in characters (or length).
    Supports HierarchicalChunk instances, dicts, or generic objects.
    """
    if chunk is None:
        return 0
    if isinstance(chunk, dict):
        cnt = chunk.get("char_count")
        if isinstance(cnt, int) and cnt > 0:
            return cnt
        text = str(chunk.get("text") or "")
        return len(text)
    if hasattr(chunk, "char_count") and getattr(chunk, "char_count", 0):
        return int(chunk.char_count)
    if hasattr(chunk, "text") and chunk.text:
        return len(chunk.text)
    return 0


@dataclass
class RoutedChunk:
    chunk: Any
    priority: ChunkPriority
    similarity: float = 0.0
    matched_idea: Optional[str] = None
    matched_idea_summary: str = ""
    recovered_from_limbo: bool = False


@dataclass
class BatchRoutingResult:
    unassigned: List[RoutedChunk] = field(default_factory=list)
    high_confidence: List[RoutedChunk] = field(default_factory=list)
    limbo: List[RoutedChunk] = field(default_factory=list)
    recovered_from_limbo: List[RoutedChunk] = field(default_factory=list)
    reused_embeddings_count: int = 0
    computed_embeddings_count: int = 0

    @property
    def embeddings_updated(self) -> bool:
        return self.computed_embeddings_count > 0

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

    @property
    def recovered_from_limbo_count(self) -> int:
        return len(self.recovered_from_limbo)

    def recover_limbo_chunks(
        self,
        max_batch_chunk_size: Optional[int] = None,
        threshold: Optional[int] = None,
    ) -> int:
        """
        Recover chunks currently in limbo whose size is >= max_batch_chunk_size,
        restoring them to unassigned (Priority 1 unprocessed LLM extraction queue).
        Returns number of recovered chunks.
        """
        th = max_batch_chunk_size if max_batch_chunk_size is not None else threshold
        if th is None or int(th) <= 0:
            return 0
        th_val = int(th)
        remaining_limbo: List[RoutedChunk] = []
        recovered = 0
        for rc in self.limbo:
            if get_chunk_size(rc.chunk) >= th_val:
                rc.priority = ChunkPriority.UNASSIGNED
                rc.recovered_from_limbo = True
                self.unassigned.append(rc)
                self.recovered_from_limbo.append(rc)
                recovered += 1
            else:
                remaining_limbo.append(rc)
        self.limbo = remaining_limbo
        return recovered

    def get_llm_queue_chunks(self, skip_limbo: bool = False) -> List[Any]:
        """
        Return chunks ordered for LLM extraction:
        Priority 1 (Unassigned, including any chunks recovered from limbo) first,
        followed by Priority 3 (Limbo) unless skip_limbo=True.
        """
        if skip_limbo:
            return [rc.chunk for rc in self.unassigned]
        return [rc.chunk for rc in self.unassigned] + [rc.chunk for rc in self.limbo]


@dataclass
class IdeaEmbeddingIndex:
    """
    Vector index of existing knowledge graph ideas for fast cosine similarity routing.
    Reuses persisted :Idea node embeddings and supports Neo4j HNSW vector index queries.
    """
    idea_names: List[str] = field(default_factory=list)
    idea_summaries: List[str] = field(default_factory=list)
    idea_months: Dict[str, Optional[str]] = field(default_factory=dict)
    vectors: Optional[np.ndarray] = None
    reused_count: int = 0
    embedded_count: int = 0

    @classmethod
    def build_from_graph(
        cls,
        store: ConceptGraphStore,
        embeddings: Optional[Embeddings] = None,
        batch_size: int = 64,
    ) -> "IdeaEmbeddingIndex":
        """
        Extract all Concept/Idea nodes from knowledge graph and reuse or compute their embeddings.
        Newly computed embeddings are cached back onto the Concept/Idea nodes in store.graph.
        """
        names: List[str] = []
        texts: List[str] = []
        node_ids: List[str] = []
        months: Dict[str, Optional[str]] = {}
        cached_vecs: Dict[int, List[float]] = {}
        missing_indices: List[int] = []

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

                    idx = len(names)
                    names.append(clean_name)
                    texts.append(
                        format_idea_for_embedding(
                            clean_name,
                            brief_description=brief,
                            detailed_explanation=detailed,
                            summary=summary,
                        )
                    )
                    node_ids.append(node_id)
                    months[clean_name] = month

                    existing_emb = attrs.get("embedding")
                    if isinstance(existing_emb, (list, np.ndarray)) and len(existing_emb) > 1:
                        cached_vecs[idx] = [float(x) for x in existing_emb]
                    else:
                        missing_indices.append(idx)

        if not names:
            return cls(idea_names=names, idea_summaries=texts, idea_months=months, vectors=None)

        # If all embeddings are missing and no embeddings client is provided, return without vectors
        if not cached_vecs and embeddings is None:
            return cls(idea_names=names, idea_summaries=texts, idea_months=months, vectors=None)

        # Batch embed only the ideas that do not yet have a cached embedding (size-aware batching for full ideas)
        newly_embedded: Dict[int, List[float]] = {}
        if missing_indices and embeddings is not None:
            from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings

            missing_texts = [texts[idx] for idx in missing_indices]
            computed_raw: List[List[float]] = []
            if isinstance(embeddings, FailoverOllamaEmbeddings):
                try:
                    computed_raw = embeddings.embed_documents(
                        missing_texts,
                        batch_size=batch_size,
                        max_batch_chars=12000,
                    )
                except Exception as e:
                    logger.warning(f"Error embedding idea texts across pool: {e}")
                    computed_raw = []
            if not computed_raw:
                idea_batches = partition_texts_for_embedding(
                    missing_texts,
                    max_batch_chars=12000,
                    max_batch_items=batch_size,
                )
                for b_idx, sub_batch in enumerate(idea_batches):
                    try:
                        v = embeddings.embed_documents(sub_batch)
                        computed_raw.extend(v)
                    except Exception as e:
                        logger.warning(f"Error embedding idea texts batch {b_idx}: {e}")
                        for _ in sub_batch:
                            computed_raw.append([])
            for idx, vec in zip(missing_indices, computed_raw):
                if vec and len(vec) > 1:
                    newly_embedded[idx] = [float(x) for x in vec]

        # Determine vector dimension from cached or newly embedded vectors
        dim = 0
        for v in cached_vecs.values():
            if len(v) > 1:
                dim = len(v)
                break
        if dim == 0:
            for v in newly_embedded.values():
                if len(v) > 1:
                    dim = len(v)
                    break

        if dim == 0:
            return cls(idea_names=names, idea_summaries=texts, idea_months=months, vectors=None)

        all_vecs: List[List[float]] = []
        for idx in range(len(names)):
            v = cached_vecs.get(idx) or newly_embedded.get(idx)
            if v and len(v) == dim:
                all_vecs.append(v)
            else:
                all_vecs.append([0.0] * dim)

        arr = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = arr / norms

        # Persist newly computed normalized embeddings back onto Concept/Idea nodes in store.graph
        if store and store.graph and newly_embedded:
            for idx in newly_embedded:
                nid = node_ids[idx]
                if store.graph.has_node(nid):
                    store.graph.nodes[nid]["embedding"] = normalized[idx].tolist()

        return cls(
            idea_names=names,
            idea_summaries=texts,
            idea_months=months,
            vectors=normalized,
            reused_count=len(cached_vecs),
            embedded_count=len(newly_embedded),
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

        if q_unit.shape[0] != self.vectors.shape[1]:
            return (None, 0.0, "")

        sims = np.dot(self.vectors, q_unit)
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])
        return (
            self.idea_names[best_idx],
            max_sim,
            self.idea_summaries[best_idx],
        )

    def query_batch(
        self,
        query_vectors: Sequence[Sequence[float]],
        top_k: int = 3,
        neo4j_exporter: Optional[Any] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Query top-k most similar Idea nodes for a batch of vectors.
        Uses Neo4j native HNSW vector index ('idea_idx') in a single batched UNWIND transaction
        if neo4j_exporter is provided; otherwise uses vectorized matrix cosine similarity.
        """
        if not query_vectors:
            return []

        if neo4j_exporter is not None and hasattr(neo4j_exporter, "query_ideas_hnsw_batch"):
            try:
                hnsw_res = neo4j_exporter.query_ideas_hnsw_batch(query_vectors, top_k=top_k)
                if hnsw_res and any(hnsw_res):
                    return hnsw_res
            except Exception as e:
                logger.debug(f"Neo4j HNSW batch query fallback to in-memory IdeaEmbeddingIndex: {e}")

        if self.vectors is None or len(self.idea_names) == 0:
            return [[] for _ in query_vectors]

        q_arr = np.array(query_vectors, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
        if q_arr.shape[1] != self.vectors.shape[1]:
            return [[] for _ in query_vectors]

        q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
        zero_mask = (q_norms.squeeze(-1) == 0.0)
        q_norms[q_norms == 0.0] = 1.0
        q_unit = q_arr / q_norms

        # Vectorized cosine similarity matrix: (num_queries, num_ideas)
        sim_matrix = np.dot(q_unit, self.vectors.T)
        k = min(top_k, len(self.idea_names))

        results: List[List[Dict[str, Any]]] = []
        for row_idx in range(sim_matrix.shape[0]):
            if zero_mask[row_idx]:
                results.append([])
                continue
            row_sims = sim_matrix[row_idx]
            if k == len(self.idea_names):
                top_indices = np.argsort(row_sims)[::-1]
            else:
                part_idx = np.argpartition(row_sims, -k)[-k:]
                top_indices = part_idx[np.argsort(row_sims[part_idx])[::-1]]
            row_matches: List[Dict[str, Any]] = []
            for idx in top_indices:
                name = self.idea_names[int(idx)]
                summary = self.idea_summaries[int(idx)]
                row_matches.append(
                    {
                        "node_id": f"concept:{name}",
                        "name": name,
                        "brief_description": summary,
                        "detailed_explanation": summary,
                        "summary": summary,
                        "month": self.idea_months.get(name),
                        "score": float(row_sims[int(idx)]),
                    }
                )
            results.append(row_matches)
        return results


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
        neo4j_exporter: Optional[Any] = None,
        max_batch_chunk_size: Optional[int] = None,
    ):
        self.idea_index = idea_index
        self.embeddings = embeddings
        self.unassigned_threshold = unassigned_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.neo4j_exporter = neo4j_exporter
        self.max_batch_chunk_size = max_batch_chunk_size

    @property
    def recovery_threshold(self) -> Optional[int]:
        if self.max_batch_chunk_size is not None and int(self.max_batch_chunk_size) > 0:
            return int(self.max_batch_chunk_size)
        return None

    def should_recover_from_limbo(self, chunk: Any) -> bool:
        th = self.recovery_threshold
        if th is None or th <= 0:
            return False
        return get_chunk_size(chunk) >= th

    def route_chunks(
        self,
        chunks: Sequence[Any],
        batch_size: int = 64,
    ) -> BatchRoutingResult:
        """
        Reuse cached chunk embeddings (or compute missing ones) and route each chunk
        into Unassigned, High-Confidence, or Limbo.
        """
        result = BatchRoutingResult()
        if not chunks:
            return result

        # If knowledge base is empty (e.g. Batch 1), all chunks are unassigned brand new ideas
        has_ideas = self.idea_index.vectors is not None and len(self.idea_index.idea_names) > 0
        if not has_ideas and self.neo4j_exporter is None:
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

        # Check which chunks already have cached embeddings from chunking / ChunkStore
        cached_vecs: Dict[int, List[float]] = {}
        missing_indices: List[int] = []
        for idx, chk in enumerate(chunks):
            existing_emb = getattr(chk, "embedding", None)
            if isinstance(existing_emb, (list, np.ndarray)) and len(existing_emb) > 1:
                cached_vecs[idx] = [float(x) for x in existing_emb]
            else:
                missing_indices.append(idx)

        result.reused_embeddings_count = len(cached_vecs)

        # Embed only chunks that are missing embeddings
        newly_embedded: Dict[int, List[float]] = {}
        if missing_indices:
            if self.embeddings is None and not cached_vecs:
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

            if self.embeddings is not None:
                from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings

                missing_texts = [getattr(chunks[idx], "text", "") or "" for idx in missing_indices]
                computed_raw: List[List[float]] = []
                if isinstance(self.embeddings, FailoverOllamaEmbeddings):
                    try:
                        computed_raw = self.embeddings.embed_documents(
                            missing_texts,
                            batch_size=min(batch_size, 16),
                            max_batch_chars=12000,
                        )
                    except Exception as e:
                        logger.warning(f"Error embedding chunks across pool: {e}")
                        computed_raw = []
                if not computed_raw:
                    chunk_batches = partition_texts_for_embedding(
                        missing_texts,
                        max_batch_chars=12000,
                        max_batch_items=min(batch_size, 16),
                    )
                    for b_idx, sub_batch in enumerate(chunk_batches):
                        try:
                            v = self.embeddings.embed_documents(sub_batch)
                            computed_raw.extend(v)
                        except Exception as e:
                            logger.warning(f"Error embedding chunks batch {b_idx}: {e}")
                            for _ in sub_batch:
                                computed_raw.append([])

                for idx, vec in zip(missing_indices, computed_raw):
                    if vec and len(vec) > 1:
                        arr_v = np.array(vec, dtype=np.float32)
                        norm_v = float(np.linalg.norm(arr_v))
                        unit_v = (arr_v / norm_v).tolist() if norm_v > 0.0 else arr_v.tolist()
                        newly_embedded[idx] = unit_v
                        try:
                            setattr(chunks[idx], "embedding", unit_v)
                        except Exception:
                            pass

        result.computed_embeddings_count = len(newly_embedded)

        # Assemble chunk vectors
        dim = 0
        if self.idea_index.vectors is not None and self.idea_index.vectors.ndim == 2:
            dim = int(self.idea_index.vectors.shape[1])
        if dim == 0:
            for v in cached_vecs.values():
                if len(v) > 1:
                    dim = len(v)
                    break
        if dim == 0:
            for v in newly_embedded.values():
                if len(v) > 1:
                    dim = len(v)
                    break

        if dim == 0:
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

        all_chunk_vecs: List[List[float]] = []
        for idx in range(len(chunks)):
            v = cached_vecs.get(idx) or newly_embedded.get(idx)
            if v and len(v) == dim:
                all_chunk_vecs.append(v)
            else:
                all_chunk_vecs.append([0.0] * dim)

        batch_matches = self.idea_index.query_batch(
            all_chunk_vecs,
            top_k=1,
            neo4j_exporter=self.neo4j_exporter,
        )

        for idx, chk in enumerate(chunks):
            matches = batch_matches[idx] if idx < len(batch_matches) else []
            if matches:
                top_m = matches[0]
                best_idea = top_m.get("name")
                sim = float(top_m.get("score") or 0.0)
                best_summary = top_m.get("brief_description") or top_m.get("summary") or ""
            else:
                best_idea, sim, best_summary = None, 0.0, ""

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
                if self.should_recover_from_limbo(chk):
                    rc = RoutedChunk(
                        chunk=chk,
                        priority=ChunkPriority.UNASSIGNED,
                        similarity=sim,
                        matched_idea=best_idea,
                        matched_idea_summary=best_summary,
                        recovered_from_limbo=True,
                    )
                    result.unassigned.append(rc)
                    result.recovered_from_limbo.append(rc)
                else:
                    result.limbo.append(
                        RoutedChunk(
                            chunk=chk,
                            priority=ChunkPriority.LIMBO,
                            similarity=sim,
                            matched_idea=best_idea,
                            matched_idea_summary=best_summary,
                            recovered_from_limbo=False,
                        )
                    )

        return result
