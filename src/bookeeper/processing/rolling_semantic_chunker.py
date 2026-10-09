"""
Rolling-Window Semantic Distance Chunker.

Implements a rolling-window semantic distance chunking strategy with hard
token-length safety guards and structural boundary awareness.

Pairwise sentence similarity (comparing sentence i to i+1) suffers from high
variance and splits text prematurely on momentary stylistic tangents or local
narrative shifts. This chunker solves that by comparing multi-sentence context
windows on both sides of candidate boundaries, while enforcing strict min/max
token bounds, respecting paragraph breaks, and supporting sentence overlap.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

try:
    from langchain_core.documents import Document
except ImportError:
    class Document(BaseModel):  # type: ignore
        page_content: str
        metadata: Dict[str, Any] = Field(default_factory=dict)

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    Embeddings = Any  # type: ignore

logger = logging.getLogger(__name__)


class RollingWindowSemanticChunker:
    """
    Chunks text by evaluating rolling-window semantic distance between
    consecutive sentence groups, with hard token limits and paragraph break awareness.

    Parameters
    ----------
    embedding_model : Optional[Union[str, Any, Callable[[List[str]], np.ndarray]]]
        An embedding model name (e.g. 'sentence-transformers/all-MiniLM-L6-v2'),
        a SentenceTransformer instance, a LangChain Embeddings instance, or a custom
        callable accepting a list of strings and returning a 2D numpy array of embeddings.
        If None, structural token-based fallback chunking is used without crashing.
    window_size : int, default=3
        Number of consecutive sentences grouped on the left and right sides of each
        potential split point.
    min_chunk_tokens : int, default=150
        Minimum token count before a split is allowed.
    max_chunk_tokens : int, default=500
        Maximum token count; if reached, forces a split at the best candidate boundary
        or paragraph boundary.
    similarity_threshold : float, default=0.65
        Cosine similarity below which a split point is considered a candidate.
    overlap_sentences : int, default=1
        Number of sentences carried over from the end of Chunk N into the start of Chunk N+1
        to prevent boundary clipping.
    token_counter : Optional[Callable[[str], int]], default=None
        Custom token counting function. If None, uses tiktoken (if available) or a robust
        regex token counter.
    batch_size : int, default=64
        Batch size for vectorizer embedding calls to prevent OOM on large texts.
    breakpoint_threshold_type : str, default='similarity_threshold'
        Strategy for candidate split selection: 'similarity_threshold', 'percentile',
        or 'standard_deviation'.
    breakpoint_threshold_amount : float, default=85.0
        Percentile (e.g. 85.0) or standard deviation multiplier if dynamic threshold is used.
    respect_paragraphs : bool, default=True
        Whether paragraph breaks bias candidate split points.
    """

    def __init__(
        self,
        embedding_model: Optional[Union[str, Any, Callable[[List[str]], np.ndarray]]] = None,
        window_size: int = 3,
        min_chunk_tokens: int = 150,
        max_chunk_tokens: int = 500,
        similarity_threshold: float = 0.65,
        overlap_sentences: int = 1,
        token_counter: Optional[Callable[[str], int]] = None,
        batch_size: int = 64,
        breakpoint_threshold_type: str = "similarity_threshold",
        breakpoint_threshold_amount: float = 85.0,
        respect_paragraphs: bool = True,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if min_chunk_tokens < 1:
            raise ValueError(f"min_chunk_tokens must be >= 1, got {min_chunk_tokens}")
        if max_chunk_tokens < min_chunk_tokens:
            raise ValueError(
                f"max_chunk_tokens ({max_chunk_tokens}) cannot be less than "
                f"min_chunk_tokens ({min_chunk_tokens})"
            )
        if overlap_sentences < 0:
            raise ValueError(f"overlap_sentences must be >= 0, got {overlap_sentences}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.embedding_model = embedding_model
        self.window_size = window_size
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.similarity_threshold = similarity_threshold
        self.overlap_sentences = overlap_sentences
        self.batch_size = batch_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.respect_paragraphs = respect_paragraphs

        self._embed_fn = self._resolve_embed_fn(embedding_model)
        self._token_counter = token_counter or self._default_token_counter

    @property
    def embedding_stats(self) -> dict:
        """Return embedding throughput statistics if supported by underlying embeddings."""
        if hasattr(self.embedding_model, "embedding_stats"):
            return self.embedding_model.embedding_stats
        return {}

    def reset_embedding_stats(self) -> None:
        """Reset embedding throughput counters if supported."""
        if hasattr(self.embedding_model, "reset_stats"):
            self.embedding_model.reset_stats()

    def _resolve_embed_fn(
        self, model: Optional[Union[str, Any, Callable[[List[str]], np.ndarray]]]
    ) -> Optional[Callable[[List[str]], np.ndarray]]:
        """Wrap diverse embedding interfaces into a uniform (List[str]) -> np.ndarray callable."""
        if model is None:
            return None

        if isinstance(model, str):
            try:
                from sentence_transformers import SentenceTransformer

                st = SentenceTransformer(model)
                return lambda texts: np.asarray(
                    st.encode(texts, convert_to_numpy=True), dtype=np.float32
                )
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required when passing a model name string. "
                    "Install it via 'pip install sentence-transformers' or supply an Embeddings instance."
                ) from exc
        elif callable(model) and not hasattr(model, "embed_documents") and not hasattr(model, "encode"):
            return lambda texts: np.asarray(model(texts), dtype=np.float32)
        elif hasattr(model, "encode"):
            # SentenceTransformer or compatible instance
            return lambda texts: np.asarray(model.encode(texts), dtype=np.float32)
        elif hasattr(model, "embed_documents"):
            # LangChain Embeddings instance
            return lambda texts: np.asarray(
                model.embed_documents(texts), dtype=np.float32
            )
        elif hasattr(model, "embed_query"):
            # Embeddings with query only
            return lambda texts: np.asarray(
                [model.embed_query(t) for t in texts], dtype=np.float32
            )
        else:
            raise TypeError(
                f"Unsupported embedding_model type: {type(model)}. Expected str, callable, "
                "SentenceTransformer, or LangChain Embeddings."
            )

    @staticmethod
    def _default_token_counter(text: str) -> int:
        """Estimate token count using tiktoken if available, else regex word/punctuation tokens."""
        if not text:
            return 0
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Fallback regex tokenization: matches words, contractions, numbers, and individual punctuation
            return len(re.findall(r"\w+|[^\w\s]", text))

    def count_tokens(self, text: str) -> int:
        """Return token count for the given string using the configured tokenizer."""
        return self._token_counter(text)

    def _split_into_sentences(self, text: str) -> List[Tuple[str, bool]]:
        """
        Split raw book text into sentences while detecting paragraph breaks (\\n\\n).

        Returns
        -------
        List[Tuple[str, bool]]
            A list of tuples (sentence_text, is_paragraph_end), where is_paragraph_end
            is True if the sentence immediately precedes a paragraph boundary.
        """
        if not text or not text.strip():
            return []

        # Attempt mojibake repair if BookParser is available
        try:
            from bookeeper.calibre.parser import BookParser

            text = BookParser.repair_mojibake(text)
        except Exception:
            pass

        # Normalize line breaks
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        raw_paragraphs = re.split(r"\n\s*\n+", normalized)

        # Common abbreviations to avoid false positive sentence boundaries
        abbr_pattern = re.compile(
            r"\b(mr|mrs|ms|dr|prof|sr|jr|vs|etc|e\.g|i\.e|inc|ltd|co|corp|al|vol|no|fig|st|dept|approx|est)\.\s*$",
            re.IGNORECASE,
        )

        results: List[Tuple[str, bool]] = []

        for para in raw_paragraphs:
            para_clean = para.strip()
            if not para_clean:
                continue

            # Preserve markdown headers or fenced code blocks as atomic units
            if para_clean.startswith("#") or para_clean.startswith("```"):
                results.append((para_clean, True))
                continue

            # Split on terminal punctuation followed by space, respecting quotes/brackets
            raw_splits = re.split(
                r'(?<=[.!?])\s+|(?<=[.!?]["\'”’\)])\s+', para_clean
            )

            merged: List[str] = []
            for s in raw_splits:
                s_clean = s.strip()
                if not s_clean:
                    continue
                # If preceded by an abbreviation, merge back
                if merged and abbr_pattern.search(merged[-1]):
                    merged[-1] = f"{merged[-1]} {s_clean}"
                # If preceded by a single capital initial (e.g., 'J.' in 'J. K. Rowling'), merge back
                elif merged and re.search(r"\b[A-Z]\.\s*$", merged[-1]):
                    merged[-1] = f"{merged[-1]} {s_clean}"
                # Lowercase continuation (dialogue tags like '"Where?" asked Watson.')
                elif (
                    merged
                    and re.match(r"^[a-z0-9]", s_clean)
                    and not merged[-1].endswith((".", "!", "?", '"', "'", "”", "’"))
                ):
                    merged[-1] = f"{merged[-1]} {s_clean}"
                elif (
                    merged
                    and re.match(r"^[a-z]", s_clean)
                    and merged[-1].endswith(('"', "'", "”", "’"))
                ):
                    merged[-1] = f"{merged[-1]} {s_clean}"
                else:
                    merged.append(s_clean)

            for s_idx, sentence_str in enumerate(merged):
                is_para_end = s_idx == len(merged) - 1
                results.append((sentence_str, is_para_end))

        return results

    def _compute_cosine_similarities(
        self, left_contexts: List[str], right_contexts: List[str]
    ) -> np.ndarray:
        """
        Batch compute cosine similarities between paired left and right context strings.
        Deduplicates context strings and computes embeddings in batches of `self.batch_size`.

        Returns
        -------
        np.ndarray
            1D float array of cosine similarities bounded in [-1.0, 1.0].
        """
        if not left_contexts:
            return np.array([], dtype=np.float32)

        if self._embed_fn is None:
            # Fallback when no embeddings are provided: return uniform similarity
            return np.ones(len(left_contexts), dtype=np.float32)

        # 1. Deduplicate unique texts to minimize embedding calls and memory overhead
        unique_texts = list(dict.fromkeys(left_contexts + right_contexts))
        text_to_idx = {t: i for i, t in enumerate(unique_texts)}

        # 2. Batch embed unique texts
        all_embeddings_list: List[np.ndarray] = []
        for b_start in range(0, len(unique_texts), self.batch_size):
            batch = unique_texts[b_start : b_start + self.batch_size]
            b_emb = self._embed_fn(batch)
            if b_emb.ndim == 1:
                b_emb = b_emb.reshape(1, -1)
            all_embeddings_list.append(b_emb)

        all_embeddings = np.vstack(all_embeddings_list)

        # 3. Safe L2 normalization on unique embeddings
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed_embeddings = all_embeddings / norms

        # 4. Map back to paired similarities
        left_indices = [text_to_idx[t] for t in left_contexts]
        right_indices = [text_to_idx[t] for t in right_contexts]

        left_normed = normed_embeddings[left_indices]
        right_normed = normed_embeddings[right_indices]

        similarities = np.sum(left_normed * right_normed, axis=1)
        return np.clip(similarities, -1.0, 1.0)

    def _calculate_effective_threshold(self, similarities: np.ndarray) -> float:
        """Calculate similarity threshold based on configured type."""
        if len(similarities) == 0:
            return self.similarity_threshold

        if self.breakpoint_threshold_type == "similarity_threshold":
            return self.similarity_threshold

        elif self.breakpoint_threshold_type == "percentile":
            # In similarity, a split corresponds to a drop in similarity (lower percentile of similarities)
            pct = max(0.0, min(100.0, 100.0 - self.breakpoint_threshold_amount))
            return float(np.percentile(similarities, pct))

        elif self.breakpoint_threshold_type == "standard_deviation":
            mean_val = float(np.mean(similarities))
            std_val = float(np.std(similarities))
            # Semantic drop is below mean - k * std
            return float(mean_val - self.breakpoint_threshold_amount * std_val)

        return self.similarity_threshold

    def _select_best_split_candidate(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Select the best candidate split point when max_chunk_tokens is reached.

        Picks the split point with lowest cosine similarity, preferring paragraph breaks
        if their similarity is within 10% of the lowest observed similarity.
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")

        min_sim = min(c["similarity"] for c in candidates)
        sim_margin = min_sim * 1.10 if min_sim > 0 else min_sim + 0.10

        if self.respect_paragraphs:
            para_candidates = [
                c for c in candidates if c["is_paragraph"] and c["similarity"] <= sim_margin
            ]
            if para_candidates:
                return min(para_candidates, key=lambda c: c["similarity"])

        return min(candidates, key=lambda c: c["similarity"])

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text using rolling-window semantic distance with safety guards.

        Parameters
        ----------
        text : str
            Raw text to chunk.
        metadata : Optional[Dict[str, Any]], default=None
            Optional parent metadata (book_id, chapter_idx, title) attached to chunks.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing:
            - 'chunk_id': int (sequential identifier)
            - 'text': str (reconstructed chunk text)
            - 'token_count': int (number of tokens in the chunk)
            - 'char_count': int (number of characters)
            - 'word_count': int (number of words)
            - 'start_sentence_idx': int (start sentence index in document)
            - 'end_sentence_idx': int (end sentence index in document, inclusive)
            - 'split_reason': str ('semantic_drop', 'max_tokens_exceeded', or 'document_end')
            - 'metadata': dict (attached metadata)
        """
        base_meta = dict(metadata) if metadata else {}

        # Step A: Split into sentences and detect paragraph boundaries
        sentence_data = self._split_into_sentences(text)
        if not sentence_data:
            return []

        sentences = [s[0] for s in sentence_data]
        is_para_boundary = [s[1] for s in sentence_data]
        sentence_tokens = [self.count_tokens(s) for s in sentences]
        num_sentences = len(sentences)

        # Edge case: single-sentence document
        if num_sentences == 1:
            chunk_str = sentences[0]
            return [
                {
                    "chunk_id": 0,
                    "text": chunk_str,
                    "token_count": self.count_tokens(chunk_str),
                    "char_count": len(chunk_str),
                    "word_count": len(chunk_str.split()),
                    "start_sentence_idx": 0,
                    "end_sentence_idx": 0,
                    "split_reason": "document_end",
                    "metadata": base_meta,
                }
            ]

        # Step B: Compute rolling window embeddings
        num_split_points = num_sentences - 1
        left_contexts: List[str] = []
        right_contexts: List[str] = []

        for i in range(num_split_points):
            left_slice = sentences[max(0, i - self.window_size + 1) : i + 1]
            right_slice = sentences[
                i + 1 : min(num_sentences, i + 1 + self.window_size)
            ]
            left_contexts.append(" ".join(left_slice))
            right_contexts.append(" ".join(right_slice))

        similarities = self._compute_cosine_similarities(left_contexts, right_contexts)
        effective_threshold = self._calculate_effective_threshold(similarities)

        # Step C: Iterate through sentences and construct chunks
        chunks: List[Dict[str, Any]] = []
        start_idx = 0
        chunk_id = 0
        prev_split_idx = -1

        while start_idx < num_sentences:
            current_tokens = 0
            valid_candidates: List[Dict[str, Any]] = []
            split_idx: Optional[int] = None
            split_reason: Optional[str] = None

            for curr in range(start_idx, num_sentences):
                current_tokens += sentence_tokens[curr]

                # Check if we reached the final sentence of the document
                if curr == num_sentences - 1:
                    if (
                        current_tokens >= self.max_chunk_tokens
                        and valid_candidates
                    ):
                        best = self._select_best_split_candidate(valid_candidates)
                        split_idx = best["split_idx"]
                        split_reason = "max_tokens_exceeded"
                    else:
                        split_idx = curr
                        split_reason = "document_end"
                    break

                # Do not split within the carried-over overlap window to ensure forward progress
                if curr <= prev_split_idx:
                    continue

                sim = float(similarities[curr]) if len(similarities) > curr else 1.0
                is_para = is_para_boundary[curr]

                # If current_tokens < min_chunk_tokens: accumulate
                if current_tokens < self.min_chunk_tokens:
                    continue

                # If min_chunk_tokens <= current_tokens <= max_chunk_tokens:
                if self.min_chunk_tokens <= current_tokens <= self.max_chunk_tokens:
                    valid_candidates.append(
                        {
                            "split_idx": curr,
                            "similarity": sim,
                            "is_paragraph": is_para,
                            "tokens": current_tokens,
                        }
                    )

                    # Prefer splitting immediately after paragraph breaks if similarity
                    # is within 10% of the threshold
                    thresh = (
                        effective_threshold * 1.10
                        if (is_para and self.respect_paragraphs)
                        else effective_threshold
                    )

                    if sim < thresh:
                        split_idx = curr
                        split_reason = "semantic_drop"
                        break

                # If current_tokens >= max_chunk_tokens:
                elif current_tokens >= self.max_chunk_tokens:
                    if valid_candidates:
                        best = self._select_best_split_candidate(valid_candidates)
                        split_idx = best["split_idx"]
                    else:
                        split_idx = curr
                    split_reason = "max_tokens_exceeded"
                    break

            if split_idx is None:
                split_idx = num_sentences - 1
                split_reason = "document_end"

            # Reconstruct chunk text preserving original paragraph breaks
            chunk_pieces: List[str] = []
            for j in range(start_idx, split_idx + 1):
                chunk_pieces.append(sentences[j])
                if j < split_idx:
                    if is_para_boundary[j]:
                        chunk_pieces.append("\n\n")
                    else:
                        chunk_pieces.append(" ")
            chunk_text = "".join(chunk_pieces)
            final_token_count = self.count_tokens(chunk_text)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "token_count": final_token_count,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "start_sentence_idx": start_idx,
                    "end_sentence_idx": split_idx,
                    "split_reason": split_reason,
                    "metadata": base_meta,
                }
            )
            chunk_id += 1
            prev_split_idx = split_idx

            if split_reason == "document_end" or split_idx >= num_sentences - 1:
                break

            # Apply sentence overlap: next chunk starts from sentence (split_idx - overlap_sentences + 1)
            next_start = split_idx - self.overlap_sentences + 1
            start_idx = max(start_idx + 1, next_start)

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantically cohesive chunk strings.
        Standard LangChain text splitter interface.
        """
        chunks = self.chunk_text(text)
        return [c["text"] for c in chunks]

    def create_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Document]:
        """
        Create LangChain Document objects from raw texts with rich metadata.
        Standard LangChain text splitter interface.
        """
        docs: List[Document] = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            chunks = self.chunk_text(text, metadata=meta)
            for c in chunks:
                chunk_meta = dict(c.get("metadata", {}))
                chunk_meta.update(
                    {
                        "chunk_id": c["chunk_id"],
                        "token_count": c["token_count"],
                        "char_count": c["char_count"],
                        "word_count": c["word_count"],
                        "start_sentence_idx": c["start_sentence_idx"],
                        "end_sentence_idx": c["end_sentence_idx"],
                        "split_reason": c["split_reason"],
                    }
                )
                docs.append(Document(page_content=c["text"], metadata=chunk_meta))
        return docs

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """
        Split existing LangChain Document objects into semantically chunked Documents.
        Standard LangChain text splitter interface.
        """
        all_docs: List[Document] = []
        for doc in documents:
            text = doc.page_content if hasattr(doc, "page_content") else str(doc)
            meta = dict(doc.metadata) if hasattr(doc, "metadata") and doc.metadata else {}
            chunks = self.chunk_text(text, metadata=meta)
            for c in chunks:
                chunk_meta = dict(meta)
                chunk_meta.update(
                    {
                        "chunk_id": c["chunk_id"],
                        "token_count": c["token_count"],
                        "char_count": c["char_count"],
                        "word_count": c["word_count"],
                        "start_sentence_idx": c["start_sentence_idx"],
                        "end_sentence_idx": c["end_sentence_idx"],
                        "split_reason": c["split_reason"],
                    }
                )
                all_docs.append(Document(page_content=c["text"], metadata=chunk_meta))
        return all_docs

    def get_sentence_similarities(self, text: str) -> List[Dict[str, Any]]:
        """
        Inspect rolling-window similarity scores across candidate split points.
        Useful for debugging, visualization, and auditing topic coherence.
        """
        sentence_data = self._split_into_sentences(text)
        if len(sentence_data) <= 1:
            return []

        sentences = [s[0] for s in sentence_data]
        is_para = [s[1] for s in sentence_data]
        num_sentences = len(sentences)

        left_contexts: List[str] = []
        right_contexts: List[str] = []
        for i in range(num_sentences - 1):
            left_slice = sentences[max(0, i - self.window_size + 1) : i + 1]
            right_slice = sentences[
                i + 1 : min(num_sentences, i + 1 + self.window_size)
            ]
            left_contexts.append(" ".join(left_slice))
            right_contexts.append(" ".join(right_slice))

        similarities = self._compute_cosine_similarities(left_contexts, right_contexts)
        threshold = self._calculate_effective_threshold(similarities)

        records: List[Dict[str, Any]] = []
        for i, sim in enumerate(similarities):
            records.append(
                {
                    "split_idx": i,
                    "sentence_before": sentences[i],
                    "sentence_after": sentences[i + 1],
                    "similarity": round(float(sim), 4),
                    "threshold": round(float(threshold), 4),
                    "is_below_threshold": float(sim) < threshold,
                    "is_paragraph_end": is_para[i],
                    "left_context": left_contexts[i],
                    "right_context": right_contexts[i],
                }
            )
        return records

    def get_statistics(self, text: str) -> Dict[str, Any]:
        """Return diagnostic metrics on document sentences, similarities, and chunks."""
        chunks = self.chunk_text(text)
        sim_records = self.get_sentence_similarities(text)

        sims = [r["similarity"] for r in sim_records]
        tokens = [c["token_count"] for c in chunks]

        return {
            "total_sentences": len(sim_records) + 1 if sim_records else (1 if text.strip() else 0),
            "total_transitions": len(sim_records),
            "total_chunks": len(chunks),
            "mean_chunk_tokens": round(float(np.mean(tokens)), 2) if tokens else 0.0,
            "min_chunk_tokens": min(tokens) if tokens else 0,
            "max_chunk_tokens": max(tokens) if tokens else 0,
            "mean_similarity": round(float(np.mean(sims)), 4) if sims else 1.0,
            "median_similarity": round(float(np.median(sims)), 4) if sims else 1.0,
            "min_similarity": round(float(np.min(sims)), 4) if sims else 1.0,
            "max_similarity": round(float(np.max(sims)), 4) if sims else 1.0,
            "window_size": self.window_size,
        }
