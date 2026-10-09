"""
Embedding-based entity resolution and concept deduplication using OllamaEmbeddings.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from bookeeper.calibre.parser import BookParser
from bookeeper.config import Settings
from bookeeper.processing.extractor import Concept

logger = logging.getLogger(__name__)


class EntityDeduplicator:
    """
    Deduplicates and canonicalizes concept names across books and sections
    using vector embeddings and rolling cosine similarity cache.
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        similarity_threshold: float = 0.85,
    ):
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

        # Canonical name -> (Normalized unit vector, Concept instance)
        self._cache: Dict[str, Tuple[Optional[np.ndarray], Concept]] = {}

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> "EntityDeduplicator":
        from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings, OllamaPool

        e_model = embedding_model or settings.embedding_model
        s_thresh = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )
        try:
            if base_url:
                urls = [u.strip() for u in base_url.split(",") if u.strip()]
                pool = OllamaPool.from_urls(urls, cooldown_seconds=settings.failover_cooldown_seconds)
                embeddings = FailoverOllamaEmbeddings(pool=pool, model=e_model)
            else:
                embeddings = FailoverOllamaEmbeddings.from_settings(settings, model=e_model)
        except Exception as e:
            logger.warning(f"Could not initialize FailoverOllamaEmbeddings: {e}")
            embeddings = None

        return cls(
            embeddings=embeddings,
            similarity_threshold=s_thresh,
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Strip punctuation and redundant whitespace for basic equality matching."""
        cleaned = re.sub(r"[^\w\s-]", "", name).strip().lower()
        return re.sub(r"\s+", " ", cleaned)

    def resolve_concept(self, concept: Concept) -> Concept:
        """
        Compare incoming concept against existing graph concepts.
        If similarity exceeds similarity_threshold, merge and return the canonical Concept.
        Otherwise, register and return the new Concept.
        """
        concept.name = BookParser.repair_mojibake(concept.name.strip())
        raw_name = concept.name
        norm_name = self._normalize_name(raw_name)

        # 1. Exact or case-insensitive string match
        for canon_name, (_, existing) in self._cache.items():
            if self._normalize_name(canon_name) == norm_name:
                self._merge_into(existing, concept)
                return existing

        # 2. Embedding-based semantic cosine similarity
        concept_vec = self._embed(raw_name)
        if concept_vec is not None and self._cache:
            best_sim = -1.0
            best_match: Optional[Concept] = None

            for canon_name, (cached_vec, existing_concept) in self._cache.items():
                if cached_vec is not None:
                    sim = float(np.dot(concept_vec, cached_vec))
                    if sim > best_sim:
                        best_sim = sim
                        best_match = existing_concept

            if best_match is not None and best_sim >= self.similarity_threshold:
                logger.debug(
                    f"Merged concept '{raw_name}' -> '{best_match.name}' (similarity: {best_sim:.3f})"
                )
                self._merge_into(best_match, concept)
                return best_match

        # 3. Register as new canonical node
        self._cache[raw_name] = (concept_vec, concept)
        return concept

    def _merge_into(self, canonical: Concept, incoming: Concept) -> None:
        """Merge incoming concept's related concepts and details into canonical node."""
        for rel in incoming.related_concepts:
            if rel not in canonical.related_concepts and rel != canonical.name:
                canonical.related_concepts.append(rel)

        # Merge descriptions
        if incoming.detailed_explanation and len(incoming.detailed_explanation) > len(getattr(canonical, "detailed_explanation", "")):
            canonical.detailed_explanation = incoming.detailed_explanation
        if incoming.brief_description and len(incoming.brief_description) > len(getattr(canonical, "brief_description", "")):
            canonical.brief_description = incoming.brief_description
        if not getattr(canonical, "supporting_quote", None) and getattr(incoming, "supporting_quote", None):
            canonical.supporting_quote = incoming.supporting_quote

        # If incoming has a significantly richer summary, update canonical summary
        if len(incoming.summary) > len(canonical.summary) + 30:
            if hasattr(canonical, "summary"):
                try:
                    canonical.summary = incoming.summary
                except Exception:
                    pass

        # Retain highest significance weight observed
        canonical.weight = max(getattr(canonical, "weight", 5), getattr(incoming, "weight", 5))

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Compute normalized unit vector for input text."""
        if self.embeddings is None:
            return None

        try:
            vec = self.embeddings.embed_query(text)
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            return (arr / norm) if norm > 0 else arr
        except Exception as e:
            logger.debug(f"Failed to generate embedding for '{text}': {e}")
            return None

    def get_canonical_concepts(self) -> List[Concept]:
        """Return all unique canonical concepts registered so far."""
        return [c for _, c in self._cache.values()]


# Backward-compatibility alias
ConceptDeduplicator = EntityDeduplicator
