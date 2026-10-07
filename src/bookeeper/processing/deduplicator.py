"""
Embedding-based entity resolution and concept deduplication.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np

from bookeeper.config import OllamaSettings, ProcessingSettings
from bookeeper.processing.extractor import ConceptNode

logger = logging.getLogger(__name__)


class EntityDeduplicator:
    """Resolves and merges semantically equivalent concepts across books and sections."""

    def __init__(
        self,
        ollama_settings: Optional[OllamaSettings] = None,
        processing_settings: Optional[ProcessingSettings] = None,
    ):
        self.ollama_settings = ollama_settings or OllamaSettings()
        self.processing_settings = processing_settings or ProcessingSettings()
        self.embeddings_url = f"{self.ollama_settings.base_url.rstrip('/')}/api/embeddings"

        # Canonical name -> (embedding_vector, ConceptNode)
        self.known_concepts: Dict[str, Tuple[Optional[np.ndarray], ConceptNode]] = {}
        # Alias / alternative name -> Canonical name
        self.alias_map: Dict[str, str] = {}

    def normalize_name(self, name: str) -> str:
        """Strip punctuation and redundant whitespace for base key matching."""
        cleaned = re.sub(r"[^\w\s-]", "", name).strip().lower()
        return re.sub(r"\s+", " ", cleaned)

    def resolve(self, concept: ConceptNode) -> ConceptNode:
        """
        Check if concept exists in registry; if semantically similar entity is found,
        merge and return canonical ConceptNode. Otherwise register new node.
        """
        raw_name = concept.name.strip()
        norm_key = self.normalize_name(raw_name)

        # 1. Exact alias / normalized match
        if norm_key in self.alias_map:
            canonical_name = self.alias_map[norm_key]
            _, existing_node = self.known_concepts[canonical_name]
            self._merge_into(existing_node, concept)
            return existing_node

        # Check existing known concepts by exact case-insensitive match
        for canon_name, (_, node) in self.known_concepts.items():
            if self.normalize_name(canon_name) == norm_key:
                self.alias_map[norm_key] = canon_name
                self._merge_into(node, concept)
                return node

        # 2. Embedding similarity matching
        concept_vec = self._get_embedding(concept.name)
        if concept_vec is not None and self.known_concepts:
            best_match_name = None
            best_sim = -1.0

            for canon_name, (existing_vec, _) in self.known_concepts.items():
                if existing_vec is not None:
                    sim = self._cosine_similarity(concept_vec, existing_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_match_name = canon_name

            if (
                best_match_name is not None
                and best_sim >= self.processing_settings.dedup_similarity_threshold
            ):
                logger.debug(
                    f"Resolved '{concept.name}' -> '{best_match_name}' (sim: {best_sim:.3f})"
                )
                _, existing_node = self.known_concepts[best_match_name]
                self.alias_map[norm_key] = best_match_name
                self._merge_into(existing_node, concept)
                return existing_node

        # 3. New concept registration
        self.known_concepts[concept.name] = (concept_vec, concept)
        self.alias_map[norm_key] = concept.name
        for alias in concept.aliases:
            self.alias_map[self.normalize_name(alias)] = concept.name

        return concept

    def _merge_into(self, target: ConceptNode, source: ConceptNode) -> None:
        """Merge aliases and supplementary metadata into the canonical node."""
        if source.name != target.name and source.name not in target.aliases:
            target.aliases.append(source.name)
        for a in source.aliases:
            if a != target.name and a not in target.aliases:
                target.aliases.append(a)
        # Keep richer description if source has more detail
        if len(source.description) > len(target.description) and len(target.description) < 40:
            target.description = source.description

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Fetch embedding from local Ollama instance."""
        payload = {
            "model": self.ollama_settings.embedding_model,
            "prompt": text,
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(self.embeddings_url, json=payload)
                if resp.status_code == 200:
                    vec = resp.json().get("embedding")
                    if vec:
                        arr = np.array(vec, dtype=np.float32)
                        norm = np.linalg.norm(arr)
                        return arr / norm if norm > 0 else arr
        except Exception as e:
            logger.debug(f"Ollama embedding unavailable for '{text}': {e}")
        return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between normalized vectors."""
        dot = np.dot(a, b)
        return float(dot)
