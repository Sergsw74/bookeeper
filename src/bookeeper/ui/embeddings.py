"""
Concept Embedding Index and Semantic Search for Bookeeper.
Embeds Concept/Idea nodes using dense neural embeddings (JEPA-768 / bge-m3 / nomic-embed-text)
and calculates (cosine_similarity * node_weight) priority ranking.
Persists normalized embedding matrices to disk (.npz) for microsecond retrieval.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from bookeeper.graph.embedding_codec import get_node_embedding

logger = logging.getLogger(__name__)


class ConceptEmbeddingIndex:
    """
    Dense semantic embedding index for Knowledge Graph Concept and Idea nodes.
    Supports:
    - Precomputed vector storage in .npz format (instant disk load in <20ms).
    - Dynamic vector extraction from knowledge_graph.json / spine.
    - JEPA-768 and Ollama embeddings generation for queries.
    - Cosine similarity calculation via fast BLAS matrix multiplication.
    - Priority scoring: priority = cosine_similarity * node_weight.
    - Similarity threshold filtering (e.g. >= 0.65 or 0.70).
    - Top-k seed idea selection (top 5-6 core ideas).
    """

    def __init__(
        self,
        cache_path: Path,
        ollama_url: str = "http://192.168.50.15:11434",
        model_name: str = "jepa-768",
        settings: Optional[Any] = None,
        embeddings_client: Optional[Any] = None,
    ):
        self.cache_path = Path(cache_path)
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.settings = settings
        self.embeddings_client = embeddings_client
        self.ids: List[str] = []
        self.names: List[str] = []
        self.weights: np.ndarray = np.array([], dtype=np.float32)
        self.vectors: np.ndarray = np.array([], dtype=np.float32)  # shape (N, dim), L2-normalized
        self.dim: int = 768

    @property
    def concept_ids(self) -> List[str]:
        """Alias for self.ids to support backend vector search lookups."""
        return self.ids

    @property
    def _norm_matrix(self) -> np.ndarray:
        """Alias for self.vectors to support backend matrix multiplication."""
        return self.vectors

    def is_loaded(self) -> bool:
        """Check if embedding index is loaded in memory."""
        return len(self.ids) > 0 and self.vectors.size > 0

    def load(self, graph_path: Optional[Path] = None) -> bool:
        """Load precomputed embedding index from .npz file or extract from knowledge_graph.json."""
        # 1. Try loading from .npz cache
        if self.cache_path.is_file():
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                self.ids = list(data["ids"])
                self.names = list(data["names"])
                self.weights = data["weights"].astype(np.float32)
                self.vectors = data["vectors"].astype(np.float32)
                self.dim = self.vectors.shape[1] if len(self.vectors.shape) > 1 else 768
                logger.info(f"Loaded {len(self.ids)} concept embeddings from {self.cache_path} (dim={self.dim})")
                return True
            except Exception as e:
                logger.warning(f"Failed to load embedding cache {self.cache_path}: {e}")

        # 2. Fallback: extract embeddings directly from knowledge_graph.json / spine
        target_graph = graph_path or (self.cache_path.parent / "knowledge_graph.json")
        if not target_graph.is_file() and (self.cache_path.parent / "knowledge_graph_spine.json").is_file():
            target_graph = self.cache_path.parent / "knowledge_graph_spine.json"

        if target_graph.is_file():
            try:
                with open(target_graph, "r", encoding="utf-8") as gf:
                    g_data = json.load(gf)

                loaded_ids = []
                loaded_names = []
                loaded_weights = []
                loaded_vectors = []

                nodes = g_data.get("nodes", [])
                for n in nodes:
                    ntype = n.get("type", "")
                    nid = str(n.get("id") or "")
                    if ntype == "Chunk" or nid.startswith("chunk:"):
                        continue
                    emb = get_node_embedding(n, return_numpy=True)
                    if emb is not None and len(emb) > 1:
                        name = str(n.get("name") or nid.replace("concept:", ""))
                        w = float(n.get("weight") or 1.0)
                        arr = np.asarray(emb, dtype=np.float32)
                        norm = float(np.linalg.norm(arr))
                        if norm > 0:
                            arr = arr / norm
                        loaded_ids.append(nid)
                        loaded_names.append(name)
                        loaded_weights.append(w)
                        loaded_vectors.append(arr)

                if loaded_vectors:
                    self.ids = loaded_ids
                    self.names = loaded_names
                    self.weights = np.array(loaded_weights, dtype=np.float32)
                    self.vectors = np.vstack(loaded_vectors)
                    self.dim = self.vectors.shape[1]
                    logger.info(f"Extracted {len(self.ids)} concept embeddings from {target_graph} (dim={self.dim})")
                    self.save()
                    return True
            except Exception as e:
                logger.warning(f"Could not extract embeddings from {target_graph}: {e}")

        return False

    def save(self):
        """Save embedding index to .npz file."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache_path,
                ids=np.array(self.ids, dtype=object),
                names=np.array(self.names, dtype=object),
                weights=self.weights,
                vectors=self.vectors,
            )
            logger.info(f"Saved {len(self.ids)} concept embeddings to {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")

    def _get_embeddings_client(self) -> Optional[Any]:
        if self.embeddings_client is not None:
            return self.embeddings_client
        if self.settings is not None:
            try:
                from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings
                self.embeddings_client = FailoverOllamaEmbeddings.from_settings(self.settings)
                return self.embeddings_client
            except Exception as e:
                logger.debug(f"Could not initialize FailoverOllamaEmbeddings in UI: {e}")
        return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate a dense embedding vector for a single query text (JEPA and Ollama compatible)."""
        cleaned = text.strip()
        if not cleaned:
            return None

        # 1. Primary: Use configured Embeddings client (JEPA / whitened 768-d pipeline)
        client = self._get_embeddings_client()
        if client is not None:
            try:
                if hasattr(client, "embed_query"):
                    vec = client.embed_query(cleaned)
                else:
                    vecs = client.embed_documents([cleaned])
                    vec = vecs[0] if vecs else None
                if vec is not None and len(vec) > 1:
                    arr = np.asarray(vec, dtype=np.float32)
                    norm = float(np.linalg.norm(arr))
                    return arr / norm if norm > 0 else arr
            except Exception as e:
                logger.warning(f"Embeddings client query failed in UI: {e}")

        # 2. Fallback: Direct Ollama HTTP endpoints
        urls = [self.ollama_url, "http://192.168.50.15:11434", "http://localhost:11434", "http://127.0.0.1:11434"]
        seen_urls = set()
        candidate_urls = []
        for u in urls:
            if u and u not in seen_urls:
                seen_urls.add(u)
                candidate_urls.append(u)

        for u in candidate_urls:
            # Try /api/embed first
            try:
                with httpx.Client(timeout=2.5) as client:
                    res = client.post(
                        f"{u}/api/embed",
                        json={"model": self.model_name, "input": cleaned},
                    )
                    if res.status_code == 200:
                        embs = res.json().get("embeddings", [])
                        if embs:
                            self.ollama_url = u  # Cache working URL
                            vec = np.array(embs[0], dtype=np.float32)
                            norm = np.linalg.norm(vec)
                            return vec / norm if norm > 0 else vec
            except Exception as e:
                logger.debug(f"/api/embed attempt at {u} failed: {e}")

            # Fallback to legacy /api/embeddings endpoint
            try:
                with httpx.Client(timeout=2.5) as client:
                    res = client.post(
                        f"{u}/api/embeddings",
                        json={"model": self.model_name, "prompt": cleaned},
                    )
                    if res.status_code == 200:
                        emb = res.json().get("embedding", [])
                        if emb:
                            self.ollama_url = u  # Cache working URL
                            vec = np.array(emb, dtype=np.float32)
                            norm = np.linalg.norm(vec)
                            return vec / norm if norm > 0 else vec
            except Exception as e:
                logger.debug(f"/api/embeddings attempt at {u} failed: {e}")

        logger.warning(f"Failed to generate query embedding via Ollama endpoints {candidate_urls}")
        return None

    def query_core_ideas_with_telemetry(
        self,
        query: str,
        similarity_threshold: float = 0.70,
        top_k: int = 6,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract up to top-k core ideas based on embedding similarity and node weight,
        returning both the ranked core ideas and full embedding debug telemetry:
        1. Embed user query with neural embedding model.
        2. Compute cosine similarity against all concept embeddings.
        3. Calculate priority = cosine_similarity * node_weight.
        4. Filter candidates above similarity_threshold.
        5. Compute pairwise cosine similarity matrix between retrieved ideas.
        6. Return (core_ideas, debug_meta).
        """
        if not self.is_loaded():
            self.load()

        debug_meta: Dict[str, Any] = {
            "model_name": self.model_name,
            "dimensions": int(self.vectors.shape[1]) if self.vectors is not None else 0,
            "total_indexed_concepts": len(self.ids) if self.ids is not None else 0,
            "endpoint": self.ollama_url,
            "similarity_threshold": similarity_threshold,
            "top_k_requested": top_k,
            "candidates_above_threshold_count": 0,
            "candidates_above_relaxed_count": 0,
            "top_k_retrieved": 0,
            "pairwise_similarity": {
                "names": [],
                "ids": [],
                "matrix": [],
            },
        }

        if not self.is_loaded():
            logger.warning("ConceptEmbeddingIndex has no embeddings loaded.")
            return [], debug_meta

        query_vec = self.embed_text(query)
        if query_vec is None:
            return [], debug_meta

        # Cosine similarity (vectors are L2-normalized, so dot product = cosine)
        cosines = self.vectors @ query_vec

        # Priority calculation: product of cosine similarity and node weight
        priorities = cosines * self.weights

        # Candidate pool: include candidates above relaxed similarity threshold (min 0.50)
        # so high-weight concepts (e.g. weight 8+ with cosine 0.68) aren't prematurely eliminated
        candidate_pool = np.where(cosines >= min(0.50, similarity_threshold * 0.80))[0]
        if len(candidate_pool) == 0:
            candidate_pool = np.argsort(-cosines)[:top_k]

        debug_meta["candidates_above_threshold_count"] = int(np.sum(cosines >= similarity_threshold))
        debug_meta["candidates_above_relaxed_count"] = int(len(candidate_pool))

        # Sort candidate pool by priority score (cosine * weight) descending
        ranked_indices = candidate_pool[np.argsort(-priorities[candidate_pool])]

        results = []
        seen_names = set()
        top_indices = []

        for idx in ranked_indices:
            cid = str(self.ids[idx])
            name = str(self.names[idx])
            norm_name = name.lower().strip()
            # Deduplicate case variations (e.g. 'Brier score' vs 'Brier Score'), keeping the highest priority one
            if norm_name in seen_names:
                continue
            seen_names.add(norm_name)
            top_indices.append(idx)

            w = float(self.weights[idx])
            cos = float(cosines[idx])
            prio = float(priorities[idx])
            results.append({
                "rank": len(results) + 1,
                "id": cid,
                "name": name,
                "weight": w,
                "cosine_similarity": round(cos, 4),
                "priority_score": round(prio, 4),
                "threshold_met": bool(cos >= similarity_threshold or (cos >= 0.65 and w >= 7.0)),
            })
            if len(results) >= top_k:
                break

        debug_meta["top_k_retrieved"] = len(results)

        # Compute pairwise cosine similarity matrix between the top retrieved ideas
        if len(top_indices) > 0:
            top_vecs = self.vectors[top_indices]
            # Since vectors are normalized, dot product = cosine similarity
            pairwise_mat = top_vecs @ top_vecs.T
            debug_meta["pairwise_similarity"] = {
                "names": [str(self.names[i]) for i in top_indices],
                "ids": [str(self.ids[i]) for i in top_indices],
                "matrix": [[round(float(val), 4) for val in row] for row in pairwise_mat],
            }

        return results, debug_meta

    def query_core_ideas(
        self,
        query: str,
        similarity_threshold: float = 0.70,
        top_k: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Extract up to top-k core ideas based on embedding similarity and node weight.
        Convenience method returning just the list of ideas (backwards compatible).
        """
        results, _ = self.query_core_ideas_with_telemetry(
            query=query,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
        )
        return results
