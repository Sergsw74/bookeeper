"""
Embedding-based entity resolution and concept deduplication using OllamaEmbeddings
and two-tier LLM-assisted verification and synthesis.
"""

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from bookeeper.calibre.parser import BookParser
from bookeeper.config import Settings
from bookeeper.processing.extractor import Concept
from bookeeper.processing.ollama_pool import (
    DaemonThreadPoolExecutor,
    FailoverOllamaEmbeddings,
    format_idea_for_embedding,
    partition_texts_for_embedding,
)

logger = logging.getLogger(__name__)


class ConceptDeduplicationDecision(BaseModel):
    """Structured decision and synthesized concept details from LLM disambiguation."""

    is_same_concept: bool = Field(
        description="True if both concepts refer to the same underlying entity, idea, technique, strategy, or theme.",
    )
    canonical_name: Optional[str] = Field(
        default=None,
        description="If similar, provide the most concise, accurate, canonical name representing both concepts (2-4 words, capitalized noun phrase).",
    )
    brief_description: Optional[str] = Field(
        default=None,
        description="If similar, provide a synthesized, general 1-2 sentence definition combining the essence of both concepts.",
    )
    detailed_explanation: Optional[str] = Field(
        default=None,
        description="If similar, provide a synthesized comprehensive explanation combining details, mechanisms, and nuances from both concepts.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="A concise sentence explaining why they are or are not the same concept.",
    )


class EntityDeduplicator:
    """
    Deduplicates and canonicalizes concept names across books and sections
    using vector embeddings, two-tier thresholding, and LLM-assisted synthesis.
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        similarity_threshold: float = 0.80,
        high_similarity_threshold: float = 0.95,
        extractor: Optional[Any] = None,
    ):
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.high_similarity_threshold = high_similarity_threshold
        self.extractor = extractor

        # Canonical name -> (Normalized unit vector, Concept instance)
        self._cache: Dict[str, Tuple[Optional[np.ndarray], Concept]] = {}
        # Normalized name -> Canonical name in _cache for O(1) lookup
        self._norm_to_canon: Dict[str, str] = {}
        # Cache of LLM disambiguation decisions: (norm_name_a, norm_name_b) -> decision
        self._disambiguation_cache: Dict[Tuple[str, str], ConceptDeduplicationDecision] = {}
        # Native C++ HNSW usearch index for logarithmic vector similarity search
        self._usearch_index: Optional[Any] = None
        self._cached_canon_order: List[str] = []

    def _register_in_cache(
        self,
        canon_name: str,
        vec: Optional[np.ndarray],
        concept: Concept,
    ) -> None:
        """Register or update a concept in _cache and maintain _norm_to_canon index and usearch index."""
        self._cache[canon_name] = (vec, concept)
        norm_key = self._normalize_name(canon_name)
        if norm_key:
            self._norm_to_canon[norm_key] = canon_name

        if vec is not None and isinstance(vec, np.ndarray) and vec.ndim == 1 and len(vec) > 1:
            arr = vec.astype(np.float32)
            norm = float(np.linalg.norm(arr))
            unit_v = (arr / norm) if norm > 0.0 else arr

            if self._usearch_index is None:
                from usearch.index import Index

                self._usearch_index = Index(ndim=int(unit_v.shape[0]), metric="cos")

            if canon_name in self._cached_canon_order:
                idx = self._cached_canon_order.index(canon_name)
                if idx in self._usearch_index:
                    self._usearch_index.remove(idx)
                self._usearch_index.add(idx, unit_v)
            else:
                idx = len(self._cached_canon_order)
                self._cached_canon_order.append(canon_name)
                self._usearch_index.add(idx, unit_v)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        high_similarity_threshold: Optional[float] = None,
        extractor: Optional[Any] = None,
        embeddings: Optional[Embeddings] = None,
    ) -> "EntityDeduplicator":
        from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings, OllamaPool

        e_model = embedding_model or settings.embedding_model
        s_thresh = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(settings, "similarity_threshold", 0.80)
        )
        high_s_thresh = (
            high_similarity_threshold
            if high_similarity_threshold is not None
            else getattr(settings, "high_similarity_threshold", 0.95)
        )
        if embeddings is None:
            try:
                if base_url:
                    cd = getattr(settings, "resolved_analysis_cooldown_seconds", getattr(settings, "failover_cooldown_seconds", 600))
                    pool = OllamaPool.from_urls([u.strip() for u in base_url.split(",") if u.strip()], cooldown_seconds=cd)
                    embeddings = FailoverOllamaEmbeddings(pool=pool, model=e_model)
                else:
                    embeddings = FailoverOllamaEmbeddings.from_settings(settings, model=e_model)
            except Exception as e:
                logger.warning(f"Could not initialize FailoverOllamaEmbeddings: {e}")
                embeddings = None

        return cls(
            embeddings=embeddings,
            similarity_threshold=s_thresh,
            high_similarity_threshold=high_s_thresh,
            extractor=extractor,
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Strip punctuation and redundant whitespace for basic equality matching."""
        cleaned = re.sub(r"[^\w\s-]", "", name).strip().lower()
        return re.sub(r"\s+", " ", cleaned)

    def seed_from_graph(self, store: Any) -> int:
        """
        Seed canonical concept cache from existing Concept/Idea nodes in ConceptGraphStore,
        reusing stored node embeddings so existing ideas do not need to be re-embedded.
        """
        if store is None or getattr(store, "graph", None) is None:
            return 0
        seeded = 0
        for node_id, attrs in store.graph.nodes(data=True):
            if attrs.get("type") != "Concept" and not attrs.get("is_idea"):
                continue
            raw_name = attrs.get("name") or str(node_id).replace("concept:", "")
            clean_name = BookParser.repair_mojibake(raw_name).strip()
            if not clean_name or clean_name in self._cache:
                continue

            brief = attrs.get("brief_description") or attrs.get("summary") or ""
            detailed = attrs.get("detailed_explanation") or ""
            cat = attrs.get("category") or "Idea"
            weight = int(attrs.get("weight") or 5)
            existing_emb = attrs.get("embedding")
            vec_arr: Optional[np.ndarray] = None
            emb_list: Optional[List[float]] = None
            if isinstance(existing_emb, (list, np.ndarray)) and len(existing_emb) > 1:
                arr = np.array(existing_emb, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                vec_arr = (arr / norm) if norm > 0.0 else arr
                emb_list = vec_arr.tolist()

            concept_obj = Concept(
                name=clean_name,
                category=cat,
                brief_description=brief,
                detailed_explanation=detailed,
                summary=brief or detailed,
                weight=weight,
            )
            if emb_list is not None:
                concept_obj.embedding = emb_list
            self._register_in_cache(clean_name, vec_arr, concept_obj)
            seeded += 1
        return seeded

    def seed_from_idea_index(self, idea_index: Any) -> int:
        """
        Seed the deduplicator cache and usearch index from an existing IdeaEmbeddingIndex.
        """
        if not getattr(idea_index, "idea_names", None):
            return 0
        names = idea_index.idea_names
        summaries = getattr(idea_index, "idea_summaries", [])
        vectors = getattr(idea_index, "vectors", None)
        seeded = 0
        for idx, name in enumerate(names):
            clean_name = BookParser.repair_mojibake(name).strip()
            if not clean_name:
                continue
            summary = summaries[idx] if idx < len(summaries) else ""
            vec_arr = None
            if vectors is not None and idx < len(vectors):
                vec_arr = np.asarray(vectors[idx], dtype=np.float32)
            concept_obj = Concept(
                name=clean_name,
                category="Idea",
                brief_description=summary,
                detailed_explanation=summary,
                summary=summary,
            )
            if vec_arr is not None:
                concept_obj.embedding = vec_arr.tolist()
            self._register_in_cache(clean_name, vec_arr, concept_obj)
            seeded += 1
        return seeded

    def resolve_concept(self, concept: Concept) -> Concept:
        """
        Compare incoming concept against existing graph concepts.
        - Similarity >= high_similarity_threshold (0.95) or exact name match:
          Definitely the same concept. Merge into canonical Concept directly without LLM.
        - Similarity between similarity_threshold (0.80) and high_similarity_threshold (0.95):
          Probably similar. Ask LLM model with concept names and descriptions to verify
          if they are actually similar, and synthesize general name and descriptions.
        - Similarity < similarity_threshold (0.80):
          Distinct concept. Register and return as new canonical Concept.
        """
        concept.name = BookParser.repair_mojibake(concept.name.strip())
        raw_name = concept.name
        norm_name = self._normalize_name(raw_name)

        # 1. O(1) Exact or case-insensitive normalized string match (highest confidence)
        matched_canon = self._norm_to_canon.get(norm_name)
        if matched_canon and matched_canon in self._cache:
            _, existing = self._cache[matched_canon]
            self._merge_into(existing, concept)
            return existing
        for canon_name, (_, existing) in self._cache.items():
            if self._normalize_name(canon_name) == norm_name:
                self._norm_to_canon[norm_name] = canon_name
                self._merge_into(existing, concept)
                return existing

        # 2. Embedding-based semantic cosine similarity (reuse concept.embedding if already set)
        if concept.embedding and len(concept.embedding) > 1:
            arr = np.array(concept.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            concept_vec: Optional[np.ndarray] = (arr / norm) if norm > 0.0 else arr
        else:
            concept_vec = self._embed(raw_name)
            if concept_vec is not None:
                concept.embedding = concept_vec.tolist()

        if concept_vec is not None and self._cache:
            best_sim = -1.0
            best_match: Optional[Concept] = None
            best_canon_name: Optional[str] = None

            # Fast in-memory native C++ HNSW usearch lookup
            if self._usearch_index is not None and len(self._usearch_index) > 0:
                matches = self._usearch_index.search(concept_vec, 1)
                if len(matches.keys) > 0 and len(matches.distances) > 0:
                    d = float(matches.distances[0])
                    if not np.isnan(d):
                        sim = float(max(-1.0, min(1.0, 1.0 - d)))
                        k_idx = int(matches.keys[0])
                        if 0 <= k_idx < len(self._cached_canon_order):
                            c_canon = self._cached_canon_order[k_idx]
                            if c_canon in self._cache:
                                best_sim = sim
                                best_match = self._cache[c_canon][1]
                                best_canon_name = c_canon
            else:
                for canon_name, (cached_vec, existing_concept) in self._cache.items():
                    if cached_vec is not None and cached_vec.shape == concept_vec.shape:
                        sim = float(np.dot(concept_vec, cached_vec))
                        if sim > best_sim:
                            best_sim = sim
                            best_match = existing_concept
                            best_canon_name = canon_name

            # Case A: High Confidence (>= 0.95) -> Definitive match, merge directly
            if best_match is not None and best_sim >= self.high_similarity_threshold:
                logger.debug(
                    f"Direct merge '{raw_name}' -> '{best_match.name}' (high confidence similarity: {best_sim:.3f})"
                )
                self._merge_into(best_match, concept)
                return best_match

            # Case B: Probable match (0.80 <= similarity < 0.95) -> Disambiguate and synthesize via LLM
            if best_match is not None and best_sim >= self.similarity_threshold:
                if self.extractor is not None:
                    decision = self._ask_llm_disambiguation(best_match, concept, best_sim)
                    if decision.is_same_concept:
                        logger.info(
                            f"LLM confirmed merge '{raw_name}' -> '{best_match.name}' "
                            f"(sim: {best_sim:.3f}, reasoning: {decision.reasoning})"
                        )
                        self._merge_with_llm_decision(best_match, concept, decision, best_canon_name)
                        return best_match
                    else:
                        logger.info(
                            f"LLM rejected merge between '{raw_name}' and '{best_match.name}' "
                            f"(sim: {best_sim:.3f}, reasoning: {decision.reasoning}). Keeping as separate concepts."
                        )
                else:
                    # Fallback if no LLM extractor is configured: merge based on vector similarity
                    logger.debug(
                        f"Merged concept '{raw_name}' -> '{best_match.name}' (similarity: {best_sim:.3f}, no LLM)"
                    )
                    self._merge_into(best_match, concept)
                    return best_match

        # 3. Register as new canonical node
        self._register_in_cache(raw_name, concept_vec, concept)
        return concept

    def resolve_concepts_batch(
        self,
        concepts: Sequence[Concept],
        neo4j_exporter: Optional[Any] = None,
        top_k: int = 3,
        batch_size: int = 64,
        on_progress: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> List[Concept]:
        """
        Batch-resolve all extracted ideas/concepts from an article or book:
        1. Batch-embed all unique concept names that do not yet have embeddings.
        2. Query all extracted idea vectors in a single batched transaction against
           the native Neo4j HNSW vector index on :Idea(embedding)
           (UNWIND $idea_vectors AS vec CALL db.index.vector.queryNodes('idea_idx', 3, vec)),
           plus vectorized comparison against in-memory cached ideas.
        3. Concurrently disambiguate any probable matches (0.80 <= sim < 0.95) across the LLM pool
           and resolve each concept with live progress callbacks.
        """
        if not concepts:
            return []

        # Clean names first
        for c in concepts:
            c.name = BookParser.repair_mojibake(c.name.strip())

        # 1. Batch-embed any concepts missing embeddings and not already in exact-name cache
        missing_texts: List[str] = []
        missing_text_to_vec: Dict[str, np.ndarray] = {}
        concept_to_embed_text: Dict[str, str] = {}
        seen_missing: set[str] = set()

        for c in concepts:
            raw_name = c.name
            norm_name = self._normalize_name(raw_name)
            embed_text = format_idea_for_embedding(
                raw_name,
                brief_description=getattr(c, "brief_description", "") or "",
                detailed_explanation=getattr(c, "detailed_explanation", "") or "",
                summary=getattr(c, "summary", "") or "",
                description=getattr(c, "description", "") or "",
                short_description=getattr(c, "short_description", "") or "",
            )
            concept_to_embed_text[raw_name] = embed_text

            if c.embedding and len(c.embedding) > 1:
                arr = np.array(c.embedding, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                unit = (arr / norm) if norm > 0.0 else arr
                c.embedding = unit.tolist()
                missing_text_to_vec[embed_text] = unit
                continue

            canon_hit = self._norm_to_canon.get(norm_name) or (raw_name if raw_name in self._cache else None)
            if canon_hit and canon_hit in self._cache and self._cache[canon_hit][0] is not None:
                vec = self._cache[canon_hit][0]
                if vec is not None:
                    c.embedding = vec.tolist()
                    missing_text_to_vec[embed_text] = vec
                    continue

            if embed_text not in seen_missing:
                seen_missing.add(embed_text)
                missing_texts.append(embed_text)

        if missing_texts and self.embeddings is not None:
            embedded_so_far = 0
            prog_lock = threading.Lock()
            if on_progress:
                on_progress("embed_ideas", 0, len(missing_texts), f"Embedding {len(missing_texts)} unique ideas")

            def _on_batch_done(count: int) -> None:
                nonlocal embedded_so_far
                with prog_lock:
                    embedded_so_far += count
                    cur = embedded_so_far
                if on_progress:
                    on_progress(
                        "embed_ideas",
                        cur,
                        len(missing_texts),
                        f"Embedded {cur}/{len(missing_texts)} ideas",
                    )

            pooled_ok = False
            if isinstance(self.embeddings, FailoverOllamaEmbeddings):
                try:
                    raw_vecs = self.embeddings.embed_documents(
                        missing_texts,
                        batch_size=batch_size,
                        max_batch_chars=12000,
                        on_progress=_on_batch_done,
                    )
                    for txt, v in zip(missing_texts, raw_vecs):
                        if v and len(v) > 1:
                            arr = np.array(v, dtype=np.float32)
                            norm = float(np.linalg.norm(arr))
                            unit = (arr / norm) if norm > 0.0 else arr
                            missing_text_to_vec[txt] = unit
                    pooled_ok = True
                except Exception as e:
                    logger.debug(f"Pooled batch concept embedding fallback to sub-batches: {e}")

            if not pooled_ok:
                text_batches = partition_texts_for_embedding(
                    missing_texts,
                    max_batch_chars=12000,
                    max_batch_items=batch_size,
                )
                for b_idx, sub in enumerate(text_batches):
                    try:
                        if hasattr(self.embeddings, "embed_documents"):
                            raw_vecs = self.embeddings.embed_documents(sub)
                        else:
                            raw_vecs = [self.embeddings.embed_query(t) for t in sub]
                        for txt, v in zip(sub, raw_vecs):
                            if v and len(v) > 1:
                                arr = np.array(v, dtype=np.float32)
                                norm = float(np.linalg.norm(arr))
                                unit = (arr / norm) if norm > 0.0 else arr
                                missing_text_to_vec[txt] = unit
                    except Exception as e:
                        logger.debug(f"Batch concept embedding failed on sub-batch {b_idx}: {e}")
                    _on_batch_done(len(sub))

        for c in concepts:
            embed_text = concept_to_embed_text.get(c.name, c.name)
            if not c.embedding and embed_text in missing_text_to_vec:
                c.embedding = missing_text_to_vec[embed_text].tolist()

        # 2. Query Neo4j HNSW vector index ('idea_idx') for all extracted idea vectors in a single batched transaction:
        #    UNWIND $idea_vectors AS vec CALL db.index.vector.queryNodes('idea_idx', 3, vec)
        hnsw_candidates_by_idx: Dict[int, List[Dict[str, Any]]] = {}
        if neo4j_exporter is not None and hasattr(neo4j_exporter, "query_ideas_hnsw_batch"):
            query_indices: List[int] = []
            idea_vectors: List[List[float]] = []
            for idx, c in enumerate(concepts):
                if c.embedding and len(c.embedding) > 1:
                    query_indices.append(idx)
                    idea_vectors.append(c.embedding)
            if idea_vectors:
                if on_progress:
                    on_progress("hnsw_query", 0, len(idea_vectors), f"Querying Neo4j HNSW index for {len(idea_vectors)} ideas")
                try:
                    batch_hnsw = neo4j_exporter.query_ideas_hnsw_batch(idea_vectors, top_k=top_k)
                    for q_pos, idx in enumerate(query_indices):
                        if q_pos < len(batch_hnsw):
                            hnsw_candidates_by_idx[idx] = batch_hnsw[q_pos]
                except Exception as e:
                    logger.debug(f"Neo4j HNSW batch query in resolve_concepts_batch fallback to cache: {e}")
                if on_progress:
                    on_progress("hnsw_query", len(idea_vectors), len(idea_vectors), f"Queried Neo4j HNSW index ({len(idea_vectors)} ideas)")

        # Build a vectorized snapshot of in-memory cached vectors for fast matrix cosine comparison
        cached_names: List[str] = []
        cached_concepts: List[Concept] = []
        cached_vecs_list: List[np.ndarray] = []
        target_dim: Optional[int] = None
        for cname, (cvec, cobj) in self._cache.items():
            if cvec is not None and cvec.ndim == 1 and len(cvec) > 1:
                if target_dim is None:
                    target_dim = cvec.shape[0]
                if cvec.shape[0] == target_dim:
                    cached_names.append(cname)
                    cached_concepts.append(cobj)
                    cached_vecs_list.append(cvec)
        cached_matrix: Optional[np.ndarray] = (
            np.vstack(cached_vecs_list) if cached_vecs_list else None
        )

        # Pre-pass: identify any probable matches (0.80 <= sim < 0.95) against existing cache/HNSW
        # so we can run LLM disambiguation concurrently across the Ollama pool instead of sequentially!
        if self.extractor is not None:
            pending_disambiguations: Dict[Tuple[str, str], Tuple[Concept, Concept, float]] = {}
            for idx, concept in enumerate(concepts):
                raw_name = concept.name
                norm_name = self._normalize_name(raw_name)
                if norm_name in self._norm_to_canon:
                    continue

                embed_text = concept_to_embed_text.get(raw_name, raw_name)
                concept_vec = missing_text_to_vec.get(embed_text)
                if concept_vec is None and concept.embedding and len(concept.embedding) > 1:
                    arr = np.array(concept.embedding, dtype=np.float32)
                    norm = float(np.linalg.norm(arr))
                    concept_vec = (arr / norm) if norm > 0.0 else arr

                best_sim = -1.0
                best_match: Optional[Concept] = None
                for cand in hnsw_candidates_by_idx.get(idx, []):
                    cand_name = BookParser.repair_mojibake(str(cand.get("name") or "")).strip()
                    cand_score = float(cand.get("score") or 0.0)
                    if not cand_name:
                        continue
                    if cand_name in self._cache:
                        _, existing_c = self._cache[cand_name]
                    else:
                        existing_c = Concept(
                            name=cand_name,
                            brief_description=cand.get("brief_description") or cand.get("summary") or "",
                            detailed_explanation=cand.get("detailed_explanation") or "",
                            summary=cand.get("summary") or cand.get("brief_description") or "",
                            category="Idea",
                        )
                        self._register_in_cache(cand_name, None, existing_c)
                    if cand_score > best_sim:
                        best_sim = cand_score
                        best_match = existing_c

                if concept_vec is not None and self._usearch_index is not None and len(self._usearch_index) > 0:
                    matches = self._usearch_index.search(concept_vec, 1)
                    if len(matches.keys) > 0 and len(matches.distances) > 0:
                        d = float(matches.distances[0])
                        if not np.isnan(d):
                            sim_val = float(max(-1.0, min(1.0, 1.0 - d)))
                            k_idx = int(matches.keys[0])
                            if 0 <= k_idx < len(self._cached_canon_order):
                                cand_canon = self._cached_canon_order[k_idx]
                                if cand_canon in self._cache:
                                    cand_concept = self._cache[cand_canon][1]
                                    if sim_val > best_sim:
                                        best_sim = sim_val
                                        best_match = cand_concept
                elif concept_vec is not None and cached_matrix is not None and concept_vec.shape[0] == cached_matrix.shape[1]:
                    sims = np.dot(cached_matrix, concept_vec)
                    max_i = int(np.argmax(sims))
                    sim_val = float(sims[max_i])
                    if sim_val > best_sim:
                        best_sim = sim_val
                        best_match = cached_concepts[max_i]

                if (
                    best_match is not None
                    and self.similarity_threshold <= best_sim < self.high_similarity_threshold
                ):
                    pair_key = (self._normalize_name(best_match.name), norm_name)
                    if pair_key not in self._disambiguation_cache and pair_key not in pending_disambiguations:
                        pending_disambiguations[pair_key] = (best_match, concept, best_sim)

            if pending_disambiguations:
                total_disambig = len(pending_disambiguations)
                pool_obj = getattr(self.extractor, "pool", None)
                max_workers = max(1, min(total_disambig, len(getattr(pool_obj, "nodes", [1])) * 2 if pool_obj else 4))
                if on_progress:
                    on_progress(
                        "llm_disambiguation",
                        0,
                        total_disambig,
                        f"Disambiguating {total_disambig} probable idea match(es) via LLM",
                    )
                done_disambig = 0
                with DaemonThreadPoolExecutor(max_workers=max_workers) as ex:
                    fut_to_key = {
                        ex.submit(self._ask_llm_disambiguation, b_match, inc_c, b_sim): p_key
                        for p_key, (b_match, inc_c, b_sim) in pending_disambiguations.items()
                    }
                    for fut in as_completed(fut_to_key):
                        p_key = fut_to_key[fut]
                        try:
                            self._disambiguation_cache[p_key] = fut.result()
                        except Exception as e:
                            logger.debug(f"Parallel LLM disambiguation failed for {p_key}: {e}")
                            self._disambiguation_cache[p_key] = ConceptDeduplicationDecision(is_same_concept=False)
                        done_disambig += 1
                        if on_progress:
                            on_progress(
                                "llm_disambiguation",
                                done_disambig,
                                total_disambig,
                                f"Disambiguated {done_disambig}/{total_disambig} probable idea match(es)",
                            )

        # 3. Resolve each concept in order, combining HNSW candidates and in-memory cache
        resolved_list: List[Concept] = []
        for idx, concept in enumerate(concepts):
            raw_name = concept.name
            norm_name = self._normalize_name(raw_name)

            # 3a. O(1) Exact or normalized name match in cache
            exact_hit: Optional[Concept] = None
            canon_hit = self._norm_to_canon.get(norm_name)
            if canon_hit and canon_hit in self._cache:
                _, exact_hit = self._cache[canon_hit]
            else:
                for canon_name, (_, existing) in self._cache.items():
                    if self._normalize_name(canon_name) == norm_name:
                        self._norm_to_canon[norm_name] = canon_name
                        exact_hit = existing
                        break
            if exact_hit is not None:
                self._merge_into(exact_hit, concept)
                if exact_hit.embedding is None and concept.embedding:
                    exact_hit.embedding = concept.embedding
                resolved_list.append(exact_hit)
                continue

            embed_text = concept_to_embed_text.get(raw_name, raw_name)
            concept_vec = missing_text_to_vec.get(embed_text)
            if concept_vec is None and concept.embedding and len(concept.embedding) > 1:
                arr = np.array(concept.embedding, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                concept_vec = (arr / norm) if norm > 0.0 else arr

            best_sim = -1.0
            best_match: Optional[Concept] = None
            best_canon_name: Optional[str] = None

            # Check top-k candidates returned from Neo4j HNSW 'idea_idx'
            hnsw_cands = hnsw_candidates_by_idx.get(idx, [])
            for cand in hnsw_cands:
                cand_name = BookParser.repair_mojibake(str(cand.get("name") or "")).strip()
                cand_score = float(cand.get("score") or 0.0)
                if not cand_name:
                    continue
                if cand_name in self._cache:
                    _, existing_c = self._cache[cand_name]
                else:
                    existing_c = Concept(
                        name=cand_name,
                        brief_description=cand.get("brief_description") or cand.get("summary") or "",
                        detailed_explanation=cand.get("detailed_explanation") or "",
                        summary=cand.get("summary") or cand.get("brief_description") or "",
                        category="Idea",
                    )
                    self._register_in_cache(cand_name, None, existing_c)
                if cand_score > best_sim:
                    best_sim = cand_score
                    best_match = existing_c
                    best_canon_name = cand_name

            # Also check in-memory _cache vectors via fast usearch HNSW index
            if concept_vec is not None and self._cache:
                if self._usearch_index is not None and len(self._usearch_index) > 0:
                    matches = self._usearch_index.search(concept_vec, 1)
                    if len(matches.keys) > 0 and len(matches.distances) > 0:
                        d = float(matches.distances[0])
                        if not np.isnan(d):
                            sim_val = float(max(-1.0, min(1.0, 1.0 - d)))
                            k_idx = int(matches.keys[0])
                            if 0 <= k_idx < len(self._cached_canon_order):
                                cand_canon = self._cached_canon_order[k_idx]
                                if cand_canon in self._cache:
                                    cand_concept = self._cache[cand_canon][1]
                                    if sim_val > best_sim:
                                        best_sim = sim_val
                                        best_match = cand_concept
                                        best_canon_name = cand_canon
                elif cached_matrix is not None and concept_vec.shape[0] == cached_matrix.shape[1]:
                    sims = np.dot(cached_matrix, concept_vec)
                    max_i = int(np.argmax(sims))
                    sim_val = float(sims[max_i])
                    if sim_val > best_sim:
                        best_sim = sim_val
                        best_match = cached_concepts[max_i]
                        best_canon_name = cached_names[max_i]
                # Also check any newly added concepts in _cache during this loop beyond cached_matrix if usearch is not present
                if self._usearch_index is None and len(self._cache) > len(cached_names):
                    for canon_name, (cached_vec, existing_concept) in list(self._cache.items())[len(cached_names) :]:
                        if cached_vec is not None and cached_vec.shape == concept_vec.shape:
                            sim = float(np.dot(concept_vec, cached_vec))
                            if sim > best_sim:
                                best_sim = sim
                                best_match = existing_concept
                                best_canon_name = canon_name

            # Case A: High Confidence (>= 0.95) -> Direct merge
            if best_match is not None and best_sim >= self.high_similarity_threshold:
                self._merge_into(best_match, concept)
                if best_match.embedding is None and concept.embedding:
                    best_match.embedding = concept.embedding
                resolved_list.append(best_match)
                continue

            # Case B: Probable match (0.80 <= similarity < 0.95) -> LLM disambiguation (reusing parallel pre-pass cache)
            if best_match is not None and best_sim >= self.similarity_threshold:
                if self.extractor is not None:
                    pair_key = (self._normalize_name(best_match.name), norm_name)
                    decision = self._disambiguation_cache.get(pair_key)
                    if decision is None:
                        decision = self._ask_llm_disambiguation(best_match, concept, best_sim)
                        self._disambiguation_cache[pair_key] = decision
                    if decision.is_same_concept:
                        self._merge_with_llm_decision(best_match, concept, decision, best_canon_name)
                        if best_match.embedding is None and concept.embedding:
                            best_match.embedding = concept.embedding
                        resolved_list.append(best_match)
                        continue
                else:
                    self._merge_into(best_match, concept)
                    if best_match.embedding is None and concept.embedding:
                        best_match.embedding = concept.embedding
                    resolved_list.append(best_match)
                    continue

            # Case C: Register as new canonical concept
            self._register_in_cache(raw_name, concept_vec, concept)
            resolved_list.append(concept)

        return resolved_list

    def _ask_llm_disambiguation(
        self,
        canonical: Concept,
        incoming: Concept,
        similarity: float,
    ) -> ConceptDeduplicationDecision:
        """Ask LLM to determine if two concepts are similar and synthesize unified descriptions."""
        system_instruction = (
            "You are an expert knowledge graph ontologist and entity resolution specialist.\n"
            "Your task is to determine whether two concept candidates extracted from literature "
            "refer to the exact same underlying concept, entity, strategic principle, or motif, "
            "or whether they represent distinct concepts that must remain separate in the knowledge graph.\n\n"
            "Guidelines:\n"
            "1. Same Concept Criteria:\n"
            "   - They refer to the identical core idea, tactic, theme, or phenomenon, even if phrased slightly differently "
            "or originating from different narrative contexts.\n"
            "   - E.g. 'Tactical Ambush' and 'Surprise Ambush Attack' -> SAME.\n"
            "   - E.g. 'Radioactive Fallout Shelter' and 'Underground Nuclear Bunker' -> SAME.\n"
            "2. Distinct Concepts Criteria:\n"
            "   - They describe different actions, opposite strategies, or unrelated phenomena, even if they share a general domain.\n"
            "   - E.g. 'Tactical Retreat' vs 'Ambush Defense' -> DIFFERENT (opposing tactical maneuvers).\n"
            "   - E.g. 'Wall Graffiti' vs 'Propaganda Slogan' -> DIFFERENT.\n"
            "3. If they are the SAME concept (is_same_concept: true):\n"
            "   - Synthesize the most accurate canonical capitalized name ('canonical_name', 2-4 words).\n"
            "   - Synthesize a clear, unified 1-2 sentence definition ('brief_description').\n"
            "   - Synthesize a comprehensive explanation ('detailed_explanation') merging mechanisms, nuances, and context from both.\n"
            "4. If they are DIFFERENT concepts (is_same_concept: false):\n"
            "   - Do not synthesize names or descriptions; explain why in 'reasoning'."
        )

        user_content = (
            f"Concept 1 (Existing Canonical in Knowledge Graph):\n"
            f"- Name: {canonical.name}\n"
            f"- Category: {getattr(canonical, 'category', 'General')}\n"
            f"- Brief Description: {getattr(canonical, 'brief_description', '') or getattr(canonical, 'summary', '')}\n"
            f"- Detailed Explanation: {getattr(canonical, 'detailed_explanation', '')}\n\n"
            f"Concept 2 (Incoming Candidate):\n"
            f"- Name: {incoming.name}\n"
            f"- Category: {getattr(incoming, 'category', 'General')}\n"
            f"- Brief Description: {getattr(incoming, 'brief_description', '') or getattr(incoming, 'summary', '')}\n"
            f"- Detailed Explanation: {getattr(incoming, 'detailed_explanation', '')}\n\n"
            f"Cosine Similarity Score between names: {similarity:.3f}\n\n"
            f"Are these two concepts referring to the same underlying concept? "
            f"If yes, synthesize the canonical name, brief description, and detailed explanation."
        )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_content),
        ]

        try:
            if hasattr(self.extractor, "_execute_structured_invoke"):
                res = self.extractor._execute_structured_invoke(
                    ConceptDeduplicationDecision,
                    messages,
                    task_type="deduplication",
                )
                if isinstance(res, ConceptDeduplicationDecision):
                    return res
                return ConceptDeduplicationDecision(**dict(res))
            elif hasattr(self.extractor, "invoke"):
                res = self.extractor.invoke(messages)
                if isinstance(res, ConceptDeduplicationDecision):
                    return res
                return ConceptDeduplicationDecision(**dict(res))
        except Exception as e:
            logger.warning(f"LLM deduplication disambiguation failed: {e}. Defaulting to distinct concepts.")
            return ConceptDeduplicationDecision(
                is_same_concept=False,
                reasoning=f"LLM invocation failed: {e}",
            )

        return ConceptDeduplicationDecision(is_same_concept=False)

    def _merge_with_llm_decision(
        self,
        canonical: Concept,
        incoming: Concept,
        decision: ConceptDeduplicationDecision,
        canon_cache_key: Optional[str] = None,
    ) -> None:
        """Merge incoming concept using LLM-synthesized name and descriptions."""
        # 1. Update related concepts
        for rel in incoming.related_concepts:
            if rel not in canonical.related_concepts and rel != canonical.name:
                canonical.related_concepts.append(rel)

        # 2. Update descriptions using synthesized LLM output
        if decision.brief_description and decision.brief_description.strip():
            canonical.brief_description = decision.brief_description.strip()
            canonical.summary = decision.brief_description.strip()
        elif incoming.brief_description and len(incoming.brief_description) > len(getattr(canonical, "brief_description", "")):
            canonical.brief_description = incoming.brief_description

        if decision.detailed_explanation and decision.detailed_explanation.strip():
            canonical.detailed_explanation = decision.detailed_explanation.strip()
        elif incoming.detailed_explanation and len(incoming.detailed_explanation) > len(getattr(canonical, "detailed_explanation", "")):
            canonical.detailed_explanation = incoming.detailed_explanation

        # 3. Retain highest significance weight observed
        canonical.weight = max(getattr(canonical, "weight", 5), getattr(incoming, "weight", 5))

        # 4. Canonical name update if synthesized
        if decision.canonical_name and decision.canonical_name.strip():
            new_name = decision.canonical_name.strip()
            if new_name != canonical.name:
                old_name = canonical.name
                canonical.name = new_name
                if canon_cache_key and canon_cache_key in self._cache:
                    cached_vec, _ = self._cache[canon_cache_key]
                    self._register_in_cache(new_name, cached_vec, canonical)
                    self._register_in_cache(old_name, cached_vec, canonical)

    def _merge_into(self, canonical: Concept, incoming: Concept) -> None:
        """Merge incoming concept's related concepts and details into canonical node (heuristic fallback)."""
        for rel in incoming.related_concepts:
            if rel not in canonical.related_concepts and rel != canonical.name:
                canonical.related_concepts.append(rel)

        # Merge descriptions
        if incoming.detailed_explanation and len(incoming.detailed_explanation) > len(getattr(canonical, "detailed_explanation", "")):
            canonical.detailed_explanation = incoming.detailed_explanation
        if incoming.brief_description and len(incoming.brief_description) > len(getattr(canonical, "brief_description", "")):
            canonical.brief_description = incoming.brief_description

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

    def is_same_concept(
        self,
        c1: Any,
        c2: Any,
    ) -> Tuple[bool, float, str, Optional[str], Optional[str]]:
        """
        Evaluate whether two concepts refer to the same underlying entity/idea.
        Accepts either Concept objects or string concept names.
        Returns:
            (is_same, similarity_score, match_method, canonical_name, reasoning)
            where match_method is in ("exact", "high_vector", "llm_disambiguated", "none").
        """
        raw_name1 = c1 if isinstance(c1, str) else getattr(c1, "name", str(c1))
        raw_name2 = c2 if isinstance(c2, str) else getattr(c2, "name", str(c2))

        name1 = BookParser.repair_mojibake(raw_name1.strip())
        name2 = BookParser.repair_mojibake(raw_name2.strip())
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)

        # 1. Exact or case-insensitive string match
        if norm1 == norm2:
            return True, 1.0, "exact", name1, "Exact normalized name match"

        # 2. Embedding-based semantic cosine similarity
        sim = 0.0
        vec1 = self._embed(name1)
        vec2 = self._embed(name2)
        if vec1 is not None and vec2 is not None:
            sim = float(np.dot(vec1, vec2))

            # Case A: High confidence vector match (>= 0.95)
            if sim >= self.high_similarity_threshold:
                return True, sim, "high_vector", name1, f"High confidence vector similarity ({sim:.3f})"

            # Case B: Probable match (0.80 <= similarity < 0.95) -> Disambiguate with LLM
            if sim >= self.similarity_threshold:
                if self.extractor is not None:
                    concept_obj1 = c1 if isinstance(c1, Concept) else Concept(name=name1)
                    concept_obj2 = c2 if isinstance(c2, Concept) else Concept(name=name2)
                    decision = self._ask_llm_disambiguation(concept_obj1, concept_obj2, sim)
                    if decision.is_same_concept:
                        return True, sim, "llm_disambiguated", decision.canonical_name or name1, decision.reasoning
                    else:
                        return False, sim, "none", None, decision.reasoning
        # 3. Fallback when embeddings are unavailable but an LLM extractor is configured
        if (vec1 is None or vec2 is None) and self.extractor is not None:
            t1 = set(norm1.split())
            t2 = set(norm2.split())
            if (t1 & t2) or (norm1 in norm2) or (norm2 in norm1):
                concept_obj1 = c1 if isinstance(c1, Concept) else Concept(name=name1)
                concept_obj2 = c2 if isinstance(c2, Concept) else Concept(name=name2)
                try:
                    decision = self._ask_llm_disambiguation(concept_obj1, concept_obj2, 0.85)
                    if decision.is_same_concept:
                        return True, 0.85, "llm_disambiguated", decision.canonical_name or name1, decision.reasoning
                    return False, 0.0, "none", None, decision.reasoning
                except Exception as e:
                    logger.debug(f"Fallback LLM disambiguation failed: {e}")

        return False, sim, "none", None, "Concepts differ below similarity threshold"

    def get_canonical_concepts(self) -> List[Concept]:
        """Return all unique canonical concepts registered so far."""
        return [c for _, c in self._cache.values()]


# Backward-compatibility alias
ConceptDeduplicator = EntityDeduplicator
