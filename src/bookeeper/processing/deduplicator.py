"""
Embedding-based entity resolution and concept deduplication using OllamaEmbeddings
and two-tier LLM-assisted verification and synthesis.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from bookeeper.calibre.parser import BookParser
from bookeeper.config import Settings
from bookeeper.processing.extractor import Concept

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

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        high_similarity_threshold: Optional[float] = None,
        extractor: Optional[Any] = None,
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
            high_similarity_threshold=high_s_thresh,
            extractor=extractor,
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Strip punctuation and redundant whitespace for basic equality matching."""
        cleaned = re.sub(r"[^\w\s-]", "", name).strip().lower()
        return re.sub(r"\s+", " ", cleaned)

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

        # 1. Exact or case-insensitive string match (highest confidence)
        for canon_name, (_, existing) in self._cache.items():
            if self._normalize_name(canon_name) == norm_name:
                self._merge_into(existing, concept)
                return existing

        # 2. Embedding-based semantic cosine similarity
        concept_vec = self._embed(raw_name)
        if concept_vec is not None and self._cache:
            best_sim = -1.0
            best_match: Optional[Concept] = None
            best_canon_name: Optional[str] = None

            for canon_name, (cached_vec, existing_concept) in self._cache.items():
                if cached_vec is not None:
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
        self._cache[raw_name] = (concept_vec, concept)
        return concept

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
                    self._cache[new_name] = (cached_vec, canonical)
                    self._cache[old_name] = (cached_vec, canonical)

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
                else:
                    return True, sim, "high_vector", name1, f"Vector similarity match ({sim:.3f}, no LLM)"

        return False, sim, "none", None, "Concepts differ below similarity threshold"

    def get_canonical_concepts(self) -> List[Concept]:
        """Return all unique canonical concepts registered so far."""
        return [c for _, c in self._cache.values()]


# Backward-compatibility alias
ConceptDeduplicator = EntityDeduplicator
