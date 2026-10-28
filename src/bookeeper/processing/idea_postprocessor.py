"""
Two-pass idea aggregation and synthesis postprocessor.

Implements second-pass aggregation over knowledge graph ideas:
1. Deterministic normalization & grouping + cosine embedding similarity clustering (>= 0.92).
2. Source provenance aggregation (top 3-5 quotes) and chronological linearization.
3. Branching: Single-occurrence clusters (N=1) bypass LLM; multi-occurrence (N>=2) synthesized by LLM.
4. Graph ingestion: upsert canonical entity node, provenance edges (:MENTIONED_IN, :SUPPORTED_BY),
   and relational edges (:RELATES_TO).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from bookeeper.calibre.parser import BookParser, format_idea_month
from bookeeper.config import Settings
from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings, OllamaPool, format_idea_for_embedding

if TYPE_CHECKING:
    from bookeeper.graph.store import ConceptGraphStore

logger = logging.getLogger(__name__)


# ==============================================================================
# Helper & Normalization Functions
# ==============================================================================


def _format_eta(seconds: Optional[float]) -> str:
    """Format remaining seconds as (min, sec), e.g. '2m 15s' or '0m 45s'."""
    if seconds is None:
        return "--m --s"
    if seconds <= 0:
        return "0m 00s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def normalize_idea_name(name: str) -> str:
    """
    Deterministically normalize an idea/entity name:
    - Case folding (lowercase)
    - Stripping English determiners ('the', 'a', 'an')
    - Trimming punctuation and special characters
    - Collapsing consecutive whitespaces
    """
    if not name:
        return ""
    # Mojibake cleanup
    s = BookParser.repair_mojibake(name).strip().lower()
    # Strip leading determiners
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    # Strip trailing determiners or punctuation
    s = re.sub(r"[^\w\s-]", "", s)
    # Normalize whitespaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_canonical_name(name: str) -> str:
    """
    Clean name for canonical entity representation:
    - Strips leading determiners ('the', 'a', 'an')
    - Repairs mojibake and trims punctuation
    """
    if not name:
        return ""
    s = BookParser.repair_mojibake(name).strip()
    stripped = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.IGNORECASE).strip()
    return stripped if stripped else s


def prune_near_duplicate_quotes(quotes: List[str], max_quotes: int = 5) -> List[str]:
    """
    Retain top 3-5 most representative supporting quotes, pruning near-duplicates.
    """
    clean_quotes = []
    for q in quotes:
        q_clean = BookParser.repair_mojibake(q).strip()
        if len(q_clean) >= 15 and q_clean not in clean_quotes:
            clean_quotes.append(q_clean)

    if not clean_quotes:
        return []

    # Sort descending by length as a proxy for explanatory richness
    sorted_quotes = sorted(clean_quotes, key=lambda x: len(x), reverse=True)
    retained: List[str] = []

    for cand in sorted_quotes:
        cand_norm = re.sub(r"\W+", " ", cand.lower()).strip()
        cand_tokens = set(cand_norm.split())

        is_dup = False
        for existing in retained:
            ext_norm = re.sub(r"\W+", " ", existing.lower()).strip()
            # Direct substring match
            if cand_norm in ext_norm or ext_norm in cand_norm:
                is_dup = True
                break
            # Token overlap similarity (Jaccard)
            ext_tokens = set(ext_norm.split())
            if cand_tokens and ext_tokens:
                overlap = len(cand_tokens & ext_tokens) / float(min(len(cand_tokens), len(ext_tokens)))
                if overlap > 0.85:
                    is_dup = True
                    break

        if not is_dup:
            retained.append(cand)
        if len(retained) >= max_quotes:
            break

    return retained


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class IdeaObservation:
    """A localized mention/observation of an idea within a specific chunk/section."""

    concept_name: str
    chunk_id: Optional[str] = None
    book_id: int = 0
    chapter_idx: int = 0
    chunk_idx: int = 0
    position_str: str = ""
    local_brief: str = ""
    local_detailed: str = ""
    quote: str = ""
    weight: int = 5
    category: str = "Idea"
    related_concepts: List[str] = field(default_factory=list)
    month: Optional[str] = None


@dataclass
class IdeaCluster:
    """A clustered entity grouping raw observations across chunks."""

    canonical_name: str
    normalized_name: str
    aliases: Set[str] = field(default_factory=set)
    observations: List[IdeaObservation] = field(default_factory=list)
    chunk_ids: Set[str] = field(default_factory=set)
    category: str = "Idea"
    max_weight: int = 5
    embedding: Optional[np.ndarray] = None
    first_occurrence_month: Optional[str] = None
    existing_node_ids: Set[str] = field(default_factory=set)
    already_refined: bool = False
    already_postprocessed: bool = False
    was_merged: bool = False
    refined_brief: str = ""
    refined_detailed: str = ""
    refined_related_entities: List[str] = field(default_factory=list)

    def add_observation(self, obs: IdeaObservation) -> None:
        self.observations.append(obs)
        if obs.concept_name:
            self.aliases.add(obs.concept_name)
        if obs.chunk_id:
            self.chunk_ids.add(obs.chunk_id)
        if obs.weight > self.max_weight:
            self.max_weight = obs.weight
        if obs.category and obs.category not in ("Idea", "Concept", ""):
            self.category = obs.category
        if obs.month:
            if not self.first_occurrence_month or obs.month < self.first_occurrence_month:
                self.first_occurrence_month = obs.month

    def get_first_occurrence_month(self) -> Optional[str]:
        """Find the earliest occurrence month across all observations and cluster metadata (earlier date wins)."""
        valid = [o.month for o in self.observations if o.month]
        if self.first_occurrence_month:
            valid.append(self.first_occurrence_month)
        return min(valid) if valid else None

    @property
    def total_occurrences(self) -> int:
        return max(len(self.observations), len(self.chunk_ids))

    def get_representative_brief(self) -> str:
        """Find the longest/most informative local brief description."""
        candidates = [o.local_brief for o in self.observations if o.local_brief]
        if candidates:
            return max(candidates, key=len)
        detailed = [o.local_detailed for o in self.observations if o.local_detailed]
        if detailed:
            return max(detailed, key=len)
        return self.canonical_name

    def get_representative_detailed(self) -> str:
        """Find the longest/most informative local detailed description."""
        detailed = [o.local_detailed for o in self.observations if o.local_detailed]
        if detailed:
            return max(detailed, key=len)
        candidates = [o.local_brief for o in self.observations if o.local_brief]
        if candidates:
            return max(candidates, key=len)
        return self.canonical_name

    def get_linearized_observations(self) -> List[IdeaObservation]:
        """Order observations chronologically by (book_id, chapter_idx, chunk_idx)."""
        return sorted(
            self.observations,
            key=lambda o: (o.book_id, o.chapter_idx, o.chunk_idx, o.chunk_id or ""),
        )

    def get_representative_quotes(self, max_quotes: int = 5) -> List[str]:
        raw_quotes = [o.quote for o in self.observations if o.quote]
        return prune_near_duplicate_quotes(raw_quotes, max_quotes=max_quotes)


class ConsolidatedIdeaOutput(BaseModel):
    """Structured LLM synthesis for a consolidated idea."""

    brief_description: str = Field(
        description="Cohesive 1-2 sentence description capturing the macro-identity."
    )
    detailed_explanation: str = Field(
        description="Synthesized comprehensive explanation integrating distinct events, cause-and-effect chains, and conditions across all chunks without dropping earlier facts."
    )
    related_entities: List[str] = Field(
        default_factory=list,
        description="Explicit related entities or concepts mentioned across the pooled observations.",
    )
    category: Optional[str] = Field(
        default=None,
        description="Domain category classification (e.g. Architectural Pattern, Principle, Mechanism).",
    )
    weight: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="Significance and depth weight from 0 to 10.",
    )


class IdeaPostProcessorReport(BaseModel):
    """Execution summary report for the dual-pass idea postprocessor."""

    total_input_concepts: int = 0
    total_clusters: int = 0
    single_occurrence_clusters_count: int = 0
    multi_occurrence_clusters_count: int = 0
    synthesized_count: int = 0
    resumed_count: int = 0
    merged_clusters_count: int = 0
    edges_added_count: int = 0
    duration_seconds: float = 0.0
    synthesized_ideas: List[str] = Field(default_factory=list)
    bypassed_ideas: List[str] = Field(default_factory=list)


# ==============================================================================
# Persistent Checkpoint State Management
# ==============================================================================


class IdeaPostProcessorState:
    """
    Persistent checkpoint state manager for two-pass idea postprocessing.
    Saves progress every N ideas (default: 100) and enables auto-continue on resume.
    """

    def __init__(self, state_path: Path | str):
        self.state_path = Path(state_path).expanduser().resolve()
        self.version: int = 1
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.total_clusters: int = 0
        self.model_name: str = ""
        self.similarity_threshold: float = 0.92
        self.min_occurrences: int = 2
        self.processed_ideas: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> bool:
        """Load state from disk. Returns True if existing valid state was loaded."""
        if not self.state_path.is_file():
            return False
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.version = data.get("version", 1)
                self.created_at = data.get("created_at", self.created_at)
                self.updated_at = data.get("updated_at", time.time())
                self.total_clusters = data.get("total_clusters", 0)
                self.model_name = data.get("model_name", "")
                self.similarity_threshold = data.get("similarity_threshold", 0.92)
                self.min_occurrences = data.get("min_occurrences", 2)
                self.processed_ideas = data.get("processed_ideas", {})
                return True
        except Exception as e:
            logger.warning(f"Could not load postprocessor state from {self.state_path} ({e}); starting fresh.")
        return False

    def save(self) -> None:
        """Atomically persist state to disk using a temporary file rename."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.updated_at = time.time()
            data = {
                "version": self.version,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "total_clusters": self.total_clusters,
                "model_name": self.model_name,
                "similarity_threshold": self.similarity_threshold,
                "min_occurrences": self.min_occurrences,
                "completed_count": len(self.processed_ideas),
                "processed_ideas": self.processed_ideas,
            }
            dir_name = self.state_path.parent
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                json.dump(data, tf, indent=2, ensure_ascii=False)
                temp_name = tf.name
            os.replace(temp_name, self.state_path)
            logger.debug(f"Saved postprocessor state ({len(self.processed_ideas)} ideas) to {self.state_path}")
        except Exception as e:
            logger.warning(f"Failed to persist postprocessor state to {self.state_path}: {e}")

    def is_processed(self, canonical_name: str) -> bool:
        """Check if an idea has already been processed in this state."""
        return canonical_name in self.processed_ideas

    def get_status(self, canonical_name: str) -> Optional[str]:
        """Retrieve processing status ('synthesized' or 'bypassed') for an idea if present."""
        rec = self.processed_ideas.get(canonical_name)
        if not rec or not isinstance(rec, dict):
            return None
        return rec.get("status")

    def get_output(self, canonical_name: str) -> Optional[ConsolidatedIdeaOutput]:
        """Retrieve cached ConsolidatedIdeaOutput for an idea if present."""
        rec = self.processed_ideas.get(canonical_name)
        if not rec or "output" not in rec:
            return None
        try:
            return ConsolidatedIdeaOutput.model_validate(rec["output"])
        except Exception as e:
            logger.warning(f"Failed to validate cached ConsolidatedIdeaOutput for '{canonical_name}': {e}")
            return None

    def get_synthesized_output_for_cluster(self, cluster: IdeaCluster) -> Optional[ConsolidatedIdeaOutput]:
        """
        Retrieve cached ConsolidatedIdeaOutput if the cluster's canonical name or any of its aliases
        has already been refined (status == 'synthesized'). Ensures refinement happens at most once.
        """
        candidates = [cluster.canonical_name] + sorted(
            a for a in cluster.aliases if a != cluster.canonical_name
        )
        for name in candidates:
            rec = self.processed_ideas.get(name)
            if rec and isinstance(rec, dict) and rec.get("status") == "synthesized" and "output" in rec:
                try:
                    res = ConsolidatedIdeaOutput.model_validate(rec["output"])
                    if "embedding" in rec and (cluster.embedding is None or len(cluster.embedding) <= 1):
                        cluster.embedding = np.array(rec["embedding"], dtype=np.float32)
                    return res
                except Exception as e:
                    logger.warning(f"Failed to validate cached synthesized output for '{name}': {e}")
        return None

    def record_processed(
        self,
        cluster: IdeaCluster,
        output: ConsolidatedIdeaOutput,
        status: str = "synthesized",
    ) -> None:
        """Record a processed idea cluster and its consolidated output."""
        rec: Dict[str, Any] = {
            "canonical_name": cluster.canonical_name,
            "normalized_name": cluster.normalized_name,
            "total_occurrences": cluster.total_occurrences,
            "status": status,
            "output": output.model_dump(),
            "timestamp": time.time(),
        }
        if cluster.embedding is not None and len(cluster.embedding) > 1:
            rec["embedding"] = [round(float(x), 6) for x in cluster.embedding]
        self.processed_ideas[cluster.canonical_name] = rec

    def clear(self) -> None:
        """Clear all in-memory and on-disk state."""
        self.processed_ideas.clear()
        if self.state_path.is_file():
            try:
                self.state_path.unlink()
            except Exception as e:
                logger.warning(f"Could not remove state file {self.state_path}: {e}")


# ==============================================================================
# IdeaPostProcessor Engine
# ==============================================================================


class IdeaPostProcessor:
    """
    Second-pass aggregation and synthesis engine for Knowledge Graph ideas.
    """

    def __init__(
        self,
        pool: Optional[OllamaPool] = None,
        embeddings: Optional[Embeddings] = None,
        model_name: str = "llama3.1:8b",
        similarity_threshold: float = 0.92,
        max_quotes: int = 5,
        min_occurrences: int = 2,
        temperature: float = 0.1,
        timeout: int = 60,
        retries: int = 1,
        state_file: Optional[Path | str] = None,
        checkpoint_interval: int = 100,
        resume: bool = True,
    ):
        self.pool = pool
        self.embeddings = embeddings
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_quotes = max_quotes
        self.min_occurrences = min_occurrences
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.state_file = Path(state_file).expanduser().resolve() if state_file else None
        self.checkpoint_interval = checkpoint_interval
        self.resume = resume

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        model_name: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        pool: Optional[OllamaPool] = None,
        embeddings: Optional[Embeddings] = None,
        state_file: Optional[Path | str] = None,
        checkpoint_interval: Optional[int] = None,
        resume: Optional[bool] = None,
    ) -> "IdeaPostProcessor":
        """Instantiate IdeaPostProcessor using application settings."""
        eff_model = model_name or settings.resolved_dual_pass_model
        eff_sim = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.dual_pass.similarity_threshold
        )
        max_q = settings.dual_pass.max_quotes
        min_occ = settings.dual_pass.min_occurrences
        temp = settings.dual_pass.temperature
        t_out = getattr(settings, "request_timeout", 60)
        retries = getattr(settings, "max_retries", 1)
        eff_state_file = state_file or getattr(settings, "resolved_dual_pass_state_file", None)
        eff_interval = (
            checkpoint_interval
            if checkpoint_interval is not None
            else getattr(settings.dual_pass, "checkpoint_interval", 100)
        )
        eff_resume = resume if resume is not None else True

        # Initialize pool if not provided
        if pool is None:
            servers = getattr(settings, "resolved_llm_servers", None) or settings.resolved_ollama_servers
            cooldown = getattr(settings, "resolved_analysis_cooldown_seconds", getattr(settings, "failover_cooldown_seconds", 600))
            pool = OllamaPool(
                servers=servers,
                cooldown_seconds=cooldown,
                max_tasks_per_server=1,
            )

        # Initialize embeddings if not provided
        if embeddings is None:
            try:
                embeddings = FailoverOllamaEmbeddings.from_settings(settings, model=settings.embedding_model)
            except Exception as e:
                logger.warning(f"Could not initialize embeddings for IdeaPostProcessor: {e}")
                embeddings = None

        return cls(
            pool=pool,
            embeddings=embeddings,
            model_name=eff_model,
            similarity_threshold=eff_sim,
            max_quotes=max_q,
            min_occurrences=min_occ,
            temperature=temp,
            timeout=t_out,
            retries=retries,
            state_file=eff_state_file,
            checkpoint_interval=eff_interval,
            resume=eff_resume,
        )

    # --------------------------------------------------------------------------
    # Step 1: Entity Grouping & Clustering (Deterministic Pass + Similarity)
    # --------------------------------------------------------------------------

    def collect_observations_from_graph(
        self,
        store: ConceptGraphStore,
        console: Optional[Console] = None,
    ) -> Dict[str, IdeaCluster]:
        """
        Step 1a: Collect all idea objects from the graph, normalize names, and group
        ideas sharing an identical normalized name into the same cluster.
        """
        status = None
        if console is not None:
            status = console.status(
                "[bold cyan]Scanning Knowledge Graph & normalizing idea mentions...[/bold cyan]",
                spinner="dots",
            )
            status.start()

        clusters_by_norm: Dict[str, IdeaCluster] = {}
        try:
            # 1. Inspect all Concept nodes
            for node_id, attrs in store.graph.nodes(data=True):
                if attrs.get("type") != "Concept" and not attrs.get("is_idea"):
                    continue

                raw_name = attrs.get("name") or node_id.replace("concept:", "")
                norm_name = normalize_idea_name(raw_name)
                if not norm_name:
                    continue

                brief = attrs.get("brief_description") or attrs.get("summary") or ""
                detailed = attrs.get("detailed_explanation") or attrs.get("summary") or ""
                cat = attrs.get("category") or "Idea"
                weight = int(attrs.get("weight") or 5)
                clean_cand = clean_canonical_name(raw_name)
                node_month = format_idea_month(attrs.get("first_occurrence_month") or attrs.get("month"))

                # 2. Gather edges connecting this concept to Chunks
                # SUPPORTED_BY / MENTIONED_IN edges (Concept -> Chunk)
                chunk_edges = []
                for _, target, edge_data in store.graph.out_edges(node_id, data=True):
                    if edge_data.get("relation") in ("SUPPORTED_BY", "MENTIONED_IN"):
                        chunk_edges.append((target, edge_data))

                # SUPPORTS_IDEA edges (Chunk -> Concept)
                for source, _, edge_data in store.graph.in_edges(node_id, data=True):
                    if edge_data.get("relation") == "SUPPORTS_IDEA":
                        chunk_edges.append((source, edge_data))

                # Skip empty stub nodes created solely as targets of :RELATES_TO edges
                if not chunk_edges and not attrs.get("is_idea") and not (brief or detailed):
                    continue

                if norm_name not in clusters_by_norm:
                    clusters_by_norm[norm_name] = IdeaCluster(
                        canonical_name=clean_cand,
                        normalized_name=norm_name,
                        category=cat,
                        max_weight=weight,
                        first_occurrence_month=node_month,
                        already_postprocessed=bool(attrs.get("postprocessed")),
                    )
                else:
                    clusters_by_norm[norm_name].was_merged = True
                    clusters_by_norm[norm_name].already_postprocessed = False

                cluster = clusters_by_norm[norm_name]
                cluster.existing_node_ids.add(node_id)
                if attrs.get("refined"):
                    cluster.already_refined = True
                    if brief and not cluster.refined_brief:
                        cluster.refined_brief = brief
                    if detailed and not cluster.refined_detailed:
                        cluster.refined_detailed = detailed
                    for rel_ent in attrs.get("related_entities") or []:
                        if rel_ent and rel_ent not in cluster.refined_related_entities:
                            cluster.refined_related_entities.append(rel_ent)
                    if not cluster.refined_related_entities:
                        for _, tgt_nid, edata in store.graph.out_edges(node_id, data=True):
                            if edata.get("relation") == "RELATES_TO":
                                tgt_name = store.graph.nodes.get(tgt_nid, {}).get("name") or tgt_nid.replace("concept:", "")
                                if tgt_name and tgt_name not in cluster.refined_related_entities:
                                    cluster.refined_related_entities.append(tgt_name)

                existing_emb = attrs.get("embedding")
                if cluster.embedding is None and isinstance(existing_emb, (list, np.ndarray)) and len(existing_emb) > 1:
                    arr = np.array(existing_emb, dtype=np.float32)
                    norm = float(np.linalg.norm(arr))
                    cluster.embedding = (arr / norm) if norm > 0.0 else arr
                cluster.aliases.add(raw_name)
                if clean_cand:
                    cluster.aliases.add(clean_cand)
                if node_month:
                    if not cluster.first_occurrence_month or node_month < cluster.first_occurrence_month:
                        cluster.first_occurrence_month = node_month

                # Update canonical name if current has leading determiner or candidate is better capitalized
                if bool(re.match(r"^(?:the|a|an)\s+", cluster.canonical_name, re.IGNORECASE)):
                    cluster.canonical_name = clean_cand
                elif len(clean_cand) > len(cluster.canonical_name) and not clean_cand.islower():
                    cluster.canonical_name = clean_cand

                seen_chunk_ids_for_node: Set[str] = set()

                for chunk_nid, edge_data in chunk_edges:
                    chunk_node = store.graph.nodes.get(chunk_nid, {})
                    chunk_id = str(chunk_node.get("chunk_id") or edge_data.get("chunk_id") or chunk_nid.replace("chunk:", ""))
                    if chunk_id in seen_chunk_ids_for_node:
                        continue
                    seen_chunk_ids_for_node.add(chunk_id)

                    b_id = int(chunk_node.get("book_id") or 0)
                    ch_idx = int(chunk_node.get("chapter_idx") or 0)
                    c_idx = int(chunk_node.get("chunk_idx") or 0)
                    pos = (
                        chunk_node.get("breadcrumb")
                        or f"{chunk_node.get('chapter_title', f'Chapter {ch_idx}')} > {chunk_node.get('section_title', '')}"
                    ).strip(" >")

                    quote = edge_data.get("quote") or ""
                    edge_brief = edge_data.get("brief_description") or brief
                    edge_detailed = edge_data.get("detailed_explanation") or detailed

                    book_node = store.graph.nodes.get(f"book:{b_id}", {})
                    book_pub = book_node.get("publication_date") or book_node.get("pubdate")
                    obs_month = format_idea_month(book_pub) or node_month

                    obs = IdeaObservation(
                        concept_name=raw_name,
                        chunk_id=chunk_id,
                        book_id=b_id,
                        chapter_idx=ch_idx,
                        chunk_idx=c_idx,
                        position_str=pos,
                        local_brief=edge_brief,
                        local_detailed=edge_detailed,
                        quote=quote,
                        weight=weight,
                        category=cat,
                        month=obs_month,
                    )
                    cluster.add_observation(obs)

                # If concept has no chunk edges at all, record a single node observation
                if not seen_chunk_ids_for_node:
                    cluster.add_observation(
                        IdeaObservation(
                            concept_name=raw_name,
                            chunk_id=None,
                            local_brief=brief,
                            local_detailed=detailed,
                            weight=weight,
                            category=cat,
                            month=node_month,
                        )
                    )
        finally:
            if status is not None:
                status.stop()

        if console is not None:
            total_cand = sum(len(c.aliases) for c in clusters_by_norm.values())
            console.print(
                f"[bold green]✓[/bold green] [dim]Collected {len(clusters_by_norm)} normalized idea clusters ({total_cand} mentions)[/dim]"
            )

        return clusters_by_norm

    def cluster_by_semantic_similarity(
        self,
        clusters: List[IdeaCluster],
        console: Optional[Console] = None,
    ) -> List[IdeaCluster]:
        """
        Step 1b: Semantic Similarity Clustering (Within Same Book / Corpus).
        Reuses cached embeddings on IdeaCluster when available and only embeds missing clusters.
        Merges unmerged clusters if cosine similarity >= 0.92.
        """
        if not clusters:
            return clusters

        missing_indices: List[int] = [
            idx for idx, c in enumerate(clusters) if c.embedding is None or len(c.embedding) <= 1
        ]

        if missing_indices and self.embeddings is not None:
            from bookeeper.processing.ollama_pool import (
                format_idea_for_embedding,
                partition_texts_for_embedding,
            )

            texts = [
                format_idea_for_embedding(
                    clusters[idx].canonical_name,
                    brief_description=clusters[idx].get_representative_brief(),
                    detailed_explanation=clusters[idx].get_representative_detailed(),
                    summary=clusters[idx].get_representative_brief(),
                    description=clusters[idx].get_representative_detailed(),
                    short_description=clusters[idx].get_representative_brief(),
                )
                for idx in missing_indices
            ]
            emb_model = getattr(self.embeddings, "model", "embeddings")
            text_batches = partition_texts_for_embedding(
                texts,
                max_batch_chars=12000,
                max_batch_items=32,
            )

            try:
                vectors_raw = []
                if console is not None and len(texts) > 0:
                    emb_progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[bold cyan]Generating Semantic Embeddings:[/bold cyan]"),
                        BarColumn(bar_width=25),
                        TaskProgressColumn(),
                        MofNCompleteColumn(),
                        TextColumn(f"[dim]• {emb_model}[/dim]"),
                        TimeElapsedColumn(),
                        TextColumn("• ETA:"),
                        TimeRemainingColumn(),
                        console=console,
                    )
                    with emb_progress:
                        task = emb_progress.add_task("embed", total=len(texts))
                        if isinstance(self.embeddings, FailoverOllamaEmbeddings):
                            vectors_raw = self.embeddings.embed_documents(
                                texts,
                                batch_size=32,
                                max_batch_chars=12000,
                                on_progress=lambda n: emb_progress.update(task, advance=n),
                            )
                        else:
                            for batch in text_batches:
                                vecs = self.embeddings.embed_documents(batch)
                                vectors_raw.extend(vecs)
                                emb_progress.update(task, advance=len(batch))
                else:
                    if isinstance(self.embeddings, FailoverOllamaEmbeddings):
                        vectors_raw = self.embeddings.embed_documents(
                            texts,
                            batch_size=32,
                            max_batch_chars=12000,
                        )
                    else:
                        for batch in text_batches:
                            vecs = self.embeddings.embed_documents(batch)
                            vectors_raw.extend(vecs)

                if len(vectors_raw) != len(missing_indices):
                    logger.warning(
                        f"Mismatch between missing clusters ({len(missing_indices)}) and embeddings ({len(vectors_raw)}). Skipping semantic merge."
                    )
                    return clusters

                if console is not None:
                    reused_cnt = len(clusters) - len(missing_indices)
                    reused_note = f", {reused_cnt} reused" if reused_cnt > 0 else ""
                    console.print(
                        f"[bold green]✓[/bold green] [dim]Generated semantic embeddings for {len(texts)} clusters ({emb_model}{reused_note})[/dim]"
                    )

                for idx, v in zip(missing_indices, vectors_raw):
                    arr = np.array(v, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    clusters[idx].embedding = (arr / norm) if norm > 0 else arr
            except Exception as e:
                logger.warning(f"Error computing cluster embeddings: {e}. Skipping semantic merge.")
                if console is not None:
                    console.print(f"[yellow]⚠ Semantic embedding skipped: {e}[/yellow]")
                return clusters
        elif missing_indices and self.embeddings is None:
            logger.warning(
                f"Missing embeddings for {len(missing_indices)} clusters and no embeddings client provided. Skipping semantic merge."
            )
            if console is not None:
                console.print(f"[yellow]⚠ Semantic embedding skipped: no embeddings client[/yellow]")
            return clusters
        elif not missing_indices and console is not None:
            console.print(
                f"[bold green]✓[/bold green] [dim]Reused cached semantic embeddings for all {len(clusters)} clusters[/dim]"
            )

        merged: List[IdeaCluster] = []
        consumed_indices: Set[int] = set()

        if console is not None and len(clusters) > 0:
            cluster_progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]Clustering concepts (similarity >= {task.fields[thresh]}):[/bold cyan]"),
                BarColumn(bar_width=25),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("• ETA:"),
                TimeRemainingColumn(),
                console=console,
            )
        else:
            cluster_progress = nullcontext()

        with cluster_progress as c_prog:
            c_task = c_prog.add_task("clustering", total=len(clusters), thresh=f"{self.similarity_threshold:.2f}") if c_prog else None

            for i in range(len(clusters)):
                if c_prog and c_task is not None:
                    c_prog.update(c_task, completed=i)
                if i in consumed_indices:
                    continue

                current = clusters[i]
                consumed_indices.add(i)

                # Search remaining clusters for cosine similarity >= threshold
                for j in range(i + 1, len(clusters)):
                    if j in consumed_indices:
                        continue

                    other = clusters[j]
                    if (
                        current.embedding is not None
                        and other.embedding is not None
                        and current.embedding.shape == other.embedding.shape
                    ):
                        sim = float(np.dot(current.embedding, other.embedding))
                        if sim >= self.similarity_threshold:
                            logger.info(
                                f"Semantic merge ({sim:.3f} >= {self.similarity_threshold}): "
                                f"'{other.canonical_name}' into '{current.canonical_name}'"
                            )
                            # Merge other into current
                            current.was_merged = True
                            current.already_postprocessed = False
                            current.existing_node_ids.update(other.existing_node_ids)
                            if other.already_refined:
                                if not current.already_refined:
                                    current.already_refined = True
                                    current.refined_brief = other.refined_brief
                                    current.refined_detailed = other.refined_detailed
                                    current.refined_related_entities = list(other.refined_related_entities)
                                else:
                                    for r_ent in other.refined_related_entities:
                                        if r_ent not in current.refined_related_entities:
                                            current.refined_related_entities.append(r_ent)
                            current.aliases.update(other.aliases)
                            current.aliases.add(other.canonical_name)
                            for obs in other.observations:
                                current.add_observation(obs)
                            current.chunk_ids.update(other.chunk_ids)
                            current.max_weight = max(current.max_weight, other.max_weight)
                            consumed_indices.add(j)

                merged.append(current)

            if c_prog and c_task is not None:
                c_prog.update(c_task, completed=len(clusters))

        if console is not None:
            merged_diff = len(clusters) - len(merged)
            console.print(
                f"[bold green]✓[/bold green] [dim]Cosine similarity clustering complete: {len(merged)} canonical ideas ({merged_diff} merged)[/dim]"
            )

        return merged

    # --------------------------------------------------------------------------
    # Step 2: Source Provenance & Observation Aggregation
    # --------------------------------------------------------------------------

    def format_consolidation_payload(
        self, cluster: IdeaCluster
    ) -> Tuple[str, List[str]]:
        """
        Format chronologically linearized observations and top representative quotes
        into structured bullet points for the LLM.
        """
        quotes = cluster.get_representative_quotes(max_quotes=self.max_quotes)
        ordered_obs = cluster.get_linearized_observations()

        bullet_points = []
        for idx, obs in enumerate(ordered_obs, 1):
            pos_label = obs.position_str or f"Chunk {obs.chunk_id or idx}"
            desc = obs.local_detailed or obs.local_brief or "Mentioned concept."
            quote_part = f' | Supporting Quote: "{obs.quote}"' if obs.quote else ""
            bullet_points.append(f"- [Obs #{idx} | Pos: {pos_label}]{quote_part}\n  Context: {desc}")

        payload = (
            f"Target Idea / Concept: {cluster.canonical_name}\n"
            f"Known Aliases: {', '.join(sorted(cluster.aliases)) if cluster.aliases else 'None'}\n"
            f"Total Occurrences across Corpus: {cluster.total_occurrences}\n\n"
            f"Chronological Observations:\n"
            + ("\n".join(bullet_points) if bullet_points else "- Single mention.")
            + "\n\n"
            f"Top Source Quotes:\n"
            + ("\n".join(f'* "{q}"' for q in quotes) if quotes else "* None.")
        )
        return payload, quotes

    # --------------------------------------------------------------------------
    # Step 3: LLM Consolidation & Synthesis (Second Pass)
    # --------------------------------------------------------------------------

    def synthesize_cluster(self, cluster: IdeaCluster) -> ConsolidatedIdeaOutput:
        """
        Submit aggregated observation list to consolidation model to synthesize:
        - cohesive brief_description
        - detailed_explanation integrating distinct events and conditions
        - related_entities
        """
        payload, quotes = self.format_consolidation_payload(cluster)

        system_prompt = (
            "You are a Principal Knowledge Graph Architect and Domain Ontologist.\n"
            "Your task is Second-Pass Idea Consolidation: synthesize a unified, definitive definition "
            "for a multi-occurrence idea based strictly on the chronologically ordered observations "
            "and supporting quotes collected across the source material.\n\n"
            "Requirements:\n"
            "1. brief_description: A cohesive 1-2 sentence definition capturing the macro-identity and core essence of the idea.\n"
            "2. detailed_explanation: A comprehensive, synthesized explanation integrating distinct events, "
            "cause-and-effect chains, mechanisms, and conditions across all chunks without dropping earlier facts.\n"
            "3. related_entities: List 2-5 explicit related entities, concepts, or architectural patterns "
            "discussed across the pooled observations.\n"
            "4. category: Domain classification (e.g. Architectural Pattern, Principle, Mechanism, Theory, Data Structure).\n"
            "5. weight: Significance and depth weight from 0 to 10 (higher for novel/specialized architectural concepts)."
        )

        user_prompt = (
            f"Consolidate and synthesize the following idea observations into a unified canonical concept:\n\n"
            f"{payload}\n\n"
            f"Respond with valid JSON conforming to the ConsolidatedIdeaOutput schema."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Invoke LLM via pool / Ollama
        if self.pool and self.pool.nodes:
            def _invoke_fn(url: str) -> ConsolidatedIdeaOutput:
                # 1. Structured ChatOllama invoke
                try:
                    llm = ChatOllama(
                        model=self.model_name,
                        base_url=url,
                        temperature=self.temperature,
                        top_p=0.9,
                        sync_client_kwargs={"timeout": float(self.timeout)},
                        client_kwargs={"timeout": float(self.timeout)},
                    )
                    structured_llm = llm.with_structured_output(ConsolidatedIdeaOutput)
                    res = structured_llm.invoke(messages)
                    if isinstance(res, ConsolidatedIdeaOutput):
                        return res
                    if isinstance(res, dict):
                        return ConsolidatedIdeaOutput.model_validate(res)
                except Exception as exc:
                    logger.debug(f"Structured invoke failed on {url}, falling back to curl JSON: {exc}")

                # 2. Curl fallback with format="json"
                if shutil.which("curl"):
                    msgs_payload = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    body = {
                        "model": self.model_name,
                        "messages": msgs_payload,
                        "format": "json",
                        "options": {"temperature": self.temperature, "top_p": 0.9},
                        "stream": False,
                    }
                    import subprocess

                    cmd = [
                        "curl",
                        "-s",
                        "--connect-timeout",
                        "5",
                        "--max-time",
                        str(self.timeout),
                        "-X",
                        "POST",
                        f"{url.rstrip('/')}/api/chat",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        json.dumps(body),
                    ]
                    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if p.returncode == 0 and p.stdout.strip():
                        resp_json = json.loads(p.stdout)
                        content = resp_json.get("message", {}).get("content", "")
                        return ConsolidatedIdeaOutput.model_validate_json(content)

                raise RuntimeError(f"Could not synthesize idea on {url}")

            try:
                input_chars = len(system_prompt) + len(user_prompt)
                result = self.pool.execute_with_failover(
                    _invoke_fn,
                    retries=self.retries,
                    model_name=self.model_name,
                    capability="llm",
                    input_chars=input_chars,
                    task_type="llm",
                )
                return result
            except Exception as e:
                logger.warning(
                    f"LLM synthesis failed for cluster '{cluster.canonical_name}': {e}. "
                    "Falling back to deterministic observation aggregation."
                )

        # Deterministic fallback when LLM is unavailable / failed
        return self._deterministic_consolidation(cluster, quotes)

    def _deterministic_consolidation(
        self, cluster: IdeaCluster, quotes: List[str]
    ) -> ConsolidatedIdeaOutput:
        """Fallback synthesis when LLM is offline or disabled."""
        brief = cluster.get_representative_brief()
        detailed_parts = []
        for o in cluster.get_linearized_observations():
            t = o.local_detailed or o.local_brief
            if t and t not in detailed_parts:
                detailed_parts.append(t)

        detailed = " ".join(detailed_parts) if detailed_parts else brief
        related = set()
        for o in cluster.observations:
            for r in o.related_concepts:
                if r and r.lower() != cluster.canonical_name.lower():
                    related.add(r)

        return ConsolidatedIdeaOutput(
            brief_description=brief,
            detailed_explanation=detailed,
            related_entities=list(related)[:5],
            category=cluster.category,
            weight=cluster.max_weight,
        )

    # --------------------------------------------------------------------------
    # Step 4: Graph Ingestion & Overwrite
    # --------------------------------------------------------------------------

    def apply_consolidation_to_graph(
        self,
        store: ConceptGraphStore,
        cluster: IdeaCluster,
        output: ConsolidatedIdeaOutput,
    ) -> int:
        """
        Step 4: Upsert canonical entity node, merge aliases, link provenance
        relationships (:SUPPORTED_BY, :SUPPORTS_IDEA), and emit directed :RELATES_TO edges.
        Strictly marks only genuinely created or modified nodes/edges as dirty.
        Returns the number of new edges added.
        """
        canonical_name = cluster.canonical_name
        canonical_nid = f"concept:{canonical_name}"
        edges_added = 0

        brief = BookParser.repair_mojibake(output.brief_description)
        detailed = BookParser.repair_mojibake(output.detailed_explanation)
        cat = BookParser.repair_mojibake(output.category or cluster.category or "Idea")
        weight = output.weight if output.weight is not None else cluster.max_weight
        occurrences = cluster.total_occurrences
        is_refined = bool(occurrences >= self.min_occurrences or cluster.already_refined)
        clean_related = [
            BookParser.repair_mojibake(r).strip()
            for r in (output.related_entities or [])
            if r and BookParser.repair_mojibake(r).strip().lower() != canonical_name.lower()
        ]

        # 1. Upsert Canonical Entity Node
        earliest_month = cluster.get_first_occurrence_month()
        emb_list: Optional[List[float]] = None
        if cluster.embedding is not None and len(cluster.embedding) > 1:
            emb_list = [round(float(x), 6) for x in cluster.embedding]

        # Calculate or update embedding against the full synthesized idea
        # (name, description, short description) after each post-processing update
        if self.embeddings is not None and (emb_list is None or (is_refined and self.pool is not None)):
            full_embed_text = format_idea_for_embedding(
                canonical_name,
                brief_description=brief,
                detailed_explanation=detailed,
                summary=brief or detailed,
                description=detailed,
                short_description=brief,
            )
            try:
                if hasattr(self.embeddings, "embed_query"):
                    raw_vec = self.embeddings.embed_query(full_embed_text)
                elif hasattr(self.embeddings, "embed_documents"):
                    docs_embs = self.embeddings.embed_documents([full_embed_text])
                    raw_vec = docs_embs[0] if docs_embs else None
                else:
                    raw_vec = None

                if raw_vec and len(raw_vec) > 1:
                    arr = np.array(raw_vec, dtype=np.float32)
                    norm = float(np.linalg.norm(arr))
                    unit = (arr / norm) if norm > 0 else arr
                    emb_list = [round(float(x), 6) for x in unit]
                    cluster.embedding = unit
            except Exception as e:
                logger.debug(f"Could not calculate updated post-processing embedding for '{canonical_name}': {e}")

        node_changed = False
        if not store.graph.has_node(canonical_nid):
            node_attrs: Dict[str, Any] = {
                "type": "Concept",
                "tag": "idea",
                "tags": ["idea", cat.lower().replace(" ", "-")],
                "is_idea": True,
                "name": canonical_name,
                "category": cat,
                "brief_description": brief,
                "short_description": brief,
                "detailed_explanation": detailed,
                "description": detailed,
                "summary": brief or detailed,
                "weight": weight,
                "occurrences": occurrences,
                "postprocessed": True,
                "refined": is_refined,
            }
            if clean_related:
                node_attrs["related_entities"] = clean_related
            if earliest_month:
                node_attrs["first_occurrence_month"] = earliest_month
                node_attrs["month"] = earliest_month
            if emb_list is not None:
                node_attrs["embedding"] = emb_list
            store.graph.add_node(canonical_nid, **node_attrs)
            node_changed = True
        else:
            nd = store.graph.nodes[canonical_nid]
            prev_brief = nd.get("brief_description")
            prev_detailed = nd.get("detailed_explanation")
            prev_cat = nd.get("category")
            prev_weight = int(nd.get("weight") or 0)
            prev_occ = int(nd.get("occurrences") or 0)
            prev_month = nd.get("first_occurrence_month") or nd.get("month")
            prev_emb = nd.get("embedding")
            prev_refined = bool(nd.get("refined"))

            new_weight = max(prev_weight, weight)
            new_occ = max(prev_occ, occurrences)

            nd["type"] = "Concept"
            nd["is_idea"] = True
            nd["tag"] = "idea"
            nd["name"] = canonical_name
            nd["brief_description"] = brief
            nd["short_description"] = brief
            nd["detailed_explanation"] = detailed
            nd["description"] = detailed
            nd["summary"] = brief or detailed
            nd["category"] = cat
            nd["weight"] = new_weight
            nd["occurrences"] = new_occ
            nd["postprocessed"] = True
            nd["refined"] = prev_refined or is_refined
            if clean_related:
                nd["related_entities"] = clean_related
            if earliest_month:
                curr_month = nd.get("first_occurrence_month") or nd.get("month")
                if not curr_month or earliest_month < curr_month:
                    nd["first_occurrence_month"] = earliest_month
                    nd["month"] = earliest_month
            if emb_list is not None:
                nd["embedding"] = emb_list
            if "tags" not in nd:
                nd["tags"] = ["idea", cat.lower().replace(" ", "-")]

            if (
                prev_brief != brief
                or prev_detailed != detailed
                or prev_cat != cat
                or prev_weight != new_weight
                or prev_occ != new_occ
                or (nd.get("first_occurrence_month") or nd.get("month")) != prev_month
                or prev_emb != emb_list
                or prev_refined != bool(nd.get("refined"))
            ):
                node_changed = True

        if node_changed:
            store.mark_node_dirty(canonical_nid)

        # 2. Redirect alias nodes into canonical node and remove redundant nodes
        for alias in cluster.aliases:
            if alias == canonical_name:
                continue
            alias_nid = f"concept:{alias}"
            if store.graph.has_node(alias_nid):
                store.mark_node_dirty(canonical_nid)
                alias_month = store.graph.nodes[alias_nid].get("first_occurrence_month") or store.graph.nodes[alias_nid].get("month")
                if alias_month:
                    curr_m = store.graph.nodes[canonical_nid].get("first_occurrence_month") or store.graph.nodes[canonical_nid].get("month")
                    if not curr_m or alias_month < curr_m:
                        store.graph.nodes[canonical_nid]["first_occurrence_month"] = alias_month
                        store.graph.nodes[canonical_nid]["month"] = alias_month
                # Redirect in-edges to canonical
                in_edges = list(store.graph.in_edges(alias_nid, data=True))
                for src, _, edata in in_edges:
                    if src != canonical_nid:
                        had_edge = store.graph.has_edge(src, canonical_nid)
                        store.graph.add_edge(src, canonical_nid, **edata)
                        store.mark_edge_dirty(src, canonical_nid)
                        if not had_edge:
                            edges_added += 1
                # Redirect out-edges to canonical
                out_edges = list(store.graph.out_edges(alias_nid, data=True))
                for _, tgt, edata in out_edges:
                    if tgt != canonical_nid:
                        had_edge = store.graph.has_edge(canonical_nid, tgt)
                        store.graph.add_edge(canonical_nid, tgt, **edata)
                        store.mark_edge_dirty(canonical_nid, tgt)
                        if not had_edge:
                            edges_added += 1
                # Remove alias node
                store.graph.remove_node(alias_nid)
                store.mark_node_removed(alias_nid)

        # 3. Link Provenance & Structural Edges
        # Connect consolidated idea node to all source chunks without overwriting existing edges
        quotes = cluster.get_representative_quotes(max_quotes=self.max_quotes)
        rep_quote = quotes[0] if quotes else ""

        for chunk_id in cluster.chunk_ids:
            chunk_nid = f"chunk:{chunk_id}"
            if store.graph.has_node(chunk_nid):
                # (:Concept) -[:SUPPORTED_BY]-> (:Chunk)
                if not store.graph.has_edge(canonical_nid, chunk_nid):
                    store.graph.add_edge(
                        canonical_nid,
                        chunk_nid,
                        relation="SUPPORTED_BY",
                        quote=rep_quote,
                        brief_description=brief,
                        detailed_explanation=detailed,
                        chunk_id=chunk_id,
                    )
                    store.mark_edge_dirty(canonical_nid, chunk_nid)
                    edges_added += 1

                # (:Chunk) -[:SUPPORTS_IDEA]-> (:Concept)
                if not store.graph.has_edge(chunk_nid, canonical_nid):
                    store.graph.add_edge(
                        chunk_nid,
                        canonical_nid,
                        relation="SUPPORTS_IDEA",
                        quote=rep_quote,
                        brief_description=brief,
                        detailed_explanation=detailed,
                    )
                    store.mark_edge_dirty(chunk_nid, canonical_nid)
                    edges_added += 1

        # 4. Emit directed relational edges ([:RELATES_TO]) only when not already present
        for rel_name in clean_related:
            if store.add_concept_relation(
                src_concept_name=canonical_name,
                tgt_concept_name=rel_name,
                relation_type="RELATES_TO",
            ):
                edges_added += 1

        return edges_added

    def _is_cluster_already_refined(
        self,
        cluster: IdeaCluster,
        state_manager: Optional[IdeaPostProcessorState] = None,
    ) -> bool:
        """
        Check whether a cluster (or any of its merged aliases) has ALREADY been refined
        by the LLM. Refinement must be done no more than 1 time per idea.
        """
        if not self.resume:
            return False
        if cluster.already_refined:
            return True
        if state_manager is not None and state_manager.get_synthesized_output_for_cluster(cluster) is not None:
            return True
        return False

    def _is_cluster_affected(
        self,
        store: ConceptGraphStore,
        cluster: IdeaCluster,
        state_manager: Optional[IdeaPostProcessorState] = None,
    ) -> bool:
        """
        Determine whether a cluster was affected in the current cycle and needs
        post-processing / graph consolidation, or if it was already post-processed
        in an earlier cycle and can be skipped without re-ingestion.
        """
        if not self.resume:
            return True

        canonical_name = cluster.canonical_name
        canonical_nid = f"concept:{canonical_name}"
        if not store.graph.has_node(canonical_nid):
            return True

        # If multiple graph nodes were merged into this cluster (by normalization or cosine similarity),
        # or if any alias node still exists separately in the graph, we must consolidate them.
        if cluster.was_merged or len(cluster.existing_node_ids) > 1:
            return True
        for alias in cluster.aliases:
            if alias != canonical_name and store.graph.has_node(f"concept:{alias}"):
                return True

        nd = store.graph.nodes[canonical_nid]
        is_multi = cluster.total_occurrences >= self.min_occurrences
        already_refined = self._is_cluster_already_refined(cluster, state_manager)

        # If cluster has >= min_occurrences and has never been refined yet, it needs its 1-time LLM refinement!
        if is_multi and not already_refined:
            return True

        # Check whether the node in store.graph already has its post-processed state applied
        if not nd.get("postprocessed"):
            # Backward-compatibility with graphs saved before the 'postprocessed' node flag existed:
            # If state_manager has a cached output whose brief/detailed already match the graph node
            # (and the node wasn't modified in the current batch), stamp it in-memory as postprocessed.
            if state_manager is not None and canonical_nid not in store.dirty_nodes:
                cached_out = (
                    state_manager.get_synthesized_output_for_cluster(cluster)
                    if is_multi
                    else state_manager.get_output(canonical_name)
                )
                if cached_out is not None:
                    node_brief = BookParser.repair_mojibake(nd.get("brief_description") or "")
                    node_det = BookParser.repair_mojibake(nd.get("detailed_explanation") or "")
                    cache_brief = BookParser.repair_mojibake(cached_out.brief_description or "")
                    cache_det = BookParser.repair_mojibake(cached_out.detailed_explanation or "")
                    if node_brief == cache_brief and node_det == cache_det:
                        nd["postprocessed"] = True
                        if is_multi or state_manager.get_status(canonical_name) == "synthesized":
                            nd["refined"] = True
                    else:
                        return True
                else:
                    return True
            else:
                return True

        # If multi-occurrence and already refined in state_manager, ensure graph node has refined state applied
        if is_multi and not nd.get("refined"):
            return True

        # Check if occurrence count, earliest month, embedding, or chunk provenance edges need updating
        if int(nd.get("occurrences") or 0) < cluster.total_occurrences:
            return True

        earliest_month = cluster.get_first_occurrence_month()
        curr_month = nd.get("first_occurrence_month") or nd.get("month")
        if earliest_month and (not curr_month or earliest_month < curr_month):
            return True

        if cluster.embedding is not None and len(cluster.embedding) > 1 and not nd.get("embedding"):
            return True

        for chunk_id in cluster.chunk_ids:
            chunk_nid = f"chunk:{chunk_id}"
            if store.graph.has_node(chunk_nid):
                if not store.graph.has_edge(canonical_nid, chunk_nid) or not store.graph.has_edge(chunk_nid, canonical_nid):
                    return True

        return False

    def _process_single_cluster(
        self,
        cluster: IdeaCluster,
        state_manager: Optional[IdeaPostProcessorState] = None,
    ) -> Tuple[ConsolidatedIdeaOutput, bool, bool]:
        """
        Synthesize or retrieve cached cluster output.
        Refinement via LLM is performed no more than 1 time per idea:
        - If the cluster (or any merged alias) was already refined once, reuses the refined output.
        - If the cluster was previously single-occurrence ('bypassed') and now has >= min_occurrences,
          performs its 1-time LLM refinement now.
        Returns: (output, is_multi, was_resumed)
        """
        is_multi = cluster.total_occurrences >= self.min_occurrences
        if self.resume:
            # 1. Check if already refined in state_manager (canonical or any merged alias)
            if state_manager is not None:
                synth_cached = state_manager.get_synthesized_output_for_cluster(cluster)
                if synth_cached is not None:
                    cluster.already_refined = True
                    if state_manager.get_status(cluster.canonical_name) != "synthesized":
                        state_manager.record_processed(cluster, synth_cached, status="synthesized")
                    return synth_cached, True, True

            # 2. Check if already refined on the graph node itself
            if cluster.already_refined:
                refined_out = ConsolidatedIdeaOutput(
                    brief_description=cluster.refined_brief or cluster.get_representative_brief(),
                    detailed_explanation=cluster.refined_detailed or cluster.get_representative_detailed(),
                    related_entities=list(cluster.refined_related_entities),
                    category=cluster.category,
                    weight=cluster.max_weight,
                )
                if state_manager is not None:
                    state_manager.record_processed(cluster, refined_out, status="synthesized")
                return refined_out, True, True

            # 3. For single-occurrence ideas, reuse cached bypassed output if present
            if not is_multi and state_manager is not None:
                cached = state_manager.get_output(cluster.canonical_name)
                if cached is not None:
                    return cached, False, True

        if is_multi:
            output = self.synthesize_cluster(cluster)
            cluster.already_refined = True
            if state_manager:
                state_manager.record_processed(cluster, output, status="synthesized")
            return output, True, False
        else:
            default_output = ConsolidatedIdeaOutput(
                brief_description=cluster.get_representative_brief(),
                detailed_explanation=cluster.get_representative_detailed(),
                related_entities=[],
                category=cluster.category,
                weight=cluster.max_weight,
            )
            if state_manager:
                state_manager.record_processed(cluster, default_output, status="bypassed")
            return default_output, False, False

    # --------------------------------------------------------------------------
    # Main Postprocessing Pipeline
    # --------------------------------------------------------------------------

    def process_graph(
        self,
        store: ConceptGraphStore,
        console: Optional[Console] = None,
        on_progress: Optional[Callable[[int, int, str, Dict[str, Any]], None]] = None,
    ) -> IdeaPostProcessorReport:
        """
        Execute full two-pass aggregation and synthesis pipeline on the graph:
        1. Group ideas deterministically by normalized name.
        2. Merge clusters by semantic similarity (>= 0.92).
        3. Filter out already post-processed and untouched clusters so only affected
           clusters are processed/ingested, ensuring LLM refinement runs at most once per idea.
        4. Overwrite/upsert canonical concept nodes and provenance/relational edges for affected clusters.
        Supports state file checkpointing (save every N ideas) and auto-continue.
        """
        t0 = time.time()
        t_start = time.perf_counter()

        # 1. Deterministic normalization grouping
        clusters_by_norm = self.collect_observations_from_graph(store, console=console)
        initial_clusters = list(clusters_by_norm.values())
        total_input_concepts = sum(len(c.aliases) for c in initial_clusters)

        # 2. Semantic similarity clustering (across all ideas in the graph)
        merged_clusters = self.cluster_by_semantic_similarity(initial_clusters, console=console)
        merged_diff = len(initial_clusters) - len(merged_clusters)
        total_clusters = len(merged_clusters)

        target_multi = [c for c in merged_clusters if c.total_occurrences >= self.min_occurrences]
        target_single = [c for c in merged_clusters if c.total_occurrences < self.min_occurrences]

        # Checkpoint state manager setup
        state_manager: Optional[IdeaPostProcessorState] = None
        resumed_count = 0
        if self.state_file:
            state_manager = IdeaPostProcessorState(self.state_file)
            state_manager.model_name = self.model_name
            state_manager.similarity_threshold = self.similarity_threshold
            state_manager.min_occurrences = self.min_occurrences
            state_manager.total_clusters = total_clusters
            if not self.resume:
                state_manager.clear()
            elif len(state_manager.processed_ideas) > 0:
                resumed_count = len(state_manager.processed_ideas)
                if console is not None:
                    console.print(
                        f"[bold cyan]• Found existing postprocessor state file: [bold]{state_manager.state_path.name}[/bold] "
                        f"({resumed_count} ideas previously processed). Auto-continuing...[/bold cyan]"
                    )

        # Partition clusters into untouched (already post-processed & up-to-date in graph) vs affected
        affected_clusters: List[IdeaCluster] = []
        untouched_clusters: List[IdeaCluster] = []
        for c in merged_clusters:
            if self._is_cluster_affected(store, c, state_manager):
                affected_clusters.append(c)
            else:
                untouched_clusters.append(c)

        # Display Section Header
        if console is not None:
            resumed_info = (
                f"\n• Resumed from State:       [bold green]{resumed_count}[/bold green] ideas (auto-continued from {state_manager.state_path.name})"
                if resumed_count > 0 and state_manager else ""
            )
            already_proc_info = (
                f"\n• Already Post-Processed:   [bold cyan]{len(untouched_clusters)}[/bold cyan] ideas skipped (affected in this cycle: [bold yellow]{len(affected_clusters)}[/bold yellow])"
                if len(untouched_clusters) > 0 else ""
            )
            checkpoint_info = (
                f"\n• State Checkpoint File:    [bold]{state_manager.state_path}[/bold] (saves every {self.checkpoint_interval} ideas)"
                if state_manager else ""
            )
            header_text = (
                f"[bold cyan]🧠 Two-Pass Idea Aggregation & Synthesis (Pass 2)[/bold cyan]\n"
                f"• Consolidation Model:       [bold green]{self.model_name}[/bold green]\n"
                f"• Total Concepts in Graph:   [bold]{total_input_concepts}[/bold]\n"
                f"• Resolved Entity Clusters:  [bold]{total_clusters}[/bold] (Merged: [yellow]{merged_diff}[/yellow] via cosine sim >= {self.similarity_threshold})\n"
                f"• Multi-Occurrence Ideas:    [bold green]{len(target_multi)}[/bold green] (filtered in for LLM consolidation, >= {self.min_occurrences} mentions)\n"
                f"• Single-Occurrence Ideas:   [dim]{len(target_single)} (filtered out / bypassed to conserve compute)[/dim]\n"
                f"• Cosine Similarity Thresh:  [bold yellow]{self.similarity_threshold}[/bold yellow]"
                f"{already_proc_info}"
                f"{checkpoint_info}"
                f"{resumed_info}"
            )
            console.print(Panel.fit(header_text, title="Idea Postprocessor", border_style="cyan"))
        else:
            logger.info(
                f"Starting IdeaPostProcessor (model={self.model_name}): {total_input_concepts} raw concepts -> "
                f"{total_clusters} clusters ({len(affected_clusters)} affected, {len(untouched_clusters)} already post-processed)."
            )

        single_occ_count = 0
        multi_occ_count = 0
        synthesized_count = 0
        total_edges_added = 0
        actual_resumed_count = 0
        synthesized_names: List[str] = []
        bypassed_names: List[str] = []

        # Account for untouched clusters without re-ingesting them or marking them dirty
        for u_cluster in untouched_clusters:
            actual_resumed_count += 1
            if u_cluster.total_occurrences >= self.min_occurrences or u_cluster.already_refined:
                synthesized_count += 1
                multi_occ_count += 1
                synthesized_names.append(u_cluster.canonical_name)
            else:
                single_occ_count += 1
                bypassed_names.append(u_cluster.canonical_name)

        num_affected = len(affected_clusters)

        try:
            # 3 & 4. Process and apply ONLY affected clusters to the graph
            if console is not None and num_affected > 0:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]Consolidation Progress:[/bold cyan]"),
                    BarColumn(bar_width=25),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TextColumn("{task.fields[stats_line]}"),
                    console=console,
                )
                tot_task = progress.add_task("total", total=num_affected, completed=0, stats_line="")
                first_c = affected_clusters[0]
                first_is_cached = self._is_cluster_already_refined(first_c, state_manager) or bool(
                    not (first_c.total_occurrences >= self.min_occurrences)
                    and state_manager
                    and state_manager.is_processed(first_c.canonical_name)
                )
                if first_c.total_occurrences >= self.min_occurrences:
                    if first_is_cached:
                        first_init_text = f"[bold blue]Current Idea:[/bold blue] '{first_c.canonical_name}' | [bold green]✓ Synthesized (resumed)[/bold green]"
                    else:
                        first_init_text = (
                            f"[bold blue]Current Idea:[/bold blue] '{first_c.canonical_name}' "
                            f"({first_c.total_occurrences} mentions) | [bold yellow]Consolidating with {self.model_name}...[/bold yellow]"
                        )
                else:
                    first_init_text = (
                        f"[bold blue]Current Idea:[/bold blue] '{first_c.canonical_name}' | [dim]Bypassing (single-occurrence)[/dim]"
                    )
                cur_line = Text.from_markup(first_init_text)

                with Live(Group(cur_line, progress), console=console, refresh_per_second=10) as live:
                    for idx, cluster in enumerate(affected_clusters, start=1):
                        is_multi = cluster.total_occurrences >= self.min_occurrences
                        is_already_refined = self._is_cluster_already_refined(cluster, state_manager)

                        if is_multi and not is_already_refined:
                            cur_text = (
                                f"[bold blue]Current Idea:[/bold blue] '{cluster.canonical_name}' "
                                f"({cluster.total_occurrences} mentions) | [bold yellow]Consolidating with {self.model_name}...[/bold yellow]"
                            )
                            live.update(Group(Text.from_markup(cur_text), progress))

                        output, eff_is_multi, was_resumed = self._process_single_cluster(cluster, state_manager)
                        edges_added = self.apply_consolidation_to_graph(store, cluster, output)
                        total_edges_added += edges_added

                        if was_resumed:
                            actual_resumed_count += 1

                        if eff_is_multi:
                            synthesized_count += 1
                            multi_occ_count += 1
                            synthesized_names.append(cluster.canonical_name)
                            status_badge = "[bold green]✓ Synthesized (resumed)[/bold green]" if was_resumed else "[bold green]✓ Synthesized[/bold green]"
                        else:
                            single_occ_count += 1
                            bypassed_names.append(cluster.canonical_name)
                            status_badge = "[dim]Bypassed (resumed)[/dim]" if was_resumed else "[dim]Bypassed (single-occurrence)[/dim]"

                        completed = idx
                        cur_text = f"[bold blue]Current Idea:[/bold blue] '{cluster.canonical_name}' | {status_badge}"

                        elapsed = time.perf_counter() - t_start
                        rate = (completed / elapsed) if elapsed > 0 else 0.0
                        if completed > 0 and num_affected > completed and rate > 0:
                            rem_sec = (num_affected - completed) / rate
                            eta_str = _format_eta(rem_sec)
                        elif completed >= num_affected:
                            eta_str = "0m 00s"
                        else:
                            eta_str = "--m --s"

                        rate_str = f"{rate:.2f} idea/s" if rate > 0 else "-- idea/s"
                        resumed_stats = f" | [cyan]{actual_resumed_count} resumed[/cyan]" if actual_resumed_count > 0 else ""
                        stats_str = (
                            f" | [bold yellow]{rate_str}[/bold yellow] | ETA: {eta_str} | "
                            f"[bold green]✓ {synthesized_count} synthesized[/bold green] | "
                            f"[dim]{single_occ_count} bypassed[/dim]{resumed_stats}"
                        )

                        progress.update(tot_task, completed=completed, total=num_affected, stats_line=stats_str)
                        live.update(Group(Text.from_markup(cur_text), progress))

                        # Save checkpoint every checkpoint_interval (default: 100)
                        if state_manager and (completed % self.checkpoint_interval == 0 or completed == num_affected):
                            state_manager.save()
                            if completed % self.checkpoint_interval == 0 and completed < num_affected:
                                console.print(
                                    f"[bold cyan][Idea Postprocessor][/bold cyan] [dim]💾 Checkpoint saved at {completed}/{num_affected} ideas -> {state_manager.state_path.name}[/dim]"
                                )
                                sys.stdout.flush()

                        # Periodic statistical line info every 10 ideas (or on final completion)
                        if completed % 10 == 0 or completed == num_affected:
                            pct = (completed / num_affected * 100) if num_affected > 0 else 100.0
                            console.print(
                                f"[bold cyan][Idea Postprocessor][/bold cyan] {completed}/{num_affected} ideas ({pct:.1f}%)"
                                f" | {rate_str} | ETA: {eta_str} | [bold green]✓ {synthesized_count} synthesized[/bold green]"
                                f" | [dim]{single_occ_count} bypassed[/dim]{resumed_stats} | +{total_edges_added} edges"
                            )
                            sys.stdout.flush()

                        if on_progress:
                            on_progress(completed, num_affected, cluster.canonical_name, {
                                "synthesized": synthesized_count,
                                "bypassed": single_occ_count,
                                "resumed": actual_resumed_count,
                                "edges_added": total_edges_added,
                            })
            elif console is not None and total_clusters > 0 and num_affected == 0:
                console.print(
                    f"[bold cyan][Idea Postprocessor][/bold cyan] All {total_clusters} ideas already post-processed and up-to-date "
                    f"([cyan]{actual_resumed_count} resumed[/cyan], 0 affected ideas to re-ingest, +0 edges)."
                )
            else:
                for idx, cluster in enumerate(affected_clusters, start=1):
                    output, eff_is_multi, was_resumed = self._process_single_cluster(cluster, state_manager)
                    edges_added = self.apply_consolidation_to_graph(store, cluster, output)
                    total_edges_added += edges_added

                    if was_resumed:
                        actual_resumed_count += 1

                    if eff_is_multi:
                        synthesized_count += 1
                        multi_occ_count += 1
                        synthesized_names.append(cluster.canonical_name)
                    else:
                        single_occ_count += 1
                        bypassed_names.append(cluster.canonical_name)

                    completed = idx
                    if state_manager and (completed % self.checkpoint_interval == 0 or completed == num_affected):
                        state_manager.save()

                    if completed % 10 == 0 or completed == num_affected:
                        pct = (completed / num_affected * 100) if num_affected > 0 else 100.0
                        logger.info(
                            f"[Idea Postprocessor] {completed}/{num_affected} affected ideas ({pct:.1f}%)"
                            f" | Synthesized: {synthesized_count} | Bypassed: {single_occ_count} | Resumed: {actual_resumed_count} | Edges: +{total_edges_added}"
                        )

                    if on_progress:
                        on_progress(completed, num_affected, cluster.canonical_name, {
                            "synthesized": synthesized_count,
                            "bypassed": single_occ_count,
                            "resumed": actual_resumed_count,
                            "edges_added": total_edges_added,
                        })
        finally:
            if state_manager:
                state_manager.save()

        duration = time.time() - t0
        report = IdeaPostProcessorReport(
            total_input_concepts=total_input_concepts,
            total_clusters=len(merged_clusters),
            single_occurrence_clusters_count=single_occ_count,
            multi_occurrence_clusters_count=multi_occ_count,
            synthesized_count=synthesized_count,
            resumed_count=actual_resumed_count,
            merged_clusters_count=merged_diff,
            edges_added_count=total_edges_added,
            duration_seconds=round(duration, 2),
            synthesized_ideas=synthesized_names,
            bypassed_ideas=bypassed_names,
        )

        logger.info(
            f"IdeaPostProcessor completed in {duration:.2f}s: "
            f"{len(merged_clusters)} total clusters ({num_affected} affected, {len(untouched_clusters)} untouched; "
            f"{multi_occ_count} synthesized, {single_occ_count} bypassed, {actual_resumed_count} resumed)."
        )
        return report
