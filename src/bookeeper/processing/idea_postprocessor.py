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
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
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
)
from rich.text import Text

from bookeeper.calibre.parser import BookParser
from bookeeper.config import Settings
from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings, OllamaPool

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
    merged_clusters_count: int = 0
    edges_added_count: int = 0
    duration_seconds: float = 0.0
    synthesized_ideas: List[str] = Field(default_factory=list)
    bypassed_ideas: List[str] = Field(default_factory=list)


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

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        model_name: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        pool: Optional[OllamaPool] = None,
        embeddings: Optional[Embeddings] = None,
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

                if norm_name not in clusters_by_norm:
                    clusters_by_norm[norm_name] = IdeaCluster(
                        canonical_name=clean_cand,
                        normalized_name=norm_name,
                        category=cat,
                        max_weight=weight,
                    )

                cluster = clusters_by_norm[norm_name]
                cluster.aliases.add(raw_name)
                if clean_cand:
                    cluster.aliases.add(clean_cand)

                # Update canonical name if current has leading determiner or candidate is better capitalized
                if bool(re.match(r"^(?:the|a|an)\s+", cluster.canonical_name, re.IGNORECASE)):
                    cluster.canonical_name = clean_cand
                elif len(clean_cand) > len(cluster.canonical_name) and not clean_cand.islower():
                    cluster.canonical_name = clean_cand

                # 2. Gather edges connecting this concept to Chunks
                # SUPPORTED_BY edges (Concept -> Chunk)
                chunk_edges = []
                for _, target, edge_data in store.graph.out_edges(node_id, data=True):
                    if edge_data.get("relation") == "SUPPORTED_BY":
                        chunk_edges.append((target, edge_data))

                # SUPPORTS_IDEA edges (Chunk -> Concept)
                for source, _, edge_data in store.graph.in_edges(node_id, data=True):
                    if edge_data.get("relation") == "SUPPORTS_IDEA":
                        chunk_edges.append((source, edge_data))

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
        Computes dense embeddings for canonical_name + brief_description.
        Merges unmerged clusters if cosine similarity >= 0.92.
        """
        if not clusters or self.embeddings is None:
            return clusters

        # Generate texts for embedding
        texts = [f"{c.canonical_name}: {c.get_representative_brief()}" for c in clusters]
        emb_model = getattr(self.embeddings, "model", "embeddings")
        batch_size = 16

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
                    console=console,
                )
                with emb_progress:
                    task = emb_progress.add_task("embed", total=len(texts))
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i : i + batch_size]
                        vecs = self.embeddings.embed_documents(batch)
                        vectors_raw.extend(vecs)
                        emb_progress.update(task, advance=len(batch))
            else:
                vectors_raw = self.embeddings.embed_documents(texts)

            if len(vectors_raw) != len(clusters):
                logger.warning(
                    f"Mismatch between clusters ({len(clusters)}) and embeddings ({len(vectors_raw)}). Skipping semantic merge."
                )
                return clusters

            if console is not None:
                console.print(
                    f"[bold green]✓[/bold green] [dim]Generated semantic embeddings for {len(texts)} clusters ({emb_model})[/dim]"
                )

            # Normalize vectors to unit norm
            vectors = []
            for v in vectors_raw:
                arr = np.array(v, dtype=np.float32)
                norm = np.linalg.norm(arr)
                vectors.append(arr / norm if norm > 0 else arr)
        except Exception as e:
            logger.warning(f"Error computing cluster embeddings: {e}. Skipping semantic merge.")
            if console is not None:
                console.print(f"[yellow]⚠ Semantic embedding skipped: {e}[/yellow]")
            return clusters

        status = None
        if console is not None:
            status = console.status(
                f"[bold cyan]Clustering concepts via cosine similarity (>= {self.similarity_threshold})...[/bold cyan]",
                spinner="dots",
            )
            status.start()

        try:
            for i, cluster in enumerate(clusters):
                cluster.embedding = vectors[i]

            merged: List[IdeaCluster] = []
            consumed_indices: Set[int] = set()

            for i in range(len(clusters)):
                if i in consumed_indices:
                    continue

                current = clusters[i]
                consumed_indices.add(i)

                # Search remaining clusters for cosine similarity >= threshold
                for j in range(i + 1, len(clusters)):
                    if j in consumed_indices:
                        continue

                    other = clusters[j]
                    if current.embedding is not None and other.embedding is not None:
                        sim = float(np.dot(current.embedding, other.embedding))
                        if sim >= self.similarity_threshold:
                            logger.info(
                                f"Semantic merge ({sim:.3f} >= {self.similarity_threshold}): "
                                f"'{other.canonical_name}' into '{current.canonical_name}'"
                            )
                            # Merge other into current
                            current.aliases.update(other.aliases)
                            current.aliases.add(other.canonical_name)
                            for obs in other.observations:
                                current.add_observation(obs)
                            current.chunk_ids.update(other.chunk_ids)
                            current.max_weight = max(current.max_weight, other.max_weight)
                            consumed_indices.add(j)

                merged.append(current)
        finally:
            if status is not None:
                status.stop()

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
        relationships (:MENTIONED_IN, :SUPPORTED_BY), and emit directed :RELATES_TO edges.
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

        # 1. Upsert Canonical Entity Node
        if not store.graph.has_node(canonical_nid):
            store.graph.add_node(
                canonical_nid,
                type="Concept",
                tag="idea",
                tags=["idea", cat.lower().replace(" ", "-")],
                is_idea=True,
                name=canonical_name,
                category=cat,
                brief_description=brief,
                detailed_explanation=detailed,
                summary=brief or detailed,
                weight=weight,
                occurrences=occurrences,
            )
        else:
            nd = store.graph.nodes[canonical_nid]
            nd["type"] = "Concept"
            nd["is_idea"] = True
            nd["tag"] = "idea"
            nd["name"] = canonical_name
            nd["brief_description"] = brief
            nd["detailed_explanation"] = detailed
            nd["summary"] = brief or detailed
            nd["category"] = cat
            nd["weight"] = max(int(nd.get("weight") or 0), weight)
            nd["occurrences"] = max(int(nd.get("occurrences") or 0), occurrences)
            if "tags" not in nd:
                nd["tags"] = ["idea", cat.lower().replace(" ", "-")]

        # 2. Redirect alias nodes into canonical node and remove redundant nodes
        for alias in cluster.aliases:
            if alias == canonical_name:
                continue
            alias_nid = f"concept:{alias}"
            if store.graph.has_node(alias_nid):
                # Redirect in-edges to canonical
                in_edges = list(store.graph.in_edges(alias_nid, data=True))
                for src, _, edata in in_edges:
                    if src != canonical_nid:
                        store.graph.add_edge(src, canonical_nid, **edata)
                # Redirect out-edges to canonical
                out_edges = list(store.graph.out_edges(alias_nid, data=True))
                for _, tgt, edata in out_edges:
                    if tgt != canonical_nid:
                        store.graph.add_edge(canonical_nid, tgt, **edata)
                # Remove alias node
                store.graph.remove_node(alias_nid)

        # 3. Link Provenance & Structural Edges
        # Connect consolidated idea node to all source chunks
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
                    edges_added += 1

                # Explicit [:MENTIONED_IN] provenance relation
                if not store.graph.has_edge(canonical_nid, chunk_nid) or store.graph[canonical_nid][chunk_nid].get("relation") != "MENTIONED_IN":
                    # We can store a specific MENTIONED_IN edge or mark on SUPPORTED_BY
                    store.graph.add_edge(
                        canonical_nid,
                        chunk_nid,
                        relation="MENTIONED_IN",
                        chunk_id=chunk_id,
                    )
                    edges_added += 1

        # 4. Emit directed relational edges ([:RELATES_TO])
        for rel_entity in output.related_entities:
            rel_name = BookParser.repair_mojibake(rel_entity).strip()
            if rel_name and rel_name.lower() != canonical_name.lower():
                store.add_concept_relation(
                    src_concept_name=canonical_name,
                    tgt_concept_name=rel_name,
                    relation_type="RELATES_TO",
                )
                edges_added += 1

        return edges_added

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
        3. For N == 1: pass through as-is (bypass LLM).
           For N >= 2: synthesize cohesive brief and detailed explanations with LLM.
        4. Overwrite/upsert canonical concept nodes and provenance/relational edges.
        """
        t0 = time.time()
        t_start = time.perf_counter()

        # 1. Deterministic normalization grouping
        clusters_by_norm = self.collect_observations_from_graph(store, console=console)
        initial_clusters = list(clusters_by_norm.values())
        total_input_concepts = sum(len(c.aliases) for c in initial_clusters)

        # 2. Semantic similarity clustering
        merged_clusters = self.cluster_by_semantic_similarity(initial_clusters, console=console)
        merged_diff = len(initial_clusters) - len(merged_clusters)
        total_clusters = len(merged_clusters)

        target_multi = [c for c in merged_clusters if c.total_occurrences >= self.min_occurrences]
        target_single = [c for c in merged_clusters if c.total_occurrences < self.min_occurrences]

        # Display Section Header
        if console is not None:
            header_text = (
                f"[bold cyan]🧠 Two-Pass Idea Aggregation & Synthesis (Pass 2)[/bold cyan]\n"
                f"• Consolidation Model:       [bold green]{self.model_name}[/bold green]\n"
                f"• Total Concepts in Graph:   [bold]{total_input_concepts}[/bold]\n"
                f"• Resolved Entity Clusters:  [bold]{total_clusters}[/bold] (Merged: [yellow]{merged_diff}[/yellow] via cosine sim >= {self.similarity_threshold})\n"
                f"• Multi-Occurrence Ideas:    [bold green]{len(target_multi)}[/bold green] (filtered in for LLM consolidation, >= {self.min_occurrences} mentions)\n"
                f"• Single-Occurrence Ideas:   [dim]{len(target_single)} (filtered out / bypassed to conserve compute)[/dim]\n"
                f"• Cosine Similarity Thresh:  [bold yellow]{self.similarity_threshold}[/bold yellow]"
            )
            console.print(Panel.fit(header_text, title="Idea Postprocessor", border_style="cyan"))
        else:
            logger.info(
                f"Starting IdeaPostProcessor (model={self.model_name}): {total_input_concepts} raw concepts -> "
                f"{total_clusters} clusters ({len(target_multi)} multi-occurrence filtered for synthesis, "
                f"{len(target_single)} single-occurrence bypassed)."
            )

        single_occ_count = 0
        multi_occ_count = 0
        synthesized_count = 0
        total_edges_added = 0
        synthesized_names: List[str] = []
        bypassed_names: List[str] = []

        # 3 & 4. Branch by mention count and apply to graph with live progress
        if console is not None and total_clusters > 0:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]Consolidation Progress:[/bold cyan]"),
                BarColumn(bar_width=25),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TextColumn("{task.fields[stats_line]}"),
                console=console,
            )
            tot_task = progress.add_task("total", total=total_clusters, completed=0, stats_line="")
            first_c = merged_clusters[0]
            if first_c.total_occurrences >= self.min_occurrences:
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
                for idx, cluster in enumerate(merged_clusters, start=1):
                    is_multi = cluster.total_occurrences >= self.min_occurrences
                    if is_multi:
                        cur_text = (
                            f"[bold blue]Current Idea:[/bold blue] '{cluster.canonical_name}' "
                            f"({cluster.total_occurrences} mentions) | [bold yellow]Consolidating with {self.model_name}...[/bold yellow]"
                        )
                        live.update(Group(Text.from_markup(cur_text), progress))
                        output = self.synthesize_cluster(cluster)
                        edges_added = self.apply_consolidation_to_graph(store, cluster, output)
                        total_edges_added += edges_added
                        synthesized_count += 1
                        multi_occ_count += 1
                        synthesized_names.append(cluster.canonical_name)
                        status_badge = "[bold green]✓ Synthesized[/bold green]"
                    else:
                        default_output = ConsolidatedIdeaOutput(
                            brief_description=cluster.get_representative_brief(),
                            detailed_explanation=cluster.get_representative_detailed(),
                            related_entities=[],
                            category=cluster.category,
                            weight=cluster.max_weight,
                        )
                        edges_added = self.apply_consolidation_to_graph(store, cluster, default_output)
                        total_edges_added += edges_added
                        single_occ_count += 1
                        bypassed_names.append(cluster.canonical_name)
                        status_badge = "[dim]Bypassed (single-occurrence)[/dim]"

                    completed = idx
                    cur_text = f"[bold blue]Current Idea:[/bold blue] '{cluster.canonical_name}' | {status_badge}"

                    elapsed = time.perf_counter() - t_start
                    rate = (completed / elapsed) if elapsed > 0 else 0.0
                    if completed > 0 and total_clusters > completed and rate > 0:
                        rem_sec = (total_clusters - completed) / rate
                        eta_str = _format_eta(rem_sec)
                    elif completed >= total_clusters:
                        eta_str = "0m 00s"
                    else:
                        eta_str = "--m --s"

                    rate_str = f"{rate:.2f} idea/s" if rate > 0 else "-- idea/s"
                    stats_str = (
                        f" | [bold yellow]{rate_str}[/bold yellow] | ETA: {eta_str} | "
                        f"[bold green]✓ {synthesized_count} synthesized[/bold green] | "
                        f"[dim]{single_occ_count} bypassed[/dim]"
                    )

                    progress.update(tot_task, completed=completed, total=total_clusters, stats_line=stats_str)
                    live.update(Group(Text.from_markup(cur_text), progress))

                    # Periodic statistical line info every 10 ideas (or on final completion)
                    if completed % 10 == 0 or completed == total_clusters:
                        pct = (completed / total_clusters * 100) if total_clusters > 0 else 100.0
                        console.print(
                            f"[bold cyan][Idea Postprocessor][/bold cyan] {completed}/{total_clusters} ideas ({pct:.1f}%)"
                            f" | {rate_str} | ETA: {eta_str} | [bold green]✓ {synthesized_count} synthesized[/bold green]"
                            f" | [dim]{single_occ_count} bypassed[/dim] | +{total_edges_added} edges"
                        )
                        sys.stdout.flush()

                    if on_progress:
                        on_progress(completed, total_clusters, cluster.canonical_name, {
                            "synthesized": synthesized_count,
                            "bypassed": single_occ_count,
                            "edges_added": total_edges_added,
                        })
        else:
            for idx, cluster in enumerate(merged_clusters, start=1):
                is_multi = cluster.total_occurrences >= self.min_occurrences
                if is_multi:
                    output = self.synthesize_cluster(cluster)
                    edges_added = self.apply_consolidation_to_graph(store, cluster, output)
                    total_edges_added += edges_added
                    synthesized_count += 1
                    multi_occ_count += 1
                    synthesized_names.append(cluster.canonical_name)
                else:
                    default_output = ConsolidatedIdeaOutput(
                        brief_description=cluster.get_representative_brief(),
                        detailed_explanation=cluster.get_representative_detailed(),
                        related_entities=[],
                        category=cluster.category,
                        weight=cluster.max_weight,
                    )
                    edges_added = self.apply_consolidation_to_graph(store, cluster, default_output)
                    total_edges_added += edges_added
                    single_occ_count += 1
                    bypassed_names.append(cluster.canonical_name)

                completed = idx
                if completed % 10 == 0 or completed == total_clusters:
                    pct = (completed / total_clusters * 100) if total_clusters > 0 else 100.0
                    logger.info(
                        f"[Idea Postprocessor] {completed}/{total_clusters} ideas ({pct:.1f}%)"
                        f" | Synthesized: {synthesized_count} | Bypassed: {single_occ_count} | Edges: +{total_edges_added}"
                    )

                if on_progress:
                    on_progress(completed, total_clusters, cluster.canonical_name, {
                        "synthesized": synthesized_count,
                        "bypassed": single_occ_count,
                        "edges_added": total_edges_added,
                    })

        duration = time.time() - t0
        report = IdeaPostProcessorReport(
            total_input_concepts=total_input_concepts,
            total_clusters=len(merged_clusters),
            single_occurrence_clusters_count=single_occ_count,
            multi_occurrence_clusters_count=multi_occ_count,
            synthesized_count=synthesized_count,
            merged_clusters_count=merged_diff,
            edges_added_count=total_edges_added,
            duration_seconds=round(duration, 2),
            synthesized_ideas=synthesized_names,
            bypassed_ideas=bypassed_names,
        )

        logger.info(
            f"IdeaPostProcessor completed in {duration:.2f}s: "
            f"{len(merged_clusters)} clusters ({multi_occ_count} synthesized, {single_occ_count} bypassed)."
        )
        return report
