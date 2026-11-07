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
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    orjson = None
    HAS_ORJSON = False

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, model_validator
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


class ConsolidatedIdeaBatchItem(BaseModel):
    """Structured LLM synthesis for an individual idea inside a batched prompt."""

    idea_id: Optional[str] = Field(
        default=None,
        description="Exact idea ID matching the <idea id=\"...\"> tag in the prompt.",
    )
    canonical_name: Optional[str] = Field(
        default=None,
        description="Canonical name of the target idea / concept.",
    )
    brief_description: str = Field(
        description="Cohesive 1-2 sentence description capturing the macro-identity."
    )
    detailed_explanation: str = Field(
        description="Synthesized comprehensive explanation integrating distinct events, cause-and-effect chains, mechanisms, and conditions across all chunks without dropping earlier facts."
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


class BatchConsolidatedIdeaOutput(BaseModel):
    """Synthesis output containing consolidated ideas for multiple combined idea clusters."""

    ideas: List[ConsolidatedIdeaBatchItem] = Field(
        default_factory=list,
        description="List of consolidated ideas matching each provided idea tag.",
    )

    @model_validator(mode="before")
    @classmethod
    def sanitize_batch(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"ideas": data}
        if isinstance(data, dict):
            if "ideas" in data and isinstance(data["ideas"], list):
                return data
            for alt_key in ("consolidated_ideas", "items", "results", "concepts", "clusters", "chunks"):
                if alt_key in data and isinstance(data[alt_key], list):
                    data["ideas"] = data[alt_key]
                    return data
            # Check if dict keys are idea IDs (e.g. {"idea_1": {...}, "idea_2": {...}})
            items = []
            for k, val in data.items():
                if isinstance(val, dict):
                    val_copy = dict(val)
                    val_copy.setdefault("idea_id", str(k))
                    items.append(val_copy)
            if items:
                return {"ideas": items}
        return data


def salvage_truncated_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to recover completed idea items from a JSON payload that was truncated mid-stream (e.g. at token limit).
    """
    if not raw or not isinstance(raw, str):
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    last_brace = raw.rfind('}')
    while last_brace > 0:
        candidate = raw[:last_brace + 1].strip()
        for suffix in (']}', '}', ']}'):
            try:
                data = json.loads(candidate + suffix)
                if isinstance(data, dict):
                    for k in ("ideas", "consolidated_ideas", "items", "results"):
                        if k in data and isinstance(data[k], list) and len(data[k]) > 0:
                            return data
            except Exception:
                pass
        last_brace = raw.rfind('}', 0, last_brace)
    return None


def batch_ideas_by_context_size(
    clusters: Sequence[IdeaCluster],
    format_payload_fn: Callable[[IdeaCluster], Tuple[str, List[str]]],
    max_context_size: int = 7168,
    max_idea_batch: Optional[int] = 15,
    tag_overhead: int = 150,
) -> List[List[IdeaCluster]]:
    """
    Combine idea clusters into batches up to max_context_size bytes (default: 7168 bytes / 7KB)
    and up to max_idea_batch items (default: 15).
    Each idea's formatted payload and XML tag overhead are accounted for.
    If a single idea cluster exceeds max_context_size, it is placed in its own single-item batch.
    """
    if max_context_size <= 0:
        return [[c] for c in clusters]

    batches: List[List[IdeaCluster]] = []
    current_batch: List[IdeaCluster] = []
    current_size = 0

    for cluster in clusters:
        payload, _ = format_payload_fn(cluster)
        payload_size = len(payload.encode("utf-8")) + tag_overhead
        can_fit_size = (current_size + payload_size <= max_context_size)
        can_fit_count = (max_idea_batch is None or max_idea_batch <= 0 or len(current_batch) < max_idea_batch)

        if not current_batch:
            current_batch.append(cluster)
            current_size = payload_size
        elif can_fit_size and can_fit_count:
            current_batch.append(cluster)
            current_size += payload_size
        else:
            batches.append(current_batch)
            current_batch = [cluster]
            current_size = payload_size

    if current_batch:
        batches.append(current_batch)

    return batches


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
    Supports high-speed C-accelerated serialization (orjson) and asynchronous
    background persistence to eliminate IO pauses on processing safe points.
    """

    def __init__(self, state_path: Path | str, async_save: bool = True):
        self.state_path = Path(state_path).expanduser().resolve()
        self.version: int = 1
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.total_clusters: int = 0
        self.model_name: str = ""
        self.similarity_threshold: float = 0.92
        self.min_occurrences: int = 2
        self.processed_ideas: Dict[str, Dict[str, Any]] = {}
        self.async_save_enabled: bool = async_save
        self._dirty: bool = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending_future: Optional[Future] = None
        self._lock = threading.Lock()
        self.load()

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="postproc_saver")
        return self._executor

    def load(self) -> bool:
        """Load state from disk. Returns True if existing valid state was loaded."""
        if not self.state_path.is_file():
            return False
        try:
            if HAS_ORJSON:
                with open(self.state_path, "rb") as f:
                    data = orjson.loads(f.read())
            else:
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
                self._dirty = False
                return True
        except Exception as e:
            logger.warning(f"Could not load postprocessor state from {self.state_path} ({e}); starting fresh.")
        return False

    @staticmethod
    def _write_snapshot_to_disk(data: Dict[str, Any], state_path: Path) -> None:
        """Helper executed in background thread or synchronously to serialize and atomically replace state file."""
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            dir_name = state_path.parent
            temp_name = None
            try:
                if HAS_ORJSON:
                    payload = orjson.dumps(data)
                    with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tf:
                        tf.write(payload)
                        temp_name = tf.name
                else:
                    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                        json.dump(data, tf, separators=(',', ':'), ensure_ascii=False)
                        temp_name = tf.name
                os.replace(temp_name, state_path)
                logger.debug(f"Saved postprocessor state ({data.get('completed_count', 0)} ideas) to {state_path}")
            finally:
                if temp_name and os.path.exists(temp_name):
                    try:
                        os.unlink(temp_name)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to persist postprocessor state to {state_path}: {e}")

    def save_async(self) -> None:
        """
        Asynchronously dispatch checkpoint save in background thread.
        Never pauses the main worker thread. If a save is already in-flight, coalesces.
        """
        if not self._dirty:
            return

        with self._lock:
            if self._pending_future is not None and not self._pending_future.done():
                # Previous save still writing to disk; coalesce and let subsequent checkpoint/flush write newer snapshot
                return

            self.updated_at = time.time()
            snapshot = {
                "version": self.version,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "total_clusters": self.total_clusters,
                "model_name": self.model_name,
                "similarity_threshold": self.similarity_threshold,
                "min_occurrences": self.min_occurrences,
                "completed_count": len(self.processed_ideas),
                "processed_ideas": dict(self.processed_ideas),
            }
            self._dirty = False
            self._pending_future = self._get_executor().submit(
                self._write_snapshot_to_disk, snapshot, self.state_path
            )

    def save(self, async_save: bool = False) -> None:
        """
        Persist state to disk.
        If async_save=True, delegates to save_async().
        If async_save=False (default), flushes any pending background write and writes synchronously.
        """
        if async_save:
            self.save_async()
            return

        self.flush()

        # If dirty or file does not exist yet, write snapshot synchronously
        if self._dirty or not self.state_path.is_file():
            with self._lock:
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
                    "processed_ideas": dict(self.processed_ideas),
                }
                self._dirty = False
                self._write_snapshot_to_disk(data, self.state_path)

    def flush(self) -> None:
        """Wait for any in-flight background save to complete, and write any remaining dirty state."""
        with self._lock:
            if self._pending_future is not None:
                try:
                    self._pending_future.result(timeout=120)
                except Exception as e:
                    logger.warning(f"Background save failed during flush: {e}")
                self._pending_future = None

            if self._dirty:
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
                    "processed_ideas": dict(self.processed_ideas),
                }
                self._dirty = False
                self._write_snapshot_to_disk(data, self.state_path)

    def close(self) -> None:
        """Cleanly flush and shut down background executor."""
        self.flush()
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    def __enter__(self) -> "IdeaPostProcessorState":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_executor") and self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass

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
        self._dirty = True

    def clear(self) -> None:
        """Clear all in-memory and on-disk state."""
        self.flush()
        self.processed_ideas.clear()
        self._dirty = False
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
        async_save: bool = True,
        max_idea_batch: int = 15,
        max_context_size: int = 7168,
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
        self.async_save = async_save
        self.max_idea_batch = max_idea_batch
        self.max_context_size = max_context_size

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
        async_save: Optional[bool] = None,
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
        eff_async_save = (
            async_save
            if async_save is not None
            else getattr(settings.dual_pass, "async_save", True)
        )
        eff_max_idea_batch = (
            settings.dual_pass.max_idea_batch
            if getattr(settings.dual_pass, "max_idea_batch", None) is not None
            else min(getattr(settings, "max_chunk_batch", 15), 15)
        )
        eff_max_context_size = (
            settings.dual_pass.max_context_size
            if getattr(settings.dual_pass, "max_context_size", None) is not None
            else getattr(settings, "max_context_size", 7168)
        )

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
            async_save=eff_async_save,
            max_idea_batch=eff_max_idea_batch,
            max_context_size=eff_max_context_size,
        )

    # --------------------------------------------------------------------------
    # Step 1: Entity Grouping & Clustering (Deterministic Pass + Similarity)
    # --------------------------------------------------------------------------

    def collect_observations_from_graph(
        self,
        store: ConceptGraphStore,
        console: Optional[Console] = None,
        affected_concepts: Optional[Set[str]] = None,
    ) -> Dict[str, IdeaCluster]:
        """
        Step 1a: Collect idea objects from the graph, normalize names, and group
        ideas sharing an identical normalized name into the same cluster.
        If affected_concepts is provided, restricts collection strictly to nodes
        associated with those affected concepts.
        """
        status = None
        if console is not None:
            status = console.status(
                "[bold cyan]Scanning Knowledge Graph & normalizing idea mentions...[/bold cyan]",
                spinner="dots",
            )
            status.start()

        clusters_by_norm: Dict[str, IdeaCluster] = {}
        if affected_concepts is not None and not affected_concepts:
            if status is not None:
                status.stop()
            return clusters_by_norm

        clean_affected_names: Set[str] = set()
        affected_lower_names: Set[str] = set()
        affected_norm_names: Set[str] = set()
        affected_node_ids: Set[str] = set()

        if affected_concepts is not None:
            for c in affected_concepts:
                if not c:
                    continue
                c_clean = c[len("concept:"):].strip() if c.startswith("concept:") else c.strip()
                if not c_clean:
                    continue
                clean_affected_names.add(c_clean)
                affected_lower_names.add(c_clean.lower())
                c_norm = normalize_idea_name(c_clean)
                if c_norm:
                    affected_norm_names.add(c_norm)
                affected_node_ids.add(f"concept:{c_clean}")
                if c.startswith("concept:"):
                    affected_node_ids.add(c)

        try:
            # 1. Inspect Concept nodes
            for node_id, attrs in store.graph.nodes(data=True):
                if attrs.get("type") != "Concept" and not attrs.get("is_idea"):
                    continue

                raw_name = attrs.get("name") or node_id.replace("concept:", "")
                norm_cand = normalize_idea_name(raw_name)
                if not norm_cand:
                    continue

                if affected_concepts is not None:
                    if (
                        node_id not in affected_node_ids
                        and raw_name.lower() not in affected_lower_names
                        and norm_cand not in affected_norm_names
                    ):
                        continue

                norm_name = norm_cand

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

        # Prepare embedding matrix for vectorized fast similarity checks
        valid_emb_indices = [
            idx for idx, c in enumerate(clusters)
            if c.embedding is not None and isinstance(c.embedding, np.ndarray) and c.embedding.ndim == 1 and len(c.embedding) > 1
        ]
        valid_emb_set = set(valid_emb_indices)
        emb_dim = len(clusters[valid_emb_indices[0]].embedding) if valid_emb_indices else 0
        emb_mat = np.zeros((len(clusters), emb_dim), dtype=np.float32) if emb_dim > 0 else None
        if emb_mat is not None:
            for idx in valid_emb_indices:
                emb_mat[idx] = clusters[idx].embedding

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

        # Block-wise vectorized clustering (BLAS accelerated dot products in bounded RAM blocks)
        BLOCK_SIZE = 1024
        with cluster_progress as c_prog:
            c_task = c_prog.add_task("clustering", total=len(clusters), thresh=f"{self.similarity_threshold:.2f}") if c_prog else None

            for start_idx in range(0, len(clusters), BLOCK_SIZE):
                end_idx = min(start_idx + BLOCK_SIZE, len(clusters))
                if c_prog and c_task is not None:
                    c_prog.update(c_task, completed=start_idx)

                # Compute batch similarity block (end_idx - start_idx, len(clusters)) via BLAS
                block_sims = (emb_mat[start_idx:end_idx] @ emb_mat.T) if emb_mat is not None else None

                for row_idx in range(end_idx - start_idx):
                    i = start_idx + row_idx
                    if i in consumed_indices:
                        continue

                    current = clusters[i]
                    consumed_indices.add(i)

                    # Search remaining clusters for cosine similarity >= threshold (vectorized)
                    if block_sims is not None and i in valid_emb_set:
                        row_sims = block_sims[row_idx, i + 1:]
                        candidates = np.where(row_sims >= self.similarity_threshold)[0]
                        for c_rel in candidates:
                            j = i + 1 + int(c_rel)
                            if j in consumed_indices or j not in valid_emb_set:
                                continue
                            other = clusters[j]
                            sim = float(row_sims[c_rel])
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
                    elif emb_mat is None or i not in valid_emb_set:
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

    def _merge_graph_node_into_cluster(
        self,
        store: ConceptGraphStore,
        cluster: IdeaCluster,
        node_id: str,
    ) -> None:
        """
        Merge an existing graph concept node into an active IdeaCluster, incorporating
        its observations, aliases, weight, occurrences, and provenance edges.
        """
        attrs = store.graph.nodes.get(node_id, {})
        if not attrs:
            return

        raw_name = (attrs.get("name") or node_id.replace("concept:", "")).strip()
        clean_cand = clean_canonical_name(raw_name)
        brief = attrs.get("brief_description") or attrs.get("summary") or ""
        detailed = attrs.get("detailed_explanation") or attrs.get("summary") or ""
        cat = attrs.get("category") or "Idea"
        weight = int(attrs.get("weight") or 5)
        node_month = format_idea_month(attrs.get("first_occurrence_month") or attrs.get("month"))

        cluster.was_merged = True
        cluster.already_postprocessed = False
        cluster.existing_node_ids.add(node_id)
        cluster.aliases.add(raw_name)
        if clean_cand:
            cluster.aliases.add(clean_cand)
        cluster.max_weight = max(cluster.max_weight, weight)
        if node_month:
            if not cluster.first_occurrence_month or node_month < cluster.first_occurrence_month:
                cluster.first_occurrence_month = node_month

        if attrs.get("refined"):
            cluster.already_refined = True
            if brief and not cluster.refined_brief:
                cluster.refined_brief = brief
            if detailed and not cluster.refined_detailed:
                cluster.refined_detailed = detailed
            for rel_ent in attrs.get("related_entities") or []:
                if rel_ent and rel_ent not in cluster.refined_related_entities:
                    cluster.refined_related_entities.append(rel_ent)

        # Gather chunk edges for node_id
        chunk_edges = []
        for _, target, edge_data in store.graph.out_edges(node_id, data=True):
            if edge_data.get("relation") in ("SUPPORTED_BY", "MENTIONED_IN"):
                chunk_edges.append((target, edge_data))
        for source, _, edge_data in store.graph.in_edges(node_id, data=True):
            if edge_data.get("relation") == "SUPPORTS_IDEA":
                chunk_edges.append((source, edge_data))

        seen_chunk_ids_for_node: Set[str] = set()
        for chunk_nid, edge_data in chunk_edges:
            chunk_node = store.graph.nodes.get(chunk_nid, {})
            chunk_id = str(chunk_node.get("chunk_id") or edge_data.get("chunk_id") or chunk_nid.replace("chunk:", ""))
            if chunk_id in seen_chunk_ids_for_node or chunk_id in cluster.chunk_ids:
                continue
            seen_chunk_ids_for_node.add(chunk_id)
            cluster.chunk_ids.add(chunk_id)

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

        if not seen_chunk_ids_for_node and not cluster.observations:
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

    def _merge_with_existing_graph_concepts(
        self,
        store: ConceptGraphStore,
        clusters: List[IdeaCluster],
        console: Optional[Console] = None,
    ) -> int:
        """
        In incremental/batch mode, check if any affected clusters semantically match (cosine similarity >= threshold)
        existing concept nodes in the graph that were not part of the active batch.
        If a match is found, merges the existing graph node into the affected cluster.
        Returns the number of existing graph nodes merged.
        """
        if not clusters or store.graph.number_of_nodes() == 0:
            return 0

        # Collect node IDs and alias names already included in the active clusters
        already_clustered_nids: Set[str] = set()
        already_clustered_names: Set[str] = set()
        for c in clusters:
            already_clustered_nids.update(c.existing_node_ids)
            for alias in c.aliases:
                already_clustered_names.add(alias.lower())
                already_clustered_nids.add(f"concept:{alias}")

        # Gather existing graph concepts that have embeddings and are not already in active clusters
        candidate_nids: List[str] = []
        candidate_embs: List[np.ndarray] = []

        for nid, attrs in store.graph.nodes(data=True):
            if attrs.get("type") != "Concept" and not attrs.get("is_idea"):
                continue
            if nid in already_clustered_nids:
                continue
            raw_name = (attrs.get("name") or nid.replace("concept:", "")).strip()
            if raw_name.lower() in already_clustered_names:
                continue

            emb = attrs.get("embedding")
            if emb is not None and isinstance(emb, (list, np.ndarray)) and len(emb) > 1:
                arr = np.array(emb, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 0.0:
                    candidate_nids.append(nid)
                    candidate_embs.append(arr / norm)

        if not candidate_embs:
            return 0

        G_mat = np.stack(candidate_embs, axis=0)  # Shape: (N_cand, D)
        merged_count = 0
        consumed_cand_indices: Set[int] = set()

        for cluster in clusters:
            if cluster.embedding is None or len(cluster.embedding) <= 1:
                continue

            c_emb = cluster.embedding
            sims = G_mat @ c_emb  # Shape: (N_cand,)
            matching_indices = np.where(sims >= self.similarity_threshold)[0]

            for idx in matching_indices:
                if idx in consumed_cand_indices:
                    continue
                match_nid = candidate_nids[idx]
                sim = float(sims[idx])
                logger.info(
                    f"Cross-batch semantic merge ({sim:.3f} >= {self.similarity_threshold}): "
                    f"'{match_nid}' into '{cluster.canonical_name}'"
                )
                self._merge_graph_node_into_cluster(store, cluster, match_nid)
                consumed_cand_indices.add(idx)
                G_mat[idx] = 0.0  # Zero out vector so it doesn't merge again
                merged_count += 1

        if console is not None and merged_count > 0:
            console.print(
                f"[bold green]✓[/bold green] [dim]Merged {merged_count} existing graph concepts into batch clusters (sim >= {self.similarity_threshold:.2f})[/dim]"
            )

        return merged_count

    # --------------------------------------------------------------------------
    # Step 2: Source Provenance & Observation Aggregation
    # --------------------------------------------------------------------------

    def format_consolidation_payload(
        self,
        cluster: IdeaCluster,
        max_obs: int = 3,
        max_desc_chars: int = 150,
        max_quotes: Optional[int] = None,
    ) -> Tuple[str, List[str]]:
        """
        Format chronologically linearized observations and top representative quotes
        into compact structured bullet points for the LLM.
        """
        eff_max_quotes = max_quotes if max_quotes is not None else min(self.max_quotes, 2)
        quotes = cluster.get_representative_quotes(max_quotes=eff_max_quotes)
        ordered_obs = cluster.get_linearized_observations()
        selected_obs = ordered_obs[:max_obs] if len(ordered_obs) > max_obs else ordered_obs

        bullet_points = []
        for idx, obs in enumerate(selected_obs, 1):
            pos_label = obs.position_str or f"Chunk {obs.chunk_id or idx}"
            desc = (obs.local_detailed or obs.local_brief or "Mentioned concept.").strip()
            if len(desc) > max_desc_chars:
                desc = desc[:max_desc_chars].rsplit(" ", 1)[0] + "..."
            quote_part = ""
            if obs.quote:
                q_clean = obs.quote.strip()
                if len(q_clean) > 100:
                    q_clean = q_clean[:100].rsplit(" ", 1)[0] + "..."
                quote_part = f' | Quote: "{q_clean}"'
            bullet_points.append(f"- [Pos: {pos_label}]{quote_part} Context: {desc}")

        compact_quotes = []
        for q in quotes:
            q_clean = q.strip()
            if len(q_clean) > 120:
                q_clean = q_clean[:120].rsplit(" ", 1)[0] + "..."
            compact_quotes.append(f'* "{q_clean}"')

        payload = (
            f"Concept: {cluster.canonical_name}\n"
            f"Occurrences: {cluster.total_occurrences}\n"
            f"Observations:\n"
            + ("\n".join(bullet_points) if bullet_points else "- Single mention.")
            + (("\nTop Quotes:\n" + "\n".join(compact_quotes)) if compact_quotes else "")
        )
        return payload, quotes

    # --------------------------------------------------------------------------
    # Step 3: LLM Consolidation & Synthesis (Second Pass)
    # --------------------------------------------------------------------------

    def batch_ideas(
        self,
        clusters: Sequence[IdeaCluster],
        max_context_size: Optional[int] = None,
        max_idea_batch: Optional[int] = None,
    ) -> List[List[IdeaCluster]]:
        """
        Combine idea clusters into batches up to maximum context size (default: self.max_context_size / 5KB)
        and maximum idea count (default: self.max_idea_batch / 3).
        """
        size_limit = self.max_context_size if max_context_size is None else max_context_size
        count_limit = self.max_idea_batch if max_idea_batch is None else max_idea_batch
        return batch_ideas_by_context_size(
            clusters,
            format_payload_fn=self.format_consolidation_payload,
            max_context_size=size_limit,
            max_idea_batch=count_limit,
        )

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
                    if isinstance(res, BatchConsolidatedIdeaOutput):
                        if res.ideas:
                            item = res.ideas[0]
                            return ConsolidatedIdeaOutput(
                                brief_description=item.brief_description,
                                detailed_explanation=item.detailed_explanation,
                                related_entities=item.related_entities,
                                category=item.category or cluster.category,
                                weight=item.weight if item.weight is not None else cluster.max_weight,
                            )
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
                        try:
                            from bookeeper.processing.token_tracker import TokenTracker
                            TokenTracker.get_instance().record_response_data(
                                resp_json,
                                prompt_text=f"{system_prompt}\n{user_prompt}",
                                completion_text=content,
                                task_type="llm",
                            )
                        except Exception as track_err:
                            logger.debug(f"Failed to record token usage in postprocessor: {track_err}")
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
                    quarantine_server=False,
                )
                if isinstance(result, BatchConsolidatedIdeaOutput):
                    if result.ideas:
                        item = result.ideas[0]
                        return ConsolidatedIdeaOutput(
                            brief_description=item.brief_description,
                            detailed_explanation=item.detailed_explanation,
                            related_entities=item.related_entities,
                            category=item.category or cluster.category,
                            weight=item.weight if item.weight is not None else cluster.max_weight,
                        )
                return result
            except Exception as e:
                logger.warning(
                    f"LLM synthesis failed for cluster '{cluster.canonical_name}': {e}. "
                    "Falling back to deterministic observation aggregation."
                )

        # Deterministic fallback when LLM is unavailable / failed
        return self._deterministic_consolidation(cluster, quotes)

    def synthesize_cluster_batch(
        self, clusters: List[IdeaCluster]
    ) -> Dict[str, ConsolidatedIdeaOutput]:
        """
        Submit multiple idea clusters simultaneously to consolidation model
        in a single LLM prompt with XML tags, returning a mapping of canonical_name -> ConsolidatedIdeaOutput.
        If a cluster is missing from batch output or if the call fails, gracefully falls back to individual synthesis.
        """
        if not clusters:
            return {}
        if len(clusters) == 1:
            res_single = self.synthesize_cluster(clusters[0])
            if isinstance(res_single, BatchConsolidatedIdeaOutput):
                if res_single.ideas:
                    item = res_single.ideas[0]
                    res_single = ConsolidatedIdeaOutput(
                        brief_description=item.brief_description,
                        detailed_explanation=item.detailed_explanation,
                        related_entities=item.related_entities,
                        category=item.category or clusters[0].category,
                        weight=item.weight if item.weight is not None else clusters[0].max_weight,
                    )
            return {clusters[0].canonical_name: res_single}

        idea_blocks: List[str] = []
        cluster_by_id: Dict[str, IdeaCluster] = {}
        cluster_quotes: Dict[str, List[str]] = {}

        for idx, cluster in enumerate(clusters, start=1):
            idea_id = f"idea_{idx}"
            cluster_by_id[idea_id] = cluster
            payload, quotes = self.format_consolidation_payload(cluster)
            cluster_quotes[cluster.canonical_name] = quotes
            block = (
                f'<idea id="{idea_id}" name="{cluster.canonical_name}">\n'
                f"{payload}\n"
                f"</idea>"
            )
            idea_blocks.append(block)

        system_prompt = (
            "You are a Principal Knowledge Graph Architect and Domain Ontologist.\n"
            "Your task is Second-Pass Idea Consolidation: synthesize a unified, definitive definition "
            "for multiple distinct ideas based strictly on their chronologically ordered observations "
            "and supporting quotes collected across the source material.\n"
            'Multiple ideas are provided simultaneously, each enclosed within <idea id="..." name="..."> ... </idea> tags.\n\n'
            "Requirements for each idea:\n"
            "1. Process each idea independently and return a separate consolidated result matching its idea_id.\n"
            "2. brief_description: A cohesive 1-2 sentence definition capturing the macro-identity and core essence of the idea.\n"
            "3. detailed_explanation: A focused, cohesive explanation (100-200 words) integrating distinct events, "
            "cause-and-effect chains, mechanisms, and conditions across all chunks without dropping earlier facts.\n"
            "4. related_entities: List 2-5 explicit related entities, concepts, or architectural patterns "
            "discussed across the pooled observations for that specific idea.\n"
            "5. category: Domain classification (e.g. Architectural Pattern, Principle, Mechanism, Theory, Data Structure).\n"
            "6. weight: Significance and depth weight from 0 to 10 (higher for novel/specialized architectural concepts).\n"
            "7. Return strictly valid JSON conforming to the BatchConsolidatedIdeaOutput schema."
        )

        user_prompt = (
            "Consolidate and synthesize the following ideas into unified canonical concepts:\n\n"
            + "\n\n".join(idea_blocks)
            + "\n\nRespond with valid JSON conforming to the BatchConsolidatedIdeaOutput schema."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        results: Dict[str, ConsolidatedIdeaOutput] = {}

        if self.pool and self.pool.nodes:
            def _invoke_fn(url: str) -> BatchConsolidatedIdeaOutput:
                # 1. Structured ChatOllama invoke
                try:
                    llm = ChatOllama(
                        model=self.model_name,
                        base_url=url,
                        temperature=self.temperature,
                        top_p=0.9,
                        model_kwargs={"num_predict": 8192},
                        sync_client_kwargs={"timeout": float(self.timeout)},
                        client_kwargs={"timeout": float(self.timeout)},
                    )
                    structured_llm = llm.with_structured_output(BatchConsolidatedIdeaOutput)
                    res = structured_llm.invoke(messages)
                    if isinstance(res, BatchConsolidatedIdeaOutput):
                        return res
                    if isinstance(res, dict):
                        return BatchConsolidatedIdeaOutput.model_validate(res)
                except Exception as exc:
                    logger.debug(f"Structured batch invoke failed on {url}, falling back to curl JSON: {exc}")

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
                        "options": {"temperature": self.temperature, "top_p": 0.9, "num_predict": 8192},
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
                        try:
                            from bookeeper.processing.token_tracker import TokenTracker
                            TokenTracker.get_instance().record_response_data(
                                resp_json,
                                prompt_text=f"{system_prompt}\n{user_prompt}",
                                completion_text=content,
                                task_type="llm",
                            )
                        except Exception as track_err:
                            logger.debug(f"Failed to record token usage in postprocessor batch: {track_err}")
                        try:
                            return BatchConsolidatedIdeaOutput.model_validate_json(content)
                        except Exception as parse_err:
                            salvaged = salvage_truncated_json(content)
                            if salvaged:
                                logger.info(
                                    f"Salvaged {len(salvaged.get('ideas', []))} ideas from partially truncated batch JSON on {url}"
                                )
                                return BatchConsolidatedIdeaOutput.model_validate(salvaged)
                            raise parse_err

                raise RuntimeError(f"Could not synthesize idea batch on {url}")

            try:
                input_chars = len(system_prompt) + len(user_prompt)
                batch_res = self.pool.execute_with_failover(
                    _invoke_fn,
                    retries=self.retries,
                    model_name=self.model_name,
                    capability="llm",
                    input_chars=input_chars,
                    task_type="llm",
                    quarantine_server=False,
                )
                if isinstance(batch_res, BatchConsolidatedIdeaOutput):
                    model_ideas: Dict[str, ConsolidatedIdeaBatchItem] = {}
                    for item in batch_res.ideas:
                        if item.idea_id:
                            model_ideas[str(item.idea_id).strip()] = item

                    for idx_pos, (idea_id, cluster) in enumerate(cluster_by_id.items(), start=1):
                        matching_item = model_ideas.get(idea_id)
                        if not matching_item:
                            # Try fuzzy match by idea_id, integer index, or cluster canonical_name
                            for m_id, m_val in model_ideas.items():
                                if m_id in (idea_id, idea_id.replace("idea_", ""), str(idx_pos), cluster.canonical_name):
                                    matching_item = m_val
                                    break
                                if m_val.canonical_name and m_val.canonical_name.lower() == cluster.canonical_name.lower():
                                    matching_item = m_val
                                    break
                        if not matching_item and idx_pos <= len(batch_res.ideas):
                            matching_item = batch_res.ideas[idx_pos - 1]
                        if matching_item:
                            results[cluster.canonical_name] = ConsolidatedIdeaOutput(
                                brief_description=matching_item.brief_description,
                                detailed_explanation=matching_item.detailed_explanation,
                                related_entities=matching_item.related_entities,
                                category=matching_item.category or cluster.category,
                                weight=matching_item.weight if matching_item.weight is not None else cluster.max_weight,
                            )
                elif isinstance(batch_res, ConsolidatedIdeaOutput):
                    results[clusters[0].canonical_name] = batch_res
            except Exception as e:
                logger.warning(f"Batch idea synthesis LLM call failed across pool: {e}. Falling back to individual synthesis.")

        # For any cluster missing from batch results, fall back to individual synthesis
        for cluster in clusters:
            if cluster.canonical_name not in results:
                try:
                    res_indiv = self.synthesize_cluster(cluster)
                    if isinstance(res_indiv, BatchConsolidatedIdeaOutput):
                        if res_indiv.ideas:
                            item = res_indiv.ideas[0]
                            res_indiv = ConsolidatedIdeaOutput(
                                brief_description=item.brief_description,
                                detailed_explanation=item.detailed_explanation,
                                related_entities=item.related_entities,
                                category=item.category or cluster.category,
                                weight=item.weight if item.weight is not None else cluster.max_weight,
                            )
                    results[cluster.canonical_name] = res_indiv
                except Exception as fallback_err:
                    logger.warning(f"Fallback synthesis failed for '{cluster.canonical_name}': {fallback_err}")
                    results[cluster.canonical_name] = self._deterministic_consolidation(
                        cluster, cluster_quotes.get(cluster.canonical_name, [])
                    )

        return results

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
        affected_concepts: Optional[Set[str]] = None,
    ) -> IdeaPostProcessorReport:
        """
        Execute two-pass aggregation and synthesis pipeline on the graph:
        1. Group ideas deterministically by normalized name.
        2. Merge clusters by semantic similarity (>= 0.92).
        3. Filter out already post-processed and untouched clusters so only affected
           clusters are processed/ingested, ensuring LLM refinement runs at most once per idea.
        4. Overwrite/upsert canonical concept nodes and provenance/relational edges for affected clusters.
        Supports state file checkpointing (save every N ideas) and auto-continue.

        If affected_concepts is provided (e.g. during inter-batch execution), post-processing
        is strictly scoped to the touched concepts and their candidate merge partners in the graph,
        bypassing the expensive full graph re-scan.
        """
        t0 = time.time()
        t_start = time.perf_counter()

        if affected_concepts is not None and len(affected_concepts) == 0:
            return IdeaPostProcessorReport(
                total_input_concepts=0,
                total_clusters=0,
                multi_occurrence_clusters_count=0,
                single_occurrence_clusters_count=0,
                merged_clusters_count=0,
                edges_added_count=0,
                synthesized_ideas=[],
                bypassed_ideas=[],
                duration_seconds=0.0,
            )

        # 1. Deterministic normalization grouping
        clusters_by_norm = self.collect_observations_from_graph(
            store,
            console=console,
            affected_concepts=affected_concepts,
        )
        initial_clusters = list(clusters_by_norm.values())
        total_input_concepts = sum(len(c.aliases) for c in initial_clusters)

        # 2. Semantic similarity clustering (across candidate clusters)
        merged_clusters = self.cluster_by_semantic_similarity(initial_clusters, console=console)
        merged_diff = len(initial_clusters) - len(merged_clusters)

        # If running with affected_concepts, check if any affected clusters merge with existing graph concepts
        if affected_concepts is not None and store.graph.number_of_nodes() > 0:
            cross_merged = self._merge_with_existing_graph_concepts(store, merged_clusters, console=console)
            merged_diff += cross_merged

        total_clusters = len(merged_clusters)

        target_multi = [c for c in merged_clusters if c.total_occurrences >= self.min_occurrences]
        target_single = [c for c in merged_clusters if c.total_occurrences < self.min_occurrences]

        # Checkpoint state manager setup
        state_manager: Optional[IdeaPostProcessorState] = None
        resumed_count = 0
        if self.state_file:
            state_manager = IdeaPostProcessorState(self.state_file, async_save=self.async_save)
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
            scope_line = (
                f"• Batch Affected Concepts:   [bold]{total_input_concepts}[/bold] ({len(affected_concepts)} candidates)\n"
                if affected_concepts is not None
                else f"• Total Concepts in Graph:   [bold]{total_input_concepts}[/bold]\n"
            )
            panel_title = "Batch Idea Postprocessor" if affected_concepts is not None else "Idea Postprocessor"
            header_text = (
                f"[bold cyan]🧠 Two-Pass Idea Aggregation & Synthesis (Pass 2)[/bold cyan]\n"
                f"• Consolidation Model:       [bold green]{self.model_name}[/bold green]\n"
                f"{scope_line}"
                f"• Resolved Entity Clusters:  [bold]{total_clusters}[/bold] (Merged: [yellow]{merged_diff}[/yellow] via cosine sim >= {self.similarity_threshold})\n"
                f"• Multi-Occurrence Ideas:    [bold green]{len(target_multi)}[/bold green] (filtered in for LLM consolidation, >= {self.min_occurrences} mentions)\n"
                f"• Single-Occurrence Ideas:   [dim]{len(target_single)} (filtered out / bypassed to conserve compute)[/dim]\n"
                f"• Cosine Similarity Thresh:  [bold yellow]{self.similarity_threshold}[/bold yellow]"
                f"{already_proc_info}"
                f"{checkpoint_info}"
                f"{resumed_info}"
            )
            console.print(Panel.fit(header_text, title=panel_title, border_style="cyan"))
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

        # Partition affected clusters into:
        # 1. Single-occurrence (bypassed, 0 LLM calls needed)
        # 2. Already refined / cached (resumed, 0 LLM calls needed)
        # 3. Needing synthesis (multi-occurrence >= min_occurrences, batched LLM calls)
        bypassed_clusters: List[IdeaCluster] = []
        cached_clusters: List[IdeaCluster] = []
        clusters_needing_synthesis: List[IdeaCluster] = []

        for c in affected_clusters:
            if c.total_occurrences < self.min_occurrences and not self._is_cluster_already_refined(c, state_manager):
                bypassed_clusters.append(c)
            elif self._is_cluster_already_refined(c, state_manager):
                cached_clusters.append(c)
            else:
                clusters_needing_synthesis.append(c)

        completed = 0

        try:
            # Phase 1: Fast in-memory resolution of single-occurrence & cached clusters
            for c in bypassed_clusters:
                output, eff_is_multi, was_resumed = self._process_single_cluster(c, state_manager)
                edges_added = self.apply_consolidation_to_graph(store, c, output)
                total_edges_added += edges_added
                single_occ_count += 1
                bypassed_names.append(c.canonical_name)
                if was_resumed:
                    actual_resumed_count += 1
                completed += 1
                if completed % 10 == 0 or completed == num_affected:
                    pct = (completed / num_affected * 100) if num_affected > 0 else 100.0
                    if console is not None:
                        console.print(
                            f"[bold cyan][Idea Postprocessor][/bold cyan] {completed}/{num_affected} ideas ({pct:.1f}%)"
                            f" | [dim]{single_occ_count} bypassed[/dim] | +{total_edges_added} edges"
                        )
                if on_progress:
                    on_progress(completed, num_affected, c.canonical_name, {
                        "synthesized": synthesized_count,
                        "bypassed": single_occ_count,
                        "resumed": actual_resumed_count,
                        "edges_added": total_edges_added,
                    })

            for c in cached_clusters:
                output, eff_is_multi, was_resumed = self._process_single_cluster(c, state_manager)
                edges_added = self.apply_consolidation_to_graph(store, c, output)
                total_edges_added += edges_added
                actual_resumed_count += 1
                if eff_is_multi:
                    synthesized_count += 1
                    multi_occ_count += 1
                    synthesized_names.append(c.canonical_name)
                else:
                    single_occ_count += 1
                    bypassed_names.append(c.canonical_name)
                completed += 1
                if completed % 10 == 0 or completed == num_affected:
                    pct = (completed / num_affected * 100) if num_affected > 0 else 100.0
                    if console is not None:
                        console.print(
                            f"[bold cyan][Idea Postprocessor][/bold cyan] {completed}/{num_affected} ideas ({pct:.1f}%)"
                            f" | [bold green]✓ {synthesized_count} synthesized[/bold green]"
                            f" | [dim]{single_occ_count} bypassed[/dim] | +{total_edges_added} edges"
                        )
                if on_progress:
                    on_progress(completed, num_affected, c.canonical_name, {
                        "synthesized": synthesized_count,
                        "bypassed": single_occ_count,
                        "resumed": actual_resumed_count,
                        "edges_added": total_edges_added,
                    })

            if console is not None and (bypassed_clusters or cached_clusters):
                console.print(
                    f"  [dim green]⚡ Instant resolution: {len(bypassed_clusters)} single-occurrence ideas bypassed, "
                    f"{len(cached_clusters)} cached ideas reused (+{total_edges_added} edges).[/dim green]"
                )

            # Phase 2: Batch Consolidation for multi-occurrence clusters needing synthesis
            synthesis_batches = self.batch_ideas(clusters_needing_synthesis)
            total_batches = len(synthesis_batches)
            total_needing_synthesis = len(clusters_needing_synthesis)

            if total_needing_synthesis > 0:
                t_synth_start = time.perf_counter()
                synth_completed = 0

                if console is not None:
                    progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[bold cyan]Consolidation Progress:[/bold cyan]"),
                        BarColumn(bar_width=25),
                        TaskProgressColumn(),
                        MofNCompleteColumn(),
                        TextColumn("{task.fields[stats_line]}"),
                        console=console,
                    )
                    synth_task = progress.add_task(
                        "synthesize",
                        total=total_needing_synthesis,
                        completed=0,
                        stats_line="",
                    )
                    first_b = synthesis_batches[0]
                    first_b_names = ", ".join(f"'{b_item.canonical_name}'" for b_item in first_b[:3])
                    first_more = f" (+{len(first_b)-3} more)" if len(first_b) > 3 else ""
                    init_text = (
                        f"[bold blue]Batch 1/{total_batches} ({len(first_b)} ideas):[/bold blue] {first_b_names}{first_more} | "
                        f"[bold yellow]Consolidating with {self.model_name}...[/bold yellow]"
                    )
                    cur_line = Text.from_markup(init_text)

                    with Live(Group(cur_line, progress), console=console, refresh_per_second=10) as live:
                        for b_idx, batch in enumerate(synthesis_batches, 1):
                            batch_names = ", ".join(f"'{b_item.canonical_name}'" for b_item in batch[:3])
                            more_str = f" (+{len(batch)-3} more)" if len(batch) > 3 else ""
                            total_bytes = sum(len(self.format_consolidation_payload(c)[0].encode("utf-8")) + 150 for c in batch)
                            batch_kb = total_bytes / 1024.0

                            cur_text = (
                                f"[bold blue]Batch {b_idx}/{total_batches} ({len(batch)} ideas, {batch_kb:.1f}KB):[/bold blue] {batch_names}{more_str} | "
                                f"[bold yellow]Consolidating with {self.model_name}...[/bold yellow]"
                            )
                            live.update(Group(Text.from_markup(cur_text), progress))

                            t_b_start = time.perf_counter()
                            batch_results = self.synthesize_cluster_batch(batch)
                            b_dur = time.perf_counter() - t_b_start

                            for c in batch:
                                output = batch_results.get(c.canonical_name) or self.synthesize_cluster(c)
                                c.already_refined = True
                                if state_manager:
                                    state_manager.record_processed(c, output, status="synthesized")
                                edges_added = self.apply_consolidation_to_graph(store, c, output)
                                total_edges_added += edges_added
                                synthesized_count += 1
                                multi_occ_count += 1
                                synthesized_names.append(c.canonical_name)
                                completed += 1

                                if completed % 10 == 0 or completed == num_affected:
                                    pct = (completed / num_affected * 100) if num_affected > 0 else 100.0
                                    console.print(
                                        f"[bold cyan][Idea Postprocessor][/bold cyan] {completed}/{num_affected} ideas ({pct:.1f}%)"
                                        f" | [bold green]✓ {synthesized_count} synthesized[/bold green]"
                                        f" | [dim]{single_occ_count} bypassed[/dim] | +{total_edges_added} edges"
                                    )

                                if on_progress:
                                    on_progress(completed, num_affected, c.canonical_name, {
                                        "synthesized": synthesized_count,
                                        "bypassed": single_occ_count,
                                        "resumed": actual_resumed_count,
                                        "edges_added": total_edges_added,
                                    })

                            synth_completed += len(batch)
                            elapsed_synth = time.perf_counter() - t_synth_start
                            rate = (synth_completed / elapsed_synth) if elapsed_synth > 0 else 0.0
                            rem_sec = ((total_needing_synthesis - synth_completed) / rate) if rate > 0 and total_needing_synthesis > synth_completed else 0.0
                            eta_str = _format_eta(rem_sec) if synth_completed < total_needing_synthesis else "0m 00s"
                            rate_str = f"{rate:.1f} ideas/s" if rate > 0 else "-- ideas/s"
                            stats_str = (
                                f" | [bold yellow]{rate_str}[/bold yellow] | ETA: {eta_str} | "
                                f"[bold green]✓ {synth_completed}/{total_needing_synthesis} synthesized[/bold green] (Batch {b_idx}/{total_batches})"
                            )
                            progress.update(synth_task, completed=synth_completed, total=total_needing_synthesis, stats_line=stats_str)

                            srv = self.pool.get_last_used_server() or "ollama" if self.pool else "ollama"
                            b_rate = (len(batch) / b_dur) if b_dur > 0 else 0.0
                            finish_text = (
                                f"  [dim green]✓ [Batch {b_idx}/{total_batches} ({len(batch)} ideas, {batch_kb:.1f}KB)][/dim green] "
                                f"Synthesized [bold cyan]{len(batch)} ideas[/bold cyan] [dim]({b_dur:.1f}s, {b_rate:.1f} ideas/s via [bold green]{srv}[/bold green])[/dim]"
                            )
                            live.update(Group(Text.from_markup(finish_text), progress))
                            console.print(finish_text)

                            # Checkpoint save
                            if state_manager and (synth_completed % self.checkpoint_interval == 0 or synth_completed == total_needing_synthesis):
                                if synth_completed == total_needing_synthesis:
                                    state_manager.flush()
                                else:
                                    state_manager.save_async()
                else:
                    for b_idx, batch in enumerate(synthesis_batches, 1):
                        t_b_start = time.perf_counter()
                        batch_results = self.synthesize_cluster_batch(batch)
                        b_dur = time.perf_counter() - t_b_start

                        for c in batch:
                            output = batch_results.get(c.canonical_name) or self.synthesize_cluster(c)
                            c.already_refined = True
                            if state_manager:
                                state_manager.record_processed(c, output, status="synthesized")
                            edges_added = self.apply_consolidation_to_graph(store, c, output)
                            total_edges_added += edges_added
                            synthesized_count += 1
                            multi_occ_count += 1
                            synthesized_names.append(c.canonical_name)
                            completed += 1

                            if completed % 10 == 0 or completed == num_affected:
                                pct = (completed / num_affected * 100) if num_affected > 0 else 100.0
                                logger.info(
                                    f"[Idea Postprocessor] {completed}/{num_affected} affected ideas ({pct:.1f}%)"
                                    f" | Synthesized: {synthesized_count} | Bypassed: {single_occ_count} | Resumed: {actual_resumed_count} | Edges: +{total_edges_added}"
                                )

                            if on_progress:
                                on_progress(completed, num_affected, c.canonical_name, {
                                    "synthesized": synthesized_count,
                                    "bypassed": single_occ_count,
                                    "resumed": actual_resumed_count,
                                    "edges_added": total_edges_added,
                                })

                        synth_completed += len(batch)
                        if state_manager and (synth_completed % self.checkpoint_interval == 0 or synth_completed == total_needing_synthesis):
                            if synth_completed == total_needing_synthesis:
                                state_manager.flush()
                            else:
                                state_manager.save_async()

                        logger.info(
                            f"[Idea Postprocessor] Batch {b_idx}/{total_batches} ({len(batch)} ideas) in {b_dur:.1f}s: "
                            f"{synth_completed}/{total_needing_synthesis} synthesized | +{total_edges_added} edges"
                        )
            elif console is not None and num_affected == 0:
                console.print(
                    f"[bold cyan][Idea Postprocessor][/bold cyan] All {total_clusters} ideas already post-processed and up-to-date "
                    f"([cyan]{actual_resumed_count} resumed[/cyan], 0 affected ideas to re-ingest, +0 edges)."
                )
        finally:
            if state_manager:
                state_manager.close()

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
