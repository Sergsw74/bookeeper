"""
Knowledge graph factual verification and discrepancy audit engine.
Audits extracted ideas and concepts against source text chunks using a dedicated LLM model.
"""

import json
import logging
import math
import random
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import ChunkStore, HierarchicalChunk
from bookeeper.processing.ollama_pool import OllamaPool

logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    """Structured evaluation output from the dedicated verifier LLM."""

    is_supported: bool = Field(
        description="True if the text chunk directly or inferentially contains, discusses, or exemplifies this idea/concept; False otherwise."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    explanation: str = Field(
        description="Clear, concise explanation (1-2 sentences) of why the chunk supports the idea or why it represents a discrepancy."
    )


class VerificationItem(BaseModel):
    """Detailed record of an evaluated idea-chunk pairing."""

    idea_name: str
    idea_category: str = "Concept"
    idea_summary: str = ""
    idea_weight: int = 5
    chunk_id: str
    book_id: Optional[int] = None
    book_title: str = "Unknown"
    section_title: str = "Unknown"
    quote: str = ""
    chunk_snippet: str = ""
    is_supported: bool
    confidence: float
    explanation: str


class VerificationStats(BaseModel):
    """Summary metrics of the verification run."""

    total_ideas_in_graph: int
    candidate_ideas_with_chunks: int
    sampled_ideas: int
    sample_percentage: float
    mode: str
    model_name: str
    total_evaluations: int
    verified_count: int  # Good / Pass
    discrepancy_count: int  # Failed
    verified_percentage: float
    discrepancy_percentage: float
    total_duration_seconds: float


class VerificationReport(BaseModel):
    """Complete verification report containing statistics and discrepancy records."""

    stats: VerificationStats
    discrepancies: List[VerificationItem]
    verified_samples: List[VerificationItem]


def _clean_json_str(text: str) -> str:
    """Clean model response removing thinking tags and markdown backticks."""
    text = text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


class IdeaVerifier:
    """
    Dedicated verifier using an Ollama pool to audit whether assigned chunks
    genuinely support extracted ideas/concepts.
    """

    def __init__(
        self,
        pool: OllamaPool,
        model_name: str = "llama3.1:8b",
        temperature: float = 0.0,
        timeout: int = 30,
        retries: int = 3,
    ):
        self.pool = pool
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
    ) -> "IdeaVerifier":
        """Instantiate IdeaVerifier using Ollama pool configured for verification capability."""
        servers = getattr(settings, "resolved_verification_servers", None) or getattr(settings, "resolved_ollama_servers", [])
        cooldown = getattr(settings, "failover_cooldown_seconds", 600)
        pool = OllamaPool(
            servers=servers,
            cooldown_seconds=cooldown,
            max_tasks_per_server=1,
        )
        verif_cfg = getattr(settings, "verification", None)
        eff_model = model_name or getattr(settings, "resolved_verifier_model", "llama3.1:8b")
        eff_temp = temperature if temperature is not None else (getattr(verif_cfg, "temperature", 0.0) if verif_cfg else 0.0)
        eff_timeout = timeout or getattr(settings, "request_timeout", 30)
        eff_retries = retries or getattr(settings, "max_retries", 3)
        return cls(
            pool=pool,
            model_name=eff_model,
            temperature=eff_temp,
            timeout=eff_timeout,
            retries=eff_retries,
        )

    def warmup(self, timeout: int = 30) -> None:
        """Ping Ollama server pool nodes supporting verification to preload model into VRAM."""
        target_nodes = [n for n in self.pool.nodes if n.has_capability("verification")] or self.pool.nodes
        for node in target_nodes:
            if shutil.which("curl"):
                try:
                    cmd = [
                        "curl",
                        "-s",
                        "--max-time",
                        str(timeout),
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        json.dumps({
                            "model": self.model_name,
                            "prompt": "ping",
                            "options": {"num_predict": 1},
                            "stream": False,
                        }),
                        f"{node.url.rstrip('/')}/api/generate",
                    ]
                    subprocess.run(cmd, capture_output=True, text=True, check=False)
                except Exception as e:
                    logger.debug(f"Warmup ping error for {node.url}: {e}")

    def _execute_structured_invoke(self, messages: List[Any]) -> VerificationResult:
        """
        Execute structured LLM invoke with automatic pool failover and curl fallback
        to ensure resilience against network timeouts and macOS Local Network Privacy.
        """
        def _invoke(url: str) -> VerificationResult:
            try:
                # Clean sampling parameters: pass only valid parameters (no mirostat, mirostat_eta, mirostat_tau, or tfs_z)
                llm = ChatOllama(
                    model=self.model_name,
                    base_url=url,
                    temperature=self.temperature,
                    top_p=0.9,
                    sync_client_kwargs={"timeout": float(self.timeout)},
                    client_kwargs={"timeout": float(self.timeout)},
                )
                structured_llm = llm.with_structured_output(VerificationResult)
                res = structured_llm.invoke(messages)
                if isinstance(res, VerificationResult):
                    return res
                if isinstance(res, dict):
                    return VerificationResult.model_validate(res)
                cleaned = _clean_json_str(str(res))
                return VerificationResult.model_validate_json(cleaned)
            except Exception as exc:
                # Resilient fallback via curl for macOS Sequoia LAN routing
                if shutil.which("curl"):
                    msgs_payload = []
                    for m in messages:
                        role = "user"
                        if hasattr(m, "type"):
                            role = "system" if m.type == "system" else "user"
                        content = getattr(m, "content", str(m))
                        msgs_payload.append({"role": role, "content": content})

                    schema_dict = VerificationResult.model_json_schema()
                    body = {
                        "model": self.model_name,
                        "messages": msgs_payload,
                        "format": schema_dict,
                        "options": {
                            "temperature": self.temperature,
                            "top_p": 0.9,
                        },
                        "stream": False,
                    }
                    cmd = [
                        "curl",
                        "-s",
                        "--max-time",
                        str(self.timeout),
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        json.dumps(body),
                        f"{url.rstrip('/')}/api/chat",
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if res.returncode == 0 and res.stdout.strip():
                        data = json.loads(res.stdout)
                        if "error" in data:
                            raise RuntimeError(f"Ollama server {url} error: {data['error']}")
                        content = data.get("message", {}).get("content", "")
                        if not content and "thinking" in data.get("message", {}):
                            content = data["message"]["thinking"]
                        cleaned = _clean_json_str(content)
                        if cleaned:
                            return VerificationResult.model_validate_json(cleaned)
                        raise ValueError(f"Ollama server {url} returned empty content")
                raise exc

        return self.pool.execute_with_failover(_invoke, retries=self.retries, capability="verification")

    def verify_idea_chunk(
        self,
        idea_name: str,
        idea_category: str,
        idea_summary: str,
        quote: str,
        book_title: str,
        section_title: str,
        chunk_text: str,
    ) -> VerificationResult:
        """
        Verify whether the assigned text chunk contains or supports the given idea.
        """
        system_prompt = (
            "You are a strict, objective knowledge graph auditor and fact-checking specialist.\n"
            "Your task is to verify whether an assigned text chunk from a book genuinely contains, discusses, "
            "exemplifies, or directly supports an extracted concept/idea.\n\n"
            "EVALUATION CRITERIA:\n"
            "1. DIRECT SUBSTANTIVE EVIDENCE:\n"
            "   - Set `is_supported: true` (Good) if the text chunk directly mentions, explains, defines, "
            "or clearly explores and substantiates the concept.\n"
            "   - Set `is_supported: false` (Failed / Discrepancy) if the concept is completely absent, "
            "fabricated, or not grounded in this specific text chunk.\n\n"
            "2. DETECT OVER-ABSTRACTION & CATEGORY MISMATCH:\n"
            "   - Set `is_supported: false` if mundane story events, casual dialogue banter, or routine physical actions "
            "have been artificially elevated into formal engineering principles, scientific laws, or abstract theoretical frameworks.\n"
            "   - Generic Example: Claiming a routine character mishap or slip is a 'System Design Principle' like 'Human Error' "
            "or 'Fault Tolerance', or claiming casual greeting dialogue is a 'Communication Protocol'. "
            "Such conceptual over-generalizations must be marked `is_supported: false`.\n"
            "   - Set `is_supported: false` if the excerpt describes unrelated interactions while the concept is nowhere to be found.\n\n"
            "3. SUPPORTING QUOTE GROUNDING:\n"
            "   - If a Supporting Quote is provided, check whether it actually exists within the assigned text chunk. "
            "If the quote is missing from this chunk (e.g., extracted from another passage or made up) and the chunk text itself "
            "does not substantiate the idea, mark `is_supported: false`.\n\n"
            "4. OBJECTIVE JUSTIFICATION:\n"
            "   - In `explanation`, provide a concise 1-2 sentence objective justification. "
            "If unsupported, state what the chunk is actually about and specifically why the concept is absent or unjustified.\n"
            "   - In `confidence`, provide a score from 0.0 to 1.0 reflecting your verification certainty."
        )

        user_content = (
            f"Target Idea/Concept:\n"
            f"- Name: \"{idea_name}\"\n"
            f"- Category: \"{idea_category}\"\n"
            f"- Summary: \"{idea_summary}\"\n"
        )
        if quote:
            user_content += f"- Supporting Quote: \"{quote}\"\n"

        user_content += (
            f"\nAssigned Text Chunk (from '{book_title}', Section '{section_title}'):\n"
            f"\"\"\"\n{chunk_text[:3500]}\n\"\"\"\n\n"
            f"Does this assigned text chunk genuinely contain or support the idea \"{idea_name}\"?"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            return self._execute_structured_invoke(messages)
        except Exception as e:
            logger.warning(f"Verification call failed for '{idea_name}' on chunk: {e}")
            return VerificationResult(
                is_supported=False,
                confidence=0.0,
                explanation=f"LLM verification error across server pool: {e}",
            )


def verify_graph(
    store: ConceptGraphStore,
    pool: OllamaPool,
    model_name: str = "llama3.1:8b",
    percent: float = 1.0,
    mode: str = "ideas",
    max_examples: int = 20,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    concurrency: int = 3,
    timeout: int = 300,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> VerificationReport:
    """
    Perform factual verification of ideas in the Knowledge Graph against their assigned chunks.
    Randomly samples `percent`% of ideas, pulls their related chunks, and audits grounding.
    """
    start_time = time.time()
    verifier = IdeaVerifier(pool=pool, model_name=model_name, temperature=temperature, timeout=timeout)

    # 1. Pull all Concept nodes
    all_concept_nodes = [
        (n, d)
        for n, d in store.graph.nodes(data=True)
        if d.get("type") == "Concept"
    ]
    total_ideas = len(all_concept_nodes)

    # 2. Build index of candidate ideas that have associated chunks
    # We look for direct SUPPORTED_BY edges to Chunk nodes
    candidates: List[Tuple[str, Dict[str, Any], List[Tuple[str, Dict[str, Any], Dict[str, Any]]]]] = []

    for c_id, c_attrs in all_concept_nodes:
        # Check direct outgoing SUPPORTED_BY edges: (:Concept) -[:SUPPORTED_BY]-> (:Chunk)
        chunks: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for _, target, edge_data in store.graph.out_edges(c_id, data=True):
            if edge_data.get("relation") == "SUPPORTED_BY" and store.graph.nodes[target].get("type") == "Chunk":
                chunks.append((target, store.graph.nodes[target], edge_data))

        # Check incoming SUPPORTS_IDEA edges: (:Chunk) -[:SUPPORTS_IDEA]-> (:Concept)
        for src, _, edge_data in store.graph.in_edges(c_id, data=True):
            if edge_data.get("relation") == "SUPPORTS_IDEA" and store.graph.nodes[src].get("type") == "Chunk":
                if not any(c[0] == src for c in chunks):
                    chunks.append((src, store.graph.nodes[src], edge_data))

        if chunks:
            candidates.append((c_id, c_attrs, chunks))

    candidate_count = len(candidates)
    if candidate_count == 0:
        # Return empty report if no ideas have supporting chunks
        stats = VerificationStats(
            total_ideas_in_graph=total_ideas,
            candidate_ideas_with_chunks=0,
            sampled_ideas=0,
            sample_percentage=percent,
            mode=mode,
            model_name=model_name,
            total_evaluations=0,
            verified_count=0,
            discrepancy_count=0,
            verified_percentage=0.0,
            discrepancy_percentage=0.0,
            total_duration_seconds=round(time.time() - start_time, 2),
        )
        return VerificationReport(stats=stats, discrepancies=[], verified_samples=[])

    # 3. Randomly select X percent of candidate ideas
    clamped_percent = max(0.01, min(100.0, percent))
    sample_size = max(1, int(round(candidate_count * (clamped_percent / 100.0))))
    sample_size = min(sample_size, candidate_count)

    rng = random.Random(seed) if seed is not None else random.Random()
    sampled_candidates = rng.sample(candidates, sample_size)

    # 4. Gather all idea-chunk tasks
    eval_tasks: List[Dict[str, Any]] = []
    for c_id, c_attrs, chunks in sampled_candidates:
        c_name = c_attrs.get("name", c_id.replace("concept:", ""))
        c_cat = c_attrs.get("category", "Concept")
        c_sum = c_attrs.get("summary") or c_attrs.get("brief_description", "")
        c_wt = c_attrs.get("weight", 5)

        for chunk_node_id, chunk_attrs, edge_data in chunks:
            quote = edge_data.get("quote", "")
            chunk_text = chunk_attrs.get("text", "")
            if not chunk_text and quote:
                chunk_text = quote

            eval_tasks.append({
                "idea_name": c_name,
                "idea_category": c_cat,
                "idea_summary": c_sum,
                "idea_weight": c_wt,
                "quote": quote,
                "chunk_id": chunk_attrs.get("chunk_id", chunk_node_id.replace("chunk:", "")),
                "book_id": chunk_attrs.get("book_id"),
                "book_title": chunk_attrs.get("book_title", "Unknown"),
                "section_title": chunk_attrs.get("section_title", chunk_attrs.get("chapter_title", "Unknown")),
                "chunk_text": chunk_text,
            })

    total_evals = len(eval_tasks)
    verified_samples: List[VerificationItem] = []
    discrepancies: List[VerificationItem] = []

    def _eval_single(task_item: Dict[str, Any]) -> VerificationItem:
        res = verifier.verify_idea_chunk(
            idea_name=task_item["idea_name"],
            idea_category=task_item["idea_category"],
            idea_summary=task_item["idea_summary"],
            quote=task_item["quote"],
            book_title=task_item["book_title"],
            section_title=task_item["section_title"],
            chunk_text=task_item["chunk_text"],
        )
        return VerificationItem(
            idea_name=task_item["idea_name"],
            idea_category=task_item["idea_category"],
            idea_summary=task_item["idea_summary"],
            idea_weight=task_item["idea_weight"],
            chunk_id=task_item["chunk_id"],
            book_id=task_item["book_id"],
            book_title=task_item["book_title"],
            section_title=task_item["section_title"],
            quote=task_item["quote"],
            chunk_snippet=task_item["chunk_text"][:250].strip() + ("..." if len(task_item["chunk_text"]) > 250 else ""),
            is_supported=res.is_supported,
            confidence=res.confidence,
            explanation=res.explanation,
        )

    # 5. Execute evaluations with worker concurrency
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {executor.submit(_eval_single, t): t for t in eval_tasks}
        for fut in as_completed(futures):
            item = fut.result()
            completed_count += 1
            if item.is_supported:
                verified_samples.append(item)
            else:
                discrepancies.append(item)

            if progress_callback:
                progress_callback(completed_count, total_evals, item.idea_name)

    # 6. Calculate statistics
    duration = round(time.time() - start_time, 2)
    verified_cnt = len(verified_samples)
    discrepancy_cnt = len(discrepancies)
    ver_pct = round((verified_cnt / total_evals * 100.0), 2) if total_evals > 0 else 0.0
    disc_pct = round((discrepancy_cnt / total_evals * 100.0), 2) if total_evals > 0 else 0.0

    stats = VerificationStats(
        total_ideas_in_graph=total_ideas,
        candidate_ideas_with_chunks=candidate_count,
        sampled_ideas=sample_size,
        sample_percentage=percent,
        mode=mode,
        model_name=model_name,
        total_evaluations=total_evals,
        verified_count=verified_cnt,
        discrepancy_count=discrepancy_cnt,
        verified_percentage=ver_pct,
        discrepancy_percentage=disc_pct,
        total_duration_seconds=duration,
    )

    # Limit discrepancies in the output report to max_examples
    limited_discrepancies = discrepancies[:max_examples]

    return VerificationReport(
        stats=stats,
        discrepancies=limited_discrepancies,
        verified_samples=verified_samples[:max_examples],
    )


class DistanceBucket(BaseModel):
    """Histogram bucket representing a range of sentence-expansion semantic distances."""

    range_label: str
    coherence_level: str
    count: int
    percentage: float
    bar: str


class ChunkingDiscrepancy(BaseModel):
    """Record of a chunking discrepancy (internal semantic drift or boundary mismatch)."""

    chunk_id: str
    book_id: int
    book_title: str
    section_title: str
    chunk_idx: int
    discrepancy_type: str  # "internal_semantic_drift" or "boundary_mismatch"
    max_distance: float
    threshold: float
    drift_sentence: str = ""
    prior_text_snippet: str = ""
    explanation: str


class ChunkingVerificationStats(BaseModel):
    """Statistical distribution of smart chunking verification across a book."""

    book_id: int
    book_title: str
    total_chunks: int
    total_sentences: int
    total_transitions: int
    coherent_chunks: int
    divergent_chunks: int
    coherence_rate: float
    boundary_match_rate: float
    exact_matches_count: int
    reconstructed_chunks_count: int
    actual_chunks_count: int
    mean_distance: float
    median_distance: float
    min_distance: float
    max_distance: float
    p80_distance: float
    p95_distance: float
    std_dev_distance: float
    distance_threshold: float
    distribution_buckets: List[DistanceBucket]
    total_duration_seconds: float


class ChunkingVerificationReport(BaseModel):
    """Complete report for chunking verification mode."""

    mode: str = "chunking"
    embedding_model: str
    distance_threshold: float
    stats: ChunkingVerificationStats
    discrepancies: List[ChunkingDiscrepancy]
    all_books_summary: Optional[List[ChunkingVerificationStats]] = None


def cosine_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine distance (1.0 - cosine_similarity) between two vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 1.0
    similarity = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    similarity = max(-1.0, min(1.0, similarity))
    return max(0.0, 1.0 - similarity)


def split_sentences(text: str) -> List[str]:
    """Split text into clean, non-empty sentences respecting punctuation and whitespace."""
    if not text or not text.strip():
        return []
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text.strip()]


class SmartSentenceChunker:
    """
    Expands chunk borders sentence by sentence using embedding distance thresholds.
    A chunk remains whole as long as the semantic distance within these sentences remains within accepted thresholds.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        distance_threshold: float = 0.20,
        max_chunk_chars: int = 2500,
        min_chunk_chars: int = 120,
    ):
        self.embeddings = embeddings
        self.distance_threshold = distance_threshold
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_text(self, text: str) -> List[str]:
        """
        Segment text into smart semantic chunks using sentence-level incremental expansion.
        """
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        chunks: List[str] = []
        current_sentences = [sentences[0]]
        current_text = sentences[0]
        current_emb = self.embeddings.embed_query(current_text)

        for i in range(1, len(sentences)):
            next_sent = sentences[i]
            candidate_text = current_text + " " + next_sent

            # Cap chunk size
            if len(candidate_text) > self.max_chunk_chars and len(current_text) >= self.min_chunk_chars:
                chunks.append(current_text)
                current_sentences = [next_sent]
                current_text = next_sent
                current_emb = self.embeddings.embed_query(current_text)
                continue

            candidate_emb = self.embeddings.embed_query(candidate_text)
            dist = cosine_distance(current_emb, candidate_emb)

            if dist <= self.distance_threshold:
                # Semantic distance remains within accepted threshold -> expand chunk borders
                current_sentences.append(next_sent)
                current_text = candidate_text
                current_emb = candidate_emb
            else:
                # Semantic distance exceeded threshold -> semantic shift detected, finalize chunk
                chunks.append(current_text)
                current_sentences = [next_sent]
                current_text = next_sent
                current_emb = self.embeddings.embed_query(current_text)

        if current_text:
            chunks.append(current_text)

        return chunks


class ChunkingVerifier:
    """
    Audits actual chunks stored in output chunks folder using the sentence-expansion distance metric:
    - Calculates embeddings for one sentence, then the same sentence plus the next one,
      and measures the semantic distance vector.
    - Evaluates whether internal sentence expansions remain within the accepted threshold.
    - Computes distance distribution and percentiles across all chunks within the book.
    - Reconstructs smart chunks and verifies boundary consistency against actual chunks.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        distance_threshold: float = 0.20,
        max_chunk_chars: int = 2500,
        min_chunk_chars: int = 120,
    ):
        self.embeddings = embeddings
        self.distance_threshold = distance_threshold
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.smart_chunker = SmartSentenceChunker(
            embeddings=embeddings,
            distance_threshold=distance_threshold,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        )

    def verify_chunk(
        self, chunk: HierarchicalChunk
    ) -> Tuple[bool, float, List[float], Optional[Tuple[str, str, float]]]:
        """
        Verify internal semantic coherence of a chunk using incremental sentence expansion.
        Returns: (is_coherent, max_step_distance, step_distances, drift_info_or_None)
        """
        sentences = split_sentences(chunk.text)
        if len(sentences) <= 1:
            return True, 0.0, [], None

        cum_texts: List[str] = []
        accum = ""
        for s in sentences:
            accum = (accum + " " + s).strip() if accum else s
            cum_texts.append(accum)

        embs = self.embeddings.embed_documents(cum_texts)
        step_distances: List[float] = []
        max_dist = 0.0
        worst_drift: Optional[Tuple[str, str, float]] = None

        for j in range(len(embs) - 1):
            d = cosine_distance(embs[j], embs[j + 1])
            step_distances.append(d)
            if d > max_dist:
                max_dist = d
            if d > self.distance_threshold and worst_drift is None:
                worst_drift = (sentences[j + 1], cum_texts[j], d)

        is_coherent = max_dist <= self.distance_threshold
        return is_coherent, max_dist, step_distances, worst_drift

    def verify_book(
        self,
        book_id: int,
        chunks: List[HierarchicalChunk],
        book_title: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[ChunkingVerificationStats, List[ChunkingDiscrepancy]]:
        """
        Verify all chunks of a book, produce distribution across chunks, and compare against smart chunks.
        """
        start_time = time.time()
        total_chunks = len(chunks)
        total_sentences = 0
        all_step_distances: List[float] = []
        coherent_chunks_cnt = 0
        divergent_chunks_cnt = 0
        discrepancies: List[ChunkingDiscrepancy] = []

        # 1. Audit internal coherence of all actual chunks
        for idx, chk in enumerate(chunks, 1):
            sents = split_sentences(chk.text)
            total_sentences += len(sents)

            is_coh, max_d, step_dists, drift_info = self.verify_chunk(chk)
            all_step_distances.extend(step_dists)

            if is_coh:
                coherent_chunks_cnt += 1
            else:
                divergent_chunks_cnt += 1
                if drift_info:
                    drift_sent, prior_text, drift_dist = drift_info
                    discrepancies.append(
                        ChunkingDiscrepancy(
                            chunk_id=chk.chunk_id,
                            book_id=book_id,
                            book_title=book_title,
                            section_title=chk.section_title or chk.chapter_title,
                            chunk_idx=chk.chunk_idx,
                            discrepancy_type="internal_semantic_drift",
                            max_distance=round(drift_dist, 4),
                            threshold=self.distance_threshold,
                            drift_sentence=drift_sent[:150],
                            prior_text_snippet=prior_text[-150:],
                            explanation=(
                                f"Incremental sentence expansion distance {drift_dist:.3f} "
                                f"exceeded threshold {self.distance_threshold:.3f}."
                            ),
                        )
                    )

            if progress_callback:
                progress_callback(idx, total_chunks, chk.chunk_id)

        # 2. Section-level reconstruction & comparison with actual chunks
        from collections import defaultdict
        section_to_chunks: Dict[Tuple[int, str], List[HierarchicalChunk]] = defaultdict(list)
        for chk in chunks:
            section_to_chunks[(chk.chapter_idx, chk.section_title)].append(chk)

        exact_matches = 0
        total_reconstructed = 0

        for sec_key, sec_chunks in section_to_chunks.items():
            combined_sec_text = "\n\n".join(c.text for c in sec_chunks)
            smart_chunks = self.smart_chunker.chunk_text(combined_sec_text)
            total_reconstructed += len(smart_chunks)

            for act_chk in sec_chunks:
                act_norm = " ".join(act_chk.text.split())
                matched = False
                for sm_chk in smart_chunks:
                    sm_norm = " ".join(sm_chk.split())
                    if act_norm == sm_norm:
                        matched = True
                        exact_matches += 1
                        break
                    if abs(len(act_norm) - len(sm_norm)) < 0.1 * len(act_norm) and (
                        act_norm[:50] == sm_norm[:50] or act_norm[-50:] == sm_norm[-50:]
                    ):
                        matched = True
                        exact_matches += 1
                        break

                if not matched and len(sec_chunks) > 1:
                    discrepancies.append(
                        ChunkingDiscrepancy(
                            chunk_id=act_chk.chunk_id,
                            book_id=book_id,
                            book_title=book_title,
                            section_title=act_chk.section_title,
                            chunk_idx=act_chk.chunk_idx,
                            discrepancy_type="boundary_mismatch",
                            max_distance=0.0,
                            threshold=self.distance_threshold,
                            drift_sentence="",
                            prior_text_snippet="",
                            explanation=(
                                f"Actual chunk boundaries ({len(act_chk.text)} chars) differ from "
                                f"smart expanding chunking ({len(smart_chunks)} smart chunk(s) generated)."
                            ),
                        )
                    )

        # 3. Calculate distance distribution statistics
        total_trans = len(all_step_distances)
        if total_trans > 0:
            sorted_dists = sorted(all_step_distances)
            mean_d = sum(sorted_dists) / total_trans
            median_d = sorted_dists[total_trans // 2]
            min_d = sorted_dists[0]
            max_d = sorted_dists[-1]
            p80_d = sorted_dists[int(total_trans * 0.80)]
            p95_d = sorted_dists[min(total_trans - 1, int(total_trans * 0.95))]
            std_d = math.sqrt(sum((x - mean_d) ** 2 for x in sorted_dists) / total_trans)
        else:
            mean_d = median_d = min_d = max_d = p80_d = p95_d = std_d = 0.0

        # 4. Build distribution histogram buckets
        bucket_defs = [
            (0.000, 0.050, "0.000 - 0.050", "Very High Coherence"),
            (0.050, 0.100, "0.050 - 0.100", "High Coherence"),
            (0.100, 0.150, "0.100 - 0.150", "Moderate Coherence"),
            (0.150, 0.200, "0.150 - 0.200", "Borderline Coherence"),
            (0.200, float("inf"), ">= 0.200", "Semantic Shift / Drift"),
        ]

        buckets: List[DistanceBucket] = []
        for low, high, lbl, coh in bucket_defs:
            cnt = sum(1 for d in all_step_distances if (low <= d < high if high != float("inf") else d >= low))
            pct = round((cnt / total_trans * 100.0), 1) if total_trans > 0 else 0.0
            bar_len = int(round(pct / 4.0))
            bar_str = "█" * bar_len if bar_len > 0 else ("▏" if cnt > 0 else "")
            buckets.append(
                DistanceBucket(
                    range_label=lbl,
                    coherence_level=coh,
                    count=cnt,
                    percentage=pct,
                    bar=bar_str,
                )
            )

        duration = round(time.time() - start_time, 2)
        coherence_rate = round((coherent_chunks_cnt / total_chunks * 100.0), 2) if total_chunks > 0 else 100.0
        boundary_match_rate = round((exact_matches / total_chunks * 100.0), 2) if total_chunks > 0 else 100.0

        stats = ChunkingVerificationStats(
            book_id=book_id,
            book_title=book_title,
            total_chunks=total_chunks,
            total_sentences=total_sentences,
            total_transitions=total_trans,
            coherent_chunks=coherent_chunks_cnt,
            divergent_chunks=divergent_chunks_cnt,
            coherence_rate=coherence_rate,
            boundary_match_rate=boundary_match_rate,
            exact_matches_count=exact_matches,
            reconstructed_chunks_count=total_reconstructed,
            actual_chunks_count=total_chunks,
            mean_distance=round(mean_d, 4),
            median_distance=round(median_d, 4),
            min_distance=round(min_d, 4),
            max_distance=round(max_d, 4),
            p80_distance=round(p80_d, 4),
            p95_distance=round(p95_d, 4),
            std_dev_distance=round(std_d, 4),
            distance_threshold=self.distance_threshold,
            distribution_buckets=buckets,
            total_duration_seconds=duration,
        )

        return stats, discrepancies


def verify_chunking(
    chunk_store: ChunkStore,
    embeddings: Embeddings,
    distance_threshold: float = 0.20,
    book_id: Optional[int] = None,
    all_books: bool = False,
    seed: Optional[int] = None,
    max_examples: int = 20,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> ChunkingVerificationReport:
    """
    Run smart chunking verification across stored chunks.
    If book_id is provided, verifies that book.
    If all_books is True, verifies all stored books.
    Otherwise, randomly selects a stored book.
    """
    stored_ids = chunk_store.list_stored_book_ids()
    if not stored_ids:
        raise ValueError(f"No stored book chunks found in {chunk_store.storage_dir}.")

    if book_id is not None:
        if book_id not in stored_ids:
            raise ValueError(f"Book #{book_id} not found in chunks directory. Available: {stored_ids}")
        target_ids = [book_id]
    elif all_books:
        target_ids = stored_ids
    else:
        rng = random.Random(seed) if seed is not None else random.Random()
        target_ids = [rng.choice(stored_ids)]

    verifier = ChunkingVerifier(
        embeddings=embeddings,
        distance_threshold=distance_threshold,
    )

    all_stats: List[ChunkingVerificationStats] = []
    all_discrepancies: List[ChunkingDiscrepancy] = []

    for bid in target_ids:
        chunks = chunk_store.load_chunks(bid) or []
        btitle = chunk_store.get_book_title(bid) or f"Book #{bid}"
        if not chunks:
            continue
        stats, discs = verifier.verify_book(
            book_id=bid,
            chunks=chunks,
            book_title=btitle,
            progress_callback=progress_callback,
        )
        all_stats.append(stats)
        all_discrepancies.extend(discs)

    if not all_stats:
        raise ValueError("No chunks were loaded for verification.")

    primary_stats = all_stats[0]
    raw_emb = getattr(embeddings, "model", None)
    emb_model_name = raw_emb if isinstance(raw_emb, str) else "embeddings"

    return ChunkingVerificationReport(
        mode="chunking",
        embedding_model=emb_model_name,
        distance_threshold=distance_threshold,
        stats=primary_stats,
        discrepancies=all_discrepancies[:max_examples],
        all_books_summary=all_stats if len(all_stats) > 1 else None,
    )
