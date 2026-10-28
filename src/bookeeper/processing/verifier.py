"""
Knowledge graph factual verification and discrepancy audit engine.
Audits extracted ideas and concepts against source text chunks using a dedicated LLM model.
"""

import datetime
import json
import logging
import math
import random
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import ChunkStore, HierarchicalChunk
from bookeeper.processing.cross_checker import (
    CrossCheckAuditItem,
    CrossCheckBlockResult,
    CrossCheckReport,
    CrossCheckSummary,
    SystemBlockMetrics,
    get_ideas_for_chunks,
    run_cross_check,
)
from bookeeper.processing.extractor import Concept
from bookeeper.processing.ollama_pool import DaemonThreadPoolExecutor, OllamaPool, _thread_local
from bookeeper.processing.qa_probe import (
    AtomicAssertionItem,
    NLIAssertionEvaluation,
    build_nli_verifier_prompt,
    calculate_ternary_scores,
    evaluate_claims_atomic_assertions,
    format_ideas_for_context,
    generate_atomic_assertions,
    generate_chunk_probes,
    verify_assertion_entailment,
)

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
    raw_prompt: Optional[str] = None
    raw_response: Optional[str] = None


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
    chunk_text: Optional[str] = None
    sentences: List[str] = Field(default_factory=list)
    raw_prompt: Optional[str] = None
    raw_response: Optional[str] = None
    is_supported: bool
    confidence: float
    explanation: str
    server_node: Optional[str] = None


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


class ChunkMatchedPair(BaseModel):
    """Pairing of an Oracle idea with an Original idea confirmed as the same concept."""

    oracle_idea_name: str
    original_idea_name: str
    similarity_score: float = 1.0
    match_method: str = "exact"  # "exact", "high_vector", "llm_disambiguated"
    canonical_name: Optional[str] = None
    reasoning: Optional[str] = None


class ChunkAuditItem(BaseModel):
    """Detailed audit record for a single chunk."""

    chunk_id: str
    book_id: Optional[int] = None
    book_title: str = "Unknown"
    section_title: str = "Unknown"
    chunk_text_snippet: str = ""

    # Full text context and sentence breakdown
    chunk_text: Optional[str] = None
    sentences: List[str] = Field(default_factory=list)
    formatted_claims_context: Optional[str] = None
    probe_generation_prompt: Optional[str] = None
    probe_generation_response: Optional[str] = None
    ideas_details: List[Dict[str, Any]] = Field(default_factory=list)

    # Concept ideas in graph
    original_ideas: List[str] = Field(default_factory=list)
    oracle_ideas: List[str] = Field(default_factory=list)

    # NLI Atomic Assertion & Evaluation Probes
    probes: List[AtomicAssertionItem] = Field(default_factory=list)
    assertions: List[AtomicAssertionItem] = Field(default_factory=list)
    evaluations: List[NLIAssertionEvaluation] = Field(default_factory=list)

    # Assertion Counts
    total_probes: int = 0
    passed_probes: int = 0
    failed_probes: int = 0
    supported_count: int = 0
    not_mentioned_count: int = 0
    against_count: int = 0

    # Rates & Continuous Scores
    qa_recall: float = 0.0
    support_ratio: float = 0.0
    hallucination_ratio: float = 0.0
    net_integrity_score: float = 0.0
    calibrated_fidelity: float = 0.0
    seam_integrity_rate: float = 1.0
    seam_net_integrity: float = 0.0
    seam_calibrated_fidelity: float = 0.0

    # Legacy alignment / count fields for backwards compatibility
    matched_pairs: List[ChunkMatchedPair] = Field(default_factory=list)
    missed_oracle_ideas: List[str] = Field(default_factory=list)  # In Oracle/Probes, missing/unsubstantiated in claims
    extra_original_ideas: List[str] = Field(default_factory=list)
    original_count: int = 0
    oracle_count: int = 0
    matched_count: int = 0
    missed_count: int = 0
    success_rate: float = 0.0  # Matched / Oracle % or QA Recall %
    fail_rate: float = 0.0  # Missed % or 100 - QA Recall %

    duration_seconds: float = 0.0
    server_node: Optional[str] = None

    def __init__(self, **data: Any):
        if "assertions" in data and "probes" not in data:
            data["probes"] = data["assertions"]
        elif "probes" in data and "assertions" not in data:
            data["assertions"] = data["probes"]

        # Sync legacy fields if new ones are passed
        if "qa_recall" in data and "success_rate" not in data:
            data["success_rate"] = round(float(data["qa_recall"]) * 100.0, 2)
        if "qa_recall" in data and "fail_rate" not in data:
            data["fail_rate"] = round((1.0 - float(data["qa_recall"])) * 100.0, 2)
        if "supported_count" in data and "matched_count" not in data:
            data["matched_count"] = int(data["supported_count"])
        if "failed_probes" in data and "missed_count" not in data:
            data["missed_count"] = int(data["failed_probes"])
        if "total_probes" in data and "oracle_count" not in data:
            data["oracle_count"] = int(data["total_probes"])

        # Sync forward if legacy fields are passed
        if "matched_count" in data and "supported_count" not in data:
            data["supported_count"] = int(data["matched_count"])
            if "passed_probes" not in data:
                data["passed_probes"] = int(data["matched_count"])
        if "missed_count" in data and "failed_probes" not in data:
            data["failed_probes"] = int(data["missed_count"])
            if "not_mentioned_count" not in data:
                data["not_mentioned_count"] = int(data["missed_count"])
        if "oracle_count" in data and "total_probes" not in data:
            data["total_probes"] = int(data["oracle_count"])
        if "success_rate" in data and "qa_recall" not in data:
            data["qa_recall"] = round(float(data["success_rate"]) / 100.0, 4)

        super().__init__(**data)


class ChunkVerificationStats(BaseModel):
    """Summary statistics across all audited chunks."""

    mode: str = "chunk"
    model_name: str
    embedding_model: str = "embeddings"
    total_chunks_in_graph: int = 0
    sampled_chunks_count: int = 0
    sample_percentage: float = 0.0

    # Continuous ternary verification metrics across all chunks
    total_assertions_count: int = 0
    total_supported_count: int = 0
    total_not_mentioned_count: int = 0
    total_against_count: int = 0

    mean_qa_recall: float = 0.0
    mean_support_ratio: float = 0.0
    mean_hallucination_ratio: float = 0.0
    mean_net_integrity_score: float = 0.0
    mean_calibrated_fidelity: float = 0.0
    mean_seam_integrity_rate: float = 1.0

    chunks_with_perfect_recall: int = 0
    chunks_with_omissions: int = 0
    chunks_with_hallucinations: int = 0

    # Legacy fields for backward compatibility
    total_original_ideas: int = 0
    total_oracle_ideas: int = 0
    total_matched_ideas: int = 0
    total_missed_ideas: int = 0
    total_extra_original_ideas: int = 0

    overall_success_rate: float = 0.0
    overall_fail_rate: float = 0.0
    chunks_with_perfect_match: int = 0

    total_duration_seconds: float = 0.0

    def __init__(self, **data: Any):
        # Sync forward or backward
        if "mean_qa_recall" in data and "overall_success_rate" not in data:
            data["overall_success_rate"] = round(float(data["mean_qa_recall"]) * 100.0, 2)
            if "overall_fail_rate" not in data:
                data["overall_fail_rate"] = round((1.0 - float(data["mean_qa_recall"])) * 100.0, 2)
        elif "overall_success_rate" in data and "mean_qa_recall" not in data:
            data["mean_qa_recall"] = round(float(data["overall_success_rate"]) / 100.0, 4)

        if "chunks_with_perfect_recall" in data and "chunks_with_perfect_match" not in data:
            data["chunks_with_perfect_match"] = int(data["chunks_with_perfect_recall"])
        elif "chunks_with_perfect_match" in data and "chunks_with_perfect_recall" not in data:
            data["chunks_with_perfect_recall"] = int(data["chunks_with_perfect_match"])

        if "total_assertions_count" in data and "total_oracle_ideas" not in data:
            data["total_oracle_ideas"] = int(data["total_assertions_count"])
        elif "total_oracle_ideas" in data and "total_assertions_count" not in data:
            data["total_assertions_count"] = int(data["total_oracle_ideas"])

        if "total_supported_count" in data and "total_matched_ideas" not in data:
            data["total_matched_ideas"] = int(data["total_supported_count"])
        elif "total_matched_ideas" in data and "total_supported_count" not in data:
            data["total_supported_count"] = int(data["total_matched_ideas"])

        super().__init__(**data)


class ChunkVerificationReport(BaseModel):
    """Full serialization report for chunk verification mode."""

    mode: str = "chunk"
    stats: ChunkVerificationStats
    audited_chunks: List[ChunkAuditItem]
    discrepancy_examples: List[ChunkAuditItem] = Field(default_factory=list)
    omission_examples: List[ChunkAuditItem] = Field(default_factory=list)

    def __init__(self, **data: Any):
        if "omission_examples" in data and "discrepancy_examples" not in data:
            data["discrepancy_examples"] = data["omission_examples"]
        elif "discrepancy_examples" in data and "omission_examples" not in data:
            data["omission_examples"] = data["discrepancy_examples"]
        super().__init__(**data)


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


def _parse_verification_text(text: str) -> Optional[VerificationResult]:
    """
    Resilient parser to extract a valid VerificationResult from model output:
    1. Direct JSON parse (cleaned of thinking tags / markdown fences)
    2. Embedded JSON regex extraction { ... }
    3. Heuristic text fallback for freeform text or markdown bullet points
       (e.g., '* Target Idea: "Rampart" *does* substantiate the...')
    """
    if not text or not str(text).strip():
        return None

    raw_str = str(text).strip()
    cleaned = _clean_json_str(raw_str)

    # 1. Direct JSON parse
    try:
        return VerificationResult.model_validate_json(cleaned)
    except Exception:
        pass

    # 2. Search for embedded JSON object block
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return VerificationResult.model_validate_json(match.group(0))
        except Exception:
            pass

    # 3. Heuristic text analysis for markdown/freeform model answers
    lower = raw_str.lower()
    negative_patterns = [
        "does not substantiate", "does not support", "is not supported",
        "not substantiated", "unsupported", "absent", "fabricated",
        "not grounded", "*does not*", "fails to substantiate",
        "no evidence", "not mentioned", "is unsupported: true",
        "is_supported: false", 'is_supported": false', "is_supported': false",
    ]
    positive_patterns = [
        "does substantiate", "substantiates", "is supported",
        "genuinely contains", "directly supports", "supports the idea",
        "*does* substantiate", "clearly explores", "directly mentions",
        "well supported", "is_supported: true", 'is_supported": true',
        "is_supported': true",
    ]

    is_supported: Optional[bool] = None
    if any(p in lower for p in negative_patterns):
        is_supported = False
    elif any(p in lower for p in positive_patterns):
        is_supported = True
    else:
        if "verdict: pass" in lower or "verdict: true" in lower:
            is_supported = True
        elif "verdict: fail" in lower or "verdict: false" in lower:
            is_supported = False

    if is_supported is not None:
        clean_text = " ".join(raw_str.replace("*", "").split())
        explanation = clean_text[:250] + ("..." if len(clean_text) > 250 else "")
        return VerificationResult(
            is_supported=is_supported,
            confidence=0.85,
            explanation=explanation,
        )

    return None


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
        cooldown = getattr(settings, "resolved_verification_cooldown_seconds", getattr(settings, "failover_cooldown_seconds", 60))
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
                        "--connect-timeout",
                        "3",
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

    def _execute_structured_invoke(
        self,
        messages: List[Any],
        input_chars: int = 0,
    ) -> VerificationResult:
        """
        Execute structured LLM invoke with automatic pool failover and curl fallback
        to ensure resilience against network timeouts and macOS Local Network Privacy.
        """
        if input_chars <= 0 and messages:
            for m in messages:
                content = getattr(m, "content", "")
                if isinstance(content, str):
                    input_chars += len(content)

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
                parsed = _parse_verification_text(str(res))
                if parsed:
                    return parsed
                cleaned = _clean_json_str(str(res))
                return VerificationResult.model_validate_json(cleaned)
            except Exception as exc:
                # 1. Recover if exception contains raw model output (e.g. Pydantic ValidationError)
                if hasattr(exc, "errors"):
                    try:
                        for err in exc.errors():
                            inp = err.get("input")
                            if inp and isinstance(inp, str):
                                parsed = _parse_verification_text(inp)
                                if parsed:
                                    return parsed
                    except Exception:
                        pass

                err_str = str(exc)
                if "input_value=" in err_str:
                    match = re.search(r"input_value=['\"]([\s\S]*?)['\"], input_type=", err_str)
                    if match:
                        raw_input = match.group(1).replace(r"\n", "\n").replace(r"\'", "'").replace(r'\"', '"')
                        parsed = _parse_verification_text(raw_input)
                        if parsed:
                            return parsed

                # 2. Resilient fallback via curl with standard "format": "json"
                if shutil.which("curl"):
                    msgs_payload = []
                    for m in messages:
                        role = "user"
                        if hasattr(m, "type"):
                            role = "system" if m.type == "system" else "user"
                        content = getattr(m, "content", str(m))
                        msgs_payload.append({"role": role, "content": content})

                    body = {
                        "model": self.model_name,
                        "messages": msgs_payload,
                        "format": "json",
                        "options": {
                            "temperature": self.temperature,
                            "top_p": 0.9,
                        },
                        "stream": False,
                    }
                    cmd = [
                        "curl",
                        "-s",
                        "--connect-timeout",
                        "3",
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
                        try:
                            data = json.loads(res.stdout)
                            if "error" in data:
                                raise RuntimeError(f"Ollama server {url} error: {data['error']}")
                            content = data.get("message", {}).get("content", "")
                            if not content and "thinking" in data.get("message", {}):
                                content = data["message"]["thinking"]
                            parsed = _parse_verification_text(content)
                            if parsed:
                                return parsed
                        except Exception:
                            pass
                raise exc

        return self.pool.execute_with_failover(
            _invoke,
            retries=self.retries,
            capability="verification",
            input_chars=input_chars,
            task_type="verification",
            quarantine_server=False,
            max_task_duration=max(180.0, 3.0 * float(self.timeout)),
        )

    def verify_idea_chunk(
        self,
        idea_name: str,
        idea_category: str,
        idea_summary: str,
        quote: str,
        book_title: str,
        section_title: str,
        chunk_text: str,
        prior_context: str = "",
        subsequent_context: str = "",
    ) -> VerificationResult:
        """
        Verify whether the assigned text chunk (with surrounding scene context) contains or supports the given idea.
        """
        system_prompt = (
            "You are a strict, objective knowledge graph auditor and factual verification specialist.\n"
            "Your task is to verify whether an assigned text chunk from a book genuinely contains, discusses, "
            "exemplifies, dramatizes, or directly supports an extracted concept/idea.\n\n"
            "EVALUATION CRITERIA:\n"
            "1. DIRECT & NARRATIVE SUBSTANTIVE EVIDENCE:\n"
            "   - Set `is_supported: true` (Good) if the assigned text chunk directly mentions, explains, defines, "
            "depicts, dramatizes, or clearly explores the concept/idea. In narrative works, narrative enactment, tactical choices, "
            "character dialogues, or conflicts that directly demonstrate the concept count as valid support.\n"
            "   - Set `is_supported: false` (Failed / Discrepancy) if the concept is completely absent, fabricated, "
            "or entirely disconnected from the passage.\n\n"
            "2. DETECT OVER-ABSTRACTION & CATEGORY MISMATCH:\n"
            "   - Set `is_supported: false` if trivial mundane actions or casual conversational banter have been artificially "
            "elevated into formal engineering principles, scientific laws, or military frameworks.\n"
            "   - Example: Claiming a routine physical misstep is a 'System Design Principle' like 'Fault Tolerance', or "
            "claiming a routine greeting is a 'Military Tactical Retreat'. Such conceptual over-generalizations must be marked `is_supported: false`.\n"
            "   - However, genuine story actions (e.g. retreating from a battle, evading an ambush, organizing a defense) do support "
            "the corresponding concept, even if the characters do not speak in formal academic terminology.\n\n"
            "3. SUPPORTING QUOTE EVALUATION:\n"
            "   - If a Supporting Quote is provided, check whether it is grounded in the passage. If a quote is absent or approximate, "
            "evaluate whether the text chunk itself still substantively depicts or supports the concept. Do NOT reject a genuinely "
            "grounded idea solely because an extracted quote was paraphrased or loosely referenced, as long as the passage itself supports the idea.\n\n"
            "4. SURROUNDING CONTEXT:\n"
            "   - Preceding or following excerpts (if provided) provide scene context. The primary chunk is the focus, but surrounding context "
            "may establish characters, setting, or ongoing actions.\n\n"
            "5. OBJECTIVE JUSTIFICATION:\n"
            "   - In `explanation`, provide a concise 1-2 sentence objective justification.\n"
            "   - In `confidence`, provide a score from 0.0 to 1.0 reflecting your verification certainty.\n\n"
            "MANDATORY RESPONSE FORMAT:\n"
            "You MUST respond ONLY with a raw JSON object matching the following schema. "
            "Do NOT wrap in markdown backticks, do NOT write bullet points, and do NOT include any introductory or concluding text:\n"
            "{\n"
            '  "is_supported": true,\n'
            '  "confidence": 0.95,\n'
            '  "explanation": "Clear 1-2 sentence justification here."\n'
            "}"
        )

        user_content = (
            f"Target Idea/Concept:\n"
            f"- Name: \"{idea_name}\"\n"
            f"- Category: \"{idea_category}\"\n"
        )
        if idea_summary:
            user_content += f"- General Concept Summary / Definition: \"{idea_summary}\"\n"
        if quote:
            user_content += f"- Supporting Quote: \"{quote}\"\n"

        user_content += f"\nSource Location: Book '{book_title}', Section '{section_title}'\n\n"

        if prior_context:
            user_content += f"[Preceding Excerpt Context]:\n...{prior_context.strip()}\n\n"

        user_content += (
            f"[Primary Assigned Text Chunk (Focus of Evaluation)]:\n"
            f"\"\"\"\n{chunk_text.strip()[:3500]}\n\"\"\"\n\n"
        )

        if subsequent_context:
            user_content += f"[Following Excerpt Context]:\n{subsequent_context.strip()}...\n\n"

        user_content += (
            f"Does this assigned text chunk (understood within its surrounding context) genuinely contain, "
            f"discuss, dramatize, or support the idea \"{idea_name}\"?"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        raw_prompt = f"{system_prompt}\n\n{user_content}"
        try:
            res = self._execute_structured_invoke(messages)
            if res.raw_prompt is None:
                res.raw_prompt = raw_prompt
            return res
        except Exception as e:
            logger.warning(f"Verification call failed for '{idea_name}' on chunk: {e}")
            return VerificationResult(
                is_supported=False,
                confidence=0.0,
                explanation=f"LLM verification error across server pool: {e}",
                raw_prompt=raw_prompt,
                raw_response=f"Error: {e}",
            )


def get_surrounding_chunk_context(
    chunk_id: str,
    book_id: Optional[int],
    chunk_store: Optional[ChunkStore] = None,
    graph: Optional[Any] = None,
    window_chars: int = 500,
    book_chunks_cache: Optional[Dict[int, List[HierarchicalChunk]]] = None,
) -> Tuple[str, str]:
    """
    Retrieve predecessor and successor context excerpts for an atomic chunk.
    Uses ChunkStore if available, or falls back to graph sibling chunk nodes in the section.
    Returns (prior_excerpt, subsequent_excerpt).
    """
    if window_chars <= 0:
        return "", ""

    # 1. Look up through ChunkStore
    if chunk_store is not None and book_id is not None:
        try:
            if book_chunks_cache is not None:
                if book_id not in book_chunks_cache:
                    loaded = chunk_store.load_chunks(book_id)
                    book_chunks_cache[book_id] = loaded or []
                b_chunks = book_chunks_cache[book_id]
            else:
                b_chunks = chunk_store.load_chunks(book_id) or []

            if b_chunks:
                match_idx = next((i for i, c in enumerate(b_chunks) if c.chunk_id == chunk_id), None)
                if match_idx is not None:
                    prior = ""
                    if match_idx > 0:
                        txt = b_chunks[match_idx - 1].text.strip()
                        prior = txt[-window_chars:].strip()
                    subsequent = ""
                    if match_idx < len(b_chunks) - 1:
                        txt = b_chunks[match_idx + 1].text.strip()
                        subsequent = txt[:window_chars].strip()
                    return prior, subsequent
        except Exception as e:
            logger.debug(f"Failed to retrieve chunk context from ChunkStore for {chunk_id}: {e}")

    # 2. Look up through Knowledge Graph sibling chunks connected to the same Section
    if graph is not None:
        try:
            target_node_id = f"chunk:{chunk_id}"
            if target_node_id in graph:
                section_parents = [
                    src for src, _, d in graph.in_edges(target_node_id, data=True)
                    if d.get("relation") == "HAS_CHUNK"
                ]
                if section_parents:
                    sec_id = section_parents[0]
                    sibling_ids = [
                        tgt for _, tgt, d in graph.out_edges(sec_id, data=True)
                        if d.get("relation") == "HAS_CHUNK"
                    ]
                    if target_node_id in sibling_ids:
                        idx = sibling_ids.index(target_node_id)
                        prior = ""
                        if idx > 0:
                            p_txt = graph.nodes[sibling_ids[idx - 1]].get("text", "").strip()
                            prior = p_txt[-window_chars:].strip()
                        subsequent = ""
                        if idx < len(sibling_ids) - 1:
                            s_txt = graph.nodes[sibling_ids[idx + 1]].get("text", "").strip()
                            subsequent = s_txt[:window_chars].strip()
                        return prior, subsequent
        except Exception as e:
            logger.debug(f"Failed to retrieve chunk context from graph for {chunk_id}: {e}")

    return "", ""


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
    progress_callback: Optional[Callable[..., None]] = None,
    chunk_store: Optional[ChunkStore] = None,
    context_window_chars: int = 500,
) -> VerificationReport:
    """
    Perform factual verification of ideas in the Knowledge Graph against their assigned chunks.
    Randomly samples `percent`% of ideas, pulls their related chunks, sanitizes quotes,
    enriches with surrounding scene context, and audits grounding.
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

    # Cache loaded book chunks across evaluation tasks
    book_chunks_cache: Dict[int, List[HierarchicalChunk]] = {}

    # 4. Gather all idea-chunk tasks with sanitized quotes and surrounding context
    eval_tasks: List[Dict[str, Any]] = []
    for c_id, c_attrs, chunks in sampled_candidates:
        c_name = c_attrs.get("name", c_id.replace("concept:", ""))
        c_cat = c_attrs.get("category", "Concept")
        c_sum = c_attrs.get("summary") or c_attrs.get("brief_description", "")
        c_wt = c_attrs.get("weight", 5)

        for chunk_node_id, chunk_attrs, edge_data in chunks:
            raw_quote = edge_data.get("quote", "")
            chunk_text = chunk_attrs.get("text", "")
            if not chunk_text and raw_quote:
                chunk_text = raw_quote

            # 1. Quote sanitization:
            # Merged canonical concepts from previous runs/books may leak cross-book quotes.
            # Only pass quote to the verifier if it is actually grounded in this chunk's text!
            sanitized_quote = ""
            if raw_quote and chunk_text:
                q_norm = " ".join(raw_quote.lower().split())
                t_norm = " ".join(chunk_text.lower().split())
                if q_norm in t_norm:
                    sanitized_quote = raw_quote
                else:
                    # Check significant word overlap
                    q_words = set(re.findall(r"\w{4,}", q_norm))
                    if q_words:
                        t_words = set(re.findall(r"\w{4,}", t_norm))
                        overlap = len(q_words & t_words) / len(q_words)
                        if overlap >= 0.5:
                            sanitized_quote = raw_quote

            # 2. Localized description / summary:
            # Prioritize chunk-level description on the edge over global canonical summary
            brief = (edge_data.get("brief_description") or "").strip()
            detailed = (edge_data.get("detailed_explanation") or "").strip()
            if brief and detailed and brief != detailed:
                local_desc = f"{brief}. {detailed}"
            else:
                local_desc = detailed or brief
            effective_summary = local_desc if local_desc else c_sum

            bid = chunk_attrs.get("book_id")
            chk_id = chunk_attrs.get("chunk_id", chunk_node_id.replace("chunk:", ""))

            prior_ctx, sub_ctx = get_surrounding_chunk_context(
                chunk_id=chk_id,
                book_id=bid,
                chunk_store=chunk_store,
                graph=store.graph,
                window_chars=context_window_chars,
                book_chunks_cache=book_chunks_cache,
            )

            eval_tasks.append({
                "idea_name": c_name,
                "idea_category": c_cat,
                "idea_summary": effective_summary,
                "idea_weight": c_wt,
                "quote": sanitized_quote,
                "chunk_id": chk_id,
                "book_id": bid,
                "book_title": chunk_attrs.get("book_title", "Unknown"),
                "section_title": chunk_attrs.get("section_title", chunk_attrs.get("chapter_title", "Unknown")),
                "chunk_text": chunk_text,
                "prior_context": prior_ctx,
                "subsequent_context": sub_ctx,
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
            prior_context=task_item.get("prior_context", ""),
            subsequent_context=task_item.get("subsequent_context", ""),
        )
        server_lbl = getattr(_thread_local, "last_used_server", None)
        c_text = task_item["chunk_text"]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", c_text) if s.strip()]
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
            chunk_snippet=c_text[:250].strip() + ("..." if len(c_text) > 250 else ""),
            chunk_text=c_text,
            sentences=sentences,
            raw_prompt=res.raw_prompt,
            raw_response=res.raw_response,
            is_supported=res.is_supported,
            confidence=res.confidence,
            explanation=res.explanation,
            server_node=server_lbl,
        )

    # 5. Execute evaluations with worker concurrency
    completed_count = 0
    with DaemonThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {executor.submit(_eval_single, t): t for t in eval_tasks}
        for fut in as_completed(futures):
            item = fut.result()
            completed_count += 1
            if item.is_supported:
                verified_samples.append(item)
            else:
                discrepancies.append(item)

            if progress_callback:
                try:
                    progress_callback(completed_count, total_evals, item.idea_name, item)
                except TypeError:
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
                try:
                    progress_callback(idx, total_chunks, chk.chunk_id, is_coh, chk)
                except TypeError:
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


def verify_chunks(
    store: ConceptGraphStore,
    pool: OllamaPool,
    model_name: str = "llama3.1:8b",
    percent: float = 1.0,
    chunk_store: Optional[ChunkStore] = None,
    embeddings: Optional[Embeddings] = None,
    max_examples: int = 20,
    seed: Optional[int] = None,
    concurrency: int = 3,
    timeout: int = 300,
    progress_callback: Optional[Callable[..., None]] = None,
    num_probes_per_chunk: int = 1,
) -> ChunkVerificationReport:
    """
    Verify Knowledge Graph idea extraction quality by auditing sampled chunks against an Oracle model.
    1. Randomly samples `percent`% of chunks from knowledge_graph.json / chunk_store.
    2. Uses Oracle model to formulate benchmark question(s) for the selected chunk in ternary answer format.
    3. Retrieves ideas extracted from that chunk with complete metadata (short description, full description, quotes).
    4. Evaluates NLI entailment in ternary format: SUPPORTED (+1.0), NOT_MENTIONED (-0.25), CONTRADICTED (-1.0).
    5. Computes QA Recall, Net Semantic Integrity, Calibrated Fidelity, and Seam Integrity per chunk and roll-up.
    """
    start_time = time.time()

    all_chunk_nodes = [
        (n, d)
        for n, d in store.graph.nodes(data=True)
        if d.get("type") == "Chunk"
    ]

    valid_chunks: List[Tuple[str, str, Optional[int], str, str]] = []
    for cid, cattrs in all_chunk_nodes:
        text = cattrs.get("text", "")
        book_title = cattrs.get("book_title", "Unknown")
        book_id = cattrs.get("book_id")
        section_title = cattrs.get("section_title") or cattrs.get("chapter_title", "Unknown")

        if (not text or len(text.strip()) < 20) and chunk_store is not None:
            stored = chunk_store.get_chunk(cid)
            if stored is not None:
                text = getattr(stored, "text", text)
                if getattr(stored, "book_title", None):
                    book_title = stored.book_title
                if getattr(stored, "section_title", None):
                    section_title = stored.section_title
                if getattr(stored, "book_id", None):
                    book_id = stored.book_id

        if text and len(text.strip()) >= 20:
            valid_chunks.append((cid, text, book_id, book_title, section_title))

    raw_emb = getattr(embeddings, "model", None)
    emb_model_name = raw_emb if isinstance(raw_emb, str) else "embeddings"

    if not valid_chunks:
        empty_stats = ChunkVerificationStats(
            mode="chunk",
            model_name=model_name,
            embedding_model=emb_model_name,
            total_chunks_in_graph=len(all_chunk_nodes),
            sampled_chunks_count=0,
            sample_percentage=percent,
            total_assertions_count=0,
            total_supported_count=0,
            total_not_mentioned_count=0,
            total_against_count=0,
            mean_qa_recall=1.0,
            mean_support_ratio=1.0,
            mean_hallucination_ratio=0.0,
            mean_net_integrity_score=1.0,
            mean_calibrated_fidelity=1.0,
            mean_seam_integrity_rate=1.0,
            chunks_with_perfect_recall=0,
            chunks_with_omissions=0,
            chunks_with_hallucinations=0,
            total_original_ideas=0,
            total_oracle_ideas=0,
            total_matched_ideas=0,
            total_missed_ideas=0,
            total_extra_original_ideas=0,
            overall_success_rate=100.0,
            overall_fail_rate=0.0,
            chunks_with_perfect_match=0,
            total_duration_seconds=0.0,
        )
        return ChunkVerificationReport(
            mode="chunk",
            stats=empty_stats,
            audited_chunks=[],
            discrepancy_examples=[],
            omission_examples=[],
        )

    clamped_percent = max(0.01, min(100.0, percent))
    sample_size = max(1, int(round(len(valid_chunks) * (clamped_percent / 100.0))))
    sample_size = min(sample_size, len(valid_chunks))

    rng = random.Random(seed) if seed is not None else random.Random()
    sampled_chunks = rng.sample(valid_chunks, sample_size)

    def _audit_single_chunk(chunk_item: Tuple[str, str, Optional[int], str, str]) -> ChunkAuditItem:
        cid, text, bid, btitle, stitle = chunk_item
        t_start = time.time()

        # 1. Retrieve Original Ideas extracted from this chunk with complete metadata (short & full descriptions, quotes)
        orig_concepts = get_ideas_for_chunks(store, [cid])

        # 2. Formulate Question for Selected Chunk & Evaluate Entailment in Ternary Answer Format
        try:
            assertions, metrics = evaluate_claims_atomic_assertions(
                w_raw=text,
                ideas=orig_concepts,
                pool=pool,
                model_name=model_name,
                num_probes=num_probes_per_chunk,
                is_chunk_mode=True,
            )
        except Exception as exc:
            logger.warning(f"Chunk question NLI audit failed for chunk '{cid}': {exc}")
            assertions = []
            metrics = None

        dur = round(time.time() - t_start, 2)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        ideas_details = [
            {
                "name": c.name,
                "category": getattr(c, "category", "Concept"),
                "brief_description": (getattr(c, "brief_description", "") or getattr(c, "summary", "") or "").strip(),
                "detailed_explanation": (getattr(c, "detailed_explanation", "") or "").strip(),
                "quote": (getattr(c, "quote", "") or getattr(c, "supporting_quote", "") or "").strip(),
            }
            for c in orig_concepts
        ]
        probe_gen_prompt = None
        probe_gen_response = None
        if assertions:
            probe_gen_prompt = getattr(assertions[0], "raw_prompt", None)
            probe_gen_response = getattr(assertions[0], "raw_response", None)
        formatted_claims = getattr(metrics, "formatted_claims_context", None) if metrics else None

        if metrics is not None:
            missed_assertions = [
                getattr(e, "question", None) or e.assertion
                for e in metrics.evaluations
                if e.classification in ("NOT_MENTIONED", "AGAINST", "CONTRADICTED")
            ]
            matched_assertions = [
                ChunkMatchedPair(
                    oracle_idea_name=getattr(e, "question", None) or e.assertion,
                    original_idea_name="Claims Context",
                    similarity_score=1.0 if e.classification == "SUPPORTED" else 0.0,
                    match_method="nli_entailment",
                    canonical_name=e.assertion,
                    reasoning=e.rationale,
                )
                for e in metrics.evaluations
                if e.classification == "SUPPORTED"
            ]

            return ChunkAuditItem(
                chunk_id=cid,
                book_id=bid,
                book_title=btitle,
                section_title=stitle,
                chunk_text_snippet=text[:150],
                chunk_text=text,
                sentences=sentences,
                formatted_claims_context=formatted_claims,
                probe_generation_prompt=probe_gen_prompt,
                probe_generation_response=probe_gen_response,
                ideas_details=ideas_details,
                original_ideas=[c.name for c in orig_concepts],
                oracle_ideas=[getattr(a, "question", None) or a.assertion for a in assertions],
                probes=assertions,
                assertions=assertions,
                evaluations=metrics.evaluations,
                total_probes=metrics.total_probes,
                passed_probes=metrics.passed_probes,
                failed_probes=metrics.failed_probes,
                supported_count=metrics.supported_count,
                not_mentioned_count=metrics.not_mentioned_count,
                against_count=metrics.against_count,
                qa_recall=metrics.qa_recall,
                support_ratio=metrics.support_ratio,
                hallucination_ratio=metrics.hallucination_ratio,
                net_integrity_score=metrics.net_integrity_score,
                calibrated_fidelity=metrics.calibrated_fidelity,
                seam_integrity_rate=metrics.seam_integrity_rate,
                seam_net_integrity=metrics.seam_net_integrity,
                seam_calibrated_fidelity=metrics.seam_calibrated_fidelity,
                matched_pairs=matched_assertions,
                missed_oracle_ideas=missed_assertions,
                extra_original_ideas=[],
                original_count=len(orig_concepts),
                oracle_count=metrics.total_probes,
                matched_count=metrics.supported_count,
                missed_count=metrics.failed_probes,
                success_rate=round(metrics.qa_recall * 100.0, 2),
                fail_rate=round((1.0 - metrics.qa_recall) * 100.0, 2),
                duration_seconds=dur,
            )
        else:
            return ChunkAuditItem(
                chunk_id=cid,
                book_id=bid,
                book_title=btitle,
                section_title=stitle,
                chunk_text_snippet=text[:150],
                chunk_text=text,
                sentences=sentences,
                formatted_claims_context=None,
                probe_generation_prompt=None,
                probe_generation_response=None,
                ideas_details=ideas_details,
                original_ideas=[c.name for c in orig_concepts],
                oracle_ideas=[],
                probes=[],
                assertions=[],
                evaluations=[],
                original_count=len(orig_concepts),
                duration_seconds=dur,
            )

    audited_items: List[ChunkAuditItem] = []
    completed_count = 0
    total_count = len(sampled_chunks)

    num_alive = len(pool.alive_nodes) if hasattr(pool, "alive_nodes") else 1
    eff_concurrency = max(1, min(concurrency, num_alive or 1))

    if eff_concurrency <= 1 or total_count <= 1:
        for item in sampled_chunks:
            res = _audit_single_chunk(item)
            audited_items.append(res)
            completed_count += 1
            if progress_callback:
                try:
                    progress_callback(completed_count, total_count, item[0], res)
                except Exception:
                    pass
    else:
        with DaemonThreadPoolExecutor(max_workers=eff_concurrency) as executor:
            futures = {executor.submit(_audit_single_chunk, item): item for item in sampled_chunks}
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    logger.error(f"Error auditing chunk {item[0]}: {exc}")
                    cid, txt, bid, bt, st = item
                    res = ChunkAuditItem(
                        chunk_id=cid,
                        book_id=bid,
                        book_title=bt,
                        section_title=st,
                        chunk_text_snippet=txt[:150],
                    )
                audited_items.append(res)
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_count, item[0], res)
                    except Exception:
                        pass

    # Sort audited items by chunk_id
    audited_items.sort(key=lambda x: x.chunk_id)

    tot_assertions = sum(it.total_probes for it in audited_items)
    tot_sup = sum(it.supported_count for it in audited_items)
    tot_not = sum(it.not_mentioned_count for it in audited_items)
    tot_opp = sum(it.against_count for it in audited_items)
    tot_orig = sum(it.original_count for it in audited_items)

    mean_rec = round(float(np.mean([it.qa_recall for it in audited_items])), 4) if audited_items else 0.0
    mean_sup_ratio = round(float(np.mean([it.support_ratio for it in audited_items])), 4) if audited_items else 0.0
    mean_opp_ratio = round(float(np.mean([it.hallucination_ratio for it in audited_items])), 4) if audited_items else 0.0
    mean_net = round(float(np.mean([it.net_integrity_score for it in audited_items])), 4) if audited_items else 0.0
    mean_fid = round(float(np.mean([it.calibrated_fidelity for it in audited_items])), 4) if audited_items else 0.0
    mean_seam = round(float(np.mean([it.seam_integrity_rate for it in audited_items])), 4) if audited_items else 1.0

    perfect_cnt = sum(1 for it in audited_items if it.qa_recall >= 1.0 and it.total_probes > 0)
    omission_cnt = sum(1 for it in audited_items if it.not_mentioned_count > 0)
    hallucination_cnt = sum(1 for it in audited_items if it.against_count > 0)

    tot_dur = round(time.time() - start_time, 2)

    stats = ChunkVerificationStats(
        mode="chunk",
        model_name=model_name,
        embedding_model=emb_model_name,
        total_chunks_in_graph=len(all_chunk_nodes),
        sampled_chunks_count=len(audited_items),
        sample_percentage=clamped_percent,
        total_assertions_count=tot_assertions,
        total_supported_count=tot_sup,
        total_not_mentioned_count=tot_not,
        total_against_count=tot_opp,
        mean_qa_recall=mean_rec,
        mean_support_ratio=mean_sup_ratio,
        mean_hallucination_ratio=mean_opp_ratio,
        mean_net_integrity_score=mean_net,
        mean_calibrated_fidelity=mean_fid,
        mean_seam_integrity_rate=mean_seam,
        chunks_with_perfect_recall=perfect_cnt,
        chunks_with_omissions=omission_cnt,
        chunks_with_hallucinations=hallucination_cnt,
        total_original_ideas=tot_orig,
        total_oracle_ideas=tot_assertions,
        total_matched_ideas=tot_sup,
        total_missed_ideas=tot_not + tot_opp,
        total_extra_original_ideas=0,
        overall_success_rate=round(mean_rec * 100.0, 2),
        overall_fail_rate=round((1.0 - mean_rec) * 100.0, 2),
        chunks_with_perfect_match=perfect_cnt,
        total_duration_seconds=tot_dur,
    )

    discrepancy_examples = sorted(
        [it for it in audited_items if it.failed_probes > 0 or it.missed_count > 0],
        key=lambda x: (x.against_count, x.not_mentioned_count, 1.0 - x.qa_recall),
        reverse=True,
    )[:max_examples]

    return ChunkVerificationReport(
        mode="chunk",
        stats=stats,
        audited_chunks=audited_items,
        discrepancy_examples=discrepancy_examples,
        omission_examples=discrepancy_examples,
    )


def export_verification_stats(
    report: Any,
    path: Union[str, Path],
    mode: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export clean statistical summary file (e.g. verification_stats.json)
    standardized across idea verification, chunk verification, and cross-check.
    Used for A/B testing comparing Model A versus Model B.
    """
    eff_mode = mode or getattr(report, "mode", "ideas")
    stats_obj = getattr(report, "stats", None)
    if stats_obj is not None:
        stats_dict = (
            stats_obj.model_dump()
            if hasattr(stats_obj, "model_dump")
            else (stats_obj.__dict__ if hasattr(stats_obj, "__dict__") else dict(stats_obj))
        )
    elif isinstance(report, dict) and "stats" in report:
        stats_dict = report["stats"]
    else:
        stats_dict = {}

    eff_model = (
        model_name
        or (stats_obj and getattr(stats_obj, "model_name", None))
        or stats_dict.get("model_name")
        or "unknown"
    )
    duration = (
        (stats_obj and getattr(stats_obj, "total_duration_seconds", 0.0))
        or stats_dict.get("total_duration_seconds", 0.0)
    )

    stats_data = {
        "mode": eff_mode,
        "model_name": eff_model,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "duration_seconds": duration,
        "stats": stats_dict,
    }
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    return stats_data


def export_verification_debug(
    report: Any,
    path: Union[str, Path],
    mode: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export comprehensive debug JSON (e.g. verification_debug.json) containing:
    - Raw request prompts and raw responses for verification
    - Every sentence and every question/assertion evaluated
    - Full context provided within the requests (full chunk text, formatted ideas claims context)
    - Detailed evaluations and outcomes
    """
    eff_mode = mode or getattr(report, "mode", "ideas")
    stats_obj = getattr(report, "stats", None)
    if stats_obj is not None:
        stats_dict = (
            stats_obj.model_dump()
            if hasattr(stats_obj, "model_dump")
            else (stats_obj.__dict__ if hasattr(stats_obj, "__dict__") else dict(stats_obj))
        )
    elif isinstance(report, dict) and "stats" in report:
        stats_dict = report["stats"]
    else:
        stats_dict = {}

    eff_model = (
        model_name
        or (stats_obj and getattr(stats_obj, "model_name", None))
        or stats_dict.get("model_name")
        or "unknown"
    )

    debug_items = []

    if eff_mode == "chunk" or hasattr(report, "audited_chunks"):
        chunks = getattr(report, "audited_chunks", [])
        if not chunks and isinstance(report, dict):
            chunks = report.get("audited_chunks", [])

        for it in chunks:
            if isinstance(it, dict):
                cid = it.get("chunk_id", "")
                bid = it.get("book_id")
                btitle = it.get("book_title", "Unknown")
                stitle = it.get("section_title", "Unknown")
                ctext = it.get("chunk_text") or it.get("chunk_text_snippet", "")
                sents = it.get("sentences", [])
                ideas_details = it.get("ideas_details", [])
                claims_ctx = it.get("formatted_claims_context")
                probe_prompt = it.get("probe_generation_prompt")
                probe_resp = it.get("probe_generation_response")
                probes = it.get("probes") or it.get("assertions", [])
                evals = it.get("evaluations", [])
                qa_recall = it.get("qa_recall", 0.0)
                net_integrity = it.get("net_integrity_score", 0.0)
                calibrated_fid = it.get("calibrated_fidelity", 0.0)
                seam_rate = it.get("seam_integrity_rate", 1.0)
                sup_cnt = it.get("supported_count", 0)
                not_cnt = it.get("not_mentioned_count", 0)
                against_cnt = it.get("against_count", 0)
                tot_probes = it.get("total_probes", 0)
                dur = it.get("duration_seconds", 0.0)
                server = it.get("server_node")
            else:
                cid = it.chunk_id
                bid = it.book_id
                btitle = it.book_title
                stitle = it.section_title
                ctext = getattr(it, "chunk_text", None) or it.chunk_text_snippet
                sents = getattr(it, "sentences", [])
                ideas_details = getattr(it, "ideas_details", [])
                claims_ctx = getattr(it, "formatted_claims_context", None)
                probe_prompt = getattr(it, "probe_generation_prompt", None)
                probe_resp = getattr(it, "probe_generation_response", None)
                probes = getattr(it, "probes", []) or getattr(it, "assertions", [])
                evals = getattr(it, "evaluations", [])
                qa_recall = it.qa_recall
                net_integrity = it.net_integrity_score
                calibrated_fid = it.calibrated_fidelity
                seam_rate = it.seam_integrity_rate
                sup_cnt = it.supported_count
                not_cnt = it.not_mentioned_count
                against_cnt = it.against_count
                tot_probes = it.total_probes
                dur = it.duration_seconds
                server = it.server_node

            if not sents and ctext:
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctext) if s.strip()]

            parsed_questions = []
            for p in probes:
                if isinstance(p, dict):
                    parsed_questions.append({
                        "question_id": p.get("claim_id", p.get("question_id", 1)),
                        "question": p.get("question") or p.get("assertion", ""),
                        "assertion": p.get("assertion", ""),
                        "source_quote": p.get("source_quote", ""),
                        "is_cross_boundary": p.get("is_cross_boundary", False),
                        "raw_prompt": p.get("raw_prompt"),
                        "raw_response": p.get("raw_response"),
                    })
                else:
                    parsed_questions.append({
                        "question_id": p.claim_id,
                        "question": getattr(p, "question", None) or p.assertion,
                        "assertion": p.assertion,
                        "source_quote": getattr(p, "source_quote", ""),
                        "is_cross_boundary": p.is_cross_boundary,
                        "raw_prompt": getattr(p, "raw_prompt", None),
                        "raw_response": getattr(p, "raw_response", None),
                    })

            probes_eval = []
            for e in evals:
                if isinstance(e, dict):
                    probes_eval.append({
                        "probe_id": e.get("claim_id", e.get("question_id", 1)),
                        "question": e.get("question") or e.get("assertion", ""),
                        "assertion": e.get("assertion", ""),
                        "source_quote": e.get("gold_answer") or e.get("source_quote", ""),
                        "is_cross_boundary": e.get("is_cross_boundary", False),
                        "raw_verifier_prompt": e.get("raw_prompt") or e.get("verifier_prompt", ""),
                        "raw_verifier_response": e.get("raw_response", ""),
                        "result": {
                            "classification": e.get("classification", "NOT_MENTIONED"),
                            "score": e.get("score", -0.25),
                            "verdict": e.get("verdict", "FAIL"),
                            "rationale": e.get("rationale") or e.get("reason", ""),
                        },
                    })
                else:
                    probes_eval.append({
                        "probe_id": e.claim_id,
                        "question": getattr(e, "question", None) or e.assertion,
                        "assertion": e.assertion,
                        "source_quote": getattr(e, "gold_answer", "") or getattr(e, "source_quote", ""),
                        "is_cross_boundary": e.is_cross_boundary,
                        "raw_verifier_prompt": getattr(e, "raw_prompt", "") or e.verifier_prompt,
                        "raw_verifier_response": getattr(e, "raw_response", ""),
                        "result": {
                            "classification": e.classification,
                            "score": e.score,
                            "verdict": e.verdict,
                            "rationale": e.rationale or e.reason,
                        },
                    })

            debug_items.append({
                "chunk_id": cid,
                "book_id": bid,
                "book_title": btitle,
                "section_title": stitle,
                "chunk_text": ctext,
                "sentences": sents,
                "ideas_context": {
                    "extracted_ideas": ideas_details,
                    "formatted_prompt_context": claims_ctx,
                },
                "question_generation": {
                    "raw_prompt": probe_prompt,
                    "raw_response": probe_resp,
                    "parsed_questions": parsed_questions,
                },
                "probes_evaluation": probes_eval,
                "metrics": {
                    "qa_recall": qa_recall,
                    "net_integrity_score": net_integrity,
                    "calibrated_fidelity": calibrated_fid,
                    "seam_integrity_rate": seam_rate,
                    "supported_count": sup_cnt,
                    "not_mentioned_count": not_cnt,
                    "against_count": against_cnt,
                    "total_probes": tot_probes,
                    "duration_seconds": dur,
                    "server_node": server,
                },
            })

    elif eff_mode in ("cross-check", "cross_check") or hasattr(report, "blocks"):
        blocks = getattr(report, "blocks", [])
        if not blocks and isinstance(report, dict):
            blocks = report.get("blocks", [])

        for blk in blocks:
            bdata = blk.model_dump() if hasattr(blk, "model_dump") else (blk if isinstance(blk, dict) else blk.__dict__)
            raw_t = bdata.get("raw_text", "")
            debug_items.append({
                "block_index": bdata.get("block_index"),
                "book_id": bdata.get("book_id"),
                "book_title": bdata.get("book_title"),
                "section_title": bdata.get("section_title"),
                "raw_text": raw_t,
                "sentences": [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_t) if s.strip()],
                "ideas_old": bdata.get("ideas_old", []),
                "ideas_new": bdata.get("ideas_new", []),
                "probes": bdata.get("probes", []),
                "metrics_old": bdata.get("metrics_old", {}),
                "metrics_new": bdata.get("metrics_new", {}),
                "delta_recall": bdata.get("delta_recall", 0.0),
                "delta_net_integrity": bdata.get("delta_net_integrity", 0.0),
            })

    else:
        # Ideas mode
        items = list(getattr(report, "discrepancies", [])) + list(getattr(report, "verified_samples", []))
        if not items and isinstance(report, dict):
            items = list(report.get("discrepancies", [])) + list(report.get("verified_samples", []))

        for it in items:
            if isinstance(it, dict):
                iname = it.get("idea_name", "")
                icat = it.get("idea_category", "Concept")
                isum = it.get("idea_summary", "")
                iwt = it.get("idea_weight", 5)
                cid = it.get("chunk_id", "")
                bid = it.get("book_id")
                btitle = it.get("book_title", "Unknown")
                stitle = it.get("section_title", "Unknown")
                q = it.get("quote", "")
                ctext = it.get("chunk_text") or it.get("chunk_snippet", "")
                sents = it.get("sentences", [])
                prompt = it.get("raw_prompt")
                resp = it.get("raw_response")
                sup = it.get("is_supported", False)
                conf = it.get("confidence", 0.0)
                exp = it.get("explanation", "")
                srv = it.get("server_node")
            else:
                iname = it.idea_name
                icat = getattr(it, "idea_category", "Concept")
                isum = getattr(it, "idea_summary", "")
                iwt = getattr(it, "idea_weight", 5)
                cid = it.chunk_id
                bid = getattr(it, "book_id", None)
                btitle = getattr(it, "book_title", "Unknown")
                stitle = getattr(it, "section_title", "Unknown")
                q = getattr(it, "quote", "")
                ctext = getattr(it, "chunk_text", None) or it.chunk_snippet
                sents = getattr(it, "sentences", [])
                prompt = getattr(it, "raw_prompt", None)
                resp = getattr(it, "raw_response", None)
                sup = it.is_supported
                conf = it.confidence
                exp = it.explanation
                srv = it.server_node

            if not sents and ctext:
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctext) if s.strip()]

            debug_items.append({
                "idea_name": iname,
                "idea_category": icat,
                "idea_summary": isum,
                "idea_weight": iwt,
                "chunk_id": cid,
                "book_id": bid,
                "book_title": btitle,
                "section_title": stitle,
                "quote": q,
                "chunk_text": ctext,
                "sentences": sents,
                "raw_verifier_prompt": prompt,
                "raw_verifier_response": resp,
                "result": {
                    "is_supported": sup,
                    "confidence": conf,
                    "explanation": exp,
                },
                "server_node": srv,
            })

    debug_data = {
        "mode": eff_mode,
        "model_name": eff_model,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stats": stats_dict,
        "debug_items": debug_items,
    }

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)
    return debug_data
