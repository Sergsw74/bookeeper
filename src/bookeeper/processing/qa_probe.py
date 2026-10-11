"""
Natural Language Inference (NLI) Atomic Assertion Oracle for Knowledge Graph Cross-Check Verification.

Evaluates retrieval utility and seam integrity by:
1. Extracting 5 factual atomic declarative assertions directly from unbroken passage W_raw,
   with mandatory multi-sentence/cross-boundary assertions.
2. Directly evaluating semantic entailment against candidate claims context using ternary NLI:
   SUPPORTED, CONTRADICTED, NOT_MENTIONED.
3. Computing deterministic Recall and Seam Integrity without free-form answering drift.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field

from bookeeper.processing.extractor import Concept
from bookeeper.processing.ollama_pool import OllamaPool

logger = logging.getLogger(__name__)


# ==============================================================================
# Ternary Continuous Verification Scoring Model
# ==============================================================================


@dataclass
class EvaluationWeights:
    supported: float = 1.0
    not_mentioned: float = -0.25
    against: float = -1.0


def calculate_ternary_scores(
    results: List[str],
    weights: EvaluationWeights = EvaluationWeights(),
) -> dict:
    """
    Calculates weighted semantic integrity and calibrated fidelity scores.
    results: list of strings, each in ['SUPPORTED', 'NOT_MENTIONED', 'AGAINST', 'CONTRADICTED']
    """
    n_total = len(results)
    if n_total == 0:
        return {
            "total_probes": 0,
            "supported_count": 0,
            "not_mentioned_count": 0,
            "against_count": 0,
            "support_ratio": 0.0,
            "hallucination_ratio": 0.0,
            "net_integrity_score": 0.0,
            "calibrated_fidelity": 0.0,
            "net_score": 0.0,
        }

    # Normalize classifications to canonical forms: SUPPORTED, NOT_MENTIONED, AGAINST
    normalized: List[str] = []
    for r in results:
        r_str = str(r).strip().upper()
        if r_str in ("SUPPORTED", "PASS"):
            normalized.append("SUPPORTED")
        elif r_str in ("AGAINST", "CONTRADICTED", "CONFRONTED"):
            normalized.append("AGAINST")
        else:
            normalized.append("NOT_MENTIONED")

    n_sup = normalized.count("SUPPORTED")
    n_not = normalized.count("NOT_MENTIONED")
    n_opp = normalized.count("AGAINST")

    # 1. Bounded [-1.0, 1.0] Net Score
    raw_score = (
        (n_sup * weights.supported) +
        (n_not * weights.not_mentioned) +
        (n_opp * weights.against)
    )
    net_score = raw_score / n_total

    # 2. Normalized [0.0, 1.0] Benchmark Fidelity
    # Formula: max(0.0, (Sup - 0.25*Not - 1.0*Against) / Total)
    calibrated_fidelity = max(0.0, (n_sup - (0.25 * n_not) - (1.0 * n_opp)) / n_total)

    return {
        "total_probes": n_total,
        "supported_count": n_sup,
        "not_mentioned_count": n_not,
        "against_count": n_opp,
        "support_ratio": round(n_sup / n_total, 4),
        "hallucination_ratio": round(n_opp / n_total, 4),
        "net_integrity_score": round(net_score, 4),
        "calibrated_fidelity": round(calibrated_fidelity, 4),
        "net_score": round(net_score, 4),
    }


# ==============================================================================
# Data Models
# ==============================================================================


class AtomicAssertionItem(BaseModel):
    """An atomic declarative assertion grounded in the raw passage."""

    claim_id: int = 1
    assertion: str = ""
    is_cross_boundary: bool = False
    source_quote: str = ""

    # Backwards compatibility fields for QA probe callers
    question_id: Optional[int] = None
    question: Optional[str] = None
    gold_answer: Optional[str] = None
    is_cross_sentence: Optional[bool] = None
    source_sentence_references: List[str] = Field(default_factory=list)

    def __init__(self, **data: Any):
        # Sync legacy question fields if provided
        if "question_id" in data and "claim_id" not in data:
            data["claim_id"] = data["question_id"]
        if "question" in data and "assertion" not in data:
            data["assertion"] = data["question"]
        if "is_cross_sentence" in data and "is_cross_boundary" not in data:
            data["is_cross_boundary"] = data["is_cross_sentence"]
        if "gold_answer" in data and "source_quote" not in data:
            data["source_quote"] = data["gold_answer"]
        if "source_sentence_references" in data and data["source_sentence_references"] and not data.get("source_quote"):
            data["source_quote"] = data["source_sentence_references"][0]

        # Sync forward
        if "claim_id" in data and "question_id" not in data:
            data["question_id"] = data["claim_id"]
        if "assertion" in data and "question" not in data:
            data["question"] = data["assertion"]
        if "is_cross_boundary" in data and "is_cross_sentence" not in data:
            data["is_cross_sentence"] = data["is_cross_boundary"]
        if "source_quote" in data and "gold_answer" not in data:
            data["gold_answer"] = data["source_quote"]
        if "source_quote" in data and not data.get("source_sentence_references"):
            data["source_sentence_references"] = [data["source_quote"]] if data["source_quote"] else []

        super().__init__(**data)


QAProbeItem = AtomicAssertionItem


class NLIAssertionEvaluation(BaseModel):
    """NLI entailment classification of an atomic assertion against knowledge base claims."""

    claim_id: int = 1
    assertion: str = ""
    is_cross_boundary: bool = False
    classification: Literal["SUPPORTED", "CONTRADICTED", "NOT_MENTIONED", "AGAINST"] = "NOT_MENTIONED"
    score: float = -0.25
    verdict: Literal["PASS", "FAIL"] = "FAIL"
    rationale: str = ""
    verifier_prompt: str = ""

    # Backwards compatibility fields for QA probe callers
    question_id: Optional[int] = None
    question: Optional[str] = None
    gold_answer: Optional[str] = None
    is_cross_sentence: Optional[bool] = None
    predicted_answer: str = ""
    reason: str = ""
    answering_prompt: str = ""
    judge_prompt: str = ""

    def __init__(self, **data: Any):
        if "question_id" in data and "claim_id" not in data:
            data["claim_id"] = data["question_id"]
        if "question" in data and "assertion" not in data:
            data["assertion"] = data["question"]
        if "is_cross_sentence" in data and "is_cross_boundary" not in data:
            data["is_cross_boundary"] = data["is_cross_sentence"]
        if "reason" in data and "rationale" not in data:
            data["rationale"] = data["reason"]
        if "classification" in data and "verdict" not in data:
            data["verdict"] = "PASS" if data["classification"] == "SUPPORTED" else "FAIL"
        elif "verdict" in data and "classification" not in data:
            data["classification"] = "SUPPORTED" if data["verdict"] == "PASS" else "NOT_MENTIONED"

        # Sync numeric encoding: SUPPORTED=+1.0, CONTRADICTED/AGAINST=-1.0, NOT_MENTIONED=-0.25
        if "score" not in data or data["score"] is None:
            cls_upper = str(data.get("classification", "")).strip().upper()
            if cls_upper in ("SUPPORTED", "PASS"):
                data["score"] = 1.0
            elif cls_upper in ("CONTRADICTED", "AGAINST", "CONFRONTED"):
                data["score"] = -1.0
            else:
                data["score"] = -0.25
        elif "classification" not in data:
            sc = float(data["score"])
            if sc > 0:
                data["classification"] = "SUPPORTED"
                data["verdict"] = "PASS"
            elif sc <= -0.5:
                data["classification"] = "CONTRADICTED"
                data["verdict"] = "FAIL"
            else:
                data["classification"] = "NOT_MENTIONED"
                data["verdict"] = "FAIL"

        # Sync forward
        if "claim_id" in data and "question_id" not in data:
            data["question_id"] = data["claim_id"]
        if "assertion" in data and "question" not in data:
            data["question"] = data["assertion"]
        if "is_cross_boundary" in data and "is_cross_sentence" not in data:
            data["is_cross_sentence"] = data["is_cross_boundary"]
        if "rationale" in data and "reason" not in data:
            data["reason"] = data["rationale"]
        if "classification" in data and not data.get("predicted_answer"):
            data["predicted_answer"] = data["classification"]
        if "verifier_prompt" in data:
            if not data.get("answering_prompt"):
                data["answering_prompt"] = data["verifier_prompt"]
            if not data.get("judge_prompt"):
                data["judge_prompt"] = data["verifier_prompt"]

        super().__init__(**data)


QAProbeEvaluation = NLIAssertionEvaluation


class NLISystemBlockMetrics(BaseModel):
    """NLI entailment results for a single system on an audited block."""

    system: Literal["old", "new"]
    total_probes: int = 0
    passed_probes: int = 0
    failed_probes: int = 0
    qa_recall: float = 0.0  # passed_probes / total_probes
    cross_sentence_total: int = 0
    cross_sentence_passed: int = 0
    seam_integrity_rate: float = 0.0  # cross_sentence_passed / cross_sentence_total
    evaluations: List[NLIAssertionEvaluation] = Field(default_factory=list)

    # Continuous ternary verification metrics
    supported_count: int = 0
    not_mentioned_count: int = 0
    against_count: int = 0
    support_ratio: float = 0.0
    hallucination_ratio: float = 0.0
    net_integrity_score: float = 0.0
    calibrated_fidelity: float = 0.0
    seam_net_integrity: float = 0.0
    seam_calibrated_fidelity: float = 0.0


QASystemBlockMetrics = NLISystemBlockMetrics


# ==============================================================================
# Claims Context Formatting
# ==============================================================================


def format_ideas_for_context(ideas: List[Any]) -> str:
    """
    Format deduplicated Concept objects or dictionary ideas into structured Markdown blocks
    including concept name, short description, full description, and supporting quotes.
    """
    if not ideas:
        return "(No knowledge claims extracted for this block)"

    formatted_blocks = []
    for idx, idea in enumerate(ideas, 1):
        if isinstance(idea, Concept):
            name = idea.name or "Unnamed Concept"
            brief = (getattr(idea, "brief_description", "") or getattr(idea, "summary", "") or "").strip()
            detailed = (getattr(idea, "detailed_explanation", "") or "").strip()
            quote = (getattr(idea, "supporting_quote", "") or getattr(idea, "quote", "") or "").strip()
        elif isinstance(idea, dict):
            name = idea.get("name") or "Unnamed Concept"
            brief = (idea.get("brief_description") or idea.get("summary") or "").strip()
            detailed = (idea.get("detailed_explanation") or "").strip()
            quote = (idea.get("supporting_quote") or idea.get("quote") or "").strip()
        else:
            name = str(idea).strip()
            brief = ""
            detailed = ""
            quote = ""

        block_lines = [f"[{idx}] Concept: {name}"]
        if brief and detailed and brief != detailed:
            block_lines.append(f"    Short Description: {brief}")
            block_lines.append(f"    Explanation: {detailed}")
        elif detailed:
            block_lines.append(f"    Explanation: {detailed}")
        elif brief:
            block_lines.append(f"    Explanation: {brief}")
        else:
            block_lines.append("    Explanation: (No description provided)")
        if quote:
            block_lines.append(f'    Direct Context/Quote: "{quote}"')
        formatted_blocks.append("\n".join(block_lines))

    return "\n\n".join(formatted_blocks)


def format_ideas_as_claims(ideas: List[Any]) -> str:
    """Backward-compatible alias for format_ideas_for_context."""
    return format_ideas_for_context(ideas)


# ==============================================================================
# Helper for JSON extraction from LLM response
# ==============================================================================


def clean_llm_json_response(raw_text: str) -> str:
    """Clean markdown code fences and thinking tags from LLM response."""
    text = raw_text.strip()
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


def parse_json_array_safely(text: str) -> List[Dict[str, Any]]:
    """Safely parse a JSON array from LLM response text with fallback heuristics."""
    clean = clean_llm_json_response(text)
    try:
        data = json.loads(clean)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("assertions", "claims", "probes", "questions", "items", "qa_pairs"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
    except Exception:
        pass

    # Regex extraction of JSON array
    match = re.search(r"\[\s*\{.*\}\s*\]", clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return []


# ==============================================================================
# 1. Atomic Assertion Generation
# ==============================================================================


def build_assertion_generator_prompt(canonical_window: str, num_assertions: int = 5) -> str:
    """
    Construct prompt sent to model to decompose passage into factual atomic assertions.
    Enforces independent declarative statements, cross-sentence dependencies, and forbids
    sentence-order/grammar probes.
    """
    return (
        "You are an expert NLP benchmark engineer.\n"
        f"Analyze the passage below and extract exactly {num_assertions} clear, factual, declarative statements "
        "(atomic claims) that represent the key events, entity actions, and conditions described.\n\n"
        "CONSTRAINTS:\n"
        "1. Every claim must be an independent, self-contained declarative sentence "
        '(e.g., "Sircolo is a male marsh harrier who agreed to carry the group to firm ground").\n'
        "2. At least 3 claims MUST be CROSS-SENTENCE (connecting a premise or condition with an outcome).\n"
        "3. DO NOT output questions. Output ONLY declarative assertions.\n"
        '4. DO NOT make statements about sentence order, grammar, or punctuation (no "Sentence A precedes Sentence B").\n\n'
        f"Passage:\n\"\"\"\n{canonical_window[:7000]}\n\"\"\"\n\n"
        "Return strictly valid JSON:\n"
        "[\n"
        "  {\n"
        '    "claim_id": 1,\n'
        '    "assertion": "<factual declarative statement>",\n'
        '    "is_cross_boundary": true,\n'
        '    "source_quote": "<verbatim sentence from passage>"\n'
        "  }\n"
        "]\n"
    )


def build_generator_prompt(w_raw: str, num_questions: int = 5) -> str:
    """Backward-compatible alias for build_assertion_generator_prompt."""
    return build_assertion_generator_prompt(canonical_window=w_raw, num_assertions=num_questions)


def generate_atomic_assertions(
    canonical_window: str,
    pool: OllamaPool,
    model_name: str,
    num_assertions: int = 5,
) -> List[AtomicAssertionItem]:
    """
    Extract factual atomic assertions directly from unbroken passage canonical_window.
    Targets key events, entity actions, and conditions with mandatory cross-sentence claims.
    """
    prompt = build_assertion_generator_prompt(canonical_window, num_assertions=num_assertions)

    try:
        raw_text, _ = pool.generate(model_name, prompt, temperature=0.0)
        items = parse_json_array_safely(raw_text)

        assertions: List[AtomicAssertionItem] = []
        for idx, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            cid = item.get("claim_id", item.get("question_id", idx))
            try:
                cid = int(cid)
            except (ValueError, TypeError):
                cid = idx

            assertion_text = str(
                item.get("assertion") or item.get("claim") or item.get("question", "")
            ).strip()

            is_cross = bool(
                item.get("is_cross_boundary", item.get("is_cross_sentence", False))
            )

            source_quote = str(item.get("source_quote") or item.get("gold_answer") or "").strip()
            if not source_quote:
                refs = item.get("source_sentence_references", [])
                if isinstance(refs, list) and refs:
                    source_quote = str(refs[0]).strip()
                elif isinstance(refs, str) and refs.strip():
                    source_quote = refs.strip()

            if assertion_text:
                assertions.append(
                    AtomicAssertionItem(
                        claim_id=cid,
                        assertion=assertion_text,
                        is_cross_boundary=is_cross,
                        source_quote=source_quote,
                    )
                )

        if assertions:
            # Enforce at least some cross-boundary assertions if model omitted flags
            if not any(a.is_cross_boundary for a in assertions) and len(assertions) >= 2:
                for idx, a in enumerate(assertions):
                    # Flag assertions containing causal or conditional markers
                    text_lower = a.assertion.lower()
                    if any(w in text_lower for w in ("because", "when", "after", "in order to", "led to", "resulting in", "so that", "if")):
                        a.is_cross_boundary = True
                if not any(a.is_cross_boundary for a in assertions):
                    for a in assertions[: min(3, len(assertions))]:
                        a.is_cross_boundary = True
            return assertions
    except Exception as exc:
        logger.warning(f"Atomic assertion generation failed: {exc}")

    # Deterministic fallback extracting factual sentences directly from passage
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", canonical_window) if len(s.strip()) > 15]
    if len(sentences) >= 2:
        return [
            AtomicAssertionItem(
                claim_id=1,
                assertion=sentences[0],
                is_cross_boundary=False,
                source_quote=sentences[0],
            ),
            AtomicAssertionItem(
                claim_id=2,
                assertion=f"{sentences[0]} and {sentences[1]}",
                is_cross_boundary=True,
                source_quote=sentences[1],
            ),
        ]

    fallback_text = canonical_window.strip()[:200]
    return [
        AtomicAssertionItem(
            claim_id=1,
            assertion=fallback_text,
            is_cross_boundary=False,
            source_quote=fallback_text,
        )
    ]


def generate_qa_probes(
    w_raw: str,
    pool: OllamaPool,
    model_name: str,
    num_questions: int = 5,
) -> List[AtomicAssertionItem]:
    """Backward-compatible alias for generate_atomic_assertions."""
    return generate_atomic_assertions(
        canonical_window=w_raw,
        pool=pool,
        model_name=model_name,
        num_assertions=num_questions,
    )


# ==============================================================================
# 2. Ternary Entailment Verifier Prompt & Classification
# ==============================================================================


def build_nli_verifier_prompt(
    formatted_claims: str,
    assertion: str,
    question: Optional[str] = None,
) -> str:
    """
    Construct prompt for Natural Language Inference entailment verification against knowledge base claims.
    Supports either an assertion, a question, or a question + assertion pair.
    """
    target_lines = []
    if question and question.strip() and question.strip() != assertion.strip():
        target_lines.append(f'Question: "{question.strip()}"')
        target_lines.append(f'Target Assertion: "{assertion.strip()}"')
    else:
        target_lines.append(f'Target Assertion: "{assertion.strip()}"')

    target_block = "\n".join(target_lines)

    return (
        "You are an objective Natural Language Inference (NLI) evaluator.\n"
        "Given the Extracted Knowledge Base below, determine whether the Target Question / Assertion is supported by the verified claims.\n\n"
        f"Extracted Knowledge Base:\n\"\"\"\n{formatted_claims}\n\"\"\"\n\n"
        f"Target to Verify:\n{target_block}\n\n"
        "Choose exactly ONE classification:\n"
        "- SUPPORTED: The question's answer or assertion's core facts and relationships are explicitly stated, explained, or logically entailed by the knowledge base.\n"
        "- CONTRADICTED: The knowledge base explicitly states something that conflicts with or negates the question/assertion.\n"
        "- NOT_MENTIONED: The knowledge base lacks sufficient facts or explanations to verify/answer the question/assertion.\n\n"
        "Return strictly valid JSON:\n"
        "{\n"
        '  "classification": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED",\n'
        '  "rationale": "<brief 1-sentence reason>"\n'
        "}\n"
    )


def verify_assertion_entailment(
    claims_context: str,
    assertion: str,
    pool: OllamaPool,
    model_name: str,
    question: Optional[str] = None,
) -> Tuple[Literal["SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"], str]:
    """
    Classify whether Target Assertion/Question is SUPPORTED, CONTRADICTED, or NOT_MENTIONED
    by the extracted knowledge base context.
    Returns (classification, rationale).
    """
    if not claims_context.strip() or "(No knowledge claims extracted" in claims_context:
        return "NOT_MENTIONED", "Knowledge base contains no claims for this block."

    if not assertion.strip() and not (question and question.strip()):
        return "NOT_MENTIONED", "Target assertion and question are empty."

    prompt = build_nli_verifier_prompt(claims_context, assertion, question=question)

    try:
        raw_text, _ = pool.generate(model_name, prompt, temperature=0.0)
        clean = clean_llm_json_response(raw_text)
        data = {}
        try:
            data = json.loads(clean)
        except Exception:
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    pass

        raw_cls = str(data.get("classification", "")).strip().upper()
        rationale = str(data.get("rationale", "")).strip()

        if "SUPPORTED" in raw_cls and "NOT_MENTIONED" not in raw_cls and "NOT SUPPORTED" not in raw_cls and "UNSUPPORTED" not in raw_cls:
            return "SUPPORTED", rationale or "Entailed by claims."
        elif "CONTRADICT" in raw_cls or "AGAINST" in raw_cls or "CONFRONT" in raw_cls:
            return "CONTRADICTED", rationale or "Contradicts claims."
        elif "NOT_MENTIONED" in raw_cls or "UNSUPPORTED" in raw_cls:
            return "NOT_MENTIONED", rationale or "Not mentioned in claims."

        # Heuristic fallback
        clean_upper = clean.upper()
        if "SUPPORTED" in clean_upper and "NOT_MENTIONED" not in clean_upper and "NOT SUPPORTED" not in clean_upper and "UNSUPPORTED" not in clean_upper:
            return "SUPPORTED", rationale or clean[:120]
        elif "CONTRADICT" in clean_upper or "AGAINST" in clean_upper or "CONFRONT" in clean_upper:
            return "CONTRADICTED", rationale or clean[:120]
        else:
            return "NOT_MENTIONED", rationale or clean[:120]
    except Exception as exc:
        logger.warning(f"NLI classification failed for assertion '{assertion[:40]}': {exc}")
        return "NOT_MENTIONED", f"Evaluation error: {exc}"


def generate_chunk_probes(
    chunk_text: str,
    pool: OllamaPool,
    model_name: str,
    num_probes: int = 1,
) -> List[AtomicAssertionItem]:
    """
    Formulate question-assertion probe(s) for a selected chunk using the Oracle model
    to test whether the chunk's core idea, event, or mechanism was captured.
    """
    prompt = (
        "You are an expert NLP benchmark auditor evaluating knowledge graph extraction quality.\n"
        f"Analyze the text chunk below and formulate exactly {num_probes} clear, specific question(s) "
        "that test whether the core conceptual idea, key event, mechanism, or principle described in this chunk "
        "has been captured and preserved.\n\n"
        "CONSTRAINTS:\n"
        "1. Focus on the PRIMARY concept or substantive factual relationship described in the chunk.\n"
        "2. Formulate a specific question that tests this core knowledge in natural language.\n"
        "3. Provide the factual declarative statement (assertion) that directly answers the question.\n"
        "4. Include a direct verbatim quote from the chunk supporting the answer.\n\n"
        f"Chunk Text:\n\"\"\"\n{chunk_text[:5000]}\n\"\"\"\n\n"
        "Return strictly valid JSON:\n"
        "[\n"
        "  {\n"
        '    "question_id": 1,\n'
        '    "question": "<specific factual question testing the chunk\'s core concept>",\n'
        '    "assertion": "<concise declarative factual statement answering the question>",\n'
        '    "source_quote": "<verbatim sentence from chunk text>"\n'
        "  }\n"
        "]\n"
    )

    try:
        raw_text, _ = pool.generate(model_name, prompt, temperature=0.0)
        items = parse_json_array_safely(raw_text)

        probes: List[AtomicAssertionItem] = []
        for idx, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            qid = item.get("question_id", item.get("claim_id", idx))
            try:
                qid = int(qid)
            except (ValueError, TypeError):
                qid = idx

            question_text = str(item.get("question") or "").strip()
            assertion_text = str(item.get("assertion") or item.get("claim") or question_text).strip()
            if not question_text and assertion_text:
                question_text = f"Is the following fact verified: {assertion_text}"

            source_quote = str(item.get("source_quote") or item.get("gold_answer") or "").strip()
            if not source_quote:
                refs = item.get("source_sentence_references", [])
                if isinstance(refs, list) and refs:
                    source_quote = str(refs[0]).strip()
                elif isinstance(refs, str) and refs.strip():
                    source_quote = refs.strip()

            if question_text or assertion_text:
                probes.append(
                    AtomicAssertionItem(
                        claim_id=qid,
                        question_id=qid,
                        question=question_text or assertion_text,
                        assertion=assertion_text or question_text,
                        source_quote=source_quote,
                        is_cross_boundary=False,
                    )
                )

        if probes:
            return probes
    except Exception as exc:
        logger.warning(f"Chunk probe generation failed: {exc}")

    # Deterministic fallback extracting key sentence from chunk
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if len(s.strip()) > 15]
    if sentences:
        s0 = sentences[0]
        words = [w for w in re.findall(r"[A-Z][a-z]+|\b\w{4,}\b", s0) if w.lower() not in ("what", "that", "this", "there", "they", "then", "with", "from")]
        anchor = words[0] if words else "the core concept"
        return [
            AtomicAssertionItem(
                claim_id=1,
                question_id=1,
                question=f"What key fact or principle is established regarding {anchor}?",
                assertion=s0,
                source_quote=s0,
                is_cross_boundary=False,
            )
        ]

    return [
        AtomicAssertionItem(
            claim_id=1,
            question_id=1,
            question="What is the central concept discussed in this chunk?",
            assertion=chunk_text[:150],
            source_quote=chunk_text[:150],
            is_cross_boundary=False,
        )
    ]


# Legacy helper functions preserved for backward compatibility
def build_answering_prompt(claims_context: str, question: str) -> str:
    """Legacy helper preserved for backward compatibility."""
    return build_nli_verifier_prompt(claims_context, question)


def answer_probe(
    claims_context: str,
    question: str,
    pool: OllamaPool,
    model_name: str,
) -> str:
    """Legacy helper preserved for backward compatibility."""
    cls, _ = verify_assertion_entailment(claims_context, question, pool, model_name)
    return "INSUFFICIENT_INFORMATION" if cls == "NOT_MENTIONED" else cls


def build_judge_prompt(gold_answer: str, candidate_answer: str, w_raw: str) -> str:
    """Legacy helper preserved for backward compatibility."""
    return (
        "You are an objective judge evaluating whether a candidate answer accurately conveys ground truth facts.\n\n"
        f'Ground Truth Answer: "{gold_answer}"\n'
        f'Candidate Answer: "{candidate_answer}"\n\n'
        f"Passage Context (for reference):\n\"\"\"\n{w_raw[:4000]}\n\"\"\"\n\n"
        "Judge Verdict Guidelines:\n"
        '- If Candidate Answer conveys the substantive answer to the question asked, mark as "PASS".\n'
        "- DO NOT penalize the Candidate Answer for omitting the premise or condition if the question already stated that condition.\n"
        '- Mark as "FAIL" ONLY if the answer is "INSUFFICIENT_INFORMATION", directly contradicts the ground truth, or asserts a hallucinated fact.\n\n'
        "Return strictly valid JSON:\n"
        "{\n"
        '  "verdict": "PASS" | "FAIL",\n'
        '  "reason": "concise rationale"\n'
        "}\n"
    )


def judge_probe_answer(
    gold_answer: str,
    predicted_answer: str,
    w_raw: str,
    pool: OllamaPool,
    model_name: str,
) -> Tuple[Literal["PASS", "FAIL"], str]:
    """Legacy helper preserved for backward compatibility."""
    pred_clean = predicted_answer.strip()
    if not pred_clean:
        return "FAIL", "Candidate answer is empty."
    if pred_clean.upper().startswith("INSUFFICIENT_INFORMATION") or "INSUFFICIENT_INFORMATION" in pred_clean.upper():
        return "FAIL", "Candidate indicated insufficient information in extracted claims."

    prompt = build_judge_prompt(gold_answer, pred_clean, w_raw)
    try:
        raw_text, _ = pool.generate(model_name, prompt, temperature=0.0)
        clean = clean_llm_json_response(raw_text)
        data = json.loads(clean)
        verdict = str(data.get("verdict", "")).strip().upper()
        if verdict not in ("PASS", "FAIL"):
            verdict = "PASS" if "pass" in verdict.lower() else "FAIL"
        reason = str(data.get("reason", "")).strip()
        return verdict, reason  # type: ignore
    except Exception as exc:
        logger.warning(f"Judging probe answer failed: {exc}")
        return "FAIL", f"Judge evaluation error: {exc}"


# ==============================================================================
# 3. End-to-End Block Atomic Assertion NLI Evaluation
# ==============================================================================


def evaluate_claims_atomic_assertions(
    w_raw: str,
    ideas: List[Any],
    pool: OllamaPool,
    model_name: str,
    assertions: Optional[List[AtomicAssertionItem]] = None,
    system_label: Literal["old", "new"] = "old",
    num_probes: int = 1,
    is_chunk_mode: bool = False,
) -> Tuple[List[AtomicAssertionItem], NLISystemBlockMetrics]:
    """
    Evaluate NLI entailment of atomic assertions/questions for W_raw against a single claims knowledge base.
    Returns:
      (assertions, system_metrics)
    """
    if assertions is None:
        if is_chunk_mode:
            assertions = generate_chunk_probes(w_raw, pool, model_name, num_probes=num_probes)
        else:
            assertions = generate_atomic_assertions(w_raw, pool, model_name)

    claims = format_ideas_as_claims(ideas)
    evals: List[NLIAssertionEvaluation] = []

    for a in assertions:
        q_text = getattr(a, "question", None) or a.assertion
        prompt = build_nli_verifier_prompt(claims, a.assertion, question=q_text)
        cls, rat = verify_assertion_entailment(claims, a.assertion, pool, model_name, question=q_text)
        verdict: Literal["PASS", "FAIL"] = "PASS" if cls == "SUPPORTED" else "FAIL"

        evals.append(
            NLIAssertionEvaluation(
                claim_id=a.claim_id,
                assertion=a.assertion,
                is_cross_boundary=a.is_cross_boundary,
                classification=cls,
                verdict=verdict,
                rationale=rat,
                verifier_prompt=prompt,
                # compatibility
                question_id=a.claim_id,
                question=q_text,
                gold_answer=a.source_quote or a.assertion,
                is_cross_sentence=a.is_cross_boundary,
                predicted_answer=cls,
                reason=rat,
                answering_prompt=prompt,
                judge_prompt=prompt,
            )
        )

    tot = len(assertions)
    cross_tot = sum(1 for a in assertions if a.is_cross_boundary)
    passed = sum(1 for e in evals if e.verdict == "PASS")
    cross_passed = sum(1 for e in evals if e.is_cross_boundary and e.verdict == "PASS")
    t_scores = calculate_ternary_scores([e.classification for e in evals])
    cross_evals = [e for e in evals if e.is_cross_boundary]
    cross_t = calculate_ternary_scores([e.classification for e in cross_evals])

    metrics = NLISystemBlockMetrics(
        system=system_label,
        total_probes=tot,
        passed_probes=passed,
        failed_probes=tot - passed,
        qa_recall=round(passed / tot, 4) if tot > 0 else 0.0,
        cross_sentence_total=cross_tot,
        cross_sentence_passed=cross_passed,
        seam_integrity_rate=round(cross_passed / cross_tot, 4) if cross_tot > 0 else 1.0,
        evaluations=evals,
        supported_count=t_scores["supported_count"],
        not_mentioned_count=t_scores["not_mentioned_count"],
        against_count=t_scores["against_count"],
        support_ratio=t_scores["support_ratio"],
        hallucination_ratio=t_scores["hallucination_ratio"],
        net_integrity_score=t_scores["net_integrity_score"],
        calibrated_fidelity=t_scores["calibrated_fidelity"],
        seam_net_integrity=cross_t["net_integrity_score"],
        seam_calibrated_fidelity=cross_t["calibrated_fidelity"],
    )

    return assertions, metrics


def evaluate_block_atomic_assertions(
    w_raw: str,
    ideas_old: List[Any],
    ideas_new: List[Any],
    pool: OllamaPool,
    model_name: str,
    assertions: Optional[List[AtomicAssertionItem]] = None,
) -> Tuple[List[AtomicAssertionItem], NLISystemBlockMetrics, NLISystemBlockMetrics]:
    """
    Generate atomic assertions for W_raw (or use provided ones), classify entailment against
    Old and New claims knowledge bases, and compute recall & seam integrity metrics.
    Returns:
      (assertions, old_system_metrics, new_system_metrics)
    """
    if assertions is None:
        assertions = generate_atomic_assertions(w_raw, pool, model_name)

    _, metrics_old = evaluate_claims_atomic_assertions(
        w_raw=w_raw,
        ideas=ideas_old,
        pool=pool,
        model_name=model_name,
        assertions=assertions,
        system_label="old",
    )
    _, metrics_new = evaluate_claims_atomic_assertions(
        w_raw=w_raw,
        ideas=ideas_new,
        pool=pool,
        model_name=model_name,
        assertions=assertions,
        system_label="new",
    )

    return assertions, metrics_old, metrics_new


def evaluate_block_qa_probes(
    w_raw: str,
    ideas_old: List[Any],
    ideas_new: List[Any],
    pool: OllamaPool,
    model_name: str,
    probes: Optional[List[AtomicAssertionItem]] = None,
) -> Tuple[List[AtomicAssertionItem], NLISystemBlockMetrics, NLISystemBlockMetrics]:
    """Backward-compatible wrapper for evaluate_block_atomic_assertions."""
    return evaluate_block_atomic_assertions(
        w_raw=w_raw,
        ideas_old=ideas_old,
        ideas_new=ideas_new,
        pool=pool,
        model_name=model_name,
        assertions=probes,
    )
