"""
QA Probe Oracle for Knowledge Graph Cross-Check Verification.

Evaluates retrieval utility and seam integrity by:
1. Generating fact-based QA probes directly from unbroken passage W_raw,
   with mandatory multi-sentence/cross-boundary probes.
2. Executing constrained closed-book answering using ONLY candidate claims as context.
3. Conducting objective arbitration with an Oracle judge against gold ground-truth answers.
"""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from bookeeper.processing.extractor import Concept
from bookeeper.processing.ollama_pool import OllamaPool

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Models
# ==============================================================================


class QAProbeItem(BaseModel):
    """A probe question-answer pair grounded in the raw passage."""

    question_id: int
    question: str
    gold_answer: str
    is_cross_sentence: bool = False
    source_sentence_references: List[str] = Field(default_factory=list)


class QAProbeEvaluation(BaseModel):
    """Evaluation of a single system's response to a probe question."""

    question_id: int
    question: str
    gold_answer: str
    is_cross_sentence: bool
    predicted_answer: str
    verdict: Literal["PASS", "FAIL"]
    reason: str = ""
    answering_prompt: str = ""
    judge_prompt: str = ""


class QASystemBlockMetrics(BaseModel):
    """QA evaluation results for a single system on an audited block."""

    system: Literal["old", "new"]
    total_probes: int = 0
    passed_probes: int = 0
    failed_probes: int = 0
    qa_recall: float = 0.0  # passed_probes / total_probes
    cross_sentence_total: int = 0
    cross_sentence_passed: int = 0
    seam_integrity_rate: float = 0.0  # cross_sentence_passed / cross_sentence_total
    evaluations: List[QAProbeEvaluation] = Field(default_factory=list)


# ==============================================================================
# Claims Context Formatting
# ==============================================================================


def format_ideas_for_context(ideas: List[Any]) -> str:
    """
    Format deduplicated Concept objects or dictionary ideas into structured Markdown blocks
    including supporting quotes and detailed explanations.
    """
    if not ideas:
        return "(No knowledge claims extracted for this block)"

    formatted_blocks = []
    for idx, idea in enumerate(ideas, 1):
        if isinstance(idea, Concept):
            name = idea.name or "Unnamed Concept"
            explanation = (
                getattr(idea, "detailed_explanation", "")
                or getattr(idea, "brief_description", "")
                or getattr(idea, "summary", "")
                or ""
            ).strip()
            quote = (getattr(idea, "supporting_quote", "") or getattr(idea, "quote", "") or "").strip()
        elif isinstance(idea, dict):
            name = idea.get("name") or "Unnamed Concept"
            explanation = (
                idea.get("detailed_explanation")
                or idea.get("brief_description")
                or idea.get("summary")
                or ""
            ).strip()
            quote = (idea.get("supporting_quote") or idea.get("quote") or "").strip()
        else:
            name = str(idea).strip()
            explanation = ""
            quote = ""

        block = f"[{idx}] Concept: {name}\n    Explanation: {explanation}"
        if quote:
            block += f'\n    Direct Context/Quote: "{quote}"'
        formatted_blocks.append(block)

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
            # Check for keys like 'probes', 'questions', 'items'
            for key in ("probes", "questions", "items", "qa_pairs"):
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
# 1. Targeted QA Probe Generation
# ==============================================================================


# ==============================================================================
# 1. Targeted QA Probe Generation
# ==============================================================================


def build_generator_prompt(w_raw: str, num_questions: int = 5) -> str:
    """Construct the complete prompt sent to the Oracle model to generate relational QA probes."""
    return (
        "You are an expert NLP benchmark engineer evaluating a conceptual Knowledge Base and Knowledge Graph.\n"
        f"Read the passage below and construct {num_questions} fact-based Question-Answer probes testing the retention of core knowledge.\n\n"
        "### PROBE SELECTION CRITERIA (WHAT TO ASK):\n"
        "1. Focus strictly on:\n"
        "   - ENTITY RELATIONSHIPS: Functional, hierarchical, causal, or social links between characters, groups, systems, or components.\n"
        "   - CAUSAL CHAINS & STATE CHANGES: Why an action occurred, what decision was made, what condition was triggered, and the direct consequence.\n"
        "   - STRUCTURAL RULES & THEMATIC MECHANISMS: Explicit operational rules, lore constraints, or tactical configurations.\n"
        "2. At least 3 questions MUST be CROSS-BOUNDARY:\n"
        "   - They must require synthesizing facts that span across multiple sentences (e.g., premise or condition in Sentence A, outcome or qualification in Sentence B).\n\n"
        "### CRITICAL PROBE REQUIREMENTS:\n"
        "- DO NOT quote raw sentences in the question (e.g., NEVER ask \"What relationship connects 'Quote A' and 'Quote B'?\").\n"
        '- Frame all questions around entities, actions, or decisions in natural language (e.g., "Why did X decide to do Y?" or "What consequence occurred after Z?").\n'
        '- Do NOT generate questions testing the physical order of sentences in the text (e.g., no "precedes" or chronological sentence order questions).\n\n'
        "### NEGATIVE CONSTRAINTS (STRICTLY FORBIDDEN):\n"
        "- NO TRIVIA: Do not ask about superficial, incidental dialogue details, single-use slang, minor character quips, or verbatim insult nicknames "
        '(e.g., do NOT ask "What did X call Y?", "What was the exact phrase shouted?", or "What specific insult was used?").\n'
        '- NO GENERIC QUESTIONS: Avoid high-level vagueness like "What is the main topic of this paragraph?", "What are the characters doing?", or "Summarize the text".\n'
        "- NO UNSUPPORTED INFERENCES: Every probe must have unambiguous, objective factual ground truth directly stated in the text.\n\n"
        "### CITATION / REFERENCE REQUIREMENT:\n"
        "For every generated probe, you MUST provide the exact verbatim excerpt or sentence(s) from the passage that supply the ground truth. "
        "If the probe is cross-sentence, you must provide the separate excerpts that are being bridged.\n\n"
        f"Passage:\n\"\"\"\n{w_raw[:7000]}\n\"\"\"\n\n"
        "Return strictly valid JSON matching this schema:\n"
        "[\n"
        "  {\n"
        '    "question_id": 1,\n'
        '    "question": "<specific, non-trivial relational or causal question in natural language>",\n'
        '    "gold_answer": "<concise 1-2 sentence ground truth containing the necessary facts>",\n'
        '    "is_cross_sentence": true,\n'
        '    "source_sentence_references": [\n'
        '      "<exact verbatim quote of premise or first fact from passage>",\n'
        '      "<exact verbatim quote of consequence or second fact from passage (if cross-sentence)>"\n'
        "    ]\n"
        "  }\n"
        "]\n"
    )


def generate_qa_probes(
    w_raw: str,
    pool: OllamaPool,
    model_name: str,
    num_questions: int = 5,
) -> List[QAProbeItem]:
    """
    Generate fact-based QA probes directly from unbroken passage W_raw (canonical_window).
    Targets structural concepts, actions, and causal relationships with explicit negative constraints
    and verbatim sentence references.
    """
    prompt = build_generator_prompt(w_raw, num_questions=num_questions)

    try:
        raw_text, _ = pool.generate(model_name, prompt, temperature=0.0)
        items = parse_json_array_safely(raw_text)

        probes: List[QAProbeItem] = []
        for idx, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            qid = item.get("question_id", idx)
            try:
                qid = int(qid)
            except (ValueError, TypeError):
                qid = idx
            q = str(item.get("question", "")).strip()
            a = str(item.get("gold_answer", item.get("answer", ""))).strip()
            is_cross = bool(item.get("is_cross_sentence", False))
            refs = item.get("source_sentence_references", [])
            if isinstance(refs, str):
                refs = [refs]
            elif not isinstance(refs, list):
                refs = []
            refs = [str(r).strip() for r in refs if str(r).strip()]

            if q and a:
                probes.append(
                    QAProbeItem(
                        question_id=qid,
                        question=q,
                        gold_answer=a,
                        is_cross_sentence=is_cross,
                        source_sentence_references=refs,
                    )
                )

        if probes:
            # Enforce at least some cross-sentence probes if model didn't set flags
            if not any(p.is_cross_sentence for p in probes) and len(probes) >= 2:
                # Mark questions starting with Why / How / What caused as cross-sentence
                for p in probes:
                    q_lower = p.question.lower()
                    if any(q_lower.startswith(w) for w in ("why", "how", "what caused", "under what", "when")):
                        p.is_cross_sentence = True
                # If still none, mark the first 3
                if not any(p.is_cross_sentence for p in probes):
                    for p in probes[:3]:
                        p.is_cross_sentence = True
            return probes
    except Exception as exc:
        logger.warning(f"QA Probe generation failed: {exc}")

    # Fallback minimal probe based on passage content without quoting raw sentences or chronological order
    sentences = [s.strip() for s in re.split(r"[.!?]", w_raw) if len(s.strip()) > 15]
    if len(sentences) >= 2:
        return [
            QAProbeItem(
                question_id=1,
                question="What initial premise or condition is established in the passage?",
                gold_answer=sentences[0],
                is_cross_sentence=False,
                source_sentence_references=[sentences[0]],
            ),
            QAProbeItem(
                question_id=2,
                question="What outcome or consequence followed the initial condition?",
                gold_answer=sentences[1],
                is_cross_sentence=True,
                source_sentence_references=[sentences[0], sentences[1]],
            ),
        ]

    return [
        QAProbeItem(
            question_id=1,
            question="What core fact is stated in the passage?",
            gold_answer=w_raw[:150],
            is_cross_sentence=False,
            source_sentence_references=[w_raw[:150]],
        )
    ]


# ==============================================================================
# 2. Constrained Closed-Book Answering
# ==============================================================================


def build_answering_prompt(claims_context: str, question: str) -> str:
    """Construct prompt sent to the answering model constrained strictly to extracted claims."""
    return (
        "Answer the question below using ONLY the provided verified claims and supporting context.\n"
        "Do not assume or invent facts outside these claims.\n"
        'If the claims do not contain sufficient evidence to answer the question, output EXACTLY: "INSUFFICIENT_INFORMATION".\n\n'
        f"Extracted Knowledge Base:\n\"\"\"\n{claims_context}\n\"\"\"\n\n"
        f"Question: {question}\n\n"
        "Concise Answer:"
    )


def answer_probe(
    claims_context: str,
    question: str,
    pool: OllamaPool,
    model_name: str,
) -> str:
    """
    Answer the question using ONLY the provided list of extracted claims.
    If information is missing, respond with exactly 'INSUFFICIENT_INFORMATION'.
    """
    if not claims_context.strip() or "(No knowledge claims extracted" in claims_context:
        return "INSUFFICIENT_INFORMATION"

    prompt = build_answering_prompt(claims_context, question)

    try:
        raw_text, _ = pool.generate(model_name, prompt, temperature=0.0)
        clean = raw_text.strip()
        if "<think>" in clean and "</think>" in clean:
            clean = clean.split("</think>")[-1].strip()
        return clean
    except Exception as exc:
        logger.warning(f"Answering probe '{question[:40]}' failed: {exc}")
        return "INSUFFICIENT_INFORMATION"


# ==============================================================================
# 3. Objective Arbitration / Oracle Scoring LLM
# ==============================================================================


def build_judge_prompt(gold_answer: str, candidate_answer: str, w_raw: str) -> str:
    """
    Construct the judge arbitration prompt evaluating core semantic entailment
    instead of verbatim sub-fact matching.
    """
    return (
        "You are an objective judge evaluating whether a candidate answer accurately conveys ground truth facts.\n\n"
        f'Ground Truth Answer: "{gold_answer}"\n'
        f'Candidate Answer: "{candidate_answer}"\n\n'
        f"Passage Context (for reference):\n\"\"\"\n{w_raw[:4000]}\n\"\"\"\n\n"
        "Judge Verdict Guidelines:\n"
        "- If Candidate Answer is \"INSUFFICIENT_INFORMATION\" or indicates lack of evidence, mark as \"FAIL\".\n"
        "- If Candidate Answer directly contradicts the ground truth or hallucinates untrue facts, mark as \"FAIL\".\n"
        "- If Candidate Answer conveys the primary causal mechanism or core entity fact, mark as \"PASS\" "
        "(even if secondary details from other sentences are missing).\n"
        "- Do not require verbatim token matching. Evaluate whether the semantic proposition is asserted.\n\n"
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
    """
    Compare Candidate Answer against Ground Truth Answer.
    Uses core semantic entailment guidelines and stops using raw token overlap fallback.
    Returns (verdict, reason).
    """
    pred_clean = predicted_answer.strip()

    # Fast short-circuits
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
# 4. End-to-End Block QA Evaluation Orchestration
# ==============================================================================


def evaluate_block_qa_probes(
    w_raw: str,
    ideas_old: List[Any],
    ideas_new: List[Any],
    pool: OllamaPool,
    model_name: str,
    probes: Optional[List[QAProbeItem]] = None,
) -> Tuple[List[QAProbeItem], QASystemBlockMetrics, QASystemBlockMetrics]:
    """
    Generate probes for W_raw, query both systems (Old and New claims), and arbitrate verdicts.
    Returns:
      (probes, old_system_metrics, new_system_metrics)
    """
    if probes is None:
        probes = generate_qa_probes(w_raw, pool, model_name)

    claims_old = format_ideas_as_claims(ideas_old)
    claims_new = format_ideas_as_claims(ideas_new)

    evals_old: List[QAProbeEvaluation] = []
    evals_new: List[QAProbeEvaluation] = []

    for p in probes:
        # 1. Old / Baseline evaluation
        prompt_ans_old = build_answering_prompt(claims_old, p.question)
        pred_old = answer_probe(claims_old, p.question, pool, model_name)
        prompt_judge_old = build_judge_prompt(p.gold_answer, pred_old, w_raw)
        verdict_old, reason_old = judge_probe_answer(p.gold_answer, pred_old, w_raw, pool, model_name)
        evals_old.append(
            QAProbeEvaluation(
                question_id=p.question_id,
                question=p.question,
                gold_answer=p.gold_answer,
                is_cross_sentence=p.is_cross_sentence,
                predicted_answer=pred_old,
                verdict=verdict_old,
                reason=reason_old,
                answering_prompt=prompt_ans_old,
                judge_prompt=prompt_judge_old,
            )
        )

        # 2. New / Candidate evaluation
        prompt_ans_new = build_answering_prompt(claims_new, p.question)
        pred_new = answer_probe(claims_new, p.question, pool, model_name)
        prompt_judge_new = build_judge_prompt(p.gold_answer, pred_new, w_raw)
        verdict_new, reason_new = judge_probe_answer(p.gold_answer, pred_new, w_raw, pool, model_name)
        evals_new.append(
            QAProbeEvaluation(
                question_id=p.question_id,
                question=p.question,
                gold_answer=p.gold_answer,
                is_cross_sentence=p.is_cross_sentence,
                predicted_answer=pred_new,
                verdict=verdict_new,
                reason=reason_new,
                answering_prompt=prompt_ans_new,
                judge_prompt=prompt_judge_new,
            )
        )

    tot = len(probes)
    cross_tot = sum(1 for p in probes if p.is_cross_sentence)

    # Metrics Baseline (Old)
    passed_old = sum(1 for e in evals_old if e.verdict == "PASS")
    cross_passed_old = sum(1 for e in evals_old if e.is_cross_sentence and e.verdict == "PASS")
    metrics_old = QASystemBlockMetrics(
        system="old",
        total_probes=tot,
        passed_probes=passed_old,
        failed_probes=tot - passed_old,
        qa_recall=round(passed_old / tot, 4) if tot > 0 else 0.0,
        cross_sentence_total=cross_tot,
        cross_sentence_passed=cross_passed_old,
        seam_integrity_rate=round(cross_passed_old / cross_tot, 4) if cross_tot > 0 else 1.0,
        evaluations=evals_old,
    )

    # Metrics Candidate (New)
    passed_new = sum(1 for e in evals_new if e.verdict == "PASS")
    cross_passed_new = sum(1 for e in evals_new if e.is_cross_sentence and e.verdict == "PASS")
    metrics_new = QASystemBlockMetrics(
        system="new",
        total_probes=tot,
        passed_probes=passed_new,
        failed_probes=tot - passed_new,
        qa_recall=round(passed_new / tot, 4) if tot > 0 else 0.0,
        cross_sentence_total=cross_tot,
        cross_sentence_passed=cross_passed_new,
        seam_integrity_rate=round(cross_passed_new / cross_tot, 4) if cross_tot > 0 else 1.0,
        evaluations=evals_new,
    )

    return probes, metrics_old, metrics_new
