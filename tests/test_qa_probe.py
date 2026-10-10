"""
Unit tests for QA Probe Oracle module (qa_probe.py).
"""

import json
from unittest.mock import MagicMock
import pytest

from bookeeper.processing.extractor import Concept
from bookeeper.processing.qa_probe import (
    QAProbeItem,
    QAProbeEvaluation,
    QASystemBlockMetrics,
    format_ideas_as_claims,
    format_ideas_for_context,
    clean_llm_json_response,
    parse_json_array_safely,
    generate_qa_probes,
    answer_probe,
    judge_probe_answer,
    evaluate_block_qa_probes,
    build_generator_prompt,
    build_answering_prompt,
    build_judge_prompt,
)


def test_format_ideas_for_context():
    # 1. Concept objects
    c1 = Concept(
        name="Format Conversion",
        category="Tech",
        summary="Converts EPUB to MOBI",
        detailed_explanation="Automatic format conversion occurs when sync is initiated.",
    )
    c2 = Concept(
        name="Metadata Tagging",
        category="Tech",
        brief_description="Tags book genres",
    )
    claims_text = format_ideas_as_claims([c1, c2])
    assert "[1] Concept: Format Conversion" in claims_text
    assert "Explanation: Automatic format conversion occurs when sync is initiated." in claims_text
    assert "[2] Concept: Metadata Tagging" in claims_text
    assert "Explanation: Tags book genres" in claims_text

    # 2. Concept with quote and category
    c3 = Concept(
        name="DRM Protection",
        category="Security",
        brief_description="Digital rights management restricts reading.",
        detailed_explanation="Encrypted keys prevent unauthorized copying.",
        supporting_quote="DRM was used to restrict user freedom.",
    )
    claims_text3 = format_ideas_for_context([c3])
    assert "[1] Concept: DRM Protection" in claims_text3
    assert "Explanation: Encrypted keys prevent unauthorized copying." in claims_text3
    assert 'Direct Context/Quote: "DRM was used to restrict user freedom."' in claims_text3

    # 3. String fallback
    assert "[1] Concept: Simple Idea" in format_ideas_for_context(["Simple Idea"])

    # 4. Empty list
    assert format_ideas_for_context([]) == "(No knowledge claims extracted for this block)"


def test_json_parsing_and_cleaning():
    # Markdown code blocks and thinking tags
    raw = (
        "<think>Generating 2 questions...</think>\n"
        "```json\n"
        '[\n  {"question": "Q1", "gold_answer": "A1", "is_cross_sentence": true},\n'
        '  {"question": "Q2", "gold_answer": "A2", "is_cross_sentence": false}\n]\n'
        "```"
    )
    clean = clean_llm_json_response(raw)
    assert not clean.startswith("```")
    assert "<think>" not in clean

    items = parse_json_array_safely(raw)
    assert len(items) == 2
    assert items[0]["question"] == "Q1"
    assert items[0]["is_cross_sentence"] is True

    # Dict containing array key
    wrapped = '{"questions": [{"question": "Q_wrapped", "gold_answer": "A_wrapped"}]}'
    items_wrapped = parse_json_array_safely(wrapped)
    assert len(items_wrapped) == 1
    assert items_wrapped[0]["question"] == "Q_wrapped"


def test_generate_qa_probes():
    mock_pool = MagicMock()
    mock_resp = json.dumps([
        {
            "question_id": 1,
            "question": "Why did X fail?",
            "gold_answer": "Because Y happened.",
            "is_cross_sentence": True,
            "source_sentence_references": ["X failed because Y happened.", "Y happened."],
        },
        {
            "question_id": 2,
            "question": "What is Z?",
            "gold_answer": "Z is a standard format.",
            "is_cross_sentence": False,
            "source_sentence_references": ["Z is a standard format maintained by the community."],
        },
    ])
    mock_pool.generate.return_value = (mock_resp, {})

    probes = generate_qa_probes(
        w_raw="X failed because Y happened. Z is a standard format maintained by the community.",
        pool=mock_pool,
        model_name="mock-model",
        num_questions=2,
    )

    assert len(probes) == 2
    assert probes[0].question == "Why did X fail?"
    assert probes[0].is_cross_sentence is True
    assert len(probes[0].source_sentence_references) == 2
    assert probes[0].source_sentence_references[0] == "X failed because Y happened."
    assert probes[1].gold_answer == "Z is a standard format."
    assert len(probes[1].source_sentence_references) == 1


def test_answer_probe():
    mock_pool = MagicMock()
    mock_pool.generate.return_value = ("The conversion is done automatically.", {})

    claims = "Claim 1: eBook Conversion\nSummary: Converts books automatically."
    ans = answer_probe(claims, "How is conversion done?", mock_pool, "mock-model")
    assert ans == "The conversion is done automatically."

    # Empty context fast return
    insufficient = answer_probe("", "Any question?", mock_pool, "mock-model")
    assert insufficient == "INSUFFICIENT_INFORMATION"


def test_judge_probe_answer():
    mock_pool = MagicMock()

    # Fast short-circuit for INSUFFICIENT_INFORMATION (0 LLM calls)
    verdict, reason = judge_probe_answer("EPUB format.", "INSUFFICIENT_INFORMATION", "Context", mock_pool, "mock-model")
    assert verdict == "FAIL"
    assert mock_pool.generate.call_count == 0

    # Short-circuit for empty string
    verdict_empty, _ = judge_probe_answer("EPUB format.", "", "Context", mock_pool, "mock-model")
    assert verdict_empty == "FAIL"

    # PASS verdict from LLM
    mock_pool.generate.return_value = (json.dumps({"verdict": "PASS", "reason": "Accurately conveys facts."}), {})
    verdict_pass, reason_pass = judge_probe_answer("EPUB format.", "It is the EPUB format.", "Context", mock_pool, "mock-model")
    assert verdict_pass == "PASS"
    assert "Accurately conveys facts." in reason_pass

    # FAIL verdict from LLM
    mock_pool.generate.return_value = (json.dumps({"verdict": "FAIL", "reason": "Hallucinates PDF format."}), {})
    verdict_fail, reason_fail = judge_probe_answer("EPUB format.", "PDF format.", "Context", mock_pool, "mock-model")
    assert verdict_fail == "FAIL"


def test_evaluate_block_qa_probes():
    mock_pool = MagicMock()

    probes = [
        QAProbeItem(
            question_id=1,
            question="Why did X happen?",
            gold_answer="Due to reason R.",
            is_cross_sentence=True,
        ),
        QAProbeItem(
            question_id=2,
            question="What is item Y?",
            gold_answer="Item Y is a widget.",
            is_cross_sentence=False,
        ),
    ]

    # Model returns:
    # 1. Answer Q1 for Old: "Due to reason R."
    # 2. Judge Q1 for Old: PASS
    # 3. Answer Q1 for New: "INSUFFICIENT_INFORMATION" (Judge short-circuits to FAIL)
    # 4. Answer Q2 for Old: "INSUFFICIENT_INFORMATION" (Judge short-circuits to FAIL)
    # 5. Answer Q2 for New: "Item Y is a widget."
    # 6. Judge Q2 for New: PASS

    def fake_generate(model, prompt, **kwargs):
        if "Concise Answer:" in prompt:
            if "Why did X happen?" in prompt and "Old Claim" in prompt:
                return ("Due to reason R.", {})
            if "What is item Y?" in prompt and "New Claim" in prompt:
                return ("Item Y is a widget.", {})
            return ("INSUFFICIENT_INFORMATION", {})
        if "Ground Truth Answer:" in prompt:
            return (json.dumps({"verdict": "PASS", "reason": "Correct match."}), {})
        return ("", {})

    mock_pool.generate.side_effect = fake_generate

    ideas_old = [Concept(name="Old Claim", summary="Due to reason R.")]
    ideas_new = [Concept(name="New Claim", summary="Item Y is a widget.")]

    probes_res, m_old, m_new = evaluate_block_qa_probes(
        w_raw="Passage text here.",
        ideas_old=ideas_old,
        ideas_new=ideas_new,
        pool=mock_pool,
        model_name="mock-model",
        probes=probes,
    )

    assert len(probes_res) == 2
    # Old passed Q1 (cross-sentence), failed Q2
    assert m_old.passed_probes == 1
    assert m_old.failed_probes == 1
    assert m_old.qa_recall == 0.5
    assert m_old.seam_integrity_rate == 1.0  # 1/1 cross-sentence passed

    # New failed Q1, passed Q2 (single-sentence)
    assert m_new.passed_probes == 1
    assert m_new.failed_probes == 1
    assert m_new.qa_recall == 0.5
    assert m_new.seam_integrity_rate == 0.0  # 0/1 cross-sentence passed

    # Verify answering_prompt and judge_prompt are captured on evaluations
    assert len(m_old.evaluations[0].answering_prompt) > 20
    assert "Extracted Knowledge Base:" in m_old.evaluations[0].answering_prompt
    assert len(m_old.evaluations[0].judge_prompt) > 20
    assert "Judge Verdict Guidelines:" in m_old.evaluations[0].judge_prompt


def test_prompt_builders_and_guidelines():
    """Verify prompt templates enforce critical probe requirements and relaxed judge guidelines."""
    # 1. Generator prompt requirements
    gen_p = build_generator_prompt("Some passage about King Arthur.", 5)
    assert "DO NOT quote raw sentences in the question" in gen_p
    assert "Frame all questions around entities, actions, or decisions in natural language" in gen_p
    assert 'no "precedes" or chronological sentence order questions' in gen_p
    assert "source_sentence_references" in gen_p

    # 2. Judge prompt guidelines
    judge_p = build_judge_prompt("Arthur became king.", "Arthur assumed the crown.", "Context")
    assert 'If Candidate Answer conveys the primary causal mechanism or core entity fact, mark as "PASS"' in judge_p
    assert "Do not require verbatim token matching. Evaluate whether the semantic proposition is asserted." in judge_p


def test_judge_no_token_overlap_fallback():
    """Verify that failed LLM judge calls fail without token overlap fallback."""
    mock_pool = MagicMock()
    mock_pool.generate.side_effect = RuntimeError("Connection timed out")

    # Even with identical text (100% token overlap), must NOT pass via fallback token overlap
    verdict, reason = judge_probe_answer("The quick brown fox", "The quick brown fox", "Passage", mock_pool, "model")
    assert verdict == "FAIL"
    assert "Judge evaluation error" in reason
