"""
Unit tests for NLI Atomic Assertion Oracle module (qa_probe.py).
"""

import json
from unittest.mock import MagicMock
import pytest

from bookeeper.processing.extractor import Concept
from bookeeper.processing.qa_probe import (
    AtomicAssertionItem,
    NLIAssertionEvaluation,
    NLISystemBlockMetrics,
    QAProbeItem,
    QAProbeEvaluation,
    QASystemBlockMetrics,
    format_ideas_as_claims,
    format_ideas_for_context,
    clean_llm_json_response,
    parse_json_array_safely,
    generate_atomic_assertions,
    generate_qa_probes,
    verify_assertion_entailment,
    answer_probe,
    judge_probe_answer,
    evaluate_block_atomic_assertions,
    evaluate_block_qa_probes,
    build_assertion_generator_prompt,
    build_generator_prompt,
    build_nli_verifier_prompt,
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
        "<think>Generating 2 assertions...</think>\n"
        "```json\n"
        '[\n  {"assertion": "A1", "is_cross_boundary": true, "source_quote": "Q1"},\n'
        '  {"assertion": "A2", "is_cross_boundary": false, "source_quote": "Q2"}\n]\n'
        "```"
    )
    clean = clean_llm_json_response(raw)
    assert not clean.startswith("```")
    assert "<think>" not in clean

    items = parse_json_array_safely(raw)
    assert len(items) == 2
    assert items[0]["assertion"] == "A1"
    assert items[0]["is_cross_boundary"] is True

    # Dict containing array key
    wrapped = '{"assertions": [{"assertion": "A_wrapped", "source_quote": "Q_wrapped"}]}'
    items_wrapped = parse_json_array_safely(wrapped)
    assert len(items_wrapped) == 1
    assert items_wrapped[0]["assertion"] == "A_wrapped"


def test_atomic_assertion_data_model_compat():
    """Verify bidirectional compatibility between AtomicAssertionItem and QAProbeItem."""
    # Instantiated with new assertion fields
    item1 = AtomicAssertionItem(
        claim_id=1,
        assertion="Sircolo is a male marsh harrier who agreed to carry the group.",
        is_cross_boundary=True,
        source_quote="Sircolo agreed to carry them.",
    )
    assert item1.claim_id == 1
    assert item1.question_id == 1
    assert item1.assertion == "Sircolo is a male marsh harrier who agreed to carry the group."
    assert item1.question == "Sircolo is a male marsh harrier who agreed to carry the group."
    assert item1.is_cross_boundary is True
    assert item1.is_cross_sentence is True
    assert item1.source_quote == "Sircolo agreed to carry them."
    assert item1.gold_answer == "Sircolo agreed to carry them."

    # Instantiated with legacy question fields
    item2 = QAProbeItem(
        question_id=2,
        question="What format is used?",
        gold_answer="EPUB format.",
        is_cross_sentence=False,
    )
    assert item2.claim_id == 2
    assert item2.question_id == 2
    assert item2.assertion == "What format is used?"
    assert item2.is_cross_boundary is False
    assert item2.source_quote == "EPUB format."


def test_generate_atomic_assertions():
    mock_pool = MagicMock()
    mock_resp = json.dumps([
        {
            "claim_id": 1,
            "assertion": "Sircolo is a male marsh harrier who agreed to carry the group across the marsh.",
            "is_cross_boundary": True,
            "source_quote": "Sircolo agreed to carry them to firm ground.",
        },
        {
            "claim_id": 2,
            "assertion": "Rekaby called Sircolo using a whistle.",
            "is_cross_boundary": False,
            "source_quote": "Rekaby blew his whistle.",
        },
    ])
    mock_pool.generate.return_value = (mock_resp, {})

    assertions = generate_atomic_assertions(
        canonical_window="Rekaby blew his whistle. Sircolo agreed to carry them to firm ground.",
        pool=mock_pool,
        model_name="mock-model",
        num_assertions=2,
    )

    assert len(assertions) == 2
    assert assertions[0].claim_id == 1
    assert "Sircolo is a male marsh harrier" in assertions[0].assertion
    assert assertions[0].is_cross_boundary is True
    assert assertions[0].source_quote == "Sircolo agreed to carry them to firm ground."
    assert assertions[1].claim_id == 2
    assert assertions[1].is_cross_boundary is False


def test_verify_assertion_entailment():
    mock_pool = MagicMock()

    # Empty context fast return
    cls_empty, rat_empty = verify_assertion_entailment("", "Any claim?", mock_pool, "mock-model")
    assert cls_empty == "NOT_MENTIONED"
    assert "no claims" in rat_empty.lower()

    # Empty assertion fast return
    cls_blank, rat_blank = verify_assertion_entailment("Some context", "", mock_pool, "mock-model")
    assert cls_blank == "NOT_MENTIONED"

    # SUPPORTED classification
    mock_pool.generate.return_value = (
        json.dumps({"classification": "SUPPORTED", "rationale": "Directly stated in Concept 1."}),
        {},
    )
    cls1, rat1 = verify_assertion_entailment("Claims context", "Target claim", mock_pool, "mock-model")
    assert cls1 == "SUPPORTED"
    assert "Directly stated" in rat1

    # CONTRADICTED classification
    mock_pool.generate.return_value = (
        json.dumps({"classification": "CONTRADICTED", "rationale": "The claims state the exact opposite."}),
        {},
    )
    cls2, rat2 = verify_assertion_entailment("Claims context", "Target claim", mock_pool, "mock-model")
    assert cls2 == "CONTRADICTED"

    # NOT_MENTIONED classification
    mock_pool.generate.return_value = (
        json.dumps({"classification": "NOT_MENTIONED", "rationale": "No facts mention this entity."}),
        {},
    )
    cls3, rat3 = verify_assertion_entailment("Claims context", "Target claim", mock_pool, "mock-model")
    assert cls3 == "NOT_MENTIONED"


def test_evaluate_block_atomic_assertions():
    mock_pool = MagicMock()

    assertions = [
        AtomicAssertionItem(
            claim_id=1,
            assertion="Rekaby used a whistle to summon Sircolo.",
            is_cross_boundary=True,
            source_quote="Rekaby whistled.",
        ),
        AtomicAssertionItem(
            claim_id=2,
            assertion="Sircolo is a harrier.",
            is_cross_boundary=False,
            source_quote="Sircolo the harrier.",
        ),
    ]

    # Baseline (Old) claims entail Claim 1 (cross-boundary), but NOT Claim 2
    # Candidate (New) claims entail Claim 2, but NOT Claim 1
    def fake_generate(model, prompt, **kwargs):
        if 'Target Assertion: "Rekaby used a whistle' in prompt:
            if "Old Claim" in prompt:
                return (json.dumps({"classification": "SUPPORTED", "rationale": "Found in Old Claim"}), {})
            else:
                return (json.dumps({"classification": "NOT_MENTIONED", "rationale": "Missing in New Claim"}), {})
        elif 'Target Assertion: "Sircolo is a harrier.' in prompt:
            if "New Claim" in prompt:
                return (json.dumps({"classification": "SUPPORTED", "rationale": "Found in New Claim"}), {})
            else:
                return (json.dumps({"classification": "NOT_MENTIONED", "rationale": "Missing in Old Claim"}), {})
        return (json.dumps({"classification": "NOT_MENTIONED", "rationale": "Unknown"}), {})

    mock_pool.generate.side_effect = fake_generate

    ideas_old = [Concept(name="Old Claim", summary="Rekaby used a whistle to summon Sircolo.")]
    ideas_new = [Concept(name="New Claim", summary="Sircolo is a harrier.")]

    res_assertions, m_old, m_new = evaluate_block_atomic_assertions(
        w_raw="Rekaby whistled for Sircolo the harrier.",
        ideas_old=ideas_old,
        ideas_new=ideas_new,
        pool=mock_pool,
        model_name="mock-model",
        assertions=assertions,
    )

    assert len(res_assertions) == 2
    # Old: passed Claim 1 (cross-boundary), failed Claim 2
    assert m_old.passed_probes == 1
    assert m_old.failed_probes == 1
    assert m_old.qa_recall == 0.5
    assert m_old.seam_integrity_rate == 1.0  # 1/1 cross-boundary supported

    # New: failed Claim 1 (cross-boundary), passed Claim 2
    assert m_new.passed_probes == 1
    assert m_new.failed_probes == 1
    assert m_new.qa_recall == 0.5
    assert m_new.seam_integrity_rate == 0.0  # 0/1 cross-boundary supported

    # Check that verifier_prompt is preserved on evaluations
    assert len(m_old.evaluations[0].verifier_prompt) > 20
    assert "Target Assertion:" in m_old.evaluations[0].verifier_prompt
    assert m_old.evaluations[0].classification == "SUPPORTED"
    assert m_old.evaluations[0].verdict == "PASS"
    assert m_old.evaluations[1].classification == "NOT_MENTIONED"
    assert m_old.evaluations[1].verdict == "FAIL"


def test_prompt_builders_and_constraints():
    """Verify prompt templates enforce strict constraints from NLI architecture."""
    # 1. Assertion Generator prompt constraints
    gen_p = build_assertion_generator_prompt("Some passage about King Arthur.", 5)
    assert "You are an expert NLP benchmark engineer." in gen_p
    assert "extract exactly 5 clear, factual, declarative statements" in gen_p
    assert "Every claim must be an independent, self-contained declarative sentence" in gen_p
    assert "At least 3 claims MUST be CROSS-SENTENCE" in gen_p
    assert "DO NOT output questions. Output ONLY declarative assertions." in gen_p
    assert 'DO NOT make statements about sentence order, grammar, or punctuation (no "Sentence A precedes Sentence B").' in gen_p
    assert "claim_id" in gen_p
    assert "is_cross_boundary" in gen_p
    assert "source_quote" in gen_p

    # Backward compatible alias
    gen_p_alias = build_generator_prompt("Passage", 5)
    assert gen_p_alias == gen_p.replace("Some passage about King Arthur.", "Passage")

    # 2. NLI Verifier prompt constraints
    ver_p = build_nli_verifier_prompt("Formatted claims context", "Sircolo is a harrier.")
    assert "You are an objective Natural Language Inference (NLI) evaluator." in ver_p
    assert "Target Assertion: \"Sircolo is a harrier.\"" in ver_p
    assert "SUPPORTED" in ver_p
    assert "CONTRADICTED" in ver_p
    assert "NOT_MENTIONED" in ver_p
    assert "classification" in ver_p
    assert "rationale" in ver_p
