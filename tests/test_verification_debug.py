"""
Unit tests for Verification Debug JSON output and Standalone Statistics files.
Verifies that debug JSON captures raw requests, every sentence, every question,
full ideas context, raw responses, and final statistics.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.extractor import Concept
from bookeeper.processing.ollama_pool import OllamaPool, OllamaServerNode
from bookeeper.processing.qa_probe import (
    AtomicAssertionItem,
    NLIAssertionEvaluation,
    NLISystemBlockMetrics,
)
from bookeeper.processing.verifier import (
    ChunkAuditItem,
    ChunkVerificationReport,
    ChunkVerificationStats,
    VerificationItem,
    VerificationReport,
    VerificationStats,
    export_verification_debug,
    export_verification_stats,
    verify_chunks,
)


@pytest.fixture
def sample_chunk_report():
    a1 = AtomicAssertionItem(
        claim_id=1,
        question="What enables deliberate practice?",
        assertion="Deliberate practice requires focused attention.",
        source_quote="Focused attention is required.",
        is_cross_boundary=False,
        raw_prompt="Formulate question prompt for chunk",
        raw_response='[{"question_id": 1, "question": "What enables deliberate practice?", "assertion": "Deliberate practice requires focused attention.", "source_quote": "Focused attention is required."}]',
    )
    e1 = NLIAssertionEvaluation(
        claim_id=1,
        question="What enables deliberate practice?",
        assertion=a1.assertion,
        classification="SUPPORTED",
        score=1.0,
        verdict="PASS",
        rationale="Clearly confirmed in chunk claims.",
        verifier_prompt="Verifier prompt with ideas claims",
        raw_prompt="Verifier prompt with ideas claims",
        raw_response='{"classification": "SUPPORTED", "rationale": "Clearly confirmed in chunk claims."}',
    )
    item = ChunkAuditItem(
        chunk_id="chunk_1",
        book_id=1,
        book_title="Mastery of Learning",
        section_title="Chapter 1",
        chunk_text_snippet="Deliberate practice requires focused attention.",
        chunk_text="Deliberate practice requires focused attention. Immediate feedback loops sustain improvement.",
        sentences=[
            "Deliberate practice requires focused attention.",
            "Immediate feedback loops sustain improvement.",
        ],
        formatted_claims_context="[1] Concept: Deliberate Practice\n    Short Description: Focus and feedback\n    Explanation: Full deliberate practice description.",
        probe_generation_prompt="Oracle prompt for chunk 1",
        probe_generation_response='[{"question_id": 1, "question": "What enables deliberate practice?"}]',
        ideas_details=[
            {
                "name": "Deliberate Practice",
                "category": "Learning",
                "brief_description": "Focus and feedback",
                "detailed_explanation": "Full deliberate practice description.",
                "quote": "focused attention",
            }
        ],
        original_ideas=["Deliberate Practice"],
        oracle_ideas=["What enables deliberate practice?"],
        probes=[a1],
        assertions=[a1],
        evaluations=[e1],
        total_probes=1,
        passed_probes=1,
        failed_probes=0,
        supported_count=1,
        not_mentioned_count=0,
        against_count=0,
        qa_recall=1.0,
        support_ratio=1.0,
        hallucination_ratio=0.0,
        net_integrity_score=1.0,
        calibrated_fidelity=1.0,
        seam_integrity_rate=1.0,
    )
    stats = ChunkVerificationStats(
        mode="chunk",
        model_name="oracle-llama",
        total_chunks_in_graph=10,
        sampled_chunks_count=1,
        sample_percentage=10.0,
        total_assertions_count=1,
        total_supported_count=1,
        total_not_mentioned_count=0,
        total_against_count=0,
        mean_qa_recall=1.0,
        mean_net_integrity_score=1.0,
        mean_calibrated_fidelity=1.0,
        chunks_with_perfect_recall=1,
        total_duration_seconds=2.5,
    )
    return ChunkVerificationReport(
        mode="chunk",
        stats=stats,
        audited_chunks=[item],
        discrepancy_examples=[],
        omission_examples=[],
    )


@pytest.fixture
def sample_ideas_report():
    v1 = VerificationItem(
        idea_name="Deliberate Practice",
        idea_category="Learning",
        idea_summary="Focus and feedback",
        idea_weight=5,
        chunk_id="chunk_1",
        book_id=1,
        book_title="Mastery of Learning",
        section_title="Chapter 1",
        quote="focused attention",
        chunk_snippet="Deliberate practice requires focused attention.",
        chunk_text="Deliberate practice requires focused attention. Immediate feedback loops sustain improvement.",
        sentences=[
            "Deliberate practice requires focused attention.",
            "Immediate feedback loops sustain improvement.",
        ],
        raw_prompt="Verify idea prompt",
        raw_response='{"is_supported": true, "confidence": 1.0, "explanation": "Directly supported."}',
        is_supported=True,
        confidence=1.0,
        explanation="Directly supported by the text.",
    )
    stats = VerificationStats(
        total_ideas_in_graph=50,
        candidate_ideas_with_chunks=45,
        sampled_ideas=1,
        sample_percentage=2.0,
        mode="ideas",
        model_name="verifier-llama",
        total_evaluations=1,
        verified_count=1,
        discrepancy_count=0,
        verified_percentage=100.0,
        discrepancy_percentage=0.0,
        total_duration_seconds=1.8,
    )
    return VerificationReport(
        stats=stats,
        discrepancies=[],
        verified_samples=[v1],
    )


def test_export_verification_stats_and_debug_chunk_mode(tmp_path, sample_chunk_report):
    """Test exporting standalone stats and comprehensive debug JSON for chunk verification."""
    stats_file = tmp_path / "verification_stats.json"
    debug_file = tmp_path / "verification_debug.json"

    stats_data = export_verification_stats(sample_chunk_report, stats_file)
    assert stats_file.is_file()
    assert stats_data["mode"] == "chunk"
    assert stats_data["model_name"] == "oracle-llama"
    assert "timestamp" in stats_data
    assert stats_data["stats"]["mean_qa_recall"] == 1.0
    assert stats_data["stats"]["total_supported_count"] == 1

    debug_data = export_verification_debug(sample_chunk_report, debug_file)
    assert debug_file.is_file()
    assert debug_data["mode"] == "chunk"
    assert len(debug_data["debug_items"]) == 1

    item = debug_data["debug_items"][0]
    assert item["chunk_id"] == "chunk_1"
    # Full chunk text context
    assert "Immediate feedback loops sustain improvement." in item["chunk_text"]
    # Every sentence
    assert len(item["sentences"]) == 2
    assert "Deliberate practice requires focused attention." in item["sentences"]
    # Full ideas context provided within request
    assert "Formatted_claims_context" in str(item["ideas_context"]) or "formatted_prompt_context" in item["ideas_context"]
    assert "[1] Concept: Deliberate Practice" in item["ideas_context"]["formatted_prompt_context"]
    # Raw request and response for question generation
    assert item["question_generation"]["raw_prompt"] == "Oracle prompt for chunk 1"
    assert "What enables deliberate practice?" in item["question_generation"]["raw_response"]
    # Questions & Evaluations
    assert len(item["probes_evaluation"]) == 1
    probe = item["probes_evaluation"][0]
    assert probe["question"] == "What enables deliberate practice?"
    assert probe["assertion"] == "Deliberate practice requires focused attention."
    assert probe["raw_verifier_prompt"] == "Verifier prompt with ideas claims"
    assert probe["raw_verifier_response"] == '{"classification": "SUPPORTED", "rationale": "Clearly confirmed in chunk claims."}'
    assert probe["result"]["classification"] == "SUPPORTED"
    assert probe["result"]["score"] == 1.0
    assert probe["result"]["verdict"] == "PASS"


def test_export_verification_stats_and_debug_ideas_mode(tmp_path, sample_ideas_report):
    """Test exporting standalone stats and debug JSON for ideas verification mode."""
    stats_file = tmp_path / "verification_stats.json"
    debug_file = tmp_path / "verification_debug.json"

    stats_data = export_verification_stats(sample_ideas_report, stats_file)
    assert stats_file.is_file()
    assert stats_data["mode"] == "ideas"
    assert stats_data["stats"]["verified_count"] == 1
    assert stats_data["stats"]["discrepancy_count"] == 0

    debug_data = export_verification_debug(sample_ideas_report, debug_file)
    assert debug_file.is_file()
    assert debug_data["mode"] == "ideas"
    assert len(debug_data["debug_items"]) == 1

    item = debug_data["debug_items"][0]
    assert item["idea_name"] == "Deliberate Practice"
    assert item["chunk_id"] == "chunk_1"
    assert len(item["sentences"]) == 2
    assert item["raw_verifier_prompt"] == "Verify idea prompt"
    assert item["result"]["is_supported"] is True


def test_cli_verify_saves_stats_and_debug_files(tmp_path):
    """Test that CLI verify command saves verification_report.json, verification_stats.json, and verification_debug.json."""
    store = ConceptGraphStore()
    store.add_book(1, title="Mastery", author="Author")
    c1 = HierarchicalChunk(
        chunk_id="chk_1",
        chunk_idx=1,
        book_id=1,
        book_title="Mastery",
        chapter_idx=1,
        chapter_title="Ch 1",
        section_title="Sec 1",
        subtitle="",
        breadcrumb="Ch 1",
        text="Deliberate practice requires focused attention.",
        char_count=45,
        word_count=5,
    )
    store.add_chunk(c1)
    con = Concept(name="Deliberate Practice", category="Learning", summary="Practice model")
    store.add_concept(con)
    store.add_idea_support_link("Deliberate Practice", c1, "focused attention", "Brief", "Detailed")

    kg_file = tmp_path / "knowledge_graph.json"
    store.save(str(kg_file))

    out_report = tmp_path / "verification_report.json"
    out_stats = tmp_path / "verification_stats.json"
    out_debug = tmp_path / "verification_debug.json"

    def mock_eval_claims(w_raw, ideas, pool, model_name, *args, **kwargs):
        a = AtomicAssertionItem(
            claim_id=1,
            question="What is deliberate practice?",
            assertion="Deliberate practice requires focused attention.",
            source_quote="focused attention",
            raw_prompt="Formulate question",
            raw_response="Oracle response",
        )
        e = NLIAssertionEvaluation(
            claim_id=1,
            question="What is deliberate practice?",
            assertion=a.assertion,
            classification="SUPPORTED",
            score=1.0,
            verdict="PASS",
            rationale="Supported in claims",
            raw_prompt="NLI Prompt",
            raw_response='{"classification": "SUPPORTED"}',
        )
        m = NLISystemBlockMetrics(
            system="old",
            total_probes=1,
            passed_probes=1,
            failed_probes=0,
            qa_recall=1.0,
            evaluations=[e],
            supported_count=1,
            net_integrity_score=1.0,
            calibrated_fidelity=1.0,
        )
        return [a], m

    runner = CliRunner()
    with patch("bookeeper.cli.IdeaVerifier.warmup", return_value=True), \
         patch("bookeeper.processing.verifier.evaluate_claims_atomic_assertions", side_effect=mock_eval_claims):
        res = runner.invoke(
            app,
            [
                "verify",
                "--graph-file",
                str(kg_file),
                "--mode",
                "chunk",
                "--percent",
                "100.0",
                "--debug",
                "--output",
                str(out_report),
                "--stats-output",
                str(out_stats),
                "--debug-output",
                str(out_debug),
            ],
        )

    assert res.exit_code == 0
    assert out_report.is_file()
    assert out_stats.is_file()
    assert out_debug.is_file()

    stats_content = json.loads(out_stats.read_text())
    assert stats_content["mode"] == "chunk"
    assert "stats" in stats_content

    debug_content = json.loads(out_debug.read_text())
    assert debug_content["mode"] == "chunk"
    assert len(debug_content["debug_items"]) == 1
    assert "sentences" in debug_content["debug_items"][0]
    assert "ideas_context" in debug_content["debug_items"][0]
    assert "question_generation" in debug_content["debug_items"][0]


def test_cli_verify_debug_disabled_by_default(tmp_path):
    """Test that bookeeper verify does not generate debug JSON when --debug is not specified."""
    kg_file = tmp_path / "knowledge_graph.json"
    dummy_kg = {
        "nodes": [
            {"id": "book:1", "type": "Book", "title": "Test Book", "author": "Author"},
            {"id": "c1", "type": "Chunk", "book_id": 1, "text": "Deliberate practice is key. Regular feedback helps."},
            {
                "id": "idea:1",
                "type": "Concept",
                "name": "Deliberate Practice",
                "chunk_ids": ["c1"],
                "brief_description": "Deliberate practice",
                "detailed_explanation": "Detailed explanation of deliberate practice.",
            },
        ],
        "edges": [{"source": "idea:1", "target": "c1", "type": "EXTRACTED_FROM"}],
    }
    kg_file.write_text(json.dumps(dummy_kg), encoding="utf-8")

    out_report = tmp_path / "custom_report.json"
    out_stats = tmp_path / "custom_stats.json"
    out_debug = tmp_path / "custom_debug.json"

    def mock_eval_claims(probe_oracle, verifier_pool, claims, probes, chunk_text, **kwargs):
        a = AtomicAssertionItem(
            claim_id=1,
            question="What is key?",
            assertion="Deliberate practice is key.",
            source_quote="Deliberate practice is key.",
        )
        e = NLIAssertionEvaluation(
            claim_id=1,
            question="What is key?",
            assertion=a.assertion,
            classification="SUPPORTED",
            score=1.0,
            verdict="PASS",
            rationale="Supported",
        )
        m = NLISystemBlockMetrics(
            system="old",
            total_probes=1,
            passed_probes=1,
            failed_probes=0,
            qa_recall=1.0,
            evaluations=[e],
            supported_count=1,
            net_integrity_score=1.0,
            calibrated_fidelity=1.0,
        )
        return [a], m

    runner = CliRunner()
    with patch("bookeeper.cli.IdeaVerifier.warmup", return_value=True), \
         patch("bookeeper.processing.verifier.evaluate_claims_atomic_assertions", side_effect=mock_eval_claims):
        res = runner.invoke(
            app,
            [
                "verify",
                "--graph-file",
                str(kg_file),
                "--mode",
                "chunk",
                "--percent",
                "100.0",
                "--output",
                str(out_report),
                "--stats-output",
                str(out_stats),
                "--debug-output",
                str(out_debug),
            ],
        )

    assert res.exit_code == 0
    assert out_report.is_file()
    assert out_stats.is_file()
    # Debug file should NOT be created because debug is disabled by default
    assert not out_debug.is_file()


def test_ab_test_compare_consumes_verification_stats(tmp_path):
    """Test that scripts/ab_test.py compare_results loads verification_stats.json seamlessly."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from scripts.ab_test import ABTestRunner

    dir_a = tmp_path / "baseline"
    dir_b = tmp_path / "branch_b"
    dir_a.mkdir()
    dir_b.mkdir()

    stats_a = {
        "mode": "chunk",
        "model_name": "model_a",
        "stats": {
            "total_chunks_in_graph": 100,
            "sampled_chunks_count": 10,
            "mean_qa_recall": 0.50,
            "overall_success_rate": 50.0,
            "overall_fail_rate": 50.0,
            "total_supported_count": 5,
            "total_not_mentioned_count": 5,
            "total_against_count": 0,
        },
    }
    stats_b = {
        "mode": "chunk",
        "model_name": "model_b",
        "stats": {
            "total_chunks_in_graph": 100,
            "sampled_chunks_count": 10,
            "mean_qa_recall": 0.80,
            "overall_success_rate": 80.0,
            "overall_fail_rate": 20.0,
            "total_supported_count": 8,
            "total_not_mentioned_count": 2,
            "total_against_count": 0,
        },
    }

    # Save as verification_stats.json
    (dir_a / "verification_stats.json").write_text(json.dumps(stats_a))
    (dir_b / "verification_stats.json").write_text(json.dumps(stats_b))

    repo_root = Path(__file__).parents[1]
    runner = ABTestRunner(branch1="master", branch2="feat-model", repo_dir=str(repo_root), mode="chunk")
    comp = runner.compare_results(dir_a, dir_b)

    assert comp["mode"] == "chunk"
    metric_names = [m["metric"] for m in comp["metrics"]]
    assert "Success Rate (Recall %)" in metric_names
    # Check that the metric values match what was in stats_a and stats_b
    success_metric = next(m for m in comp["metrics"] if m["metric"] == "Success Rate (Recall %)")
    assert success_metric["branch_a"] == 50.0  # baseline
    assert success_metric["branch_b"] == 80.0  # candidate
