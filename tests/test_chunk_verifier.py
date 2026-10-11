"""
Unit tests for Chunk Verification Mode (verify_chunks, EntityDeduplicator.is_same_concept, CLI verify --mode chunk).
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import Concept
from bookeeper.processing.ollama_pool import OllamaPool, OllamaServerNode
from bookeeper.processing.verifier import (
    ChunkAuditItem,
    ChunkMatchedPair,
    ChunkVerificationReport,
    ChunkVerificationStats,
    verify_chunks,
)


@pytest.fixture
def mock_pool():
    return OllamaPool(
        servers=[OllamaServerNode(url="http://localhost:11434", priority=1)],
        cooldown_seconds=60,
    )


@pytest.fixture
def populated_chunk_store():
    store = ConceptGraphStore()
    store.add_book(1, title="Mastery of Learning", author="Test Author")

    # Chunk 1: has 2 concepts connected
    c1 = HierarchicalChunk(
        chunk_id="chunk_1",
        chunk_idx=1,
        book_id=1,
        book_title="Mastery of Learning",
        chapter_idx=1,
        chapter_title="Chapter 1",
        section_title="Introduction",
        subtitle="",
        breadcrumb="Book > Ch 1",
        text="Deliberate practice requires focused attention and constant feedback loops.",
        char_count=73,
        word_count=10,
    )
    store.add_chunk(c1)
    con1 = Concept(name="Deliberate Practice", category="Learning", summary="Focus and feedback", weight=5)
    con2 = Concept(name="Feedback Loops", category="Learning", summary="Constant iterative signals", weight=4)
    store.add_concept(con1)
    store.add_concept(con2)
    store.add_idea_support_link("Deliberate Practice", c1, "focused attention", "Brief 1", "Detailed 1")
    store.add_idea_support_link("Feedback Loops", c1, "constant feedback", "Brief 2", "Detailed 2")

    # Chunk 2: has 1 concept connected
    c2 = HierarchicalChunk(
        chunk_id="chunk_2",
        chunk_idx=2,
        book_id=1,
        book_title="Mastery of Learning",
        chapter_idx=1,
        chapter_title="Chapter 1",
        section_title="Memory",
        subtitle="",
        breadcrumb="Book > Ch 1",
        text="Spaced repetition leverages the spacing effect to dramatically improve long-term retention.",
        char_count=87,
        word_count=12,
    )
    store.add_chunk(c2)
    con3 = Concept(name="Spaced Repetition", category="Memory", summary="Spacing intervals", weight=5)
    store.add_concept(con3)
    store.add_idea_support_link("Spaced Repetition", c2, "spacing effect", "Brief 3", "Detailed 3")

    return store


def test_chunk_verification_models_and_metrics():
    """Verify data structures and mathematical metric definitions."""
    # Test case 1: Standard match (2 original, 3 oracle, 2 matched, 1 missed)
    item = ChunkAuditItem(
        chunk_id="chk_1",
        book_id=1,
        book_title="Test Book",
        section_title="Ch 1",
        chunk_text_snippet="Snippet text",
        original_ideas=["Idea A", "Idea B"],
        oracle_ideas=["Idea A", "Idea B", "Idea C"],
        matched_pairs=[
            ChunkMatchedPair(oracle_idea_name="Idea A", original_idea_name="Idea A", similarity_score=1.0),
            ChunkMatchedPair(oracle_idea_name="Idea B", original_idea_name="Idea B", similarity_score=1.0),
        ],
        missed_oracle_ideas=["Idea C"],
        extra_original_ideas=[],
        original_count=2,
        oracle_count=3,
        matched_count=2,
        missed_count=1,
        success_rate=round(2 / 3 * 100.0, 2),
        fail_rate=round(1 / 2 * 100.0, 2),
    )
    assert item.success_rate == 66.67
    assert item.fail_rate == 50.0

    # Test stats rollup
    stats = ChunkVerificationStats(
        mode="chunk",
        model_name="oracle-test",
        total_chunks_in_graph=10,
        sampled_chunks_count=1,
        sample_percentage=10.0,
        total_original_ideas=2,
        total_oracle_ideas=3,
        total_matched_ideas=2,
        total_missed_ideas=1,
        total_extra_original_ideas=0,
        overall_success_rate=66.67,
        overall_fail_rate=50.0,
        chunks_with_perfect_match=0,
        chunks_with_omissions=1,
        total_duration_seconds=2.5,
    )
    report = ChunkVerificationReport(
        mode="chunk",
        stats=stats,
        audited_chunks=[item],
        omission_examples=[item],
    )
    assert report.stats.overall_success_rate == 66.67
    assert report.stats.overall_fail_rate == 50.0
    assert len(report.omission_examples) == 1


def test_entity_deduplicator_is_same_concept_exact_and_normalized():
    """Test is_same_concept exact and normalized string matches."""
    dedup = EntityDeduplicator()

    # Exact match
    is_same, score, method, canon, reasoning = dedup.is_same_concept("Deep Learning", "Deep Learning")
    assert is_same is True
    assert score == 1.0
    assert method == "exact"

    # Normalized match (case-insensitive and punctuation)
    is_same, score, method, canon, reasoning = dedup.is_same_concept("Deep Learning!", "deep learning")
    assert is_same is True
    assert score == 1.0
    assert method == "exact"

    # Completely different concept without embeddings
    is_same, score, method, canon, reasoning = dedup.is_same_concept("Quantum Mechanics", "Cooking Pasta")
    assert is_same is False
    assert score == 0.0


def test_entity_deduplicator_is_same_concept_with_embeddings():
    """Test is_same_concept when vector embeddings indicate high similarity."""
    mock_embeddings = MagicMock()
    # Return identical vector embeddings
    mock_embeddings.embed_query.return_value = [0.5, 0.5, 0.5, 0.5]

    dedup = EntityDeduplicator(embeddings=mock_embeddings)
    is_same, score, method, canon, reasoning = dedup.is_same_concept(
        "Artificial Intelligence", "Synthetic Intelligence"
    )
    assert is_same is True
    assert score >= 0.88
    assert method == "high_vector"


def test_verify_chunks_execution(populated_chunk_store, mock_pool):
    """Test verify_chunks pipeline on graph store with mock NLI atomic assertions per-chunk evaluation."""
    from bookeeper.processing.qa_probe import AtomicAssertionItem, NLIAssertionEvaluation, NLISystemBlockMetrics

    def mock_evaluate_claims(w_raw, ideas, pool, model_name, assertions=None, system_label="old", *args, **kwargs):
        if "Deliberate practice" in w_raw:
            a1 = AtomicAssertionItem(claim_id=1, assertion="Deliberate practice requires focused attention.", is_cross_boundary=False)
            a2 = AtomicAssertionItem(claim_id=2, assertion="Feedback loops provide constant iterative signals.", is_cross_boundary=True)
            e1 = NLIAssertionEvaluation(claim_id=1, assertion=a1.assertion, classification="SUPPORTED", score=1.0, verdict="PASS")
            e2 = NLIAssertionEvaluation(claim_id=2, assertion=a2.assertion, classification="SUPPORTED", score=1.0, verdict="PASS")
            m = NLISystemBlockMetrics(
                system="old",
                total_probes=2,
                passed_probes=2,
                failed_probes=0,
                qa_recall=1.0,
                cross_sentence_total=1,
                cross_sentence_passed=1,
                seam_integrity_rate=1.0,
                evaluations=[e1, e2],
                supported_count=2,
                not_mentioned_count=0,
                against_count=0,
                support_ratio=1.0,
                hallucination_ratio=0.0,
                net_integrity_score=1.0,
                calibrated_fidelity=1.0,
                seam_net_integrity=1.0,
                seam_calibrated_fidelity=1.0,
            )
            return [a1, a2], m
        else:
            a3 = AtomicAssertionItem(claim_id=3, assertion="Spaced repetition leverages the spacing effect.", is_cross_boundary=False)
            a4 = AtomicAssertionItem(claim_id=4, assertion="Spacing effect causes memory interference.", is_cross_boundary=True)
            e3 = NLIAssertionEvaluation(claim_id=3, assertion=a3.assertion, classification="SUPPORTED", score=1.0, verdict="PASS")
            e4 = NLIAssertionEvaluation(claim_id=4, assertion=a4.assertion, classification="NOT_MENTIONED", score=-0.25, verdict="FAIL")
            m = NLISystemBlockMetrics(
                system="old",
                total_probes=2,
                passed_probes=1,
                failed_probes=1,
                qa_recall=0.5,
                cross_sentence_total=1,
                cross_sentence_passed=0,
                seam_integrity_rate=0.0,
                evaluations=[e3, e4],
                supported_count=1,
                not_mentioned_count=1,
                against_count=0,
                support_ratio=0.5,
                hallucination_ratio=0.0,
                net_integrity_score=0.375,
                calibrated_fidelity=0.375,
                seam_net_integrity=-0.25,
                seam_calibrated_fidelity=0.0,
            )
            return [a3, a4], m

    with patch("bookeeper.processing.verifier.evaluate_claims_atomic_assertions", side_effect=mock_evaluate_claims):
        report = verify_chunks(
            store=populated_chunk_store,
            pool=mock_pool,
            model_name="oracle-llama",
            percent=100.0,
            seed=42,
            concurrency=1,
        )

    assert isinstance(report, ChunkVerificationReport)
    assert report.mode == "chunk"
    assert report.stats.sampled_chunks_count == 2
    assert report.stats.total_chunks_in_graph == 2

    # Continuous ternary metrics assertions
    assert report.stats.total_assertions_count == 4
    assert report.stats.total_supported_count == 3
    assert report.stats.total_not_mentioned_count == 1
    assert report.stats.total_against_count == 0
    assert report.stats.mean_qa_recall == 0.75  # mean(1.0, 0.5)
    assert report.stats.chunks_with_perfect_recall == 1
    assert report.stats.chunks_with_omissions == 1

    # Backwards compatibility assertions
    assert report.stats.total_oracle_ideas == 4
    assert report.stats.total_matched_ideas == 3
    assert report.stats.overall_success_rate == 75.0

    # Ensure audited chunks are present
    assert len(report.audited_chunks) == 2
    assert report.audited_chunks[0].qa_recall in (1.0, 0.5)
    assert len(report.discrepancy_examples) == 1


def test_cli_verify_chunk_mode(populated_chunk_store, tmp_path, mock_pool):
    """Test CLI `bookeeper verify --mode chunk` command."""
    from bookeeper.processing.qa_probe import AtomicAssertionItem, NLIAssertionEvaluation, NLISystemBlockMetrics

    kg_file = tmp_path / "knowledge_graph.json"
    populated_chunk_store.save(str(kg_file))

    out_file = tmp_path / "verification_report.json"

    def mock_evaluate_claims(w_raw, ideas, pool, model_name, assertions=None, system_label="old", *args, **kwargs):
        a1 = AtomicAssertionItem(claim_id=1, assertion="Deliberate practice requires focused attention.", is_cross_boundary=False)
        e1 = NLIAssertionEvaluation(claim_id=1, assertion=a1.assertion, classification="SUPPORTED", score=1.0, verdict="PASS")
        m = NLISystemBlockMetrics(
            system="old",
            total_probes=1,
            passed_probes=1,
            failed_probes=0,
            qa_recall=1.0,
            cross_sentence_total=0,
            cross_sentence_passed=0,
            seam_integrity_rate=1.0,
            evaluations=[e1],
            supported_count=1,
            not_mentioned_count=0,
            against_count=0,
            support_ratio=1.0,
            hallucination_ratio=0.0,
            net_integrity_score=1.0,
            calibrated_fidelity=1.0,
            seam_net_integrity=0.0,
            seam_calibrated_fidelity=0.0,
        )
        return [a1], m

    runner = CliRunner()
    with patch("bookeeper.cli.IdeaVerifier.warmup", return_value=True), \
         patch("bookeeper.processing.verifier.evaluate_claims_atomic_assertions", side_effect=mock_evaluate_claims):
        result = runner.invoke(
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
                str(out_file),
            ],
        )

    assert result.exit_code == 0
    assert out_file.is_file()
    data = json.loads(out_file.read_text())
    assert data["mode"] == "chunk"
    assert "mean_qa_recall" in data["stats"]
    assert "total_supported_count" in data["stats"]
    assert "overall_success_rate" in data["stats"]
    assert "overall_fail_rate" in data["stats"]


def test_knowledge_extractor_extract_ideas(mock_pool):
    """Test that extract_ideas delegates to extract_section properly."""
    from bookeeper.processing.extractor import KnowledgeExtractor, SectionExtraction

    extractor = KnowledgeExtractor(pool=mock_pool, model="oracle-llama")
    expected_concept = Concept(name="Domain Concept", category="Core", summary="Summary text")

    with patch.object(extractor, "extract_section", return_value=SectionExtraction(concepts=[expected_concept])) as mock_sec:
        concepts = extractor.extract_ideas("Some chunk text", book_title="Book A", section_title="Sec 1")
        assert len(concepts) == 1
        assert concepts[0].name == "Domain Concept"
        mock_sec.assert_called_once()
