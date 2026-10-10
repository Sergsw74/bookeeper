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
    """Test verify_chunks pipeline on graph store with mock oracle extraction and deduplication."""
    def mock_extract_ideas(text, book_title="Unknown", section_title="Unknown", **kwargs):
        if "Deliberate practice" in text:
            return [
                Concept(name="Deliberate Practice", category="Learning", summary="Focus"),
                Concept(name="Feedback Loops", category="Learning", summary="Signals"),
                Concept(name="Continuous Improvement", category="Learning", summary="Nuance"),
            ]
        else:
            return [
                Concept(name="Spaced Repetition", category="Memory", summary="Retention"),
            ]

    with patch("bookeeper.processing.extractor.KnowledgeExtractor.extract_ideas", side_effect=mock_extract_ideas):
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

    # Check that ideas were extracted and matched
    assert report.stats.total_oracle_ideas >= 3
    assert report.stats.total_matched_ideas >= 3
    assert report.stats.overall_success_rate > 0.0

    # Ensure audited chunks are present
    assert len(report.audited_chunks) == 2


def test_cli_verify_chunk_mode(populated_chunk_store, tmp_path, mock_pool):
    """Test CLI `bookeeper verify --mode chunk` command."""
    kg_file = tmp_path / "knowledge_graph.json"
    populated_chunk_store.save(str(kg_file))

    out_file = tmp_path / "verification_report.json"

    def mock_extract_ideas(text, book_title="Unknown", section_title="Unknown", **kwargs):
        return [
            Concept(name="Deliberate Practice", category="Learning", summary="Focus"),
            Concept(name="Feedback Loops", category="Learning", summary="Signals"),
        ]

    runner = CliRunner()
    with patch("bookeeper.cli.IdeaVerifier.warmup", return_value=True), \
         patch("bookeeper.processing.extractor.KnowledgeExtractor.extract_ideas", side_effect=mock_extract_ideas):
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
    assert "overall_success_rate" in data["stats"]
    assert "overall_fail_rate" in data["stats"]
