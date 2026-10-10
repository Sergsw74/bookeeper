"""
Unit tests for Cross-Check Verification Mode (run_cross_check, stitch_chunks_dedup_text,
compute_chunk_disproportion, CLI verify --mode cross-check).
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.cross_checker import (
    CrossCheckAuditItem,
    CrossCheckBlockResult,
    CrossCheckReport,
    CrossCheckSummary,
    SystemBlockMetrics,
    audit_unmatched_idea,
    compute_chunk_disproportion,
    evaluate_system_against_oracle,
    find_overlapping_chunks,
    run_cross_check,
    stitch_chunks_dedup_text,
    tiered_deduplicate_ideas,
)
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import Concept
from bookeeper.processing.ollama_pool import OllamaPool, OllamaServerNode


@pytest.fixture
def mock_pool():
    return OllamaPool(
        servers=[OllamaServerNode(url="http://localhost:11434", priority=1)],
        cooldown_seconds=60,
    )


@pytest.fixture
def sample_graphs():
    """Create two small graphs representing baseline (store_a) and candidate (store_b)."""
    store_a = ConceptGraphStore()
    store_a.add_book(1, title="Test Book", author="Author A")

    # Store A: 3 contiguous chunks
    c1 = HierarchicalChunk(
        chunk_id="chunk_a1",
        chunk_idx=1,
        book_id=1,
        book_title="Test Book",
        chapter_idx=1,
        chapter_title="Chapter 1",
        section_title="Sec 1",
        subtitle="",
        breadcrumb="Ch 1",
        text="The quick brown fox jumps over the lazy dog. It was a bright sunny morning.",
        char_count=76,
        word_count=14,
    )
    c2 = HierarchicalChunk(
        chunk_id="chunk_a2",
        chunk_idx=2,
        book_id=1,
        book_title="Test Book",
        chapter_idx=1,
        chapter_title="Chapter 1",
        section_title="Sec 1",
        subtitle="",
        breadcrumb="Ch 1",
        text="It was a bright sunny morning. Animals gathered around the sparkling river.",
        char_count=74,
        word_count=11,
    )
    c3 = HierarchicalChunk(
        chunk_id="chunk_a3",
        chunk_idx=3,
        book_id=1,
        book_title="Test Book",
        chapter_idx=1,
        chapter_title="Chapter 1",
        section_title="Sec 1",
        subtitle="",
        breadcrumb="Ch 1",
        text="Animals gathered around the sparkling river. The birds sang melodic songs above.",
        char_count=79,
        word_count=12,
    )
    store_a.add_chunk(c1)
    store_a.add_chunk(c2)
    store_a.add_chunk(c3)

    con_a1 = Concept(name="Fox Agility", category="Observation", summary="Fox jumping", weight=5)
    con_a2 = Concept(name="Sunny Morning", category="Weather", summary="Bright conditions", weight=4)
    con_a3 = Concept(name="River Gathering", category="Ecology", summary="Animals at river", weight=5)
    store_a.add_concept(con_a1)
    store_a.add_concept(con_a2)
    store_a.add_concept(con_a3)
    store_a.add_idea_support_link("Fox Agility", c1, "jumps over dog", "Fox agility", "Detailed")
    store_a.add_idea_support_link("Sunny Morning", c2, "bright sunny morning", "Sunny morning", "Detailed")
    store_a.add_idea_support_link("River Gathering", c3, "gathered around river", "River gathering", "Detailed")

    # Store B: 2 chunks covering roughly the same span
    store_b = ConceptGraphStore()
    store_b.add_book(1, title="Test Book", author="Author A")

    cb1 = HierarchicalChunk(
        chunk_id="chunk_b1",
        chunk_idx=1,
        book_id=1,
        book_title="Test Book",
        chapter_idx=1,
        chapter_title="Chapter 1",
        section_title="Sec 1",
        subtitle="",
        breadcrumb="Ch 1",
        text="The quick brown fox jumps over the lazy dog. It was a bright sunny morning. Animals gathered around the sparkling river.",
        char_count=124,
        word_count=20,
    )
    cb2 = HierarchicalChunk(
        chunk_id="chunk_b2",
        chunk_idx=2,
        book_id=1,
        book_title="Test Book",
        chapter_idx=1,
        chapter_title="Chapter 1",
        section_title="Sec 1",
        subtitle="",
        breadcrumb="Ch 1",
        text="Animals gathered around the sparkling river. The birds sang melodic songs above.",
        char_count=79,
        word_count=12,
    )
    store_b.add_chunk(cb1)
    store_b.add_chunk(cb2)

    con_b1 = Concept(name="Fox Agility", category="Observation", summary="Fox jumping", weight=5)
    con_b2 = Concept(name="Sunny Morning", category="Weather", summary="Bright conditions", weight=4)
    con_b3 = Concept(name="River Gathering", category="Ecology", summary="Animals at river", weight=5)
    con_b4 = Concept(name="Bird Songs", category="Ecology", summary="Melodic songs", weight=3)
    store_b.add_concept(con_b1)
    store_b.add_concept(con_b2)
    store_b.add_concept(con_b3)
    store_b.add_concept(con_b4)
    store_b.add_idea_support_link("Fox Agility", cb1, "jumps over dog", "Fox agility", "Detailed")
    store_b.add_idea_support_link("Sunny Morning", cb1, "bright sunny morning", "Sunny morning", "Detailed")
    store_b.add_idea_support_link("River Gathering", cb1, "gathered around river", "River gathering", "Detailed")
    store_b.add_idea_support_link("Bird Songs", cb2, "melodic songs", "Bird songs", "Detailed")

    return store_a, store_b


def test_stitch_chunks_dedup_text():
    """Verify seam overlap stitching removes repeated sentences."""
    texts = [
        "First sentence. Overlap sentence.",
        "Overlap sentence. Third sentence.",
        "Third sentence. Final conclusion.",
    ]
    stitched = stitch_chunks_dedup_text(texts)
    assert stitched.count("Overlap sentence.") == 1
    assert stitched.count("Third sentence.") == 1
    assert "First sentence." in stitched
    assert "Final conclusion." in stitched

    # Empty list and single string handling
    assert stitch_chunks_dedup_text([]) == ""
    assert stitch_chunks_dedup_text(["Only one sentence."]) == "Only one sentence."


def test_compute_chunk_disproportion():
    """Verify mathematical disproportion metric: sum(len(A)) / sum(len(A ∪ B))."""
    texts_a = ["Hello world from Branch A.", "Second piece of text."]
    texts_b = ["Hello world from Branch A.", "Second piece of text.", "Extra extension."]
    w_raw = stitch_chunks_dedup_text(texts_a)

    sum_a, sum_b, union_len, disproportion = compute_chunk_disproportion(texts_a, texts_b, w_raw)
    assert sum_a == len(texts_a[0]) + len(texts_a[1])
    assert sum_b == len(texts_b[0]) + len(texts_b[1]) + len(texts_b[2])
    assert 0.0 < disproportion <= 1.0

    # Identical sets
    _, _, _, ident = compute_chunk_disproportion(texts_a, texts_a, w_raw)
    assert ident == 1.0

    # Empty inputs
    _, _, _, empty_disp = compute_chunk_disproportion([], texts_b, "")
    assert empty_disp == 0.0


def test_find_overlapping_chunks(sample_graphs):
    """Verify chunk matching via character/token boundary overlap."""
    _, store_b = sample_graphs
    reference = "The quick brown fox jumps over the lazy dog. It was a bright sunny morning."

    matched = find_overlapping_chunks(store_b, book_id=1, w_raw=reference, min_overlap=0.80)
    assert len(matched) == 1
    chunk_ids = [m.get("chunk_id") for m in matched]
    assert chunk_ids == ["chunk_b1"]


def test_tiered_deduplicate_ideas():
    """Verify pairwise deduplication of candidate concepts."""
    mock_dedup = MagicMock(spec=EntityDeduplicator)
    mock_dedup.is_same_concept.return_value = (False, 0.5, "none", "", "")

    c1 = Concept(name="Idea Alpha", category="C", summary="A", weight=1)
    c2 = Concept(name="Idea Alpha", category="C", summary="A copy", weight=1)
    c3 = Concept(name="Idea Beta", category="C", summary="B", weight=1)

    # Make second identical concept match
    def mock_is_same(u, c):
        if u.name == c.name:
            return (True, 1.0, "exact", u.name, "exact")
        return (False, 0.5, "none", "", "")

    mock_dedup.is_same_concept.side_effect = mock_is_same

    unique = tiered_deduplicate_ideas([c1, c2, c3], deduplicator=mock_dedup)
    assert len(unique) == 2
    assert {c.name for c in unique} == {"Idea Alpha", "Idea Beta"}


def test_evaluate_system_against_oracle():
    """Verify recall, grounded precision, and truncation rate computation."""
    mock_dedup = MagicMock(spec=EntityDeduplicator)
    # Idea A matches Oracle A, Idea B matches Oracle B
    mock_dedup.is_same_concept.side_effect = lambda a, b: (
        (True, 1.0, "exact", a, "same") if str(a).lower() == str(b).lower() else (False, 0.4, "none", "", "")
    )

    oracle_ideas = ["Idea A", "Idea B", "Idea C"]
    system_ideas = ["Idea A", "Idea B", "Idea D"]

    mock_pool = MagicMock()
    mock_pool.generate.return_value = (
        json.dumps({"verdict": "TRUNCATION_ARTIFACT", "rationale": "Missing condition."}),
        {},
    )

    metrics = evaluate_system_against_oracle(
        candidate_ideas=system_ideas,
        oracle_ideas=oracle_ideas,
        w_raw="Reference passage text",
        system_label="new",
        deduplicator=mock_dedup,
        pool=mock_pool,
        model_name="test-model",
    )

    assert metrics.oracle_recall == pytest.approx(2 / 3, 0.01)
    assert len(metrics.retained_oracle_ideas) == 2
    assert len(metrics.dropped_oracle_ideas) == 1
    assert metrics.truncation_rate == pytest.approx(1 / 3, 0.01)


def test_run_cross_check_e2e(sample_graphs, mock_pool):
    """Verify end-to-end cross-check pipeline execution with mocked LLM generation."""
    store_a, store_b = sample_graphs

    def fake_generate(model, prompt, **kwargs):
        if "extract all core standalone ideas" in prompt.lower() or "knowledge extractor" in prompt.lower():
            return (
                json.dumps({
                    "ideas": [
                        {"name": "Fox Agility", "reasoning": "Fox jumping over dog"},
                        {"name": "Sunny Morning", "reasoning": "Bright morning"},
                        {"name": "River Gathering", "reasoning": "Animals at river"},
                    ]
                }),
                {},
            )
        else:
            return (
                json.dumps({
                    "verdict": "VALID_DETAIL",
                    "rationale": "Accurate detail present in text",
                }),
                {},
            )

    mock_pool.generate = MagicMock(side_effect=fake_generate)

    report = run_cross_check(
        store_a=store_a,
        store_b=store_b,
        num_blocks=1,
        branch_a_name="Baseline",
        branch_b_name="Candidate",
        model_name="test-model",
        pool=mock_pool,
    )

    assert isinstance(report, CrossCheckReport)
    assert report.summary.total_samples == 1
    assert report.baseline_branch == "Baseline"
    assert report.candidate_branch == "Candidate"
    assert len(report.samples) == 1
    assert report.summary.mean_chunk_disproportion > 0.0
    assert report.summary.mean_old_recall >= 0.0
    assert report.summary.mean_new_recall >= 0.0


def test_cross_check_cli_command(tmp_path, sample_graphs):
    """Test CLI command `bookeeper verify --mode cross-check --graph-file ... --compare-graph ...`"""
    store_a, store_b = sample_graphs
    path_a = tmp_path / "graph_a.json"
    path_b = tmp_path / "graph_b.json"
    out_rep = tmp_path / "cross_check_out.json"

    store_a.save(path_a)
    store_b.save(path_b)

    runner = CliRunner()

    with patch("bookeeper.cli.run_cross_check") as mock_run:
        summary_obj = CrossCheckSummary(
            total_samples=1,
            mean_old_recall=0.85,
            mean_new_recall=0.90,
            mean_delta_recall=0.05,
            mean_old_precision=0.95,
            mean_new_precision=0.97,
            mean_old_truncation_rate=0.0,
            mean_new_truncation_rate=0.01,
            mean_chunk_disproportion=0.92,
            decision_pass=True,
            pass_reason="All gates passed.",
        )
        mock_rep = CrossCheckReport(
            model_name="test-model",
            baseline_branch="graph_a.json",
            candidate_branch="graph_b.json",
            summary=summary_obj,
            samples=[],
        )
        mock_run.return_value = mock_rep

        result = runner.invoke(
            app,
            [
                "verify",
                "--mode",
                "cross-check",
                "--graph-file",
                str(path_a),
                "--compare-graph",
                str(path_b),
                "--blocks",
                "1",
                "--output",
                str(out_rep),
            ],
        )

        assert result.exit_code == 0
        assert "Cross-Check Verification Summary" in result.stdout
        assert out_rep.exists()


def test_cross_check_cli_with_ab_test_run_dir(tmp_path, sample_graphs):
    """Test CLI command `bookeeper verify --mode cross-check --compare-graph <run_dir>` discovers both branches."""
    store_a, store_b = sample_graphs
    run_dir = tmp_path / "ab_test_runs" / "run_20261010_000000_master_vs_candidate"
    dir_a = run_dir / "branch_A_master"
    dir_b = run_dir / "branch_B_candidate"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    store_a.save(dir_a / "knowledge_graph.json")
    store_b.save(dir_b / "knowledge_graph.json")
    out_rep = tmp_path / "cross_check_run_out.json"

    runner = CliRunner()

    with patch("bookeeper.cli.run_cross_check") as mock_run:
        summary_obj = CrossCheckSummary(
            total_samples=1,
            mean_old_recall=0.85,
            mean_new_recall=0.90,
            mean_delta_recall=0.05,
            mean_old_precision=0.95,
            mean_new_precision=0.97,
            mean_old_truncation_rate=0.0,
            mean_new_truncation_rate=0.01,
            mean_chunk_disproportion=0.92,
            decision_pass=True,
            pass_reason="All gates passed.",
        )
        mock_rep = CrossCheckReport(
            model_name="test-model",
            baseline_branch="master",
            candidate_branch="candidate",
            summary=summary_obj,
            samples=[],
        )
        mock_run.return_value = mock_rep

        # Only pass --compare-graph with the run directory!
        result = runner.invoke(
            app,
            [
                "verify",
                "--mode",
                "cross-check",
                "--compare-graph",
                str(run_dir),
                "--blocks",
                "1",
                "--output",
                str(out_rep),
            ],
        )

        assert result.exit_code == 0
        assert "Cross-Check Verification Summary" in result.stdout
        assert out_rep.exists()
        # Verify run_cross_check was called with discovered branch names
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["branch_a_name"] == "master"
        assert call_kwargs["branch_b_name"] == "candidate"

