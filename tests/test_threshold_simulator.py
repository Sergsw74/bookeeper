"""
Unit and Integration Tests for Knowledge Base Threshold Simulation & Optimization Suite.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.evaluation.threshold_simulator import (
    PreloadedBookData,
    SimulationSuiteReport,
    SimulationVariationResult,
    ThresholdPair,
    parse_threshold_pairs,
    preload_simulation_data,
    run_threshold_simulation_suite,
    simulate_variation,
)

runner = CliRunner()


# ==============================================================================
# 1. Parsing & Validation Tests
# ==============================================================================


def test_parse_threshold_pairs_strings():
    # Standard colon-delimited
    p1 = parse_threshold_pairs("0.60:0.90, 0.65:0.90, 0.70:0.90")
    assert len(p1) == 3
    assert p1[0].low == 0.60 and p1[0].high == 0.90
    assert p1[1].low == 0.65 and p1[1].high == 0.90
    assert p1[2].low == 0.70 and p1[2].high == 0.90

    # Dash-delimited
    p2 = parse_threshold_pairs("0.60-0.90, 0.65-0.95")
    assert len(p2) == 2
    assert p2[1].high == 0.95

    # Parentheses-tuples format
    p3 = parse_threshold_pairs("(0.55, 0.85), (0.70, 0.90)")
    assert len(p3) == 2
    assert p3[0].low == 0.55 and p3[0].high == 0.85

    # Slash-delimited
    p4 = parse_threshold_pairs("0.62/0.88, 0.68/0.92")
    assert len(p4) == 2
    assert p4[0].low == 0.62 and p4[0].high == 0.88


def test_parse_threshold_pairs_sequences():
    # Tuples
    p_tuples = parse_threshold_pairs([(0.60, 0.90), (0.70, 0.95)])
    assert len(p_tuples) == 2
    assert p_tuples[0].low == 0.60

    # Dicts
    p_dicts = parse_threshold_pairs([
        {"low": 0.65, "high": 0.90, "label": "Custom-1"},
        {"unassigned_threshold": 0.70, "high_confidence_threshold": 0.92},
    ])
    assert len(p_dicts) == 2
    assert p_dicts[0].label == "Custom-1"
    assert p_dicts[1].low == 0.70 and p_dicts[1].high == 0.92

    # Default fallback on empty
    p_default = parse_threshold_pairs("")
    assert len(p_default) == 5
    assert p_default[1].low == 0.65 and p_default[1].high == 0.90


def test_parse_threshold_pairs_invalid():
    # low > high
    with pytest.raises(ValueError, match="cannot exceed"):
        ThresholdPair(low=0.90, high=0.60)

    # Empty string with no matches
    with pytest.raises(ValueError, match="Could not parse"):
        parse_threshold_pairs("invalid_text_without_numbers")


# ==============================================================================
# 2. Simulation Logic & Dynamic Routing Tests
# ==============================================================================


@pytest.fixture
def mock_simulation_corpus():
    """Generates synthetic preloaded book data with known unit vectors."""
    dim = 16
    np.random.seed(42)

    # Chunk 0 has vector v0
    v0 = np.random.randn(dim).astype(np.float32)
    v0 /= np.linalg.norm(v0)

    # Chunk 1 is orthogonal to v0
    v1 = np.random.randn(dim).astype(np.float32)
    v1 -= np.dot(v1, v0) * v0
    v1 /= np.linalg.norm(v1)

    # Chunk 2 is nearly identical to v0 (sim ~ 0.95)
    v2 = v0 * 0.95 + v1 * np.sqrt(1 - 0.95**2)
    v2 /= np.linalg.norm(v2)

    # Chunk 3 has intermediate similarity to v0 (sim ~ 0.70)
    v3 = v0 * 0.70 + v1 * np.sqrt(1 - 0.70**2)
    v3 /= np.linalg.norm(v3)

    book = PreloadedBookData(
        book_id="1",
        book_title="Test Book",
        publication_date="2024-01-01",
        chunks=[
            {"chunk_id": "c0", "chunk_idx": 0, "section_title": "Sec 1", "char_count": 500, "embedding": v0.tolist()},
            {"chunk_id": "c1", "chunk_idx": 1, "section_title": "Sec 1", "char_count": 500, "embedding": v1.tolist()},
            {"chunk_id": "c2", "chunk_idx": 2, "section_title": "Sec 2", "char_count": 500, "embedding": v2.tolist()},
            {"chunk_id": "c3", "chunk_idx": 3, "section_title": "Sec 2", "char_count": 500, "embedding": v3.tolist()},
        ],
        extractions={
            "c0": [{"name": "Concept A", "summary": "Idea from c0"}],
            "c1": [{"name": "Concept B", "summary": "Idea from c1"}],
            "c2": [{"name": "Concept C", "summary": "Idea from c2"}],
            "c3": [{"name": "Concept D", "summary": "Idea from c3"}],
        },
    )

    concept_map = {
        "concept a": v0.tolist(),
        "concept b": v1.tolist(),
        "concept c": v2.tolist(),
        "concept d": v3.tolist(),
    }

    return ([book], concept_map)


def test_simulate_variation_threshold_dynamics(mock_simulation_corpus):
    books, concept_map = mock_simulation_corpus

    # Case 1: Low boundary = 0.65, High boundary = 0.90
    # c0: unassigned (initial, KB empty) -> adds Concept A (v0)
    # c1: sim(c1, v0) = 0.0 < 0.65 -> unassigned -> adds Concept B (v1)
    # c2: sim(c2, v0) ~ 0.95 >= 0.90 -> high confidence -> direct assignment (no new concept)
    # c3: sim(c3, v0) ~ 0.70 in [0.65, 0.90) -> limbo -> skipped when skip_limbo=True
    pair_mid = ThresholdPair(low=0.65, high=0.90)
    res_mid = simulate_variation(pair_mid, books, concept_map, tau=0.75, skip_limbo=True)

    assert res_mid.unassigned_count == 2  # c0, c1
    assert res_mid.high_confidence_count == 1  # c2
    assert res_mid.limbo_count == 1  # c3
    assert res_mid.nodes_count == 2  # Concept A, Concept B
    assert res_mid.parent_chunks_count == 2
    assert res_mid.skipped_chunks_count == 2
    assert res_mid.parent_chunk_ratio_pct == 50.0  # 2 of 4 chunks
    assert res_mid.coverage_at_tau_pct > 0.0

    # Case 2: Raise low boundary to 0.75 (so c3 with sim 0.70 becomes unassigned)
    pair_high = ThresholdPair(low=0.75, high=0.90)
    res_high = simulate_variation(pair_high, books, concept_map, tau=0.75, skip_limbo=True)

    assert res_high.unassigned_count == 3  # c0, c1, c3
    assert res_high.high_confidence_count == 1  # c2
    assert res_high.limbo_count == 0
    assert res_high.nodes_count == 3  # Concept A, Concept B, Concept D
    assert res_high.parent_chunks_count == 3
    assert res_high.parent_chunk_ratio_pct == 75.0


def test_simulate_variation_no_skip_limbo(mock_simulation_corpus):
    books, concept_map = mock_simulation_corpus

    # When skip_limbo=False, limbo chunks are processed and extracted
    pair = ThresholdPair(low=0.65, high=0.90)
    res = simulate_variation(pair, books, concept_map, tau=0.75, skip_limbo=False)

    assert res.limbo_count == 1
    # Parent chunks include unassigned + limbo
    assert res.parent_chunks_count == 3  # c0, c1, c3
    assert res.nodes_count == 3


def test_simulate_variation_clean_start_between_runs(mock_simulation_corpus):
    """
    Explicitly verify that each variation starts with a 100% clean, empty graph and index,
    and that no concepts, chunks, or metrics leak between consecutive variations.
    """
    books, concept_map = mock_simulation_corpus

    # Run 1: High low threshold (0.75) -> extracts 3 concepts (A, B, D)
    pair_high = ThresholdPair(low=0.75, high=0.90, label="High-Run")
    res_high = simulate_variation(pair_high, books, concept_map, tau=0.75, skip_limbo=True)
    assert res_high.nodes_count == 3

    # Run 2: Immediately run low threshold (0.65) -> must extract exactly 2 concepts (A, B)
    # If state leaked from Run 1, nodes_count would be >= 3
    pair_mid = ThresholdPair(low=0.65, high=0.90, label="Mid-Run")
    res_mid = simulate_variation(pair_mid, books, concept_map, tau=0.75, skip_limbo=True)
    assert res_mid.nodes_count == 2
    assert res_mid.unassigned_count == 2
    assert res_mid.parent_chunks_count == 2

    # Run 3: Run again to ensure complete determinism and clean start
    res_mid_repeat = simulate_variation(pair_mid, books, concept_map, tau=0.75, skip_limbo=True)
    assert res_mid_repeat.nodes_count == 2
    assert res_mid_repeat.coverage_at_tau_pct == res_mid.coverage_at_tau_pct


# ==============================================================================
# 3. Report Generation & Mermaid Charts
# ==============================================================================


def test_simulation_suite_report_markdown_and_rich(mock_simulation_corpus):
    books, concept_map = mock_simulation_corpus

    p1 = ThresholdPair(low=0.60, high=0.90, label="0.60:0.90")
    p2 = ThresholdPair(low=0.70, high=0.90, label="0.70:0.90")

    res1 = simulate_variation(p1, books, concept_map, tau=0.75)
    res2 = simulate_variation(p2, books, concept_map, tau=0.75)

    suite = SimulationSuiteReport(
        tau=0.75,
        total_corpus_chunks=4,
        total_corpus_chars=2000,
        total_books=1,
        skip_limbo=True,
        variations=[res1, res2],
    )

    # 1. Rich Table test
    rich_table = suite.to_rich_table()
    assert rich_table is not None
    assert len(rich_table.columns) == 12

    # 2. Markdown output test
    md = suite.to_markdown()
    assert "# Knowledge Base Threshold Simulation & Optimization Report" in md
    assert "| **0.60:0.90** |" in md
    assert "| **0.70:0.90** |" in md
    assert "xychart-beta" in md
    assert 'title "Coverage vs Compute Ratio Across Variations"' in md
    assert 'title "Compression Leverage Multiplier Across Variations"' in md
    assert 'title "Mean Proximity (mu) vs Worst-Decile (p10)"' in md
    assert 'title "Embedding Space Dispersion Regularity (sigma)"' in md
    assert "## 3. Comparative Observations & Pareto Analysis" in md


# ==============================================================================
# 4. CLI Command Integration Test
# ==============================================================================


def test_cli_simulate_thresholds(tmp_path):
    # Setup mock state cache directory
    state_dir = tmp_path / "state_cache"
    state_dir.mkdir(parents=True)
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir(parents=True)

    dim = 8
    v0 = (np.ones(dim, dtype=np.float32) / np.sqrt(dim)).tolist()

    # Create dummy state file
    state_file = state_dir / "book_99_state.json"
    state_data = {
        "version": 1,
        "book_id": "99",
        "book_title": "Test Title",
        "publication_date": "2023-01-01",
        "processed_chunks": {
            "c99_0": {
                "status": "extracted",
                "extraction": {
                    "concepts": [{"name": "Mock Concept", "summary": "A mock idea", "category": "Idea"}]
                },
            }
        },
    }
    state_file.write_text(json.dumps(state_data), encoding="utf-8")

    # Create companion chunk file
    chunk_file = chunks_dir / "book_99_chunks.json"
    chunk_data = {
        "chunks": [
            {
                "chunk_id": "c99_0",
                "chunk_idx": 0,
                "section_title": "Section 1",
                "char_count": 300,
                "embedding": v0,
            }
        ]
    }
    chunk_file.write_text(json.dumps(chunk_data), encoding="utf-8")

    out_report = tmp_path / "custom_report.md"

    res = runner.invoke(
        app,
        [
            "simulate-thresholds",
            "--threshold-pairs",
            "0.60:0.90,0.70:0.90",
            "--state-cache-dir",
            str(state_dir),
            "--chunks-dir",
            str(chunks_dir),
            "--output-report",
            str(out_report),
        ],
    )

    assert res.exit_code == 0
    assert "Knowledge Base Threshold Simulation Benchmark" in res.output
    assert "Simulation Finished" in res.output
    assert out_report.is_file()
    report_text = out_report.read_text(encoding="utf-8")
    assert "0.60:0.90" in report_text
    assert "0.70:0.90" in report_text
    assert "xychart-beta" in report_text
