"""
Unit tests for Knowledge Graph Embedding Space Verification Suite ('graph' / 'kg' mode).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.extractor import Concept
from bookeeper.processing.graph_evaluator import (
    GraphSpaceComparisonReport,
    GraphSpaceMetrics,
    compare_graph_spaces,
    evaluate_graph_space,
    evaluate_skipping_leverage,
    extract_chunk_embeddings,
    extract_concept_embeddings,
    format_graph_comparison_table,
    format_graph_space_table,
    get_parent_chunk_ids,
    run_graph_space_comparison,
    run_graph_space_evaluation,
)

runner = CliRunner()


@pytest.fixture
def mock_stores_with_embeddings():
    """Builds two ConceptGraphStore objects populated with Chunk and Idea embeddings."""
    dim = 32
    np.random.seed(42)

    store_a = ConceptGraphStore()
    store_b = ConceptGraphStore()

    store_a.add_book(1, title="Corpus Book Alpha")
    store_b.add_book(1, title="Corpus Book Alpha")

    # Generate 10 base universe chunk vectors
    base_chunks = np.random.randn(10, dim).astype(np.float32)
    base_chunks /= np.linalg.norm(base_chunks, axis=1, keepdims=True)

    for i in range(10):
        chk = HierarchicalChunk(
            chunk_id=f"chk_{i}",
            chunk_idx=i,
            book_id=1,
            book_title="Corpus Book Alpha",
            chapter_idx=1,
            chapter_title="Chapter 1",
            section_title="Section 1",
            subtitle="",
            breadcrumb="Ch 1",
            text=f"Text content for literature chunk number {i}.",
            char_count=50,
            word_count=8,
            embedding=base_chunks[i].tolist(),
        )
        store_a.add_chunk(chk)
        store_b.add_chunk(chk)

    # Store A (Dense Graph): 6 concepts closely matching chunks 0..5
    for i in range(6):
        c_vec = base_chunks[i] + np.random.normal(0, 0.05, dim).astype(np.float32)
        c_vec /= np.linalg.norm(c_vec)
        c = Concept(
            name=f"Concept A{i}",
            category="Domain",
            summary=f"Idea summary A{i}",
            weight=1,
            embedding=c_vec.tolist(),
        )
        store_a.add_concept(c, embedding=c_vec.tolist())

    # Store B (Sparse / Efficient Graph): 4 concepts covering chunks 0..9 with diverse dispersion
    for i in range(4):
        # Blend pairs of chunks to represent compressed semantic concepts
        c_vec = (base_chunks[i * 2] + base_chunks[min(9, i * 2 + 1)]) / 2.0
        c_vec /= np.linalg.norm(c_vec)
        c = Concept(
            name=f"Concept B{i}",
            category="Domain",
            summary=f"Idea summary B{i}",
            weight=1,
            embedding=c_vec.tolist(),
        )
        store_b.add_concept(c, embedding=c_vec.tolist())

    return store_a, store_b


def test_evaluate_graph_space_empty():
    """Verify evaluate_graph_space handles empty inputs safely without division errors."""
    u = np.empty((0, 16), dtype=np.float32)
    g = np.empty((0, 16), dtype=np.float32)
    res = evaluate_graph_space(u, g)
    assert res["nodes_count"] == 0
    assert res["chamfer_mean"] == 0.0
    assert res["vendi_diversity_score"] == 0.0


def test_evaluate_graph_space_perfect_coverage():
    """Verify evaluate_graph_space on identical universe and graph vectors achieves 100% coverage."""
    dim = 16
    vecs = np.eye(5, dim, dtype=np.float32)  # 5 orthogonal unit vectors

    res = evaluate_graph_space(universe_embeddings=vecs, graph_embeddings=vecs, tau=0.75)
    assert res["nodes_count"] == 5
    assert res["universe_chunks_count"] == 5
    assert res["chamfer_mean"] == pytest.approx(1.0, 1e-4)
    assert res["chamfer_std"] == pytest.approx(0.0, 1e-4)
    assert res["worst_decile_p10"] == pytest.approx(1.0, 1e-4)
    assert res["median_p50"] == pytest.approx(1.0, 1e-4)
    assert res["coverage_at_tau_pct"] == 100.0
    assert res["worst_case_gap"] == pytest.approx(0.0, 1e-4)
    assert res["vendi_diversity_score"] > 3.0  # High diversity across orthogonal concept modes
    assert res["coverage_per_node_efficiency"] == pytest.approx(20.0, 1e-4)


def test_vendi_score_properties():
    """Verify Vendi Score equals 1.0 for completely redundant/identical concept vectors."""
    dim = 32
    v = np.random.randn(1, dim).astype(np.float32)
    identical_g = np.vstack([v, v, v, v, v])

    res = evaluate_graph_space(universe_embeddings=identical_g, graph_embeddings=identical_g)
    assert res["vendi_diversity_score"] == pytest.approx(1.0, 1e-3)
    assert res["vendi_per_node"] == pytest.approx(1.0 / 5, 1e-3)


def test_compare_graph_spaces_decision_matrix():
    """Verify compare_graph_spaces decision matrix gating conditions."""
    dim = 16
    np.random.seed(123)
    u = np.random.randn(50, dim).astype(np.float32)

    # G1: good coverage
    g1 = u[:15].copy()

    # G2: slightly fewer nodes but preserving coverage
    g2 = u[:10].copy()

    report = compare_graph_spaces(
        universe_embeddings=u,
        baseline_embeddings=g1,
        candidate_embeddings=g2,
        tau=0.70,
        label_a="G1 Dense",
        label_b="G2 Compressed",
    )

    assert isinstance(report, GraphSpaceComparisonReport)
    assert report.universe_chunks_count == 50
    assert report.baseline_metrics.nodes_count == 15
    assert report.candidate_metrics.nodes_count == 10
    assert "gate_checklist" in report.model_dump()
    assert "coverage_retention" in report.gate_checklist
    assert "coverage_regularity" in report.gate_checklist
    assert "blind_spot_ceiling" in report.gate_checklist
    assert "concept_quality_economy" in report.gate_checklist

    # Rendering Rich table
    table = format_graph_comparison_table(report)
    assert "G1 Dense" in table.title or any("G1 Dense" in str(c.header) for c in table.columns)


def test_extract_embeddings_from_store(mock_stores_with_embeddings):
    """Verify concept and chunk embeddings are cleanly extracted from ConceptGraphStore."""
    store_a, _ = mock_stores_with_embeddings

    c_names, c_mat = extract_concept_embeddings(store_a)
    assert len(c_names) == 6
    assert c_mat.shape == (6, 32)

    u_ids, u_mat = extract_chunk_embeddings(store_a)
    assert len(u_ids) == 10
    assert u_mat.shape == (10, 32)


def test_run_graph_space_evaluation_end_to_end(mock_stores_with_embeddings):
    """Verify run_graph_space_evaluation computes GraphSpaceMetrics directly on a store."""
    store_a, _ = mock_stores_with_embeddings

    metrics = run_graph_space_evaluation(store=store_a, tau=0.75)
    assert isinstance(metrics, GraphSpaceMetrics)
    assert metrics.nodes_count == 6
    assert metrics.universe_chunks_count == 10
    assert 0.0 <= metrics.chamfer_mean <= 1.0
    assert 0.0 <= metrics.coverage_at_tau_pct <= 100.0

    tbl = format_graph_space_table(metrics)
    assert "Chamfer Mean Coverage" in str(tbl.columns[0]._cells)


def test_run_graph_space_comparison_end_to_end(mock_stores_with_embeddings):
    """Verify run_graph_space_comparison executes comparison across two stores."""
    store_a, store_b = mock_stores_with_embeddings

    report = run_graph_space_comparison(
        store_a=store_a,
        store_b=store_b,
        tau=0.75,
        label_a="Baseline (A)",
        label_b="Candidate (B)",
    )
    assert isinstance(report, GraphSpaceComparisonReport)
    assert report.baseline_metrics.nodes_count == 6
    assert report.candidate_metrics.nodes_count == 4
    assert report.universe_chunks_count == 10


def test_cli_verify_mode_kg(tmp_path: Path, mock_stores_with_embeddings):
    """Verify `bookeeper verify --mode kg` executes from CLI."""
    store_a, _ = mock_stores_with_embeddings

    kg_path = tmp_path / "knowledge_graph.json"
    store_a.save(kg_path)

    out_rep = tmp_path / "kg_report.json"
    res = runner.invoke(app, [
        "verify",
        "--graph-file", str(kg_path),
        "--mode", "kg",
        "--tau", "0.70",
        "--output", str(out_rep),
    ])

    assert res.exit_code == 0, f"Error: {res.stdout}"
    assert "Knowledge Graph Embedding Space Metrics" in res.stdout
    assert "Chamfer Mean Coverage" in res.stdout
    assert out_rep.is_file()

    with open(out_rep, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "chamfer_mean" in data
    assert "vendi_diversity_score" in data


def test_cli_ab_compare_mode_kg(tmp_path: Path, mock_stores_with_embeddings):
    """Verify `bookeeper ab-compare --mode kg` executes comparative A/B audit."""
    store_a, store_b = mock_stores_with_embeddings

    kg_a = tmp_path / "kg_a.json"
    kg_b = tmp_path / "kg_b.json"
    store_a.save(kg_a)
    store_b.save(kg_b)

    out_rep = tmp_path / "ab_kg_report.json"
    res = runner.invoke(app, [
        "ab-compare",
        str(kg_a),
        str(kg_b),
        "--mode", "kg",
        "--tau", "0.75",
        "--output", str(out_rep),
    ])

    assert res.exit_code == 0, f"Error: {res.stdout}"
    assert "Knowledge Graph Embedding Space Comparison" in res.stdout
    assert "Chamfer" in res.stdout
    assert "Vendi" in res.stdout
    assert out_rep.is_file()
    assert out_rep.with_suffix(".md").is_file()

    with open(out_rep, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "graph_space" in data
    gs = data["graph_space"]
    assert "baseline_metrics" in gs
    assert "candidate_metrics" in gs
    assert "gate_checklist" in gs


def test_evaluate_skipping_leverage_generalization_surplus():
    """Verify evaluate_skipping_leverage calculates positive surplus when centroids generalize."""
    dim = 16
    np.random.seed(42)

    # 3 parent chunks
    p_vecs = np.eye(3, dim, dtype=np.float32)

    # 6 skipped chunks that are very close to parent centroids (e.g. slight jitter)
    s_vecs = []
    for i in range(6):
        base = p_vecs[i % 3]
        jittered = base + np.random.normal(0, 0.05, dim).astype(np.float32)
        jittered /= np.linalg.norm(jittered)
        s_vecs.append(jittered)
    s_vecs = np.array(s_vecs, dtype=np.float32)

    # 3 graph concept centroids equal to parent chunks
    g_vecs = p_vecs.copy()

    res = evaluate_skipping_leverage(
        parent_chunk_embeddings=p_vecs,
        skipped_chunk_embeddings=s_vecs,
        graph_embeddings=g_vecs,
        tau=0.75,
    )

    assert res["parent_chunks_count"] == 3
    assert res["skipped_chunks_count"] == 6
    assert res["parent_chunk_ratio_pct"] == pytest.approx(33.33, 0.05)
    assert res["total_coverage_pct"] == 100.0
    assert res["skipped_retention_pct"] == 100.0
    assert res["generalization_surplus_pct"] == pytest.approx(66.67, 0.05)
    assert res["compression_leverage_factor"] == pytest.approx(3.0, 0.05)


def test_evaluate_skipping_leverage_narrow_extraction():
    """Verify evaluate_skipping_leverage when concepts only match parent chunks and leave skipped chunks orphaned."""
    dim = 16
    # 2 parent chunks along axes 0 and 1
    p_vecs = np.zeros((2, dim), dtype=np.float32)
    p_vecs[0, 0] = 1.0
    p_vecs[1, 1] = 1.0

    # 2 skipped chunks along axes 2 and 3 (completely orthogonal to ideas)
    s_vecs = np.zeros((2, dim), dtype=np.float32)
    s_vecs[0, 2] = 1.0
    s_vecs[1, 3] = 1.0

    # Ideas only cover axes 0 and 1
    g_vecs = p_vecs.copy()

    res = evaluate_skipping_leverage(
        parent_chunk_embeddings=p_vecs,
        skipped_chunk_embeddings=s_vecs,
        graph_embeddings=g_vecs,
        tau=0.75,
    )

    assert res["parent_chunks_count"] == 2
    assert res["skipped_chunks_count"] == 2
    assert res["parent_chunk_ratio_pct"] == 50.0
    assert res["total_coverage_pct"] == 50.0
    assert res["skipped_retention_pct"] == 0.0  # None of the skipped chunks retained
    assert res["generalization_surplus_pct"] == 0.0  # Zero surplus
    assert res["compression_leverage_factor"] == 1.0  # No compression leverage


def test_evaluate_skipping_leverage_zero_skipped_chunks():
    """Verify evaluate_skipping_leverage with zero skipped chunks (all parent chunks)."""
    dim = 8
    p_vecs = np.eye(4, dim, dtype=np.float32)
    s_vecs = np.empty((0, dim), dtype=np.float32)
    g_vecs = p_vecs.copy()

    res = evaluate_skipping_leverage(
        parent_chunk_embeddings=p_vecs,
        skipped_chunk_embeddings=s_vecs,
        graph_embeddings=g_vecs,
        tau=0.75,
    )

    assert res["parent_chunks_count"] == 4
    assert res["skipped_chunks_count"] == 0
    assert res["parent_chunk_ratio_pct"] == 100.0
    assert res["total_coverage_pct"] == 100.0
    assert res["skipped_retention_pct"] == 100.0
    assert res["generalization_surplus_pct"] == 0.0
    assert res["compression_leverage_factor"] == 1.0


def test_get_parent_chunk_ids_from_store():
    """Verify get_parent_chunk_ids discovers parent chunks connected to ideas."""
    store = ConceptGraphStore()
    chk1 = HierarchicalChunk(
        chunk_id="chk_parent_1",
        chunk_idx=0,
        book_id=1,
        book_title="Test",
        chapter_idx=1,
        chapter_title="Ch 1",
        section_title="Sec 1",
        subtitle="",
        breadcrumb="Ch 1",
        text="Parent chunk text 1",
        char_count=20,
        word_count=4,
    )
    chk2 = HierarchicalChunk(
        chunk_id="chk_skipped_2",
        chunk_idx=1,
        book_id=1,
        book_title="Test",
        chapter_idx=1,
        chapter_title="Ch 1",
        section_title="Sec 1",
        subtitle="",
        breadcrumb="Ch 1",
        text="Skipped chunk text 2",
        char_count=20,
        word_count=4,
    )
    store.add_chunk(chk1)
    store.add_chunk(chk2)

    c = Concept(name="Key Idea", category="Concept", summary="Summary", weight=1)
    store.add_concept(c)
    # Link only chk1 to the idea
    store.add_idea_support_link(
        concept_name="Key Idea",
        chunk=chk1,
        quote="Parent chunk text 1",
        brief_description="Brief",
        detailed_explanation="Detail",
    )

    parent_ids = get_parent_chunk_ids(store)
    assert "chk_parent_1" in parent_ids
    assert "chk_skipped_2" not in parent_ids


def test_compare_graph_spaces_skipping_gate_pass_and_fail():
    """Verify Practical Decision Checklist skipping_generalization gate logic."""
    dim = 16
    np.random.seed(99)
    u = np.random.randn(20, dim).astype(np.float32)
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    # 5 parent chunks, 15 skipped chunks
    parent_mask = np.zeros(20, dtype=bool)
    parent_mask[:5] = True

    # High coverage candidate (covers all 20 chunks)
    g_pass = u[:5].copy()
    # Ensure remaining 15 chunks are close to g_pass
    for i in range(5, 20):
        u[i] = g_pass[i % 5] + np.random.normal(0, 0.02, dim).astype(np.float32)
        u[i] /= np.linalg.norm(u[i])

    rep_pass = compare_graph_spaces(
        universe_embeddings=u,
        baseline_embeddings=u,  # 100% extraction baseline
        candidate_embeddings=g_pass,  # 25% extraction candidate with high generalization
        tau=0.75,
        parent_mask_b=parent_mask,
    )
    assert "skipping_generalization" in rep_pass.gate_checklist
    assert rep_pass.gate_checklist["skipping_generalization"]["passed"] is True
    assert rep_pass.candidate_metrics.parent_chunk_ratio_pct == 25.0
    assert rep_pass.candidate_metrics.skipped_retention_pct >= 90.0
    assert rep_pass.candidate_metrics.generalization_surplus_pct > 0.0

    # Low retention candidate (ideas completely fail to retain skipped chunks)
    # Orthogonal vectors for skipped chunks
    u_isolated = np.zeros((20, 20), dtype=np.float32)
    np.fill_diagonal(u_isolated, 1.0)
    g_fail = u_isolated[:5]  # covers only 5 parent chunks, misses 15 skipped chunks

    rep_fail = compare_graph_spaces(
        universe_embeddings=u_isolated,
        baseline_embeddings=u_isolated,
        candidate_embeddings=g_fail,
        tau=0.75,
        parent_mask_b=parent_mask,
    )
    assert "skipping_generalization" in rep_fail.gate_checklist
    assert rep_fail.gate_checklist["skipping_generalization"]["passed"] is False
    assert rep_fail.decision_pass is False
    assert "Skipping Generalization & Retention" in rep_fail.pass_reason
