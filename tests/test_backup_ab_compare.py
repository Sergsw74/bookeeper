"""
Tests for A/B Testing versus archived backups and comparative knowledge base verification.
"""

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from bookeeper.backup import (
    compare_knowledge_graphs,
    open_backup_target,
    resolve_backup_target,
    ResolvedBackupTarget,
)
from bookeeper.cli import app
from bookeeper.processing.cross_checker import CrossCheckReport, CrossCheckSummary
from bookeeper.graph.store import ConceptGraphStore
from networkx.readwrite import json_graph

runner = CliRunner(env={"COLUMNS": "200"})


@pytest.fixture
def sample_kg_payloads(tmp_path: Path):
    """Generate two sample Knowledge Graph payloads representing Baseline (A) and Candidate (B)."""
    kg_a = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "book:1", "type": "Book", "book_id": 1, "title": "Distributed Systems A"},
            {"id": "sec:1", "type": "Section", "title": "Consensus Basics"},
            {"id": "chunk:1", "type": "Chunk", "text": "Paxos guarantees safety under asynchronous network conditions."},
            {"id": "chunk:2", "type": "Chunk", "text": "Two-phase commits allow atomic multi-partition updates."},
            {"id": "concept:Paxos", "type": "Concept", "name": "Paxos", "description": "Single-decree consensus"},
            {"id": "concept:Raft", "type": "Concept", "name": "Raft", "description": "Understandable consensus algorithm"},
            {"id": "concept:2PC", "type": "Concept", "name": "Two-Phase Commit", "description": "Atomic commitment protocol"},
        ],
        "edges": [
            {"source": "book:1", "target": "sec:1", "type": "HAS_SECTION"},
            {"source": "sec:1", "target": "chunk:1", "type": "HAS_CHUNK"},
            {"source": "sec:1", "target": "chunk:2", "type": "HAS_CHUNK"},
            {"source": "concept:Paxos", "target": "chunk:1", "type": "SUPPORTED_BY"},
            {"source": "concept:2PC", "target": "chunk:2", "type": "SUPPORTED_BY"},
        ],
    }

    kg_b = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "book:1", "type": "Book", "book_id": 1, "title": "Distributed Systems A"},
            {"id": "book:2", "type": "Book", "book_id": 2, "title": "Advanced Replication B"},
            {"id": "sec:1", "type": "Section", "title": "Consensus Basics"},
            {"id": "sec:2", "type": "Section", "title": "Byzantine Faults"},
            {"id": "chunk:1", "type": "Chunk", "text": "Paxos guarantees safety under asynchronous network conditions with quorum intersection."},
            {"id": "chunk:2", "type": "Chunk", "text": "Two-phase commits allow atomic multi-partition updates."},
            {"id": "chunk:3", "type": "Chunk", "text": "PBFT tolerates Byzantine failures up to f = (n-1)/3."},
            {"id": "concept:Paxos", "type": "Concept", "name": "Paxos", "description": "Single-decree consensus"},
            {"id": "concept:Raft", "type": "Concept", "name": "Raft", "description": "Understandable consensus algorithm"},
            {"id": "concept:PBFT", "type": "Concept", "name": "PBFT", "description": "Practical Byzantine Fault Tolerance"},
            {"id": "concept:Gossip", "type": "Concept", "name": "Gossip Protocol", "description": "Epidemic information dissemination"},
        ],
        "edges": [
            {"source": "book:1", "target": "sec:1", "type": "HAS_SECTION"},
            {"source": "book:2", "target": "sec:2", "type": "HAS_SECTION"},
            {"source": "sec:1", "target": "chunk:1", "type": "HAS_CHUNK"},
            {"source": "sec:1", "target": "chunk:2", "type": "HAS_CHUNK"},
            {"source": "sec:2", "target": "chunk:3", "type": "HAS_CHUNK"},
            {"source": "concept:Paxos", "target": "chunk:1", "type": "SUPPORTED_BY"},
            {"source": "concept:2PC", "target": "chunk:2", "type": "SUPPORTED_BY"},
            {"source": "concept:PBFT", "target": "chunk:3", "type": "SUPPORTED_BY"},
        ],
    }
    return kg_a, kg_b


def _create_backup_zip(dest: Path, kg_payload: dict, chunks: dict = None) -> Path:
    """Helper creating a standard Knowledge Backup ZIP file with metadata."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w") as zf:
        zf.writestr("backup_metadata.json", json.dumps({"backup_type": "knowledge", "version": 1}))
        zf.writestr("knowledge_graph.json", json.dumps(kg_payload, indent=2))
        zf.writestr(".book_states/book_1_state.json", json.dumps({"book_id": "1", "status": "fully_indexed"}))
        if chunks:
            for c_name, c_data in chunks.items():
                zf.writestr(f"chunks/{c_name}", json.dumps(c_data))
        else:
            zf.writestr("chunks/book_1_chunks.json", json.dumps([{"chunk_id": "chunk:1", "text": "Paxos..."}]))
    return dest


def test_resolve_backup_target_from_zip(tmp_path: Path, sample_kg_payloads):
    """Verify resolve_backup_target unpacks archive into TemporaryDirectory and purges on cleanup."""
    kg_a, _ = sample_kg_payloads
    zip_path = tmp_path / "backups" / "knowledge_backup_20261020.zip"
    _create_backup_zip(zip_path, kg_a)

    target = resolve_backup_target(zip_path)
    assert target.is_temp is True
    assert target.root_dir.is_dir()
    assert target.graph_file.is_file()
    assert target.label == "knowledge_backup_20261020"
    assert target.chunks_dir is not None and target.chunks_dir.is_dir()
    assert target.states_dir is not None and target.states_dir.is_dir()

    # Verify graph can be loaded from restored target
    store = ConceptGraphStore()
    store.load(target.graph_file)
    assert len(store.graph.nodes()) > 0

    # Cleanup temporary folder
    temp_root = target.root_dir
    target.cleanup()
    assert not temp_root.exists()


def test_open_backup_target_context_manager(tmp_path: Path, sample_kg_payloads):
    """Verify open_backup_target context manager safely recovers and purges temp directory."""
    kg_a, _ = sample_kg_payloads
    zip_path = tmp_path / "knowledge_backup_test.zip"
    _create_backup_zip(zip_path, kg_a)

    temp_path = None
    with open_backup_target(zip_path) as bt:
        assert bt.is_temp is True
        temp_path = bt.root_dir
        assert temp_path.is_dir()
        assert bt.graph_file.is_file()

    # Upon exit, temp folder must be deleted
    assert not temp_path.exists()


def test_resolve_backup_target_from_direct_dir_and_file(tmp_path: Path, sample_kg_payloads):
    """Verify resolve_backup_target handles uncompressed folders and direct files without temp folders."""
    kg_a, _ = sample_kg_payloads
    out_dir = tmp_path / "live_output"
    out_dir.mkdir(parents=True)
    kg_file = out_dir / "knowledge_graph.json"
    kg_file.write_text(json.dumps(kg_a))

    # Directory input
    bt_dir = resolve_backup_target(out_dir)
    assert bt_dir.is_temp is False
    assert bt_dir.graph_file == kg_file
    bt_dir.cleanup()
    assert out_dir.is_dir()  # Permanent directory untouched

    # Direct file input
    bt_file = resolve_backup_target(kg_file)
    assert bt_file.is_temp is False
    assert bt_file.graph_file == kg_file
    bt_file.cleanup()
    assert kg_file.is_file()


def test_compare_knowledge_graphs_metrics(sample_kg_payloads):
    """Verify compare_knowledge_graphs calculates accurate topological and idea overlap metrics."""
    kg_a, kg_b = sample_kg_payloads

    store_a = ConceptGraphStore()
    store_a.graph = json_graph.node_link_graph(kg_a, directed=True, multigraph=False)

    store_b = ConceptGraphStore()
    store_b.graph = json_graph.node_link_graph(kg_b, directed=True, multigraph=False)

    metrics = compare_knowledge_graphs(store_a, store_b, label_a="Run A", label_b="Run B")

    # Books: A has 1, B has 2
    assert metrics["books"]["a"] == 1
    assert metrics["books"]["b"] == 2
    assert metrics["books"]["delta"] == 1

    # Sections: A has 1, B has 2
    assert metrics["sections"]["delta"] == 1

    # Chunks: A has 2, B has 3
    assert metrics["chunks"]["a"] == 2
    assert metrics["chunks"]["b"] == 3
    assert metrics["chunks"]["delta"] == 1

    # Ideas:
    # A: Paxos, Raft, Two-Phase Commit (3 ideas)
    # B: Paxos, Raft, PBFT, Gossip Protocol (4 ideas)
    # Common: Paxos, Raft (2 ideas)
    # Only in A: Two-Phase Commit (1 idea)
    # Only in B: PBFT, Gossip Protocol (2 ideas)
    # Union: 5 ideas -> Jaccard = 2/5 = 0.4000
    idea_st = metrics["ideas"]
    assert idea_st["a"] == 3
    assert idea_st["b"] == 4
    assert idea_st["delta"] == 1
    assert idea_st["common_count"] == 2
    assert idea_st["only_a_count"] == 1
    assert idea_st["only_b_count"] == 2
    assert idea_st["jaccard_similarity"] == 0.4000
    assert "Two-Phase Commit" in idea_st["sample_only_a"]
    assert any("PBFT" in name for name in idea_st["sample_only_b"])


def test_cli_ab_compare_stats_mode(tmp_path: Path, sample_kg_payloads):
    """Verify CLI `bookeeper ab-compare` runs between two backup ZIP archives in stats mode."""
    kg_a, kg_b = sample_kg_payloads

    zip_a = _create_backup_zip(tmp_path / "backup_a.zip", kg_a)
    zip_b = _create_backup_zip(tmp_path / "backup_b.zip", kg_b)
    out_report = tmp_path / "ab_report.json"

    res = runner.invoke(app, [
        "ab-compare",
        str(zip_a),
        str(zip_b),
        "--mode", "stats",
        "--output", str(out_report),
    ])

    assert res.exit_code == 0, f"Error output: {res.stdout}"
    assert "Unpacking backup archive" in res.stdout
    assert "Cleaned up temporal backup folder" in res.stdout
    assert "Knowledge Base Structural Comparison" in res.stdout
    assert "Unique Concepts / Ideas" in res.stdout
    assert "Shared Ideas (A ∩ B)" in res.stdout
    assert "Jaccard" in res.stdout

    # Verify JSON and Markdown reports were written
    assert out_report.is_file()
    data = json.loads(out_report.read_text(encoding="utf-8"))
    assert data["mode"] == "stats"
    assert data["structural_comparison"]["books"]["delta"] == 1
    assert data["structural_comparison"]["ideas"]["common_count"] == 2

    md_report = out_report.with_suffix(".md")
    assert md_report.is_file()
    md_content = md_report.read_text(encoding="utf-8")
    assert "Knowledge Base A/B Comparison Report" in md_content
    assert "Jaccard" in md_content


def test_cli_ab_compare_against_active_output_dir(tmp_path: Path, sample_kg_payloads):
    """Verify CLI `bookeeper ab-compare` compares a backup archive against the default active graph."""
    kg_a, kg_b = sample_kg_payloads

    zip_a = _create_backup_zip(tmp_path / "backup_baseline.zip", kg_a)

    # Candidate B is the active output_dir / knowledge_graph.json
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    kg_b_file = out_dir / "knowledge_graph.json"
    kg_b_file.write_text(json.dumps(kg_b))

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\n")

    res = runner.invoke(app, [
        "ab-compare",
        str(zip_a),
        "--mode", "stats",
        "--config", str(cfg_file),
    ])

    assert res.exit_code == 0, f"Error output: {res.stdout}"
    assert "backup_baseline" in res.stdout
    assert "Total Books" in res.stdout
    assert "Unique Concepts / Ideas" in res.stdout


def test_cli_verify_cross_check_resolves_backup_archive(tmp_path: Path, sample_kg_payloads):
    """Verify `bookeeper verify --compare-graph <backup.zip>` unpacks archive to temp and resolves graph."""
    kg_a, kg_b = sample_kg_payloads

    zip_a = _create_backup_zip(tmp_path / "baseline_run.zip", kg_a)
    zip_b = _create_backup_zip(tmp_path / "candidate_run.zip", kg_b)

    # Mock run_cross_check with typed report to avoid requiring live Ollama LLM in unit test
    summary_obj = CrossCheckSummary(
        total_samples=2,
        total_assertions_count=6,
        total_old_supported=5,
        total_new_supported=6,
        delta_supported=1,
        total_old_not_mentioned=1,
        total_new_not_mentioned=0,
        delta_not_mentioned=-1,
        total_old_against=0,
        total_new_against=0,
        delta_against=0,
        mean_old_recall=0.833,
        mean_new_recall=1.000,
        mean_delta_recall=0.167,
        mean_old_precision=0.95,
        mean_new_precision=0.97,
        mean_old_seam_integrity=0.85,
        mean_new_seam_integrity=0.95,
        mean_old_net_integrity=0.79,
        mean_new_net_integrity=1.00,
        delta_net_integrity=0.21,
        mean_chunk_disproportion=0.95,
        decision_pass=True,
        pass_reason="Recall delta +16.7% and zero contradictions",
    )
    mock_cc_report = CrossCheckReport(
        model_name="llama3.1:8b",
        baseline_branch="baseline_run",
        candidate_branch="candidate_run",
        summary=summary_obj,
        samples=[],
    )

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\n")

    with patch("bookeeper.cli.run_cross_check", return_value=mock_cc_report), \
         patch("bookeeper.processing.verifier.IdeaVerifier.warmup"):
        res = runner.invoke(app, [
            "verify",
            "--graph-file", str(zip_a),
            "--compare-graph", str(zip_b),
            "--mode", "cross-check",
            "--blocks", "2",
            "--config", str(cfg_file),
        ])

        assert res.exit_code == 0, f"Error output: {res.stdout}"
        assert "Cross-Check Verification Summary & Decision Gates" in res.stdout
        assert "PASS" in res.stdout


def test_cli_ab_compare_cross_check_executes_cleanly(tmp_path: Path, sample_kg_payloads):
    """Verify `bookeeper ab-compare --mode cross-check` executes without AttributeError on summary attributes."""
    kg_a, kg_b = sample_kg_payloads

    zip_a = _create_backup_zip(tmp_path / "baseline_run.zip", kg_a)
    zip_b = _create_backup_zip(tmp_path / "candidate_run.zip", kg_b)

    summary_obj = CrossCheckSummary(
        total_samples=2,
        total_assertions_count=6,
        total_old_supported=5,
        total_new_supported=6,
        delta_supported=1,
        total_old_not_mentioned=1,
        total_new_not_mentioned=0,
        delta_not_mentioned=-1,
        total_old_against=0,
        total_new_against=0,
        delta_against=0,
        mean_old_recall=0.833,
        mean_new_recall=1.000,
        mean_delta_recall=0.167,
        mean_old_precision=0.95,
        mean_new_precision=0.97,
        mean_old_seam_integrity=0.85,
        mean_new_seam_integrity=0.95,
        mean_old_net_integrity=0.79,
        mean_new_net_integrity=1.00,
        mean_delta_net_integrity=0.21,
        mean_old_calibrated_fidelity=0.80,
        mean_new_calibrated_fidelity=0.95,
        mean_delta_calibrated_fidelity=0.15,
        mean_chunk_disproportion=0.95,
        decision_pass=True,
        pass_reason="Recall delta +16.7% and zero contradictions",
    )
    mock_cc_report = CrossCheckReport(
        model_name="llama3.1:8b",
        baseline_branch="baseline_run",
        candidate_branch="candidate_run",
        summary=summary_obj,
        samples=[],
    )

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    report_json = out_dir / "custom_report.json"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\n")

    with patch("bookeeper.cli.run_cross_check", return_value=mock_cc_report), \
         patch("bookeeper.processing.verifier.IdeaVerifier.warmup"):
        res = runner.invoke(app, [
            "ab-compare",
            str(zip_a),
            str(zip_b),
            "--mode", "cross-check",
            "--blocks", "2",
            "--output", str(report_json),
            "--config", str(cfg_file),
        ])

        assert res.exit_code == 0, f"Error output: {res.stdout}"
        assert "Cross-Check Verification Summary & Decision Gates" in res.stdout
        assert "Mean Net Semantic Integrity (S_net)" in res.stdout
        assert "Mean Calibrated Fidelity (R_calib)" in res.stdout
        assert report_json.is_file()
        assert report_json.with_suffix(".md").is_file()

