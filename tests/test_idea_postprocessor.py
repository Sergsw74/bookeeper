"""
Unit tests for Two-Pass Idea Aggregation and Synthesis (IdeaPostProcessor).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.config import DualPassConfig, Settings
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.extractor import Concept
from bookeeper.processing.idea_postprocessor import (
    ConsolidatedIdeaOutput,
    IdeaCluster,
    IdeaObservation,
    IdeaPostProcessor,
    IdeaPostProcessorReport,
    normalize_idea_name,
    prune_near_duplicate_quotes,
)
from bookeeper.processing.ollama_pool import OllamaPool, OllamaServerNode


# ==============================================================================
# 1. Normalization & Quote Pruning Tests
# ==============================================================================


def test_normalize_idea_name():
    """Verify name normalization strips determiners, punctuation, and folds case."""
    assert normalize_idea_name("The Event Sourcing") == "event sourcing"
    assert normalize_idea_name("A Distributed Lock!") == "distributed lock"
    assert normalize_idea_name("An LSM-Tree...") == "lsm-tree"
    assert normalize_idea_name("  Write-Ahead Logging   ") == "write-ahead logging"
    assert normalize_idea_name("Event Sourcing (ES)") == "event sourcing es"
    assert normalize_idea_name("") == ""


def test_prune_near_duplicate_quotes():
    """Verify near-duplicate quotes are pruned and top quotes retained up to max."""
    quotes = [
        "Event sourcing ensures that all changes to state are stored as a sequence of events.",
        "Event sourcing ensures that all changes to state are stored as a sequence of events in order.",  # Near-duplicate
        "Write-ahead logging is essential for durability and crash recovery.",
        "Shorter quote here that should be kept if unique enough and long.",
        "A" * 10,  # Under 15 chars, filtered
    ]
    retained = prune_near_duplicate_quotes(quotes, max_quotes=3)
    assert len(retained) <= 3
    # Near duplicate should be pruned
    assert not (quotes[0] in retained and quotes[1] in retained)


# ==============================================================================
# 2. Graph Observation Collection & Linearization
# ==============================================================================


@pytest.fixture
def mock_graph_store():
    store = ConceptGraphStore()
    store.add_book(1, title="Distributed Systems", author="Martin K.")

    # Chunk 1
    chk1 = HierarchicalChunk(
        chunk_id="chk_1",
        chunk_idx=1,
        book_id=1,
        book_title="Distributed Systems",
        chapter_idx=1,
        chapter_title="Chapter 1: Storage",
        section_title="Logging",
        breadcrumb="Distributed Systems > Chapter 1 > Logging",
        text="Event sourcing stores all state modifications as an immutable log of events.",
        char_count=80,
        word_count=13,
    )
    store.add_chunk(chk1)

    # Chunk 2
    chk2 = HierarchicalChunk(
        chunk_id="chk_2",
        chunk_idx=2,
        book_id=1,
        book_title="Distributed Systems",
        chapter_idx=1,
        chapter_title="Chapter 1: Storage",
        section_title="Recovery",
        breadcrumb="Distributed Systems > Chapter 1 > Recovery",
        text="The Event Sourcing mechanism allows full historical state reconstruction upon restart.",
        char_count=90,
        word_count=13,
    )
    store.add_chunk(chk2)

    # Concept 1 (Chunk 1)
    store.add_idea_support_link(
        concept_name="Event Sourcing",
        chunk=chk1,
        quote="Event sourcing stores all state modifications as an immutable log of events.",
        brief_description="Storing state changes as events.",
        detailed_explanation="Immutable append-only event log pattern.",
    )

    # Concept 2 with determiner "The Event Sourcing" on Chunk 2 (same underlying concept)
    store.add_idea_support_link(
        concept_name="The Event Sourcing",
        chunk=chk2,
        quote="The Event Sourcing mechanism allows full historical state reconstruction upon restart.",
        brief_description="Reconstructing state from historical events.",
        detailed_explanation="Enables deterministic replay and crash recovery.",
    )

    # Single-occurrence concept (Chunk 1 only)
    store.add_idea_support_link(
        concept_name="Write-Ahead Logging",
        chunk=chk1,
        quote="WAL ensures atomicity before in-memory writes.",
        brief_description="Writing logs before committing memory.",
        detailed_explanation="Durability pattern for databases.",
    )

    return store


def test_collect_observations_from_graph(mock_graph_store):
    """Verify observations are grouped into clusters matching normalized names."""
    postprocessor = IdeaPostProcessor()
    clusters = postprocessor.collect_observations_from_graph(mock_graph_store)

    assert "event sourcing" in clusters
    cluster = clusters["event sourcing"]

    # Both "Event Sourcing" and "The Event Sourcing" merged into the same cluster
    assert len(cluster.aliases) == 2
    assert "Event Sourcing" in cluster.aliases
    assert "The Event Sourcing" in cluster.aliases

    # 2 observations gathered from 2 distinct chunks
    assert len(cluster.observations) == 2
    assert cluster.chunk_ids == {"chk_1", "chk_2"}

    # Linearization check
    linear = cluster.get_linearized_observations()
    assert linear[0].chunk_id == "chk_1"
    assert linear[1].chunk_id == "chk_2"

    # Single occurrence cluster
    assert "write-ahead logging" in clusters
    wal_cluster = clusters["write-ahead logging"]
    assert wal_cluster.total_occurrences == 1


# ==============================================================================
# 3. Semantic Similarity Clustering Tests
# ==============================================================================


class MockEmbeddings:
    """Mock embeddings returning deterministic vectors based on input words."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for t in texts:
            t_low = t.lower()
            if "event" in t_low or "sourcing" in t_low:
                # Vector close to [1.0, 0.0]
                results.append([0.98, 0.05])
            elif "audit trail" in t_low:
                # High cosine similarity to event sourcing (>= 0.92)
                results.append([0.97, 0.06])
            else:
                # Orthogonal vector [0.0, 1.0]
                results.append([0.05, 0.98])
        return results


def test_cluster_by_semantic_similarity_merges_above_threshold():
    """Verify clusters with cosine similarity >= 0.92 are merged."""
    c1 = IdeaCluster(
        canonical_name="Event Sourcing",
        normalized_name="event sourcing",
        observations=[IdeaObservation(concept_name="Event Sourcing", chunk_id="c1", local_brief="Storing state changes as events.")],
        chunk_ids={"c1"},
    )
    c2 = IdeaCluster(
        canonical_name="Audit Trail Events",
        normalized_name="audit trail events",
        observations=[IdeaObservation(concept_name="Audit Trail Events", chunk_id="c2", local_brief="Audit trail of historical events.")],
        chunk_ids={"c2"},
    )
    c3 = IdeaCluster(
        canonical_name="Consensus Protocol",
        normalized_name="consensus protocol",
        observations=[IdeaObservation(concept_name="Consensus Protocol", chunk_id="c3", local_brief="Raft leader election protocol.")],
        chunk_ids={"c3"},
    )

    pp = IdeaPostProcessor(
        embeddings=MockEmbeddings(),
        similarity_threshold=0.92,
    )
    merged = pp.cluster_by_semantic_similarity([c1, c2, c3])

    # c1 and c2 should be merged together; c3 should remain distinct
    assert len(merged) == 2
    merged_c1 = next(c for c in merged if "Event Sourcing" in c.canonical_name or "Audit Trail" in c.canonical_name)
    assert "Audit Trail Events" in merged_c1.aliases or "Event Sourcing" in merged_c1.aliases
    assert merged_c1.chunk_ids == {"c1", "c2"}


# ==============================================================================
# 4. Mention Count Branching & Synthesis
# ==============================================================================


def test_dual_pass_branching_and_graph_update(mock_graph_store):
    """
    Verify:
    - Single-occurrence idea (Write-Ahead Logging, N=1) bypasses LLM
    - Multi-occurrence idea (Event Sourcing, N=2) triggers LLM synthesis
    - Graph canonical node overwritten with cohesive description
    - Old alias node removed
    - Provenance edges (:MENTIONED_IN) and relational edges (:RELATES_TO) emitted
    """
    mock_pool = MagicMock()
    mock_pool.nodes = [OllamaServerNode(url="http://localhost:11434", priority=1)]

    synthesized_output = ConsolidatedIdeaOutput(
        brief_description="Unified architecture storing all state modifications as an append-only log of domain events.",
        detailed_explanation="Combines immutable write-ahead event streams with deterministic state reconstruction on restart.",
        related_entities=["CQRS", "Event Store"],
        category="Architectural Pattern",
        weight=8,
    )

    mock_pool.execute_with_failover.return_value = synthesized_output

    postprocessor = IdeaPostProcessor(
        pool=mock_pool,
        embeddings=None,  # skip vector embedding step in unit test
        similarity_threshold=0.92,
        min_occurrences=2,
    )

    report = postprocessor.process_graph(mock_graph_store)

    assert report.total_clusters == 2
    assert report.multi_occurrence_clusters_count == 1
    assert report.single_occurrence_clusters_count == 1
    assert report.synthesized_count == 1
    assert "Event Sourcing" in report.synthesized_ideas
    assert "Write-Ahead Logging" in report.bypassed_ideas

    # 1. LLM was invoked exactly once (only for Event Sourcing, not for Write-Ahead Logging)
    assert mock_pool.execute_with_failover.call_count == 1

    # 2. Canonical node in graph was updated
    canonical_nid = "concept:Event Sourcing"
    assert mock_graph_store.graph.has_node(canonical_nid)
    node_data = mock_graph_store.graph.nodes[canonical_nid]
    assert node_data["brief_description"] == synthesized_output.brief_description
    assert node_data["detailed_explanation"] == synthesized_output.detailed_explanation
    assert node_data["weight"] == 8
    assert node_data["occurrences"] >= 2

    # 3. Old alias node ("The Event Sourcing") was removed
    assert not mock_graph_store.graph.has_node("concept:The Event Sourcing")

    # 4. Provenance edges (:MENTIONED_IN) created to both source chunks
    assert mock_graph_store.graph.has_edge(canonical_nid, "chunk:chk_1")
    assert mock_graph_store.graph.has_edge(canonical_nid, "chunk:chk_2")

    # 5. Relational edges (:RELATES_TO) emitted for synthesized related entities
    assert mock_graph_store.graph.has_node("concept:CQRS")
    assert mock_graph_store.graph.has_edge(canonical_nid, "concept:CQRS")
    assert mock_graph_store.graph[canonical_nid]["concept:CQRS"]["relation"] == "RELATES_TO"


# ==============================================================================
# 5. Configuration & Settings Parsing Tests
# ==============================================================================


def test_dual_pass_config_yaml_loading(tmp_path: Path):
    """Verify dual_pass configuration loads correctly from YAML."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
llm_model: qwen2.5:7b-instruct
dual_pass:
  enabled: true
  model: llama3.1:8b
  similarity_threshold: 0.94
  max_quotes: 4
  min_occurrences: 2
  temperature: 0.2
"""
    )
    settings = Settings.from_yaml(cfg_file)
    assert settings.dual_pass.enabled is True
    assert settings.dual_pass.model == "llama3.1:8b"
    assert settings.dual_pass.similarity_threshold == 0.94
    assert settings.dual_pass.max_quotes == 4
    assert settings.dual_pass.min_occurrences == 2
    assert settings.dual_pass.temperature == 0.2
    assert settings.resolved_dual_pass_model == "llama3.1:8b"


# ==============================================================================
# 6. CLI Command Test
# ==============================================================================


def test_postprocess_ideas_cli_command(tmp_path: Path, mock_graph_store):
    """Test bookeeper postprocess-ideas CLI command against a stored knowledge graph."""
    runner = CliRunner()
    graph_file = tmp_path / "knowledge_graph.json"
    mock_graph_store.save(graph_file)

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"""
output_dir: {tmp_path}
dual_pass:
  enabled: true
  similarity_threshold: 0.92
  min_occurrences: 2
"""
    )

    with patch.object(IdeaPostProcessor, "synthesize_cluster") as mock_synth:
        mock_synth.return_value = ConsolidatedIdeaOutput(
            brief_description="Consolidated brief definition.",
            detailed_explanation="Consolidated detailed explanation.",
            related_entities=["Related Concept"],
            category="Pattern",
            weight=9,
        )

        res = runner.invoke(
            app,
            [
                "postprocess-ideas",
                "--graph-file",
                str(graph_file),
                "--config",
                str(cfg_file),
            ],
        )

        assert res.exit_code == 0
        assert "Two-Pass Idea Aggregation & Synthesis" in res.stdout
        assert "Dual-Pass Idea Aggregation & Synthesis Complete" in res.stdout

        # Verify graph file was updated
        updated_store = ConceptGraphStore()
        updated_store.load(graph_file)
        assert updated_store.graph.nodes["concept:Event Sourcing"]["brief_description"] == "Consolidated brief definition."


def test_process_graph_header_and_periodic_stats():
    """Verify that process_graph displays the model & filtered count header, and prints stats every 10 ideas."""
    from rich.console import Console

    store = ConceptGraphStore()
    store.add_book(1, title="Test Architecture", author="Architect")

    # Add 12 concepts, each on 2 chunks (multi-occurrence)
    for i in range(1, 13):
        chk_a = HierarchicalChunk(
            chunk_id=f"c_{i}_a",
            chunk_idx=i * 2 - 1,
            book_id=1,
            book_title="Test Architecture",
            chapter_idx=1,
            chapter_title="Chapter 1",
            section_title=f"Sec {i}",
            breadcrumb=f"Book > Ch1 > Sec {i}",
            text=f"Text for concept {i} in chunk A.",
            char_count=40,
            word_count=8,
        )
        chk_b = HierarchicalChunk(
            chunk_id=f"c_{i}_b",
            chunk_idx=i * 2,
            book_id=1,
            book_title="Test Architecture",
            chapter_idx=1,
            chapter_title="Chapter 1",
            section_title=f"Sec {i}",
            breadcrumb=f"Book > Ch1 > Sec {i}",
            text=f"Text for concept {i} in chunk B.",
            char_count=40,
            word_count=8,
        )
        store.add_chunk(chk_a)
        store.add_chunk(chk_b)
        store.add_idea_support_link(
            concept_name=f"Pattern {i}",
            chunk=chk_a,
            quote=f"Quote for pattern {i} in A",
            brief_description=f"Brief {i}",
        )
        store.add_idea_support_link(
            concept_name=f"Pattern {i}",
            chunk=chk_b,
            quote=f"Quote for pattern {i} in B",
            brief_description=f"Brief {i}",
        )

    # Add 3 single-occurrence concepts
    for j in range(1, 4):
        chk_s = HierarchicalChunk(
            chunk_id=f"s_{j}",
            chunk_idx=30 + j,
            book_id=1,
            book_title="Test Architecture",
            chapter_idx=1,
            chapter_title="Chapter 1",
            section_title=f"Single {j}",
            breadcrumb="Book > Ch1 > Single",
            text=f"Single occurrence text {j}.",
            char_count=30,
            word_count=6,
        )
        store.add_chunk(chk_s)
        store.add_idea_support_link(
            concept_name=f"Single Idea {j}",
            chunk=chk_s,
            quote=f"Single quote {j}",
            brief_description=f"Single brief {j}",
        )

    mock_pool = MagicMock()
    mock_pool.nodes = [OllamaServerNode(url="http://localhost:11434", priority=1)]
    mock_pool.execute_with_failover.return_value = ConsolidatedIdeaOutput(
        brief_description="Synthesized description.",
        detailed_explanation="Synthesized detailed explanation.",
        related_entities=[],
        category="Pattern",
        weight=7,
    )

    test_console = Console(record=True, width=120)
    pp = IdeaPostProcessor(
        pool=mock_pool,
        model_name="custom-synth:7b",
        similarity_threshold=0.92,
        min_occurrences=2,
    )

    progress_events = []
    def _track(c, tot, name, stats):
        progress_events.append((c, tot, name, stats))

    report = pp.process_graph(store, console=test_console, on_progress=_track)
    output_text = test_console.export_text()

    # 1. Header contains consolidation model and filtering summary
    assert "Consolidation Model:       custom-synth:7b" in output_text or "custom-synth:7b" in output_text
    assert "12" in output_text  # 12 multi-occurrence ideas filtered in
    assert "3" in output_text   # 3 single-occurrence ideas bypassed

    # 2. Periodic statistical line every 10 ideas
    # Total clusters = 15 (12 multi + 3 single) -> stats printed at 10/15 and 15/15
    assert "10/15 ideas" in output_text
    assert "15/15 ideas" in output_text
    assert "[Idea Postprocessor]" in output_text

    # 3. Report numbers match
    assert report.total_clusters == 15
    assert report.multi_occurrence_clusters_count == 12
    assert report.single_occurrence_clusters_count == 3
    assert report.synthesized_count == 12
    assert len(progress_events) == 15


def test_process_graph_loading_progress():
    """Verify that process_graph outputs progress during concept scanning, embeddings, and clustering."""
    store = ConceptGraphStore()
    chk = HierarchicalChunk(
        chunk_id="c1",
        chunk_idx=1,
        book_id=1,
        book_title="Book",
        chapter_idx=1,
        chapter_title="Ch1",
        section_title="Sec1",
        breadcrumb="Book > Ch1",
        text="Sample text",
        char_count=20,
        word_count=4,
    )
    store.add_chunk(chk)
    store.add_idea_support_link(
        concept_name="CQRS Architecture",
        chunk=chk,
        quote="Quote CQRS",
        brief_description="Command query responsibility segregation",
    )
    store.add_idea_support_link(
        concept_name="Event Sourcing",
        chunk=chk,
        quote="Quote Event Sourcing",
        brief_description="Storing domain events",
    )

    test_console = Console(record=True, width=120)
    pp = IdeaPostProcessor(
        pool=None,
        embeddings=MockEmbeddings(),
        similarity_threshold=0.92,
        min_occurrences=1,
    )

    report = pp.process_graph(store, console=test_console)
    output_text = test_console.export_text()

    # Verify loading stage outputs
    assert "Collected 2 normalized idea clusters" in output_text
    assert "Generated semantic embeddings for 2 clusters" in output_text
    assert "Cosine similarity clustering complete" in output_text
    assert report.total_clusters == 2

