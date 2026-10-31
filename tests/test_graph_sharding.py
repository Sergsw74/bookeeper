"""
Unit tests for Spine and Sharded Knowledge Graph architecture.
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from networkx.readwrite import json_graph

from bookeeper.graph.embedding_codec import get_node_embedding
from bookeeper.graph.sharding import GraphMigrator, ShardedGraphStore


@pytest.fixture
def sample_monolithic_graph_file(tmp_path: Path) -> Path:
    """Generate a realistic synthetic knowledge graph with multiple books, chunks, and embeddings."""
    g = nx.DiGraph()
    rng = np.random.default_rng(42)

    # 3 Books, each with 2 sections and 5 chunks
    for b in range(1, 4):
        b_nid = f"book:{b}"
        g.add_node(
            b_nid,
            type="Book",
            book_id=b,
            title=f"Book {b}: Distributed Systems",
            author="Leslie Lamport",
        )

        for s in range(1, 3):
            s_nid = f"section:{b}:{s}"
            g.add_node(
                s_nid,
                type="Section",
                book_id=b,
                section_id=s,
                title=f"Chapter {s}",
            )
            g.add_edge(b_nid, s_nid, type="HAS_SECTION")

            prev_chunk = None
            for c in range(1, 6):
                c_nid = f"chunk:{b}:{s}:{c}"
                # 128-dimensional embedding
                vec = rng.standard_normal(128).astype(np.float32)
                vec /= np.linalg.norm(vec)

                g.add_node(
                    c_nid,
                    type="Chunk",
                    book_id=b,
                    section_id=s,
                    chunk_index=c,
                    text=f"Text content for book {b} section {s} chunk {c}." * 20,
                    embedding=[round(float(x), 6) for x in vec.tolist()],
                )
                g.add_edge(s_nid, c_nid, type="HAS_CHUNK")
                if prev_chunk:
                    g.add_edge(prev_chunk, c_nid, type="NEXT_CHUNK")
                prev_chunk = c_nid

    # 5 Concepts / Ideas
    for i in range(1, 6):
        idea_nid = f"concept:idea_{i}"
        idea_vec = rng.standard_normal(128).astype(np.float32)
        idea_vec /= np.linalg.norm(idea_vec)

        g.add_node(
            idea_nid,
            type="Concept",
            name=f"Idea {i}",
            is_idea=True,
            brief_description=f"Core concept {i}",
            embedding=[round(float(x), 6) for x in idea_vec.tolist()],
        )
        # Link from some chunks
        g.add_edge(f"chunk:1:1:1", idea_nid, type="MENTIONS", quote=f"Quote for {i}")

    # Idea-to-idea relationship
    g.add_edge("concept:idea_1", "concept:idea_2", type="RELATES_TO")
    g.add_edge("concept:idea_2", "concept:idea_3", type="RELATES_TO")

    file_path = tmp_path / "knowledge_graph.json"
    data = json_graph.node_link_data(g)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return file_path


def test_migrate_monolithic_to_sharded(sample_monolithic_graph_file: Path, tmp_path: Path):
    out_dir = tmp_path / "sharded_output"

    # Migrate with 0.05 MB target shard size to force multiple shards
    report = GraphMigrator.migrate_monolithic_graph(
        source_path=sample_monolithic_graph_file,
        output_dir=out_dir,
        target_shard_size_mb=0.02,  # Tiny limit to verify multi-shard binning
        dtype="float16",
        make_backup=True,
    )

    assert report.verified is True
    assert report.spine_nodes > 0
    assert report.total_nodes_migrated == report.original_nodes
    assert report.total_edges_migrated == report.original_edges
    assert report.num_shards >= 2

    # Verify backup exists
    assert (sample_monolithic_graph_file.parent / "knowledge_graph.json.bak").is_file()

    # Verify Spine graph
    spine_file = out_dir / "knowledge_graph_spine.json"
    assert spine_file.is_file()

    # Verify Manifest
    manifest_file = out_dir / "shards_manifest.json"
    assert manifest_file.is_file()

    with open(manifest_file, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    assert len(m_data["shards"]) == report.num_shards
    assert m_data["embedding_dtype"] == "float16"

    # Verify transparent querying via ShardedGraphStore
    store = ShardedGraphStore(out_dir)
    assert store.is_sharded() is True
    store.load_spine()

    # Check spine node retrieval
    assert store.spine_graph.has_node("concept:idea_1")
    idea_emb = get_node_embedding(store.spine_graph.nodes["concept:idea_1"])
    assert idea_emb is not None
    assert idea_emb.shape == (128,)

    # Check chunk retrieval from shard
    chunk = store.get_chunk("chunk:1:1:1")
    assert chunk is not None
    assert chunk["type"] == "Chunk"
    assert chunk["book_id"] == 1
    chunk_emb = get_node_embedding(chunk)
    assert chunk_emb is not None
    assert chunk_emb.shape == (128,)

    # Check book chunks
    b1_chunks = store.get_book_chunks(1)
    assert len(b1_chunks) == 10  # 2 sections * 5 chunks


def test_roundtrip_reconstruction(sample_monolithic_graph_file: Path, tmp_path: Path):
    sharded_dir = tmp_path / "sharded_dir"
    GraphMigrator.migrate_monolithic_graph(
        source_path=sample_monolithic_graph_file,
        output_dir=sharded_dir,
        target_shard_size_mb=10,
        dtype="float16",
    )

    # Reconstruct back to monolithic
    recon_file = tmp_path / "reconstructed_graph.json"
    GraphMigrator.export_monolithic_from_shards(
        base_dir=sharded_dir,
        output_file=recon_file,
        decompress_embeddings=True,
    )

    assert recon_file.is_file()

    with open(sample_monolithic_graph_file, "r") as f:
        orig = json.load(f)
    with open(recon_file, "r") as f:
        recon = json.load(f)

    assert len(orig["nodes"]) == len(recon["nodes"])
    assert len(orig["edges"]) == len(recon["edges"])

    orig_ids = {n["id"] for n in orig["nodes"]}
    recon_ids = {n["id"] for n in recon["nodes"]}
    assert orig_ids == recon_ids


def test_cli_shard_and_inspect_commands(sample_monolithic_graph_file: Path, tmp_path: Path):
    from typer.testing import CliRunner
    from bookeeper.cli import app

    runner = CliRunner(env={"COLUMNS": "200"})
    sharded_dir = tmp_path / "cli_sharded"

    # 1. Test shard-graph CLI command
    res_shard = runner.invoke(
        app,
        [
            "shard-graph",
            "--graph-file", str(sample_monolithic_graph_file),
            "--output-dir", str(sharded_dir),
            "--shard-size-mb", "10",
            "--dtype", "float16",
        ],
    )
    assert res_shard.exit_code == 0, res_shard.stdout
    assert "Knowledge Graph Sharding & Compression Summary" in res_shard.stdout
    assert "100% Verified Parity" in res_shard.stdout
    assert (sharded_dir / "shards_manifest.json").is_file()
    assert (sharded_dir / "knowledge_graph_spine.json").is_file()

    # 2. Test inspect-shards CLI command
    res_inspect = runner.invoke(
        app,
        [
            "inspect-shards",
            "--dir", str(sharded_dir),
        ],
    )
    assert res_inspect.exit_code == 0, res_inspect.stdout
    assert "Sharded Knowledge Graph Manifest Overview" in res_inspect.stdout
    assert "Payload Shards Distribution" in res_inspect.stdout

    # 3. Test unshard-graph CLI command
    unshard_file = tmp_path / "cli_unshared.json"
    res_unshard = runner.invoke(
        app,
        [
            "unshard-graph",
            "--dir", str(sharded_dir),
            "--output", str(unshard_file),
            "--decompress-embeddings",
        ],
    )
    assert res_unshard.exit_code == 0, res_unshard.stdout
    assert "Reconstructed Monolithic Knowledge Graph" in res_unshard.stdout
    assert unshard_file.is_file()


def test_concept_graph_store_sharded_lifecycle(sample_monolithic_graph_file: Path, tmp_path: Path):
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.extractor import Concept

    sharded_dir = tmp_path / "lifecycle_sharded"
    GraphMigrator.migrate_monolithic_graph(
        source_path=sample_monolithic_graph_file,
        output_dir=sharded_dir,
        target_shard_size_mb=10,
        dtype="float16",
    )

    # 1. Load store from sharded directory
    store = ConceptGraphStore()
    store.load(sharded_dir / "knowledge_graph.json")
    assert store.is_sharded is True
    assert store.manifest is not None
    initial_books = len(store.manifest.book_to_shard)

    # 2. Add Book 99 with a section and chunks in sharded mode
    store.add_book(99, title="Quantum Computing", author="David Deutsch")
    sec_id = store.add_section(book_id=99, chapter_idx=1, title="Qubits & Superposition")

    chk_dict = {
        "chunk_id": "99_c1",
        "book_id": 99,
        "chapter_idx": 1,
        "section_title": "Qubits & Superposition",
        "text": "A qubit can exist in a superposition of states.",
        "embedding": [0.1, 0.2, -0.3, 0.4],
    }
    c_nid = store.add_chunk(chk_dict)
    assert c_nid == "chunk:99_c1"

    # Verify chunk was added to active shard
    assert store.active_shard_graph is not None
    assert store.active_shard_graph.has_node("chunk:99_c1")
    chk_attrs = store.active_shard_graph.nodes["chunk:99_c1"]
    assert "embedding_b64" in chk_attrs
    assert chk_attrs["embedding_dtype"] == "float16"

    # Add a concept and link it to the chunk
    concept = Concept(
        name="Quantum Superposition",
        category="Quantum Mechanics",
        brief_description="Ability of quantum state to be in linear combination",
        weight=9,
    )
    store.add_concept(concept)
    store.add_idea_support_link(
        concept_name=concept.name,
        chunk=chk_dict,
        quote="A qubit can exist in a superposition of states.",
    )

    # 3. Save sharded checkpoint
    saved_spine = store.save(sharded_dir)
    assert saved_spine.is_file()
    assert (sharded_dir / "shards_manifest.json").is_file()

    # 4. Verify stats report composite totals
    st = store.stats()
    assert st["is_sharded"] is True
    assert st["node_types"]["Book"] == initial_books + 1

    # 5. Remove Book 99
    rem_stats = store.remove_book(99)
    assert rem_stats["book_removed"] == 1
    assert rem_stats["chunks_removed"] == 1
    assert rem_stats["sections_removed"] == 1

    # 6. Re-load and verify Book 99 is gone
    fresh_store = ConceptGraphStore()
    fresh_store.load(sharded_dir)
    assert fresh_store.is_sharded is True
    assert not fresh_store.graph.has_node("book:99")
    assert not fresh_store.graph.has_node(sec_id)
    assert fresh_store.get_chunk("chunk:99_c1") is None
