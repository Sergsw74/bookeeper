"""
Unit and integration tests for chunking, Pydantic schemas, entity deduplication,
NetworkX graph insertions, and Obsidian export without requiring live Calibre or Ollama.
"""

from pathlib import Path
import tempfile
import pytest

from bookeeper.calibre.parser import Section
from bookeeper.graph.exporters import GraphMLExporter, ObsidianExporter
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunker
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import BookMetadata, Concept, SectionExtraction


# ==============================================================================
# 1. Chunker Verification
# ==============================================================================


def test_hierarchical_chunker_metadata_preservation():
    """Verify that HierarchicalChunker breaks sections while retaining complete metadata."""
    chunker = HierarchicalChunker(max_chunk_chars=400, min_chunk_chars=50)

    # 3 distinct paragraphs of text
    p1 = (
        "Event Sourcing ensures every state change is recorded as an immutable event in an append-only log. "
        "This architectural pattern guarantees full auditability and temporal replay capabilities."
    )
    p2 = (
        "CQRS segregates read and write workloads into dedicated specialized query models and command processors. "
        "This decouples transactional throughput from read scalability."
    )
    p3 = (
        "Distributed consensus algorithms such as Raft and Paxos provide replicated state machines across clusters. "
        "They maintain consistency despite network partitions."
    )

    full_text = f"{p1}\n\n{p2}\n\n{p3}"
    section = Section(
        title="Chapter 4: Event-Driven Architectures",
        chapter_idx=4,
        text=full_text,
    )

    chunks = chunker.chunk_section(
        section=section,
        book_id=101,
        book_title="Enterprise Patterns",
    )

    assert len(chunks) >= 2
    for idx, chk in enumerate(chunks):
        assert chk.book_id == 101
        assert chk.book_title == "Enterprise Patterns"
        assert chk.section_title == "Chapter 4: Event-Driven Architectures"
        assert chk.chapter_idx == 4
        assert chk.chunk_idx == idx + 1
        assert len(chk.text) >= 50
        assert chk.word_count > 0
        assert "Enterprise Patterns > Chapter 4: Event-Driven Architectures" in chk.breadcrumb


# ==============================================================================
# 2. Pydantic Extraction Schemas Verification
# ==============================================================================


def test_pydantic_extraction_schemas():
    """Verify BookMetadata, Concept, and SectionExtraction schemas."""
    # BookMetadata
    meta = BookMetadata(
        title="Designing Data-Intensive Applications",
        author="Martin Kleppmann",
        summary="A comprehensive guide to data systems architecture, storage engines, and consensus protocols.",
    )
    assert meta.title == "Designing Data-Intensive Applications"
    assert meta.author == "Martin Kleppmann"
    assert "data systems" in meta.summary

    # Concept
    concept = Concept(
        name="Write-Ahead Log",
        category="Architectural Pattern",
        summary="An append-only log on disk where modifications are written before being applied to in-memory tables.",
        related_concepts=["LSM-Tree", "Crash Recovery", "SSTables"],
    )
    assert concept.name == "Write-Ahead Log"
    assert len(concept.related_concepts) == 3

    # SectionExtraction
    extraction = SectionExtraction(concepts=[concept])
    assert len(extraction.concepts) == 1
    assert extraction.concepts[0].name == "Write-Ahead Log"


# ==============================================================================
# 3. Entity Deduplication Verification
# ==============================================================================


def test_entity_deduplicator_offline_merging():
    """Verify EntityDeduplicator normalizes and merges concepts without live embeddings."""
    dedup = EntityDeduplicator(embeddings=None, similarity_threshold=0.85)

    c1 = Concept(
        name="Event Sourcing",
        category="Architectural Pattern",
        summary="Stores state transitions as events.",
        related_concepts=["CQRS"],
    )
    c2 = Concept(
        name="event sourcing",  # Case-insensitive variation
        category="Pattern",
        summary="Stores state changes as a sequence of immutable events in an append-only log.",
        related_concepts=["Append-Only Log", "CQRS"],
    )

    resolved_1 = dedup.resolve_concept(c1)
    resolved_2 = dedup.resolve_concept(c2)

    # Must resolve to the canonical node
    assert resolved_1.name == "Event Sourcing"
    assert resolved_2.name == "Event Sourcing"
    # Related concepts should be merged
    assert "Append-Only Log" in resolved_1.related_concepts
    assert "CQRS" in resolved_1.related_concepts


# ==============================================================================
# 4. NetworkX Graph Store & Insertion Verification
# ==============================================================================


def test_concept_graph_store_hierarchy_and_export():
    """
    Verify complete graph lifecycle:
      (:Book) -[:HAS_SECTION]-> (:Section)
      (:Section) -[:DISCUSSES]-> (:Concept)
      (:Concept) -[:RELATES_TO]-> (:Concept)
    and export into Obsidian Markdown notes with [[wikilinks]].
    """
    store = ConceptGraphStore()

    # 1. Add Book
    b_id = store.add_book(
        book_id=1,
        title="Building Microservices",
        author="Sam Newman",
        summary="A practical guide to microservice architectures and distributed communications.",
    )
    assert b_id == "book:1"

    # 2. Add Section
    sec_id = store.add_section(
        book_id=1,
        chapter_idx=2,
        title="Chapter 2: The Evolutionary Architect",
        text="Sample chapter content...",
    )
    assert sec_id == "section:1:2"
    assert store.graph.has_edge("book:1", "section:1:2")
    assert store.graph["book:1"]["section:1:2"]["relation"] == "HAS_SECTION"

    # 3. Add Concepts
    c_circuit = Concept(
        name="Circuit Breaker",
        category="Architectural Pattern",
        summary="Prevents cascading failures by failing fast when downstream services are degraded.",
        related_concepts=["Bulkhead Pattern", "Timeouts"],
    )
    c_bulkhead = Concept(
        name="Bulkhead Pattern",
        category="Architectural Pattern",
        summary="Isolates resources to ensure a failure in one partition does not exhaust all system resources.",
        related_concepts=["Circuit Breaker"],
    )

    store.add_concept(c_circuit)
    store.add_concept(c_bulkhead)

    # 4. Connect Section -> DISCUSSES -> Concept
    store.add_section_concept_link(
        section_node_id=sec_id,
        concept_name="Circuit Breaker",
        summary="Introduces circuit breaking for RPC calls.",
        quote="Wrap calls in a circuit breaker to fail fast.",
    )
    assert store.graph.has_edge("section:1:2", "concept:Circuit Breaker")
    assert store.graph["section:1:2"]["concept:Circuit Breaker"]["relation"] == "DISCUSSES"

    # 5. Connect Concept -> RELATES_TO -> Concept
    store.add_concept_relation("Circuit Breaker", "Bulkhead Pattern", "RELATES_TO")
    assert store.graph.has_edge("concept:Circuit Breaker", "concept:Bulkhead Pattern")
    assert store.graph["concept:Circuit Breaker"]["concept:Bulkhead Pattern"]["relation"] == "RELATES_TO"

    # 6. Check Stats
    stats = store.stats()
    assert stats["total_nodes"] == 4  # 1 Book, 1 Section, 2 Concepts
    assert stats["total_edges"] == 3  # 1 HAS_SECTION, 1 DISCUSSES, 1 RELATES_TO
    assert stats["node_types"]["Book"] == 1
    assert stats["node_types"]["Section"] == 1
    assert stats["node_types"]["Concept"] == 2
    assert stats["edge_types"]["HAS_SECTION"] == 1
    assert stats["edge_types"]["DISCUSSES"] == 1
    assert stats["edge_types"]["RELATES_TO"] == 1

    # 7. Test JSON Persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "test_graph.json"
        saved_path = store.save(json_file)
        assert saved_path.is_file()

        # Load back
        new_store = ConceptGraphStore()
        new_store.load(saved_path)
        assert new_store.graph.number_of_nodes() == 4
        assert new_store.graph.number_of_edges() == 3

        # 8. Test Obsidian Exporter
        vault_dir = Path(tmpdir) / "vault"
        exporter = ObsidianExporter(output_dir=vault_dir)
        exporter.export(new_store)

        assert (vault_dir / "00_Index.md").is_file()
        assert (vault_dir / "Books" / "Building Microservices.md").is_file()
        assert (vault_dir / "Concepts" / "Circuit Breaker.md").is_file()
        assert (vault_dir / "Concepts" / "Bulkhead Pattern.md").is_file()

        # Check content and wikilinks
        circuit_content = (vault_dir / "Concepts" / "Circuit Breaker.md").read_text(encoding="utf-8")
        assert "[[Bulkhead Pattern]]" in circuit_content
        assert "type: concept" in circuit_content
        assert "Building Microservices > Chapter 2: The Evolutionary Architect" in circuit_content

        # 9. Test GraphML Exporter
        graphml_file = Path(tmpdir) / "graph.graphml"
        gml = GraphMLExporter(graphml_file)
        gml.export(new_store)
        assert graphml_file.is_file()
