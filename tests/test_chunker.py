"""
Unit and integration tests for chunking, Pydantic schemas, entity deduplication,
NetworkX graph insertions, and Obsidian export without requiring live Calibre or Ollama.
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock
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
        assert "type: idea" in circuit_content
        assert "Building Microservices > Chapter 2: The Evolutionary Architect" in circuit_content

        # 9. Test GraphML Exporter
        graphml_file = Path(tmpdir) / "graph.graphml"
        gml = GraphMLExporter(graphml_file)
        gml.export(new_store)
        assert graphml_file.is_file()


def test_chunk_store_persistence(tmp_path: Path):
    """Verify ChunkStore saves, retrieves, and checks existence of book chunks."""
    from bookeeper.processing.chunker import ChunkStore, HierarchicalChunk

    store_dir = tmp_path / "chunks"
    chunk_store = ChunkStore(store_dir)

    assert chunk_store.has_chunks(42) is False
    assert chunk_store.load_chunks(42) is None

    chunks = [
        HierarchicalChunk(
            chunk_id="chk_1",
            book_id=42,
            book_title="Test Book",
            section_title="Chapter 1",
            chapter_idx=1,
            chunk_idx=1,
            text="This is the first chunk of chapter 1.",
        ),
        HierarchicalChunk(
            chunk_id="chk_2",
            book_id=42,
            book_title="Test Book",
            section_title="Chapter 1",
            chapter_idx=1,
            chunk_idx=2,
            text="This is the second chunk of chapter 1.",
        ),
    ]

    saved_file = chunk_store.save_chunks(42, "Test Book", chunks)
    assert saved_file.is_file()
    assert chunk_store.has_chunks(42) is True

    # Load back
    loaded = chunk_store.load_chunks(42)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].chunk_id == "chk_1"
    assert loaded[0].book_title == "Test Book"
    assert loaded[1].text == "This is the second chunk of chapter 1."

    # Stats and listing
    assert chunk_store.list_stored_book_ids() == [42]
    stats = chunk_store.stats()
    assert stats["total_books"] == 1
    assert stats["total_bytes"] > 0

    # Delete
    assert chunk_store.delete_chunks(42) is True
    assert chunk_store.has_chunks(42) is False


def test_hierarchical_smart_rag_and_idea_provenance(tmp_path: Path):
    """Verify smart structural chunking, parent-child linkages, idea explanations, and chunk provenance."""
    # 1. Test smart markdown AST parsing and parent chunk linkage
    raw_markdown = """
# Chapter 5: Distributed Consensus

In distributed systems, achieving agreement across unreliable networks is a fundamental problem.

## Raft Leader Election

Raft decomposes consensus into leader election, log replication, and safety.
A leader is elected when it receives votes from a quorum of servers.
Servers start in follower state, transition to candidate upon election timeout, and become leader if a majority votes yes.

## Log Compaction

Snapshots are the simplest approach to log compaction.
In snapshotting, the entire current system state is written to stable storage.
"""

    section = Section(
        title="Chapter 5: Distributed Consensus",
        chapter_idx=5,
        text=raw_markdown,
    )

    chunker = HierarchicalChunker(max_chunk_chars=350, min_chunk_chars=40)
    chunks = chunker.chunk_section(section, book_id=7, book_title="Distributed Systems")

    assert len(chunks) >= 2
    # Check subtitle detection
    subtitles = {c.subtitle for c in chunks}
    assert "Raft Leader Election" in subtitles
    assert "Log Compaction" in subtitles

    # Check parent chunk linkage and sequential pointers
    for c in chunks:
        assert c.parent_chunk_id is not None
        assert c.parent_text is not None
        assert "Distributed Systems > Chapter 5: Distributed Consensus" in c.breadcrumb

    # Check prev/next links
    assert chunks[0].prev_chunk_id is None
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].prev_chunk_id == chunks[0].chunk_id

    # 2. Test Idea node creation with detailed explanations
    store = ConceptGraphStore()
    b_id = store.add_book(7, "Distributed Systems", "Tanenbaum")
    sec_id = store.add_section(7, 5, "Chapter 5: Distributed Consensus")

    idea = Concept(
        name="Raft Consensus",
        brief_description="A consensus algorithm designed to be understandable and equivalent to Paxos in fault-tolerance.",
        detailed_explanation="Raft separates leader election, log replication, and safety. A single elected leader manages the replicated log, reducing state-space complexity.",
        category="Distributed Systems Algorithm",
        supporting_quote="Raft decomposes consensus into leader election, log replication, and safety.",
        related_concepts=["Paxos", "State Machine Replication"],
    )

    store.add_concept(idea)

    # 3. Test Idea-to-Chunk provenance link
    target_chunk = next(c for c in chunks if c.subtitle == "Raft Leader Election")
    store.add_idea_support_link(
        concept_name=idea.name,
        chunk=target_chunk,
        quote=idea.supporting_quote,
        brief_description=idea.brief_description,
        detailed_explanation=idea.detailed_explanation,
    )

    # Verify graph node properties
    idea_node = store.graph.nodes[f"concept:{idea.name}"]
    assert idea_node["is_idea"] is True
    assert idea_node["tag"] == "idea"
    assert "understandable" in idea_node["brief_description"]
    assert "leader election" in idea_node["detailed_explanation"]

    # Verify edge connectivity
    assert store.graph.has_edge(f"concept:{idea.name}", f"chunk:{target_chunk.chunk_id}")
    edge = store.graph[f"concept:{idea.name}"][f"chunk:{target_chunk.chunk_id}"]
    assert edge["relation"] == "SUPPORTED_BY"
    assert "Raft decomposes consensus" in edge["quote"]
    assert "Raft Leader Election" in edge["subtitle"]
    assert edge["chunk_id"] == target_chunk.chunk_id

    # 4. Verify Obsidian export formatting with detailed explanation and chunk provenance
    vault_dir = tmp_path / "obsidian_vault"
    exporter = ObsidianExporter(vault_dir)
    exporter.export(store)

    idea_file = vault_dir / "Concepts" / "Raft Consensus.md"
    assert idea_file.is_file()
    idea_content = idea_file.read_text(encoding="utf-8")

    assert "type: idea" in idea_content
    assert "## Summary" in idea_content
    assert "## Detailed Explanation" in idea_content
    assert "## Supporting Evidence & Book Locations" in idea_content
    assert f"Chunk {target_chunk.chunk_id}" in idea_content
    assert "Raft Leader Election" in idea_content
    assert "Raft decomposes consensus into leader election" in idea_content


def test_book_literature_note_plane_and_idea_links(tmp_path):
    """Verify Book nodes act as literature notes with research planes and link bidirectionally with Idea nodes."""
    store = ConceptGraphStore()

    # 1. Add Book
    b_id = store.add_book(
        book_id=42,
        title="Designing Data-Intensive Applications",
        author="Martin Kleppmann",
        summary="The definitive guide to the architecture of data systems.",
    )
    assert b_id == "book:42"

    book_node = store.graph.nodes["book:42"]
    assert "literature-note" in book_node["tags"]
    assert "research-source" in book_node["tags"]
    assert "book" in book_node["tags"]

    # 2. Add Idea
    idea = Concept(
        name="LSM-Trees",
        brief_description="Log-Structured Merge-Trees maintain fast writes by indexing append-only segments.",
        detailed_explanation="Writes go to an in-memory Memtable before being flushed to immutable SSTables. Compaction runs in background to merge segments.",
        category="Storage Architecture",
    )
    store.add_concept(idea)

    # 3. Add Book <-> Idea bidirectional link
    store.add_book_idea_link(book_id=42, concept_name="LSM-Trees")

    assert store.graph.has_edge("book:42", "concept:LSM-Trees")
    assert store.graph["book:42"]["concept:LSM-Trees"]["relation"] == "EXPLORES_IDEA"
    assert store.graph.has_edge("concept:LSM-Trees", "book:42")
    assert store.graph["concept:LSM-Trees"]["book:42"]["relation"] == "FEATURED_IN"

    # 4. Export to Obsidian Vault and verify note generation
    vault_dir = tmp_path / "obsidian_vault"
    exporter = ObsidianExporter(vault_dir)
    exporter.export(store)

    # Check Book note
    book_file = vault_dir / "Books" / "Designing Data-Intensive Applications.md"
    assert book_file.is_file()
    book_content = book_file.read_text(encoding="utf-8")

    assert "type: book" in book_content
    assert "literature-note" in book_content
    assert "research-source" in book_content
    assert "## 📖 Overview & Abstract" in book_content
    assert "The definitive guide to the architecture of data systems." in book_content
    assert "## 💡 Core Ideas & Conceptual Knowledge Plane" in book_content
    assert "### Storage Architecture" in book_content
    assert "[[LSM-Trees]]" in book_content
    assert "Log-Structured Merge-Trees maintain fast writes" in book_content
    assert "## 🔬 Research & Reading Notes" in book_content
    assert "### Key Synthesis" in book_content
    assert "### Cross-Book Conceptual Connections" in book_content

    # Check Idea note
    idea_file = vault_dir / "Concepts" / "LSM-Trees.md"
    assert idea_file.is_file()
    idea_content = idea_file.read_text(encoding="utf-8")

    assert "type: idea" in idea_content
    assert "## 📚 Featured in Books" in idea_content
    assert "[[Designing Data-Intensive Applications|Designing Data-Intensive Applications]]" in idea_content
    assert "Martin Kleppmann" in idea_content
    # Ensure EXPLORES_IDEA / FEATURED_IN didn't pollute "Related Ideas"
    assert "## Related Ideas" not in idea_content


def test_concept_weight_lifecycle_and_export(tmp_path):
    """Verify concept weights (0=basic, 3=risk assessment, 7=RAG, 9=SmartRAG), merging, and export."""
    from bookeeper.processing.deduplicator import EntityDeduplicator

    # 1. Test model instantiation and clamping
    c_ident = Concept(name="Identification", category="Security", summary="Basic identity check", weight=0)
    c_risk = Concept(name="Risk Assessment", category="Security", summary="Evaluation of risk factors", weight=3)
    c_rag = Concept(name="RAG", category="AI Architecture", summary="Retrieval Augmented Generation", weight=7)
    c_smartrag = Concept(name="SmartRAG", category="AI Architecture", summary="Adaptive agentic RAG mechanism", weight=9)

    assert c_ident.weight == 0
    assert c_risk.weight == 3
    assert c_rag.weight == 7
    assert c_smartrag.weight == 9

    # Clamping test
    c_clamped_high = Concept(name="Super Concept", category="Test", summary="...", weight=15)
    c_clamped_low = Concept(name="Low Concept", category="Test", summary="...", weight=-5)
    assert c_clamped_high.weight == 10
    assert c_clamped_low.weight == 0

    # 2. Test ConceptGraphStore weight storage and update
    store = ConceptGraphStore()
    store.add_concept(c_ident)
    store.add_concept(c_rag)
    store.add_concept(c_smartrag)

    assert store.graph.nodes["concept:Identification"]["weight"] == 0
    assert store.graph.nodes["concept:RAG"]["weight"] == 7

    # Updating RAG with higher weight updates it; lower weight does not downgrade it
    c_rag_more_specific = Concept(name="RAG", category="AI Architecture", summary="Deeper RAG nuances", weight=8)
    store.add_concept(c_rag_more_specific)
    assert store.graph.nodes["concept:RAG"]["weight"] == 8

    c_rag_generic_mention = Concept(name="RAG", category="AI Architecture", summary="Brief mention", weight=5)
    store.add_concept(c_rag_generic_mention)
    assert store.graph.nodes["concept:RAG"]["weight"] == 8

    # 3. Test Deduplicator merging retains max weight
    dedup = EntityDeduplicator(similarity_threshold=0.85)
    canon = Concept(name="Risk Assessment", category="Security", summary="Base definition", weight=3)
    incoming = Concept(name="Risk Assessment", category="Security", summary="Enhanced definition", weight=6)
    resolved = dedup.resolve_concept(canon)
    resolved_merged = dedup.resolve_concept(incoming)
    assert resolved_merged.weight == 6

    # 4. Verify Obsidian Export includes weight in frontmatter and sorts MOC by weight
    vault_dir = tmp_path / "obsidian_vault"
    exporter = ObsidianExporter(vault_dir)
    exporter.export(store)

    smartrag_file = vault_dir / "Concepts" / "SmartRAG.md"
    assert smartrag_file.is_file()
    content = smartrag_file.read_text(encoding="utf-8")
    assert "weight: 9" in content
    assert "Weight / Significance: 9/10" in content

    # Check 00_Index.md sorts SmartRAG (w:9) before RAG (w:8)
    index_file = vault_dir / "00_Index.md"
    index_content = index_file.read_text(encoding="utf-8")
    smartrag_pos = index_content.find("[[SmartRAG]]")
    rag_pos = index_content.find("[[RAG]]")
    assert smartrag_pos != -1 and rag_pos != -1
    assert smartrag_pos < rag_pos  # SmartRAG (9) appears before RAG (8)

    # 5. Verify stats includes average weight
    stats = store.stats()
    assert "average_concept_weight" in stats
    assert stats["average_concept_weight"] > 0


def test_obsidian_clean_export(tmp_path):
    """Verify clean export deletes stale Obsidian notes before writing new ones."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(parents=True)
    stale_file = vault_dir / "Concepts" / "OldStaleConcept.md"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("obsolete content", encoding="utf-8")

    store = ConceptGraphStore()
    store.add_concept(Concept(name="FreshConcept", category="Idea", summary="Fresh idea"))

    exporter = ObsidianExporter(vault_dir, clean=True)
    exporter.export(store)

    assert not stale_file.exists()
    assert (vault_dir / "Concepts" / "FreshConcept.md").exists()


def test_chunk_book_progress_callback():
    """Verify that chunk_book triggers progress_callback at each section boundary."""
    chunker = HierarchicalChunker(max_chunk_chars=300, min_chunk_chars=50)
    sections = [
        Section(
            title="Chapter 1: Foundations",
            chapter_idx=1,
            text="This is chapter 1 with sufficient content to form a valid chunk. It discusses foundational concepts.",
        ),
        Section(
            title="Chapter 2: Implementations",
            chapter_idx=2,
            text="This is chapter 2 with sufficient content to form another valid chunk. It focuses on implementation.",
        ),
    ]

    calls = []

    def _cb(completed: int, total: int, sec_title: str, num_chunks: int):
        calls.append((completed, total, sec_title, num_chunks))

    chunks = chunker.chunk_book(
        sections=sections,
        book_id=42,
        book_title="Test Progress Book",
        progress_callback=_cb,
    )

    assert len(chunks) == 2
    assert len(calls) == 4  # 2 calls per section (start and completion)
    # First section start and end
    assert calls[0] == (0, 2, "Chapter 1: Foundations", 0)
    assert calls[1][0] == 1
    assert calls[1][1] == 2
    assert calls[1][2] == "Chapter 1: Foundations"
    assert calls[1][3] == 1

    # Second section start and end
    assert calls[2][0] == 1
    assert calls[2][1] == 2
    assert calls[2][2] == "Chapter 2: Implementations"
    assert calls[2][3] == 1
    assert calls[3] == (2, 2, "Chapter 2: Implementations", 2)


def test_hierarchical_chunker_embedding_stats_delegation():
    """Verify that HierarchicalChunker delegates embedding_stats and reset_embedding_stats."""
    mock_emb = MagicMock()
    mock_emb.embedding_stats = {"total_texts": 42, "speed_texts_per_sec": 19.5}

    chunker = HierarchicalChunker(embeddings=mock_emb)
    assert chunker.embedding_stats == {"total_texts": 42, "speed_texts_per_sec": 19.5}

    chunker.reset_embedding_stats()
    mock_emb.reset_stats.assert_called_once()





