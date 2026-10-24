"""
Unit tests for chronological publication date sorting, batching,
first_occurrence_month tracking, and semantic chunk routing.
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from bookeeper.calibre.parser import format_idea_month
from bookeeper.config import Settings
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.idea_postprocessor import (
    IdeaCluster,
    IdeaObservation,
    IdeaPostProcessor,
    ConsolidatedIdeaOutput,
)
from bookeeper.processing.recent_ideas import (
    BatchRoutingResult,
    ChunkPriority,
    IdeaEmbeddingIndex,
    RecentIdeaChunkRouter,
    RoutedChunk,
    normalize_sort_date,
    partition_books_into_batches,
    sort_books_chronologically,
)


def test_format_idea_month():
    """Verify normalization of various publication date formats into YYYY-MM."""
    assert format_idea_month("2024-05-18") == "2024-05"
    assert format_idea_month("2024-05") == "2024-05"
    assert format_idea_month("2024-05-18T14:30:00+00:00") == "2024-05"
    assert format_idea_month("2015") == "2015-01"
    assert format_idea_month("0101-01-01") is None
    assert format_idea_month("") is None
    assert format_idea_month(None) is None


def test_normalize_sort_date():
    """Verify sortable date representation with missing dates sorting last."""
    assert normalize_sort_date("1995-04-12") == "1995-04-12"
    assert normalize_sort_date("2020-08") == "2020-08-01"
    assert normalize_sort_date("2011") == "2011-01-01"
    assert normalize_sort_date(None) == "9999-12-31"
    assert normalize_sort_date("0101-01-01") == "9999-12-31"


def test_sort_books_chronologically():
    """Verify books are sorted from earliest publication date to latest, with undated books last."""
    books = [
        {"id": 1, "title": "Modern ML", "publication_date": "2024-01-15"},
        {"id": 2, "title": "Classical AI", "publication_date": "1998-06-20"},
        {"id": 3, "title": "Undated Book", "publication_date": None},
        {"id": 4, "title": "Deep Learning Renaissance", "publication_date": "2012-10"},
        {"id": 5, "title": "Ancient Logic", "publication_date": "1980"},
    ]
    sorted_books = sort_books_chronologically(books)
    sorted_ids = [b["id"] for b in sorted_books]
    assert sorted_ids == [5, 2, 4, 1, 3]  # 1980, 1998, 2012, 2024, Undated


def test_partition_books_into_batches():
    """Verify batching of books into blocks (e.g. 10 books)."""
    books = [{"id": i, "title": f"Book {i}"} for i in range(25)]
    batches = partition_books_into_batches(books, batch_size=10)
    assert len(batches) == 3
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5
    assert [b["id"] for b in batches[0]] == list(range(10))
    assert [b["id"] for b in batches[1]] == list(range(10, 20))
    assert [b["id"] for b in batches[2]] == list(range(20, 25))


def test_store_idea_month_earlier_date_wins():
    """Verify ConceptGraphStore records first_occurrence_month and earlier date always wins."""
    store = ConceptGraphStore()

    # 1. Book in 2023-05 introduces 'Zero Knowledge Proofs'
    store.add_book(101, title="ZKP Intro", publication_date="2023-05-10")
    store.add_book_idea_link(101, "Zero Knowledge Proofs")

    node = store.graph.nodes["concept:Zero Knowledge Proofs"]
    assert node.get("first_occurrence_month") == "2023-05"
    assert node.get("month") == "2023-05"

    # 2. Later book in 2024-08 also discusses 'Zero Knowledge Proofs' -> 2023-05 remains!
    store.add_book(102, title="Modern ZKP", publication_date="2024-08-01")
    store.add_book_idea_link(102, "Zero Knowledge Proofs")
    assert node.get("first_occurrence_month") == "2023-05"

    # 3. Earlier seminal book in 1985 is discovered and linked -> 1985-01 WINS!
    store.add_book(103, title="Original ZKP Paper", publication_date="1985-02-01")
    store.add_book_idea_link(103, "Zero Knowledge Proofs")
    assert node.get("first_occurrence_month") == "1985-02"
    assert node.get("month") == "1985-02"


def test_store_idea_support_link_tracks_month():
    """Verify add_idea_support_link updates first_occurrence_month correctly."""
    store = ConceptGraphStore()
    chunk = HierarchicalChunk(
        chunk_id="b1_c1",
        book_id=200,
        book_title="Distributed Consensus",
        section_title="Paxos Section",
        chapter_idx=1,
        chunk_idx=1,
        text="Paxos guarantees consensus under partial synchrony.",
    )
    store.add_book(200, title="Distributed Consensus", publication_date="1998-05-15")
    store.add_idea_support_link("Paxos Algorithm", chunk, quote="guarantees consensus")

    node = store.graph.nodes["concept:Paxos Algorithm"]
    assert node.get("first_occurrence_month") == "1998-05"


def test_postprocessor_earlier_date_wins_during_consolidation():
    """Verify that during IdeaPostProcessor clustering/deduplication, the earliest month wins."""
    store = ConceptGraphStore()

    # Create two alias concept nodes with different dates
    store.graph.add_node(
        "concept:Transformer Architecture",
        type="Concept",
        is_idea=True,
        name="Transformer Architecture",
        first_occurrence_month="2020-04",
        month="2020-04",
        category="Deep Learning",
        weight=5,
    )
    store.graph.add_node(
        "concept:Attention Transformers",
        type="Concept",
        is_idea=True,
        name="Attention Transformers",
        first_occurrence_month="2017-06",
        month="2017-06",
        category="Deep Learning",
        weight=5,
    )

    cluster = IdeaCluster(
        canonical_name="Transformer Architecture",
        normalized_name="transformer architecture",
        aliases={"Transformer Architecture", "Attention Transformers"},
        first_occurrence_month="2020-04",
    )
    cluster.add_observation(IdeaObservation(concept_name="Transformer Architecture", month="2020-04"))
    cluster.add_observation(IdeaObservation(concept_name="Attention Transformers", month="2017-06"))

    # The cluster's earliest month should be 2017-06
    assert cluster.get_first_occurrence_month() == "2017-06"

    postprocessor = IdeaPostProcessor()
    out = ConsolidatedIdeaOutput(
        canonical_name="Transformer Architecture",
        brief_description="Attention mechanism based neural network.",
        detailed_explanation="Introduced in Attention Is All You Need.",
        category="Deep Learning",
        weight=5,
    )
    postprocessor.apply_consolidation_to_graph(store, cluster, out)

    canonical_node = store.graph.nodes["concept:Transformer Architecture"]
    # 2017-06 must win over 2020-04!
    assert canonical_node.get("first_occurrence_month") == "2017-06"
    assert canonical_node.get("month") == "2017-06"


def test_chunk_similarity_router():
    """
    Verify ChunkSimilarityRouter routes chunks into:
    - Unassigned (<0.65)
    - High Confidence (>=0.90)
    - Limbo (0.65 <= sim < 0.90)
    """
    # Create mock embeddings with predetermined cosine similarity
    mock_embeddings = MagicMock()

    # Dimension 4 embeddings
    # Idea 1: [1.0, 0.0, 0.0, 0.0] -> "Quantum Cryptography"
    idea_index = IdeaEmbeddingIndex(
        idea_names=["Quantum Cryptography"],
        idea_summaries=["Quantum key distribution and security."],
        idea_months={"Quantum Cryptography": "2021-01"},
        vectors=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

    # Chunks:
    # c1: Cosine sim = 1.0 (>= 0.90) -> HIGH_CONFIDENCE
    # c2: Cosine sim = 0.75 (0.65 <= sim < 0.90) -> LIMBO
    # c3: Cosine sim = 0.20 (< 0.65) -> UNASSIGNED
    vec_c1 = [1.0, 0.0, 0.0, 0.0]
    vec_c2 = [0.75, np.sqrt(1 - 0.75**2), 0.0, 0.0]
    vec_c3 = [0.20, np.sqrt(1 - 0.20**2), 0.0, 0.0]

    mock_embeddings.embed_documents.return_value = [vec_c1, vec_c2, vec_c3]

    chunks = [
        HierarchicalChunk(chunk_id="c1", book_id=1, book_title="B", section_title="S1", chapter_idx=1, chunk_idx=1, text="Quantum key exchange"),
        HierarchicalChunk(chunk_id="c2", book_id=1, book_title="B", section_title="S2", chapter_idx=1, chunk_idx=2, text="Lattice based security"),
        HierarchicalChunk(chunk_id="c3", book_id=1, book_title="B", section_title="S3", chapter_idx=1, chunk_idx=3, text="Agricultural economics"),
    ]

    router = RecentIdeaChunkRouter(
        idea_index=idea_index,
        embeddings=mock_embeddings,
        unassigned_threshold=0.65,
        high_confidence_threshold=0.90,
    )
    result = router.route_chunks(chunks)

    assert result.total_chunks == 3
    assert result.high_confidence_count == 1
    assert result.limbo_count == 1
    assert result.unassigned_count == 1

    assert result.high_confidence[0].chunk.chunk_id == "c1"
    assert result.high_confidence[0].priority == ChunkPriority.HIGH_CONFIDENCE
    assert result.high_confidence[0].matched_idea == "Quantum Cryptography"
    assert result.high_confidence[0].similarity >= 0.95

    assert result.limbo[0].chunk.chunk_id == "c2"
    assert result.limbo[0].priority == ChunkPriority.LIMBO
    assert result.limbo[0].matched_idea == "Quantum Cryptography"
    assert 0.65 <= result.limbo[0].similarity < 0.90

    assert result.unassigned[0].chunk.chunk_id == "c3"
    assert result.unassigned[0].priority == ChunkPriority.UNASSIGNED
    assert result.unassigned[0].similarity < 0.65


def test_chunk_similarity_router_empty_graph():
    """Verify that when the knowledge graph has 0 ideas, all chunks are routed to UNASSIGNED."""
    empty_index = IdeaEmbeddingIndex()
    chunks = [
        HierarchicalChunk(chunk_id="c1", book_id=1, book_title="B", section_title="S", chapter_idx=1, chunk_idx=1, text="Text 1"),
        HierarchicalChunk(chunk_id="c2", book_id=1, book_title="B", section_title="S", chapter_idx=1, chunk_idx=2, text="Text 2"),
    ]
    router = RecentIdeaChunkRouter(idea_index=empty_index, embeddings=None)
    result = router.route_chunks(chunks)

    assert result.unassigned_count == 2
    assert result.high_confidence_count == 0
    assert result.limbo_count == 0
    assert result.unassigned[0].priority == ChunkPriority.UNASSIGNED


def test_build_graph_chronological_batch_integration(tmp_path: Path):
    """
    Integration test verifying build-graph runs chronological sorting,
    records first_occurrence_month on ideas, and assigns high-confidence chunks.
    """
    import sqlite3
    from unittest.mock import patch
    from typer.testing import CliRunner
    from bookeeper.cli import app
    from bookeeper.processing.extractor import SectionExtraction, Concept, KnowledgeExtractor
    from bookeeper.calibre.client import _register_sqlite_functions
    from bookeeper.calibre.parser import Section

    runner = CliRunner()
    lib_dir = tmp_path / "calibre_lib"
    lib_dir.mkdir()
    db_path = lib_dir / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    _register_sqlite_functions(conn)
    conn.executescript("""
        CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL, sort TEXT, author_sort TEXT, pubdate TIMESTAMP, path TEXT NOT NULL DEFAULT 'Author/Book');
        CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE);
        CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER NOT NULL, author INTEGER NOT NULL);
        CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER NOT NULL, format TEXT NOT NULL, name TEXT NOT NULL);
        CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER NOT NULL UNIQUE, text TEXT NOT NULL);
    """)

    # Book 1 published in 2024 (newer)
    conn.execute("INSERT INTO books (id, title, sort, author_sort, pubdate, path) VALUES (1, 'Modern Book', 'Modern Book', 'Author', '2024-05-12', 'A/B1')")
    # Book 2 published in 1999 (earlier -> should be sorted first!)
    conn.execute("INSERT INTO books (id, title, sort, author_sort, pubdate, path) VALUES (2, 'Seminal Book', 'Seminal Book', 'Author', '1999-08-20', 'A/B2')")
    conn.execute("INSERT INTO authors (id, name) VALUES (1, 'Jane Doe')")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (1, 1), (2, 1)")
    conn.execute("INSERT INTO data (book, format, name) VALUES (1, 'EPUB', 'Modern Book'), (2, 'EPUB', 'Seminal Book')")
    conn.commit()
    conn.close()

    (lib_dir / "A" / "B1").mkdir(parents=True)
    (lib_dir / "A" / "B2").mkdir(parents=True)
    (lib_dir / "A" / "B1" / "Modern Book.epub").write_text("dummy", encoding="utf-8")
    (lib_dir / "A" / "B2" / "Seminal Book.epub").write_text("dummy", encoding="utf-8")

    out_dir = tmp_path / "output"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\ncalibre_library_path: {lib_dir}\n", encoding="utf-8")

    mock_text = (
        "Vector space models represent text documents algebraically with high-dimensional term vectors. "
        "Each dimension corresponds to a separate term in the dictionary vocabulary. "
        "Term weights are commonly computed using term frequency and inverse document frequency. "
    )
    mock_concept = Concept(
        name="Vector Space Modeling",
        category="IR",
        summary="Algebraic model for representing text documents.",
        supporting_quote="Vector space models represent text documents algebraically with high-dimensional term vectors.",
    )

    mock_extract_calls = []

    def mock_extract_chunks(chunks, **kwargs):
        mock_extract_calls.append(chunks)
        res = {}
        for c in chunks:
            res[c.chunk_id] = SectionExtraction(concepts=[mock_concept])
        return res

    mock_parse_sections = [
        Section(title="Section 1", chapter_idx=1, text=mock_text)
    ]

    with patch("bookeeper.calibre.parser.BookParser.parse", return_value=mock_parse_sections), \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False), \
         patch.object(KnowledgeExtractor, "extract_chunks", side_effect=mock_extract_chunks), \
         patch.object(KnowledgeExtractor, "from_settings") as mock_from_settings, \
         patch("bookeeper.processing.deduplicator.EntityDeduplicator.from_settings") as mock_dedup_cls:

        mock_ext_instance = MagicMock()
        mock_ext_instance.extract_chunks.side_effect = mock_extract_chunks
        mock_from_settings.return_value = mock_ext_instance

        mock_dedup_instance = MagicMock()
        mock_dedup_instance.resolve_concept.side_effect = lambda c: c
        mock_dedup_cls.return_value = mock_dedup_instance

        res = runner.invoke(
            app,
            [
                "build-graph",
                "--all",
                "--calibre-path", str(lib_dir),
                "--config", str(cfg_file),
                "--skip-warmup",
                "--recent-ideas",
                "--batch-books", "2",
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        assert "Chronological Ingestion" in res.stdout
        assert "Knowledge Graph Build Complete" in res.stdout

        # Verify Knowledge Graph has concept with earliest month 1999-08
        kg_path = out_dir / "knowledge_graph.json"
        assert kg_path.is_file()
        store = ConceptGraphStore()
        store.load(kg_path)

        concept_node = store.graph.nodes.get("concept:Vector Space Modeling")
        assert concept_node is not None
        # 1999-08 should win because Book 2 was published in 1999 and sorted first!
        assert concept_node.get("first_occurrence_month") == "1999-08"
        assert concept_node.get("month") == "1999-08"


def test_build_graph_high_confidence_chunk_skips_llm(tmp_path: Path):
    """
    Verify that in batch 2, a chunk with high similarity to an existing idea (>=0.90)
    is directly assigned to the existing idea in the knowledge graph without calling the LLM.
    """
    import sqlite3
    from unittest.mock import patch
    from typer.testing import CliRunner
    from bookeeper.cli import app
    from bookeeper.processing.extractor import SectionExtraction, Concept, KnowledgeExtractor
    from bookeeper.calibre.client import _register_sqlite_functions
    from bookeeper.calibre.parser import Section
    from bookeeper.graph.store import ConceptGraphStore

    runner = CliRunner()
    lib_dir = tmp_path / "calibre_lib"
    lib_dir.mkdir()
    db_path = lib_dir / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    _register_sqlite_functions(conn)
    conn.executescript("""
        CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL, sort TEXT, author_sort TEXT, pubdate TIMESTAMP, path TEXT NOT NULL DEFAULT 'Author/Book');
        CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE);
        CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER NOT NULL, author INTEGER NOT NULL);
        CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER NOT NULL, format TEXT NOT NULL, name TEXT NOT NULL);
        CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER NOT NULL UNIQUE, text TEXT NOT NULL);
    """)

    conn.execute("INSERT INTO books (id, title, sort, author_sort, pubdate, path) VALUES (1, 'Book Alpha', 'Book Alpha', 'Author', '2020-01-01', 'A/B1')")
    conn.execute("INSERT INTO authors (id, name) VALUES (1, 'Jane Doe')")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (1, 1)")
    conn.execute("INSERT INTO data (book, format, name) VALUES (1, 'EPUB', 'Book Alpha')")
    conn.commit()
    conn.close()

    (lib_dir / "A" / "B1").mkdir(parents=True)
    (lib_dir / "A" / "B1" / "Book Alpha.epub").write_text("dummy", encoding="utf-8")

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\ncalibre_library_path: {lib_dir}\n", encoding="utf-8")

    # Seed the knowledge graph with an existing idea
    existing_store = ConceptGraphStore()
    existing_store.add_book(100, title="Preexisting Book", publication_date="2010-05-01")
    existing_store.add_book_idea_link(100, "Paxos Consensus", publication_date="2010-05-01")
    kg_path = out_dir / "knowledge_graph.json"
    existing_store.save(kg_path)

    mock_text = (
        "Paxos consensus guarantees safety and liveness under asynchronous networks. "
        "The protocol operates across two phases: prepare and accept, ensuring single-leader commit. "
        "Quorum intersection guarantees that at most one value is chosen across rounds. "
    )

    mock_extract_calls = []

    def mock_extract_chunks(chunks, **kwargs):
        mock_extract_calls.append(chunks)
        res = {}
        for c in chunks:
            res[c.chunk_id] = SectionExtraction(concepts=[])
        return res

    mock_parse_sections = [
        Section(title="Section 1", chapter_idx=1, text=mock_text)
    ]

    # Create mock embeddings where the chunk vector is identical to the idea vector (sim = 1.0 >= 0.90)
    mock_embeddings = MagicMock()
    mock_embeddings.embedding_stats = {}
    # Idea and chunk both get vector [1.0, 0.0] -> cosine similarity = 1.0 >= 0.90 (HIGH_CONFIDENCE)
    mock_embeddings.embed_documents.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    mock_embeddings.embed_query.side_effect = lambda text: [1.0, 0.0]

    with patch("bookeeper.calibre.parser.BookParser.parse", return_value=mock_parse_sections), \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False), \
         patch("bookeeper.processing.ollama_pool.FailoverOllamaEmbeddings.from_settings", return_value=mock_embeddings), \
         patch.object(KnowledgeExtractor, "extract_chunks", side_effect=mock_extract_chunks), \
         patch.object(KnowledgeExtractor, "from_settings") as mock_from_settings, \
         patch("bookeeper.processing.deduplicator.EntityDeduplicator.from_settings") as mock_dedup_cls:

        mock_ext_instance = MagicMock()
        mock_ext_instance.extract_chunks.side_effect = mock_extract_chunks
        mock_from_settings.return_value = mock_ext_instance

        mock_dedup_instance = MagicMock()
        mock_dedup_instance.resolve_concept.side_effect = lambda c: c
        mock_dedup_cls.return_value = mock_dedup_instance

        res = runner.invoke(
            app,
            [
                "build-graph",
                "--all",
                "--calibre-path", str(lib_dir),
                "--config", str(cfg_file),
                "--skip-warmup",
                "--recent-ideas",
                "--similarity-high-confidence", "0.90",
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        assert "High-confidence match" in res.stdout
        # Because the chunk had similarity 1.0 >= 0.90, LLM extract_chunks was SKIPPED!
        assert len(mock_extract_calls) == 0

        # Verify the chunk was linked directly to 'Paxos Consensus' in the knowledge graph
        updated_store = ConceptGraphStore()
        updated_store.load(kg_path)
        assert updated_store.graph.has_node("concept:Paxos Consensus")
        # Preexisting date 2010-05 was earlier than 2020-01 -> earlier date remains!
        assert updated_store.graph.nodes["concept:Paxos Consensus"].get("first_occurrence_month") == "2010-05"


def test_get_llm_queue_chunks_skip_limbo():
    """Verify BatchRoutingResult.get_llm_queue_chunks excludes limbo chunks when skip_limbo=True."""
    c_unassigned = HierarchicalChunk(chunk_id="c_u", book_id=1, book_title="B", section_title="S1", chapter_idx=1, chunk_idx=1, text="New topic")
    c_limbo = HierarchicalChunk(chunk_id="c_l", book_id=1, book_title="B", section_title="S2", chapter_idx=1, chunk_idx=2, text="Limbo topic")
    c_high = HierarchicalChunk(chunk_id="c_h", book_id=1, book_title="B", section_title="S3", chapter_idx=1, chunk_idx=3, text="High topic")

    routing = BatchRoutingResult(
        unassigned=[RoutedChunk(chunk=c_unassigned, priority=ChunkPriority.UNASSIGNED, similarity=0.2)],
        limbo=[RoutedChunk(chunk=c_limbo, priority=ChunkPriority.LIMBO, similarity=0.75, matched_idea="Existing")],
        high_confidence=[RoutedChunk(chunk=c_high, priority=ChunkPriority.HIGH_CONFIDENCE, similarity=0.95, matched_idea="Existing")],
    )

    assert [c.chunk_id for c in routing.get_llm_queue_chunks(skip_limbo=False)] == ["c_u", "c_l"]
    assert [c.chunk_id for c in routing.get_llm_queue_chunks(skip_limbo=True)] == ["c_u"]


def test_build_graph_skip_limbo_chunks_flag(tmp_path: Path):
    """
    Verify that when --skip-limbo-chunks is passed, chunks in limbo (0.65 <= sim < 0.90)
    are skipped completely and do not invoke LLM extraction.
    """
    import sqlite3
    from unittest.mock import patch
    from typer.testing import CliRunner
    from bookeeper.cli import app
    from bookeeper.processing.extractor import SectionExtraction, KnowledgeExtractor
    from bookeeper.calibre.client import _register_sqlite_functions
    from bookeeper.calibre.parser import Section
    from bookeeper.graph.store import ConceptGraphStore

    runner = CliRunner()
    lib_dir = tmp_path / "calibre_lib"
    lib_dir.mkdir()
    db_path = lib_dir / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    _register_sqlite_functions(conn)
    conn.executescript("""
        CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL, sort TEXT, author_sort TEXT, pubdate TIMESTAMP, path TEXT NOT NULL DEFAULT 'Author/Book');
        CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE);
        CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER NOT NULL, author INTEGER NOT NULL);
        CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER NOT NULL, format TEXT NOT NULL, name TEXT NOT NULL);
        CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER NOT NULL UNIQUE, text TEXT NOT NULL);
    """)

    conn.execute("INSERT INTO books (id, title, sort, author_sort, pubdate, path) VALUES (1, 'Book Beta', 'Book Beta', 'Author', '2021-03-01', 'A/B1')")
    conn.execute("INSERT INTO authors (id, name) VALUES (1, 'Jane Doe')")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (1, 1)")
    conn.execute("INSERT INTO data (book, format, name) VALUES (1, 'EPUB', 'Book Beta')")
    conn.commit()
    conn.close()

    (lib_dir / "A" / "B1").mkdir(parents=True)
    (lib_dir / "A" / "B1" / "Book Beta.epub").write_text("dummy", encoding="utf-8")

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\ncalibre_library_path: {lib_dir}\n", encoding="utf-8")

    # Seed the knowledge graph with an existing idea
    existing_store = ConceptGraphStore()
    existing_store.add_book(100, title="Preexisting Book", publication_date="2010-05-01")
    existing_store.add_book_idea_link(100, "Paxos Consensus", publication_date="2010-05-01")
    kg_path = out_dir / "knowledge_graph.json"
    existing_store.save(kg_path)

    mock_text = (
        "Raft consensus provides an understandable alternative to Paxos for replicated state machines. "
        "Leader election and log replication are decomposed into distinct subproblems. "
        "Safety properties ensure logs remain consistent across cluster terms. "
    )

    mock_extract_calls = []

    def mock_extract_chunks(chunks, **kwargs):
        mock_extract_calls.append(chunks)
        return {c.chunk_id: SectionExtraction(concepts=[]) for c in chunks}

    mock_parse_sections = [
        Section(title="Section 1", chapter_idx=1, text=mock_text)
    ]

    # Idea vector = [1.0, 0.0], Chunk vector = [0.75, sqrt(1 - 0.75^2)] -> cosine similarity = 0.75 (LIMBO: 0.65 <= 0.75 < 0.90)
    mock_embeddings = MagicMock()
    mock_embeddings.embedding_stats = {}
    limbo_vec = [0.75, float(np.sqrt(1.0 - 0.75**2))]

    def _embed_docs(texts):
        res_vecs = []
        for t in texts:
            if t.startswith("Paxos Consensus:"):
                res_vecs.append([1.0, 0.0])
            else:
                res_vecs.append(limbo_vec)
        return res_vecs

    mock_embeddings.embed_documents.side_effect = _embed_docs
    mock_embeddings.embed_query.side_effect = lambda text: limbo_vec

    with patch("bookeeper.calibre.parser.BookParser.parse", return_value=mock_parse_sections), \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False), \
         patch("bookeeper.processing.ollama_pool.FailoverOllamaEmbeddings.from_settings", return_value=mock_embeddings), \
         patch.object(KnowledgeExtractor, "extract_chunks", side_effect=mock_extract_chunks), \
         patch.object(KnowledgeExtractor, "from_settings") as mock_from_settings, \
         patch("bookeeper.processing.deduplicator.EntityDeduplicator.from_settings") as mock_dedup_cls:

        mock_ext_instance = MagicMock()
        mock_ext_instance.extract_chunks.side_effect = mock_extract_chunks
        mock_from_settings.return_value = mock_ext_instance

        mock_dedup_instance = MagicMock()
        mock_dedup_instance.resolve_concept.side_effect = lambda c: c
        mock_dedup_cls.return_value = mock_dedup_instance

        res = runner.invoke(
            app,
            [
                "build-graph",
                "--all",
                "--calibre-path", str(lib_dir),
                "--config", str(cfg_file),
                "--skip-warmup",
                "--recent-ideas",
                "--similarity-unassigned", "0.65",
                "--similarity-high-confidence", "0.90",
                "--skip-limbo-chunks",
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        assert "Limbo chunk (0.75) skipped (--skip-limbo-chunks enabled)" in res.stdout
        assert "1 limbo (skipped)" in res.stdout
        assert "Skipped Chunks (Limbo): 1" in res.stdout
        # Because the chunk was in limbo and --skip-limbo-chunks was enabled, LLM extract_chunks was SKIPPED!
        assert len(mock_extract_calls) == 0

        # Verify `bookeeper status` reports the assigned and skipped chunks
        status_res = runner.invoke(
            app,
            ["status", "--config", str(cfg_file)],
            catch_exceptions=False,
        )
        assert status_res.exit_code == 0
        assert "Assigned Chunks: 0" in status_res.stdout
        assert "Skipped Chunks (Limbo): 1" in status_res.stdout
        assert "assigned=0" in status_res.stdout
        assert "skipped=1" in status_res.stdout


def test_ui_ideas_list_and_search_with_month_range(tmp_path: Path):
    """
    Verify SearchEngineBackend.get_ideas_list_and_search supports:
      1. Full ideas listing with first_occurrence_month / month
      2. Exact lexical search for ideas
      3. Embedded semantic search for ideas
      4. Filtering by how modern the idea is (month_from and month_to in YYYY-MM / Month+Year)
    """
    import numpy as np
    from unittest.mock import patch
    from bookeeper.config import Settings
    from bookeeper.processing.extractor import Concept, SectionExtraction
    from bookeeper.ui.backend import SearchEngineBackend
    from bookeeper.ui.embeddings import ConceptEmbeddingIndex

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)

    store = ConceptGraphStore()
    store.add_book(1, title="Classical Mechanics", author="Newton", publication_date="2018-03-15")
    store.add_book(2, title="Transformer Architecture", author="Vaswani", publication_date="2021-06-10")
    store.add_book(3, title="Modern Agentic Workflows", author="Deepmind", publication_date="2024-11-05")

    store.add_concept(
        Concept(
            name="Hamiltonian Dynamics",
            category="Physics",
            brief_description="Phase space formulation of classical mechanics.",
            weight=8,
        )
    )
    store.add_book_idea_link(1, "Hamiltonian Dynamics", publication_date="2018-03-15")

    store.add_concept(
        Concept(
            name="Multi-Head Self Attention",
            category="Deep Learning",
            brief_description="Parallel attention heads attending to different representation subspaces.",
            weight=9,
        )
    )
    store.add_book_idea_link(2, "Multi-Head Self Attention", publication_date="2021-06-10")

    store.add_concept(
        Concept(
            name="Autonomous Tool-Use Agents",
            category="AI Agents",
            brief_description="LLM agents invoking external tools and reasoning loops.",
            weight=10,
        )
    )
    store.add_book_idea_link(3, "Autonomous Tool-Use Agents", publication_date="2024-11-05")
    store.save(out_dir / "knowledge_graph.json")

    cfg = Settings(output_dir=str(out_dir))
    backend = SearchEngineBackend(cfg)

    # Force Neo4j offline so local knowledge_graph.json is used deterministically
    with patch.object(backend, "get_neo4j_driver", side_effect=RuntimeError("Neo4j offline")):
        # 1. List all ideas (sorted newest month first by default)
        res_all = backend.get_ideas_list_and_search(sort_by="month_desc")
        assert res_all["total"] == 3
        assert res_all["filtered"] == 3
        assert res_all["min_month"] == "2018-03"
        assert res_all["max_month"] == "2024-11"
        names_ordered = [i["name"] for i in res_all["ideas"]]
        assert names_ordered == [
            "Autonomous Tool-Use Agents",
            "Multi-Head Self Attention",
            "Hamiltonian Dynamics",
        ]

        # 2. Filter by modernity (Month + Year From / To)
        res_modern = backend.get_ideas_list_and_search(
            month_from="2021-01",
            month_to="2024-12",
        )
        assert res_modern["filtered"] == 2
        assert [i["name"] for i in res_modern["ideas"]] == [
            "Autonomous Tool-Use Agents",
            "Multi-Head Self Attention",
        ]

        # Also test human Month + Year string formats (e.g. "Jan 2023" to "Dec 2025")
        res_very_modern = backend.get_ideas_list_and_search(
            month_from="Jan 2023",
            month_to="12/2025",
        )
        assert res_very_modern["filtered"] == 1
        assert res_very_modern["ideas"][0]["name"] == "Autonomous Tool-Use Agents"
        assert res_very_modern["ideas"][0]["month"] == "2024-11"

        # 3. Exact Search + Month Range Filter
        res_exact = backend.get_ideas_list_and_search(
            query="Attention",
            search_type="exact",
            month_from="2020-01",
            month_to="2023-12",
        )
        assert res_exact["search_type"] == "exact"
        assert res_exact["filtered"] == 1
        assert res_exact["ideas"][0]["name"] == "Multi-Head Self Attention"
        assert res_exact["ideas"][0]["match_type"] == "substring_name"

        # Exact search excluded by modernity filter
        res_exact_excluded = backend.get_ideas_list_and_search(
            query="Attention",
            search_type="exact",
            month_from="2023-01",
            month_to="2025-12",
        )
        assert res_exact_excluded["filtered"] == 0

        # 4. Embedded Search + Month Range Filter
        mock_emb_idx = ConceptEmbeddingIndex(cache_path=out_dir / "concept_embeddings.npz")
        mock_emb_idx.concept_ids = [
            "concept:Hamiltonian Dynamics",
            "concept:Multi-Head Self Attention",
            "concept:Autonomous Tool-Use Agents",
        ]
        mock_emb_idx.names = [
            "Hamiltonian Dynamics",
            "Multi-Head Self Attention",
            "Autonomous Tool-Use Agents",
        ]
        mock_emb_idx.categories = ["Physics", "Deep Learning", "AI Agents"]
        mock_emb_idx.weights = np.array([8.0, 9.0, 10.0], dtype=np.float32)
        mock_emb_idx._norm_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.85, 0.5268],
                [0.0, 0.98, 0.199],
            ],
            dtype=np.float32,
        )

        with patch.object(backend, "_get_concept_embedding_index", return_value=mock_emb_idx), \
             patch.object(mock_emb_idx, "embed_text", return_value=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
            # Without date filter, both neural concepts match >= 0.70
            res_emb = backend.get_ideas_list_and_search(
                query="neural sequence modeling and agents",
                search_type="embedded",
                similarity_threshold=0.70,
                sort_by="similarity_desc",
            )
            assert res_emb["search_type"] == "embedded"
            assert res_emb["filtered"] == 2
            assert res_emb["ideas"][0]["name"] == "Autonomous Tool-Use Agents"
            assert res_emb["ideas"][0]["cosine_similarity"] == pytest.approx(0.98, abs=1e-3)
            assert res_emb["ideas"][1]["name"] == "Multi-Head Self Attention"
            assert res_emb["ideas"][1]["cosine_similarity"] == pytest.approx(0.85, abs=1e-3)

            # With date filter restricting to 2021 only, only Multi-Head Self Attention remains
            res_emb_filtered = backend.get_ideas_list_and_search(
                query="neural sequence modeling and agents",
                search_type="embedded",
                month_from="2021-01",
                month_to="2021-12",
                similarity_threshold=0.70,
            )
            assert res_emb_filtered["filtered"] == 1
            assert res_emb_filtered["ideas"][0]["name"] == "Multi-Head Self Attention"


def test_ui_server_ideas_http_endpoint(tmp_path: Path):
    """Verify GET /api/ideas and POST /api/ideas/search HTTP endpoints and UI index.html Ideas page."""
    import json
    import threading
    from http.server import HTTPServer
    from urllib.request import Request, urlopen
    from unittest.mock import patch
    from bookeeper.config import Settings
    from bookeeper.processing.extractor import Concept, SectionExtraction
    from bookeeper.ui.backend import SearchEngineBackend
    from bookeeper.ui.server import BookeeperUIRequestHandler

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)

    store = ConceptGraphStore()
    store.add_book(10, title="Early Paper", publication_date="2019-04-01")
    store.add_book(20, title="Recent Paper", publication_date="2025-02-01")
    store.add_concept(Concept(name="Legacy Graph Kernel", category="ML", brief_description="Old kernel", weight=6))
    store.add_book_idea_link(10, "Legacy Graph Kernel", publication_date="2019-04-01")
    store.add_concept(Concept(name="Test-Time Compute Scaling", category="LLM", brief_description="Modern scaling law", weight=9))
    store.add_book_idea_link(20, "Test-Time Compute Scaling", publication_date="2025-02-01")
    store.save(out_dir / "knowledge_graph.json")

    cfg = Settings(output_dir=str(out_dir))
    backend = SearchEngineBackend(cfg)

    with patch.object(backend, "get_neo4j_driver", side_effect=RuntimeError("Offline")):
        BookeeperUIRequestHandler.backend = backend
        server = HTTPServer(("127.0.0.1", 0), BookeeperUIRequestHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            # 1. Verify GET /api/ideas with month_from filter
            with urlopen(f"http://127.0.0.1:{port}/api/ideas?month_from=2024-01&month_to=2025-12") as resp:
                assert resp.status == 200
                payload = json.loads(resp.read().decode("utf-8"))
                assert payload["total"] == 2
                assert payload["filtered"] == 1
                assert payload["ideas"][0]["name"] == "Test-Time Compute Scaling"
                assert payload["ideas"][0]["month"] == "2025-02"

            # 2. Verify POST /api/ideas/search with exact search
            req = Request(
                f"http://127.0.0.1:{port}/api/ideas/search",
                data=json.dumps({
                    "query": "Legacy Graph",
                    "search_type": "exact",
                    "month_from": "2018-01",
                    "month_to": "2020-12",
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req) as resp:
                assert resp.status == 200
                payload = json.loads(resp.read().decode("utf-8"))
                assert payload["filtered"] == 1
                assert payload["ideas"][0]["name"] == "Legacy Graph Kernel"

            # 3. Verify index.html contains the Ideas tab, search mode selector, and month+year range controls
            with urlopen(f"http://127.0.0.1:{port}/") as resp:
                html_text = resp.read().decode("utf-8")
                assert 'id="tab-btn-ideas"' in html_text
                assert 'id="pane-ideas"' in html_text
                assert 'id="ideas-search-input"' in html_text
                assert 'id="ideas-month-from"' in html_text
                assert 'id="ideas-month-to"' in html_text
        finally:
            server.shutdown()
            server.server_close()





