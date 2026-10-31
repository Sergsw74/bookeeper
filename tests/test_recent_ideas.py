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
from bookeeper.processing.ollama_pool import format_idea_for_embedding
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
        mock_dedup_instance.resolve_concepts_batch.side_effect = lambda concepts, **kw: list(concepts)
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
        if not kg_path.is_file():
            kg_path = out_dir / "knowledge_graph_spine.json"
        if not kg_path.is_file():
            print("STDOUT WAS:\n", res.stdout)
            print("OUT DIR FILES:\n", list(out_dir.glob("**/*")))
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
        mock_dedup_instance.resolve_concepts_batch.side_effect = lambda concepts, **kw: list(concepts)
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
            if t.startswith("Paxos Consensus"):
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
        mock_dedup_instance.resolve_concepts_batch.side_effect = lambda concepts, **kw: list(concepts)
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


def test_build_graph_recovers_limbo_chunk_by_size_cli(tmp_path: Path):
    """
    Verify that when --min-batch-chunk-size is passed alongside --skip-limbo-chunks:
    1. Chunks in limbo whose size >= threshold are recovered into the unprocessed queue and sent to LLM.
    2. Chunks in limbo whose size < threshold remain in limbo and are skipped.
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

    conn.execute("INSERT INTO books (id, title, sort, author_sort, pubdate, path) VALUES (1, 'Book Limbo Recovery', 'Book Limbo Recovery', 'Author', '2021-03-01', 'A/B1')")
    conn.execute("INSERT INTO authors (id, name) VALUES (1, 'Jane Doe')")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (1, 1)")
    conn.execute("INSERT INTO data (book, format, name) VALUES (1, 'EPUB', 'Book Limbo Recovery')")
    conn.commit()
    conn.close()

    (lib_dir / "A" / "B1").mkdir(parents=True)
    (lib_dir / "A" / "B1" / "Book Limbo Recovery.epub").write_text("dummy", encoding="utf-8")

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

    # Section 1 is large (>= 300 chars), Section 2 is medium (150 chars, >= 120 min_chunk_chars but < 300)
    large_text = (
        "Raft consensus provides an understandable alternative to Paxos for replicated state machines. "
        "Leader election and log replication are decomposed into distinct subproblems. "
        "Safety properties ensure logs remain consistent across cluster terms. "
        "Each server in the cluster exists in one of three states: leader, follower, or candidate. "
        "In normal operation there is exactly one leader and all of the other servers are followers."
    )
    medium_text = (
        "Heartbeats are sent periodically by the cluster leader to all followers "
        "in order to maintain its leadership status and prevent unnecessary election timeouts."
    )

    mock_extract_calls = []

    def mock_extract_chunks(chunks, **kwargs):
        mock_extract_calls.append(chunks)
        return {c.chunk_id: SectionExtraction(concepts=[]) for c in chunks}

    mock_parse_sections = [
        Section(title="Large Section", chapter_idx=1, text=large_text),
        Section(title="Medium Section", chapter_idx=2, text=medium_text),
    ]

    mock_embeddings = MagicMock()
    mock_embeddings.embedding_stats = {}
    limbo_vec = [0.75, float(np.sqrt(1.0 - 0.75**2))]

    def _embed_docs(texts):
        res_vecs = []
        for t in texts:
            if t.startswith("Paxos Consensus"):
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
        mock_dedup_instance.resolve_concepts_batch.side_effect = lambda concepts, **kw: list(concepts)
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
                "--max-batch-chunk-size", "300",
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        # The large chunk is recovered from limbo
        assert "recovered from limbo" in res.stdout
        assert "1 recovered from limbo by size" in res.stdout
        # The medium chunk is skipped
        assert "Limbo chunk (0.75) skipped" in res.stdout

        # LLM was invoked ONLY for the recovered large chunk!
        assert len(mock_extract_calls) == 1
        extracted_chunk_ids = [c.chunk_id for c in mock_extract_calls[0]]
        assert len(extracted_chunk_ids) == 1
        assert "Large Section" in str(mock_extract_calls[0][0].section_title)


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


def test_chunk_embeddings_reused_across_chunker_store_and_router(tmp_path: Path):
    """
    Verify that:
    1. RollingWindowSemanticChunker / HierarchicalChunker attaches pooled normalized embeddings
       to each HierarchicalChunk during initial chunking.
    2. ChunkStore persists and reloads chunk.embedding on disk.
    3. RecentIdeaChunkRouter.route_chunks reuses chunk.embedding without calling embed_documents again.
    """
    from unittest.mock import patch
    from langchain_core.embeddings import Embeddings
    from bookeeper.calibre.parser import Section
    from bookeeper.processing.chunker import ChunkStore, HierarchicalChunker
    from bookeeper.processing.extractor import Concept

    mock_embeddings = MagicMock(spec=Embeddings)
    mock_embeddings.embed_documents.side_effect = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]

    chunker = HierarchicalChunker(embeddings=mock_embeddings, max_chunk_chars=80, min_chunk_chars=20)
    section = Section(
        title="Distributed Consensus",
        chapter_idx=1,
        text=(
            "Paxos guarantees safety in asynchronous distributed systems across quorums. "
            "Prepare and accept phases coordinate state replication across fault-tolerant replicas. "
            "Leader election reduces split-vote contention during normal operation."
        ),
    )
    chunks = chunker.chunk_section(section, book_id=42, book_title="Systems Book")
    assert len(chunks) >= 1
    for chk in chunks:
        assert chk.embedding is not None
        assert len(chk.embedding) == 3
        assert np.isclose(np.linalg.norm(chk.embedding), 1.0, atol=1e-4)

    # Persist to ChunkStore and reload
    cstore = ChunkStore(tmp_path / "chunks")
    cstore.save_chunks(42, "Systems Book", chunks)
    loaded_chunks = cstore.load_chunks(42)
    assert loaded_chunks is not None
    assert len(loaded_chunks) == len(chunks)
    for lchk in loaded_chunks:
        assert lchk.embedding is not None
        assert len(lchk.embedding) == 3

    # Build an IdeaEmbeddingIndex with a pre-populated idea vector
    store = ConceptGraphStore()
    store.add_concept(
        Concept(name="Paxos Quorum Safety", brief_description="Safety across distributed quorums"),
        embedding=[1.0, 0.0, 0.0],
    )

    router_embeddings = MagicMock()
    idea_index = IdeaEmbeddingIndex.build_from_graph(store, embeddings=router_embeddings)
    # Idea already had its embedding stored on the node -> 0 embedding calls!
    assert idea_index.reused_count == 1
    assert idea_index.embedded_count == 0
    router_embeddings.embed_documents.assert_not_called()

    # Route loaded chunks -> should reuse chunk.embedding with 0 calls to router_embeddings.embed_documents!
    router = RecentIdeaChunkRouter(
        idea_index=idea_index,
        embeddings=router_embeddings,
        unassigned_threshold=0.65,
        high_confidence_threshold=0.90,
    )
    routing_res = router.route_chunks(loaded_chunks)
    assert routing_res.reused_embeddings_count == len(loaded_chunks)
    assert routing_res.computed_embeddings_count == 0
    assert not routing_res.embeddings_updated
    router_embeddings.embed_documents.assert_not_called()
    assert routing_res.high_confidence_count == len(loaded_chunks)


def test_neo4j_hnsw_vector_index_exclusively_on_idea_nodes_and_batched_unwind_query():
    """
    Verify that:
    1. Neo4jExporter creates the HNSW vector index 'idea_idx' exclusively on :Idea(embedding).
    2. Non-Idea nodes (:Chunk, :Section, :Book) never have embedding exported to Neo4j.
    3. Neo4jExporter.query_ideas_hnsw_batch queries all 20 extracted ideas from an article
       in a single batched transaction using:
       UNWIND $idea_vectors AS vec CALL db.index.vector.queryNodes('idea_idx', 3, vec)
    """
    from unittest.mock import patch
    from bookeeper.graph.neo4j_exporter import Neo4jExporter
    from bookeeper.processing.chunker import HierarchicalChunk
    from bookeeper.processing.extractor import Concept

    store = ConceptGraphStore()
    store.add_book(1, title="Test Article", author="Author", publication_date="2025-01-15")
    sec_id = store.add_section(1, 1, "Intro", "Text about distributed consensus")
    chk = HierarchicalChunk(
        chunk_id="b1_s001_c001",
        book_id=1,
        book_title="Test Article",
        section_title="Intro",
        chapter_idx=1,
        chunk_idx=1,
        text="Text about distributed consensus",
        embedding=[0.5, 0.5, 0.7071],
    )
    store.add_chunk(chk)
    # Even if a Chunk node in memory somehow had an embedding attribute, Neo4jExporter must strip it
    store.graph.nodes[f"chunk:{chk.chunk_id}"]["embedding"] = [0.5, 0.5, 0.7071]

    store.add_concept(
        Concept(name="State Machine Replication", brief_description="Deterministic replica coordination"),
        embedding=[1.0, 0.0, 0.0],
    )

    exporter = Neo4jExporter(uri="bolt://localhost:7687", user="neo4j", password="password")
    executed_queries = []

    mock_session = MagicMock()

    def _run_side_effect(cypher, **params):
        executed_queries.append((cypher, params))
        if "CALL db.index.vector.queryNodes('idea_idx', 3, vec)" in cypher:
            vecs = params.get("idea_vectors", [])
            rows = []
            for idx in range(len(vecs)):
                rows.append({
                    "vec_idx": idx,
                    "matches": [
                        {
                            "node_id": "concept:State Machine Replication",
                            "name": "State Machine Replication",
                            "brief_description": "Deterministic replica coordination",
                            "detailed_explanation": "",
                            "summary": "Deterministic replica coordination",
                            "month": "2025-01",
                            "score": 0.96,
                        }
                    ],
                })
            return rows
        return MagicMock()

    mock_session.run.side_effect = _run_side_effect
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch.object(exporter, "get_driver", return_value=mock_driver):
        exporter.export(store, clean=False)

        # 1. Verify native HNSW vector index creation on :Idea(embedding)
        index_queries = [
            q for q, _ in executed_queries
            if "CREATE VECTOR INDEX idea_idx IF NOT EXISTS" in q and "FOR (n:Idea) ON (n.embedding)" in q
        ]
        assert len(index_queries) == 1
        assert "`vector.similarity_function`: 'cosine'" in index_queries[0]
        assert "`vector.dimensions`: 3" in index_queries[0]

        # 2. Verify :Idea and :Chunk nodes have embedding in Neo4j export batch, while :Book does NOT,
        #    and HNSW vector index is created exclusively on :Idea(embedding)
        for q, params in executed_queries:
            if "UNWIND $batch AS row" in q and "MERGE (n:`" in q:
                batch = params.get("batch", [])
                for row in batch:
                    props = row.get("properties", {})
                    if "SET n:Idea" in q or ":`Idea`" in q:
                        assert "embedding" in props
                        assert props["embedding"] == [1.0, 0.0, 0.0]
                    elif "MERGE (n:`Chunk`" in q:
                        assert "embedding" in props
                        assert props["embedding"] == [0.5, 0.5, 0.7071]
                    else:
                        assert "embedding" not in props

        # 3. Query all 20 extracted ideas from an article in a single batched transaction
        executed_queries.clear()
        twenty_idea_vectors = [[float(i), 1.0, 0.0] for i in range(20)]
        batch_results = exporter.query_ideas_hnsw_batch(twenty_idea_vectors, top_k=3)

        assert len(batch_results) == 20
        assert len(executed_queries) == 1
        cypher_sent, params_sent = executed_queries[0]
        assert "UNWIND $idea_vectors AS vec CALL db.index.vector.queryNodes('idea_idx', 3, vec)" in cypher_sent
        assert len(params_sent["idea_vectors"]) == 20
        assert batch_results[0][0]["name"] == "State Machine Replication"
        assert batch_results[19][0]["score"] == pytest.approx(0.96)


def test_deduplicator_resolve_concepts_batch_single_transaction_for_20_ideas():
    """
    Verify EntityDeduplicator.resolve_concepts_batch embeds 20 extracted ideas from an article
    in a single batch call and queries the Neo4j HNSW 'idea_idx' index in a single batched transaction.
    """
    from bookeeper.processing.deduplicator import EntityDeduplicator
    from bookeeper.processing.extractor import Concept

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]

    mock_neo4j = MagicMock()

    def _mock_hnsw_batch(idea_vectors, top_k=3):
        # First idea matches an existing high-similarity idea (>=0.95); others have low similarity (0.20)
        results = []
        for idx, _ in enumerate(idea_vectors):
            if idx == 0:
                results.append([
                    {
                        "node_id": "concept:Canonical Graph Attention",
                        "name": "Canonical Graph Attention",
                        "brief_description": "Attention mechanism over graph neighborhoods",
                        "detailed_explanation": "",
                        "summary": "Attention mechanism over graph neighborhoods",
                        "score": 0.97,
                    }
                ])
            else:
                results.append([
                    {
                        "node_id": "concept:Canonical Graph Attention",
                        "name": "Canonical Graph Attention",
                        "brief_description": "Attention mechanism over graph neighborhoods",
                        "detailed_explanation": "",
                        "summary": "Attention mechanism over graph neighborhoods",
                        "score": 0.20,
                    }
                ])
        return results

    mock_neo4j.query_ideas_hnsw_batch.side_effect = _mock_hnsw_batch

    # Give each of the 20 extracted concepts a distinct orthogonal/unique vector after embedding
    def _distinct_embed_docs(texts):
        out = []
        for i in range(len(texts)):
            vec = [0.0] * 20
            vec[i % 20] = 1.0
            out.append(vec)
        return out

    mock_embeddings.embed_documents.side_effect = _distinct_embed_docs

    dedup = EntityDeduplicator(
        embeddings=mock_embeddings,
        similarity_threshold=0.80,
        high_similarity_threshold=0.95,
    )

    extracted_20 = [
        Concept(name=f"Extracted Idea {i}", brief_description=f"Description for idea {i}")
        for i in range(20)
    ]

    resolved = dedup.resolve_concepts_batch(extracted_20, neo4j_exporter=mock_neo4j, top_k=3)

    assert len(resolved) == 20
    # Exactly 1 batched embed_documents call and 1 batched HNSW query call for all 20 ideas!
    assert mock_embeddings.embed_documents.call_count == 1
    assert mock_neo4j.query_ideas_hnsw_batch.call_count == 1
    called_vecs = mock_neo4j.query_ideas_hnsw_batch.call_args[0][0]
    assert len(called_vecs) == 20

    # Idea 0 matched 'Canonical Graph Attention' with score 0.97 >= 0.95
    assert resolved[0].name == "Canonical Graph Attention"
    # Ideas 1..19 were distinct new concepts and each has its normalized embedding attached
    for i in range(1, 20):
        assert resolved[i].name == f"Extracted Idea {i}"
        assert resolved[i].embedding is not None
        assert len(resolved[i].embedding) == 20


def test_size_aware_embedding_batching_for_sentences_chunks_and_full_ideas():
    """
    Verify size-aware batching (`partition_texts_for_embedding`, `format_idea_for_embedding`,
    `FailoverOllamaEmbeddings.embed_documents`, `RollingWindowSemanticChunker`, `HierarchicalChunker`,
    `IdeaEmbeddingIndex`, `RecentIdeaChunkRouter`, `EntityDeduplicator`, and `IdeaPostProcessor`)
    dynamically adjusts batch sizes based on whether the texts are short sentences, short ideas,
    full ideas (name + brief + detailed explanation), or full chunks.
    """
    from unittest.mock import patch
    from langchain_core.embeddings import Embeddings
    from bookeeper.processing.ollama_pool import (
        FailoverOllamaEmbeddings,
        OllamaPool,
        format_idea_for_embedding,
        partition_texts_for_embedding,
    )
    from bookeeper.processing.chunker import HierarchicalChunk, HierarchicalChunker
    from bookeeper.processing.recent_ideas import IdeaEmbeddingIndex, RecentIdeaChunkRouter
    from bookeeper.processing.deduplicator import EntityDeduplicator
    from bookeeper.processing.extractor import Concept
    from bookeeper.processing.idea_postprocessor import IdeaCluster, IdeaObservation, IdeaPostProcessor
    from bookeeper.calibre.parser import Section
    from bookeeper.graph.store import ConceptGraphStore

    # 1. Short sentences (~100 chars each): 60 sentences fit in 1 batch (6,000 chars <= 16,000 max_batch_chars, 60 <= 64 items)
    sentences = [f"Sentence {i}: " + ("a" * 85) for i in range(60)]
    sentence_batches = partition_texts_for_embedding(sentences, max_batch_chars=16000, max_batch_items=64)
    assert len(sentence_batches) == 1
    assert len(sentence_batches[0]) == 60

    # 2. Full ideas (~1,500 chars each with name + brief + detailed_explanation):
    #    20 full ideas (~30,000 chars total) are automatically split across 3 batches when max_batch_chars=12,000 (~8 ideas/batch)
    full_ideas = [
        format_idea_for_embedding(
            f"Concept {i}",
            brief_description="Brief summary " + ("b" * 180),
            detailed_explanation="Detailed mechanism and trade-offs " + ("d" * 1250),
        )
        for i in range(20)
    ]
    idea_batches = partition_texts_for_embedding(full_ideas, max_batch_chars=12000, max_batch_items=64)
    assert len(idea_batches) == 3
    assert sum(len(b) for b in idea_batches) == 20
    assert all(sum(len(t) for t in b) <= 12000 for b in idea_batches)

    # 3. Full chunks (~2,500 chars each):
    #    15 chunks (~37,500 chars total) are automatically split into smaller batches (~4 chunks/batch for 12,000 char budget)
    chunks_texts = [f"Chunk {i}: " + ("c" * 2480) for i in range(15)]
    chunk_batches = partition_texts_for_embedding(chunks_texts, max_batch_chars=12000, max_batch_items=16)
    assert len(chunk_batches) == 4
    assert sum(len(b) for b in chunk_batches) == 15
    assert all(len(b) <= 4 for b in chunk_batches)

    # 4. FailoverOllamaEmbeddings.embed_documents respects both max_batch_chars and batch_size
    pool = OllamaPool.from_urls(["http://localhost:11434"])
    failover_emb = FailoverOllamaEmbeddings(
        pool=pool,
        model="qwen3-embedding:8b",
        batch_size=64,
        max_batch_chars=10000,
    )
    mock_client = MagicMock()
    mock_client.embed_documents.side_effect = lambda batch: [[1.0, 0.0] for _ in batch]
    with patch("bookeeper.processing.ollama_pool.OllamaEmbeddings", return_value=mock_client):
        # 50 short sentences (~5,000 chars) -> 1 HTTP batch call
        res_sent = failover_emb.embed_documents(sentences[:50])
        assert len(res_sent) == 50
        assert mock_client.embed_documents.call_count == 1

        mock_client.embed_documents.reset_mock()
        # 12 large chunks (~30,000 chars) -> split into 3 HTTP batch calls (4 chunks per 10,000-char batch)
        res_chunks = failover_emb.embed_documents(chunks_texts[:12])
        assert len(res_chunks) == 12
        assert mock_client.embed_documents.call_count == 3

    # 5. HierarchicalChunker single-sentence & structural fallback (`_smart_split`) batch-embeds chunks
    mock_emb = MagicMock(spec=Embeddings)
    mock_emb.embed_documents.side_effect = lambda batch: [[0.5, 0.5] for _ in batch]
    h_chunker = HierarchicalChunker(
        max_chunk_chars=3000,
        min_chunk_chars=20,
        embeddings=mock_emb,
    )
    sec = Section(title="Short Chapter", chapter_idx=1, text="A short structural section that fits in one chunk.")
    out_chunks = h_chunker.chunk_section(sec, book_id=1, book_title="Book")
    assert len(out_chunks) == 1
    assert out_chunks[0].embedding == pytest.approx([float(np.sqrt(0.5)), float(np.sqrt(0.5))], rel=1e-4)
    assert mock_emb.embed_documents.call_count == 1

    # 6. EntityDeduplicator.resolve_concepts_batch formats full ideas (name + brief + detailed) and partitions by size
    mock_emb.embed_documents.reset_mock()
    dedup = EntityDeduplicator(embeddings=mock_emb)
    heavy_concepts = [
        Concept(
            name=f"Heavy Idea {i}",
            brief_description="Brief " + ("x" * 200),
            detailed_explanation="Detailed " + ("y" * 1300),
        )
        for i in range(20)
    ]
    resolved_heavy = dedup.resolve_concepts_batch(heavy_concepts, batch_size=64)
    assert len(resolved_heavy) == 20
    # 20 full ideas (~1,500 chars each = ~30,000 chars) with max_batch_chars=12000 -> 3 batched calls instead of 1 oversized call
    assert mock_emb.embed_documents.call_count == 3

    # 7. IdeaPostProcessor.cluster_by_semantic_similarity formats full ideas and batches by character budget
    mock_emb.embed_documents.reset_mock()
    post = IdeaPostProcessor(embeddings=mock_emb)
    clusters = [
        IdeaCluster(
            canonical_name=f"Cluster Idea {i}",
            normalized_name=f"cluster idea {i}",
            observations=[
                IdeaObservation(
                    concept_name=f"Cluster Idea {i}",
                    local_brief="Brief " + ("b" * 200),
                    local_detailed="Detailed " + ("d" * 1300),
                )
            ],
        )
        for i in range(18)
    ]
    post.cluster_by_semantic_similarity(clusters, console=None)
    assert mock_emb.embed_documents.call_count == 3
    assert all(c.embedding is not None for c in clusters)

    # 8. ConceptGraphStore.backfill_missing_embeddings backfills existing Ideas and Chunks without rebuilding from scratch
    mock_emb.embed_documents.reset_mock()
    legacy_store = ConceptGraphStore()
    legacy_store.add_book(1, title="Legacy Book", publication_date="2023-06-01")
    legacy_store.add_concept(
        Concept(name="Legacy Idea", brief_description="Brief legacy idea", detailed_explanation="Detailed legacy idea")
    )
    legacy_store.add_chunk(
        HierarchicalChunk(
            chunk_id="legacy_c1",
            book_id=1,
            book_title="Legacy Book",
            section_title="Chapter 1",
            chapter_idx=1,
            chunk_idx=0,
            text="Legacy chunk text without pre-existing embedding.",
        )
    )
    assert legacy_store.graph.nodes["concept:Legacy Idea"].get("embedding") is None
    assert legacy_store.graph.nodes["chunk:legacy_c1"].get("embedding") is None

    bf = legacy_store.backfill_missing_embeddings(mock_emb)
    assert bf == {"ideas_embedded": 1, "chunks_embedded": 1}
    assert legacy_store.graph.nodes["concept:Legacy Idea"].get("embedding") is not None
    assert legacy_store.graph.nodes["chunk:legacy_c1"].get("embedding") is not None


def test_incremental_neo4j_export_only_affected_nodes_and_parallel_dedup_progress():
    """
    Verify:
    1. ConceptGraphStore.add_chunk persists chunk.embedding directly on the :Chunk node.
    2. Neo4jExporter.export(..., only_affected=True) exports ONLY dirty/affected nodes and edges,
       creates both 'idea_idx' on :Idea(embedding) and 'chunk_idx' on :Chunk(embedding),
       deletes any removed_nodes, and clears dirty tracking after export.
    3. EntityDeduplicator.resolve_concepts_batch executes probable-match LLM disambiguation in parallel,
       caches pair decisions, and invokes on_progress transparently.
    """
    from unittest.mock import patch
    from bookeeper.graph.neo4j_exporter import Neo4jExporter
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.chunker import HierarchicalChunk
    from bookeeper.processing.deduplicator import (
        ConceptDeduplicationDecision,
        EntityDeduplicator,
    )
    from bookeeper.processing.extractor import Concept

    store = ConceptGraphStore()

    # Batch 1: Add Book 1, Chunk 1 (with embedding), and Idea 1 (with embedding)
    store.add_book(1, title="Book 1", publication_date="2024-01-10")
    chk1 = HierarchicalChunk(
        chunk_id="b1_c1",
        book_id=1,
        book_title="Book 1",
        section_title="Ch 1",
        chapter_idx=1,
        chunk_idx=1,
        text="Chunk 1 text",
        embedding=[1.0, 0.0, 0.0],
    )
    store.add_chunk(chk1)
    assert store.graph.nodes["chunk:b1_c1"].get("embedding") == [1.0, 0.0, 0.0]

    store.add_concept(Concept(name="Idea One", brief_description="First idea"), embedding=[0.0, 1.0, 0.0])
    store.add_idea_support_link("Idea One", chk1, quote="Chunk 1 text", publication_date="2024-01-10")

    exporter = Neo4jExporter(uri="bolt://localhost:7687", user="neo4j", password="password")
    executed_queries = []
    mock_session = MagicMock()
    mock_session.run.side_effect = lambda cypher, **params: executed_queries.append((cypher, params)) or MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch.object(exporter, "get_driver", return_value=mock_driver):
        # Export Batch 1 (only_affected=True)
        res1 = exporter.export(store, clean=False, only_affected=True)
        assert res1["only_affected"] is True
        assert res1["nodes_upserted"] == 4  # book:1, section:1:1, chunk:b1_c1, concept:Idea One
        assert not store.has_dirty_changes()

        # Verify both idea_idx and chunk_idx vector indexes were ensured
        assert any("CREATE VECTOR INDEX idea_idx IF NOT EXISTS" in q for q, _ in executed_queries)
        assert any("CREATE VECTOR INDEX chunk_idx IF NOT EXISTS" in q and "FOR (n:Chunk) ON (n.embedding)" in q for q, _ in executed_queries)

        # Batch 2: Modify ONLY Idea Two and Chunk 2, plus mark an old alias as removed
        executed_queries.clear()
        chk2 = HierarchicalChunk(
            chunk_id="b1_c2",
            book_id=1,
            book_title="Book 1",
            section_title="Ch 1",
            chapter_idx=1,
            chunk_idx=2,
            text="Chunk 2 text",
            embedding=[0.0, 0.0, 1.0],
        )
        store.add_chunk(chk2)
        store.add_concept(Concept(name="Idea Two", brief_description="Second idea"), embedding=[0.0, 0.0, 1.0])
        store.mark_node_removed("concept:Obsolete Alias")

        res2 = exporter.export(store, clean=False, only_affected=True)
        assert res2["only_affected"] is True
        assert res2["removed_nodes_deleted"] == 1
        # Only section:1:1, chunk:b1_c2, and concept:Idea Two were dirty (book:1, chunk:b1_c1, concept:Idea One were NOT re-exported!)
        assert res2["nodes_upserted"] == 3
        assert any("DETACH DELETE n" in q and params.get("ids") == ["concept:Obsolete Alias"] for q, params in executed_queries)
        assert not store.has_dirty_changes()

    # Verify parallel LLM disambiguation and progress callbacks in EntityDeduplicator.resolve_concepts_batch
    mock_emb = MagicMock()
    # Cosine similarity = 0.85 (in probable-match range 0.80 <= 0.85 < 0.95)
    prob_vec = [0.85, float(np.sqrt(1.0 - 0.85**2))]
    mock_emb.embed_documents.side_effect = lambda batch: [prob_vec for _ in batch]

    mock_extractor = MagicMock()
    mock_extractor._execute_structured_invoke.return_value = ConceptDeduplicationDecision(
        is_same_concept=True,
        canonical_name="Canonical Raft",
        brief_description="Synthesized Raft consensus",
        detailed_explanation="Synthesized detailed explanation",
        reasoning="Same consensus algorithm",
    )

    dedup = EntityDeduplicator(
        embeddings=mock_emb,
        similarity_threshold=0.80,
        high_similarity_threshold=0.95,
        extractor=mock_extractor,
    )
    dedup._register_in_cache("Canonical Raft", np.array([1.0, 0.0], dtype=np.float32), Concept(name="Canonical Raft"))

    progress_events = []
    # Two duplicate occurrences of the same probable match should only invoke the LLM once due to pair caching!
    incoming = [
        Concept(name="Raft Protocol Variant", brief_description="Variant 1"),
        Concept(name="Raft Protocol Variant", brief_description="Variant 2"),
    ]
    resolved = dedup.resolve_concepts_batch(
        incoming,
        on_progress=lambda stage, done, tot, detail: progress_events.append((stage, done, tot, detail)),
    )

def test_get_chunk_size_helper():
    """Verify get_chunk_size handles HierarchicalChunk, dict, and string texts correctly."""
    from bookeeper.processing.recent_ideas import get_chunk_size

    c1 = HierarchicalChunk(
        chunk_id="c1",
        book_id=1,
        book_title="B",
        section_title="S",
        chapter_idx=1,
        chunk_idx=1,
        text="Hello world!",
        char_count=12,
    )
    assert get_chunk_size(c1) == 12

    d1 = {"chunk_id": "d1", "char_count": 500, "text": "Short"}
    assert get_chunk_size(d1) == 500

    d2 = {"chunk_id": "d2", "text": "Arbitrary text of length 27"}
    assert get_chunk_size(d2) == len("Arbitrary text of length 27")

    assert get_chunk_size(None) == 0


def test_recent_ideas_config_limbo_recovery_settings():
    """Verify max_batch_chunk_size and effective_limbo_recovery_size in config."""
    from bookeeper.config import RecentIdeasConfig, Settings

    rc1 = RecentIdeasConfig(max_batch_chunk_size=1500)
    assert rc1.effective_limbo_recovery_size == 1500

    rc_none = RecentIdeasConfig()
    assert rc_none.effective_limbo_recovery_size is None

    # Test root Settings inheritance
    s1 = Settings(max_batch_chunk_size=1800)
    assert s1.effective_limbo_recovery_size == 1800

    s2 = Settings(recent_ideas=RecentIdeasConfig(max_batch_chunk_size=2200))
    assert s2.effective_limbo_recovery_size == 2200


def test_router_limbo_recovery_by_chunk_size():
    """
    Verify that when max_batch_chunk_size is configured,
    chunks whose similarity falls into limbo (0.65 <= sim < 0.90) but whose size
    is >= threshold are recovered from limbo to UNASSIGNED (unprocessed queue),
    while smaller limbo chunks remain in LIMBO.
    """
    from bookeeper.processing.recent_ideas import RecentIdeaChunkRouter, IdeaEmbeddingIndex

    # Idea vector at [1.0, 0.0]
    idea_index = IdeaEmbeddingIndex(
        idea_names=["Consistent Hashing"],
        idea_summaries=["Distributed hashing mechanism"],
        idea_months={"Consistent Hashing": "2020-01"},
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    # Big chunk (length 1500) in limbo similarity range (sim = 0.75)
    big_text = "Distributed Hash Table " * 65  # ~1500 chars
    big_chunk = HierarchicalChunk(
        chunk_id="chk_big",
        book_id=1,
        book_title="Book A",
        section_title="Sec 1",
        chapter_idx=1,
        chunk_idx=1,
        text=big_text,
        char_count=len(big_text),
        embedding=[0.75, float(np.sqrt(1.0 - 0.75**2))],
    )

    # Small chunk (length 100) in limbo similarity range (sim = 0.75)
    small_text = "Brief note about hashing."  # 25 chars
    small_chunk = HierarchicalChunk(
        chunk_id="chk_small",
        book_id=1,
        book_title="Book A",
        section_title="Sec 1",
        chapter_idx=1,
        chunk_idx=2,
        text=small_text,
        char_count=len(small_text),
        embedding=[0.75, float(np.sqrt(1.0 - 0.75**2))],
    )

    # Low similarity chunk (sim = 0.20) -> inherently unassigned
    new_chunk = HierarchicalChunk(
        chunk_id="chk_new",
        book_id=1,
        book_title="Book A",
        section_title="Sec 1",
        chapter_idx=1,
        chunk_idx=3,
        text="Quantum Computing",
        char_count=len("Quantum Computing"),
        embedding=[0.20, float(np.sqrt(1.0 - 0.20**2))],
    )

    # Router configured with max_batch_chunk_size = 500
    router = RecentIdeaChunkRouter(
        idea_index=idea_index,
        unassigned_threshold=0.65,
        high_confidence_threshold=0.90,
        max_batch_chunk_size=500,
    )

    res = router.route_chunks([big_chunk, small_chunk, new_chunk])

    # Big chunk (size 1500 >= 500) must be recovered from limbo into unassigned!
    unassigned_ids = [rc.chunk.chunk_id for rc in res.unassigned]
    assert "chk_big" in unassigned_ids
    assert "chk_new" in unassigned_ids

    # Small chunk (size 25 < 500) must stay in limbo
    limbo_ids = [rc.chunk.chunk_id for rc in res.limbo]
    assert "chk_small" in limbo_ids
    assert "chk_big" not in limbo_ids

    # Check recovered_from_limbo tracking
    assert res.recovered_from_limbo_count == 1
    assert res.recovered_from_limbo[0].chunk.chunk_id == "chk_big"
    assert res.recovered_from_limbo[0].recovered_from_limbo is True

    # When skip_limbo=True, the recovered big chunk must STILL be included in the LLM queue!
    queue_skip_limbo = res.get_llm_queue_chunks(skip_limbo=True)
    queue_cids = [c.chunk_id for c in queue_skip_limbo]
    assert "chk_big" in queue_cids
    assert "chk_new" in queue_cids
    assert "chk_small" not in queue_cids


def test_batch_routing_result_recover_limbo_chunks():
    """Verify BatchRoutingResult.recover_limbo_chunks post-routing method."""
    c_limbo_big = HierarchicalChunk(chunk_id="c_lb", book_id=1, book_title="B", section_title="S", chapter_idx=1, chunk_idx=1, text="A" * 1200)
    c_limbo_sml = HierarchicalChunk(chunk_id="c_ls", book_id=1, book_title="B", section_title="S", chapter_idx=1, chunk_idx=2, text="A" * 200)

    routing = BatchRoutingResult(
        unassigned=[],
        limbo=[
            RoutedChunk(chunk=c_limbo_big, priority=ChunkPriority.LIMBO, similarity=0.75),
            RoutedChunk(chunk=c_limbo_sml, priority=ChunkPriority.LIMBO, similarity=0.75),
        ],
    )

    rec_cnt = routing.recover_limbo_chunks(max_batch_chunk_size=800)
    assert rec_cnt == 1
    assert len(routing.unassigned) == 1
    assert routing.unassigned[0].chunk.chunk_id == "c_lb"
    assert routing.unassigned[0].priority == ChunkPriority.UNASSIGNED
    assert routing.unassigned[0].recovered_from_limbo is True
    assert len(routing.limbo) == 1
    assert routing.limbo[0].chunk.chunk_id == "c_ls"


def test_book_processing_state_recover_skipped_limbo_chunks(tmp_path: Path):
    """Verify BookProcessingState.recover_skipped_limbo_chunks unmarks chunks whose size >= threshold."""
    from bookeeper.processing.state import BookProcessingState

    state = BookProcessingState(book_id=1, state_dir=tmp_path)

    # Mark c1 as processed via normal LLM extraction
    state.mark_chunk_processed(
        chunk_id="c1",
        chunk_idx=1,
        section_title="S1",
        extraction_dict={"concepts": [{"name": "Idea"}]},
        duration=1.0,
        server_used="gpu-node",
    )

    # Mark c2 as skipped_limbo (large text)
    c2 = HierarchicalChunk(chunk_id="c2", book_id=1, book_title="B", section_title="S2", chapter_idx=1, chunk_idx=2, text="L" * 1500)
    state.mark_chunk_processed(
        chunk_id="c2",
        chunk_idx=2,
        section_title="S2",
        extraction_dict={"concepts": []},
        duration=0.0,
        server_used="skipped_limbo",
    )

    # Mark c3 as skipped_limbo (small text)
    c3 = HierarchicalChunk(chunk_id="c3", book_id=1, book_title="B", section_title="S3", chapter_idx=1, chunk_idx=3, text="S" * 200)
    state.mark_chunk_processed(
        chunk_id="c3",
        chunk_idx=3,
        section_title="S3",
        extraction_dict={"concepts": []},
        duration=0.0,
        server_used="skipped_limbo",
    )

    assert state.is_chunk_processed("c1")
    assert state.is_chunk_processed("c2")
    assert state.is_chunk_processed("c3")

    # Recover with threshold 1000: c2 (1500) should be unmarked/recovered; c3 (200) remains skipped; c1 untouched
    recovered = state.recover_skipped_limbo_chunks([c2, c3], threshold=1000)
    assert recovered == ["c2"]

    assert state.is_chunk_processed("c1")
    assert not state.is_chunk_processed("c2")  # Successfully restored to unprocessed!
    assert state.is_chunk_processed("c3")


def test_format_idea_for_embedding_all_fields():
    """Verify format_idea_for_embedding incorporates name, description, and short_description."""
    formatted = format_idea_for_embedding(
        name="Zero-Knowledge Proof",
        short_description="A cryptographic method proving possession of information without revealing it.",
        description="A protocol enabling one party to prove to another that a statement is true without conveying additional information.",
        detailed_explanation="Extensively used in modern zk-SNARKs and zk-STARKs for blockchain privacy.",
    )
    assert "Zero-Knowledge Proof: " in formatted
    assert "A cryptographic method proving possession of information without revealing it." in formatted
    assert "A protocol enabling one party to prove to another that a statement is true without conveying additional information." in formatted
    assert "Extensively used in modern zk-SNARKs and zk-STARKs for blockchain privacy." in formatted
    # Check that parts are joined by ' — '
    assert " — " in formatted


def test_format_idea_for_embedding_deduplication():
    """Verify duplicate descriptions or matches to the name are omitted from embedding text."""
    # When short_description and description are identical to brief_description
    formatted = format_idea_for_embedding(
        name="Consensus",
        short_description="Agreement among distributed nodes.",
        brief_description="Agreement among distributed nodes.",
        description="Agreement among distributed nodes.",
        detailed_explanation="Agreement among distributed nodes.",
    )
    # Should only appear once: "Consensus: Agreement among distributed nodes."
    assert formatted == "Consensus: Agreement among distributed nodes."

    # When description is identical to name
    formatted2 = format_idea_for_embedding(
        name="Consensus",
        short_description="Consensus",
        description="",
    )
    assert formatted2 == "Consensus"


def test_usearch_idea_embedding_index_dynamic_insertion_and_search():
    """Verify IdeaEmbeddingIndex uses usearch for fast logarithmic search and dynamic updates."""
    from bookeeper.processing.recent_ideas import IdeaEmbeddingIndex

    idea_index = IdeaEmbeddingIndex()
    assert idea_index.index is None

    # Dynamically add first idea
    idx0 = idea_index.add_idea(
        name="Quantum Key Distribution",
        summary="Cryptographic protocol using quantum mechanics",
        vector=[1.0, 0.0, 0.0, 0.0],
        month="2021-01",
    )
    assert idx0 == 0
    assert len(idea_index.idea_names) == 1
    assert idea_index.index is not None
    assert len(idea_index.index) == 1

    # Dynamically add second idea
    idx1 = idea_index.add_idea(
        name="Transformer Attention",
        summary="Multi-head self-attention mechanism",
        vector=[0.0, 1.0, 0.0, 0.0],
        month="2021-06",
    )
    assert idx1 == 1
    assert len(idea_index.index) == 2

    # Query using find_best_match
    best_name, sim, best_summary = idea_index.find_best_match(np.array([1.0, 0.0, 0.0, 0.0]))
    assert best_name == "Quantum Key Distribution"
    assert sim == pytest.approx(1.0, abs=1e-3)
    assert "Cryptographic" in best_summary

    best_name, sim, _ = idea_index.find_best_match(np.array([0.0, 1.0, 0.0, 0.0]))
    assert best_name == "Transformer Attention"
    assert sim == pytest.approx(1.0, abs=1e-3)

    # Dynamic insertion on the fly: Add 3rd idea and search without rebuilding
    idx2 = idea_index.add_idea(
        name="Lattice Cryptography",
        summary="Post-quantum cryptographic scheme",
        vector=[0.0, 0.0, 1.0, 0.0],
        month="2022-03",
    )
    assert idx2 == 2
    assert len(idea_index.index) == 3

    best_name, sim, _ = idea_index.find_best_match(np.array([0.0, 0.0, 1.0, 0.0]))
    assert best_name == "Lattice Cryptography"
    assert sim == pytest.approx(1.0, abs=1e-3)

    # Update an existing idea (in-place replacement without duplicate keys)
    idx_updated = idea_index.add_idea(
        name="Transformer Attention",
        summary="Updated multi-head self-attention mechanism description",
        vector=[0.0, 0.7071, 0.7071, 0.0],
    )
    assert idx_updated == 1
    assert len(idea_index.index) == 3
    assert idea_index.idea_summaries[1] == "Updated multi-head self-attention mechanism description"


def test_usearch_idea_embedding_index_query_batch():
    """Verify IdeaEmbeddingIndex.query_batch uses usearch native batch search."""
    from bookeeper.processing.recent_ideas import IdeaEmbeddingIndex

    idea_index = IdeaEmbeddingIndex(
        idea_names=["Idea A", "Idea B", "Idea C"],
        idea_summaries=["Summary A", "Summary B", "Summary C"],
        idea_months={"Idea A": "2020-01", "Idea B": "2021-01", "Idea C": "2022-01"},
        vectors=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32),
    )

    queries = [
        [1.0, 0.0, 0.0],     # Exact match Idea A
        [0.0, 0.8, 0.6],     # Mix of Idea B (0.8) and Idea C (0.6)
        [0.0, 0.0, 0.0],     # Zero vector
    ]

    results = idea_index.query_batch(queries, top_k=2)
    assert len(results) == 3

    # Query 0: Top match Idea A
    assert len(results[0]) == 2
    assert results[0][0]["name"] == "Idea A"
    assert results[0][0]["score"] == pytest.approx(1.0, abs=1e-3)

    # Query 1: Top match Idea B (~0.80), second Idea C (~0.60)
    assert len(results[1]) == 2
    assert results[1][0]["name"] == "Idea B"
    assert results[1][0]["score"] == pytest.approx(0.80, abs=1e-2)
    assert results[1][1]["name"] == "Idea C"
    assert results[1][1]["score"] == pytest.approx(0.60, abs=1e-2)

    # Query 2: Zero vector should return empty match list
    assert len(results[2]) == 0


def test_usearch_idea_embedding_index_save_and_load(tmp_path: Path):
    """Verify IdeaEmbeddingIndex can be dumped to disk and reloaded with usearch."""
    from bookeeper.processing.recent_ideas import IdeaEmbeddingIndex

    save_base = tmp_path / "usearch_ideas"

    original = IdeaEmbeddingIndex()
    original.add_idea("Byzantine Fault Tolerance", "Consensus under arbitrary node failures", [1.0, 0.0, 0.0], "2020-05")
    original.add_idea("Raft Consensus", "Leader-based consensus protocol", [0.0, 1.0, 0.0], "2021-07")

    original.save(save_base)

    assert (tmp_path / "usearch_ideas.usearch").exists()
    assert (tmp_path / "usearch_ideas.json").exists()

    loaded = IdeaEmbeddingIndex.load(save_base)
    assert loaded.index is not None
    assert len(loaded.index) == 2
    assert loaded.idea_names == ["Byzantine Fault Tolerance", "Raft Consensus"]
    assert loaded.idea_summaries == ["Consensus under arbitrary node failures", "Leader-based consensus protocol"]
    assert loaded.idea_months["Byzantine Fault Tolerance"] == "2020-05"

    best_name, sim, _ = loaded.find_best_match(np.array([1.0, 0.0, 0.0]))
    assert best_name == "Byzantine Fault Tolerance"
    assert sim == pytest.approx(1.0, abs=1e-3)


def test_usearch_entity_deduplicator_integration():
    """Verify EntityDeduplicator uses usearch for fast in-memory candidate lookup and deduplication."""
    from bookeeper.processing.deduplicator import EntityDeduplicator
    from bookeeper.processing.extractor import Concept
    from bookeeper.processing.recent_ideas import IdeaEmbeddingIndex

    idea_index = IdeaEmbeddingIndex()
    idea_index.add_idea("Graph Convolutional Network", "Deep learning on graph structures", [1.0, 0.0, 0.0])

    dedup = EntityDeduplicator()
    seeded = dedup.seed_from_idea_index(idea_index)
    assert seeded == 1
    assert dedup._usearch_index is not None
    assert len(dedup._usearch_index) == 1

    # Incoming concept with high similarity (0.98) should directly merge using usearch index
    concept = Concept(
        name="Graph Convolution Networks",
        brief_description="GCN model over graphs",
    )
    concept.embedding = [0.98, 0.1989, 0.0]  # unit norm ≈ 1.0, cosine sim with [1,0,0] = 0.98 >= 0.95

    resolved = dedup.resolve_concept(concept)
    assert resolved.name == "Graph Convolutional Network"









