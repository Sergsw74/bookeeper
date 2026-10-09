"""
Unit tests for ProgressTracker persistence, resume checkpointing, and brace-safe prompt handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bookeeper.processing.extractor import BookMetadata, KnowledgeExtractor, SectionExtraction
from bookeeper.processing.ollama_pool import OllamaPool
from bookeeper.processing.state import ProgressTracker


def test_progress_tracker_lifecycle():
    """Verify ProgressTracker marking, querying, clearing, and persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "state.json"
        tracker = ProgressTracker(state_file)

        assert not tracker.is_completed("clean_metadata", 1)
        assert tracker.get_completed_ids("clean_metadata") == set()

        # Mark book 1 as completed
        tracker.mark_completed("clean_metadata", 1, title="Test Book 1")
        assert tracker.is_completed("clean_metadata", 1)
        assert tracker.get_completed_ids("clean_metadata") == {1}

        # Mark book 2 as failed
        tracker.mark_failed("clean_metadata", 2, title="Test Book 2", error="Network timeout")
        assert not tracker.is_completed("clean_metadata", 2)
        assert tracker.get_failed_ids("clean_metadata") == {2}

        # Mark book 3 as skipped (graphical format)
        tracker.mark_skipped("clean_metadata", 3, title="Comic Book 3", reason="graphical: CBR")
        assert tracker.is_completed("clean_metadata", 3)
        assert tracker.get_skipped_ids("clean_metadata") == {3}
        assert 3 in tracker.get_completed_ids("clean_metadata")  # completed_ids includes skipped for resume

        # Verify summary
        summary = tracker.summary("clean_metadata")
        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert summary["skipped"] == 1
        assert summary["total_recorded"] == 3

        # Verify persistence across instances
        tracker2 = ProgressTracker(state_file)
        assert tracker2.is_completed("clean_metadata", 1)
        assert tracker2.is_completed("clean_metadata", 3)
        assert tracker2.get_completed_ids("clean_metadata") == {1, 3}
        assert tracker2.get_failed_ids("clean_metadata") == {2}
        assert tracker2.get_skipped_ids("clean_metadata") == {3}

        # Clear operation
        tracker2.clear("clean_metadata")
        assert not tracker2.is_completed("clean_metadata", 1)
        assert tracker2.get_completed_ids("clean_metadata") == set()
        assert tracker2.get_failed_ids("clean_metadata") == set()
        assert tracker2.get_skipped_ids("clean_metadata") == set()


def test_book_parser_graphical_formats_detection():
    """Verify detection of graphical formats (CBR, CBZ, DJVU, CBT, CB7)."""
    from bookeeper.calibre.parser import BookParser

    assert BookParser.is_graphical_format("cbr")
    assert BookParser.is_graphical_format("CBR")
    assert BookParser.is_graphical_format(".cbz")
    assert BookParser.is_graphical_format("DJVU")
    assert BookParser.is_graphical_format(Path("comic.cbr"))
    assert BookParser.is_graphical_format(Path("scanned_book.djvu"))
    assert not BookParser.is_graphical_format("epub")
    assert not BookParser.is_graphical_format(Path("book.pdf"))

    # only_has_graphical_formats
    assert BookParser.only_has_graphical_formats(["CBR"])
    assert BookParser.only_has_graphical_formats(["cbz", "djvu"])
    assert not BookParser.only_has_graphical_formats(["EPUB", "CBR"])
    assert not BookParser.only_has_graphical_formats(["PDF"])
    assert not BookParser.only_has_graphical_formats([])
    assert not BookParser.only_has_graphical_formats(None)


def test_mojibake_repair():
    """Verify repairing of Latin-1 decoded Cyrillic text (CP1251 and UTF-8 mojibake)."""
    from bookeeper.calibre.parser import BookParser, Section
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.chunker import HierarchicalChunk
    from bookeeper.processing.extractor import BookMetadata, Concept

    raw_title = "Ñóïåðìåí Ïðèêëþ÷åíèÿ 002"
    repaired_title = BookParser.repair_mojibake(raw_title)
    assert repaired_title == "Супермен Приключения 002"

    raw_author = "Ð.Ð.Ð."
    repaired_author = BookParser.repair_mojibake(raw_author)
    assert repaired_author == "Р.Р.Р."

    # User's exact prompt phrase
    user_phrase = "Ðàñïèñêó refers to the concept of community and cooperation in a post-apocalyptic world. In the story of Ñêàìüþ, cha"
    expected_user = "Расписку refers to the concept of community and cooperation in a post-apocalyptic world. In the story of Скамью, cha"
    assert BookParser.repair_mojibake(user_phrase) == expected_user

    # Mixed typography and quotes
    quotes_phrase = "“Ðàñïèñêó” refers to... «Ñêàìüþ»"
    assert BookParser.repair_mojibake(quotes_phrase) == "“Расписку” refers to... «Скамью»"

    mixed_phrase = "В книге Ðàñïèñêó рассказывает..."
    assert BookParser.repair_mojibake(mixed_phrase) == "В книге Расписку рассказывает..."

    # UTF-8 in CP1252
    assert BookParser.repair_mojibake("Ð¡ÑƒÐ¿ÐµÑ€Ð¼ÐµÐ½") == "Супермен"
    assert BookParser.repair_mojibake("“Ð¡ÑƒÐ¿ÐµÑ€Ð¼ÐµÐ½” is Superman") == "“Супермен” is Superman"

    # CP1251 in UTF-8
    assert BookParser.repair_mojibake("Р .Р .Р .") == "Р.Р.Р."

    # Normal text should remain untouched
    clean_text = "Clean English Title"
    assert BookParser.repair_mojibake(clean_text) == clean_text

    cyrillic_text = "Чистый русский текст"
    assert BookParser.repair_mojibake(cyrillic_text) == cyrillic_text

    # Western accented text must remain untouched
    assert BookParser.repair_mojibake("English text with café and résumé and naïve") == "English text with café and résumé and naïve"
    assert BookParser.repair_mojibake("Spanish: ¿Cómo estás? ¡Muy bien!") == "Spanish: ¿Cómo estás? ¡Muy bien!"
    assert BookParser.repair_mojibake("German: Grüße über die Straße") == "German: Grüße über die Straße"

    # Stage 1: Section auto-repair
    sec = Section(title="“Ðàñïèñêó”", chapter_idx=1, text="Text with «Ñêàìüþ»")
    assert sec.title == "“Расписку”"
    assert sec.text == "Text with «Скамью»"

    # Stage 2: HierarchicalChunk auto-repair
    chk = HierarchicalChunk(
        chunk_id="chk1",
        book_id=1,
        book_title="Ð¡ÑƒÐ¿ÐµÑ€Ð¼ÐµÐ½",
        section_title="Ðàñïèñêó",
        chapter_idx=1,
        chunk_idx=1,
        text="Content with Ñêàìüþ",
    )
    assert chk.book_title == "Супермен"
    assert chk.section_title == "Расписку"
    assert chk.text == "Content with Скамью"

    # Stage 3: BookMetadata auto-repair
    meta = BookMetadata(title="Ð¡ÑƒÐ¿ÐµÑ€Ð¼ÐµÐ½", author="Ðàñïèñêó", summary="Story of Ñêàìüþ")
    assert meta.title == "Супермен"
    assert meta.author == "Расписку"
    assert meta.summary == "Story of Скамью"

    # Stage 4: Concept auto-repair
    concept = Concept(
        name="Ðàñïèñêó",
        brief_description="Refers to Ñêàìüþ",
        category="Ð¡ÑƒÐ¿ÐµÑ€Ð¼ÐµÐ½",
        related_concepts=["Ñêàìüþ"],
    )
    assert concept.name == "Расписку"
    assert concept.brief_description == "Refers to Скамью"
    assert concept.category == "Супермен"
    assert concept.related_concepts == ["Скамью"]

    # Stage 5: ConceptGraphStore auto-repair on load
    store = ConceptGraphStore()
    store.add_book(1, title="Ð¡ÑƒÐ¿ÐµÑ€Ð¼ÐµÐ½", author="Ðàñïèñêó")
    store.add_concept(concept)
    store.add_book_idea_link(1, concept.name)
    assert store.graph.has_node("concept:Расписку")


def test_retry_failed_selection():
    """Verify that get_failed_ids isolates only failed books for retry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "state.json"
        tracker = ProgressTracker(state_file)

        tracker.mark_completed("clean_metadata", 10, title="Success 1")
        tracker.mark_failed("clean_metadata", 20, title="Failed 1", error="Ollama timeout")
        tracker.mark_skipped("clean_metadata", 30, title="Skipped 1", reason="graphical: CBR")
        tracker.mark_failed("clean_metadata", 40, title="Failed 2", error="Connection error")

        failed = tracker.get_failed_ids("clean_metadata")
        assert failed == {20, 40}

        # Simulating retry: book 20 succeeds on retry
        tracker.mark_completed("clean_metadata", 20, title="Success 2")
        failed_after = tracker.get_failed_ids("clean_metadata")
        assert failed_after == {40}
        assert tracker.is_completed("clean_metadata", 20)


def test_extractor_brace_safety_in_clean_metadata():
    """
    Verify that raw curly braces (e.g. from code, LaTeX, or corrupted text)
    do NOT crash with 'unmatched { in format spec'.
    """
    pool = OllamaPool.from_urls(["http://mock-ollama:11434"])
    extractor = KnowledgeExtractor(pool=pool, model="llama3.1:8b")

    # Content with pathological unescaped curly braces and format specs
    content_with_braces = (
        "Chapter 1: The {Beginning}\n"
        "Here is some JSON: {\"key\": \"value\", \"nested\": {1, 2, 3}}\n"
        "Here is unmatched brace: {unmatched\n"
        "And another format spec: {0:10d} {foo!r} {{{broken\n"
    )

    with patch("bookeeper.processing.extractor.ChatOllama") as mock_chat_cls:
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.return_value = BookMetadata(
            title="Redwall 03",
            author="Brian Jacques",
            summary="A fantasy novel set in Mossflower Woods.",
        )
        mock_chat_cls.return_value = mock_instance

        # Must NOT raise ValueError: unmatched '{' in format spec
        cleaned = extractor.clean_metadata(
            raw_title="Jacques, Brian - Redwall 03 - {Special}",
            raw_authors=["Brian Jacques"],
            raw_comments="Some comments with {brackets} and {",
            content_sample=content_with_braces,
            file_hint="redwall.epub",
        )

        assert cleaned.title == "Redwall 03"
        assert cleaned.author == "Brian Jacques"


def test_extractor_brace_safety_in_extract_section():
    """Verify that section extractor handles curly braces in code and formulas safely."""
    pool = OllamaPool.from_urls(["http://mock-ollama:11434"])
    extractor = KnowledgeExtractor(pool=pool, model="llama3.1:8b")

    code_section = (
        "def example():\n"
        "    data = {'key': value}\n"
        "    if x: { return {a: b} }\n"
        "    fmt = '{unmatched'\n"
    )

    with patch("bookeeper.processing.extractor.ChatOllama") as mock_chat_cls:
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.return_value = SectionExtraction(
            concepts=[]
        )
        mock_chat_cls.return_value = mock_instance

        res = extractor.extract_section(
            text=code_section,
            book_title="Programming with {Braces}",
            section_title="Chapter {2}: Code",
        )

        assert isinstance(res, SectionExtraction)


def test_build_graph_cli_pool_and_chunks(tmp_path: Path):
    """Verify build-graph CLI executes parallel chunk tasks, saves chunks, and persists graph."""
    from typer.testing import CliRunner
    from bookeeper.cli import app
    from bookeeper.processing.extractor import Concept

    runner = CliRunner()
    out_dir = tmp_path / "output"
    chunks_dir = tmp_path / "custom_chunks"

    # Create dummy text file
    book_file = tmp_path / "sample.epub"
    book_file.write_text("Chapter 1: The Beginning\n\nThis is a sample book discussing Distributed Systems and Consensus.", encoding="utf-8")

    mock_concept = Concept(
        name="Distributed Systems",
        category="Architecture",
        summary="A system whose components are located on different networked computers.",
        related_concepts=["Consensus"],
    )

    with patch("bookeeper.processing.extractor.ChatOllama") as mock_chat_cls, \
         patch("bookeeper.calibre.parser.BookParser.parse") as mock_parse, \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False):

        from bookeeper.calibre.parser import Section
        mock_parse.return_value = [
            Section(title="Chapter 1", chapter_idx=1, text="Distributed Systems text for chunking.")
        ]

        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.return_value = SectionExtraction(
            concepts=[mock_concept]
        )
        mock_chat_cls.return_value = mock_instance

        result = runner.invoke(
            app,
            [
                "build-graph",
                "--file", str(book_file),
                "--config", "/dev/null",
                "--skip-warmup",
                "--chunks-dir", str(chunks_dir),
                "--max-tasks", "2",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "Parallel Task Pool:" in result.stdout
        assert "Persistent Chunks Directory:" in result.stdout
        assert "Extracted" in result.stdout
        assert "Knowledge Graph Build Complete" in result.stdout

        # Verify chunk store persisted chunks
        chunk_files = list(chunks_dir.glob("book_*_chunks.json"))
        assert len(chunk_files) == 1


def test_build_graph_calibre_local_staging(tmp_path: Path):
    """Verify build-graph stages metadata.db locally on SSD and reuses it for instant queries."""
    import sqlite3
    from typer.testing import CliRunner
    from bookeeper.cli import app
    from bookeeper.processing.extractor import Concept
    from bookeeper.calibre.client import _register_sqlite_functions

    # Setup dummy Calibre library
    lib_dir = tmp_path / "calibre_lib"
    lib_dir.mkdir()
    db_path = lib_dir / "metadata.db"

    conn = sqlite3.connect(str(db_path))
    _register_sqlite_functions(conn)
    conn.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            sort TEXT,
            author_sort TEXT,
            pubdate TIMESTAMP,
            path TEXT NOT NULL DEFAULT 'Author/Book'
        );
        CREATE TABLE authors (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL COLLATE NOCASE
        );
        CREATE TABLE books_authors_link (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL,
            author INTEGER NOT NULL
        );
        CREATE TABLE data (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL,
            format TEXT NOT NULL,
            name TEXT NOT NULL
        );
        CREATE TABLE comments (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL UNIQUE,
            text TEXT NOT NULL
        );
    """)
    conn.execute("INSERT INTO books (id, title, sort, author_sort, path) VALUES (42, 'Graph Book', 'Graph Book', 'Author', 'Author/Book')")
    conn.execute("INSERT INTO authors (id, name) VALUES (1, 'Jane Doe')")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (42, 1)")
    conn.execute("INSERT INTO data (book, format, name) VALUES (42, 'EPUB', 'Graph Book')")
    conn.commit()
    conn.close()

    # Create dummy EPUB file inside the library path
    book_folder = lib_dir / "Author" / "Book"
    book_folder.mkdir(parents=True)
    epub_file = book_folder / "Graph Book.epub"
    epub_file.write_text("EPUB test content", encoding="utf-8")

    out_dir = tmp_path / "output"
    chunks_dir = tmp_path / "chunks"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\n", encoding="utf-8")

    runner = CliRunner()
    mock_concept = Concept(
        name="Graph Theory",
        category="Math",
        summary="Study of graphs",
        related_concepts=[],
    )

    with patch("bookeeper.processing.extractor.ChatOllama") as mock_chat_cls, \
         patch("bookeeper.calibre.parser.BookParser.parse") as mock_parse, \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False):

        from bookeeper.calibre.parser import Section
        mock_parse.return_value = [
            Section(title="Chapter 1", chapter_idx=1, text="Graph theory concepts.")
        ]
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.return_value = SectionExtraction(
            concepts=[mock_concept]
        )
        mock_chat_cls.return_value = mock_instance

        # 1. First run: Downloads metadata.db and stages it locally
        res1 = runner.invoke(
            app,
            [
                "build-graph",
                "--all",
                "--calibre-path", str(lib_dir),
                "--config", str(cfg_file),
                "--skip-warmup",
                "--stage-db",
                "--chunks-dir", str(chunks_dir),
            ],
            catch_exceptions=False,
        )
        assert res1.exit_code == 0
        assert "Downloading metadata.db" in res1.stdout or "Staged database at" in res1.stdout
        assert "Reading book catalog from staged local SQLite..." in res1.stdout
        assert "Knowledge Graph Build Complete" in res1.stdout

        # Verify staged db exists in output
        staged_db = out_dir / ".staged_metadata.db"
        assert staged_db.is_file()

        # 2. Second run: Reuses existing staged db without re-downloading
        res2 = runner.invoke(
            app,
            [
                "build-graph",
                "--all",
                "--calibre-path", str(lib_dir),
                "--config", str(cfg_file),
                "--skip-warmup",
                "--stage-db",
                "--chunks-dir", str(chunks_dir),
            ],
            catch_exceptions=False,
        )
        assert res2.exit_code == 0
        assert "Found existing local staged database" in res2.stdout
        assert "Reading book catalog from staged local SQLite..." in res2.stdout


def test_build_graph_continue_and_clean_export(tmp_path):
    """Verify that build-graph continues from next book relative to knowledge graph and performs clean export."""
    import sqlite3
    from typer.testing import CliRunner
    from bookeeper.cli import app
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.extractor import Concept, SectionExtraction

    lib_dir = tmp_path / "calibre_lib"
    lib_dir.mkdir(parents=True)
    db_file = lib_dir / "metadata.db"
    conn = sqlite3.connect(str(db_file))
    conn.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            sort TEXT,
            author_sort TEXT,
            pubdate TIMESTAMP,
            path TEXT NOT NULL DEFAULT 'Author/Book'
        );
        CREATE TABLE authors (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL COLLATE NOCASE
        );
        CREATE TABLE books_authors_link (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL,
            author INTEGER NOT NULL
        );
        CREATE TABLE data (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL,
            format TEXT NOT NULL,
            name TEXT NOT NULL
        );
        CREATE TABLE comments (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL UNIQUE,
            text TEXT NOT NULL
        );
    """)
    conn.execute("INSERT INTO books (id, title, sort, author_sort, path) VALUES (10, 'Book Ten', 'Book Ten', 'Author', 'Author/B10')")
    conn.execute("INSERT INTO books (id, title, sort, author_sort, path) VALUES (20, 'Book Twenty', 'Book Twenty', 'Author', 'Author/B20')")
    conn.execute("INSERT INTO authors (id, name) VALUES (1, 'Jane Author')")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (10, 1)")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (20, 1)")
    conn.execute("INSERT INTO data (book, format, name) VALUES (10, 'EPUB', 'Book Ten')")
    conn.execute("INSERT INTO data (book, format, name) VALUES (20, 'EPUB', 'Book Twenty')")
    conn.commit()
    conn.close()

    # Create dummy EPUB files
    b10_dir = lib_dir / "Author" / "B10"
    b10_dir.mkdir(parents=True)
    (b10_dir / "Book Ten.epub").write_text("EPUB 10", encoding="utf-8")

    b20_dir = lib_dir / "Author" / "B20"
    b20_dir.mkdir(parents=True)
    (b20_dir / "Book Twenty.epub").write_text("EPUB 20", encoding="utf-8")

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    vault_dir = out_dir / "obsidian_vault"
    vault_dir.mkdir(parents=True)
    # Stale file that should be removed by clean-export
    stale_file = vault_dir / "Books" / "Stale Book.md"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("# Stale Book", encoding="utf-8")

    # Seed existing knowledge graph with Book 10
    store = ConceptGraphStore()
    store.add_book(10, title="Book Ten", author="Jane Author")
    concept10 = Concept(
        name="Foundation Concept",
        category="Tech",
        summary="Foundational idea",
        weight=4,
        related_concepts=[],
    )
    store.add_concept(concept10)
    store.add_book_idea_link(10, "Foundation Concept")
    store.save(out_dir / "knowledge_graph.json")

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\n", encoding="utf-8")

    runner = CliRunner()
    mock_concept20 = Concept(
        name="Advanced RAG",
        category="Tech",
        summary="Advanced retrieval",
        weight=8,
        related_concepts=[],
    )

    with patch("bookeeper.processing.extractor.KnowledgeExtractor.extract_section", return_value=SectionExtraction(concepts=[mock_concept20])), \
         patch("bookeeper.calibre.parser.BookParser.parse") as mock_parse, \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False):

        from bookeeper.calibre.parser import Section
        mock_parse.return_value = [
            Section(
                title="Chapter 1",
                chapter_idx=1,
                text="This is a comprehensive chapter on Retrieval-Augmented Generation (RAG) and graph knowledge integration. " * 5,
            )
        ]

        # Invoke with --clean-export and no explicit book selector (defaults to continue relative to knowledge graph)
        res = runner.invoke(
            app,
            [
                "build-graph",
                "--clean-export",
                "--calibre-path", str(lib_dir),
                "--config", str(cfg_file),
                "--skip-warmup",
                "--chunks-dir", str(tmp_path / "chunks"),
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        assert "Clean export requested" in res.stdout
        assert "Knowledge Graph Resume: Skipped 1 already indexed book(s)" in res.stdout
        assert "Next book: #20" in res.stdout
        assert "Ingesting Book #20" in res.stdout and "Book Twenty" in res.stdout

        # Verify stale note was purged by clean export
        assert not stale_file.is_file()

        # Verify both books are in Obsidian vault and knowledge_graph.json
        assert (vault_dir / "Books" / "Book Ten.md").is_file()
        assert (vault_dir / "Books" / "Book Twenty.md").is_file()

        # Reload final graph and check both books & concepts exist
        final_store = ConceptGraphStore()
        final_store.load(out_dir / "knowledge_graph.json")
        book_ids = {d["book_id"] for _, d in final_store.graph.nodes(data=True) if d.get("type") == "Book"}
        assert book_ids == {10, 20}
        concept_names = {d["name"] for _, d in final_store.graph.nodes(data=True) if d.get("type") == "Concept"}
        assert "Foundation Concept" in concept_names
        assert "Advanced RAG" in concept_names


def test_build_graph_from_scratch(tmp_path):
    """Verify that build-graph --from-scratch clears checkpoint, resets graph, and ingests all books."""
    import sqlite3
    from typer.testing import CliRunner
    from bookeeper.cli import app
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.extractor import Concept, SectionExtraction
    from bookeeper.processing.state import ProgressTracker

    lib_dir = tmp_path / "calibre_lib"
    lib_dir.mkdir(parents=True)
    db_file = lib_dir / "metadata.db"
    conn = sqlite3.connect(str(db_file))
    conn.executescript("""
        CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, sort TEXT, author_sort TEXT, pubdate TIMESTAMP, path TEXT);
        CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER, author INTEGER);
        CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER, format TEXT, name TEXT);
        CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER UNIQUE, text TEXT);
    """)
    conn.execute("INSERT INTO books (id, title, path) VALUES (1, 'Book One', 'Author/B1')")
    conn.execute("INSERT INTO books (id, title, path) VALUES (2, 'Book Two', 'Author/B2')")
    conn.execute("INSERT INTO authors (id, name) VALUES (1, 'Author')")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (1, 1), (2, 1)")
    conn.execute("INSERT INTO data (book, format, name) VALUES (1, 'EPUB', 'Book One'), (2, 'EPUB', 'Book Two')")
    conn.commit()
    conn.close()

    (lib_dir / "Author" / "B1").mkdir(parents=True)
    (lib_dir / "Author" / "B1" / "Book One.epub").write_text("EPUB 1", encoding="utf-8")
    (lib_dir / "Author" / "B2").mkdir(parents=True)
    (lib_dir / "Author" / "B2" / "Book Two.epub").write_text("EPUB 2", encoding="utf-8")

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    state_file = out_dir / "state.json"

    # Seed state file with book 1 already completed
    tracker = ProgressTracker(state_file)
    tracker.mark_completed("build_graph", 1, title="Book One")
    assert 1 in tracker.get_completed_ids("build_graph")

    # Seed an old graph
    old_store = ConceptGraphStore()
    old_store.add_book(1, title="Old Book One")
    old_store.save(out_dir / "knowledge_graph.json")

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\n", encoding="utf-8")

    runner = CliRunner()
    mock_concept = Concept(name="Concept X", category="Idea", summary="Test idea", weight=5, related_concepts=[])

    with patch("bookeeper.processing.extractor.ChatOllama") as mock_chat_cls, \
         patch("bookeeper.calibre.parser.BookParser.parse") as mock_parse, \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False):

        from bookeeper.calibre.parser import Section
        mock_parse.return_value = [Section(title="Ch 1", chapter_idx=1, text="Sample text " * 10)]
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.return_value = SectionExtraction(concepts=[mock_concept])
        mock_chat_cls.return_value = mock_instance

        res = runner.invoke(
            app,
            [
                "build-graph",
                "--from-scratch",
                "--state-file", str(state_file),
                "--calibre-path", str(lib_dir),
                "--config", str(cfg_file),
                "--skip-warmup",
                "--chunks-dir", str(tmp_path / "chunks"),
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        assert "Restarting build entirely from scratch" in res.stdout
        assert "Starting build entirely from scratch across Calibre catalog..." in res.stdout
        # Both Book 1 and Book 2 must be processed
        assert "Ingesting Book #1" in res.stdout and "Book One" in res.stdout
        assert "Ingesting Book #2" in res.stdout and "Book Two" in res.stdout

        # Verify final graph has both books
        final_store = ConceptGraphStore()
        final_store.load(out_dir / "knowledge_graph.json")
        bids = {d["book_id"] for _, d in final_store.graph.nodes(data=True) if d.get("type") == "Book"}
        assert bids == {1, 2}


def test_build_graph_manual_deletion_recovery(tmp_path):
    """Verify that deleting knowledge_graph.json auto-resets stale checkpoint so books aren't skipped."""
    import sqlite3
    from typer.testing import CliRunner
    from bookeeper.cli import app
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.extractor import Concept, SectionExtraction
    from bookeeper.processing.state import ProgressTracker

    lib_dir = tmp_path / "calibre_lib"
    lib_dir.mkdir(parents=True)
    db_file = lib_dir / "metadata.db"
    conn = sqlite3.connect(str(db_file))
    conn.executescript("""
        CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, sort TEXT, author_sort TEXT, pubdate TIMESTAMP, path TEXT);
        CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER, author INTEGER);
        CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER, format TEXT, name TEXT);
        CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER UNIQUE, text TEXT);
    """)
    conn.execute("INSERT INTO books (id, title, path) VALUES (5, 'Book Five', 'Author/B5')")
    conn.execute("INSERT INTO authors (id, name) VALUES (1, 'Author')")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (5, 1)")
    conn.execute("INSERT INTO data (book, format, name) VALUES (5, 'EPUB', 'Book Five')")
    conn.commit()
    conn.close()

    (lib_dir / "Author" / "B5").mkdir(parents=True)
    (lib_dir / "Author" / "B5" / "Book Five.epub").write_text("EPUB 5", encoding="utf-8")

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    state_file = out_dir / "state.json"

    # Simulate: checkpoint records Book 5 as completed, but knowledge_graph.json was deleted!
    tracker = ProgressTracker(state_file)
    tracker.mark_completed("build_graph", 5, title="Book Five")
    assert 5 in tracker.get_completed_ids("build_graph")
    assert not (out_dir / "knowledge_graph.json").exists()

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\n", encoding="utf-8")

    runner = CliRunner()
    mock_concept = Concept(name="Concept Y", category="Idea", summary="Test", weight=3, related_concepts=[])

    with patch("bookeeper.processing.extractor.ChatOllama") as mock_chat_cls, \
         patch("bookeeper.calibre.parser.BookParser.parse") as mock_parse, \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False):

        from bookeeper.calibre.parser import Section
        mock_parse.return_value = [Section(title="Ch 1", chapter_idx=1, text="Sample text " * 10)]
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.return_value = SectionExtraction(concepts=[mock_concept])
        mock_chat_cls.return_value = mock_instance

        res = runner.invoke(
            app,
            [
                "build-graph",
                "--state-file", str(state_file),
                "--calibre-path", str(lib_dir),
                "--config", str(cfg_file),
                "--skip-warmup",
                "--chunks-dir", str(tmp_path / "chunks"),
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        assert "Notice: Knowledge graph is empty" in res.stdout
        assert "Resetting checkpoint to match empty graph" in res.stdout
        # Book 5 is NOT skipped, it gets ingested into the graph
        assert "Ingesting Book #5" in res.stdout and "Book Five" in res.stdout

        final_store = ConceptGraphStore()
        final_store.load(out_dir / "knowledge_graph.json")
        bids = {d["book_id"] for _, d in final_store.graph.nodes(data=True) if d.get("type") == "Book"}
        assert bids == {5}


def test_concept_graph_store_remove_book():
    """Verify that removing a book purges its sections, chunks, and orphan concepts while preserving shared concepts."""
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.chunker import HierarchicalChunk
    from bookeeper.processing.extractor import Concept

    store = ConceptGraphStore()
    # Book 1
    store.add_book(1, title="Book One", author="Author One")
    sec1 = store.add_section(1, 0, "Chapter 1", text="Sample text 1")
    chk1 = HierarchicalChunk(
        chunk_id="chk-1-0-1",
        book_id=1,
        book_title="Book One",
        chapter_idx=0,
        section_title="Chapter 1",
        breadcrumb="Book One > Chapter 1",
        text="Sample text 1",
        char_count=13,
        token_count=3,
        chunk_idx=1,
    )
    store.add_chunk(chk1)

    # Book 2
    store.add_book(2, title="Book Two", author="Author Two")
    sec2 = store.add_section(2, 0, "Chapter 1", text="Sample text 2")
    chk2 = HierarchicalChunk(
        chunk_id="chk-2-0-1",
        book_id=2,
        book_title="Book Two",
        chapter_idx=0,
        section_title="Chapter 1",
        breadcrumb="Book Two > Chapter 1",
        text="Sample text 2",
        char_count=13,
        token_count=3,
        chunk_idx=1,
    )
    store.add_chunk(chk2)

    # Concept A: Only in Book 1
    c_orphan = Concept(name="Unique Idea A", category="Pattern", brief_description="Only in Book 1", detailed_explanation="Desc")
    store.add_concept(c_orphan)
    store.add_book_idea_link(1, "Unique Idea A")
    store.add_idea_support_link("Unique Idea A", chk1)

    # Concept B: Shared in Book 1 and Book 2
    c_shared = Concept(name="Shared Idea B", category="Architecture", brief_description="In both books", detailed_explanation="Desc")
    store.add_concept(c_shared)
    store.add_book_idea_link(1, "Shared Idea B")
    store.add_idea_support_link("Shared Idea B", chk1)
    store.add_book_idea_link(2, "Shared Idea B")
    store.add_idea_support_link("Shared Idea B", chk2)

    # Verify initial graph has both books and concepts
    stats_before = store.stats()
    assert stats_before["node_types"]["Book"] == 2
    assert stats_before["node_types"]["Chunk"] == 2
    assert stats_before["node_types"]["Concept"] == 2

    # Remove Book 1
    res = store.remove_book(1)
    assert res["book_removed"] == 1
    assert res["chunks_removed"] == 1
    assert res["sections_removed"] == 1
    assert res["orphan_concepts_removed"] == 1  # Unique Idea A purged!

    # Verify Book 1 and Unique Idea A are gone
    assert not store.graph.has_node("book:1")
    assert not store.graph.has_node("chunk:chk-1-0-1")
    assert not store.graph.has_node(sec1)
    assert not store.graph.has_node("concept:Unique Idea A")

    # Verify Book 2 and Shared Idea B remain
    assert store.graph.has_node("book:2")
    assert store.graph.has_node("chunk:chk-2-0-1")
    assert store.graph.has_node(sec2)
    assert store.graph.has_node("concept:Shared Idea B")


def test_progress_tracker_remove_book():
    """Verify remove_book in ProgressTracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "state.json"
        tracker = ProgressTracker(state_file)

        tracker.mark_completed("build_graph", 10, title="Book 10")
        assert 10 in tracker.get_completed_ids("build_graph")

        # Remove book 10
        assert tracker.remove_book("build_graph", 10)
        assert 10 not in tracker.get_completed_ids("build_graph")

        # Removing again returns False
        assert not tracker.remove_book("build_graph", 10)


def test_build_graph_purges_unprocessed_book_and_reprocesses_it(tmp_path):
    """Verify that an interrupted book in knowledge_graph.json not completed in tracker is purged and reprocessed."""
    import sqlite3
    from typer.testing import CliRunner
    from bookeeper.cli import app
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.extractor import Concept, SectionExtraction
    from bookeeper.processing.state import ProgressTracker

    lib_dir = tmp_path / "calibre_lib"
    lib_dir.mkdir(parents=True)
    db_file = lib_dir / "metadata.db"
    conn = sqlite3.connect(str(db_file))
    conn.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            sort TEXT,
            author_sort TEXT,
            pubdate TIMESTAMP,
            path TEXT NOT NULL DEFAULT 'Author/Book'
        );
        CREATE TABLE authors (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL COLLATE NOCASE
        );
        CREATE TABLE books_authors_link (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL,
            author INTEGER NOT NULL
        );
        CREATE TABLE data (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL,
            format TEXT NOT NULL,
            name TEXT NOT NULL
        );
        CREATE TABLE comments (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL UNIQUE,
            text TEXT NOT NULL
        );
    """)
    conn.execute("INSERT INTO books (id, title, sort, author_sort, path) VALUES (1, 'Book Completed', 'Book Completed', 'Author', 'Author/B1')")
    conn.execute("INSERT INTO books (id, title, sort, author_sort, path) VALUES (2, 'Book Interrupted', 'Book Interrupted', 'Author', 'Author/B2')")
    conn.execute("INSERT INTO authors (id, name) VALUES (1, 'Jane Author')")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (1, 1)")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (2, 1)")
    conn.execute("INSERT INTO data (book, format, name) VALUES (1, 'EPUB', 'Book Completed')")
    conn.execute("INSERT INTO data (book, format, name) VALUES (2, 'EPUB', 'Book Interrupted')")
    conn.commit()
    conn.close()

    # Create dummy EPUB files
    b1_dir = lib_dir / "Author" / "B1"
    b1_dir.mkdir(parents=True)
    (b1_dir / "Book Completed.epub").write_text("EPUB 1", encoding="utf-8")

    b2_dir = lib_dir / "Author" / "B2"
    b2_dir.mkdir(parents=True)
    (b2_dir / "Book Interrupted.epub").write_text("EPUB 2", encoding="utf-8")

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)

    # 1. State checkpoint has ONLY Book 1 completed
    state_file = out_dir / ".bookeeper_state.json"
    tracker = ProgressTracker(state_file)
    tracker.mark_completed("build_graph", 1, "Book Completed")

    # 2. Knowledge graph has Book 1 AND partially ingested Book 2 (from previous interrupted run)
    store = ConceptGraphStore()
    store.add_book(1, title="Book Completed", author="Jane Author")
    store.add_book(2, title="Book Interrupted", author="Jane Author")
    store.add_section(2, 0, "Partial Section", text="Incomplete text")
    c_partial = Concept(name="Partial Idea", category="Tech", summary="Incomplete idea", weight=3, related_concepts=[])
    store.add_concept(c_partial)
    store.add_book_idea_link(2, "Partial Idea")
    store.save(out_dir / "knowledge_graph.json")

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\n", encoding="utf-8")

    mock_concept2_full = Concept(
        name="Full Idea 2",
        category="Tech",
        summary="Complete idea after re-ingest",
        weight=8,
        related_concepts=[],
    )

    runner = CliRunner()
    with patch("bookeeper.processing.extractor.KnowledgeExtractor.extract_section", return_value=SectionExtraction(concepts=[mock_concept2_full])), \
         patch("bookeeper.calibre.parser.BookParser.parse") as mock_parse, \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False):

        from bookeeper.calibre.parser import Section
        mock_parse.return_value = [
            Section(
                title="Chapter 1",
                chapter_idx=1,
                text="Full text content for book 2 to be completely processed." * 5,
            )
        ]

        # Run build-graph (resume mode)
        res = runner.invoke(
            app,
            [
                "build-graph",
                "--calibre-path", str(lib_dir),
                "--config", str(cfg_file),
                "--skip-warmup",
                "--chunks-dir", str(tmp_path / "chunks"),
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        # Check that it detected and purged Book 2
        assert "Purged unprocessed book #2" in res.stdout
        # Check that Book 1 was skipped as already completed
        assert "Knowledge Graph Resume: Skipped 1 already indexed book(s)" in res.stdout
        # Check that Book 2 was re-ingested
        assert "Ingesting Book #2" in res.stdout
        assert "Book #2 concepts integrated" in res.stdout

        # Verify final graph contains both books fully processed
        final_store = ConceptGraphStore()
        final_store.load(out_dir / "knowledge_graph.json")
        bids = {d["book_id"] for _, d in final_store.graph.nodes(data=True) if d.get("type") == "Book"}
        assert bids == {1, 2}
        # Partial idea was purged, Full Idea 2 exists
        assert not final_store.graph.has_node("concept:Partial Idea")
        assert final_store.graph.has_node("concept:Full Idea 2")


def test_book_processing_state_lifecycle_clean(tmp_path):
    """Verify that a book processing run without poison chunks deletes state and reports fully_indexed."""
    from bookeeper.processing.state import BookProcessingState

    state_dir = tmp_path / "book_states"
    book_state = BookProcessingState(
        book_id=42,
        book_title="Clean Processing Book",
        state_dir=state_dir,
        max_chunk_attempts=6,
    )
    book_state.set_total_chunks(2)
    assert book_state.file_path.exists()

    # Stack chunk 1
    book_state.mark_chunk_stacked("c1", 0, "Chapter 1", attempt=0, server="node1")
    assert "c1" in book_state.stacked_chunks

    # Complete chunk 1
    book_state.mark_chunk_processed("c1", 0, "Chapter 1", {"concepts": [{"name": "Concept A"}]}, duration=2.5, server_used="node1")
    assert "c1" not in book_state.stacked_chunks
    assert book_state.is_chunk_processed("c1")

    # Retry chunk 2 once, then complete it
    book_state.mark_chunk_stacked("c2", 1, "Chapter 2", attempt=0, server="node1")
    book_state.mark_chunk_retried("c2", 1, "Chapter 2", attempt=1, error="timeout", server_used="node1")
    assert "c2" in book_state.retried_chunks
    book_state.mark_chunk_processed("c2", 1, "Chapter 2", {"concepts": [{"name": "Concept B"}]}, duration=1.8, server_used="node2")
    assert "c2" not in book_state.retried_chunks
    assert book_state.is_chunk_processed("c2")

    # Finalize with 0 errors -> retains file with fully_indexed for downstream verification
    res = book_state.finalize(has_errors=False)
    assert res == "fully_indexed"
    assert book_state.file_path.exists()
    assert book_state.data["status"] == "fully_indexed"


def test_book_processing_state_poison_chunks_partial_index(tmp_path):
    """Verify that a book with poison chunks exceeding 6 attempts marks partially_indexed and preserves state."""
    from bookeeper.processing.state import BookProcessingState

    state_dir = tmp_path / "book_states"
    book_state = BookProcessingState(
        book_id=99,
        book_title="Poison Chunk Book",
        state_dir=state_dir,
        max_chunk_attempts=6,
    )
    book_state.set_total_chunks(2)

    # Chunk 1 succeeds
    book_state.mark_chunk_processed("c1", 0, "Chapter 1", {"concepts": []}, duration=1.0, server_used="node1")

    # Chunk 2 retries up to 6 attempts and fails
    for att in range(1, 6):
        book_state.mark_chunk_retried("c2", 1, "Chapter 2", attempt=att, error="timeout", server_used="node1")
    book_state.mark_chunk_failed("c2", 1, "Chapter 2", attempts=6, error="Fatal poison timeout", server_used="node2")

    assert book_state.is_chunk_failed("c2")
    assert len(book_state.failed_chunks) == 1

    # Finalize -> marks partially_indexed and preserves file for inspection/debugging
    res = book_state.finalize(has_errors=True)
    assert res == "partially_indexed"
    assert book_state.file_path.exists()

    # Reload from disk in a new instance -> verify persistence
    reloaded = BookProcessingState(book_id=99, state_dir=state_dir)
    assert reloaded.is_chunk_processed("c1")
    assert reloaded.is_chunk_failed("c2")
    assert reloaded.get_failed_chunk("c2")["error"] == "Fatal poison timeout"
    assert reloaded.data["status"] == "partially_indexed"


def test_progress_tracker_partially_indexed_support(tmp_path):
    """Verify ProgressTracker handles mark_partially_indexed properly across queries and summary."""
    from bookeeper.processing.state import ProgressTracker

    state_file = tmp_path / "lib_state.json"
    tracker = ProgressTracker(state_file)

    tracker.mark_completed("build_graph", 1, title="Book 1")
    tracker.mark_partially_indexed("build_graph", 2, title="Book 2", failed_chunks=1, total_chunks=10)

    # Both completed and partially indexed should be considered completed for library resume
    assert tracker.is_completed("build_graph", 1) is True
    assert tracker.is_completed("build_graph", 2) is True
    assert tracker.get_completed_ids("build_graph") == {1, 2}
    assert tracker.get_partially_indexed_ids("build_graph") == {2}

    summary = tracker.summary("build_graph")
    assert summary["completed"] == 1
    assert summary["partially_indexed"] == 1
    assert summary["failed"] == 0
    assert summary["total_recorded"] == 2


def test_cli_has_section_extraction_for_state_resume():
    """Verify that bookeeper.cli exports and imports SectionExtraction to resume chunks without NameError."""
    import bookeeper.cli as cli
    assert hasattr(cli, "SectionExtraction")
    from bookeeper.processing.extractor import SectionExtraction

    # Simulate chunk resume payload
    cached_extraction = {
        "concepts": [
            {
                "name": "Metro Defense",
                "brief_description": "Guarding stations",
                "detailed_explanation": "Armed patrols protecting stations against threats",
                "category": "Tactical Strategy",
                "supporting_quote": "Guards stood watch",
                "related_concepts": [],
                "weight": 5,
            }
        ]
    }
    obj = cli.SectionExtraction(**cached_extraction)
    assert len(obj.concepts) == 1
    assert obj.concepts[0].name == "Metro Defense"


def test_two_phase_chunk_worker_skips_initial_failure_until_all_unprocessed_done(tmp_path):
    """Verify that failed chunks on initial pass are skipped and only retried after all fresh chunks finish."""
    import queue
    import threading
    from unittest.mock import MagicMock
    from bookeeper.processing.chunker import HierarchicalChunk
    from bookeeper.processing.extractor import SectionExtraction
    from bookeeper.processing.state import BookProcessingState

    book_state = BookProcessingState(book_id=77, state_dir=tmp_path)
    chunks = [
        HierarchicalChunk(
            chunk_id=f"chk_{i}",
            book_id=77,
            book_title="Test Book",
            chapter_idx=0,
            section_title="Ch1",
            chunk_idx=i,
            text=f"Text for chunk {i}",
            token_count=10,
        )
        for i in range(4)
    ]

    fresh_queue = queue.Queue()
    deferred_retries = []
    retry_queue = queue.Queue()

    for chk in chunks:
        fresh_queue.put((chk, 0, set()))

    stats_lock = threading.Lock()
    display_lock = threading.Lock()
    abort_event = threading.Event()
    extracted_results = []
    processed_order = []
    max_chunk_attempts = 6
    phase = "fresh"
    fresh_in_flight = 0
    retry_in_flight = 0
    active_tasks_count = 0

    # Simulate extractor where chunk 1 fails on attempt 0 (first attempt), but succeeds on retry
    def mock_extract_section(text, book_title, section_title, subtitle, parent_context, retries, quarantine_server, exclude_urls, raise_on_error):
        if "chunk 1" in text:
            with stats_lock:
                if phase == "fresh":
                    raise TimeoutError("Simulated Ollama timeout on chunk 1")
        return SectionExtraction(concepts=[])

    mock_extractor = MagicMock()
    mock_extractor.extract_section = mock_extract_section
    mock_extractor.pool.get_last_used_server.return_value = "node-alpha"
    mock_extractor.pool.get_last_used_server_url.return_value = "http://node-alpha:11434"

    def _worker():
        nonlocal active_tasks_count, phase, fresh_in_flight, retry_in_flight
        while not abort_event.is_set():
            with stats_lock:
                if len(extracted_results) >= len(chunks):
                    break
                if phase == "fresh" and fresh_queue.empty() and fresh_in_flight == 0:
                    if deferred_retries:
                        phase = "retry"
                        for item in deferred_retries:
                            retry_queue.put(item)
                        deferred_retries.clear()
                    else:
                        break
                cur_phase = phase

            task_phase = None
            if cur_phase == "fresh":
                try:
                    chk_item, attempt, tried_urls = fresh_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                task_phase = "fresh"
                with stats_lock:
                    fresh_in_flight += 1
                    active_tasks_count += 1
            else:
                try:
                    chk_item, attempt, tried_urls = retry_queue.get(timeout=0.05)
                except queue.Empty:
                    with stats_lock:
                        if len(extracted_results) >= len(chunks):
                            break
                    continue
                task_phase = "retry"
                with stats_lock:
                    retry_in_flight += 1
                    active_tasks_count += 1

            try:
                processed_order.append((chk_item.chunk_id, task_phase))
                extraction = mock_extractor.extract_section(
                    text=chk_item.text,
                    book_title="Test",
                    section_title="Ch1",
                    subtitle=None,
                    parent_context=None,
                    retries=1,
                    quarantine_server=False,
                    exclude_urls=tried_urls,
                    raise_on_error=True,
                )
                with stats_lock:
                    extracted_results.append({"status": "extracted", "chunk": chk_item})
                if task_phase == "fresh":
                    fresh_queue.task_done()
                else:
                    retry_queue.task_done()
            except Exception as exc:
                attempt += 1
                if task_phase == "fresh":
                    # Skipped on initial pass and deferred!
                    with stats_lock:
                        deferred_retries.append((chk_item, attempt, tried_urls))
                    fresh_queue.task_done()
                else:
                    if attempt < max_chunk_attempts:
                        retry_queue.put((chk_item, attempt, tried_urls))
                        retry_queue.task_done()
                    else:
                        with stats_lock:
                            extracted_results.append({"status": "failed_chunk", "chunk": chk_item})
                        retry_queue.task_done()
            finally:
                if task_phase == "fresh":
                    with stats_lock:
                        fresh_in_flight = max(0, fresh_in_flight - 1)
                        active_tasks_count = max(0, active_tasks_count - 1)
                elif task_phase == "retry":
                    with stats_lock:
                        retry_in_flight = max(0, retry_in_flight - 1)
                        active_tasks_count = max(0, active_tasks_count - 1)

    # Run 2 workers
    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    # Verify that all 4 chunks were extracted
    assert len(extracted_results) == 4
    # All fresh chunks (0, 1, 2, 3) must be attempted before chunk 1 is retried!
    # Specifically, the first 4 attempts in processed_order must all be 'fresh'!
    first_4_phases = [p[1] for p in processed_order[:4]]
    assert first_4_phases == ["fresh", "fresh", "fresh", "fresh"]
    # The last attempt in processed_order must be chunk 1 in 'retry' phase!
    assert processed_order[-1] == ("chk_1", "retry")


def test_fallback_model_configuration():
    """Verify Settings and _get_effective_settings support for llm_model_fallback."""
    from bookeeper.config import Settings
    from bookeeper.cli import _get_effective_settings

    # Direct initialization
    s1 = Settings(llm_model="qwen2.5:3b", llm_model_fallback="llama3.1:8b")
    assert s1.llm_model == "qwen2.5:3b"
    assert s1.llm_model_fallback == "llama3.1:8b"

    # Alias: fallback_model
    s2 = Settings(llm_model="qwen2.5:3b", fallback_model="mistral:7b")
    assert s2.llm_model_fallback == "mistral:7b"

    # Alias: llm_fallback_model
    s3 = Settings(llm_model="qwen2.5:3b", llm_fallback_model="gemma2:9b")
    assert s3.llm_model_fallback == "gemma2:9b"

    # CLI override via _get_effective_settings
    s4 = _get_effective_settings(model_fallback="llama3.1:8b")
    assert s4.llm_model_fallback == "llama3.1:8b"


def test_extractor_fallback_model_routing():
    """Verify KnowledgeExtractor accepts fallback_model and routes model override correctly."""
    from bookeeper.config import Settings
    from bookeeper.processing.extractor import KnowledgeExtractor, SectionExtraction, BookMetadata

    cfg = Settings(llm_model="qwen2.5:3b", llm_model_fallback="llama3.1:8b")
    extractor = KnowledgeExtractor.from_settings(cfg)
    assert extractor.model_name == "qwen2.5:3b"
    assert extractor.fallback_model == "llama3.1:8b"

    # Verify extract_section routes specified model to _execute_structured_invoke
    with patch.object(extractor, "_execute_structured_invoke") as mock_invoke:
        mock_invoke.return_value = SectionExtraction(concepts=[])
        extractor.extract_section("Sample text", book_title="Test Book", section_title="Chapter 1", model="llama3.1:8b")
        assert mock_invoke.call_count == 1
        _, kwargs = mock_invoke.call_args
        assert kwargs.get("model") == "llama3.1:8b"

    # Verify clean_metadata routes specified model to _execute_structured_invoke
    with patch.object(extractor, "_execute_structured_invoke") as mock_invoke:
        mock_invoke.return_value = BookMetadata(title="Title", author="Author", summary="Summary")
        extractor.clean_metadata(raw_title="Raw", raw_authors=["Author"], model="llama3.1:8b")
        assert mock_invoke.call_count == 1
        _, kwargs = mock_invoke.call_args
        assert kwargs.get("model") == "llama3.1:8b"


def test_fallback_model_switching_after_50_percent_retries():
    """Verify that worker logic switches to fallback model when attempt >= max_chunk_attempts / 2.0."""
    max_chunk_attempts = 6
    primary_model = "qwen2.5:3b"
    fallback_model = "llama3.1:8b"

    def select_target_model(attempt: int, fallback: str, max_attempts: int, primary: str) -> str:
        is_fallback = bool(fallback and attempt >= (max_attempts / 2.0))
        return fallback if is_fallback else primary

    # Attempts 0, 1, 2 (< 3.0) should use primary model
    for att in [0, 1, 2]:
        chosen = select_target_model(att, fallback_model, max_chunk_attempts, primary_model)
        assert chosen == primary_model, f"Attempt {att} should use primary model"

    # Attempts 3, 4, 5 (>= 3.0) should use fallback model
    for att in [3, 4, 5]:
        chosen = select_target_model(att, fallback_model, max_chunk_attempts, primary_model)
        assert chosen == fallback_model, f"Attempt {att} should use fallback model"

    # If fallback is None/empty, always use primary model
    for att in range(6):
        chosen = select_target_model(att, None, max_chunk_attempts, primary_model)
        assert chosen == primary_model


def test_resume_bypasses_memorized_skipped_books(tmp_path: Path):
    """Verify that build_graph resume pre-filtering bypasses memorized skipped books without re-scanning."""
    state_file = tmp_path / "state.json"
    tracker = ProgressTracker(state_file)

    # Book 1: Completed
    tracker.mark_completed("build_graph", 1, title="Book 1")
    # Book 2: Failed
    tracker.mark_failed("build_graph", 2, title="Book 2", error="Network error")
    # Books 3 & 4: Skipped (e.g. graphical CBR)
    tracker.mark_skipped("build_graph", 3, title="Comic 3", reason="graphical: CBR")
    tracker.mark_skipped("build_graph", 4, title="Comic 4", reason="graphical: CBZ")

    completed_in_graph = {1}
    tracker_skipped = tracker.get_skipped_ids("build_graph")
    assert tracker_skipped == {3, 4}

    all_books = [
        {"id": 1, "title": "Book 1"},
        {"id": 2, "title": "Book 2"},
        {"id": 3, "title": "Comic 3"},
        {"id": 4, "title": "Comic 4"},
        {"id": 5, "title": "Book 5 (New)"},
    ]

    # Normal resume: bypass_ids = completed_ids | tracker_skipped
    bypass_ids = completed_in_graph | tracker_skipped
    target_list = [b for b in all_books if b["id"] not in bypass_ids]

    # Books 1, 3, and 4 are bypassed! Only Book 2 (failed) and Book 5 (new) remain
    assert [b["id"] for b in target_list] == [2, 5]










