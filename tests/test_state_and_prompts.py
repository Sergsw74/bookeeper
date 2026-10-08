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
    from bookeeper.calibre.parser import BookParser

    raw_title = "Ñóïåðìåí Ïðèêëþ÷åíèÿ 002"
    repaired_title = BookParser.repair_mojibake(raw_title)
    assert repaired_title == "Супермен Приключения 002"

    raw_author = "Ð.Ð.Ð."
    repaired_author = BookParser.repair_mojibake(raw_author)
    assert repaired_author == "Р.Р.Р."

    # Normal text should remain untouched
    clean_text = "Clean English Title"
    assert BookParser.repair_mojibake(clean_text) == clean_text

    cyrillic_text = "Чистый русский текст"
    assert BookParser.repair_mojibake(cyrillic_text) == cyrillic_text


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


