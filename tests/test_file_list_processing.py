"""
Unit tests for file list parsing, matching, prioritization, skipping of already processed books,
and limit application in Bookkeeper build-graph and list-books workflows.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from bookeeper.cli import (
    app,
    parse_file_list,
    match_file_list_to_books,
    _get_effective_settings,
)
from bookeeper.config import Settings, get_settings
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.state import ProgressTracker
from bookeeper.processing.extractor import Concept, SectionExtraction
from bookeeper.calibre.parser import Section

runner = CliRunner()


def test_parse_file_list_basic(tmp_path: Path):
    fl = tmp_path / "files.txt"
    fl.write_text(
        """
# Header comment
2410.00001 - Paper One.pdf
   2410.00002 - Paper Two.pdf   

# Mid-file comment
/some/absolute/path/paper3.pdf
2410.00004
""",
        encoding="utf-8",
    )
    lines = parse_file_list(fl)
    assert lines == [
        "2410.00001 - Paper One.pdf",
        "2410.00002 - Paper Two.pdf",
        "/some/absolute/path/paper3.pdf",
        "2410.00004",
    ]


def test_parse_file_list_not_found():
    with pytest.raises(FileNotFoundError):
        parse_file_list(Path("/non/existent/path/files.txt"))


def test_match_file_list_to_books(tmp_path: Path):
    p1 = tmp_path / "2410.00001 - Alpha Paper.pdf"
    p2 = tmp_path / "2410.00002 - Beta Paper.pdf"
    p3 = tmp_path / "2410.00003 - Gamma Paper.pdf"
    p1.write_bytes(b"%PDF-1.4 dummy 1")
    p2.write_bytes(b"%PDF-1.4 dummy 2")
    p3.write_bytes(b"%PDF-1.4 dummy 3")

    all_books = [
        {
            "id": 10,
            "title": "Alpha Paper",
            "authors": ["Alice"],
            "path": str(p1),
            "arxiv_id": "2410.00001",
            "formats": ["PDF"],
        },
        {
            "id": 20,
            "title": "Beta Paper",
            "authors": ["Bob"],
            "path": str(p2),
            "arxiv_id": "2410.00002",
            "formats": ["PDF"],
        },
        {
            "id": 30,
            "title": "Gamma Paper",
            "authors": ["Carol"],
            "path": str(p3),
            "arxiv_id": "2410.00003",
            "formats": ["PDF"],
        },
    ]

    # Test order preservation: list Beta (20) first, then Alpha (10)
    lines = [
        "2410.00002 - Beta Paper.pdf",
        "2410.00001",  # bare arxiv id
        "9999.99999 - Nonexistent.pdf",
    ]

    matched, unmatched = match_file_list_to_books(lines, all_books)
    assert len(matched) == 2
    assert matched[0]["id"] == 20
    assert matched[1]["id"] == 10
    assert unmatched == ["9999.99999 - Nonexistent.pdf"]


def test_settings_file_list_configuration(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("file_list: my_list.txt\n", encoding="utf-8")
    s = get_settings(str(cfg_file))
    assert s.file_list == "my_list.txt"
    assert s.resolved_file_list == Path("my_list.txt").expanduser().resolve()

    # Check CLI override precedence
    effective = _get_effective_settings(config_path=str(cfg_file), file_list="override.txt")
    assert effective.file_list == "override.txt"


def test_list_books_with_file_list(tmp_path: Path):
    books_dir = tmp_path / "downloads"
    books_dir.mkdir(parents=True)
    p1 = books_dir / "2410.00001 - First.pdf"
    p2 = books_dir / "2410.00002 - Second.pdf"
    p3 = books_dir / "2410.00003 - Third.pdf"
    p1.write_bytes(b"%PDF-1.4 1")
    p2.write_bytes(b"%PDF-1.4 2")
    p3.write_bytes(b"%PDF-1.4 3")

    fl = tmp_path / "my_files.txt"
    fl.write_text(f"{p3.name}\n{p1.name}\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["list-books", "--books-dir", str(books_dir), "--file-list", str(fl)],
    )
    assert result.exit_code == 0
    assert "Filtered by file list: 2 of 2 matched" in result.output
    assert "2410.00003 - Third" in result.output
    assert "2410.00001 - First" in result.output


def test_build_graph_file_list_skipping_and_limit(tmp_path: Path):
    """
    Verify that build-graph with --file-list:
    1. Skips already processed books (present in checkpoint or knowledge graph).
    2. Respects user priority order from file list.
    3. Applies --limit strictly to unprocessed books from the file list.
    """
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True)

    # Create 4 test books
    p1 = downloads / "2410.00001 - One.pdf"
    p2 = downloads / "2410.00002 - Two.pdf"
    p3 = downloads / "2410.00003 - Three.pdf"
    p4 = downloads / "2410.00004 - Four.pdf"
    for p in [p1, p2, p3, p4]:
        p.write_bytes(b"%PDF-1.4 dummy")

    # File list orders: Two, One, Four, Three
    fl = tmp_path / "priority_files.txt"
    fl.write_text(
        f"{p2.name}\n{p1.name}\n{p4.name}\n{p3.name}\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\nbooks_dir: {downloads}\n", encoding="utf-8")

    # 1. Pre-populate Knowledge Graph with Book #1 ('One') as ALREADY PROCESSED
    kg_file = out_dir / "knowledge_graph.json"
    store = ConceptGraphStore()
    store.add_book(
        book_id=1,
        title="2410.00001 - One",
        author="Test Author",
        publication_date="2024-10-01",
    )
    store.save(kg_file)

    # Pre-populate ProgressTracker marking book #1 completed
    state_file = out_dir / ".bookeeper_state.json"
    tracker = ProgressTracker(state_file)
    tracker.mark_completed("build_graph", 1, "2410.00001 - One")

    from bookeeper.processing.extractor import KnowledgeExtractor
    from bookeeper.processing.deduplicator import EntityDeduplicator

    mock_concept = Concept(name="Concept Alpha", category="Idea", summary="Alpha idea", weight=5, related_concepts=[])

    def _mock_chunk_book(sections, book_id, book_title, **kwargs):
        return [
            HierarchicalChunk(
                chunk_id=f"chk_{book_id}_1",
                book_id=book_id,
                book_title=book_title,
                section_title="Ch 1",
                chapter_idx=1,
                chunk_idx=1,
                text="Alpha text content for testing purposes",
                char_count=40,
                word_count=6,
            )
        ]

    def _mock_extract(chunks, **kwargs):
        return {c.chunk_id: SectionExtraction(concepts=[mock_concept]) for c in chunks}

    with patch.object(KnowledgeExtractor, "from_settings") as mock_ext_cls, \
         patch.object(EntityDeduplicator, "from_settings") as mock_dedup_cls, \
         patch("bookeeper.processing.chunker.HierarchicalChunker.chunk_book", side_effect=_mock_chunk_book), \
         patch("bookeeper.calibre.parser.BookParser.parse") as mock_parse, \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False):

        mock_ext = MagicMock()
        mock_ext.extract_chunks.side_effect = _mock_extract
        mock_ext.pool.alive_nodes = ["http://localhost:11434"]
        mock_ext.pool.nodes = ["http://localhost:11434"]
        mock_ext.pool.get_last_used_server.return_value = "http://localhost:11434"
        mock_ext.pool.get_last_used_server_url.return_value = "http://localhost:11434"
        mock_ext_cls.return_value = mock_ext

        mock_dedup = MagicMock()
        mock_dedup.resolve_concept.side_effect = lambda c: c
        mock_dedup.resolve_concepts_batch.side_effect = lambda concepts, **kw: list(concepts)
        mock_dedup_cls.return_value = mock_dedup

        mock_parse.return_value = [Section(title="Ch 1", chapter_idx=1, text="Alpha text content " * 10)]

        # Run build-graph with --file-list and --limit 2
        # Expected behavior:
        # - Target candidates in file list order: #2 (Two), #1 (One), #4 (Four), #3 (Three)
        # - #1 (One) is detected as already processed -> SKIPPED!
        # - #2 (Two) is unprocessed -> processed (1st of limit 2)
        # - #4 (Four) is unprocessed -> processed (2nd of limit 2)
        # - Limit 2 reached -> stops, so #3 (Three) is deferred
        result = runner.invoke(
            app,
            [
                "build-graph",
                "--config", str(cfg_file),
                "--file-list", str(fl),
                "--state-file", str(state_file),
                "--limit", "2",
                "--skip-warmup",
                "--no-postprocess-ideas",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify already processed notice was printed for Book One
        assert "Skipped already processed article" in result.output
        assert "2410.00001 - One" in result.output

        # Verify starting pipeline was called for exactly 2 books (Two and Four)
        assert "Starting pipeline for 2 book(s)..." in result.output
        assert "2410.00002 - Two" in result.output
        assert "2410.00004 - Four" in result.output
        assert "2410.00003 - Three" not in result.output


def test_build_graph_file_list_prioritization_with_all(tmp_path: Path):
    """
    Verify that when both --file-list and --all are passed, file list books
    are placed first in their prioritized order, followed by the remaining catalog books.
    """
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True)
    p1 = downloads / "2410.00001 - PaperA.pdf"
    p2 = downloads / "2410.00002 - PaperB.pdf"
    p3 = downloads / "2410.00003 - PaperC.pdf"
    for p in [p1, p2, p3]:
        p.write_bytes(b"%PDF-1.4 dummy")

    # File list specifies PaperC first
    fl = tmp_path / "priority.txt"
    fl.write_text(f"{p3.name}\n", encoding="utf-8")

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\nbooks_dir: {downloads}\n", encoding="utf-8")

    from bookeeper.processing.extractor import KnowledgeExtractor
    from bookeeper.processing.deduplicator import EntityDeduplicator
    from bookeeper.processing.chunker import HierarchicalChunk

    mock_concept = Concept(name="Concept Alpha", category="Idea", summary="Alpha idea", weight=5, related_concepts=[])

    def _mock_chunk_book(sections, book_id, book_title, **kwargs):
        return [
            HierarchicalChunk(
                chunk_id=f"chk_{book_id}_1",
                book_id=book_id,
                book_title=book_title,
                section_title="Ch 1",
                chapter_idx=1,
                chunk_idx=1,
                text="Text content",
                char_count=12,
                word_count=2,
            )
        ]

    with patch.object(KnowledgeExtractor, "from_settings") as mock_ext_cls, \
         patch.object(EntityDeduplicator, "from_settings") as mock_dedup_cls, \
         patch("bookeeper.processing.chunker.HierarchicalChunker.chunk_book", side_effect=_mock_chunk_book), \
         patch("bookeeper.calibre.parser.BookParser.parse") as mock_parse, \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False):

        mock_ext = MagicMock()
        mock_ext.extract_chunks.side_effect = lambda chunks, **kwargs: {
            c.chunk_id: SectionExtraction(concepts=[mock_concept]) for c in chunks
        }
        mock_ext.pool.alive_nodes = ["http://localhost:11434"]
        mock_ext.pool.nodes = ["http://localhost:11434"]
        mock_ext.pool.get_last_used_server.return_value = "http://localhost:11434"
        mock_ext.pool.get_last_used_server_url.return_value = "http://localhost:11434"
        mock_ext_cls.return_value = mock_ext

        mock_dedup = MagicMock()
        mock_dedup.resolve_concept.side_effect = lambda c: c
        mock_dedup.resolve_concepts_batch.side_effect = lambda concepts, **kw: list(concepts)
        mock_dedup_cls.return_value = mock_dedup

        mock_parse.return_value = [Section(title="Ch 1", chapter_idx=1, text="Text content " * 5)]

        # Limit to 1 with --all and --file-list: PaperC MUST be the 1 book processed
        result = runner.invoke(
            app,
            [
                "build-graph",
                "--all",
                "--config", str(cfg_file),
                "--file-list", str(fl),
                "--limit", "1",
                "--skip-warmup",
                "--no-postprocess-ideas",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Matched 1 book(s) from file list (prioritized)" in result.output
        assert "Starting pipeline for 1 book(s)..." in result.output
        assert "2410.00003 - PaperC" in result.output
        assert "2410.00001 - PaperA" not in result.output


def test_cross_catalog_id_collision_prevention(tmp_path: Path):
    """
    Verify that an ID collision from a different catalog/library (e.g. ID 1392
    previously marked skipped as 'Trinity 009' in Calibre) does NOT cause a different
    book with the same ID (e.g. ID 1392 'La-JEPA') to be falsely skipped.
    """
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True)
    p = downloads / "2611.08156 - La-JEPA.pdf"
    p.write_bytes(b"%PDF-1.4 dummy")

    fl = tmp_path / "jepa.txt"
    fl.write_text(f"{p.name}\n", encoding="utf-8")

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"output_dir: {out_dir}\nbooks_dir: {downloads}\n", encoding="utf-8")

    # In checkpoint, record ID 1392 as skipped for an entirely different comic book 'Trinity 009'
    state_file = out_dir / ".bookeeper_state.json"
    tracker = ProgressTracker(state_file)
    tracker.mark_skipped("build_graph", 1, title="Trinity 009", reason="graphical: CBR")

    from bookeeper.processing.extractor import KnowledgeExtractor
    from bookeeper.processing.deduplicator import EntityDeduplicator
    from bookeeper.processing.chunker import HierarchicalChunk

    mock_concept = Concept(name="JEPA Concept", category="Idea", summary="JEPA summary", weight=5, related_concepts=[])

    def _mock_chunk_book(sections, book_id, book_title, **kwargs):
        return [
            HierarchicalChunk(
                chunk_id=f"chk_{book_id}_1",
                book_id=book_id,
                book_title=book_title,
                section_title="Ch 1",
                chapter_idx=1,
                chunk_idx=1,
                text="JEPA text content",
                char_count=17,
                word_count=3,
            )
        ]

    with patch.object(KnowledgeExtractor, "from_settings") as mock_ext_cls, \
         patch.object(EntityDeduplicator, "from_settings") as mock_dedup_cls, \
         patch("bookeeper.processing.chunker.HierarchicalChunker.chunk_book", side_effect=_mock_chunk_book), \
         patch("bookeeper.calibre.parser.BookParser.parse") as mock_parse, \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False):

        mock_ext = MagicMock()
        mock_ext.extract_chunks.side_effect = lambda chunks, **kwargs: {
            c.chunk_id: SectionExtraction(concepts=[mock_concept]) for c in chunks
        }
        mock_ext.pool.alive_nodes = ["http://localhost:11434"]
        mock_ext.pool.nodes = ["http://localhost:11434"]
        mock_ext.pool.get_last_used_server.return_value = "http://localhost:11434"
        mock_ext.pool.get_last_used_server_url.return_value = "http://localhost:11434"
        mock_ext_cls.return_value = mock_ext

        mock_dedup = MagicMock()
        mock_dedup.resolve_concept.side_effect = lambda c: c
        mock_dedup.resolve_concepts_batch.side_effect = lambda concepts, **kw: list(concepts)
        mock_dedup_cls.return_value = mock_dedup

        mock_parse.return_value = [Section(title="Ch 1", chapter_idx=1, text="JEPA text content " * 5)]

        result = runner.invoke(
            app,
            [
                "build-graph",
                "--config", str(cfg_file),
                "--file-list", str(fl),
                "--limit", "1",
                "--skip-warmup",
                "--no-postprocess-ideas",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # Verify it was NOT skipped as Trinity 009
        assert "Skipped previously skipped article" not in result.output
        assert "Starting pipeline for 1 book(s)..." in result.output
        assert "La-JEPA" in result.output


def test_unprocessed_books_have_no_ids_and_match_only_by_arxiv_or_title(tmp_path: Path):
    """
    Verify that:
    1. Unprocessed books have id=None.
    2. Checkpoint numeric IDs from previous runs (e.g. #1392) NEVER match an unprocessed file.
    3. Matching across systems uses exclusively arXiv ID and full name/title.
    """
    from bookeeper.cli import (
        extract_book_signatures,
        check_book_processed_or_skipped,
        build_tracker_and_graph_indices,
    )

    # 1. Signature extraction
    book_dict = {
        "id": None,
        "title": "La-JEPA: Compact Robot Policies with Latent World Modeling",
        "path": "/downloads/2611.08156 - La-JEPA_ Compact Robot Policies with Latent World Modeling.pdf",
    }
    aid, sigs = extract_book_signatures(book_dict)
    assert aid == "2611.08156"
    assert "la-jepa: compact robot policies with latent world modeling" in sigs

    # 2. Build mock tracker with #1392 skipped for Trinity 009
    state_file = tmp_path / ".bookeeper_state.json"
    tracker = ProgressTracker(state_file)
    tracker.mark_skipped("build_graph", 1392, title="Trinity 009", reason="graphical: CBR")
    tracker.mark_completed("build_graph", 1393, title="Some Other Comic")

    store = ConceptGraphStore()

    indices = build_tracker_and_graph_indices(store, tracker)
    (
        graph_book_arxiv_ids,
        graph_book_signatures,
        graph_books,
        tracker_completed_arxiv,
        tracker_completed_signatures,
        tracker_skipped_arxiv,
        tracker_skipped_signatures,
        tracker_records,
    ) = indices

    # Check unprocessed book
    is_proc, proc_reason, is_skip, skip_reason = check_book_processed_or_skipped(
        book_dict,
        graph_book_arxiv_ids,
        graph_book_signatures,
        graph_books,
        tracker_completed_arxiv,
        tracker_completed_signatures,
        tracker_skipped_arxiv,
        tracker_skipped_signatures,
        tracker_records,
    )
    assert not is_proc
    assert not is_skip

    # 3. If an article is completed in tracker by arXiv ID, it matches across systems
    tracker.mark_completed(
        "build_graph",
        5000,
        title="2611.08156 - La-JEPA",
        metadata={"arxiv_id": "2611.08156", "filename": "2611.08156.pdf"},
    )
    indices2 = build_tracker_and_graph_indices(store, tracker)
    is_proc2, proc_reason2, is_skip2, skip_reason2 = check_book_processed_or_skipped(
        book_dict,
        *indices2,
    )
    assert is_proc2
    assert "2611.08156" in proc_reason2



