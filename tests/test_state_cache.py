"""
Unit and integration tests for local book state cache (BookStateCache).
Verifies:
1. External state cache directory is never touched/purged during cleanups or start-from-scratch.
2. In case of chunking, chunks are reused if present in the state file or cache directory, falling back to standard chunking if absent.
3. In case of raw analysis, chunk extractions are reused from the state file, falling back to standard LLM extraction if absent.
4. Proper statistics are tracked and printed for reused cache items.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.config import Settings
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.state import BookProcessingState, BookStateCache


runner = CliRunner()


def test_book_state_cache_initialization_and_indexing(tmp_path: Path):
    """Verify BookStateCache indexes files by book_id and book_title."""
    cache_dir = tmp_path / "historical_cache"
    cache_dir.mkdir()

    # Empty cache dir
    cache = BookStateCache(cache_dir)
    assert cache.is_enabled()
    assert len(cache._id_to_file) == 0

    # Non-existent cache dir
    bad_cache = BookStateCache(tmp_path / "does_not_exist")
    assert not bad_cache.is_enabled()

    # Create dummy state file for book 42
    s42 = {
        "version": 1,
        "book_id": "42",
        "book_title": "The Hitchhiker's Guide to the Galaxy",
        "total_chunks": 10,
        "status": "fully_indexed",
        "processed_chunks": {
            "b42_c1": {
                "chunk_id": "b42_c1",
                "chunk_idx": 1,
                "section_title": "Chapter 1",
                "status": "extracted",
                "duration": 1.2,
                "server_used": "gpu-node-1",
                "extraction": {
                    "concepts": [
                        {
                            "name": "Babel Fish",
                            "brief_description": "Universal translator organism",
                            "detailed_explanation": "Performs real-time telepathic translation",
                            "category": "Technology",
                            "weight": 8,
                        }
                    ]
                },
            }
        },
    }
    (cache_dir / "book_42_state.json").write_text(json.dumps(s42), encoding="utf-8")

    # Re-index
    cache = BookStateCache(cache_dir)
    assert cache.is_enabled()
    assert "42" in cache._id_to_file

    # Lookup by ID
    loaded = cache.get_cached_state(42)
    assert loaded is not None
    assert loaded["book_id"] == "42"

    # Lookup by title
    loaded_by_title = cache.get_cached_state(999, book_title="The Hitchhiker's Guide to the Galaxy")
    assert loaded_by_title is not None
    assert loaded_by_title["book_id"] == "42"


def test_book_state_cache_chunks_reuse_embedded_and_companion(tmp_path: Path):
    """Verify get_cached_chunks reuses chunks embedded in state file or companion chunks file."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # 1. Test embedded chunks in state file
    chunk1 = HierarchicalChunk(
        chunk_id="b1_c1",
        book_id=1,
        book_title="Test Book",
        section_title="Intro",
        chapter_idx=1,
        chunk_idx=1,
        text="Sample chunk text content.",
        char_count=26,
    )
    s1 = {
        "book_id": "1",
        "book_title": "Test Book",
        "total_chunks": 1,
        "status": "fully_indexed",
        "chunks": [chunk1.model_dump()],
        "processed_chunks": {},
    }
    (cache_dir / "book_1_state.json").write_text(json.dumps(s1), encoding="utf-8")

    cache = BookStateCache(cache_dir)
    chunks = cache.get_cached_chunks(1, "Test Book")
    assert chunks is not None
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "b1_c1"
    assert chunks[0].text == "Sample chunk text content."
    assert cache.reused_chunks_count == 1

    # 2. Test companion chunks file (e.g. book_2_chunks.json)
    chunk2 = HierarchicalChunk(
        chunk_id="b2_c1",
        book_id=2,
        book_title="Second Book",
        section_title="Chapter 1",
        chapter_idx=1,
        chunk_idx=1,
        text="Second book text.",
        char_count=17,
    )
    (cache_dir / "book_2_chunks.json").write_text(
        json.dumps({"book_id": 2, "chunks": [chunk2.model_dump()]}), encoding="utf-8"
    )
    s2 = {
        "book_id": "2",
        "book_title": "Second Book",
        "total_chunks": 1,
        "status": "fully_indexed",
        "processed_chunks": {},
    }
    (cache_dir / "book_2_state.json").write_text(json.dumps(s2), encoding="utf-8")

    cache = BookStateCache(cache_dir)
    chunks2 = cache.get_cached_chunks(2, "Second Book")
    assert chunks2 is not None
    assert len(chunks2) == 1
    assert chunks2[0].chunk_id == "b2_c1"
    assert cache.reused_chunks_count == 1

    # 3. Fallback: missing book returns None
    assert cache.get_cached_chunks(999, "Missing") is None


def test_book_state_cache_raw_analysis_reuse_and_state_population(tmp_path: Path):
    """Verify populate_state_from_cache seeds raw extractions without touching cache files."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    s = {
        "book_id": "10",
        "book_title": "Distributed Systems",
        "total_chunks": 2,
        "status": "fully_indexed",
        "processed_chunks": {
            "b10_c1": {
                "chunk_id": "b10_c1",
                "chunk_idx": 1,
                "section_title": "Consensus",
                "status": "extracted",
                "duration": 0.95,
                "server_used": "ollama-node-1",
                "extraction": {
                    "concepts": [
                        {
                            "name": "Paxos",
                            "brief_description": "Consensus protocol",
                            "detailed_explanation": "Agreement algorithm across distributed network",
                            "category": "Protocol",
                            "weight": 9,
                        }
                    ]
                },
            }
        },
    }
    cache_file = cache_dir / "book_10_state.json"
    cache_file.write_text(json.dumps(s), encoding="utf-8")
    mtime_before = cache_file.stat().st_mtime_ns

    cache = BookStateCache(cache_dir)

    # Active output state directory
    out_dir = tmp_path / "output" / ".book_states"
    active_state = BookProcessingState(book_id=10, book_title="Distributed Systems", state_dir=out_dir)
    assert not active_state.is_chunk_processed("b10_c1")

    # Populate active state from cache
    seeded = cache.populate_state_from_cache(active_state)
    assert seeded == 1
    assert active_state.is_chunk_processed("b10_c1")
    rec = active_state.get_processed_chunk("b10_c1")
    assert rec is not None
    assert rec["extraction"]["concepts"][0]["name"] == "Paxos"

    # Verify cache file itself was NOT modified (remains 100% read-only)
    assert cache_file.stat().st_mtime_ns == mtime_before

    # Verify stats
    st = cache.stats()
    assert st["reused_books_count"] == 1
    assert st["reused_extractions_count"] == 1


def test_build_graph_start_from_scratch_never_touches_cache_dir(tmp_path: Path):
    """Verify that --from-scratch / clean deletes output .book_states but NEVER touches state-cache-dir."""
    output_dir = tmp_path / "output"
    active_states_dir = output_dir / ".book_states"
    active_states_dir.mkdir(parents=True)

    cache_dir = tmp_path / "protected_cache"
    cache_dir.mkdir(parents=True)

    # Put a state file in active output directory
    (active_states_dir / "book_1_state.json").write_text(json.dumps({"book_id": "1"}), encoding="utf-8")

    # Put a state file in cache directory
    cache_file = cache_dir / "book_1_state.json"
    cache_file.write_text(
        json.dumps({
            "book_id": "1",
            "book_title": "Cache Book",
            "processed_chunks": {
                "b1_c1": {
                    "chunk_id": "b1_c1",
                    "chunk_idx": 1,
                    "section_title": "S1",
                    "status": "extracted",
                    "duration": 1.0,
                    "server_used": "cached",
                    "extraction": {"concepts": []},
                }
            },
        }),
        encoding="utf-8",
    )

    mock_client = MagicMock()
    mock_client.is_available.return_value = True
    mock_client.list_books.return_value = []

    with patch("bookeeper.cli.get_book_client", return_value=mock_client), \
         patch("bookeeper.cli._get_effective_settings") as mock_settings:
        cfg = Settings(output_dir=str(output_dir), state_cache_dir=str(cache_dir))
        mock_settings.return_value = cfg

        # Run build-graph with --from-scratch and --state-cache-dir
        result = runner.invoke(
            app,
            [
                "build-graph",
                "--all",
                "--from-scratch",
                "--state-cache-dir",
                str(cache_dir),
                "--skip-warmup",
            ],
        )

        assert result.exit_code == 0

        # Output state directory was cleaned
        assert not (active_states_dir / "book_1_state.json").exists()

        # Cache directory was PRESERVED and UNTOUCHED
        assert cache_file.exists()
        assert "Preserving external state cache directory" in result.output or "Local State Cache Directory" in result.output


def test_build_graph_reuses_cached_chunks_and_raw_extractions(tmp_path: Path):
    """
    Integration test verifying build-graph:
    1. Loads chunks directly from cache state file (bypassing chunking).
    2. Reuses raw chunk extraction from cache state file (bypassing LLM).
    3. Prints cache statistics.
    """
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache_store"
    cache_dir.mkdir(parents=True)

    test_chunk = HierarchicalChunk(
        chunk_id="b100_c1",
        book_id=100,
        book_title="Cached Systems",
        section_title="Architecture",
        chapter_idx=1,
        chunk_idx=1,
        text="Cached chunk content for testing.",
        char_count=32,
    )

    cached_state_data = {
        "version": 1,
        "book_id": "100",
        "book_title": "Cached Systems",
        "total_chunks": 1,
        "status": "fully_indexed",
        "chunks": [test_chunk.model_dump()],
        "processed_chunks": {
            "b100_c1": {
                "chunk_id": "b100_c1",
                "chunk_idx": 1,
                "section_title": "Architecture",
                "status": "extracted",
                "duration": 0.5,
                "server_used": "gpu-node-cached",
                "extraction": {
                    "concepts": [
                        {
                            "name": "Cached Pattern",
                            "brief_description": "Pre-computed pattern",
                            "detailed_explanation": "Reused directly from local state cache",
                            "category": "Pattern",
                            "weight": 8,
                        }
                    ]
                },
            }
        },
    }
    (cache_dir / "book_100_state.json").write_text(json.dumps(cached_state_data), encoding="utf-8")

    mock_client = MagicMock()
    mock_client.is_available.return_value = True
    mock_client.list_books.return_value = [
        {"id": 100, "title": "Cached Systems", "authors": ["Author One"]}
    ]

    mock_llm_pool = MagicMock()
    mock_extractor = MagicMock()
    mock_extractor.pool = mock_llm_pool

    with patch("bookeeper.cli.get_book_client", return_value=mock_client), \
         patch("bookeeper.cli._get_effective_settings") as mock_settings, \
         patch("bookeeper.cli.KnowledgeExtractor", return_value=mock_extractor), \
         patch("bookeeper.processing.extractor.KnowledgeExtractor", return_value=mock_extractor):

        cfg = Settings(output_dir=str(output_dir), state_cache_dir=str(cache_dir))
        mock_settings.return_value = cfg

        result = runner.invoke(
            app,
            [
                "build-graph",
                "--book-id",
                "100",
                "--state-cache-dir",
                str(cache_dir),
                "--skip-warmup",
                "--no-recent-ideas",
                "--no-dual-pass",
            ],
        )

        assert result.exit_code == 0
        assert "Reused 1 cached atomic chunks from state cache" in result.output
        assert "Reused 1 raw chunk analyses from state cache" in result.output
        assert "Local State Cache Statistics" in result.output
        assert "Books Reused from Cache" in result.output

        # Verify LLM was NOT invoked because extraction was reused from cache
        assert mock_extractor.extract_section.call_count == 0
        assert mock_extractor.extract_sections_batch.call_count == 0
