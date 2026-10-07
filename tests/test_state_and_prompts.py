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

        # Verify summary
        summary = tracker.summary("clean_metadata")
        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert summary["total_recorded"] == 2

        # Verify persistence across instances
        tracker2 = ProgressTracker(state_file)
        assert tracker2.is_completed("clean_metadata", 1)
        assert tracker2.get_completed_ids("clean_metadata") == {1}
        assert tracker2.get_failed_ids("clean_metadata") == {2}

        # Clear operation
        tracker2.clear("clean_metadata")
        assert not tracker2.is_completed("clean_metadata", 1)
        assert tracker2.get_completed_ids("clean_metadata") == set()
        assert tracker2.get_failed_ids("clean_metadata") == set()


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
