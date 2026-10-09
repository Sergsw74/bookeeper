"""
Unit tests for CLI ETA remaining time formatters and custom ProgressColumn classes.
"""

import pytest
from rich.progress import Progress, Task
from rich.text import Text

from bookeeper.cli import (
    ChunkRemainingColumn,
    IngestionRemainingColumn,
    format_eta_days_h_m,
    format_eta_h_min_sec,
    format_eta_min_sec,
)


def test_format_eta_min_sec():
    """Verify remaining time formatting for chunking: (min, sec)."""
    assert format_eta_min_sec(None) == "--m --s"
    assert format_eta_min_sec(0) == "0m 00s"
    assert format_eta_min_sec(-10) == "0m 00s"
    assert format_eta_min_sec(45) == "0m 45s"
    assert format_eta_min_sec(60) == "1m 00s"
    assert format_eta_min_sec(135.8) == "2m 15s"
    assert format_eta_min_sec(3665) == "61m 05s"


def test_format_eta_h_min_sec():
    """Verify remaining time formatting for book analyses: (h, min, sec)."""
    assert format_eta_h_min_sec(None) == "--h --m --s"
    assert format_eta_h_min_sec(0) == "0h 00m 00s"
    assert format_eta_h_min_sec(-5) == "0h 00m 00s"
    assert format_eta_h_min_sec(45) == "0h 00m 45s"
    assert format_eta_h_min_sec(65) == "0h 01m 05s"
    assert format_eta_h_min_sec(3725) == "1h 02m 05s"
    assert format_eta_h_min_sec(7322.9) == "2h 02m 02s"


def test_format_eta_days_h_m():
    """Verify remaining time formatting for overall library processing: (days, h, m)."""
    assert format_eta_days_h_m(None) == "--d --h --m"
    assert format_eta_days_h_m(0) == "0d 0h 00m"
    assert format_eta_days_h_m(-1) == "0d 0h 00m"
    assert format_eta_days_h_m(59) == "0d 0h 00m"
    assert format_eta_days_h_m(120) == "0d 0h 02m"
    assert format_eta_days_h_m(3660) == "0d 1h 01m"
    assert format_eta_days_h_m(90000) == "1d 1h 00m"
    assert format_eta_days_h_m(195420) == "2d 6h 17m"


def test_chunk_remaining_column():
    """Verify ChunkRemainingColumn renders formatted ETA text."""
    col = ChunkRemainingColumn()

    with Progress(col) as progress:
        # Pending task
        task_id = progress.add_task("chunking", total=10, completed=0, eta_str="--m --s")
        task = progress.tasks[task_id]
        rendered = col.render(task)
        assert isinstance(rendered, Text)
        assert "ETA: --m --s" in rendered.plain

        # In-progress task
        progress.update(task_id, completed=5, eta_str="3m 12s")
        task = progress.tasks[task_id]
        rendered = col.render(task)
        assert "ETA: 3m 12s" in rendered.plain

        # Finished task
        progress.update(task_id, completed=10)
        task = progress.tasks[task_id]
        rendered = col.render(task)
        assert "ETA: 0m 00s" in rendered.plain


def test_ingestion_remaining_column():
    """Verify IngestionRemainingColumn renders both book (h, m, s) and library (d, h, m) ETAs."""
    col = IngestionRemainingColumn()

    with Progress(col) as progress:
        # Book task
        book_id = progress.add_task(
            "Book Task",
            total=50,
            completed=10,
            eta_type="book",
            eta_str="0h 14m 20s",
        )
        task_book = progress.tasks[book_id]
        rend_book = col.render(task_book)
        assert "ETA: 0h 14m 20s" in rend_book.plain

        # Overall task
        overall_id = progress.add_task(
            "Overall Task",
            total=100,
            completed=5,
            eta_type="overall",
            eta_str="2d 08h 15m",
        )
        task_overall = progress.tasks[overall_id]
        rend_overall = col.render(task_overall)
        assert "ETA: 2d 08h 15m" in rend_overall.plain

        # Finished book task
        progress.update(book_id, completed=50)
        task_book_fin = progress.tasks[book_id]
        rend_book_fin = col.render(task_book_fin)
        assert "ETA: 0h 00m 00s" in rend_book_fin.plain

        # Finished overall task
        progress.update(overall_id, completed=100)
        task_overall_fin = progress.tasks[overall_id]
        rend_overall_fin = col.render(task_overall_fin)
        assert "ETA: 0d 0h 00m" in rend_overall_fin.plain
