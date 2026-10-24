"""
Unit tests for CLI ETA remaining time formatters and custom ProgressColumn classes.
"""

import pytest
from rich.progress import Progress, Task
from rich.text import Text

from bookeeper.cli import (
    ChunkRemainingColumn,
    IngestionRemainingColumn,
    compute_book_chunk_eta,
    compute_overall_library_eta,
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


def test_compute_book_chunk_eta():
    """Verify compute_book_chunk_eta computes accurate remaining time using real LLM chunk durations and concurrency."""
    # 1. Zero remaining chunks -> 0s
    rem_sec, eta_str = compute_book_chunk_eta(total_chunks=100, completed_chunks_count=100, durations=[3.0, 4.0])
    assert rem_sec == 0.0
    assert eta_str == "0h 00m 00s"

    # 2. No durations yet -> None, "--h --m --s"
    rem_sec, eta_str = compute_book_chunk_eta(total_chunks=100, completed_chunks_count=10, durations=[])
    assert rem_sec is None
    assert eta_str == "--h --m --s"

    # 3. 20 chunks remaining out of 100 (e.g. 80 chunks were skipped/assigned in 0s).
    # Real LLM chunk durations: 3.0s average.
    # Concurrency: 2 workers.
    # Expected remaining time: (20 * 3.0) / 2 = 30.0s -> "0h 00m 30s"
    durations = [3.0] * 10
    rem_sec, eta_str = compute_book_chunk_eta(
        total_chunks=100,
        completed_chunks_count=80,
        durations=durations,
        pool_concurrency=2,
    )
    assert rem_sec == 30.0
    assert eta_str == "0h 00m 30s"

    # 4. Concurrency capped at remaining chunks if rem_chunks < concurrency
    rem_sec, eta_str = compute_book_chunk_eta(
        total_chunks=100,
        completed_chunks_count=99,
        durations=[4.0],
        pool_concurrency=5,
    )
    # 1 chunk remaining with 4.0s average, concurrency effectively 1 -> 4.0s
    assert rem_sec == 4.0
    assert eta_str == "0h 00m 04s"


def test_compute_overall_library_eta():
    """Verify compute_overall_library_eta uses sliding window of at most last batch and includes batch overhead."""
    # 1. Total books 0 or all completed -> 0d 0h 00m
    avg_bt, rem_sec, eta_str, avg_str = compute_overall_library_eta(
        total_books=10,
        current_book_idx=10,
        current_book_rem_sec=0.0,
        current_book_elapsed=60.0,
        completed_book_records=[],
    )
    assert rem_sec == 0.0
    assert eta_str == "0d 0h 00m"

    # 2. First book in progress (no completed books yet):
    # Book 1 has run for 30s, has 60s remaining -> estimated book time = 90s.
    # 4 books remaining after Book 1 (total 5 books).
    # Expected total remaining = 60s + 4 * 90s = 420s -> 7 minutes ("0d 0h 07m")
    avg_bt, rem_sec, eta_str, avg_str = compute_overall_library_eta(
        total_books=5,
        current_book_idx=1,
        current_book_rem_sec=60.0,
        current_book_elapsed=30.0,
        completed_book_records=[],
    )
    assert avg_bt == 90.0
    assert rem_sec == 420.0
    assert eta_str == "0d 0h 07m"
    assert avg_str == "1m 30s"

    # 3. Sliding window of at most last book batch (e.g. batch_window = 3):
    # 5 completed books: [500s, 500s, 100s, 100s, 100s]
    # Last 3 books average = 100s.
    # Current book is #6 of 10.
    # Current book remaining = 20s. Remaining after current = 4 books.
    # Expected overall remaining = 20s + 4 * 100s = 420s.
    records = [
        {"book_id": 1, "effective_total": 500.0},
        {"book_id": 2, "effective_total": 500.0},
        {"book_id": 3, "effective_total": 100.0},
        {"book_id": 4, "effective_total": 100.0},
        {"book_id": 5, "effective_total": 100.0},
    ]
    avg_bt, rem_sec, eta_str, avg_str = compute_overall_library_eta(
        total_books=10,
        current_book_idx=6,
        current_book_rem_sec=20.0,
        current_book_elapsed=40.0,
        completed_book_records=records,
        batch_window=3,
    )
    assert avg_bt == 100.0
    assert rem_sec == 420.0
    assert eta_str == "0d 0h 07m"
    assert avg_str == "1m 40s"
