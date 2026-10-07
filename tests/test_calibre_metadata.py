"""
Tests for Calibre SQLite trigger compatibility and multi-format book content sampling.
"""

import sqlite3
import tempfile
import zipfile
from pathlib import Path

import pytest

from bookeeper.calibre.client import (
    CalibreClient,
    _register_sqlite_functions,
    _sqlite_author_sort,
    _sqlite_title_sort,
)
from bookeeper.calibre.parser import BookParser


def test_sqlite_sort_helpers():
    assert _sqlite_title_sort("The Great Gatsby") == "Great Gatsby, The"
    assert _sqlite_title_sort("A Tale of Two Cities") == "Tale of Two Cities, A"
    assert _sqlite_title_sort("Метро 2033") == "Метро 2033"

    assert _sqlite_author_sort("Dmitry Glukhovsky") == "Glukhovsky, Dmitry"
    assert _sqlite_author_sort("Дмитрий Глуховский") == "Глуховский, Дмитрий"
    assert _sqlite_author_sort("SingleName") == "SingleName"


def test_calibre_sqlite_triggers_and_update(tmp_path: Path):
    """Verify that Calibre schema with triggers (title_sort, author_sort, uuid4) updates without error."""
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    _register_sqlite_functions(conn)

    # Create Calibre-like schema with real Calibre triggers
    conn.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            sort TEXT,
            author_sort TEXT,
            last_modified TIMESTAMP
        );

        CREATE TABLE authors (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL COLLATE NOCASE,
            sort TEXT COLLATE NOCASE,
            UNIQUE(name)
        );

        CREATE TABLE books_authors_link (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL,
            author INTEGER NOT NULL,
            UNIQUE(book, author)
        );

        CREATE TABLE comments (
            id INTEGER PRIMARY KEY,
            book INTEGER NOT NULL UNIQUE,
            text TEXT NOT NULL
        );

        CREATE TRIGGER books_update_trg
        AFTER UPDATE ON books
        BEGIN
            UPDATE books SET sort=title_sort(NEW.title)
            WHERE id=NEW.id AND OLD.title <> NEW.title;
        END;
    """)

    # Insert test book
    conn.execute("INSERT INTO books (id, title, sort, author_sort) VALUES (2, 'zubkov', 'zubkov', 'Unknown')")
    conn.commit()
    conn.close()

    # Use CalibreClient to update
    client = CalibreClient(library_path=str(tmp_path), calibredb_bin="non_existent_binary")
    success = client.update_metadata(
        book_id=2,
        title="Ассемблер. Для DOS, Windows и UNIX",
        authors=["Сергей Зубков"],
        comments="Исчерпывающее практическое руководство по программированию на ассемблере x86.",
    )
    assert success is True

    # Verify updated values in SQLite
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT b.id, b.title, b.sort, b.author_sort,
               (SELECT name FROM authors a JOIN books_authors_link bal ON a.id = bal.author WHERE bal.book = b.id) as author,
               (SELECT text FROM comments WHERE book = b.id) as comments
        FROM books b WHERE b.id = 2
    """)
    row = c.fetchone()
    conn.close()

    assert row["title"] == "Ассемблер. Для DOS, Windows и UNIX"
    assert row["sort"] == "Ассемблер. Для DOS, Windows и UNIX"
    assert row["author_sort"] == "Зубков, Сергей"
    assert row["author"] == "Сергей Зубков"
    assert "программированию" in row["comments"]


def test_rtf_sample_content_extraction(tmp_path: Path):
    """Verify BookParser.sample_content handles RTF with Cyrillic encoding and metadata headers."""
    rtf_content = b"""{\\rtf1\\ansi\\ansicpg1251
{\\info
{\\title \\'cc\\'e5\\'f2\\'f0\\'ee 2033}
{\\author \\'c4\\'ec\\'e8\\'f2\\'f0\\'e8\\'e9 \\'c3\\'eb\\'f3\\'f5\\'ee\\'e2\\'f1\\'ea\\'e8\\'e9}
}
\\pard
\\s1 \\qc \\'c4\\'ec\\'e8\\'f2\\'f0\\'e8\\'e9 \\'c3\\'eb\\'f3\\'f5\\'ee\\'e2\\'f1\\'ea\\'e8\\'e9\\par
\\s1 \\qc \\'cc\\'e5\\'f2\\'f0\\'ee 2033\\par
\\s12 \\qj 2033 \\'e3\\'ee\\'e4. \\'c2\\'e5\\'f1\\'fc \\'ec\\'e8\\'f0 \\'eb\\'e5\\'e6\\'e8\\'f2 \\'e2 \\'f0\\'f3\\'e8\\'ed\\'e0\\'f5.
}"""
    rtf_file = tmp_path / "2033 - .rtf"
    rtf_file.write_bytes(rtf_content)

    sample, hint = BookParser.sample_content(tmp_path)
    assert hint == "2033 - .rtf"
    assert sample is not None
    assert "Метро 2033" in sample
    assert "Дмитрий Глуховский" in sample
    assert "руинах" in sample


def test_zip_sample_content_extraction(tmp_path: Path):
    """Verify BookParser.sample_content handles ZIP archives containing technical documents."""
    zip_file = tmp_path / "zubkov - Unknown.zip"
    with zipfile.ZipFile(zip_file, "w") as z:
        z.writestr("zubkov.djvu", b"dummy djvu data")

    sample, hint = BookParser.sample_content(tmp_path)
    assert "zubkov - Unknown.zip" in hint
    assert "zubkov.djvu" in hint
    assert "zubkov.djvu" in sample
