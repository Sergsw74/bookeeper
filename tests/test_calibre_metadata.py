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


def test_warmup_check_device_cpu(monkeypatch):
    """Verify that warmup_and_check_device correctly identifies CPU execution and warns."""
    from bookeeper.processing.extractor import KnowledgeExtractor
    extractor = KnowledgeExtractor(base_url="http://localhost:11434", model="llama3.1:8b")

    mock_ps = {
        "models": [
            {
                "name": "llama3.1:8b",
                "size": 5000000000,
                "size_vram": 0,
                "runner": "llamacpp",
            }
        ]
    }

    def mock_fetch(url, *args, **kwargs):
        class MockResp:
            def read(self):
                return b'{"models": [{"name": "llama3.1:8b", "size": 5000000000, "size_vram": 0, "runner": "llamacpp"}]}'
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return MockResp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_fetch)

    result = extractor.warmup_and_check_device()
    assert result["status"] == "ok"
    assert result["is_gpu"] is False
    assert result["device"] == "CPU"
    assert result["size_vram"] == 0
    assert "CPU" in result["warning"]


def test_warmup_check_device_gpu(monkeypatch):
    """Verify that warmup_and_check_device correctly identifies GPU execution."""
    from bookeeper.processing.extractor import KnowledgeExtractor
    extractor = KnowledgeExtractor(base_url="http://localhost:11434", model="llama3.1:8b")

    def mock_fetch(url, *args, **kwargs):
        class MockResp:
            def read(self):
                return b'{"models": [{"name": "llama3.1:8b", "size": 5000000000, "size_vram": 5000000000, "runner": "cuda"}]}'
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return MockResp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_fetch)

    result = extractor.warmup_and_check_device()
    assert result["status"] == "ok"
    assert result["is_gpu"] is True
    assert "GPU" in result["device"]
    assert result["vram_pct"] == 100.0
    assert result["warning"] is None


def test_calibre_database_staging_and_sync(tmp_path: Path):
    """Verify metadata.db staging, local updates, backup creation, and atomic sync-back."""
    remote_dir = tmp_path / "remote_calibre"
    remote_dir.mkdir()
    db_path = remote_dir / "metadata.db"

    conn = sqlite3.connect(str(db_path))
    _register_sqlite_functions(conn)
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
    """)
    conn.execute("INSERT INTO books (id, title, sort, author_sort) VALUES (1, 'Old Title', 'Old Title', 'Old Author')")
    conn.commit()
    conn.close()

    client = CalibreClient(library_path=str(remote_dir), calibredb_bin="non_existent_binary")
    assert client.is_staged is False
    assert client.active_db_path == db_path

    # Stage database locally
    local_staging_dir = tmp_path / "local_staging"
    staged_path = local_staging_dir / "staged.db"
    result_path = client.stage_database(staged_path)
    assert result_path == staged_path
    assert client.is_staged is True
    assert client.active_db_path == staged_path
    assert staged_path.is_file()

    # Update metadata on staged copy
    update_res = client.update_metadata(
        book_id=1,
        title="New Cleaned Title",
        authors=["Cleaned Author"],
        comments="Cleaned summary.",
    )
    assert update_res is True

    # Verify remote database has NOT changed yet
    conn_remote = sqlite3.connect(str(db_path))
    c_remote = conn_remote.cursor()
    c_remote.execute("SELECT title FROM books WHERE id = 1")
    assert c_remote.fetchone()[0] == "Old Title"
    conn_remote.close()

    # Verify staged copy has changed
    conn_staged = sqlite3.connect(str(staged_path))
    c_staged = conn_staged.cursor()
    c_staged.execute("SELECT title FROM books WHERE id = 1")
    assert c_staged.fetchone()[0] == "New Cleaned Title"
    conn_staged.close()

    # Sync staged DB back with backup enabled
    sync_res = client.sync_database(create_backup=True)
    assert sync_res is True

    # Verify backup exists
    backup_file = remote_dir / "metadata.db.bak"
    assert backup_file.is_file()

    conn_bak = sqlite3.connect(str(backup_file))
    c_bak = conn_bak.cursor()
    c_bak.execute("SELECT title FROM books WHERE id = 1")
    assert c_bak.fetchone()[0] == "Old Title"
    conn_bak.close()

    # Verify remote metadata.db is now updated
    conn_remote = sqlite3.connect(str(db_path))
    c_remote = conn_remote.cursor()
    c_remote.execute("SELECT title FROM books WHERE id = 1")
    assert c_remote.fetchone()[0] == "New Cleaned Title"
    conn_remote.close()

    # Cleanup staged file
    client.cleanup_staged(delete_file=True)
    assert client.is_staged is False
    assert client.staged_db_path is None
    assert not staged_path.exists()


def test_calibre_staged_session_context_manager(tmp_path: Path):
    """Verify staged_session context manager automatically stages, updates, and syncs back."""
    remote_dir = tmp_path / "session_calibre"
    remote_dir.mkdir()
    db_path = remote_dir / "metadata.db"

    conn = sqlite3.connect(str(db_path))
    _register_sqlite_functions(conn)
    conn.executescript("""
        CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, sort TEXT, author_sort TEXT, last_modified TIMESTAMP);
        CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT, sort TEXT);
        CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER, author INTEGER);
        CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER, text TEXT);
    """)
    conn.execute("INSERT INTO books (id, title) VALUES (1, 'Initial Title')")
    conn.commit()
    conn.close()

    client = CalibreClient(library_path=str(remote_dir), calibredb_bin="non_existent_binary")
    staged_target = tmp_path / "session_staged.db"

    with client.staged_session(staged_path=staged_target, create_backup=True, auto_sync=True):
        assert client.is_staged is True
        client.update_metadata(book_id=1, title="Updated in Context")

    # After exit:
    assert client.is_staged is False
    assert not staged_target.exists()

    # Check updated remote db
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("SELECT title FROM books WHERE id = 1")
    assert c.fetchone()[0] == "Updated in Context"
    conn.close()


def test_config_staging_settings():
    """Verify default Settings have stage_metadata_db and backup_metadata_db enabled."""
    from bookeeper.config import Settings
    s = Settings()
    assert s.stage_metadata_db is True
    assert s.backup_metadata_db is True
    assert s.resolved_staged_db_path.name == ".staged_metadata.db"


def test_sqlite_integrity_check_with_index_warnings(tmp_path: Path, monkeypatch):
    """Verify _verify_sqlite_integrity tolerates index warnings when enabled."""
    db_file = tmp_path / "index_warn.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT);")
    conn.execute("INSERT INTO books VALUES (1, 'Book');")
    conn.commit()
    conn.close()

    # Normal check passes
    assert CalibreClient._verify_sqlite_integrity(db_file) is True

    # Monkeypatch cursor to simulate Calibre index discrepancies
    orig_connect = sqlite3.connect
    class FakeCursor:
        def execute(self, sql):
            pass
        def fetchone(self):
            return (1,)
        def fetchall(self):
            return [("wrong # of entries in index sqlite_autoindex_authors_1",)]

    class FakeConn:
        def cursor(self):
            return FakeCursor()
        def close(self):
            pass

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: FakeConn())

    # With allow_index_warnings=True, it should succeed
    assert CalibreClient._verify_sqlite_integrity(db_file, allow_index_warnings=True) is True

    # With allow_index_warnings=False, it should raise ValueError
    with pytest.raises(ValueError, match="SQLite integrity check failed"):
        CalibreClient._verify_sqlite_integrity(db_file, allow_index_warnings=False)


def test_daemon_thread_pool_executor():
    """Verify DaemonThreadPoolExecutor creates daemon threads."""
    from bookeeper.cli import DaemonThreadPoolExecutor
    import time

    executor = DaemonThreadPoolExecutor(max_workers=2)
    fut = executor.submit(time.sleep, 0.05)
    for t in executor._threads:
        assert t.daemon is True
    fut.result()
    executor.shutdown(wait=False)

