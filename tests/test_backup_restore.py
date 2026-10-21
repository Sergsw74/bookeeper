"""
Tests for Bookeeper knowledge backup and restore commands and engine.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bookeeper.backup import (
    clean_knowledge_artifacts,
    collect_knowledge_files,
    create_backup,
    detect_archive_format,
    inspect_backup,
    restore_backup,
)
from bookeeper.cli import app

runner = CliRunner(env={"COLUMNS": "200"})


@pytest.fixture
def mock_output_dir(tmp_path: Path) -> Path:
    """Populate a temporary directory with standard bookeeper knowledge artifacts."""
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Knowledge Graph files
    kg_data = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "Book:1", "type": "Book", "book_id": 1, "title": "Test Book"},
            {"id": "Concept:AI", "type": "Concept", "name": "AI", "description": "Artificial Intelligence"},
        ],
        "links": [
            {"source": "Book:1", "target": "Concept:AI", "type": "DISCUSSES"},
        ],
    }
    (out / "knowledge_graph.json").write_text(json.dumps(kg_data), encoding="utf-8")
    (out / "knowledge_graph.graphml").write_text("<graphml></graphml>", encoding="utf-8")

    # 2. Checkpoint state
    state_data = {
        "build_graph": {
            "completed": {"1": {"title": "Test Book", "timestamp": "2026-10-10T12:00:00"}},
            "failed": {},
            "skipped": {},
        }
    }
    (out / ".bookeeper_state.json").write_text(json.dumps(state_data), encoding="utf-8")

    # 3. Obsidian vault
    obs = out / "obsidian_vault"
    obs.mkdir(parents=True, exist_ok=True)
    (obs / "00_Index.md").write_text("# Knowledge Index\n[[Test Book]]", encoding="utf-8")
    books_dir = obs / "Books"
    books_dir.mkdir(parents=True, exist_ok=True)
    (books_dir / "Test Book.md").write_text("# Test Book\nTags: #book", encoding="utf-8")

    # 4. Book states
    st_dir = out / ".book_states"
    st_dir.mkdir(parents=True, exist_ok=True)
    (st_dir / "book_1_state.json").write_text('{"book_id": 1, "total_chunks": 5}', encoding="utf-8")

    # 5. Chunks
    chk_dir = out / "chunks"
    chk_dir.mkdir(parents=True, exist_ok=True)
    (chk_dir / "book_1_chunks.json").write_text('{"book_id": 1, "chunks": []}', encoding="utf-8")

    # 6. Verification Report
    (out / "verification_report.json").write_text('{"stats": {"verified_ideas": 10}}', encoding="utf-8")

    # 7. Excluded files that must NOT be backed up
    (out / ".DS_Store").write_bytes(b"\x00\x00\x00\x01")
    (out / ".staged_metadata.db").write_bytes(b"SQLite format 3\x00")
    c_ingest = out / "calibre_ingest"
    c_ingest.mkdir(parents=True, exist_ok=True)
    (c_ingest / "1_test_book.epub").write_bytes(b"PK\x03\x04test_epub_bytes")

    return out


def test_detect_archive_format(tmp_path: Path):
    assert detect_archive_format(Path("backup.zip")) == "zip"
    assert detect_archive_format(Path("backup.7z")) == "7z"
    assert detect_archive_format(Path("BACKUP.ZIP")) == "zip"
    assert detect_archive_format(Path("BACKUP.7Z")) == "7z"

    # Magic byte detection
    test_zip = tmp_path / "magic_test_zip"
    test_zip.write_bytes(b"PK\x03\x04\x14\x00")
    assert detect_archive_format(test_zip) == "zip"

    test_7z = tmp_path / "magic_test_7z"
    test_7z.write_bytes(b"7z\xbc\xaf'\x1c")
    assert detect_archive_format(test_7z) == "7z"


def test_collect_knowledge_files(mock_output_dir: Path):
    files = collect_knowledge_files(mock_output_dir)
    rel_names = {arcname for _, arcname in files}

    # Verify required knowledge files are present
    assert "knowledge_graph.json" in rel_names
    assert "knowledge_graph.graphml" in rel_names
    assert ".bookeeper_state.json" in rel_names
    assert "verification_report.json" in rel_names
    assert "obsidian_vault/00_Index.md" in rel_names
    assert "obsidian_vault/Books/Test Book.md" in rel_names
    assert ".book_states/book_1_state.json" in rel_names
    assert "chunks/book_1_chunks.json" in rel_names

    # Verify excluded files are completely absent
    assert ".DS_Store" not in rel_names
    assert ".staged_metadata.db" not in rel_names
    assert not any("calibre_ingest" in name for name in rel_names)


def test_create_and_restore_zip(mock_output_dir: Path, tmp_path: Path):
    zip_dest = tmp_path / "test_backup.zip"
    res = create_backup(mock_output_dir, destination=zip_dest, format_type="zip")

    assert res["format"] == "zip"
    assert res["destination"] == zip_dest
    assert zip_dest.is_file()
    assert res["total_files"] == 8  # 8 knowledge files

    # Inspect
    info = inspect_backup(zip_dest)
    assert info["format"] == "zip"
    assert info["total_files"] == 8
    assert info["categories"]["graph"] == 2
    assert info["categories"]["obsidian"] == 2
    assert info["categories"]["book_states"] == 1
    assert info["categories"]["chunks"] == 1
    assert info["categories"]["checkpoint"] == 1

    # Restore into clean directory
    restore_target = tmp_path / "restored_zip"
    rest_res = restore_backup(zip_dest, target_dir=restore_target)

    assert rest_res["files_restored"] == 8
    assert (restore_target / "knowledge_graph.json").is_file()
    assert (restore_target / "obsidian_vault" / "00_Index.md").is_file()
    assert (restore_target / ".book_states" / "book_1_state.json").is_file()
    assert (restore_target / "chunks" / "book_1_chunks.json").is_file()
    assert not (restore_target / ".staged_metadata.db").exists()
    assert not (restore_target / "calibre_ingest").exists()


def test_create_and_restore_7z(mock_output_dir: Path, tmp_path: Path):
    sevenz_dest = tmp_path / "test_backup.7z"
    res = create_backup(mock_output_dir, destination=sevenz_dest, format_type="7z")

    assert res["format"] == "7z"
    assert res["destination"] == sevenz_dest
    assert sevenz_dest.is_file()
    assert res["total_files"] == 8

    # Inspect
    info = inspect_backup(sevenz_dest)
    assert info["format"] == "7z"
    assert info["total_files"] == 8
    assert info["categories"]["graph"] == 2

    # Restore into clean directory
    restore_target = tmp_path / "restored_7z"
    rest_res = restore_backup(sevenz_dest, target_dir=restore_target)

    assert rest_res["files_restored"] == 8
    assert (restore_target / "knowledge_graph.json").is_file()
    assert (restore_target / "obsidian_vault" / "Books" / "Test Book.md").is_file()
    assert (restore_target / ".bookeeper_state.json").is_file()


def test_restore_clean_mode(mock_output_dir: Path, tmp_path: Path):
    zip_dest = tmp_path / "backup.zip"
    create_backup(mock_output_dir, destination=zip_dest, format_type="zip")

    target = tmp_path / "dirty_output"
    target.mkdir()
    # Create stale files in dirty target
    (target / "knowledge_graph.json").write_text('{"stale": true}')
    stale_obs = target / "obsidian_vault"
    stale_obs.mkdir()
    (stale_obs / "OldNote.md").write_text("Old")
    (target / "other_unrelated_file.txt").write_text("Keep me")

    # Clean restore
    rest_res = restore_backup(zip_dest, target_dir=target, clean=True)
    assert rest_res["cleaned"]["files_removed"] >= 1
    assert rest_res["cleaned"]["dirs_removed"] >= 1

    # Verify stale note in obsidian_vault was wiped and replaced
    assert not (target / "obsidian_vault" / "OldNote.md").exists()
    assert (target / "obsidian_vault" / "00_Index.md").exists()
    assert (target / "knowledge_graph.json").exists()
    # Unrelated files not in KNOWLEDGE_FILES or KNOWLEDGE_DIR_NAMES remain intact
    assert (target / "other_unrelated_file.txt").read_text() == "Keep me"


def test_dry_run_restore(mock_output_dir: Path, tmp_path: Path):
    zip_dest = tmp_path / "backup.zip"
    create_backup(mock_output_dir, destination=zip_dest)

    target = tmp_path / "dry_run_target"
    res = restore_backup(zip_dest, target_dir=target, dry_run=True)

    assert res["dry_run"] is True
    assert not target.exists()
    assert res["inspection"]["total_files"] == 8


def test_cli_backup_and_restore(mock_output_dir: Path, tmp_path: Path):
    archive_dest = tmp_path / "cli_backup.zip"

    # 1. Run CLI backup
    res = runner.invoke(
        app,
        [
            "backup",
            str(archive_dest),
            "--output-dir",
            str(mock_output_dir),
        ],
    )
    assert res.exit_code == 0
    assert "Knowledge Backup Summary" in res.stdout
    assert archive_dest.is_file()

    # 2. Run CLI restore dry-run
    res_dry = runner.invoke(
        app,
        [
            "restore",
            str(archive_dest),
            "--dry-run",
        ],
    )
    assert res_dry.exit_code == 0
    assert "Archive Inspection" in res_dry.stdout
    assert "Dry Run" in res_dry.stdout

    # 3. Run CLI restore
    restore_dir = tmp_path / "cli_restored"
    res_restore = runner.invoke(
        app,
        [
            "restore",
            str(archive_dest),
            "--output-dir",
            str(restore_dir),
            "--clean",
        ],
    )
    assert res_restore.exit_code == 0
    assert "Knowledge Restoration Summary" in res_restore.stdout
    assert "Knowledge successfully restored" in res_restore.stdout
    assert (restore_dir / "knowledge_graph.json").is_file()
    assert (restore_dir / "obsidian_vault" / "00_Index.md").is_file()


def test_restore_backup_sanitizes_unclean_windows_filenames(tmp_path: Path):
    """
    Verify that an archive containing filenames with newlines, control chars,
    or characters illegal on Windows (e.g. from POSIX vault creation) can be
    safely restored on Windows by encoding unsupported characters via base64.
    """
    import zipfile
    from bookeeper.backup import (
        decode_safe_relative_path,
        encode_safe_relative_path,
        sanitize_archive_relative_path,
    )

    # Check sanitize_archive_relative_path helper directly
    raw_path = "obsidian_vault/Concepts/Complexity Class Σ 0\n2.md"
    safe = sanitize_archive_relative_path(raw_path)
    assert safe == "obsidian_vault/Concepts/@b64_Q29tcGxleGl0eSBDbGFzcyDOoyAwCjI.md"
    assert "\n" not in safe
    assert decode_safe_relative_path(safe) == raw_path

    trailing_dot_path = "obsidian_vault/Concepts/trailing dot. .md"
    safe_trailing = sanitize_archive_relative_path(trailing_dot_path)
    assert safe_trailing == "obsidian_vault/Concepts/@b64_dHJhaWxpbmcgZG90LiA.md"
    assert decode_safe_relative_path(safe_trailing) == trailing_dot_path

    # Create a mock zip with invalid Windows filenames
    archive_path = tmp_path / "legacy_windows_incompatible.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("obsidian_vault/Concepts/Complexity Class Σ 0\n2.md", "# Complexity Class\n")
        zf.writestr("obsidian_vault/Concepts/Matrix V r\nj.md", "# Matrix Note\n")
        zf.writestr("obsidian_vault/Concepts/test:invalid*char?.md", "# Invalid Chars\n")

    target_dir = tmp_path / "restored_safe"
    res = restore_backup(archive_path, target_dir=target_dir)

    assert res["files_restored"] == 3
    assert (target_dir / "obsidian_vault" / "Concepts" / "@b64_Q29tcGxleGl0eSBDbGFzcyDOoyAwCjI.md").is_file()
    assert (target_dir / "obsidian_vault" / "Concepts" / "@b64_TWF0cml4IFYgcgpq.md").is_file()
    assert (target_dir / "obsidian_vault" / "Concepts" / "@b64_dGVzdDppbnZhbGlkKmNoYXI_.md").is_file()


def test_sanitize_filename_windows_compatibility():
    """Verify sanitize_filename handles newlines, tabs, and invalid Windows filesystem chars."""
    from bookeeper.graph.exporters import sanitize_filename

    assert sanitize_filename("Complexity Class Σ 0\n2") == "Complexity Class Σ 0 2"
    assert sanitize_filename("Matrix V r\nj") == "Matrix V r j"
    assert sanitize_filename("Invalid: <name> *with? \"quotes\" | pipes") == "Invalid name with quotes pipes"
    assert sanitize_filename("Trailing dots and spaces. . ") == "Trailing dots and spaces"
    assert sanitize_filename("\tTabs and \r\n multiple   spaces\t") == "Tabs and multiple spaces"


@pytest.fixture
def mock_books_dir(tmp_path: Path) -> Path:
    """Populate a temporary source books directory with indexed and unindexed books."""
    bdir = tmp_path / "downloads"
    bdir.mkdir(parents=True, exist_ok=True)

    # 1. Indexed book matching "Test Book" from mock_output_dir
    book1 = bdir / "Test Book.pdf"
    book1.write_bytes(b"%PDF-1.4 test indexed book pdf contents")
    opf1 = bdir / "Test Book.opf"
    opf1.write_text("<package><metadata><title>Test Book</title></metadata></package>", encoding="utf-8")

    # 2. Unindexed book
    book2 = bdir / "Unrelated Other Book.epub"
    book2.write_bytes(b"PK\x03\x04test epub contents")

    # 3. index.json
    index_data = {
        "1": {
            "title": "Test Book",
            "file_path": "Test Book.pdf",
            "authors": ["Author One"],
        },
        "2": {
            "title": "Unrelated Other Book",
            "file_path": "Unrelated Other Book.epub",
            "authors": ["Author Two"],
        },
    }
    (bdir / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    return bdir


def test_full_backup_creation_and_restore(mock_output_dir: Path, mock_books_dir: Path, tmp_path: Path):
    """Verify creating a full backup packages both knowledge artifacts and indexed source books."""
    archive_dest = tmp_path / "test_full_backup.zip"

    res_backup = create_backup(
        source_dir=mock_output_dir,
        destination=archive_dest,
        format_type="zip",
        backup_type="full",
        books_dir=mock_books_dir,
        indexed_only=True,
    )

    assert res_backup["backup_type"] == "full"
    assert archive_dest.is_file()
    assert res_backup["books_count"] >= 2  # Test Book.pdf, Test Book.opf, index.json

    # Inspect backup
    info = inspect_backup(archive_dest)
    assert info["backup_type"] == "full"
    assert info["categories"]["books"] >= 2
    assert info["categories"]["graph"] >= 1

    # Restore to clean targets
    restore_out = tmp_path / "restored_output"
    restore_books = tmp_path / "restored_books"

    res_restore = restore_backup(
        archive_path=archive_dest,
        target_dir=restore_out,
        books_dir=restore_books,
        clean=True,
    )

    assert res_restore["backup_type"] == "full"
    assert res_restore["knowledge_files_restored"] >= 5
    assert res_restore["book_files_restored"] >= 2

    # Verify knowledge files
    assert (restore_out / "knowledge_graph.json").is_file()
    assert (restore_out / "obsidian_vault" / "00_Index.md").is_file()

    # Verify books
    assert (restore_books / "Test Book.pdf").is_file()
    assert (restore_books / "Test Book.opf").is_file()
    assert (restore_books / "index.json").is_file()
    # Unindexed book should NOT be restored when indexed_only=True
    assert not (restore_books / "Unrelated Other Book.epub").exists()


def test_full_backup_all_books(mock_output_dir: Path, mock_books_dir: Path, tmp_path: Path):
    """Verify indexed_only=False includes all books in books_dir."""
    archive_dest = tmp_path / "test_full_all.zip"

    res = create_backup(
        source_dir=mock_output_dir,
        destination=archive_dest,
        format_type="zip",
        backup_type="full",
        books_dir=mock_books_dir,
        indexed_only=False,
    )

    info = inspect_backup(archive_dest)
    paths = [e["path"] for e in info["entries"]]
    assert "books/Test Book.pdf" in paths
    assert "books/Unrelated Other Book.epub" in paths

    restore_books = tmp_path / "restored_all_books"
    restore_out = tmp_path / "restored_all_out"
    restore_backup(archive_dest, target_dir=restore_out, books_dir=restore_books)

    assert (restore_books / "Test Book.pdf").is_file()
    assert (restore_books / "Unrelated Other Book.epub").is_file()


def test_full_backup_cli(mock_output_dir: Path, mock_books_dir: Path, tmp_path: Path):
    """Verify CLI backup and restore commands with --full and --books-dir."""
    archive_dest = tmp_path / "cli_full.zip"
    restore_out = tmp_path / "cli_full_out"
    restore_books = tmp_path / "cli_full_books"

    # 1. Run CLI backup --full
    res_backup = runner.invoke(
        app,
        [
            "backup",
            str(archive_dest),
            "--output-dir",
            str(mock_output_dir),
            "--full",
            "--books-dir",
            str(mock_books_dir),
        ],
    )
    assert res_backup.exit_code == 0
    assert "Full Backup Summary" in res_backup.stdout
    assert "Books Source" in res_backup.stdout
    assert archive_dest.is_file()

    # 2. Run CLI restore
    res_restore = runner.invoke(
        app,
        [
            "restore",
            str(archive_dest),
            "--output-dir",
            str(restore_out),
            "--books-dir",
            str(restore_books),
        ],
    )
    assert res_restore.exit_code == 0
    assert "Full Restoration Summary" in res_restore.stdout
    assert "Book Files Restored" in res_restore.stdout
    assert (restore_out / "knowledge_graph.json").is_file()
    assert (restore_books / "Test Book.pdf").is_file()


def test_full_backup_7z(mock_output_dir: Path, mock_books_dir: Path, tmp_path: Path):
    """Verify creating and restoring a full backup with 7z format."""
    archive_dest = tmp_path / "test_full_7z.7z"

    res_backup = create_backup(
        source_dir=mock_output_dir,
        destination=archive_dest,
        format_type="7z",
        backup_type="full",
        books_dir=mock_books_dir,
        indexed_only=True,
    )
    assert res_backup["backup_type"] == "full"
    assert archive_dest.is_file()

    restore_out = tmp_path / "restored_7z_out"
    restore_books = tmp_path / "restored_7z_books"

    res_restore = restore_backup(
        archive_path=archive_dest,
        target_dir=restore_out,
        books_dir=restore_books,
        clean=True,
    )
    assert res_restore["backup_type"] == "full"
    assert (restore_out / "knowledge_graph.json").is_file()
    assert (restore_books / "Test Book.pdf").is_file()
    assert (restore_books / "Test Book.opf").is_file()
    assert (restore_books / "index.json").is_file()


def test_full_restore_without_explicit_books_dir_uses_output_dir(mock_output_dir: Path, mock_books_dir: Path, tmp_path: Path):
    """
    Verify that when restoring a full backup with --output-dir (and without --books-dir),
    the books are extracted relative to the specified output directory rather than defaulting
    to the current working directory.
    """
    archive_dest = tmp_path / "server_backup.zip"
    create_backup(
        source_dir=mock_output_dir,
        destination=archive_dest,
        format_type="zip",
        backup_type="full",
        books_dir=mock_books_dir,
        indexed_only=True,
    )

    # Server target folder simulating /srv/.../bookeeper
    server_dir = tmp_path / "server_mount" / "bookeeper"

    # Run CLI restore with ONLY -o / --output-dir
    res = runner.invoke(
        app,
        [
            "restore",
            str(archive_dest),
            "-o",
            str(server_dir),
            "--clean",
        ],
    )
    assert res.exit_code == 0
    assert "Full Restoration Summary" in res.stdout

    # Verify Knowledge graph is in server_dir
    assert (server_dir / "knowledge_graph.json").is_file()

    # Verify Books are extracted under server_dir / archive_downloader / downloads
    expected_books_dir = server_dir / "archive_downloader" / "downloads"
    assert (expected_books_dir / "Test Book.pdf").is_file()
    assert (expected_books_dir / "Test Book.opf").is_file()
    assert (expected_books_dir / "index.json").is_file()
    assert "Book Files Restored" in res.stdout
    assert "Full Restoration Summary" in res.stdout


def test_cli_sync_flow(tmp_path):
    """Test client sync command packages local backup and invokes REST restore with synchronized --clean flag."""
    from unittest.mock import MagicMock, patch
    from typer.testing import CliRunner
    from bookeeper.cli import app
    import json

    runner = CliRunner(env={"COLUMNS": "200"})

    # 1. Setup local output_dir with knowledge_graph.json
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    kg_file = out_dir / "knowledge_graph.json"
    kg_file.write_text(json.dumps({
        "nodes": [{"id": "book:1", "type": "Book", "title": "Quantum Computing"}],
        "edges": []
    }), encoding="utf-8")

    books_dir = tmp_path / "books"
    books_dir.mkdir()
    (books_dir / "book.pdf").write_bytes(b"%PDF-sample")

    # Mock urllib.request.urlopen for Step 2 (REST restore)
    mock_restore_response = MagicMock()
    mock_restore_response.read.return_value = json.dumps({
        "success": True,
        "restored_files": 4,
        "neo4j_export": {"nodes_upserted": 10, "edges_upserted": 20},
    }).encode("utf-8")
    mock_restore_response.__enter__.return_value = mock_restore_response

    with patch("urllib.request.urlopen", return_value=mock_restore_response) as mock_urlopen:
        res = runner.invoke(app, [
            "sync",
            "192.168.50.20:8855",
            "--clean",
            "--output-dir", str(out_dir),
            "--books-dir", str(books_dir),
        ])

        assert res.exit_code == 0, f"Error: {res.stdout}"
        assert "Bookkeeper Sync Plan" in res.stdout
        assert "Step 1/2" in res.stdout
        assert "Step 2/2" in res.stdout
        assert "Synchronization completed successfully" in res.stdout

        # Verify Step 2: urlopen called with remote server URL and clean=true
        mock_urlopen.assert_called_once()
        req_arg = mock_urlopen.call_args[0][0]
        assert req_arg.full_url == "http://192.168.50.20:8855/api/backup/restore"
        # Multipart payload contains name="clean"\r\n\r\ntrue
        assert b'name="clean"\r\n\r\ntrue' in req_arg.data
        assert b'filename="sync_backup.zip"' in req_arg.data


def test_cli_sync_flow_no_clean(tmp_path):
    """Test client sync command without --clean translates to clean=false in remote restore."""
    from unittest.mock import MagicMock, patch
    from typer.testing import CliRunner
    from bookeeper.cli import app
    import json

    runner = CliRunner(env={"COLUMNS": "200"})

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    kg_file = out_dir / "knowledge_graph.json"
    kg_file.write_text(json.dumps({
        "nodes": [{"id": "book:1", "type": "Book", "title": "Classical Computing"}],
        "edges": []
    }), encoding="utf-8")

    mock_restore_response = MagicMock()
    mock_restore_response.read.return_value = json.dumps({
        "success": True,
        "restored_files": 2,
        "neo4j_export": {"nodes_upserted": 8, "edges_upserted": 14},
    }).encode("utf-8")
    mock_restore_response.__enter__.return_value = mock_restore_response

    with patch("urllib.request.urlopen", return_value=mock_restore_response) as mock_urlopen:
        res = runner.invoke(app, [
            "sync",
            "http://example.com:8000",
            "--output-dir", str(out_dir),
        ])

        assert res.exit_code == 0, f"Error: {res.stdout}"
        mock_urlopen.assert_called_once()
        req_arg = mock_urlopen.call_args[0][0]
        assert req_arg.full_url == "http://example.com:8000/api/backup/restore"
        # Multipart payload contains name="clean"\r\n\r\nfalse
        assert b'name="clean"\r\n\r\nfalse' in req_arg.data






