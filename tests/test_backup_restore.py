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

runner = CliRunner()


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


