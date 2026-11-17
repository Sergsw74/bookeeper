"""
Tests for scripts/daily_arxiv_sync.py automated pipeline.
"""

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts directory is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(repo_root / "scripts"))

import daily_arxiv_sync


def test_daily_arxiv_sync_parse_args():
    """Verify CLI flag parsing and defaults."""
    parser = daily_arxiv_sync.build_parser()
    args = parser.parse_args([
        "--server", "http://192.168.1.50:8855",
        "--limit", "5",
        "--category", "cs.CL",
        "--dry-run",
    ])
    assert args.server == "http://192.168.1.50:8855"
    assert args.limit == 5
    assert args.category == "cs.CL"
    assert args.dry_run is True
    assert args.diff_against == "neo4j"
    assert args.new_only is True


def test_daily_arxiv_sync_download_phase(tmp_path: Path):
    """Verify download phase detects newly downloaded files via DownloadIndex."""
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()

    # Pre-existing entry
    idx_file = downloads_dir / "index.json"
    idx_file.write_text(json.dumps({
        "2401.00001": {
            "arxiv_id": "2401.00001",
            "title": "Old Paper",
            "file_path": "2401.00001v1.Old_Paper.pdf",
            "file_size": 100,
            "downloaded_at": "2026-10-01T00:00:00Z",
        }
    }), encoding="utf-8")
    (downloads_dir / "2401.00001v1.Old_Paper.pdf").write_bytes(b"dummy old pdf")

    # Mock ArxivDownloader.run to simulate downloading 1 new paper
    def mock_run(self, progress_callback=None):
        # Update index with new paper
        idx_data = json.loads(idx_file.read_text(encoding="utf-8"))
        idx_data["2402.00002"] = {
            "arxiv_id": "2402.00002",
            "title": "New Paper",
            "file_path": "2402.00002v1.New_Paper.pdf",
            "file_size": 200,
            "downloaded_at": "2026-10-02T00:00:00Z",
        }
        idx_file.write_text(json.dumps(idx_data), encoding="utf-8")
        (downloads_dir / "2402.00002v1.New_Paper.pdf").write_bytes(b"%PDF-1.5 dummy new pdf")
        stats = MagicMock()
        stats.total_inspected = 5
        stats.already_existing = 4
        stats.failed = 0
        return stats

    with patch.object(daily_arxiv_sync.ArxivDownloader, "run", mock_run):
        new_files, stats = daily_arxiv_sync.run_download_phase(
            downloads_dir=downloads_dir,
            category="cs.AI",
            query=None,
            limit=1,
            delay=0.1,
            new_only=True,
        )

        assert len(new_files) == 1
        assert new_files[0].name == "2402.00002v1.New_Paper.pdf"
        assert stats["new_downloaded"] == 1


def test_daily_arxiv_sync_build_graph_phase(tmp_path: Path):
    """Verify build-graph phase generates manifest and calls subprocess."""
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    new_paper = downloads_dir / "2402.00002v1.New_Paper.pdf"
    new_paper.write_bytes(b"dummy")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        res = daily_arxiv_sync.run_build_graph_phase(
            new_files=[new_paper],
            downloads_dir=downloads_dir,
            output_dir=tmp_path / "output",
            config_path=None,
            config_diff=None,
            postprocess=False,
        )
        assert res["success"] is True
        assert res["articles_processed"] == 1
        assert mock_run.call_count == 1
        cmd_called = mock_run.call_args[0][0]
        assert "build-graph" in cmd_called
        assert "-L" in cmd_called or "--file-list" in cmd_called


def test_daily_arxiv_sync_incremental_backup_and_upload(tmp_path: Path):
    """Verify incremental backup creation and multipart HTTP upload to server."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "knowledge_graph.json").write_text("{}", encoding="utf-8")

    archive_path = tmp_path / "sync_incremental.zip"
    archive_path.write_bytes(b"PK0304testzip")

    with patch("daily_arxiv_sync.create_incremental_backup") as mock_inc, \
         patch("urllib.request.urlopen") as mock_urlopen:

        mock_inc.return_value = {
            "archive_path": archive_path,
            "compressed_bytes": len(archive_path.read_bytes()),
            "total_files": 3,
            "diff_summary": {
                "new_books_count": 1,
                "related_books_count": 2,
                "shared_concepts_count": 5,
            },
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "success": True,
            "restored_files": 3,
            "neo4j_export": {"nodes_upserted": 6, "edges_upserted": 10},
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        # 1. Incremental backup phase
        backup_dest, backup_res = daily_arxiv_sync.run_incremental_backup_phase(
            output_dir=out_dir,
            downloads_dir=tmp_path / "downloads",
            baseline="neo4j",
        )
        assert backup_dest.is_file() or backup_dest.parent.is_dir()
        assert backup_res["total_files"] == 3

        # 2. Upload phase
        upload_res = daily_arxiv_sync.upload_incremental_backup(
            server_url="http://remote.host:8855",
            archive_path=archive_path,
            timeout=60,
        )
        assert upload_res["success"] is True
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://remote.host:8855/api/backup/restore"
        assert b'name="incremental"\r\n\r\ntrue' in req.data
        assert b'name="clean"\r\n\r\nfalse' in req.data


def test_daily_arxiv_sync_clean_exit_when_no_new_articles(tmp_path: Path):
    """When 0 new articles are downloaded and force is False, pipeline exits cleanly with code 0."""
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    (downloads_dir / "index.json").write_text("{}", encoding="utf-8")

    with patch("daily_arxiv_sync.run_download_phase") as mock_dl:
        mock_dl.return_value = ([], {"new_downloaded": 0})

        code = daily_arxiv_sync.main([
            "--books-dir", str(downloads_dir),
            "--output-dir", str(tmp_path / "output"),
            "--limit", "5",
        ])
        assert code == 0


def test_daily_arxiv_sync_download_validation_and_ordering(tmp_path: Path):
    """
    Verify Phase 1 download engine:
    1. Captures files via live download callback in order.
    2. Excludes non-PDF or empty corrupt payloads.
    3. Guarantees only verified new PDFs are passed to Phase 2.
    """
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()

    # Create dummy files
    valid_pdf = downloads_dir / "2405.00001 - Paper One.pdf"
    valid_pdf.write_bytes(b"%PDF-1.5 \x00\x01\x02 Valid Content")

    corrupt_html = downloads_dir / "2405.00002 - Error 503.pdf"
    corrupt_html.write_bytes(b"<html><head><title>503 Service Unavailable</title></head></html>")

    empty_pdf = downloads_dir / "2405.00003 - Empty.pdf"
    empty_pdf.write_bytes(b"")

    def mock_run(self, progress_callback=None):
        # Simulate ArxivDownloader triggering progress_callback for all 3
        if progress_callback:
            # Item 1: Valid
            art1 = MagicMock()
            art1.arxiv_id = "2405.00001"
            self.indexer.add(daily_arxiv_sync.DownloadIndex(downloads_dir).entries.get("dummy") or MagicMock(
                arxiv_id="2405.00001",
                file_path=valid_pdf.name,
            ))
            progress_callback("download", 1, 3, art1, True)

            # Item 2: Corrupted HTML
            art2 = MagicMock()
            art2.arxiv_id = "2405.00002"
            self.indexer.add(MagicMock(
                arxiv_id="2405.00002",
                file_path=corrupt_html.name,
            ))
            progress_callback("download", 2, 3, art2, True)

            # Item 3: Empty file
            art3 = MagicMock()
            art3.arxiv_id = "2405.00003"
            self.indexer.add(MagicMock(
                arxiv_id="2405.00003",
                file_path=empty_pdf.name,
            ))
            progress_callback("download", 3, 3, art3, True)

        stats = MagicMock()
        stats.total_inspected = 3
        stats.already_existing = 0
        stats.failed = 0
        return stats

    with patch.object(daily_arxiv_sync.ArxivDownloader, "run", mock_run):
        new_files, stats = daily_arxiv_sync.run_download_phase(
            downloads_dir=downloads_dir,
            limit=3,
        )

        # Only the valid PDF must be accepted
        assert len(new_files) == 1
        assert new_files[0].name == "2405.00001 - Paper One.pdf"
        assert stats["new_downloaded"] == 1

