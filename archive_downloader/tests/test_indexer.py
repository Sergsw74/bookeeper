"""
Tests for DownloadIndex and deduplication.
"""

from pathlib import Path
from archive_downloader.indexer import DownloadIndex
from archive_downloader.models import IndexEntry


def test_download_index_lifecycle(tmp_path: Path):
    index = DownloadIndex(tmp_path)
    assert index.count() == 0

    # 1. Add entry
    pdf_file = tmp_path / "test_paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 mock content")

    entry = IndexEntry(
        arxiv_id="2610.00001",
        title="Test Paper",
        authors=["Alice"],
        primary_category="cs.AI",
        file_path="test_paper.pdf",
        file_size=pdf_file.stat().st_size,
        format="pdf",
        downloaded_at="2026-10-10T12:00:00Z",
        sha256="abc123",
        url="https://arxiv.org/pdf/2610.00001",
    )
    index.add(entry)
    assert index.count() == 1
    assert index.is_downloaded("2610.00001") is True
    assert index.is_downloaded("2610.99999") is False

    # 2. Reload index from disk
    new_index = DownloadIndex(tmp_path)
    assert new_index.count() == 1
    loaded = new_index.get("2610.00001")
    assert loaded is not None
    assert loaded.title == "Test Paper"

    # 3. If file is deleted, is_downloaded returns False
    pdf_file.unlink()
    assert new_index.is_downloaded("2610.00001") is False
