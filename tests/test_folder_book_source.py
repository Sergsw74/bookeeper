"""
Unit tests for FolderBookClient and folder-based book sourcing in Bookkeeper.
"""

import json
from pathlib import Path
import tempfile
import pytest
from typer.testing import CliRunner

from bookeeper.calibre import FolderBookClient, get_book_client, CalibreClient
from bookeeper.config import Settings
from bookeeper.cli import app

runner = CliRunner()


@pytest.fixture
def sample_download_dir(tmp_path: Path) -> Path:
    """Create a temporary download folder mimicking archive_downloader output."""
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    # 1. First paper (PDF + OPF + index entry)
    p1 = downloads / "2610.00001 - Paper One.pdf"
    p1.write_bytes(b"%PDF-1.4 test dummy content")
    opf1 = downloads / "2610.00001 - Paper One.opf"
    opf1.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Paper One: First Discovery</dc:title>
    <dc:creator>Alice Smith</dc:creator>
    <dc:creator>Bob Jones</dc:creator>
    <dc:description>Abstract of paper one</dc:description>
  </metadata>
</package>""",
        encoding="utf-8",
    )

    # 2. Second paper (PDF + OPF + index entry)
    p2 = downloads / "2610.00002 - Paper Two.pdf"
    p2.write_bytes(b"%PDF-1.4 test dummy content two")
    opf2 = downloads / "2610.00002 - Paper Two.opf"
    opf2.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Paper Two: Second Discovery</dc:title>
    <dc:creator>Carol White</dc:creator>
    <dc:description>Abstract of paper two</dc:description>
  </metadata>
</package>""",
        encoding="utf-8",
    )

    # 3. Third loose paper (EPUB without index.json entry)
    p3 = downloads / "Loose Ebook.epub"
    p3.write_bytes(b"dummy epub bytes")

    # index.json with entries for p1 and p2
    index_data = {
        "2610.00001": {
            "arxiv_id": "2610.00001",
            "title": "Paper One: First Discovery",
            "authors": ["Alice Smith", "Bob Jones"],
            "file_path": "2610.00001 - Paper One.pdf",
            "file_size": len(p1.read_bytes()),
            "format": "pdf",
            "downloaded_at": "2026-10-10T12:00:00",
            "primary_category": "cs.AI",
        },
        "2610.00002": {
            "arxiv_id": "2610.00002",
            "title": "Paper Two: Second Discovery",
            "authors": ["Carol White"],
            "file_path": "2610.00002 - Paper Two.pdf",
            "file_size": len(p2.read_bytes()),
            "format": "pdf",
            "downloaded_at": "2026-10-10T12:05:00",
            "primary_category": "cs.LG",
        },
    }
    (downloads / "index.json").write_text(json.dumps(index_data, indent=2), encoding="utf-8")

    return downloads


def test_folder_client_discovery_and_indexing(sample_download_dir: Path):
    """Test FolderBookClient discovers both index.json entries and loose files."""
    client = FolderBookClient(sample_download_dir)
    assert client.is_available() is True
    assert client.source_type == "folder"

    books = client.list_books()
    assert len(books) == 3

    # Check ID ordering and fields
    b1 = books[0]
    assert b1["id"] == 1
    assert b1["arxiv_id"] == "2610.00001"
    assert "Alice Smith" in b1["authors"]
    assert b1["formats"] == ["PDF"]

    b2 = books[1]
    assert b2["id"] == 2
    assert b2["arxiv_id"] == "2610.00002"
    assert "Carol White" in b2["authors"]

    b3 = books[2]
    assert b3["id"] == 3
    assert b3["title"] == "Loose Ebook"
    assert b3["formats"] == ["EPUB"]

    # Check that .book_ids.json was created
    id_file = sample_download_dir / ".book_ids.json"
    assert id_file.is_file()
    saved = json.loads(id_file.read_text(encoding="utf-8"))
    entries = saved.get("entries", saved)
    assert entries["arxiv:2610.00001"] == 1
    assert entries["arxiv:2610.00002"] == 2


def test_folder_client_id_stability(sample_download_dir: Path):
    """Test IDs remain stable when client is re-instantiated and new file is added."""
    c1 = FolderBookClient(sample_download_dir)
    books1 = c1.list_books()
    id1_map = {b["title"]: b["id"] for b in books1}

    # Add a new file
    p4 = sample_download_dir / "2610.00003 - Brand New Paper.pdf"
    p4.write_bytes(b"%PDF dummy")

    c2 = FolderBookClient(sample_download_dir)
    books2 = c2.list_books()
    assert len(books2) == 4

    # Existing books MUST maintain their exact IDs
    for b in books2:
        if b["title"] in id1_map:
            assert b["id"] == id1_map[b["title"]]

    # New book gets next ID (4)
    new_b = next(b for b in books2 if "Brand New Paper" in b["title"] or b["id"] == 4)
    assert new_b["id"] == 4


def test_folder_client_export_book(sample_download_dir: Path):
    """Test export_book returns local path to book file."""
    client = FolderBookClient(sample_download_dir)
    exported = client.export_book(1, sample_download_dir / "target")
    assert exported is not None
    assert exported.is_file()
    assert exported.name == "2610.00001 - Paper One.pdf"

    # Non-existent ID returns None
    assert client.export_book(999, sample_download_dir / "target") is None


def test_folder_client_update_metadata(sample_download_dir: Path):
    """Test update_metadata modifies in-memory, OPF, and index.json."""
    client = FolderBookClient(sample_download_dir)
    updated = client.update_metadata(
        book_id=1,
        title="Updated Title One",
        authors=["Alice S. Smith", "Bob J. Jones"],
        comments="New cleaned comments",
    )
    assert updated is True

    # Check in-memory
    b1 = client.list_books()[0]
    assert b1["title"] == "Updated Title One"
    assert b1["authors"] == ["Alice S. Smith", "Bob J. Jones"]

    # Check OPF
    opf1 = sample_download_dir / "2610.00001 - Paper One.opf"
    opf_content = opf1.read_text(encoding="utf-8")
    assert "Updated Title One" in opf_content
    assert "Alice S. Smith" in opf_content
    assert "New cleaned comments" in opf_content

    # Check index.json
    idx_content = json.loads((sample_download_dir / "index.json").read_text(encoding="utf-8"))
    assert idx_content["2610.00001"]["title"] == "Updated Title One"
    assert idx_content["2610.00001"]["authors"] == ["Alice S. Smith", "Bob J. Jones"]
    assert idx_content["2610.00001"]["comments"] == "New cleaned comments"


def test_get_book_client_resolution(sample_download_dir: Path, tmp_path: Path):
    """Test get_book_client properly resolves between Calibre and folder sources."""
    cfg = Settings(books_dir=str(sample_download_dir))

    # 1. Configured books_dir resolves to FolderBookClient
    client1 = get_book_client(cfg)
    assert isinstance(client1, FolderBookClient)
    assert client1.folder_path == sample_download_dir

    # 2. CLI books_dir override
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    client2 = get_book_client(cfg, books_dir=str(other_dir))
    assert isinstance(client2, FolderBookClient)
    assert client2.folder_path == other_dir.resolve()

    # 3. calibre_path pointing to a folder with books without metadata.db resolves to FolderBookClient
    cfg_empty = Settings(books_dir=None, calibre_library_path="/nonexistent")
    client3 = get_book_client(cfg_empty, calibre_path=str(sample_download_dir))
    assert isinstance(client3, FolderBookClient)

    # 4. calibre_path pointing to a directory with metadata.db resolves to CalibreClient
    calibre_dir = tmp_path / "calibre"
    calibre_dir.mkdir()
    (calibre_dir / "metadata.db").write_bytes(b"SQLite format 3\x00")
    client4 = get_book_client(cfg_empty, calibre_path=str(calibre_dir))
    assert isinstance(client4, CalibreClient)


def test_cli_list_books_with_books_dir(sample_download_dir: Path):
    """Test `bookeeper list-books --books-dir <path>` CLI command."""
    res = runner.invoke(app, ["list-books", "--books-dir", str(sample_download_dir)])
    assert res.exit_code == 0
    assert "Download Folder:" in res.output
    assert "Paper One: First Discovery" in res.output
    assert "2610.00001" in res.output
    assert "Carol White" in res.output


def test_cli_list_books_with_download_dir_alias(sample_download_dir: Path):
    """Test `bookeeper list-books --download-dir <path>` CLI command alias."""
    res = runner.invoke(app, ["list-books", "--download-dir", str(sample_download_dir), "--limit", "1"])
    assert res.exit_code == 0
    assert "Showing 1 of 3" in res.output


def test_book_parser_parse_publication_date():
    """Test publication date parsing across diverse formats and edge cases."""
    from bookeeper.calibre.parser import BookParser

    # ISO formats
    assert BookParser.parse_publication_date("2024-05-18T10:30:00Z") == "2024-05-18"
    assert BookParser.parse_publication_date("2023-11-05") == "2023-11-05"
    assert BookParser.parse_publication_date("2022-07") == "2022-07"
    assert BookParser.parse_publication_date("1998") == "1998"

    # PDF creation date format D:YYYYMMDD...
    assert BookParser.parse_publication_date("D:20250314120000Z") == "2025-03-14"

    # arXiv identifiers
    assert BookParser.parse_publication_date("2610.12466") == "2026-10"
    assert BookParser.parse_publication_date("arxiv:2305.18290v2") == "2023-05"

    # Calibre unset date and invalid formats
    assert BookParser.parse_publication_date("0101-01-01") is None
    assert BookParser.parse_publication_date("0101-01-01T00:00:00+00:00") is None
    assert BookParser.parse_publication_date(None) is None
    assert BookParser.parse_publication_date("") is None
    assert BookParser.parse_publication_date("Quick Start Guide") is None


def test_folder_client_publication_date_extraction(sample_download_dir: Path):
    """Verify FolderBookClient populates publication_date from files / index / opf."""
    client = FolderBookClient(sample_download_dir)
    books = client.list_books()
    assert len(books) == 3

    # Check that each paper got its publication_date inferred (e.g. 2610.00001 -> 2026-10)
    p1 = next(b for b in books if "Paper One" in b["title"])
    assert p1.get("publication_date") == "2026-10"
    assert p1.get("pubdate") == "2026-10"
