"""
Tests for ArxivDownloader orchestrator and CLI commands.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from archive_downloader.cli import app
from archive_downloader.downloader import ArxivDownloader
from archive_downloader.models import ArticleMetadata

runner = CliRunner()

PAGE1_HTML = """
<html><body>
  <div class="paging">
    Total of 4 entries :
    <span>1-2</span>
    <a href="/list/cs.AI/recent?skip=2&amp;show=2">3-4</a>
  </div>
  <dl>
    <dt><a href="/abs/2610.00001">arXiv:2610.00001</a> [<a href="/pdf/2610.00001">pdf</a>]</dt>
    <dd><div class="list-title">Paper 1</div><div class="list-authors"><a>Alice</a></div></dd>
    <dt><a href="/abs/2610.00002">arXiv:2610.00002</a> [<a href="/pdf/2610.00002">pdf</a>]</dt>
    <dd><div class="list-title">Paper 2</div><div class="list-authors"><a>Bob</a></div></dd>
  </dl>
</body></html>
"""

PAGE2_HTML = """
<html><body>
  <div class="paging">
    Total of 4 entries :
    <a href="/list/cs.AI/recent?skip=0&amp;show=2">1-2</a>
    <span>3-4</span>
  </div>
  <dl>
    <dt><a href="/abs/2610.00003">arXiv:2610.00003</a> [<a href="/pdf/2610.00003">pdf</a>]</dt>
    <dd><div class="list-title">Paper 3</div><div class="list-authors"><a>Charlie</a></div></dd>
    <dt><a href="/abs/2610.00004">arXiv:2610.00004</a> [<a href="/pdf/2610.00004">pdf</a>]</dt>
    <dd><div class="list-title">Paper 4</div><div class="list-authors"><a>David</a></div></dd>
  </dl>
</body></html>
"""


class MockResponse:
    def __init__(self, text="", status_code=200, content=b"%PDF-1.5 dummy"):
        self.text = text
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size=32768):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def mock_get(url, **kwargs):
    if "skip=2" in url:
        return MockResponse(text=PAGE2_HTML)
    return MockResponse(text=PAGE1_HTML)


def mock_stream(method, url, **kwargs):
    return MockResponse(content=b"%PDF-1.4 dummy paper stream content")


def test_arxiv_downloader_pagination(tmp_path: Path):
    downloader = ArxivDownloader(
        base_url="https://arxiv.org/list/cs.AI/recent",
        output_dir=tmp_path,
        limit=3,
        request_delay=0.0,
        write_opf=True,
    )

    with patch("httpx.Client.get", side_effect=mock_get), patch(
        "httpx.Client.stream", side_effect=mock_stream
    ):
        stats = downloader.run()

    assert stats.new_downloaded == 3
    assert stats.pages_visited == 2  # Page 1 (items 1-2) + Page 2 (item 3)
    assert downloader.indexer.count() == 3

    # Check files on disk
    pdfs = list(tmp_path.glob("*.pdf"))
    opfs = list(tmp_path.glob("*.opf"))
    assert len(pdfs) == 3
    assert len(opfs) == 3


def test_arxiv_downloader_deduplication(tmp_path: Path):
    downloader = ArxivDownloader(
        base_url="https://arxiv.org/list/cs.AI/recent",
        output_dir=tmp_path,
        limit=2,
        count_existing=True,
        request_delay=0.0,
    )

    with patch("httpx.Client.get", side_effect=mock_get), patch(
        "httpx.Client.stream", side_effect=mock_stream
    ):
        # Run 1: downloads 2
        stats1 = downloader.run()
        assert stats1.new_downloaded == 2
        assert stats1.already_existing == 0

        # Run 2: both are already in index, should skip both and download 0
        stats2 = downloader.run()
        assert stats2.new_downloaded == 0
        assert stats2.already_existing == 2


def test_arxiv_downloader_new_only_mode(tmp_path: Path):
    downloader = ArxivDownloader(
        base_url="https://arxiv.org/list/cs.AI/recent",
        output_dir=tmp_path,
        limit=2,
        count_existing=False,  # new-only mode
        request_delay=0.0,
    )

    with patch("httpx.Client.get", side_effect=mock_get), patch(
        "httpx.Client.stream", side_effect=mock_stream
    ):
        # Run 1: downloads 2
        stats1 = downloader.run()
        assert stats1.new_downloaded == 2

        # Run 2 with new-only: should skip first 2 and download next 2 from page 2!
        stats2 = downloader.run()
        assert stats2.new_downloaded == 2
        assert stats2.already_existing == 2
        assert downloader.indexer.count() == 4


def test_cli_commands(tmp_path: Path):
    with patch("httpx.Client.get", side_effect=mock_get), patch(
        "httpx.Client.stream", side_effect=mock_stream
    ):
        # 1. Download
        res = runner.invoke(
            app,
            [
                "download",
                "--output-dir",
                str(tmp_path),
                "--limit",
                "2",
                "--delay",
                "0",
            ],
        )
        assert res.exit_code == 0
        assert "Download Run Summary" in res.stdout
        assert "New Articles Downloaded" in res.stdout

        # 2. Status
        res_status = runner.invoke(
            app,
            [
                "status",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert res_status.exit_code == 0
        assert "Download Index Status" in res_status.stdout
        assert "Total Articles Indexed: 2" in res_status.stdout
