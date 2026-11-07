"""
Tests for arXiv search query construction, search page parsing, and exact match filtering.
"""

from pathlib import Path
from unittest.mock import patch

from archive_downloader.downloader import ArxivDownloader
from archive_downloader.models import ArticleMetadata
from archive_downloader.parser import (
    build_arxiv_search_url,
    normalize_arxiv_url,
    parse_arxiv_page,
    parse_arxiv_search_page,
)

SAMPLE_SEARCH_HTML = """
<!DOCTYPE html>
<html>
<body>
  <h1 class="title">Showing 1–2 of 5 results for all: jepa</h1>
  <ol class="breathe-horizontal">
    <li class="arxiv-result">
      <div class="is-marginless">
        <p class="list-title is-inline-block">
          <a href="https://arxiv.org/abs/2611.08156v1">arXiv:2611.08156v1</a>
          <span>[<a href="https://arxiv.org/pdf/2611.08156v1">pdf</a>, <a href="/format/2611.08156">other</a>]</span>
        </p>
        <div class="tags is-inline-block">
          <span class="tag is-small is-link tooltip" data-tooltip="Robotics">cs.RO</span>
          <span class="tag is-small is-grey tooltip" data-tooltip="Computer Vision">cs.CV</span>
        </div>
      </div>
      <p class="title is-5 mathjax">
        La-<span class="search-hit mathjax">JEPA</span>: Compact Robot Policies with Latent World Modeling
      </p>
      <p class="authors">
        <span class="has-text-black-bis">Authors:</span>
        <a href="/search/?searchtype=author&query=Clemente">Mateo Clemente</a>,
        <a href="/search/?searchtype=author&query=Xue">George Xue</a>
      </p>
      <p class="abstract mathjax">
        <span class="search-hit">Abstract</span>:
        <span class="abstract-full">Latent Action JEPA (La-JEPA) enables compact robot policy representation.</span>
      </p>
    </li>
    <li class="arxiv-result">
      <div class="is-marginless">
        <p class="list-title is-inline-block">
          <a href="https://arxiv.org/abs/2611.06901">arXiv:2611.06901</a>
          <span>[<a href="https://arxiv.org/pdf/2611.06901">pdf</a>]</span>
        </p>
        <div class="tags is-inline-block">
          <span class="tag is-small is-link tooltip">cs.CV</span>
        </div>
      </div>
      <p class="title is-5 mathjax">
        Motion-guided Contrastive Regularization For Latent World Models
      </p>
      <p class="authors">
        <a href="/search/?searchtype=author&query=Chen">Yiwen Chen</a>
      </p>
      <p class="abstract mathjax">
        <span class="abstract-short">JEPA-style world models learn action-conditioned latent dynamics.</span>
      </p>
    </li>
  </ol>
  <nav class="pagination">
    <a class="pagination-next" href="/search/?searchtype=all&amp;query=jepa&amp;start=2&amp;size=2">Next</a>
  </nav>
</body>
</html>
"""

SAMPLE_PAGE2_HTML = """
<!DOCTYPE html>
<html>
<body>
  <h1 class="title">Showing 3–3 of 3 results for all: jepa</h1>
  <ol class="breathe-horizontal">
    <li class="arxiv-result">
      <div class="is-marginless">
        <p class="list-title is-inline-block">
          <a href="https://arxiv.org/abs/2611.00099">arXiv:2611.00099</a>
          <span>[<a href="https://arxiv.org/pdf/2611.00099">pdf</a>]</span>
        </p>
        <div class="tags is-inline-block">
          <span class="tag is-small is-link">cs.AI</span>
        </div>
      </div>
      <p class="title is-5 mathjax">Unrelated title mentioning JEPAs</p>
      <p class="authors"><a>Author Three</a></p>
      <p class="abstract mathjax"><span class="abstract-short">This mentions JEPAs in plural form only.</span></p>
    </li>
  </ol>
  <nav class="pagination">
    <a class="pagination-next is-invisible" href="">Next</a>
  </nav>
</body>
</html>
"""


def test_build_arxiv_search_url():
    # Basic query
    url = build_arxiv_search_url("jepa", exact_match=False, size=100)
    assert "query=jepa" in url
    assert "searchtype=all" in url
    assert "size=100" in url
    assert "abstracts=show" in url

    # Exact match query adds quotes
    exact_url = build_arxiv_search_url("jepa", exact_match=True, size=50)
    assert 'query=%22jepa%22' in exact_url or 'query="jepa"' in exact_url
    assert "size=50" in exact_url


def test_normalize_search_url():
    raw_url = "https://arxiv.org/search/?searchtype=all&query=jepa"
    norm = normalize_arxiv_url(raw_url)
    assert "start=0" in norm
    assert "size=" in norm
    assert "order=-announced_date_first" in norm


def test_parse_arxiv_search_page():
    articles, next_url = parse_arxiv_search_page(
        SAMPLE_SEARCH_HTML,
        current_url="https://arxiv.org/search/?query=jepa",
    )
    assert len(articles) == 2

    a1 = articles[0]
    # Check version stripping to canonical ID
    assert a1.arxiv_id == "2611.08156"
    assert a1.title == "La-JEPA: Compact Robot Policies with Latent World Modeling"
    assert a1.authors == ["Mateo Clemente", "George Xue"]
    assert a1.primary_category == "cs.RO"
    assert a1.categories == ["cs.RO", "cs.CV"]
    assert "La-JEPA" in a1.abstract
    assert a1.pdf_url == "https://arxiv.org/pdf/2611.08156v1"

    a2 = articles[1]
    assert a2.arxiv_id == "2611.06901"
    assert a2.title == "Motion-guided Contrastive Regularization For Latent World Models"
    assert "JEPA-style" in a2.abstract

    # Next pagination URL
    assert next_url is not None
    assert "start=2" in next_url


def test_parse_arxiv_page_auto_detection():
    # Search page detection
    articles, _ = parse_arxiv_page(SAMPLE_SEARCH_HTML, current_url="https://arxiv.org/search/?query=jepa")
    assert len(articles) == 2


class MockResponse:
    def __init__(self, text="", status_code=200, content=b"%PDF-1.5 dummy paper data"):
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


def test_downloader_search_with_exact_match(tmp_path: Path):
    def mock_get(url, **kwargs):
        if "start=2" in url:
            return MockResponse(text=SAMPLE_PAGE2_HTML)
        return MockResponse(text=SAMPLE_SEARCH_HTML)

    def mock_stream(method, url, **kwargs):
        return MockResponse(content=b"%PDF-1.4 test data")

    # Exact match enabled for "jepa":
    # Item 1 has "La-JEPA: ...", Item 2 has "JEPA-style ...", Item 3 has only "JEPAs" (plural, should be skipped if exact word required)
    downloader = ArxivDownloader(
        query="jepa",
        exact_match=True,
        output_dir=tmp_path,
        limit=10,
        request_delay=0.0,
        write_opf=True,
    )

    with patch("httpx.Client.get", side_effect=mock_get), patch(
        "httpx.Client.stream", side_effect=mock_stream
    ):
        stats = downloader.run()

        # Two papers matched exact term "jepa", 1 was skipped because it only had "JEPAs"
        assert stats.new_downloaded == 2
        assert downloader.indexer.count() == 2

        # Verify that files were created with exact name notation
        pdfs = list(tmp_path.glob("*.pdf"))
        assert len(pdfs) == 2
        opfs = list(tmp_path.glob("*.opf"))
        assert len(opfs) == 2

        # Check skip if run again (deduplication)
        stats2 = downloader.run()
        assert stats2.already_existing == 2
        assert stats2.new_downloaded == 0


def test_downloader_file_list_includes_downloaded_and_skipped(tmp_path: Path):
    def mock_get(url, **kwargs):
        return MockResponse(text=SAMPLE_SEARCH_HTML)

    def mock_stream(method, url, **kwargs):
        return MockResponse(content=b"%PDF-1.4 test data")

    manifest_file = tmp_path / "file_list.txt"

    # Run 1: downloads 2 new articles
    downloader1 = ArxivDownloader(
        query="jepa",
        output_dir=tmp_path / "downloads",
        file_list=manifest_file,
        limit=2,
        request_delay=0.0,
    )

    with patch("httpx.Client.get", side_effect=mock_get), patch(
        "httpx.Client.stream", side_effect=mock_stream
    ):
        stats1 = downloader1.run()

    assert stats1.new_downloaded == 2
    assert manifest_file.is_file()
    lines1 = [line.strip() for line in manifest_file.read_text().splitlines() if line.strip()]
    assert len(lines1) == 2
    assert "2611.08156 - La-JEPA_ Compact Robot Policies with Latent World Modeling.pdf" in lines1
    assert "2611.06901 - Motion-guided Contrastive Regularization For Latent World Models.pdf" in lines1

    # Run 2: both are now skipped, write to manifest_skipped.txt
    manifest_skipped = tmp_path / "manifest_skipped.txt"
    downloader2 = ArxivDownloader(
        query="jepa",
        output_dir=tmp_path / "downloads",
        file_list=manifest_skipped,
        limit=2,
        request_delay=0.0,
    )

    with patch("httpx.Client.get", side_effect=mock_get), patch(
        "httpx.Client.stream", side_effect=mock_stream
    ):
        stats2 = downloader2.run()

    assert stats2.already_existing == 2
    assert stats2.new_downloaded == 0
    assert manifest_skipped.is_file()
    lines2 = [line.strip() for line in manifest_skipped.read_text().splitlines() if line.strip()]
    # Even though both articles were skipped, they MUST be populated in the file list!
    assert len(lines2) == 2
    assert "2611.08156 - La-JEPA_ Compact Robot Policies with Latent World Modeling.pdf" in lines2
    assert "2611.06901 - Motion-guided Contrastive Regularization For Latent World Models.pdf" in lines2

    # Run 3 with full_path=True
    manifest_full = tmp_path / "manifest_full.txt"
    downloader3 = ArxivDownloader(
        query="jepa",
        output_dir=tmp_path / "downloads",
        file_list=manifest_full,
        full_path=True,
        limit=2,
        request_delay=0.0,
    )

    with patch("httpx.Client.get", side_effect=mock_get), patch(
        "httpx.Client.stream", side_effect=mock_stream
    ):
        stats3 = downloader3.run()

    lines3 = [line.strip() for line in manifest_full.read_text().splitlines() if line.strip()]
    assert len(lines3) == 2
    assert lines3[0].startswith("/")
    assert lines3[0].endswith(".pdf")
