"""
Core downloader engine for arXiv articles and papers.
"""

import datetime
import hashlib
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import httpx

from archive_downloader.calibre_metadata import generate_calibre_opf
from archive_downloader.indexer import DownloadIndex
from archive_downloader.models import ArticleMetadata, DownloadStats, IndexEntry
from archive_downloader.parser import (
    normalize_arxiv_url,
    parse_arxiv_list_page,
    sanitize_filename,
)

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://arxiv.org/list/cs.AI/recent?skip=0&show=500"
DEFAULT_USER_AGENT = "ArchiveDownloader/1.0 (academic research & library indexing; mailto:serg@example.com)"


class ArxivDownloader:
    """
    Downloads articles and books from arXiv with pagination, deduplication, and indexing.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        output_dir: Optional[Path] = None,
        limit: int = 100,
        page_size: int = 500,
        count_existing: bool = True,
        request_delay: float = 0.5,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 30.0,
        write_opf: bool = True,
    ):
        self.page_size = page_size
        self.base_url = normalize_arxiv_url(base_url, default_show=page_size)
        self.output_dir = Path(output_dir or "./downloads").expanduser().resolve()
        self.limit = max(1, limit)
        self.count_existing = count_existing
        self.request_delay = max(0.0, request_delay)
        self.user_agent = user_agent
        self.timeout = timeout
        self.write_opf = write_opf

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.indexer = DownloadIndex(self.output_dir)

    def run(
        self,
        progress_callback: Optional[Callable[[str, int, int, ArticleMetadata, bool], None]] = None,
    ) -> DownloadStats:
        """
        Execute the crawl and download process.
        """
        t0 = time.perf_counter()
        stats = DownloadStats(target_limit=self.limit)

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
        }

        current_url = self.base_url
        pages_visited = 0
        satisfied_count = 0

        with httpx.Client(headers=headers, follow_redirects=True, timeout=self.timeout) as client:
            while current_url and satisfied_count < self.limit:
                pages_visited += 1
                stats.pages_visited = pages_visited

                logger.info(f"Fetching page {pages_visited}: {current_url}")
                try:
                    resp = client.get(current_url)
                    resp.raise_for_status()
                except Exception as e:
                    logger.error(f"Failed to fetch page {current_url}: {e}")
                    stats.failed += 1
                    break

                articles, next_page_url = parse_arxiv_list_page(resp.text, current_url=current_url)
                if not articles:
                    logger.warning(f"No articles found on page: {current_url}")
                    break

                for article in articles:
                    stats.total_inspected += 1

                    # 1. Deduplication check
                    if self.indexer.is_downloaded(article.arxiv_id):
                        stats.already_existing += 1
                        if self.count_existing:
                            satisfied_count += 1

                        if progress_callback:
                            progress_callback("skip", satisfied_count, self.limit, article, False)

                        if satisfied_count >= self.limit:
                            break
                        continue

                    # 2. Download new article
                    success = self._download_article(client, article, stats)
                    if success:
                        stats.new_downloaded += 1
                        satisfied_count += 1
                        if progress_callback:
                            progress_callback("download", satisfied_count, self.limit, article, True)
                    else:
                        stats.failed += 1

                    if satisfied_count >= self.limit:
                        break

                    # Polite rate-limiting between downloads
                    if self.request_delay > 0:
                        time.sleep(self.request_delay)

                # Advance to next page if target not yet reached
                if satisfied_count < self.limit and next_page_url and next_page_url != current_url:
                    current_url = next_page_url
                    if self.request_delay > 0:
                        time.sleep(self.request_delay)
                else:
                    break

        stats.duration_seconds = round(time.perf_counter() - t0, 2)
        return stats

    def _download_article(
        self,
        client: httpx.Client,
        article: ArticleMetadata,
        stats: DownloadStats,
    ) -> bool:
        """Download a single article and register it in the index."""
        # Sanitize filename: "<arxiv_id> - <sanitized_title>.pdf"
        safe_title = sanitize_filename(article.title, max_length=100)
        filename = f"{article.arxiv_id} - {safe_title}.pdf"
        target_path = self.output_dir / filename

        download_url = article.download_url
        if not download_url:
            download_url = f"https://arxiv.org/pdf/{article.arxiv_id}"

        logger.info(f"Downloading [{article.arxiv_id}] from {download_url} -> {filename}")

        sha256 = hashlib.sha256()
        total_bytes = 0

        try:
            with client.stream("GET", download_url) as stream:
                if stream.status_code != 200:
                    logger.warning(f"Failed to download {article.arxiv_id}: HTTP {stream.status_code}")
                    return False

                with open(target_path, "wb") as f:
                    for chunk in stream.iter_bytes(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                            sha256.update(chunk)
                            total_bytes += len(chunk)

            if total_bytes == 0:
                if target_path.exists():
                    target_path.unlink()
                return False

            stats.total_bytes += total_bytes

            # Generate accompanying Calibre-compatible .opf file
            if self.write_opf:
                opf_path = self.output_dir / f"{article.arxiv_id} - {safe_title}.opf"
                try:
                    generate_calibre_opf(article, opf_path)
                except Exception as e:
                    logger.warning(f"Could not write .opf for {article.arxiv_id}: {e}")

            # Register in index
            entry = IndexEntry(
                arxiv_id=article.arxiv_id,
                title=article.title,
                authors=article.authors,
                primary_category=article.primary_category,
                file_path=filename,
                file_size=total_bytes,
                format="pdf",
                downloaded_at=datetime.datetime.now().isoformat(),
                sha256=sha256.hexdigest(),
                url=download_url,
            )
            self.indexer.add(entry)
            return True

        except Exception as e:
            logger.error(f"Download failed for {article.arxiv_id}: {e}")
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass
            return False
