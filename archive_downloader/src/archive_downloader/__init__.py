"""
Archive Downloader - Automated arXiv & scientific paper crawler with indexing, deduplication, and Calibre metadata.
"""

__version__ = "0.1.0"

from archive_downloader.models import ArticleMetadata, DownloadStats, IndexEntry
from archive_downloader.indexer import DownloadIndex
from archive_downloader.parser import parse_arxiv_list_page, sanitize_filename
from archive_downloader.downloader import ArxivDownloader

__all__ = [
    "ArticleMetadata",
    "DownloadStats",
    "IndexEntry",
    "DownloadIndex",
    "parse_arxiv_list_page",
    "sanitize_filename",
    "ArxivDownloader",
]
