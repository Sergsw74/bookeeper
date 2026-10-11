"""
Data models for Archive Downloader.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ArticleMetadata(BaseModel):
    """Metadata extracted for a single article or paper from arXiv."""

    arxiv_id: str = Field(description="Unique arXiv identifier (e.g. 2610.12466).")
    title: str = Field(description="Article title.")
    authors: List[str] = Field(default_factory=list, description="List of author names.")
    primary_category: str = Field(default="", description="Primary subject classification (e.g. cs.AI).")
    categories: List[str] = Field(default_factory=list, description="All subject classifications.")
    abstract_url: str = Field(default="", description="URL to the abstract page.")
    pdf_url: str = Field(default="", description="Direct URL to download PDF.")
    available_formats: List[str] = Field(default_factory=list, description="List of formats found (pdf, html, other).")
    preferred_format: str = Field(default="pdf", description="Selected download format.")
    download_url: str = Field(default="", description="Resolved download URL for the preferred format.")


class IndexEntry(BaseModel):
    """Record of an article already downloaded into local storage."""

    arxiv_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    primary_category: str = ""
    file_path: str = Field(description="Relative file path within the download directory.")
    file_size: int = Field(description="File size in bytes.")
    format: str = "pdf"
    downloaded_at: str = Field(description="ISO 8601 timestamp when downloaded.")
    sha256: str = ""
    url: str = ""


class DownloadStats(BaseModel):
    """Summary statistics for a download run."""

    target_limit: int = 100
    total_inspected: int = 0
    new_downloaded: int = 0
    already_existing: int = 0
    failed: int = 0
    total_bytes: int = 0
    duration_seconds: float = 0.0
    pages_visited: int = 0
