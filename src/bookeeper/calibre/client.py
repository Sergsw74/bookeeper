"""
Calibre database client and wrapper supporting direct SQLite metadata.db access
and optional calibredb CLI execution.
"""

import json
import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from bookeeper.config import CalibreSettings

logger = logging.getLogger(__name__)


class BookRecord(BaseModel):
    """Normalized book record from Calibre library."""

    id: int
    title: str
    authors: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    pubdate: Optional[str] = None
    series: Optional[str] = None
    series_index: Optional[float] = None
    comments: Optional[str] = None
    formats: Dict[str, Path] = Field(default_factory=dict)  # e.g. {"EPUB": Path(...)}

    @property
    def author_display(self) -> str:
        return ", ".join(self.authors) if self.authors else "Unknown Author"

    def get_preferred_file(self, preferred_formats: List[str]) -> Optional[Path]:
        """Return the path of the highest-priority format present."""
        for fmt in preferred_formats:
            fmt_upper = fmt.upper()
            if fmt_upper in self.formats:
                return self.formats[fmt_upper]
        # Fallback to any format available
        if self.formats:
            return next(iter(self.formats.values()))
        return None


class CalibreClient:
    """Client for reading books and metadata from a Calibre library."""

    def __init__(self, settings: CalibreSettings):
        self.settings = settings
        self.library_path = Path(settings.library_path).expanduser().resolve()
        self.db_path = self.library_path / "metadata.db"

    def is_available(self) -> bool:
        """Check if Calibre library metadata.db exists and is accessible."""
        return self.db_path.is_file()

    def list_books(self, limit: Optional[int] = None) -> List[BookRecord]:
        """List all books from the library with their associated format files."""
        if not self.is_available():
            raise FileNotFoundError(f"Calibre metadata.db not found at {self.db_path}")

        # Connect to metadata.db in read-only URI mode to avoid locks
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
        SELECT
            b.id,
            b.title,
            b.path,
            b.pubdate,
            (SELECT GROUP_CONCAT(a.name, ' & ')
             FROM books_authors_link bal
             JOIN authors a ON bal.author = a.id
             WHERE bal.book = b.id) AS authors,
            (SELECT GROUP_CONCAT(t.name, ',')
             FROM books_tags_link btl
             JOIN tags t ON btl.tag = t.id
             WHERE btl.book = b.id) AS tags,
            (SELECT text FROM comments WHERE book = b.id) AS comments
        FROM books b
        ORDER BY b.id ASC
        """
        if limit:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query)
        rows = cursor.fetchall()

        books: List[BookRecord] = []
        for r in rows:
            book_id = r["id"]
            rel_folder = r["path"]
            book_dir = self.library_path / rel_folder

            authors = [a.strip() for a in (r["authors"] or "").split("&") if a.strip()]
            tags = [t.strip() for t in (r["tags"] or "").split(",") if t.strip()]

            # Fetch file formats for this book
            cursor.execute(
                "SELECT format, name FROM data WHERE book = ?",
                (book_id,),
            )
            data_rows = cursor.fetchall()
            formats: Dict[str, Path] = {}
            for d in data_rows:
                fmt = d["format"].upper()
                file_name = f"{d['name']}.{fmt.lower()}"
                full_path = book_dir / file_name
                if full_path.is_file():
                    formats[fmt] = full_path

            books.append(
                BookRecord(
                    id=book_id,
                    title=r["title"] or "Untitled",
                    authors=authors,
                    tags=tags,
                    pubdate=r["pubdate"],
                    comments=r["comments"],
                    formats=formats,
                )
            )

        conn.close()
        return books

    def get_book(self, book_id: int) -> Optional[BookRecord]:
        """Fetch a specific book record by its Calibre ID."""
        if not self.is_available():
            raise FileNotFoundError(f"Calibre metadata.db not found at {self.db_path}")

        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
        SELECT
            b.id,
            b.title,
            b.path,
            b.pubdate,
            (SELECT GROUP_CONCAT(a.name, ' & ')
             FROM books_authors_link bal
             JOIN authors a ON bal.author = a.id
             WHERE bal.book = b.id) AS authors,
            (SELECT GROUP_CONCAT(t.name, ',')
             FROM books_tags_link btl
             JOIN tags t ON btl.tag = t.id
             WHERE btl.book = b.id) AS tags,
            (SELECT text FROM comments WHERE book = b.id) AS comments
        FROM books b
        WHERE b.id = ?
        """
        cursor.execute(query, (book_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        rel_folder = row["path"]
        book_dir = self.library_path / rel_folder
        authors = [a.strip() for a in (row["authors"] or "").split("&") if a.strip()]
        tags = [t.strip() for t in (row["tags"] or "").split(",") if t.strip()]

        cursor.execute(
            "SELECT format, name FROM data WHERE book = ?",
            (book_id,),
        )
        data_rows = cursor.fetchall()
        formats: Dict[str, Path] = {}
        for d in data_rows:
            fmt = d["format"].upper()
            file_name = f"{d['name']}.{fmt.lower()}"
            full_path = book_dir / file_name
            if full_path.is_file():
                formats[fmt] = full_path

        conn.close()
        return BookRecord(
            id=book_id,
            title=row["title"] or "Untitled",
            authors=authors,
            tags=tags,
            pubdate=row["pubdate"],
            comments=row["comments"],
            formats=formats,
        )

    def search_books(self, query: str) -> List[BookRecord]:
        """Simple text search matching title or authors."""
        all_books = self.list_books()
        q = query.lower()
        return [
            b
            for b in all_books
            if q in b.title.lower()
            or any(q in a.lower() for a in b.authors)
            or any(q in t.lower() for t in b.tags)
        ]
