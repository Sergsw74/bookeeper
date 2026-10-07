"""
Calibre client wrapping calibredb CLI and providing direct SQLite fallback for SMB shares and local paths.
"""

import json
import logging
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CalibreClient:
    """
    Client for interacting with a Calibre library. Supports:
    1. Direct filesystem / SMB share access to metadata.db (no calibredb required).
    2. calibredb CLI tool via subprocess when available.
    3. Remote Content Server URLs (http://host:port/#library).
    """

    def __init__(
        self,
        library_path: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        calibredb_bin: Optional[str] = None,
    ):
        self.library_path = library_path
        self.user = user
        self.password = password
        self.calibredb_bin = calibredb_bin or shutil.which("calibredb") or "calibredb"

        # Expand local / SMB filesystem path
        if not self.is_remote_url:
            self.local_path = Path(self.library_path).expanduser().resolve()
            self.db_path = self.local_path / "metadata.db"
        else:
            self.local_path = None
            self.db_path = None

    @property
    def is_remote_url(self) -> bool:
        """Check if library_path is an HTTP/HTTPS remote URL."""
        p = self.library_path.strip().lower()
        return p.startswith("http://") or p.startswith("https://")

    def is_available(self) -> bool:
        """
        Verify if library is accessible.
        Returns True if:
        - metadata.db exists on filesystem / mounted SMB share, OR
        - calibredb binary is available.
        """
        if self.db_path and self.db_path.is_file():
            return True
        return shutil.which(self.calibredb_bin) is not None

    def _build_base_args(self) -> List[str]:
        """Construct CLI arguments for calibredb."""
        args = [self.calibredb_bin]
        if self.library_path:
            expanded = (
                self.library_path
                if self.is_remote_url
                else str(self.local_path)
            )
            args.extend(["--with-library", expanded])

        if self.user:
            args.extend(["--username", self.user])
        if self.password:
            args.extend(["--password", self.password])

        return args

    def list_books(self, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List books from Calibre library. Prefers calibredb CLI if available;
        falls back to direct SQLite reading on SMB share or local directory.
        """
        # 1. If calibredb is installed, use calibredb list
        if shutil.which(self.calibredb_bin):
            try:
                cmd = self._build_base_args() + ["list", "--for-machine"]
                if fields:
                    cmd.extend(["--fields", ",".join(fields)])

                logger.debug(f"Running calibredb command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    return json.loads(result.stdout)
                logger.warning(
                    f"calibredb failed (code {result.returncode}), attempting direct SQLite read..."
                )
            except Exception as e:
                logger.debug(f"calibredb execution failed: {e}")

        # 2. Fallback: Direct SQLite query on metadata.db (ideal for SMB shares)
        if self.db_path and self.db_path.is_file():
            return self._list_books_from_sqlite()

        raise RuntimeError(
            f"Unable to access Calibre library at '{self.library_path}'. "
            f"Neither calibredb CLI nor accessible metadata.db was found."
        )

    def _list_books_from_sqlite(self) -> List[Dict[str, Any]]:
        """Query metadata.db directly in read-only mode over local or SMB filesystem."""
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
            (SELECT text FROM comments WHERE book = b.id) AS comments
        FROM books b
        ORDER BY b.id ASC
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        books: List[Dict[str, Any]] = []
        for r in rows:
            book_id = r["id"]
            rel_folder = r["path"]
            book_dir = self.local_path / rel_folder if self.local_path else Path(rel_folder)

            authors = [a.strip() for a in (r["authors"] or "").split("&") if a.strip()]

            # Format lookup
            cursor.execute("SELECT format, name FROM data WHERE book = ?", (book_id,))
            formats = []
            for d in cursor.fetchall():
                fmt = d["format"].upper()
                formats.append(fmt)

            books.append(
                {
                    "id": book_id,
                    "title": r["title"] or "Untitled",
                    "authors": authors,
                    "comments": r["comments"] or "",
                    "formats": formats,
                    "path": str(book_dir),
                }
            )

        conn.close()
        return books

    def update_metadata(
        self,
        book_id: int,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        comments: Optional[str] = None,
    ) -> bool:
        """Update metadata using calibredb CLI or direct SQLite on SMB share."""
        # 1. Try calibredb CLI
        if shutil.which(self.calibredb_bin):
            try:
                cmd = self._build_base_args() + ["set_metadata", str(book_id)]
                if title:
                    cmd.extend(["--field", f"title:{title}"])
                if authors:
                    authors_str = " & ".join(authors)
                    cmd.extend(["--field", f"authors:{authors_str}"])
                if comments:
                    cmd.extend(["--field", f"comments:{comments}"])

                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    return True
                logger.warning(f"calibredb set_metadata failed (code {result.returncode}), trying direct SQLite...")
            except Exception as e:
                logger.debug(f"calibredb set_metadata error: {e}")

        # 2. Direct SQLite fallback
        if self.db_path and self.db_path.is_file():
            return self._update_metadata_sqlite(book_id, title, comments)

        raise RuntimeError(f"Cannot update metadata: calibredb failed and metadata.db not writable.")

    def _update_metadata_sqlite(
        self,
        book_id: int,
        title: Optional[str] = None,
        comments: Optional[str] = None,
    ) -> bool:
        """Perform direct SQLite update on title/comments in metadata.db."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        try:
            if title:
                cursor.execute("UPDATE books SET title = ? WHERE id = ?", (title, book_id))
            if comments:
                cursor.execute(
                    """
                    INSERT INTO comments (book, text) VALUES (?, ?)
                    ON CONFLICT(book) DO UPDATE SET text = excluded.text
                    """,
                    (book_id, comments),
                )
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Direct SQLite metadata update failed: {e}") from e
        finally:
            conn.close()

    def export_book(
        self,
        book_id: int,
        target_dir: Path | str,
        fmt: str = "EPUB",
    ) -> Optional[Path]:
        """
        Locate or export book format.
        If accessing via SMB/local path, copies file directly from book directory.
        Otherwise calls calibredb export.
        """
        out_path = Path(target_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Direct file resolution if on SMB share or local directory
        if self.local_path and self.db_path and self.db_path.is_file():
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT b.path, d.name, d.format
                FROM books b
                JOIN data d ON b.id = d.book
                WHERE b.id = ? AND UPPER(d.format) = ?
                """,
                (book_id, fmt.upper()),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                rel_path, name, ext = row
                source_file = self.local_path / rel_path / f"{name}.{ext.lower()}"
                if source_file.is_file():
                    dest_file = out_path / f"{book_id}_{name}.{ext.lower()}"
                    shutil.copy2(source_file, dest_file)
                    return dest_file

        # 2. Fallback to calibredb export
        if shutil.which(self.calibredb_bin):
            cmd = self._build_base_args() + [
                "export",
                str(book_id),
                "--to-dir",
                str(out_path),
                "--formats",
                fmt.upper(),
                "--dont-save-cover",
                "--dont-write-opf",
                "--template",
                f"{{id}}_{{title}}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                candidates = list(out_path.glob(f"{book_id}_*.{fmt.lower()}"))
                if candidates:
                    return candidates[0]
                all_matches = list(out_path.glob(f"*.{fmt.lower()}"))
                return all_matches[0] if all_matches else None

        return None
