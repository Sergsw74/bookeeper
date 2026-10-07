"""
Calibre client wrapping calibredb CLI and providing direct SQLite fallback for SMB shares and local paths.
"""

from contextlib import contextmanager
import json
import logging
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _copy_file_with_progress(
    src: Path,
    dst: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 256 * 1024,
) -> None:
    """Copy a file in chunks with optional progress callback (copied_bytes, total_bytes)."""
    total_bytes = src.stat().st_size
    copied = 0
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            copied += len(chunk)
            if progress_callback:
                progress_callback(copied, total_bytes)
    try:
        shutil.copystat(src, dst)
    except Exception:
        pass


def _sqlite_title_sort(title: Optional[str]) -> str:
    """Calibre-compatible title sort calculation."""
    if not title:
        return ""
    t = title.strip()
    for prefix in ("the ", "a ", "an "):
        if t.lower().startswith(prefix):
            orig_prefix = t[: len(prefix)].strip()
            return t[len(prefix) :].strip() + ", " + orig_prefix
    return t


def _sqlite_author_sort(author: Optional[str]) -> str:
    """Calibre-compatible author sort calculation (e.g. 'John Smith' -> 'Smith, John')."""
    if not author:
        return ""
    parts = author.strip().split()
    if len(parts) > 1:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return author.strip()


def _register_sqlite_functions(conn: sqlite3.Connection) -> None:
    """Register custom SQLite functions expected by Calibre's schema triggers."""
    import uuid

    conn.create_function("title_sort", 1, _sqlite_title_sort)
    conn.create_function("author_sort", 1, _sqlite_author_sort)
    conn.create_function("sort_author", 1, _sqlite_author_sort)
    conn.create_function("uuid4", 0, lambda: str(uuid.uuid4()))


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

        # Staging support for high-speed local transactions
        self.staged_db_path: Optional[Path] = None
        self.is_staged: bool = False

    @property
    def is_remote_url(self) -> bool:
        """Check if library_path is an HTTP/HTTPS remote URL."""
        p = self.library_path.strip().lower()
        return p.startswith("http://") or p.startswith("https://")

    @property
    def active_db_path(self) -> Optional[Path]:
        """Return staged DB path if database is staged, otherwise original db_path."""
        if self.is_staged and self.staged_db_path and self.staged_db_path.is_file():
            return self.staged_db_path
        return self.db_path

    @staticmethod
    def _verify_sqlite_integrity(path: Path, allow_index_warnings: bool = True) -> bool:
        """
        Verify SQLite database integrity.
        Ensures database is valid, non-empty, and core tables are queryable.
        Allows legacy Calibre index warnings if allow_index_warnings=True.
        """
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"SQLite file {path} is missing or empty.")

        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            cursor = conn.cursor()
            # Basic readability check on books table
            cursor.execute("SELECT count(*) FROM books;")
            cursor.fetchone()

            cursor.execute("PRAGMA integrity_check;")
            rows = cursor.fetchall()
            if not rows:
                raise ValueError(f"Integrity check returned no results on {path}")

            if len(rows) == 1 and rows[0][0] == "ok":
                return True

            errors = [str(r[0]) for r in rows if r and r[0]]
            # If all errors are merely index-related discrepancies (common in Calibre libraries after crashes),
            # log warning and proceed rather than failing into slow network mode.
            is_only_index_warnings = all("index" in err.lower() for err in errors)
            if allow_index_warnings and is_only_index_warnings:
                logger.warning(
                    f"SQLite database {path} has {len(errors)} minor index warning(s), "
                    f"but core tables and data are intact."
                )
                return True

            raise ValueError(f"SQLite integrity check failed on {path}: {errors[:5]}")
        finally:
            conn.close()

    def stage_database(
        self,
        staged_path: Optional[Path | str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Copy remote/mounted metadata.db to local disk for high-speed SSD transactions.
        Switches active_db_path to staged_path and sets is_staged=True.
        """
        if self.is_remote_url:
            raise RuntimeError("Cannot stage database for remote HTTP/HTTPS Calibre Content Server.")
        if not self.db_path or not self.db_path.is_file():
            raise FileNotFoundError(f"Source metadata.db not found at {self.db_path}")

        if staged_path:
            target = Path(staged_path).expanduser().resolve()
        else:
            import tempfile
            target = Path(tempfile.gettempdir()) / "bookeeper_staged_metadata.db"

        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Staging Calibre database from {self.db_path} to {target}...")
        _copy_file_with_progress(self.db_path, target, progress_callback=progress_callback)

        # Verify complete transfer
        source_size = self.db_path.stat().st_size
        staged_size = target.stat().st_size
        if source_size != staged_size:
            raise IOError(f"Staged file size mismatch: {staged_size} != {source_size}")

        # Verify integrity of staged copy
        self._verify_sqlite_integrity(target, allow_index_warnings=True)

        self.staged_db_path = target
        self.is_staged = True
        return target

    def sync_database(
        self,
        create_backup: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Sync staged metadata.db back to original library path.
        Optionally creates a backup (e.g. metadata.db.bak) on the remote share first.
        """
        if not self.is_staged or not self.staged_db_path or not self.staged_db_path.is_file():
            logger.debug("No staged database to sync.")
            return False

        if not self.db_path:
            raise RuntimeError("Original database path is not defined.")

        # 1. Verify staged DB integrity before uploading
        self._verify_sqlite_integrity(self.staged_db_path, allow_index_warnings=True)

        # 2. Create backup of original db if requested
        if create_backup and self.db_path.is_file():
            backup_path = self.db_path.with_name("metadata.db.bak")
            logger.info(f"Creating backup of original database at {backup_path}...")
            shutil.copy2(self.db_path, backup_path)

        # 3. Copy staged DB back to remote/original destination
        logger.info(f"Uploading staged database from {self.staged_db_path} to {self.db_path}...")
        tmp_target = self.db_path.with_name(f".metadata.db.tmp_{os.getpid()}")
        try:
            _copy_file_with_progress(self.staged_db_path, tmp_target, progress_callback=progress_callback)
            tmp_target.replace(self.db_path)
        except OSError:
            # Fallback if filesystem doesn't support atomic replace across temporary files
            if tmp_target.exists():
                try:
                    tmp_target.unlink()
                except Exception:
                    pass
            _copy_file_with_progress(self.staged_db_path, self.db_path, progress_callback=progress_callback)

        return True

    def cleanup_staged(self, delete_file: bool = True) -> None:
        """Reset staging state and optionally delete the staged copy."""
        if delete_file and self.staged_db_path and self.staged_db_path.is_file():
            try:
                self.staged_db_path.unlink()
            except Exception as e:
                logger.warning(f"Could not remove staged db file {self.staged_db_path}: {e}")
        self.staged_db_path = None
        self.is_staged = False

    @contextmanager
    def staged_session(
        self,
        staged_path: Optional[Path | str] = None,
        create_backup: bool = True,
        auto_sync: bool = True,
        cleanup_on_finish: bool = True,
    ):
        """
        Context manager for staging metadata.db locally during a session.
        Automatically syncs back upon exiting context if auto_sync=True.
        """
        staged = False
        if not self.is_remote_url and self.db_path and self.db_path.is_file():
            self.stage_database(staged_path)
            staged = True

        try:
            yield self
            if staged and auto_sync:
                self.sync_database(create_backup=create_backup)
        finally:
            if staged and cleanup_on_finish:
                self.cleanup_staged(delete_file=True)

    def is_available(self) -> bool:
        """
        Verify if library is accessible.
        Returns True if:
        - active_db_path exists on filesystem / mounted SMB share, OR
        - calibredb binary is available.
        """
        if self.active_db_path and self.active_db_path.is_file():
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
        List books from Calibre library. Prefers local staged SQLite if active,
        then calibredb CLI if available, and falls back to direct SQLite reading.
        """
        # 0. Local staged DB priority
        if self.is_staged and self.staged_db_path and self.staged_db_path.is_file():
            return self._list_books_from_sqlite()

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
        if self.active_db_path and self.active_db_path.is_file():
            return self._list_books_from_sqlite()

        raise RuntimeError(
            f"Unable to access Calibre library at '{self.library_path}'. "
            f"Neither calibredb CLI nor accessible metadata.db was found."
        )

    def _list_books_from_sqlite(self) -> List[Dict[str, Any]]:
        """Query metadata.db directly in read-only mode over local or SMB filesystem."""
        db_to_use = self.active_db_path
        if not db_to_use or not db_to_use.is_file():
            raise FileNotFoundError(f"Database file not found at {db_to_use}")

        uri = f"file:{db_to_use}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        _register_sqlite_functions(conn)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 1. Preload all book formats in ONE bulk query (avoids N individual queries)
        cursor.execute("SELECT book, format FROM data")
        book_formats: Dict[int, List[str]] = {}
        for d_row in cursor.fetchall():
            b_id = d_row[0]
            fmt = d_row[1]
            if fmt:
                book_formats.setdefault(b_id, []).append(fmt.upper())

        # 2. Bulk query all books with authors and comments
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
            formats = book_formats.get(book_id, [])

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
        """Update metadata using staged SQLite, calibredb CLI, or direct SQLite on SMB share."""
        # 0. Local staged DB priority for fast microsecond updates
        if self.is_staged and self.staged_db_path and self.staged_db_path.is_file():
            return self._update_metadata_sqlite(book_id, title, authors, comments)

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
        if self.active_db_path and self.active_db_path.is_file():
            return self._update_metadata_sqlite(book_id, title, authors, comments)

        raise RuntimeError(f"Cannot update metadata: calibredb failed and metadata.db not writable.")

    def _update_metadata_sqlite(
        self,
        book_id: int,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        comments: Optional[str] = None,
    ) -> bool:
        """Perform direct SQLite update on title, authors, and comments in metadata.db."""
        db_to_use = self.active_db_path
        if not db_to_use or not db_to_use.is_file():
            raise FileNotFoundError(f"Database file not found at {db_to_use}")

        conn = sqlite3.connect(str(db_to_use))
        _register_sqlite_functions(conn)
        cursor = conn.cursor()
        try:
            if title:
                cursor.execute("UPDATE books SET title = ? WHERE id = ?", (title.strip(), book_id))

            if comments is not None:
                cursor.execute(
                    """
                    INSERT INTO comments (book, text) VALUES (?, ?)
                    ON CONFLICT(book) DO UPDATE SET text = excluded.text
                    """,
                    (book_id, comments.strip()),
                )

            if authors:
                author_ids = []
                for a in authors:
                    a_clean = a.strip()
                    if not a_clean:
                        continue
                    cursor.execute("SELECT id FROM authors WHERE name = ? COLLATE NOCASE", (a_clean,))
                    row = cursor.fetchone()
                    if row:
                        author_ids.append(row[0])
                    else:
                        sort_val = _sqlite_author_sort(a_clean)
                        cursor.execute("INSERT INTO authors (name, sort) VALUES (?, ?)", (a_clean, sort_val))
                        author_ids.append(cursor.lastrowid)

                if author_ids:
                    cursor.execute("DELETE FROM books_authors_link WHERE book = ?", (book_id,))
                    for aid in author_ids:
                        cursor.execute(
                            "INSERT OR IGNORE INTO books_authors_link (book, author) VALUES (?, ?)",
                            (book_id, aid),
                        )
                    primary_sort = " & ".join(_sqlite_author_sort(a.strip()) for a in authors if a.strip())
                    cursor.execute("UPDATE books SET author_sort = ? WHERE id = ?", (primary_sort, book_id))

            cursor.execute("UPDATE books SET last_modified = CURRENT_TIMESTAMP WHERE id = ?", (book_id,))
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
        db_to_use = self.active_db_path
        if self.local_path and db_to_use and db_to_use.is_file():
            conn = sqlite3.connect(f"file:{db_to_use}?mode=ro", uri=True)
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
