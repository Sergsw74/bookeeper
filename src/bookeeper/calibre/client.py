"""
Calibre client wrapping the calibredb command-line interface via subprocess.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CalibreClient:
    """Wrapper around the calibredb CLI for querying and updating library metadata."""

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

    def _build_base_args(self) -> List[str]:
        """Construct common CLI parameters including library path and auth."""
        args = [self.calibredb_bin]
        if self.library_path:
            expanded = (
                self.library_path
                if self.library_path.startswith("http://") or self.library_path.startswith("https://")
                else str(Path(self.library_path).expanduser().resolve())
            )
            args.extend(["--with-library", expanded])

        if self.user:
            args.extend(["--username", self.user])
        if self.password:
            args.extend(["--password", self.password])

        return args

    def is_available(self) -> bool:
        """Verify whether calibredb CLI binary exists and is executable."""
        return shutil.which(self.calibredb_bin) is not None

    def list_books(self, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Query books from Calibre library via calibredb list.

        Args:
            fields: List of metadata field names (e.g. ['id', 'title', 'authors', 'comments', 'formats'])

        Returns:
            List of book metadata dictionaries.
        """
        if not self.is_available():
            raise FileNotFoundError(
                f"calibredb binary '{self.calibredb_bin}' not found in system PATH."
            )

        cmd = self._build_base_args() + ["list", "--for-machine"]
        if fields:
            cmd.extend(["--fields", ",".join(fields)])

        logger.debug(f"Running calibredb command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"calibredb list failed (code {result.returncode}): {error_msg}")

        try:
            books_data: List[Dict[str, Any]] = json.loads(result.stdout)
            return books_data
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse calibredb JSON output: {e}\nRaw output: {result.stdout[:500]}") from e

    def update_metadata(
        self,
        book_id: int,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        comments: Optional[str] = None,
    ) -> bool:
        """
        Update book metadata fields in Calibre via calibredb set_metadata.

        Args:
            book_id: Calibre book ID.
            title: New title string.
            authors: List of author names (formatted into '&' delimited string for calibredb).
            comments: Summary/description/comments text.

        Returns:
            True if metadata was successfully updated.
        """
        if not self.is_available():
            raise FileNotFoundError(
                f"calibredb binary '{self.calibredb_bin}' not found in system PATH."
            )

        cmd = self._build_base_args() + ["set_metadata", str(book_id)]

        if title:
            cmd.extend(["--field", f"title:{title}"])
        if authors:
            authors_str = " & ".join(authors)
            cmd.extend(["--field", f"authors:{authors_str}"])
        if comments:
            cmd.extend(["--field", f"comments:{comments}"])

        logger.debug(f"Updating metadata for book {book_id}: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"calibredb set_metadata failed for book {book_id} (code {result.returncode}): {error_msg}"
            )

        return True

    def export_book(
        self,
        book_id: int,
        target_dir: Path | str,
        fmt: str = "EPUB",
    ) -> Optional[Path]:
        """
        Export a specific book format to target directory via calibredb export.
        """
        if not self.is_available():
            raise FileNotFoundError(f"calibredb binary '{self.calibredb_bin}' not found in system PATH.")

        out_path = Path(target_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

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
        if result.returncode != 0:
            logger.warning(f"calibredb export failed: {result.stderr.strip()}")
            return None

        # Locate the exported file
        candidates = list(out_path.glob(f"{book_id}_*.{fmt.lower()}"))
        if candidates:
            return candidates[0]

        all_matches = list(out_path.glob(f"*.{fmt.lower()}"))
        return all_matches[0] if all_matches else None
