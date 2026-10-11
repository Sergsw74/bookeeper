"""
Calibre integration, folder book sources, and ebook parsing modules.
"""

from pathlib import Path
from typing import Any, Optional, Union

from bookeeper.calibre.client import CalibreClient
from bookeeper.calibre.folder_client import FolderBookClient
from bookeeper.calibre.parser import BookParser, Section


def get_book_client(
    cfg: Any,
    calibre_path: Optional[str] = None,
    books_dir: Optional[str] = None,
) -> Union[CalibreClient, FolderBookClient]:
    """
    Resolve and return appropriate book source client:
    - If books_dir is explicitly specified (CLI or config), returns FolderBookClient.
    - If calibre_path is specified:
        - If remote URL or contains metadata.db, returns CalibreClient.
        - If local directory containing books/index.json without metadata.db, returns FolderBookClient.
    - If cfg.calibre_library_path contains metadata.db or is remote URL, returns CalibreClient.
    - If cfg.calibre_library_path is a directory with books, returns FolderBookClient.
    - If Calibre is unavailable/missing but local downloads directory exists (e.g. archive_downloader/downloads),
      automatically falls back to FolderBookClient.
    - Otherwise returns CalibreClient.
    """
    # 1. Explicit books_dir parameter (CLI override)
    if books_dir and str(books_dir).strip():
        return FolderBookClient(books_dir)

    # 2. Configured books_dir in settings
    configured_books_dir = getattr(cfg, "books_dir", None)
    if configured_books_dir and str(configured_books_dir).strip():
        return FolderBookClient(configured_books_dir)

    # 3. Explicit calibre_path parameter (CLI override)
    if calibre_path and str(calibre_path).strip():
        cp = str(calibre_path).strip()
        cp_lower = cp.lower()
        if cp_lower.startswith("http://") or cp_lower.startswith("https://"):
            return CalibreClient(
                library_path=cp,
                user=getattr(cfg, "calibre_user", None),
                password=getattr(cfg, "calibre_password", None),
            )
        p = Path(cp).expanduser().resolve()
        if (p / "metadata.db").is_file():
            return CalibreClient(
                library_path=str(p),
                user=getattr(cfg, "calibre_user", None),
                password=getattr(cfg, "calibre_password", None),
            )
        if p.is_dir():
            fc = FolderBookClient(p)
            if fc.is_available():
                return fc
        return CalibreClient(
            library_path=cp,
            user=getattr(cfg, "calibre_user", None),
            password=getattr(cfg, "calibre_password", None),
        )

    # 4. Configured calibre_library_path
    lib_path = getattr(cfg, "calibre_library_path", "") or ""
    if lib_path and str(lib_path).strip():
        lp = str(lib_path).strip()
        lp_lower = lp.lower()
        if lp_lower.startswith("http://") or lp_lower.startswith("https://"):
            return CalibreClient(
                library_path=lp,
                user=getattr(cfg, "calibre_user", None),
                password=getattr(cfg, "calibre_password", None),
            )
        p = Path(lp).expanduser().resolve()
        if (p / "metadata.db").is_file():
            return CalibreClient(
                library_path=str(p),
                user=getattr(cfg, "calibre_user", None),
                password=getattr(cfg, "calibre_password", None),
            )
        if p.is_dir():
            fc = FolderBookClient(p)
            if fc.is_available():
                return fc

    # 5. Automatic fallback to local downloads folder if present and non-empty
    for candidate in ("archive_downloader/downloads", "downloads"):
        cand_p = Path(candidate).resolve()
        if cand_p.is_dir():
            fc = FolderBookClient(cand_p)
            if fc.is_available() and len(fc.list_books()) > 0:
                return fc

    # 6. Default to CalibreClient
    return CalibreClient(
        library_path=lib_path,
        user=getattr(cfg, "calibre_user", None),
        password=getattr(cfg, "calibre_password", None),
    )


__all__ = [
    "CalibreClient",
    "FolderBookClient",
    "get_book_client",
    "BookParser",
    "Section",
]
