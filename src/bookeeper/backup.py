"""
Knowledge backup and restoration engine for Bookeeper.

Provides robust compression (ZIP and 7-Zip) and restoration of knowledge graph
artifacts, Obsidian vaults, book states, cached chunks, and checkpoint files.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import py7zr

logger = logging.getLogger(__name__)

# Standard artifact directories and filenames in the output folder
KNOWLEDGE_FILES = {
    "knowledge_graph.json",
    "knowledge_graph.graphml",
    ".bookeeper_state.json",
    "concept_embeddings.npz",
    "verification_report.json",
    "cross_check_report.json",
}

KNOWLEDGE_DIR_NAMES = {
    "obsidian_vault",
    ".book_states",
    "chunks",
    "lightrag",
}

# Supported document formats for indexed books
BOOK_EXTENSIONS = {
    ".pdf",
    ".epub",
    ".mobi",
    ".azw3",
    ".azw",
    ".djvu",
    ".txt",
    ".opf",
    ".json",
    ".docx",
    ".fb2",
    ".cbz",
    ".cbr",
}

# Directories and file patterns strictly excluded from knowledge backups
EXCLUDED_DIR_NAMES = {
    "calibre_ingest",
    "__pycache__",
    ".git",
}

EXCLUDED_EXTENSIONS = {
    ".db",
    ".db-journal",
    ".db-wal",
    ".tmp",
    ".bak",
    ".zip",
    ".7z",
    ".tar",
    ".gz",
    ".tgz",
}

EXCLUDED_EXACT_FILENAMES = {
    ".DS_Store",
    "Thumbs.db",
    ".staged_metadata.db",
    ".test_reindex.db",
}


import base64

# Characters strictly forbidden on Windows NTFS / FAT32 filesystems
_WINDOWS_FORBIDDEN_CHARS = set('\\/*?:"<>|')

# Reserved device names forbidden on Windows
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def is_path_component_unsupported(name: str) -> bool:
    """Check if a directory or filename component is unsupported on Windows or cross-platform filesystems."""
    if not name:
        return True
    # Control characters (\x00-\x1f, \x7f-\x9f) or Windows forbidden characters
    if any(ord(c) < 32 or (127 <= ord(c) <= 159) or c in _WINDOWS_FORBIDDEN_CHARS for c in name):
        return True
    # Trailing dot or trailing space
    if name.endswith(" ") or name.endswith("."):
        return True
    # Reserved device names
    if "." in name and not name.startswith("."):
        stem, _ = name.rsplit(".", 1)
        if stem.endswith(" ") or stem.endswith("."):
            return True
        if stem.upper() in _WINDOWS_RESERVED_NAMES:
            return True
    else:
        if name.upper() in _WINDOWS_RESERVED_NAMES:
            return True
    return False


def encode_safe_component(name: str) -> str:
    """
    Encode an unsupported filename or folder name component into a safe cross-platform name
    using base64 (@b64_<encoded>.<ext>).
    """
    if not is_path_component_unsupported(name):
        return name
    if "." in name and not name.startswith("."):
        stem, ext = name.rsplit(".", 1)
        if not is_path_component_unsupported(ext):
            b64 = base64.urlsafe_b64encode(stem.encode("utf-8")).decode("ascii").rstrip("=")
            return f"@b64_{b64}.{ext}"
    b64 = base64.urlsafe_b64encode(name.encode("utf-8")).decode("ascii").rstrip("=")
    return f"@b64_{b64}"


def decode_safe_component(name: str) -> str:
    """Decode a @b64_<encoded> component back to its original name."""
    m = re.match(r"^@b64_([A-Za-z0-9_-]+)(?:\.([^.]+))?$", name)
    if not m:
        return name
    raw_b64 = m.group(1)
    ext = m.group(2)
    padding = "=" * (-len(raw_b64) % 4)
    decoded = base64.urlsafe_b64decode((raw_b64 + padding).encode("ascii")).decode("utf-8", errors="replace")
    if ext is not None:
        return f"{decoded}.{ext}"
    return decoded


def encode_safe_relative_path(rel_path: str) -> str:
    """Encode any unsupported path components in a relative path into safe base64-encoded names."""
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    return "/".join(encode_safe_component(p) for p in parts)


def decode_safe_relative_path(rel_path: str) -> str:
    """Decode any base64-encoded components in a relative path back to original form."""
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    return "/".join(decode_safe_component(p) for p in parts)


def sanitize_archive_relative_path(rel_path: str) -> str:
    """
    Sanitize an archive relative path for safe writing on all OS filesystems (including Windows).
    If a component contains unsupported special or control characters (e.g. newlines),
    it is encoded into a safe base64 representation.
    """
    # 1. Decode any existing @b64 components first
    decoded_path = decode_safe_relative_path(rel_path)
    # 2. Encode any unsupported components with base64
    return encode_safe_relative_path(decoded_path)



def detect_archive_format(path: Path) -> str:

    """
    Detect whether an archive is '7z' or 'zip' based on file extension or magic bytes.
    """
    suffix = path.suffix.lower()
    if suffix == ".7z":
        return "7z"
    if suffix == ".zip":
        return "zip"

    # Try detecting by magic bytes if file exists
    if path.is_file():
        try:
            with open(path, "rb") as f:
                header = f.read(6)
                if header.startswith(b"7z\xbc\xaf'\x1c"):
                    return "7z"
                if header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06"):
                    return "zip"
        except Exception:
            pass

    return "zip"


def is_excluded_file(file_path: Path, destination_archive: Optional[Path] = None) -> bool:
    """
    Check if a file should be excluded from the knowledge backup.
    """
    if destination_archive and file_path.resolve() == destination_archive.resolve():
        return True

    if file_path.name in EXCLUDED_EXACT_FILENAMES:
        return True

    if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    # Check parent directory names
    for part in file_path.parts[:-1]:
        if part in EXCLUDED_DIR_NAMES:
            return True

    return False


def collect_knowledge_files(
    source_dir: Path,
    include_chunks: bool = True,
    include_obsidian: bool = True,
    include_states: bool = True,
    include_lightrag: bool = True,
    destination_archive: Optional[Path] = None,
    custom_dirs: Optional[Dict[str, Path]] = None,
) -> List[Tuple[Path, str]]:
    """
    Collect all knowledge files to be included in the backup.
    Returns a sorted list of (absolute_path, relative_archive_path).
    """
    source_dir = source_dir.expanduser().resolve()
    items: Dict[str, Path] = {}

    if source_dir.is_dir():
        for root, dirs, files in os.walk(source_dir):
            root_path = Path(root)
            rel_root = root_path.relative_to(source_dir)

            # Filter out excluded directories in-place
            dirs[:] = [
                d
                for d in dirs
                if d not in EXCLUDED_DIR_NAMES
                and not (d == "chunks" and not include_chunks)
                and not (d == "obsidian_vault" and not include_obsidian)
                and not (d == ".book_states" and not include_states)
                and not (d == "lightrag" and not include_lightrag)
            ]

            for fname in files:
                fpath = root_path / fname
                if is_excluded_file(fpath, destination_archive):
                    continue

                rel_path = (rel_root / fname).as_posix()
                if rel_path == ".":
                    rel_path = fname

                items[rel_path] = fpath

    # Incorporate any custom locations passed outside source_dir
    if custom_dirs:
        # 1. Custom chunks directory
        if include_chunks and "chunks_dir" in custom_dirs:
            cdir = custom_dirs["chunks_dir"].expanduser().resolve()
            if cdir.is_dir() and (not source_dir.is_dir() or not cdir.is_relative_to(source_dir)):
                for cf in cdir.glob("book_*_chunks.json"):
                    if cf.is_file():
                        rel = f"chunks/{cf.name}"
                        items[rel] = cf

        # 2. Custom state file
        if "state_file" in custom_dirs:
            sfile = custom_dirs["state_file"].expanduser().resolve()
            if sfile.is_file() and (not source_dir.is_dir() or not sfile.is_relative_to(source_dir)):
                items[".bookeeper_state.json"] = sfile

        # 3. Custom book states directory
        if include_states and "states_dir" in custom_dirs:
            sdir = custom_dirs["states_dir"].expanduser().resolve()
            if sdir.is_dir() and (not source_dir.is_dir() or not sdir.is_relative_to(source_dir)):
                for sf in sdir.glob("book_*_state.json"):
                    if sf.is_file():
                        rel = f".book_states/{sf.name}"
                        items[rel] = sf

        # 4. Custom obsidian vault
        if include_obsidian and "obsidian_dir" in custom_dirs:
            odir = custom_dirs["obsidian_dir"].expanduser().resolve()
            if odir.is_dir() and (not source_dir.is_dir() or not odir.is_relative_to(source_dir)):
                for root, _, files in os.walk(odir):
                    for fname in files:
                        fpath = Path(root) / fname
                        if is_excluded_file(fpath, destination_archive):
                            continue
                        rel_note = fpath.relative_to(odir).as_posix()
                        items[f"obsidian_vault/{rel_note}"] = fpath

    result = [(p, encode_safe_relative_path(arcname)) for arcname, p in items.items()]
    result.sort(key=lambda x: x[1])
    return result


def collect_book_files(
    books_dir: Path,
    source_dir: Optional[Path] = None,
    indexed_only: bool = True,
    destination_archive: Optional[Path] = None,
) -> List[Tuple[Path, str]]:
    """
    Collect original book files from books_dir to include in a full backup.
    If indexed_only is True, extracts referenced book titles and IDs from knowledge_graph.json
    in source_dir, and only archives the matching book files (and index.json/companion .opf).
    If indexed_only is False (or no indexed books found), archives all valid book documents.
    Returns a sorted list of (absolute_path, relative_archive_path) with prefix 'books/'.
    """
    books_dir = books_dir.expanduser().resolve()
    if not books_dir.is_dir():
        logger.warning(f"Books directory not found: {books_dir}")
        return []

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower()) if s else ""

    indexed_titles: Set[str] = set()
    indexed_bids: Set[str] = set()

    if indexed_only and source_dir:
        kg_path = source_dir.expanduser().resolve() / "knowledge_graph.json"
        if kg_path.is_file():
            try:
                with open(kg_path, "r", encoding="utf-8") as f:
                    kg_data = json.load(f)
                for n in kg_data.get("nodes", []):
                    ntype = n.get("type", "") or ""
                    ncat = n.get("category", "") or ""
                    if ntype.lower() == "book" or ncat == "Book":
                        t = n.get("title") or n.get("name")
                        if t:
                            norm_t = _norm(str(t))
                            if norm_t:
                                indexed_titles.add(norm_t)
                        bid = n.get("book_id") or n.get("id")
                        if bid is not None:
                            indexed_bids.add(str(bid).replace("book:", "").strip())
                    elif ntype.lower() == "chunk" or ncat == "Chunk":
                        bt = n.get("book_title")
                        if bt:
                            norm_bt = _norm(str(bt))
                            if norm_bt:
                                indexed_titles.add(norm_bt)
                        bid = n.get("book_id")
                        if bid is not None:
                            indexed_bids.add(str(bid).replace("book:", "").strip())
            except Exception as e:
                logger.warning(f"Failed to inspect knowledge graph for indexed books: {e}")

    items: Dict[str, Path] = {}

    # Always include index.json if present in books_dir
    index_file = books_dir / "index.json"
    index_entries: Dict[str, Any] = {}
    if index_file.is_file():
        items["books/index.json"] = index_file
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_entries = json.load(f)
        except Exception:
            pass

    if indexed_only and indexed_titles:
        # 1. Match from index.json entries
        if index_entries:
            for k, val in index_entries.items():
                if not isinstance(val, dict):
                    continue
                v_title = _norm(str(val.get("title", "")))
                v_aid = str(val.get("arxiv_id", "")).strip()
                k_clean = str(k).strip()

                is_match = False
                if v_title and any(v_title == it or v_title in it or it in v_title for it in indexed_titles):
                    is_match = True
                elif (v_aid and v_aid in indexed_bids) or (k_clean and k_clean in indexed_bids):
                    is_match = True

                if is_match:
                    fpath = val.get("file_path")
                    if fpath:
                        target = books_dir / fpath
                        if target.is_file() and not is_excluded_file(target, destination_archive):
                            rel = target.relative_to(books_dir).as_posix()
                            items[f"books/{rel}"] = target
                            # Companion .opf if exists
                            opf = target.with_suffix(".opf")
                            if opf.is_file():
                                rel_opf = opf.relative_to(books_dir).as_posix()
                                items[f"books/{rel_opf}"] = opf

        # 2. Also match directly on disk filenames
        for root, dirs, files in os.walk(books_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]
            root_p = Path(root)
            for fname in files:
                fpath = root_p / fname
                if fpath.suffix.lower() not in BOOK_EXTENSIONS:
                    continue
                if is_excluded_file(fpath, destination_archive):
                    continue
                norm_stem = _norm(fpath.stem)
                if norm_stem and any(norm_stem == it or norm_stem in it or it in norm_stem for it in indexed_titles):
                    rel = fpath.relative_to(books_dir).as_posix()
                    items[f"books/{rel}"] = fpath
                    opf = fpath.with_suffix(".opf")
                    if opf.is_file():
                        rel_opf = opf.relative_to(books_dir).as_posix()
                        items[f"books/{rel_opf}"] = opf

    else:
        # All books from books_dir
        for root, dirs, files in os.walk(books_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]
            root_p = Path(root)
            for fname in files:
                fpath = root_p / fname
                if fpath.suffix.lower() not in BOOK_EXTENSIONS:
                    continue
                if is_excluded_file(fpath, destination_archive):
                    continue
                rel = fpath.relative_to(books_dir).as_posix()
                items[f"books/{rel}"] = fpath

    result = [(p, encode_safe_relative_path(arcname)) for arcname, p in items.items()]
    result.sort(key=lambda x: x[1])
    return result


def clean_book_artifacts(target_books_dir: Path) -> Dict[str, int]:
    """Remove existing index.json and temporary items from target_books_dir before clean restore."""
    target_books_dir = target_books_dir.expanduser().resolve()
    stats = {"files_removed": 0, "dirs_removed": 0}
    if not target_books_dir.is_dir():
        return stats

    idx = target_books_dir / "index.json"
    if idx.is_file():
        try:
            idx.unlink()
            stats["files_removed"] += 1
        except Exception:
            pass
    return stats


def _compute_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def clean_knowledge_artifacts(target_dir: Path) -> Dict[str, int]:
    """
    Remove existing knowledge artifacts in target_dir prior to a clean restore.
    Specifically removes:
      - knowledge_graph.json and knowledge_graph.graphml
      - obsidian_vault directory
      - .book_states directory
      - .bookeeper_state.json
      - chunks directory
      - lightrag directory
      - verification and cross-check reports
    Preserves calibre_ingest and other unrelated files.
    """
    target_dir = target_dir.expanduser().resolve()
    stats = {"files_removed": 0, "dirs_removed": 0}

    if not target_dir.is_dir():
        return stats

    # Remove known files
    for fname in KNOWLEDGE_FILES:
        fp = target_dir / fname
        if fp.is_file():
            try:
                fp.unlink()
                stats["files_removed"] += 1
            except Exception as e:
                logger.warning(f"Failed to remove {fp}: {e}")

    # Remove any verification/cross-check reports
    for rp in target_dir.glob("*verification_report*.json"):
        if rp.is_file():
            try:
                rp.unlink()
                stats["files_removed"] += 1
            except Exception:
                pass
    for cp in target_dir.glob("*cross_check*.json"):
        if cp.is_file():
            try:
                cp.unlink()
                stats["files_removed"] += 1
            except Exception:
                pass

    # Remove known directories
    for dname in KNOWLEDGE_DIR_NAMES:
        dp = target_dir / dname
        if dp.is_dir():
            try:
                shutil.rmtree(dp)
                stats["dirs_removed"] += 1
            except Exception as e:
                logger.warning(f"Failed to remove directory {dp}: {e}")

    return stats


def create_backup(
    source_dir: Path,
    destination: Optional[Path] = None,
    format_type: Optional[str] = None,
    backup_type: str = "knowledge",
    books_dir: Optional[Path] = None,
    indexed_only: bool = True,
    include_chunks: bool = True,
    include_obsidian: bool = True,
    include_states: bool = True,
    include_lightrag: bool = True,
    custom_dirs: Optional[Dict[str, Path]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Create a compressed knowledge or full backup archive (.zip or .7z).
    If backup_type is 'full', also archives the original source books from books_dir.
    """
    source_dir = source_dir.expanduser().resolve()
    is_full = backup_type.lower() == "full"

    # Determine format
    fmt = (format_type or "").lower().strip()
    if fmt in {"7z", "7-zip", "sevenzip"}:
        fmt = "7z"
    elif fmt == "zip":
        fmt = "zip"
    elif destination:
        fmt = detect_archive_format(destination)
    else:
        fmt = "zip"

    ext = f".{fmt}"

    # Determine destination archive path
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    prefix = "full_backup" if is_full else "knowledge_backup"
    default_filename = f"{prefix}_{timestamp_str}{ext}"

    if destination is None:
        dest_archive = Path.cwd() / default_filename
    else:
        dest_path = destination.expanduser().resolve()
        if dest_path.is_dir():
            dest_archive = dest_path / default_filename
        else:
            if not dest_path.suffix:
                dest_archive = dest_path.with_suffix(ext)
            else:
                dest_archive = dest_path

    dest_archive.parent.mkdir(parents=True, exist_ok=True)

    # 1. Collect knowledge files
    file_list = collect_knowledge_files(
        source_dir=source_dir,
        include_chunks=include_chunks,
        include_obsidian=include_obsidian,
        include_states=include_states,
        include_lightrag=include_lightrag,
        destination_archive=dest_archive,
        custom_dirs=custom_dirs,
    )
    knowledge_count = len(file_list)

    # 2. Collect original books if full backup
    book_files: List[Tuple[Path, str]] = []
    books_dir_resolved: Optional[Path] = None
    books_dir_rel: Optional[str] = None
    if is_full and books_dir:
        books_dir_resolved = books_dir.expanduser().resolve()
        if books_dir_resolved.is_dir():
            book_files = collect_book_files(
                books_dir=books_dir_resolved,
                source_dir=source_dir,
                indexed_only=indexed_only,
                destination_archive=dest_archive,
            )
            try:
                books_dir_rel = books_dir_resolved.relative_to(Path.cwd()).as_posix()
            except ValueError:
                books_dir_rel = str(books_dir_resolved)

    file_list.extend(book_files)
    total_files = len(file_list)
    total_uncompressed = sum(fpath.stat().st_size for fpath, _ in file_list if fpath.is_file())

    # Build metadata manifest
    manifest_items = []
    for fpath, arcname in file_list:
        if fpath.is_file():
            manifest_items.append(
                {
                    "path": arcname,
                    "size": fpath.stat().st_size,
                    "sha256": _compute_sha256(fpath),
                }
            )

    metadata = {
        "created_at": now.isoformat(),
        "backup_type": "full" if is_full else "knowledge",
        "format": fmt,
        "bookeeper_version": "0.1.0",
        "source_directory": str(source_dir),
        "books_directory": str(books_dir_resolved) if books_dir_resolved else None,
        "books_dir_relative": books_dir_rel,
        "books_count": len(book_files),
        "knowledge_files_count": knowledge_count,
        "total_files": total_files,
        "uncompressed_bytes": total_uncompressed,
        "indexed_only": indexed_only,
        "include_chunks": include_chunks,
        "include_obsidian": include_obsidian,
        "include_states": include_states,
        "include_lightrag": include_lightrag,
        "manifest": manifest_items,
    }

    # Write archive
    temp_meta_file = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(metadata, tf, indent=2)
            temp_meta_file = Path(tf.name)

        if fmt == "7z":
            with py7zr.SevenZipFile(dest_archive, "w") as sz:
                sz.write(temp_meta_file, "backup_metadata.json")
                for idx, (fpath, arcname) in enumerate(file_list, 1):
                    sz.write(fpath, arcname)
                    if progress_callback:
                        progress_callback(idx, total_files, arcname)
        else:
            with zipfile.ZipFile(dest_archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(temp_meta_file, "backup_metadata.json")
                for idx, (fpath, arcname) in enumerate(file_list, 1):
                    zf.write(fpath, arcname)
                    if progress_callback:
                        progress_callback(idx, total_files, arcname)

    finally:
        if temp_meta_file and temp_meta_file.is_file():
            try:
                temp_meta_file.unlink()
            except Exception:
                pass

    compressed_size = dest_archive.stat().st_size if dest_archive.is_file() else 0

    return {
        "destination": dest_archive,
        "backup_type": "full" if is_full else "knowledge",
        "format": fmt,
        "total_files": total_files,
        "knowledge_files_count": knowledge_count,
        "books_count": len(book_files),
        "uncompressed_bytes": total_uncompressed,
        "compressed_bytes": compressed_size,
        "metadata": metadata,
    }


def inspect_backup(archive_path: Path) -> Dict[str, Any]:
    """
    Inspect a backup archive and return detailed content metadata.
    """
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Backup archive not found at: {archive_path}")

    fmt = detect_archive_format(archive_path)
    entries: List[Dict[str, Any]] = []
    metadata: Optional[Dict[str, Any]] = None
    total_uncompressed = 0
    compressed_size = archive_path.stat().st_size

    if fmt == "7z":
        with py7zr.SevenZipFile(archive_path, "r") as sz:
            for item in sz.list():
                if not item.is_directory:
                    entries.append(
                        {
                            "path": item.filename,
                            "uncompressed": item.uncompressed,
                            "compressed": item.compressed,
                        }
                    )
                    total_uncompressed += item.uncompressed or 0

            if "backup_metadata.json" in sz.getnames():
                with tempfile.TemporaryDirectory() as td:
                    sz.extract(targets=["backup_metadata.json"], path=td)
                    mf = Path(td) / "backup_metadata.json"
                    if mf.is_file():
                        try:
                            metadata = json.loads(mf.read_text(encoding="utf-8"))
                        except Exception:
                            pass
    else:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    entries.append(
                        {
                            "path": info.filename,
                            "uncompressed": info.file_size,
                            "compressed": info.compress_size,
                        }
                    )
                    total_uncompressed += info.file_size

            if "backup_metadata.json" in zf.namelist():
                try:
                    metadata = json.loads(zf.read("backup_metadata.json").decode("utf-8"))
                except Exception:
                    pass

    # Filter out backup_metadata.json from artifact file counts
    content_entries = [e for e in entries if e["path"] != "backup_metadata.json"]

    # Category summaries
    categories = {
        "graph": [e for e in content_entries if e["path"].startswith("knowledge_graph.")],
        "obsidian": [e for e in content_entries if e["path"].startswith("obsidian_vault/")],
        "book_states": [e for e in content_entries if e["path"].startswith(".book_states/")],
        "chunks": [e for e in content_entries if e["path"].startswith("chunks/")],
        "checkpoint": [e for e in content_entries if e["path"] == ".bookeeper_state.json"],
        "books": [e for e in content_entries if e["path"].startswith("books/")],
        "reports": [e for e in content_entries if "report" in e["path"]],
    }

    backup_type = (
        metadata.get("backup_type")
        if metadata and "backup_type" in metadata
        else ("full" if len(categories["books"]) > 0 else "knowledge")
    )

    return {
        "archive_path": archive_path,
        "backup_type": backup_type,
        "format": fmt,
        "compressed_bytes": compressed_size,
        "uncompressed_bytes": total_uncompressed,
        "total_files": len(content_entries),
        "entries": content_entries,
        "categories": {k: len(v) for k, v in categories.items()},
        "metadata": metadata,
    }


def restore_backup(
    archive_path: Path,
    target_dir: Path,
    books_dir: Optional[Path] = None,
    clean: bool = False,
    dry_run: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Restore knowledge artifacts and original source books from a backup archive into target_dir
    and books_dir.
    """
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Backup archive not found at: {archive_path}")

    target_dir = target_dir.expanduser().resolve()
    fmt = detect_archive_format(archive_path)

    # 1. Inspect archive first
    inspection = inspect_backup(archive_path)
    is_full = inspection.get("backup_type") == "full" or inspection.get("categories", {}).get("books", 0) > 0

    target_books_dir: Optional[Path] = None
    if is_full:
        if books_dir:
            target_books_dir = books_dir.expanduser().resolve()
        else:
            meta = inspection.get("metadata") or {}
            rel = meta.get("books_dir_relative")
            if rel and not Path(rel).is_absolute():
                target_books_dir = (Path.cwd() / rel).resolve()
            elif (target_dir.parent / "archive_downloader" / "downloads").exists():
                target_books_dir = (target_dir.parent / "archive_downloader" / "downloads").resolve()
            else:
                target_books_dir = (target_dir.parent / "downloads").resolve()

    if dry_run:
        return {
            "dry_run": True,
            "backup_type": "full" if is_full else "knowledge",
            "target_dir": target_dir,
            "target_books_dir": target_books_dir if is_full else None,
            "inspection": inspection,
            "cleaned": {},
            "files_restored": 0,
            "bytes_restored": 0,
        }

    # 2. Clean existing artifacts if requested
    cleaned_stats = {}
    if clean:
        cleaned_stats = clean_knowledge_artifacts(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    if is_full and target_books_dir:
        target_books_dir.mkdir(parents=True, exist_ok=True)

    # 3. Extract all files safely
    restored_count = 0
    restored_knowledge_count = 0
    restored_books_count = 0
    restored_bytes = 0
    total_files = inspection["total_files"]

    if fmt == "7z":
        with py7zr.SevenZipFile(archive_path, "r") as sz:
            for item in sz.list():
                if not item.is_directory and item.filename != "backup_metadata.json":
                    restored_count += 1
                    restored_bytes += item.uncompressed or 0
                    if item.filename.startswith("books/"):
                        restored_books_count += 1
                    else:
                        restored_knowledge_count += 1
                    if progress_callback:
                        progress_callback(restored_count, total_files, item.filename)

            # Sanitize internal filenames before extraction so py7zr creates safe filenames on Windows
            for f in sz.files:
                if not f.is_directory and f.filename != "backup_metadata.json":
                    f._file_info["filename"] = sanitize_archive_relative_path(f.filename)

            sz.extractall(path=target_dir)

            # Relocate extracted books directory to target_books_dir if separate
            if is_full and target_books_dir and (target_dir / "books").is_dir():
                if target_books_dir.resolve() != (target_dir / "books").resolve():
                    for root, _, files in os.walk(target_dir / "books"):
                        for f in files:
                            src_f = Path(root) / f
                            rel = src_f.relative_to(target_dir / "books")
                            dest_f = target_books_dir / rel
                            dest_f.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(src_f), str(dest_f))
                    shutil.rmtree(target_dir / "books", ignore_errors=True)

    else:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if not info.is_dir() and info.filename != "backup_metadata.json":
                    rel_parts = Path(info.filename).parts
                    if ".." in rel_parts or Path(info.filename).is_absolute():
                        logger.warning(f"Skipping dangerous archive path: {info.filename}")
                        continue

                    if info.filename.startswith("books/") and is_full and target_books_dir:
                        rel_file = decode_safe_relative_path(info.filename[6:])
                        safe_rel = sanitize_archive_relative_path(rel_file)
                        dest = (target_books_dir / safe_rel).resolve()
                        if not dest.is_relative_to(target_books_dir):
                            logger.warning(f"Skipping path outside books directory: {info.filename}")
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        restored_books_count += 1
                    else:
                        rel_file = decode_safe_relative_path(info.filename)
                        safe_rel = sanitize_archive_relative_path(rel_file)
                        dest = (target_dir / safe_rel).resolve()
                        if not dest.is_relative_to(target_dir):
                            logger.warning(f"Skipping path outside target directory: {info.filename}")
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        restored_knowledge_count += 1

                    restored_count += 1
                    restored_bytes += info.file_size
                    if progress_callback:
                        progress_callback(restored_count, total_files, info.filename)

    # Remove backup_metadata.json if extracted into target_dir
    extracted_meta = target_dir / "backup_metadata.json"
    if extracted_meta.is_file():
        try:
            extracted_meta.unlink()
        except Exception:
            pass

    return {
        "dry_run": False,
        "backup_type": "full" if is_full else "knowledge",
        "target_dir": target_dir,
        "target_books_dir": target_books_dir if is_full else None,
        "format": fmt,
        "cleaned": cleaned_stats,
        "files_restored": restored_count,
        "knowledge_files_restored": restored_knowledge_count,
        "book_files_restored": restored_books_count,
        "bytes_restored": restored_bytes,
        "inspection": inspection,
    }
