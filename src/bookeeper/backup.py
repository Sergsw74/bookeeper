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
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import py7zr

logger = logging.getLogger(__name__)

# Standard artifact directories and filenames in the output folder
KNOWLEDGE_FILES = {
    "knowledge_graph.json",
    "knowledge_graph_spine.json",
    "shards_manifest.json",
    ".bookeeper_state.json",
    "concept_embeddings.npz",
    "verification_report.json",
    "cross_check_report.json",
}

KNOWLEDGE_DIR_NAMES = {
    "obsidian_vault",
    ".book_states",
    "chunks",
    "shards",
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
    ".graphml",
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

    # Remove optional exported GraphML if present
    gml = target_dir / "knowledge_graph.graphml"
    if gml.is_file():
        try:
            gml.unlink()
            stats["files_removed"] += 1
        except Exception:
            pass

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
    t0 = time.time()
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
    logger.info(
        f"[BACKUP] Packaging {'full' if is_full else 'knowledge'} backup -> '{dest_archive.name}' (format: {fmt})"
    )

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
    elapsed = time.time() - t0
    logger.info(
        f"[BACKUP] Successfully created '{dest_archive.name}' "
        f"({compressed_size / (1024 * 1024):.2f} MB, {total_files} files: "
        f"{knowledge_count} knowledge, {len(book_files)} books) in {elapsed:.2f}s"
    )

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
        "graph": [
            e
            for e in content_entries
            if e["path"].startswith("knowledge_graph.")
            or e["path"] in ("knowledge_graph_spine.json", "shards_manifest.json")
            or e["path"].startswith("shards/")
        ],
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


def resolve_target_books_dir(
    target_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
    explicit_books_dir: Optional[Path] = None,
) -> Path:
    """
    Resolve destination directory for books during full backup restoration.
    If explicit_books_dir is passed, it is used directly.
    Otherwise, resolves relative to target_dir (or target_dir.parent if target_dir is named 'output' or 'out').
    """
    if explicit_books_dir:
        return explicit_books_dir.expanduser().resolve()

    target_dir = target_dir.expanduser().resolve()
    base_dir = target_dir.parent if target_dir.name.lower() in ("output", "out") else target_dir

    meta = metadata or {}
    rel = meta.get("books_dir_relative")
    if rel and not Path(rel).is_absolute():
        return (base_dir / rel).resolve()

    if (base_dir / "archive_downloader" / "downloads").exists():
        return (base_dir / "archive_downloader" / "downloads").resolve()
    if (base_dir / "downloads").exists():
        return (base_dir / "downloads").resolve()
    if (base_dir / "books").exists():
        return (base_dir / "books").resolve()

    return (base_dir / "archive_downloader" / "downloads").resolve()


def restore_backup(
    archive_path: Path,
    target_dir: Path,
    books_dir: Optional[Path] = None,
    clean: bool = False,
    dry_run: bool = False,
    incremental: bool = False,
    export_neo4j: bool = True,
    neo4j_config: Optional[Any] = None,
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
    t0 = time.time()

    # 1. Inspect archive first
    inspection = inspect_backup(archive_path)
    b_type = inspection.get("backup_type", "knowledge")
    is_incremental = (b_type == "incremental") or incremental
    is_full = b_type == "full" or inspection.get("categories", {}).get("books", 0) > 0

    mode_label = "dry-run" if dry_run else ("incremental" if is_incremental else ("clean" if clean else "standard"))
    logger.info(
        f"[RESTORE] Starting restore from '{archive_path.name}' ({fmt}) -> '{target_dir}' "
        f"(type: {b_type}, mode: {mode_label}, {inspection.get('total_files', 0)} files in archive)"
    )

    if is_incremental:
        if dry_run:
            return {
                "dry_run": True,
                "backup_type": "incremental",
                "target_dir": target_dir,
                "target_books_dir": books_dir,
                "inspection": inspection,
                "cleaned": {},
                "files_restored": 0,
                "bytes_restored": 0,
            }
        return apply_incremental_update(
            archive_path=archive_path,
            target_dir=target_dir,
            books_dir=books_dir,
            neo4j_config=neo4j_config,
            export_neo4j=export_neo4j,
            progress_callback=progress_callback,
        )

    target_books_dir: Optional[Path] = None
    if is_full:
        meta = inspection.get("metadata") or {}
        target_books_dir = resolve_target_books_dir(
            target_dir=target_dir,
            metadata=meta,
            explicit_books_dir=books_dir,
        )

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
        if is_full and target_books_dir:
            cleaned_book_stats = clean_book_artifacts(target_books_dir)
            cleaned_stats["files_removed"] = cleaned_stats.get("files_removed", 0) + cleaned_book_stats.get("files_removed", 0)

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

    # Create a convenience symlink target_dir / 'books' -> target_books_dir if appropriate
    if is_full and target_books_dir and target_books_dir.is_dir():
        books_alias = target_dir / "books"
        if not books_alias.exists() and target_books_dir.resolve() != books_alias.resolve():
            try:
                books_alias.symlink_to(target_books_dir, target_is_directory=True)
            except (OSError, NotImplementedError):
                pass

    elapsed = time.time() - t0
    logger.info(
        f"[RESTORE] Completed restore from '{archive_path.name}': {restored_count} files "
        f"({restored_bytes / (1024 * 1024):.2f} MB) restored in {elapsed:.2f}s "
        f"({restored_knowledge_count} knowledge, {restored_books_count} books)"
    )

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


@dataclass
class ResolvedBackupTarget:
    """
    Encapsulates a resolved Knowledge Graph target from an active directory, file, or restored archive.
    Provides lifecycle cleanup for temporary folders created during archive extraction.
    """

    root_dir: Path
    graph_file: Path
    chunks_dir: Optional[Path] = None
    states_dir: Optional[Path] = None
    label: str = ""
    is_temp: bool = False
    temp_dir_obj: Optional[Any] = None
    console: Optional[Any] = None

    def cleanup(self) -> None:
        """Purge temporary extraction folder if this target was restored from an archive."""
        if self.temp_dir_obj is not None:
            r_dir = str(self.root_dir)
            try:
                self.temp_dir_obj.cleanup()
                msg = f"[dim green]✓ Cleaned up temporal backup folder:[/dim green] [dim]{r_dir}[/dim]"
                if self.console is not None:
                    self.console.print(msg)
                else:
                    logger.info(f"Cleaned up temporal backup directory: {r_dir}")
            except Exception as e:
                logger.debug(f"Could not clean up temporary backup directory: {e}")
            self.temp_dir_obj = None
            import gc
            gc.collect()

    def __enter__(self) -> "ResolvedBackupTarget":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


def resolve_backup_target(
    target: Union[Path, str],
    label_fallback: str = "Backup",
    console: Optional[Any] = None,
) -> ResolvedBackupTarget:
    """
    Resolve a backup target from either:
    1. A backup archive (.zip, .7z, .tar.gz) -> automatically unpacks into a secure TemporaryDirectory.
    2. A directory containing knowledge_graph.json -> used directly.
    3. A direct knowledge_graph.json file -> used directly.
    """
    p = Path(target).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Target path does not exist: {target}")

    # 1. Archive file (.zip, .7z, .tar, .gz, .tgz)
    archive_suffixes = (".zip", ".7z", ".tar", ".gz", ".tgz")
    if p.is_file() and any(p.name.lower().endswith(s) for s in archive_suffixes):
        td = tempfile.TemporaryDirectory(prefix="bk_ab_")
        target_dir = Path(td.name).resolve()
        if console is not None:
            console.print(
                f"[bold cyan]📦 Unpacking backup archive '[magenta]{p.name}[/magenta]' into temporal folder:[/bold cyan] [yellow]{target_dir}[/yellow]"
            )
        else:
            logger.info(f"Unpacking backup archive '{p.name}' into temporal folder: {target_dir}")
        restore_backup(archive_path=p, target_dir=target_dir)

        kg_file = target_dir / "knowledge_graph.json"
        if not kg_file.is_file():
            # Check subdirectories if archive had nested top-level folder
            for sub in target_dir.rglob("knowledge_graph.json"):
                kg_file = sub
                break

        if not kg_file.is_file():
            td.cleanup()
            raise FileNotFoundError(f"No knowledge_graph.json found in backup archive '{p.name}'.")

        chunks_d = target_dir / "chunks"
        if not chunks_d.is_dir() and kg_file.parent != target_dir:
            chunks_d = kg_file.parent / "chunks"

        states_d = target_dir / ".book_states"
        if not states_d.is_dir() and kg_file.parent != target_dir:
            states_d = kg_file.parent / ".book_states"

        clean_label = p.stem
        if clean_label.endswith(".tar"):
            clean_label = clean_label[:-4]

        return ResolvedBackupTarget(
            root_dir=target_dir,
            graph_file=kg_file,
            chunks_dir=chunks_d if chunks_d.is_dir() else None,
            states_dir=states_d if states_d.is_dir() else None,
            label=clean_label,
            is_temp=True,
            temp_dir_obj=td,
            console=console,
        )

    # 2. Directory containing knowledge_graph.json
    if p.is_dir():
        kg_file = p / "knowledge_graph.json"
        if not kg_file.is_file():
            for sub in p.rglob("knowledge_graph.json"):
                kg_file = sub
                break

        if not kg_file.is_file():
            raise FileNotFoundError(f"No knowledge_graph.json found in directory '{p}'.")

        chunks_d = p / "chunks" if (p / "chunks").is_dir() else (kg_file.parent / "chunks")
        states_d = p / ".book_states" if (p / ".book_states").is_dir() else (kg_file.parent / ".book_states")

        return ResolvedBackupTarget(
            root_dir=p,
            graph_file=kg_file,
            chunks_dir=chunks_d if chunks_d.is_dir() else None,
            states_dir=states_d if states_d.is_dir() else None,
            label=p.name or label_fallback,
            is_temp=False,
            console=console,
        )

    # 3. Direct knowledge_graph.json file
    if p.is_file():
        chunks_d = p.parent / "chunks"
        states_d = p.parent / ".book_states"
        return ResolvedBackupTarget(
            root_dir=p.parent,
            graph_file=p,
            chunks_dir=chunks_d if chunks_d.is_dir() else None,
            states_dir=states_d if states_d.is_dir() else None,
            label=p.parent.name or label_fallback,
            is_temp=False,
            console=console,
        )

    raise ValueError(f"Unsupported backup target: {target}")


@contextmanager
def open_backup_target(
    target: Union[Path, str],
    label_fallback: str = "Backup",
    console: Optional[Any] = None,
) -> Iterator[ResolvedBackupTarget]:
    """Context manager for safely extracting, inspecting, and auto-cleaning backup targets."""
    bt = resolve_backup_target(target, label_fallback=label_fallback, console=console)
    try:
        yield bt
    finally:
        bt.cleanup()


def compare_knowledge_graphs(
    store_a: Any,
    store_b: Any,
    label_a: str = "Baseline (A)",
    label_b: str = "Candidate (B)",
) -> Dict[str, Any]:
    """
    Compute fast topological, lexical, and entity comparative metrics between two Knowledge Graphs.
    Does not require LLM inference.
    """
    g_a = store_a.graph
    g_b = store_b.graph

    books_a = {n: d.get("title", n) for n, d in g_a.nodes(data=True) if d.get("type") == "Book"}
    books_b = {n: d.get("title", n) for n, d in g_b.nodes(data=True) if d.get("type") == "Book"}

    sections_a = {n for n, d in g_a.nodes(data=True) if d.get("type") == "Section"}
    sections_b = {n for n, d in g_b.nodes(data=True) if d.get("type") == "Section"}

    chunks_a = {n: len(d.get("text", "") or "") for n, d in g_a.nodes(data=True) if d.get("type") == "Chunk"}
    chunks_b = {n: len(d.get("text", "") or "") for n, d in g_b.nodes(data=True) if d.get("type") == "Chunk"}

    ideas_a = {d.get("name") or n.replace("concept:", "") for n, d in g_a.nodes(data=True) if d.get("type") in ("Concept", "Idea")}
    ideas_b = {d.get("name") or n.replace("concept:", "") for n, d in g_b.nodes(data=True) if d.get("type") in ("Concept", "Idea")}

    common_ideas = ideas_a.intersection(ideas_b)
    only_a_ideas = ideas_a - ideas_b
    only_b_ideas = ideas_b - ideas_a
    union_ideas = ideas_a.union(ideas_b)

    jaccard = (len(common_ideas) / len(union_ideas)) if union_ideas else 1.0
    overlap_a = (len(common_ideas) / len(ideas_a)) if ideas_a else 1.0
    overlap_b = (len(common_ideas) / len(ideas_b)) if ideas_b else 1.0

    edges_a = len(g_a.edges())
    edges_b = len(g_b.edges())

    return {
        "labels": {"a": label_a, "b": label_b},
        "books": {
            "a": len(books_a),
            "b": len(books_b),
            "delta": len(books_b) - len(books_a),
            "common_titles": list(set(books_a.values()).intersection(set(books_b.values()))),
        },
        "sections": {
            "a": len(sections_a),
            "b": len(sections_b),
            "delta": len(sections_b) - len(sections_a),
        },
        "chunks": {
            "a": len(chunks_a),
            "b": len(chunks_b),
            "delta": len(chunks_b) - len(chunks_a),
            "avg_chars_a": round(sum(chunks_a.values()) / len(chunks_a), 1) if chunks_a else 0,
            "avg_chars_b": round(sum(chunks_b.values()) / len(chunks_b), 1) if chunks_b else 0,
        },
        "ideas": {
            "a": len(ideas_a),
            "b": len(ideas_b),
            "delta": len(ideas_b) - len(ideas_a),
            "common_count": len(common_ideas),
            "only_a_count": len(only_a_ideas),
            "only_b_count": len(only_b_ideas),
            "jaccard_similarity": round(jaccard, 4),
            "overlap_ratio_a": round(overlap_a, 4),
            "overlap_ratio_b": round(overlap_b, 4),
            "sample_only_b": sorted(list(only_b_ideas))[:10],
            "sample_only_a": sorted(list(only_a_ideas))[:10],
        },
        "edges": {
            "a": edges_a,
            "b": edges_b,
            "delta": edges_b - edges_a,
        },
    }


# ---------------------------------------------------------------------------
# Incremental Update & Backup Engine
# ---------------------------------------------------------------------------


@dataclass
class IncrementalDiffResult:
    """Detailed result of calculating incremental diff against a baseline."""

    new_books: List[Dict[str, Any]]
    related_books: List[Dict[str, Any]]
    diff_store: Any  # ConceptGraphStore
    diff_node_count: int
    diff_edge_count: int
    baseline_type: str
    baseline_reference: str
    baseline_book_count: int
    current_book_count: int


def extract_book_signatures(b: Dict[str, Any]) -> Tuple[Optional[str], Set[str]]:
    """
    Extract arXiv ID and normalized title/filename signatures from a book dict.
    Returns (arxiv_id, signatures_set).
    """
    aid = b.get("arxiv_id")
    b_path_str = b.get("path") or b.get("file_path")
    p = Path(b_path_str) if b_path_str else None
    b_title = str(b.get("title") or b.get("name") or "").strip()

    if not aid and p:
        m = re.search(r"(\d{4}\.\d{4,5})", p.name)
        if m:
            aid = m.group(1)
    if not aid and b_title:
        m = re.search(r"(\d{4}\.\d{4,5})", b_title)
        if m:
            aid = m.group(1)

    signatures = set()
    if b_title:
        signatures.add(b_title.lower())
        clean_t = re.sub(r"[^a-z0-9]", "", b_title.lower())
        if clean_t:
            signatures.add(clean_t)
        title_no_aid = re.sub(r"^\d{4}\.\d{4,5}\s*[-_:]\s*", "", b_title, flags=re.IGNORECASE).strip()
        if title_no_aid and title_no_aid != b_title:
            signatures.add(title_no_aid.lower())
            clean_no_aid = re.sub(r"[^a-z0-9]", "", title_no_aid.lower())
            if clean_no_aid:
                signatures.add(clean_no_aid)

    if p:
        signatures.add(p.name.lower())
        signatures.add(p.stem.lower())
        clean_s = re.sub(r"[^a-z0-9]", "", p.stem.lower())
        if clean_s:
            signatures.add(clean_s)

    aid_clean = str(aid).strip().lower() if aid else None
    return aid_clean, signatures


def get_baseline_books(
    baseline: Union[str, Path, Any] = "neo4j",
    neo4j_config: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Extract baseline books from either:
    1. 'neo4j' or Neo4jConfig / Neo4jExporter -> queries connected Neo4j database.
    2. Backup archive (.zip, .7z, .tar.gz) -> unpacks / reads knowledge_graph.json.
    3. JSON file or directory -> reads knowledge_graph.json.
    Returns (baseline_books, baseline_type, baseline_reference).
    """
    from bookeeper.graph.store import ConceptGraphStore

    # Case 1: Neo4j baseline
    is_neo4j = (
        (isinstance(baseline, str) and baseline.lower() == "neo4j")
        or hasattr(baseline, "fetch_remote_books")
        or type(baseline).__name__ in ("Neo4jConfig", "Neo4jExporter")
    )
    if is_neo4j:
        from bookeeper.graph.neo4j_exporter import Neo4jExporter

        owns_exporter = False
        if hasattr(baseline, "fetch_remote_books"):
            exporter = baseline
        else:
            if neo4j_config is None and type(baseline).__name__ == "Neo4jConfig":
                neo4j_config = baseline
            if neo4j_config is None:
                from bookeeper.config import get_settings

                cfg = get_settings()
                neo4j_config = cfg.neo4j
            exporter = Neo4jExporter.from_config(neo4j_config)
            owns_exporter = True

        try:
            if not exporter.verify_connectivity():
                raise ConnectionError(
                    f"Neo4j database is not reachable at {getattr(exporter, 'uri', 'localhost:7687')}. "
                    "Please ensure Neo4j is running or provide a backup archive as baseline."
                )
            books = exporter.fetch_remote_books()
            return books, "neo4j", getattr(exporter, "uri", "neo4j")
        finally:
            if owns_exporter:
                exporter.close()

    # Case 2: Backup archive or directory or file
    p = Path(baseline).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Baseline backup target not found at: {p}")

    if p.is_file() and p.suffix.lower() in (".zip", ".7z", ".tar", ".gz", ".tgz"):
        with open_backup_target(p) as r_target:
            b_store = ConceptGraphStore()
            b_store.load(r_target.graph_file)
            books = []
            for nid, attrs in b_store.graph.nodes(data=True):
                if attrs.get("type") == "Book":
                    bid = attrs.get("book_id")
                    if bid is None and str(nid).startswith("book:"):
                        try:
                            bid = int(str(nid).replace("book:", ""))
                        except ValueError:
                            pass
                    books.append({
                        "id": nid,
                        "book_id": bid,
                        "title": attrs.get("title", f"Book {bid}"),
                        "author": attrs.get("author", "Unknown"),
                        "arxiv_id": attrs.get("arxiv_id"),
                        "content_hash": attrs.get("content_hash"),
                    })
        return books, "backup_archive", p.name

    if p.is_dir():
        kg_file = p / "knowledge_graph.json"
        if not kg_file.is_file():
            kg_file = p / "knowledge_graph_spine.json"
        if not kg_file.is_file():
            raise FileNotFoundError(f"No knowledge graph found in baseline directory: {p}")
        b_store = ConceptGraphStore()
        b_store.load(kg_file)
        books = []
        for nid, attrs in b_store.graph.nodes(data=True):
            if attrs.get("type") == "Book":
                bid = attrs.get("book_id")
                if bid is None and str(nid).startswith("book:"):
                    try:
                        bid = int(str(nid).replace("book:", ""))
                    except ValueError:
                        pass
                books.append({
                    "id": nid,
                    "book_id": bid,
                    "title": attrs.get("title", f"Book {bid}"),
                    "author": attrs.get("author", "Unknown"),
                    "arxiv_id": attrs.get("arxiv_id"),
                    "content_hash": attrs.get("content_hash"),
                })
        return books, "directory", str(p)

    if p.is_file():
        b_store = ConceptGraphStore()
        b_store.load(p)
        books = []
        for nid, attrs in b_store.graph.nodes(data=True):
            if attrs.get("type") == "Book":
                bid = attrs.get("book_id")
                if bid is None and str(nid).startswith("book:"):
                    try:
                        bid = int(str(nid).replace("book:", ""))
                    except ValueError:
                        pass
                books.append({
                    "id": nid,
                    "book_id": bid,
                    "title": attrs.get("title", f"Book {bid}"),
                    "author": attrs.get("author", "Unknown"),
                    "arxiv_id": attrs.get("arxiv_id"),
                    "content_hash": attrs.get("content_hash"),
                })
        return books, "file", p.name

    raise ValueError(f"Unsupported baseline target: {baseline}")


def calculate_incremental_diff(
    source_dir: Path,
    baseline: Union[str, Path, Any] = "neo4j",
    neo4j_config: Optional[Any] = None,
    current_store: Optional[Any] = None,
) -> IncrementalDiffResult:
    """
    Calculate the incremental difference between the current Knowledge Graph and a baseline
    (either an active Neo4j database or a previous backup archive/file).
    Identifies:
      1. New books in the current KG not in the baseline (or with changed content hash).
      2. Books in the current KG related to the new books via shared concepts and concept relations.
      3. An extracted diff subgraph (ConceptGraphStore) containing exactly the new & related books,
         their connecting sections/chunks, and the bridging concepts.
    """
    from bookeeper.graph.store import ConceptGraphStore

    source_dir = Path(source_dir).expanduser().resolve()
    if current_store is None:
        current_store = ConceptGraphStore()
        kg_path = source_dir / "knowledge_graph.json"
        if kg_path.is_file() or (source_dir / "knowledge_graph_spine.json").is_file():
            current_store.load(kg_path if kg_path.is_file() else source_dir)
        else:
            raise FileNotFoundError(f"No knowledge graph found in source directory: {source_dir}")

    # 1. Extract books from current store
    current_books: List[Dict[str, Any]] = []
    all_book_nodes: Dict[str, Dict[str, Any]] = {}
    for nid, attrs in current_store.graph.nodes(data=True):
        if attrs.get("type") == "Book":
            bid = attrs.get("book_id")
            if bid is None and str(nid).startswith("book:"):
                try:
                    bid = int(str(nid).replace("book:", ""))
                except ValueError:
                    pass
            b_info = {
                "node_id": nid,
                "book_id": bid,
                "title": attrs.get("title", f"Book {bid}"),
                "author": attrs.get("author", "Unknown"),
                "arxiv_id": attrs.get("arxiv_id"),
                "content_hash": attrs.get("content_hash"),
            }
            current_books.append(b_info)
            all_book_nodes[str(nid)] = b_info

    # 2. Extract books from baseline
    baseline_books, baseline_type, baseline_ref = get_baseline_books(
        baseline=baseline,
        neo4j_config=neo4j_config,
    )

    # 3. Index baseline books by arXiv ID, title signatures, and ID
    baseline_arxiv_map: Dict[str, Dict[str, Any]] = {}
    baseline_sig_map: Dict[str, Dict[str, Any]] = {}
    baseline_id_map: Dict[str, Dict[str, Any]] = {}

    for bb in baseline_books:
        b_aid, b_sigs = extract_book_signatures(bb)
        if b_aid:
            baseline_arxiv_map[b_aid] = bb
        for s in b_sigs:
            if len(s) >= 4:
                baseline_sig_map[s] = bb
        bid = bb.get("book_id") or bb.get("id")
        if bid is not None:
            bid_clean = str(bid).replace("book:", "").strip()
            baseline_id_map[bid_clean] = bb

    # 4. Identify New Books (present in current KG, missing or modified in baseline)
    new_books: List[Dict[str, Any]] = []
    for cb in current_books:
        cb_aid, cb_sigs = extract_book_signatures(cb)
        matched_bb = None
        if cb_aid and cb_aid in baseline_arxiv_map:
            matched_bb = baseline_arxiv_map[cb_aid]
        if not matched_bb:
            for s in cb_sigs:
                if len(s) >= 4 and s in baseline_sig_map:
                    matched_bb = baseline_sig_map[s]
                    break
        if not matched_bb and cb.get("book_id") is not None:
            cid_str = str(cb["book_id"]).replace("book:", "").strip()
            if cid_str in baseline_id_map:
                cand_bb = baseline_id_map[cid_str]
                _, cand_sigs = extract_book_signatures(cand_bb)
                if cb_sigs & cand_sigs:
                    matched_bb = cand_bb

        if matched_bb is None:
            new_books.append(cb)
        else:
            cb_hash = cb.get("content_hash")
            bb_hash = matched_bb.get("content_hash")
            if cb_hash and bb_hash and cb_hash != bb_hash:
                new_books.append(cb)

    new_book_nodes = {cb["node_id"] for cb in new_books}

    # 5. Identify Books Related to New Books from KG via shared concepts and concept relations
    new_connected_concepts = set()
    for b_node in new_book_nodes:
        if not current_store.graph.has_node(b_node):
            continue
        # Direct concept edges from book node
        for neighbor in current_store.graph.neighbors(b_node):
            if current_store.graph.nodes[neighbor].get("type") in ("Concept", "Idea"):
                new_connected_concepts.add(neighbor)
        for pred in current_store.graph.predecessors(b_node):
            if current_store.graph.nodes[pred].get("type") in ("Concept", "Idea"):
                new_connected_concepts.add(pred)

        # Concept edges from sections and chunks of this book
        bid = current_store.graph.nodes[b_node].get("book_id")
        for n, attrs in current_store.graph.nodes(data=True):
            if attrs.get("book_id") == bid and attrs.get("type") in ("Section", "Chunk"):
                for neighbor in current_store.graph.neighbors(n):
                    if current_store.graph.nodes[neighbor].get("type") in ("Concept", "Idea"):
                        new_connected_concepts.add(neighbor)
                for pred in current_store.graph.predecessors(n):
                    if current_store.graph.nodes[pred].get("type") in ("Concept", "Idea"):
                        new_connected_concepts.add(pred)

    # 1-hop expansion across RELATED_TO inter-concept edges
    expanded_concepts = set(new_connected_concepts)
    for c_node in new_connected_concepts:
        for neighbor in current_store.graph.neighbors(c_node):
            if current_store.graph.nodes[neighbor].get("type") in ("Concept", "Idea"):
                expanded_concepts.add(neighbor)
        for pred in current_store.graph.predecessors(c_node):
            if current_store.graph.nodes[pred].get("type") in ("Concept", "Idea"):
                expanded_concepts.add(pred)

    # Find existing/other books linked to any of these expanded concepts
    related_books_map: Dict[str, Dict[str, Any]] = {}
    for c_node in expanded_concepts:
        c_name = current_store.graph.nodes[c_node].get("name", str(c_node).replace("concept:", ""))
        connected_nodes = set(current_store.graph.neighbors(c_node)) | set(current_store.graph.predecessors(c_node))
        for node in connected_nodes:
            node_attrs = current_store.graph.nodes[node]
            ntype = node_attrs.get("type")
            b_target = None
            if ntype == "Book" and node not in new_book_nodes:
                b_target = node
            elif ntype in ("Section", "Chunk") and "book_id" in node_attrs:
                cand = f"book:{node_attrs['book_id']}"
                if cand in all_book_nodes and cand not in new_book_nodes:
                    b_target = cand

            if b_target:
                if b_target not in related_books_map:
                    b_data = all_book_nodes[b_target]
                    related_books_map[b_target] = {
                        "node_id": b_target,
                        "book_id": b_data.get("book_id"),
                        "title": b_data.get("title", f"Book {b_data.get('book_id')}"),
                        "author": b_data.get("author", "Unknown"),
                        "arxiv_id": b_data.get("arxiv_id"),
                        "shared_concepts": set(),
                    }
                related_books_map[b_target]["shared_concepts"].add(c_name)

    related_books_list = []
    for r in related_books_map.values():
        r_copy = dict(r)
        r_copy["shared_concepts"] = sorted(list(r["shared_concepts"]))
        related_books_list.append(r_copy)

    # 6. Extract Diff Subgraph (ConceptGraphStore)
    diff_nodes = set()
    # 1. New books and all their sections, chunks, formulas
    for b_node in new_book_nodes:
        diff_nodes.add(b_node)
        bid = current_store.graph.nodes[b_node].get("book_id")
        for n, attrs in current_store.graph.nodes(data=True):
            if attrs.get("book_id") == bid:
                diff_nodes.add(n)

    # 2. Related books and their connecting chunks/sections
    for r_node in related_books_map.keys():
        diff_nodes.add(r_node)
        r_bid = all_book_nodes[r_node].get("book_id")
        for n, attrs in current_store.graph.nodes(data=True):
            if attrs.get("book_id") == r_bid:
                n_neighbors = set(current_store.graph.neighbors(n)) | set(current_store.graph.predecessors(n))
                if any(c in expanded_concepts for c in n_neighbors):
                    diff_nodes.add(n)

    # 3. All expanded concepts
    for c_node in expanded_concepts:
        diff_nodes.add(c_node)

    diff_subgraph = current_store.graph.subgraph(diff_nodes).copy()
    diff_store = ConceptGraphStore(diff_subgraph)

    logger.info(
        f"[INCREMENTAL-DIFF] Completed diff vs '{baseline_ref}' ({baseline_type}): "
        f"{len(new_books)} new books, {len(related_books_list)} related books, "
        f"subgraph: {diff_store.graph.number_of_nodes()} nodes, {diff_store.graph.number_of_edges()} edges"
    )

    return IncrementalDiffResult(
        new_books=new_books,
        related_books=related_books_list,
        diff_store=diff_store,
        diff_node_count=diff_store.graph.number_of_nodes(),
        diff_edge_count=diff_store.graph.number_of_edges(),
        baseline_type=baseline_type,
        baseline_reference=baseline_ref,
        baseline_book_count=len(baseline_books),
        current_book_count=len(current_books),
    )


def collect_incremental_source_books(
    books_dir: Path,
    books: List[Dict[str, Any]],
    target_books_dir: Path,
    source_dir: Optional[Path] = None,
    destination_archive: Optional[Path] = None,
) -> int:
    """
    Collect original source book files ONLY for the specified books (from incremental diff)
    and copy them into target_books_dir preserving relative paths.
    Returns the count of distinct book document files copied.
    """
    b_dir_p = Path(books_dir).expanduser().resolve()
    if not b_dir_p.is_dir() or not books:
        return 0

    target_books_dir.mkdir(parents=True, exist_ok=True)

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower()) if s else ""

    book_targets = []
    for b in books:
        bid = b.get("book_id")
        bid_str = str(bid).replace("book:", "").strip() if bid is not None else None
        aid = str(b.get("arxiv_id") or "").strip()
        t = str(b.get("title") or b.get("name") or "").strip()
        norm_t = _norm(t)
        exp_p = b.get("path") or b.get("file_path")
        # Check .book_states for extra metadata if available
        if source_dir and bid_str:
            st_file = Path(source_dir).expanduser().resolve() / ".book_states" / f"book_{bid_str}_state.json"
            if st_file.is_file():
                try:
                    with open(st_file, "r", encoding="utf-8") as sf:
                        s_data = json.load(sf)
                    if not exp_p:
                        exp_p = s_data.get("file_path") or s_data.get("book_path")
                    if not aid and s_data.get("arxiv_id"):
                        aid = str(s_data.get("arxiv_id")).strip()
                    if not t and s_data.get("book_title"):
                        t = str(s_data.get("book_title")).strip()
                        norm_t = _norm(t)
                except Exception:
                    pass

        book_targets.append({
            "book_id": bid_str,
            "arxiv_id": aid if aid else None,
            "title": t,
            "norm_title": norm_t,
            "explicit_path": exp_p,
            "matched_files": set(),
        })

    # 1. Match from index.json in books_dir if present
    index_file = b_dir_p / "index.json"
    index_entries = {}
    matched_index_entries = {}
    if index_file.is_file():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_entries = json.load(f)
        except Exception:
            pass

    if index_entries:
        for k, val in index_entries.items():
            if not isinstance(val, dict):
                continue
            v_title = _norm(str(val.get("title", "")))
            v_aid = str(val.get("arxiv_id", "")).strip()
            v_bid = str(val.get("book_id", "")).strip()
            k_clean = str(k).strip()

            for bt in book_targets:
                is_match = False
                if bt["arxiv_id"] and (bt["arxiv_id"] == v_aid or bt["arxiv_id"] == k_clean):
                    is_match = True
                elif bt["book_id"] and (bt["book_id"] == v_bid or bt["book_id"] == k_clean):
                    is_match = True
                elif bt["norm_title"] and v_title and (bt["norm_title"] == v_title or bt["norm_title"] in v_title or v_title in bt["norm_title"]):
                    is_match = True

                if is_match:
                    fpath_str = val.get("file_path")
                    if fpath_str:
                        candidate = b_dir_p / fpath_str
                        if candidate.is_file() and not is_excluded_file(candidate, destination_archive):
                            bt["matched_files"].add(candidate)
                            matched_index_entries[k] = val

    # 2. Recursive scan in books_dir for book targets
    for root, dirs, files in os.walk(b_dir_p):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]
        root_p = Path(root)
        for fname in files:
            fpath = root_p / fname
            if fpath.suffix.lower() not in BOOK_EXTENSIONS:
                continue
            if is_excluded_file(fpath, destination_archive):
                continue

            stem_norm = _norm(fpath.stem)
            parent_name = fpath.parent.name

            for bt in book_targets:
                is_match = False
                # Explicit path or filename match
                if bt["explicit_path"]:
                    exp_name = Path(bt["explicit_path"]).name
                    if fname == exp_name or str(fpath).endswith(str(bt["explicit_path"])):
                        is_match = True
                # arXiv ID match in filename
                if not is_match and bt["arxiv_id"] and bt["arxiv_id"] in fname:
                    is_match = True
                # Calibre ID token in folder or filename: "(42)", "42_", "42 - "
                if not is_match and bt["book_id"]:
                    bid_token = f"({bt['book_id']})"
                    if bid_token in parent_name or bid_token in fpath.stem or fpath.stem.startswith(f"{bt['book_id']}_") or fpath.stem.startswith(f"{bt['book_id']} -"):
                        if not bt["norm_title"] or bt["norm_title"] in stem_norm or stem_norm in bt["norm_title"] or len(stem_norm) >= 3:
                            is_match = True
                # Normalized title match
                if not is_match and bt["norm_title"]:
                    if bt["norm_title"] == stem_norm or (len(bt["norm_title"]) >= 6 and (bt["norm_title"] in stem_norm or stem_norm in bt["norm_title"])):
                        is_match = True

                if is_match:
                    bt["matched_files"].add(fpath)

    # 3. Copy matched files and companion .opf into target_books_dir
    copied_files = set()
    for bt in book_targets:
        for fpath in bt["matched_files"]:
            if fpath in copied_files:
                continue
            copied_files.add(fpath)

            try:
                rel = fpath.relative_to(b_dir_p)
                dest = target_books_dir / rel
            except ValueError:
                dest = target_books_dir / fpath.name

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fpath, dest)

            # Check for companion .opf
            opf1 = fpath.with_suffix(".opf")
            if opf1.is_file() and opf1 not in copied_files:
                try:
                    rel_opf = opf1.relative_to(b_dir_p)
                    dest_opf = target_books_dir / rel_opf
                except ValueError:
                    dest_opf = target_books_dir / opf1.name
                dest_opf.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(opf1, dest_opf)
                copied_files.add(opf1)

            opf_meta = fpath.parent / "metadata.opf"
            if opf_meta.is_file() and opf_meta not in copied_files:
                try:
                    rel_meta = opf_meta.relative_to(b_dir_p)
                    dest_meta = target_books_dir / rel_meta
                except ValueError:
                    dest_meta = target_books_dir / opf_meta.name
                dest_meta.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(opf_meta, dest_meta)
                copied_files.add(opf_meta)

    # 4. If matched index entries exist, write pruned index.json for incremental books
    if matched_index_entries:
        (target_books_dir / "index.json").write_text(
            json.dumps(matched_index_entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Return distinct book document count
    return sum(1 for p in copied_files if p.suffix.lower() in BOOK_EXTENSIONS)


def create_incremental_backup(
    source_dir: Path,
    baseline: Union[str, Path, Any] = "neo4j",
    neo4j_config: Optional[Any] = None,
    destination: Optional[Path] = None,
    format_type: str = "zip",
    books_dir: Optional[Path] = None,
    include_books: bool = True,
    include_obsidian: bool = True,
    include_states: bool = True,
    include_chunks: bool = True,
    current_store: Optional[Any] = None,
    custom_dirs: Optional[Dict[str, Path]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Create an incremental backup archive containing only the diff (new books, related books,
    and their interconnecting concepts/sections/chunks) compared against a baseline (Neo4j or backup archive).
    """
    source_dir = Path(source_dir).expanduser().resolve()
    diff_res = calculate_incremental_diff(
        source_dir=source_dir,
        baseline=baseline,
        neo4j_config=neo4j_config,
        current_store=current_store,
    )

    fmt = (format_type or "zip").lower().strip()
    if fmt in {"7z", "7-zip", "sevenzip"}:
        fmt = "7z"
    else:
        fmt = "zip"
    ext = f".{fmt}"

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    default_filename = f"incremental_backup_{timestamp_str}{ext}"

    if destination is None:
        dest_archive = Path.cwd() / default_filename
    else:
        dest_path = Path(destination).expanduser().resolve()
        if dest_path.is_dir():
            dest_archive = dest_path / default_filename
        else:
            dest_archive = dest_path if dest_path.suffix else dest_path.with_suffix(ext)

    dest_archive.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logger.info(
        f"[INCREMENTAL-BACKUP] Packaging incremental archive -> '{dest_archive.name}' "
        f"({len(diff_res.new_books)} new books, {len(diff_res.related_books)} related books, "
        f"subgraph: {diff_res.diff_node_count} nodes, format: {fmt})"
    )

    with tempfile.TemporaryDirectory() as td:
        stage_dir = Path(td)

        # 1. Save diff knowledge graph
        diff_res.diff_store.save(stage_dir / "knowledge_graph.json")

        # 2. Collect chunks for new books
        if include_chunks:
            c_dir = custom_dirs.get("chunks_dir") if custom_dirs else None
            chunks_src = Path(c_dir).expanduser().resolve() if c_dir else (source_dir / "chunks")
            if chunks_src.is_dir():
                target_chunks = stage_dir / "chunks"
                target_chunks.mkdir(parents=True, exist_ok=True)
                for b in diff_res.new_books:
                    bid = b.get("book_id")
                    if bid is not None:
                        cf = chunks_src / f"book_{bid}_chunks.json"
                        if cf.is_file():
                            shutil.copy2(cf, target_chunks / cf.name)

        # 3. Collect state files for new books
        if include_states:
            s_dir = custom_dirs.get("states_dir") if custom_dirs else None
            states_src = Path(s_dir).expanduser().resolve() if s_dir else (source_dir / ".book_states")
            if states_src.is_dir():
                target_states = stage_dir / ".book_states"
                target_states.mkdir(parents=True, exist_ok=True)
                for b in diff_res.new_books:
                    bid = b.get("book_id")
                    if bid is not None:
                        sf = states_src / f"book_{bid}_state.json"
                        if sf.is_file():
                            shutil.copy2(sf, target_states / sf.name)

        # 4. Collect Obsidian notes for new & related books and concepts
        if include_obsidian:
            o_dir = custom_dirs.get("obsidian_dir") if custom_dirs else None
            obs_src = Path(o_dir).expanduser().resolve() if o_dir else (source_dir / "obsidian_vault")
            if obs_src.is_dir():
                target_obs = stage_dir / "obsidian_vault"
                target_obs.mkdir(parents=True, exist_ok=True)
                # Match titles of new and related books
                match_titles = set()
                for b in diff_res.new_books + diff_res.related_books:
                    t = b.get("title")
                    if t:
                        match_titles.add(re.sub(r"[^a-z0-9]", "", str(t).lower()))

                for root, _, files in os.walk(obs_src):
                    for f in files:
                        if f.endswith(".md"):
                            f_stem_clean = re.sub(r"[^a-z0-9]", "", Path(f).stem.lower())
                            if f_stem_clean in match_titles or f == "00_Index.md":
                                src_f = Path(root) / f
                                rel_f = src_f.relative_to(obs_src)
                                dest_f = target_obs / rel_f
                                dest_f.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(src_f, dest_f)

        # 5. Collect original source book files for new books (ONLY from incremental diff)
        books_copied_count = 0
        if include_books and books_dir:
            target_books = stage_dir / "books"
            books_copied_count = collect_incremental_source_books(
                books_dir=Path(books_dir),
                books=diff_res.new_books,
                target_books_dir=target_books,
                source_dir=source_dir,
                destination_archive=dest_archive,
            )

        # 6. Write backup_metadata.json
        meta = {
            "backup_type": "incremental",
            "created_at": now.isoformat(),
            "format": fmt,
            "bookeeper_version": "0.1.0",
            "source_directory": str(source_dir),
            "baseline_type": diff_res.baseline_type,
            "baseline_reference": diff_res.baseline_reference,
            "include_books": include_books,
            "books_count": books_copied_count,
            "new_books": diff_res.new_books,
            "related_books": diff_res.related_books,
            "stats": {
                "new_books_count": len(diff_res.new_books),
                "related_books_count": len(diff_res.related_books),
                "books_count": books_copied_count,
                "diff_nodes": diff_res.diff_node_count,
                "diff_edges": diff_res.diff_edge_count,
                "baseline_book_count": diff_res.baseline_book_count,
                "current_book_count": diff_res.current_book_count,
            },
        }
        (stage_dir / "backup_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # 7. Compress into target archive
        file_list: List[Tuple[Path, str]] = []
        for root, _, files in os.walk(stage_dir):
            for f in files:
                fp = Path(root) / f
                rel = fp.relative_to(stage_dir).as_posix()
                file_list.append((fp, rel))

        total_files = len(file_list)
        total_uncompressed = sum(fp.stat().st_size for fp, _ in file_list)

        if fmt == "7z":
            with py7zr.SevenZipFile(dest_archive, "w") as sz:
                for idx, (fpath, arcname) in enumerate(file_list, start=1):
                    sz.write(fpath, arcname)
                    if progress_callback:
                        progress_callback(idx, total_files, f"Packing {arcname}")
        else:
            with zipfile.ZipFile(dest_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for idx, (fpath, arcname) in enumerate(file_list, start=1):
                    zf.write(fpath, arcname)
                    if progress_callback:
                        progress_callback(idx, total_files, f"Packing {arcname}")

    compressed_size = dest_archive.stat().st_size
    elapsed = time.time() - t0
    logger.info(
        f"[INCREMENTAL-BACKUP] Created incremental archive '{dest_archive.name}' "
        f"({compressed_size / (1024 * 1024):.2f} MB compressed, {len(diff_res.new_books)} new books, "
        f"{len(diff_res.related_books)} related books, {total_files} files) in {elapsed:.2f}s"
    )
    return {
        "success": True,
        "destination": str(dest_archive),
        "filename": dest_archive.name,
        "format": fmt,
        "backup_type": "incremental",
        "baseline_type": diff_res.baseline_type,
        "baseline_reference": diff_res.baseline_reference,
        "new_books_count": len(diff_res.new_books),
        "related_books_count": len(diff_res.related_books),
        "include_books": include_books,
        "books_count": books_copied_count,
        "diff_nodes": diff_res.diff_node_count,
        "diff_edges": diff_res.diff_edge_count,
        "new_books": diff_res.new_books,
        "related_books": diff_res.related_books,
        "total_files": total_files,
        "uncompressed_bytes": total_uncompressed,
        "compressed_bytes": compressed_size,
        "created_at": now.isoformat(),
    }


def apply_incremental_update(
    archive_path: Path,
    target_dir: Path,
    books_dir: Optional[Path] = None,
    neo4j_config: Optional[Any] = None,
    export_neo4j: bool = True,
    target_store: Optional[Any] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Apply an incremental update archive to the existing Knowledge Graph and Neo4j database (if active).
    Merges new and related book subgraphs without wiping existing knowledge.
    """
    from bookeeper.graph.store import ConceptGraphStore

    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Incremental archive not found at: {archive_path}")

    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logger.info(f"[RESTORE-INC] Applying incremental update from '{archive_path.name}' into '{target_dir}'")

    with tempfile.TemporaryDirectory() as td:
        extract_dir = Path(td)
        fmt = detect_archive_format(archive_path)

        if fmt == "7z":
            with py7zr.SevenZipFile(archive_path, "r") as sz:
                sz.extractall(extract_dir)
        else:
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)

        # 1. Read metadata
        meta_file = extract_dir / "backup_metadata.json"
        meta: Dict[str, Any] = {}
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        new_books = meta.get("new_books", [])
        related_books = meta.get("related_books", [])

        # 2. Merge Knowledge Graph
        diff_kg_file = extract_dir / "knowledge_graph.json"
        merge_stats = {"nodes_added": 0, "nodes_updated": 0, "edges_added": 0, "edges_updated": 0}

        if diff_kg_file.is_file():
            diff_store = ConceptGraphStore()
            diff_store.load(diff_kg_file)

            if target_store is None:
                target_store = ConceptGraphStore()
                tgt_kg = target_dir / "knowledge_graph.json"
                if tgt_kg.is_file() or (target_dir / "knowledge_graph_spine.json").is_file():
                    target_store.load(tgt_kg if tgt_kg.is_file() else target_dir)

            merge_stats = target_store.merge_store(diff_store)
            target_store.save(target_dir / "knowledge_graph.json")

        # 3. Merge auxiliary files
        files_restored = 0
        bytes_restored = 0

        # Chunks
        inc_chunks = extract_dir / "chunks"
        if inc_chunks.is_dir():
            tgt_chunks = target_dir / "chunks"
            tgt_chunks.mkdir(parents=True, exist_ok=True)
            for cf in inc_chunks.iterdir():
                if cf.is_file():
                    shutil.copy2(cf, tgt_chunks / cf.name)
                    files_restored += 1
                    bytes_restored += cf.stat().st_size

        # States
        inc_states = extract_dir / ".book_states"
        if inc_states.is_dir():
            tgt_states = target_dir / ".book_states"
            tgt_states.mkdir(parents=True, exist_ok=True)
            for sf in inc_states.iterdir():
                if sf.is_file():
                    shutil.copy2(sf, tgt_states / sf.name)
                    files_restored += 1
                    bytes_restored += sf.stat().st_size

        # Obsidian notes
        inc_obs = extract_dir / "obsidian_vault"
        if inc_obs.is_dir():
            tgt_obs = target_dir / "obsidian_vault"
            for root, _, files in os.walk(inc_obs):
                for f in files:
                    src_f = Path(root) / f
                    rel_f = src_f.relative_to(inc_obs)
                    dest_f = tgt_obs / rel_f
                    dest_f.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_f, dest_f)
                    files_restored += 1
                    bytes_restored += src_f.stat().st_size

        # Source books
        inc_books = extract_dir / "books"
        if inc_books.is_dir() and books_dir:
            b_dir_p = Path(books_dir).expanduser().resolve()
            b_dir_p.mkdir(parents=True, exist_ok=True)
            for root, _, files in os.walk(inc_books):
                for f in files:
                    src_f = Path(root) / f
                    rel_f = src_f.relative_to(inc_books)
                    dest_f = b_dir_p / rel_f
                    dest_f.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_f, dest_f)
                    files_restored += 1
                    bytes_restored += src_f.stat().st_size

        # 4. Synchronize into Neo4j if active
        neo4j_result = None
        if export_neo4j and diff_kg_file.is_file():
            try:
                from bookeeper.graph.neo4j_exporter import Neo4jExporter

                if neo4j_config is None:
                    from bookeeper.config import get_settings

                    cfg = get_settings()
                    neo4j_config = cfg.neo4j
                exporter = Neo4jExporter.from_config(neo4j_config)
                if exporter.verify_connectivity():
                    neo4j_result = exporter.export(diff_store, clean=False, show_progress=False)
                    logger.info(
                        f"[RESTORE-INC] Neo4j incremental sync complete: "
                        f"{neo4j_result.get('nodes_upserted', 0):,} nodes, "
                        f"{neo4j_result.get('edges_upserted', 0):,} relationships"
                    )
            except Exception as e:
                logger.warning(f"[RESTORE-INC] Neo4j sync warning: {e}")
                neo4j_result = {"error": str(e)}

    elapsed = time.time() - t0
    logger.info(
        f"[RESTORE-INC] Incremental restore '{archive_path.name}' finished in {elapsed:.2f}s: "
        f"+{merge_stats.get('nodes_added', 0)} nodes, +{merge_stats.get('edges_added', 0)} edges, "
        f"{files_restored} files extracted ({bytes_restored / (1024 * 1024):.2f} MB)"
    )

    return {
        "success": True,
        "backup_type": "incremental",
        "archive": archive_path.name,
        "target_dir": str(target_dir),
        "files_restored": files_restored,
        "bytes_restored": bytes_restored,
        "merge_stats": merge_stats,
        "new_books": new_books,
        "related_books": related_books,
        "neo4j_export": neo4j_result,
        "completed_at": datetime.datetime.now().isoformat(),
    }


