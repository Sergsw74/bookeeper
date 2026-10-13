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
    "verification_report.json",
    "cross_check_report.json",
}

KNOWLEDGE_DIR_NAMES = {
    "obsidian_vault",
    ".book_states",
    "chunks",
    "lightrag",
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
    include_chunks: bool = True,
    include_obsidian: bool = True,
    include_states: bool = True,
    include_lightrag: bool = True,
    custom_dirs: Optional[Dict[str, Path]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Create a compressed knowledge backup archive (.zip or .7z).
    """
    source_dir = source_dir.expanduser().resolve()

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
    default_filename = f"knowledge_backup_{timestamp_str}{ext}"

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

    # Collect files
    file_list = collect_knowledge_files(
        source_dir=source_dir,
        include_chunks=include_chunks,
        include_obsidian=include_obsidian,
        include_states=include_states,
        include_lightrag=include_lightrag,
        destination_archive=dest_archive,
        custom_dirs=custom_dirs,
    )

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
        "format": fmt,
        "bookeeper_version": "0.1.0",
        "source_directory": str(source_dir),
        "total_files": total_files,
        "uncompressed_bytes": total_uncompressed,
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
        "format": fmt,
        "total_files": total_files,
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
        "reports": [e for e in content_entries if "report" in e["path"]],
    }

    return {
        "archive_path": archive_path,
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
    clean: bool = False,
    dry_run: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Restore knowledge artifacts from a backup archive into target_dir.
    """
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Backup archive not found at: {archive_path}")

    target_dir = target_dir.expanduser().resolve()
    fmt = detect_archive_format(archive_path)

    # 1. Inspect archive first
    inspection = inspect_backup(archive_path)
    if dry_run:
        return {
            "dry_run": True,
            "target_dir": target_dir,
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

    # 3. Extract all files safely
    restored_count = 0
    restored_bytes = 0
    total_files = inspection["total_files"]

    if fmt == "7z":
        with py7zr.SevenZipFile(archive_path, "r") as sz:
            for item in sz.list():
                if not item.is_directory and item.filename != "backup_metadata.json":
                    safe_rel = sanitize_archive_relative_path(item.filename)
                    dest = (target_dir / safe_rel).resolve()
                    if not dest.is_relative_to(target_dir):
                        logger.warning(f"Skipping dangerous archive path: {item.filename}")
                        continue
                    restored_count += 1
                    restored_bytes += item.uncompressed or 0
                    if progress_callback:
                        progress_callback(restored_count, total_files, item.filename)

            # Sanitize internal filenames before extraction so py7zr creates safe filenames on Windows
            for f in sz.files:
                if not f.is_directory and f.filename != "backup_metadata.json":
                    f._file_info["filename"] = sanitize_archive_relative_path(f.filename)

            sz.extractall(path=target_dir)

    else:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if not info.is_dir() and info.filename != "backup_metadata.json":
                    # Path traversal safety check
                    rel_parts = Path(info.filename).parts
                    if ".." in rel_parts or Path(info.filename).is_absolute():
                        logger.warning(f"Skipping dangerous archive path: {info.filename}")
                        continue

                    safe_rel = sanitize_archive_relative_path(info.filename)
                    dest = (target_dir / safe_rel).resolve()
                    if not dest.is_relative_to(target_dir):
                        logger.warning(f"Skipping path outside target directory: {info.filename}")
                        continue

                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)

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
        "target_dir": target_dir,
        "format": fmt,
        "cleaned": cleaned_stats,
        "files_restored": restored_count,
        "bytes_restored": restored_bytes,
        "inspection": inspection,
    }
