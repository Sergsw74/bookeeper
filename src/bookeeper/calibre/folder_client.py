"""
Folder-based book source client.
Enables Bookkeeper to ingest books/articles directly from a download folder
(such as archive_downloader/downloads or any folder of PDFs/EPUBs/etc.)
instead of requiring a Calibre database.
"""

from contextlib import contextmanager
import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Union
import xml.etree.ElementTree as ET

from bookeeper.calibre.parser import BookParser

logger = logging.getLogger(__name__)


def _parse_opf_metadata(opf_path: Path) -> Dict[str, Any]:
    """Parse a Calibre-compatible .opf XML file."""
    if not opf_path.is_file():
        return {}
    try:
        tree = ET.parse(opf_path)
        root = tree.getroot()
        ns = {
            "dc": "http://purl.org/dc/elements/1.1/",
            "opf": "http://www.idpf.org/2007/opf",
        }

        title_elem = root.find(".//dc:title", ns)
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

        authors = []
        for creator in root.findall(".//dc:creator", ns):
            if creator.text and creator.text.strip():
                authors.append(creator.text.strip())

        desc_elem = root.find(".//dc:description", ns)
        description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

        subjects = []
        for subj in root.findall(".//dc:subject", ns):
            if subj.text and subj.text.strip():
                subjects.append(subj.text.strip())

        identifiers: Dict[str, str] = {}
        for ident in root.findall(".//dc:identifier", ns):
            scheme = (
                ident.attrib.get("{http://www.idpf.org/2007/opf}scheme")
                or ident.attrib.get("id")
                or "id"
            )
            if ident.text and ident.text.strip():
                identifiers[scheme.lower()] = ident.text.strip()

        date_elem = root.find(".//dc:date", ns) or root.find(".//date")
        pub_date = (
            BookParser.parse_publication_date(date_elem.text)
            if date_elem is not None and date_elem.text
            else None
        )
        if not pub_date:
            for ident_key, ident_val in identifiers.items():
                if "arxiv" in ident_key:
                    pub_date = BookParser.parse_publication_date(ident_val)
                    if pub_date:
                        break
        if not pub_date:
            pub_date = BookParser.parse_publication_date(opf_path.stem)

        return {
            "title": title,
            "authors": authors,
            "description": description,
            "subjects": subjects,
            "identifiers": identifiers,
            "pubdate": pub_date,
            "publication_date": pub_date,
        }
    except Exception as e:
        logger.debug(f"Failed to parse OPF file {opf_path}: {e}")
        return {}


def _update_opf_metadata(
    opf_path: Path,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    comments: Optional[str] = None,
) -> bool:
    """Update title, authors, or description in an existing .opf XML file."""
    if not opf_path.is_file():
        return False
    try:
        ET.register_namespace("", "http://www.idpf.org/2007/opf")
        ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
        ET.register_namespace("opf", "http://www.idpf.org/2007/opf")
        tree = ET.parse(opf_path)
        root = tree.getroot()
        ns = {"dc": "http://purl.org/dc/elements/1.1/"}

        meta = root.find(".//{http://www.idpf.org/2007/opf}metadata")
        if meta is None:
            meta = root.find(".//metadata")

        if title:
            title_elem = root.find(".//dc:title", ns)
            if title_elem is not None:
                title_elem.text = title
            elif meta is not None:
                t = ET.SubElement(meta, "{http://purl.org/dc/elements/1.1/}title")
                t.text = title

        if comments is not None:
            desc_elem = root.find(".//dc:description", ns)
            if desc_elem is not None:
                desc_elem.text = comments
            elif meta is not None:
                d = ET.SubElement(meta, "{http://purl.org/dc/elements/1.1/}description")
                d.text = comments

        if authors and meta is not None:
            for c in list(meta):
                if c.tag in ("{http://purl.org/dc/elements/1.1/}creator", "creator"):
                    meta.remove(c)
            for a in authors:
                a_clean = a.strip()
                if a_clean:
                    c = ET.SubElement(meta, "{http://purl.org/dc/elements/1.1/}creator")
                    c.text = a_clean

        tree.write(opf_path, encoding="utf-8", xml_declaration=True)
        return True
    except Exception as e:
        logger.warning(f"Could not update .opf metadata at {opf_path}: {e}")
        return False


class FolderBookClient:
    """
    Client for interacting with a directory of downloaded books, papers, or articles.
    Provides full drop-in compatibility with CalibreClient for listing, reading,
    exporting, and metadata updates.
    """

    def __init__(self, folder_path: Union[str, Path]):
        self.folder_path = Path(folder_path).expanduser().resolve()
        self.local_path = self.folder_path
        self._catalog: List[Dict[str, Any]] = []
        self._catalog_by_id: Dict[int, Dict[str, Any]] = {}
        self._index_json_path: Optional[Path] = None

        # Staging compatibility attributes (no-op for direct folder access)
        self.staged_db_path: Optional[Path] = None
        self.is_staged: bool = False

        self._load_catalog()

    @property
    def display_name(self) -> str:
        return f"Download Folder: {self.folder_path}"

    @property
    def source_type(self) -> str:
        return "folder"

    @property
    def is_remote_url(self) -> bool:
        return False

    @property
    def db_path(self) -> Optional[Path]:
        return None

    @property
    def active_db_path(self) -> Optional[Path]:
        return None

    def is_available(self) -> bool:
        """Check if folder exists and is a directory."""
        return self.folder_path.is_dir()

    def stage_database(
        self,
        staged_path: Optional[Union[Path, str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Optional[Path]:
        """No-op staging for folder book source."""
        return None

    def sync_database(
        self,
        create_backup: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """No-op sync for folder book source."""
        return True

    def cleanup_staged(self, delete_file: bool = True) -> None:
        """No-op cleanup for folder book source."""
        pass

    @contextmanager
    def staged_session(self, *args: Any, **kwargs: Any):
        """No-op context manager for folder book source."""
        yield self

    def _load_catalog(self) -> None:
        """Scan folder, parse index.json and .opf files, and assign stable IDs."""
        if not self.folder_path.is_dir():
            return

        discovered_items: List[Dict[str, Any]] = []
        seen_paths: set = set()

        # 1. Inspect index.json from Archive Downloader if present
        index_file = self.folder_path / "index.json"
        if index_file.is_file():
            self._index_json_path = index_file
            try:
                raw_data = json.loads(index_file.read_text(encoding="utf-8"))
                entries: List[Dict[str, Any]] = []
                if isinstance(raw_data, dict):
                    entries = list(raw_data.values())
                elif isinstance(raw_data, list):
                    entries = raw_data

                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    arxiv_id = str(entry.get("arxiv_id", "")).strip()
                    title = str(entry.get("title", "")).strip()
                    authors = entry.get("authors", [])
                    if isinstance(authors, str):
                        authors = [a.strip() for a in authors.split(",") if a.strip()]

                    rel_path = entry.get("file_path", "")
                    cand_path = self.folder_path / rel_path if rel_path else None

                    # If not found at recorded relative path, search by arxiv_id
                    if not cand_path or not cand_path.is_file():
                        if arxiv_id:
                            matches = list(self.folder_path.glob(f"{arxiv_id}*.*"))
                            matches = [
                                m for m in matches
                                if m.is_file() and not m.name.endswith((".opf", ".json", ".part", ".tmp"))
                            ]
                            if matches:
                                cand_path = matches[0]

                    if cand_path and cand_path.is_file():
                        seen_paths.add(cand_path.resolve())
                        # Check for matching .opf to enrich description
                        opf_cand = cand_path.with_suffix(".opf")
                        opf_meta = _parse_opf_metadata(opf_cand) if opf_cand.is_file() else {}

                        resolved_title = title or opf_meta.get("title") or cand_path.stem
                        resolved_authors = authors or opf_meta.get("authors") or ["Unknown"]
                        resolved_comments = (
                            entry.get("comments")
                            or opf_meta.get("description")
                            or entry.get("primary_category", "")
                        )

                        fmt = (
                            entry.get("format")
                            or cand_path.suffix.lstrip(".").upper()
                            or "PDF"
                        ).upper()

                        resolved_pubdate = (
                            opf_meta.get("publication_date")
                            or BookParser.parse_publication_date(
                                entry.get("publication_date")
                                or entry.get("published")
                                or entry.get("published_date")
                                or arxiv_id
                            )
                            or BookParser.extract_publication_date(cand_path)
                        )

                        key = f"arxiv:{arxiv_id}" if arxiv_id else cand_path.name
                        discovered_items.append(
                            {
                                "key": key,
                                "arxiv_id": arxiv_id,
                                "title": resolved_title,
                                "authors": resolved_authors,
                                "comments": resolved_comments,
                                "formats": [fmt],
                                "path": str(cand_path.resolve()),
                                "downloaded_at": entry.get("downloaded_at", ""),
                                "pubdate": resolved_pubdate,
                                "publication_date": resolved_pubdate,
                                "file_size": cand_path.stat().st_size,
                                "opf_path": str(opf_cand.resolve()) if opf_cand.is_file() else None,
                            }
                        )
            except Exception as e:
                logger.warning(f"Error parsing index.json in {self.folder_path}: {e}")

        # 2. Discover loose files in directory not tracked by index.json
        valid_extensions = (
            BookParser.SUPPORTED_BOOK_EXTENSIONS | BookParser.ARCHIVE_EXTENSIONS
        )

        for item in self.folder_path.iterdir():
            if not item.is_file():
                continue
            if item.name.startswith(".") or item.name.endswith((".opf", ".json", ".part", ".tmp")):
                continue
            if item.suffix.lower() not in valid_extensions:
                continue
            if item.resolve() in seen_paths:
                continue

            seen_paths.add(item.resolve())
            opf_cand = item.with_suffix(".opf")
            opf_meta = _parse_opf_metadata(opf_cand) if opf_cand.is_file() else {}

            title = opf_meta.get("title") or item.stem
            authors = opf_meta.get("authors") or ["Unknown"]
            comments = opf_meta.get("description", "")
            fmt = item.suffix.lstrip(".").upper()

            # Attempt to extract arXiv ID if stem starts with standard arXiv ID pattern
            arxiv_id = ""
            import re
            m = re.match(r"^(\d{4}\.\d{4,5})", item.name)
            if m:
                arxiv_id = m.group(1)

            resolved_pubdate = (
                opf_meta.get("publication_date")
                or BookParser.parse_publication_date(arxiv_id)
                or BookParser.extract_publication_date(item)
            )

            key = f"arxiv:{arxiv_id}" if arxiv_id else item.name
            discovered_items.append(
                {
                    "key": key,
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "comments": comments,
                    "formats": [fmt],
                    "path": str(item.resolve()),
                    "downloaded_at": "",
                    "pubdate": resolved_pubdate,
                    "publication_date": resolved_pubdate,
                    "file_size": item.stat().st_size,
                    "opf_path": str(opf_cand.resolve()) if opf_cand.is_file() else None,
                }
            )

        # 3. Load or generate stable, persistent integer IDs
        id_map_file = self.folder_path / ".book_ids.json"
        id_map: Dict[str, int] = {}
        if id_map_file.is_file():
            try:
                saved_map = json.loads(id_map_file.read_text(encoding="utf-8"))
                if isinstance(saved_map, dict):
                    entries = saved_map.get("entries", saved_map)
                    for k, v in entries.items():
                        try:
                            id_map[str(k)] = int(v)
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                logger.debug(f"Could not load .book_ids.json: {e}")

        # Assign IDs
        needs_save = False
        unassigned: List[Dict[str, Any]] = []

        for item in discovered_items:
            key = item["key"]
            if key in id_map:
                item["id"] = id_map[key]
            else:
                unassigned.append(item)

        if unassigned:
            if not id_map:
                # Fresh assignment: sort deterministically by download date (oldest first), then arXiv ID, then filename
                unassigned.sort(
                    key=lambda x: (
                        0 if x.get("downloaded_at") else (1 if x.get("arxiv_id") else 2),
                        x.get("downloaded_at") or "",
                        x.get("arxiv_id") or "",
                        Path(x["path"]).name.lower(),
                    )
                )
                for idx, item in enumerate(unassigned, start=1):
                    item["id"] = idx
                    id_map[item["key"]] = idx
            else:
                # Append to existing IDs
                next_id = max(id_map.values(), default=0) + 1
                for item in unassigned:
                    item["id"] = next_id
                    id_map[item["key"]] = next_id
                    next_id += 1
            needs_save = True

        if needs_save:
            try:
                temp_file = None
                with tempfile.NamedTemporaryFile("w", dir=self.folder_path, delete=False, encoding="utf-8") as tf:
                    json.dump(
                        {"next_id": max(id_map.values(), default=0) + 1, "entries": id_map},
                        tf,
                        indent=2,
                        ensure_ascii=False,
                    )
                    temp_file = Path(tf.name)
                temp_file.replace(id_map_file)
            except Exception as e:
                logger.debug(f"Could not save .book_ids.json in {self.folder_path}: {e}")
                if temp_file and temp_file.is_file():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass

        # 4. Construct sorted final catalog
        discovered_items.sort(key=lambda x: x.get("id", 0))
        self._catalog = discovered_items
        self._catalog_by_id = {b["id"]: b for b in self._catalog}

    def list_books(self, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List books from the download directory."""
        res: List[Dict[str, Any]] = []
        for b in self._catalog:
            entry = {
                "id": b["id"],
                "title": b["title"] or "Untitled",
                "authors": b["authors"],
                "comments": b.get("comments", ""),
                "formats": b["formats"],
                "path": b["path"],
            }
            if b.get("arxiv_id"):
                entry["arxiv_id"] = b["arxiv_id"]
            if b.get("file_size"):
                entry["file_size"] = b["file_size"]
            if b.get("pubdate"):
                entry["pubdate"] = b["pubdate"]
            if b.get("publication_date"):
                entry["publication_date"] = b["publication_date"]
            res.append(entry)
        return res

    def export_book(
        self,
        book_id: int,
        target_dir: Union[Path, str],
        fmt: Optional[Union[str, List[str]]] = None,
    ) -> Optional[Path]:
        """
        Locate book file for the given book_id.
        Since the book is already stored locally on disk, returns the local file path directly.
        """
        book = self._catalog_by_id.get(book_id)
        if not book:
            return None

        src_path = Path(book["path"])
        if src_path.is_file():
            return src_path

        return None

    def update_metadata(
        self,
        book_id: int,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        comments: Optional[str] = None,
    ) -> bool:
        """
        Update metadata for a book in the download folder:
        - Updates in-memory catalog
        - Updates index.json if present
        - Updates accompanying .opf XML file if present
        """
        book = self._catalog_by_id.get(book_id)
        if not book:
            return False

        if title:
            book["title"] = title.strip()
        if authors is not None:
            book["authors"] = [a.strip() for a in authors if a.strip()]
        if comments is not None:
            book["comments"] = comments.strip()

        # Update accompanying .opf file if present
        opf_path = book.get("opf_path")
        if opf_path and Path(opf_path).is_file():
            _update_opf_metadata(
                Path(opf_path),
                title=book["title"],
                authors=book["authors"],
                comments=book.get("comments"),
            )

        # Update index.json if present
        if self._index_json_path and self._index_json_path.is_file():
            try:
                raw_data = json.loads(self._index_json_path.read_text(encoding="utf-8"))
                arxiv_id = book.get("arxiv_id")
                updated = False
                if isinstance(raw_data, dict):
                    if arxiv_id and arxiv_id in raw_data:
                        if title:
                            raw_data[arxiv_id]["title"] = book["title"]
                        if authors is not None:
                            raw_data[arxiv_id]["authors"] = book["authors"]
                        if comments is not None:
                            raw_data[arxiv_id]["comments"] = book["comments"]
                        updated = True
                elif isinstance(raw_data, list):
                    for item in raw_data:
                        if arxiv_id and item.get("arxiv_id") == arxiv_id:
                            if title:
                                item["title"] = book["title"]
                            if authors is not None:
                                item["authors"] = book["authors"]
                            if comments is not None:
                                item["comments"] = book["comments"]
                            updated = True
                            break

                if updated:
                    temp_file = None
                    with tempfile.NamedTemporaryFile("w", dir=self.folder_path, delete=False, encoding="utf-8") as tf:
                        json.dump(raw_data, tf, indent=2, ensure_ascii=False)
                        temp_file = Path(tf.name)
                    temp_file.replace(self._index_json_path)
            except Exception as e:
                logger.warning(f"Could not update index.json for book #{book_id}: {e}")

        return True
