"""
EPUB and ebook parser with Table of Contents (TOC) extraction and spine fallback.
"""

import io
import logging
import os
import re
import shutil
import tempfile
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


RUSSIAN_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
MOJIBAKE_CHARS = (
    r"(?<![a-zA-Z])[\u0080-\u00FF\u0192\u201A\u201E\u2026\u2020\u2021\u20AC\u2030\u2039\u203A\u2122]+(?![a-zA-Z])"
)
MOJIBAKE_RE = re.compile(MOJIBAKE_CHARS)
CP1251_UTF8_RE = re.compile(r"[РрСс][\u0080-\u00FF\u0400-\u04FF]+")


def _is_plausible_russian_word(word: str) -> bool:
    cyr = [c for c in word if "\u0400" <= c <= "\u04FF"]
    if len(cyr) < 2:
        return False
    if not any(c in RUSSIAN_VOWELS for c in cyr):
        return False
    return len(cyr) >= len(word) * 0.7


def _repair_segment(seg: str) -> str:
    # 1. UTF-8 decoded as CP1252 or Latin-1 (e.g. "Ð¡ÑƒÐ¿ÐµÑ€Ð¼ÐµÐ½" -> "Супермен")
    for enc in ("cp1252", "latin1"):
        try:
            cand = seg.encode(enc).decode("utf-8")
            cyr_cand = sum(1 for c in cand if "\u0400" <= c <= "\u04FF")
            if cyr_cand > 0:
                return cand
        except Exception:
            pass

    # 2. CP1251 decoded as Latin-1 or CP1252 (e.g. "Ðàñïèñêó" -> "Расписку", "Ñêàìüþ" -> "Скамью")
    if len(seg) >= 2:
        for enc in ("latin1", "cp1252"):
            try:
                cand = seg.encode(enc).decode("cp1251")
                if _is_plausible_russian_word(cand):
                    return cand
            except Exception:
                pass

    # 3. UTF-8 decoded as CP1251 (e.g. "Р .Р .Р ." -> "Р.Р.Р.")
    try:
        cand = seg.encode("cp1251").decode("utf-8")
        cyr_cand = sum(1 for c in cand if "\u0400" <= c <= "\u04FF")
        if cyr_cand > 0 and len(cand) < len(seg):
            return cand
    except Exception:
        pass

    return seg


class Section(BaseModel):
    """Structured chapter or section extracted from an ebook."""

    title: str = Field(description="Title of the chapter or section.")
    chapter_idx: int = Field(description="1-based sequence index.")
    text: str = Field(description="Cleaned extracted plain text content.")

    def model_post_init(self, __context) -> None:
        self.title = BookParser.repair_mojibake(self.title)
        self.text = BookParser.repair_mojibake(self.text)


class BookParser:
    """Parser extracting structured sections from EPUB, PDF, FB2, TXT, RTF, and archive files."""

    GRAPHICAL_FORMATS = {"cbr", "cbz", "cbt", "cb7", "djvu"}
    SUPPORTED_BOOK_EXTENSIONS = {".epub", ".pdf", ".txt", ".fb2", ".rtf"}
    ARCHIVE_EXTENSIONS = {
        ".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".rar", ".7z"
    }

    @classmethod
    def is_graphical_format(cls, fmt_or_path: str | Path) -> bool:
        """Check if format or file extension represents image-based graphical content (CBR, CBZ, DJVU)."""
        if isinstance(fmt_or_path, Path):
            ext = fmt_or_path.suffix.lstrip(".").lower()
        else:
            ext = str(fmt_or_path).strip().lstrip(".").lower()
        return ext in cls.GRAPHICAL_FORMATS

    @classmethod
    def only_has_graphical_formats(cls, formats: Optional[List[str]]) -> bool:
        """Return True if formats is non-empty and all formats are graphical (e.g. CBR, CBZ, DJVU)."""
        if not formats:
            return False
        return all(cls.is_graphical_format(f) for f in formats)

    @classmethod
    def directory_only_has_graphical_formats(cls, book_dir: Path | str) -> bool:
        """Check if all content files inside the book directory are graphical/comic formats."""
        p = Path(book_dir).expanduser().resolve()
        if not p.is_dir():
            return False
        content_files = [
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() not in (".opf", ".jpg", ".jpeg", ".png", ".json")
        ]
        if not content_files:
            return False
        return all(cls.is_graphical_format(f) for f in content_files)

    @classmethod
    def repair_mojibake(cls, text: Optional[str]) -> str:
        """
        Detect and repair common character encoding corruption / mojibake:
        - Windows-1251 decoded as Latin-1 / ISO-8859-1 (e.g. 'Ðàñïèñêó' -> 'Расписку', 'Ñêàìüþ' -> 'Скамью', 'Ñóïåðìåí' -> 'Супермен')
        - UTF-8 decoded as CP1252 / Latin-1 (e.g. 'Ð¡ÑƒÐ¿ÐµÑ€Ð¼ÐµÐ½' -> 'Супермен')
        - UTF-8 decoded as CP1251 (e.g. 'Р .Р .Р .' -> 'Р.Р.Р.')
        Supports mixed sentences, punctuation, and Unicode typography without corrupting valid Western accented or Russian text.
        """
        if not text or not isinstance(text, str):
            return text or ""

        s = text

        # 1. Whole-string fast checks if applicable
        for enc in ("cp1252", "latin1"):
            try:
                cand = s.encode(enc).decode("utf-8")
                cyr_cand = sum(1 for c in cand if "\u0400" <= c <= "\u04FF")
                cyr_orig = sum(1 for c in s if "\u0400" <= c <= "\u04FF")
                if cyr_cand > cyr_orig and cyr_cand > 0:
                    s = cand
                    break
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass

        try:
            cand = s.encode("latin1").decode("cp1251")
            cyr_cand = sum(1 for c in cand if "\u0400" <= c <= "\u04FF")
            cyr_orig = sum(1 for c in s if "\u0400" <= c <= "\u04FF")
            if cyr_cand > cyr_orig and cyr_cand > 0 and cyr_cand >= len(s.strip()) * 0.5:
                s = cand
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        try:
            cand = s.encode("cp1251").decode("utf-8")
            if len(cand) < len(s) and sum(1 for c in cand if "\u0400" <= c <= "\u04FF") > 0:
                s = cand
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        # 2. Segment-level repair for mixed sentences and strings with typography
        s = MOJIBAKE_RE.sub(lambda m: _repair_segment(m.group(0)), s)
        s = CP1251_UTF8_RE.sub(lambda m: _repair_segment(m.group(0)), s)

        return s

    @classmethod
    def is_archive(cls, path: Path | str) -> bool:
        """Check if path or filename represents a supported compressed archive."""
        p = Path(path)
        ext = p.suffix.lower()
        if ext in cls.ARCHIVE_EXTENSIONS:
            return True
        name_lower = p.name.lower()
        return any(name_lower.endswith(arch_ext) for arch_ext in cls.ARCHIVE_EXTENSIONS)

    @classmethod
    def parse(cls, file_path: Path | str) -> List[Section]:
        """Dispatch to appropriate parser by file extension."""
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Book file not found: {path}")

        ext = path.suffix.lower()
        if ext == ".epub":
            sections = cls.extract_epub_sections(path)
        elif ext == ".pdf":
            sections = cls.extract_pdf_sections(path)
        elif ext == ".txt":
            sections = cls.extract_txt_sections(path)
        elif ext == ".fb2":
            sections = cls.extract_fb2_sections(path)
        elif ext == ".rtf":
            sections = cls.extract_rtf_sections(path)
        elif ext in cls.ARCHIVE_EXTENSIONS or cls.is_archive(path):
            sections = cls.extract_archive_sections(path)
        else:
            raise ValueError(
                f"Unsupported ebook extension: '{ext}'. "
                f"Supported: .epub, .pdf, .txt, .fb2, .rtf, .zip, .tar, .rar, .7z"
            )

        for s in sections:
            s.title = cls.repair_mojibake(s.title)
            s.text = cls.repair_mojibake(s.text)
        return sections

    @classmethod
    def extract_epub_sections(cls, epub_path: Path | str) -> List[Section]:
        """
        Extract Table of Contents (TOC) hierarchy from EPUB file.
        Returns a list of Section(title, chapter_idx, text).
        Falls back to reading spine documents in sequential order if the TOC is missing or broken.
        """
        path = Path(epub_path).expanduser().resolve()
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError as e:
            raise ImportError(
                "ebooklib is required to parse EPUB files. Install with `pip install ebooklib`"
            ) from e

        book = epub.read_epub(str(path), options={"ignore_ncx": False})
        sections: List[Section] = []

        # 1. Attempt TOC Extraction
        toc_entries = cls._flatten_toc(book.toc)
        item_by_href = {
            item.get_name(): item
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        }

        if toc_entries:
            seen_hrefs = set()
            for title, href in toc_entries:
                base_href = href.split("#")[0] if href else ""
                if not base_href or base_href not in item_by_href:
                    continue

                item = item_by_href[base_href]
                content = item.get_content().decode("utf-8", errors="ignore")
                text = cls._html_to_clean_text(content)

                # Skip empty pages or already seen file hrefs unless there is substantial content
                if len(text) < 50:
                    continue

                # Avoid duplicate whole-document reads if multiple TOC links point to the same doc
                if base_href in seen_hrefs:
                    continue
                seen_hrefs.add(base_href)

                sections.append(
                    Section(
                        title=title.strip() or f"Section {len(sections) + 1}",
                        chapter_idx=len(sections) + 1,
                        text=text,
                    )
                )

        # 2. Fallback to spine items if TOC produced zero sections or was broken
        if not sections:
            logger.info(f"TOC empty or broken for {path.name}; falling back to spine documents.")
            spine_items = []
            item_by_id = {item.get_id(): item for item in book.get_items()}

            for entry in book.spine:
                item_id = entry[0] if isinstance(entry, (tuple, list)) else entry
                if item_id in item_by_id:
                    spine_items.append(item_by_id[item_id])

            if not spine_items:
                spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

            for item in spine_items:
                content = item.get_content().decode("utf-8", errors="ignore")
                title, text = cls._extract_title_and_text(content)

                if len(text) < 50:
                    continue

                sections.append(
                    Section(
                        title=title or f"Chapter {len(sections) + 1}",
                        chapter_idx=len(sections) + 1,
                        text=text,
                    )
                )

        return sections

    @classmethod
    def extract_pdf_sections(cls, pdf_path: Path | str) -> List[Section]:
        """Extract pages from PDF batched into sequential sections."""
        path = Path(pdf_path).expanduser().resolve()
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("pypdf is required to parse PDF files. Install with `pip install pypdf`") from e

        # First attempt standard open
        try:
            reader = PdfReader(str(path), strict=False)
        except Exception:
            # Fallback for PDFs with malformed EOF markers (e.g. %%%%EOF or trailing garbage)
            try:
                raw_bytes = path.read_bytes()
                cleaned_bytes = re.sub(rb"%+EOF\s*$", b"%%EOF\n", raw_bytes)
                reader = PdfReader(io.BytesIO(cleaned_bytes), strict=False)
            except Exception as e:
                logger.warning(f"Could not parse PDF '{path.name}': {e}")
                return []

        try:
            num_pages = len(reader.pages)
        except Exception:
            return []

        sections: List[Section] = []
        current_text: List[str] = []
        current_page_start = 1
        total_extracted_chars = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            cleaned = cls._clean_whitespace(page_text)
            if not cleaned:
                continue

            current_text.append(cleaned)
            total_extracted_chars += len(cleaned)
            combined = "\n\n".join(current_text)

            # Flush roughly every 5 pages or ~6,000 characters
            if len(combined) >= 6000 or page_idx == num_pages:
                sections.append(
                    Section(
                        title=f"Pages {current_page_start}-{page_idx}",
                        chapter_idx=len(sections) + 1,
                        text=combined,
                    )
                )
                current_text = []
                current_page_start = page_idx + 1

        if not sections and total_extracted_chars == 0:
            logger.info(f"PDF '{path.name}' has 0 extractable text characters (image/infographic scan).")

        return sections

    @classmethod
    def _split_into_sections(cls, text: str, default_title: str = "Section") -> List[Section]:
        """Split plain text into Section objects by chapter headings or paragraph batches."""
        chapter_pattern = re.compile(
            r"(?:^|\n)(?=#{1,3}\s+[^\n]+|(?:Chapter|Глава|Section|Part|Act)\s+[\dIVXLCDM]+[^\n]*)",
            re.IGNORECASE,
        )
        parts = [p for p in chapter_pattern.split(text) if p and p.strip()]
        sections: List[Section] = []

        if len(parts) > 1:
            pending_prefix = ""
            for idx, part in enumerate(parts, start=1):
                cleaned = cls._clean_whitespace(part)
                if len(cleaned) < 20:
                    pending_prefix = (pending_prefix + "\n\n" + cleaned).strip()
                    continue
                if pending_prefix:
                    cleaned = pending_prefix + "\n\n" + cleaned
                    pending_prefix = ""
                lines = part.strip().split("\n")
                first_line = lines[0].strip().lstrip("#").strip() if lines else f"Section {idx}"
                title = first_line if len(first_line) <= 80 else f"Section {idx}"
                sections.append(Section(title=title, chapter_idx=len(sections) + 1, text=cleaned))
            if pending_prefix and sections:
                sections[-1].text = sections[-1].text + "\n\n" + pending_prefix
        else:
            cleaned_full = cls._clean_whitespace(text)
            if len(cleaned_full) >= 20:
                paragraphs = text.split("\n\n")
                cur_batch: List[str] = []
                cur_len = 0
                sec_idx = 1
                for p in paragraphs:
                    p_clean = cls._clean_whitespace(p)
                    if not p_clean:
                        continue
                    cur_batch.append(p_clean)
                    cur_len += len(p_clean)
                    if cur_len >= 6000:
                        sections.append(
                            Section(
                                title=f"{default_title} {sec_idx}" if default_title != "Section" else f"Section {sec_idx}",
                                chapter_idx=sec_idx,
                                text="\n\n".join(cur_batch),
                            )
                        )
                        sec_idx += 1
                        cur_batch = []
                        cur_len = 0
                if cur_batch:
                    sections.append(
                        Section(
                            title=f"{default_title} {sec_idx}" if default_title != "Section" else f"Section {sec_idx}",
                            chapter_idx=sec_idx,
                            text="\n\n".join(cur_batch),
                        )
                    )

        if not sections:
            cleaned_full = cls._clean_whitespace(text)
            if cleaned_full:
                sections.append(Section(title=default_title, chapter_idx=1, text=cleaned_full))

        return sections

    @classmethod
    def extract_txt_sections(cls, txt_path: Path | str) -> List[Section]:
        """Extract sections from plain text file."""
        path = Path(txt_path).expanduser().resolve()
        raw = path.read_bytes()
        text = ""
        for enc in ("utf-8", "cp1251", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if not text:
            text = raw.decode("utf-8", errors="ignore")

        return cls._split_into_sections(text, default_title=path.stem)

    @classmethod
    def extract_rtf_sections(cls, rtf_path: Path | str) -> List[Section]:
        """
        Extract plain text and structured sections from an RTF (Rich Text Format) file.
        Supports standard ANSI, Windows-1251 (Cyrillic), UTF-8, and other codepages.
        """
        path = Path(rtf_path).expanduser().resolve()
        raw_bytes = path.read_bytes()

        # 1. Detect codepage from RTF header if present (e.g. \ansicpg1251)
        cpg_match = re.search(rb"\\ansicpg(\d+)", raw_bytes[:1000])
        cpg = cpg_match.group(1).decode("ascii") if cpg_match else None
        target_encoding = f"cp{cpg}" if cpg else None

        text = ""
        # 2. Extract using striprtf if available
        try:
            from striprtf.striprtf import rtf_to_text
            try:
                raw_str = raw_bytes.decode("latin1")
                text = rtf_to_text(raw_str, encoding=target_encoding or "cp1251")
            except Exception:
                raw_str = raw_bytes.decode("utf-8", errors="ignore")
                text = rtf_to_text(raw_str, encoding=target_encoding)
        except ImportError:
            # 3. Fallback regex RTF stripper
            text = cls._fallback_rtf_to_text(raw_bytes, encoding=target_encoding or "cp1251")

        # Repair any residual mojibake (e.g. latin1 CP1251 confusion)
        text = cls.repair_mojibake(text)

        return cls._split_into_sections(text, default_title=path.stem)

    @classmethod
    def _fallback_rtf_to_text(cls, raw_bytes: bytes, encoding: str = "cp1251") -> str:
        """Fallback lightweight RTF plain text extractor when striprtf is not installed."""
        def _replace_hex(match):
            return bytes([int(match.group(1), 16)])

        cleaned_bytes = re.sub(rb"\\'([0-9a-fA-F]{2})", _replace_hex, raw_bytes)
        try:
            decoded = cleaned_bytes.decode(encoding, errors="ignore")
        except Exception:
            decoded = cleaned_bytes.decode("utf-8", errors="ignore")

        decoded = re.sub(r"\{\\(?:fonttbl|colortbl|stylesheet|info)[^}]*\}", "", decoded)
        decoded = re.sub(r"\\(?:par|line)\b", "\n", decoded)
        decoded = re.sub(r"\\[a-zA-Z]+-?\d* ? ", " ", decoded)
        decoded = re.sub(r"\\[a-zA-Z]+-?\d*", "", decoded)
        decoded = re.sub(r"[{}]", "", decoded)
        return cls._clean_whitespace(decoded)

    @classmethod
    def extract_archive_sections(cls, archive_path: Path | str) -> List[Section]:
        """
        Open and inspect an archive (.zip, .tar, .rar, .7z), discover the ebook inside,
        extract it to a temporary directory, and parse its sections.
        """
        path = Path(archive_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Archive file not found: {path}")

        name_lower = path.name.lower()

        is_zip = zipfile.is_zipfile(path) or name_lower.endswith(".zip")
        is_tar = (
            tarfile.is_tarfile(path)
            or any(name_lower.endswith(ext) for ext in (".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"))
        )
        is_rar = name_lower.endswith(".rar")
        is_7z = name_lower.endswith(".7z")

        with tempfile.TemporaryDirectory() as extract_dir:
            tmp_dir = Path(extract_dir)

            if is_zip:
                with zipfile.ZipFile(path, "r") as zf:
                    member_names = [m for m in zf.namelist() if not m.endswith("/")]
                    candidates = cls._select_archive_candidates(member_names)
                    if not candidates:
                        cls._check_archive_rejection(path, member_names)
                    # Check if archive has multiple text parts
                    txt_parts = [c for c in candidates if Path(c).suffix.lower() == ".txt"]
                    if len(txt_parts) > 1 and len(candidates) == len(txt_parts):
                        txt_parts.sort()
                        all_sections = []
                        for part in txt_parts:
                            zf.extract(part, path=tmp_dir)
                            all_sections.extend(cls.extract_txt_sections(tmp_dir / part))
                        for i, s in enumerate(all_sections, start=1):
                            s.chapter_idx = i
                        return all_sections

                    chosen = candidates[0]
                    zf.extract(chosen, path=tmp_dir)
                    extracted_file = tmp_dir / chosen

            elif is_tar:
                with tarfile.open(path, "r:*") as tf:
                    member_names = [m.name for m in tf.getmembers() if m.isfile()]
                    candidates = cls._select_archive_candidates(member_names)
                    if not candidates:
                        cls._check_archive_rejection(path, member_names)
                    chosen = candidates[0]
                    member = tf.getmember(chosen)
                    try:
                        tf.extract(member, path=tmp_dir, filter="data")
                    except TypeError:
                        tf.extract(member, path=tmp_dir)
                    extracted_file = tmp_dir / chosen

            elif is_rar:
                extracted_file = cls._extract_rar(path, tmp_dir)

            elif is_7z:
                try:
                    import py7zr
                    with py7zr.SevenZipFile(path, "r") as szf:
                        member_names = szf.getnames()
                        candidates = cls._select_archive_candidates(member_names)
                        if not candidates:
                            cls._check_archive_rejection(path, member_names)
                        chosen = candidates[0]
                        szf.extract(path=tmp_dir, targets=[chosen])
                        extracted_file = tmp_dir / chosen
                except ImportError as e:
                    raise ImportError("py7zr is required to unpack .7z files. Install with `pip install py7zr`") from e

            else:
                # Generic zip fallback
                try:
                    with zipfile.ZipFile(path, "r") as zf:
                        member_names = [m for m in zf.namelist() if not m.endswith("/")]
                        candidates = cls._select_archive_candidates(member_names)
                        if not candidates:
                            cls._check_archive_rejection(path, member_names)
                        chosen = candidates[0]
                        zf.extract(chosen, path=tmp_dir)
                        extracted_file = tmp_dir / chosen
                except Exception:
                    raise ValueError(f"Unsupported archive format: '{path.suffix}'.")

            return cls.parse(extracted_file)

    @classmethod
    def _select_archive_candidates(cls, names: List[str]) -> List[str]:
        """Filter and rank archive members to find best ebook candidate."""
        valid = [
            n for n in names
            if not any(part.startswith(".") or part.startswith("__MACOSX") for part in Path(n).parts)
        ]

        # Priority ranking: EPUB > FB2 > PDF > RTF > TXT
        priority = {
            ".epub": 10,
            ".fb2": 9,
            ".pdf": 8,
            ".rtf": 7,
            ".txt": 6,
        }

        candidates = []
        for name in valid:
            ext = Path(name).suffix.lower()
            if ext in priority:
                candidates.append((priority[ext], name))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates]

    @classmethod
    def _extract_rar(cls, path: Path, tmp_dir: Path) -> Path:
        """Extract candidate ebook file from RAR archive using rarfile, bsdtar, or unrar."""
        # 1. Try rarfile
        try:
            import rarfile
            with rarfile.RarFile(path, "r") as rf:
                names = [n for n in rf.namelist() if not n.endswith("/")]
                candidates = cls._select_archive_candidates(names)
                if not candidates:
                    cls._check_archive_rejection(path, names)
                chosen = candidates[0]
                rf.extract(chosen, path=tmp_dir)
                return tmp_dir / chosen
        except Exception:
            pass

        # 2. Try bsdtar / unrar subprocess
        import subprocess
        for tool in ("/usr/bin/bsdtar", "bsdtar", "unrar"):
            tool_path = shutil.which(tool)
            if tool_path:
                try:
                    list_cmd = [tool_path, "-tf", str(path)] if "tar" in tool else [tool_path, "lb", str(path)]
                    res = subprocess.run(list_cmd, capture_output=True, text=True, check=False)
                    if res.returncode == 0:
                        names = [line.strip() for line in res.stdout.splitlines() if line.strip()]
                        candidates = cls._select_archive_candidates(names)
                        if candidates:
                            chosen = candidates[0]
                            extract_cmd = (
                                [tool_path, "-xf", str(path), "-C", str(tmp_dir), chosen]
                                if "tar" in tool
                                else [tool_path, "e", "-y", str(path), chosen, str(tmp_dir)]
                            )
                            subprocess.run(extract_cmd, capture_output=True, check=False)
                            extracted = tmp_dir / chosen
                            if extracted.is_file():
                                return extracted
                except Exception:
                    continue

        raise ValueError(f"Could not extract RAR archive '{path.name}'.")

    @classmethod
    def _check_archive_rejection(cls, archive_path: Path, names: List[str]) -> None:
        """Raise appropriate error if archive contains only images or no supported files."""
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
        non_meta = [
            n for n in names
            if not any(part.startswith(".") or part.startswith("__MACOSX") for part in Path(n).parts)
        ]
        if non_meta and all(Path(n).suffix.lower() in image_exts for n in non_meta):
            raise ValueError(f"Archive '{archive_path.name}' only contains images (comic/graphical archive).")
        raise ValueError(
            f"Archive '{archive_path.name}' contains no supported ebook files "
            f"({', '.join(sorted(cls.SUPPORTED_BOOK_EXTENSIONS))}). Found: {names[:10]}"
        )

    @classmethod
    def extract_fb2_sections(cls, fb2_path: Path | str) -> List[Section]:
        """Extract structured sections from FB2 (FictionBook) file."""
        path = Path(fb2_path).expanduser().resolve()
        raw = path.read_bytes()
        enc = "windows-1251" if b"windows-1251" in raw[:200].lower() else "utf-8"
        try:
            soup = BeautifulSoup(raw.decode(enc, errors="replace"), "xml")
        except Exception:
            try:
                soup = BeautifulSoup(raw.decode(enc, errors="replace"), "html.parser")
            except Exception:
                soup = BeautifulSoup(raw.decode("utf-8", errors="ignore"), "html.parser")

        sections: List[Section] = []
        fb2_sections = soup.find_all("section")
        for idx, s in enumerate(fb2_sections, start=1):
            title_tag = s.find("title")
            title = cls._clean_whitespace(title_tag.text) if title_tag else f"Section {idx}"
            text = cls._clean_whitespace(s.text)
            if len(text) < 40:
                continue
            sections.append(Section(title=title, chapter_idx=len(sections) + 1, text=text))

        if not sections:
            body = soup.find("body")
            if body and len(body.text.strip()) > 30:
                sections.append(Section(title=path.stem, chapter_idx=1, text=cls._clean_whitespace(body.text)))

        return sections

    @classmethod
    def _flatten_toc(cls, toc_tree: List[Any]) -> List[Tuple[str, str]]:
        """Recursively traverse ebooklib TOC hierarchy to extract (title, href) tuples."""
        results: List[Tuple[str, str]] = []
        for item in toc_tree:
            if isinstance(item, (tuple, list)):
                # Nested section list: (Section, [subsections])
                if len(item) == 2 and isinstance(item[1], (list, tuple)):
                    header, children = item
                    if hasattr(header, "title") and hasattr(header, "href"):
                        results.append((header.title, header.href))
                    results.extend(cls._flatten_toc(children))
                else:
                    results.extend(cls._flatten_toc(list(item)))
            elif hasattr(item, "title") and hasattr(item, "href"):
                results.append((item.title, item.href))
        return results

    @classmethod
    def _html_to_clean_text(cls, html_content: str) -> str:
        """Parse HTML, strip script/style/nav tags, and extract readable text."""
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "nav", "aside", "header", "footer"]):
            tag.decompose()
        raw_text = soup.get_text(separator="\n")
        return cls._clean_whitespace(raw_text)

    @classmethod
    def _extract_title_and_text(cls, html_content: str) -> Tuple[Optional[str], str]:
        """Extract first heading (h1-h3) as title along with cleaned text."""
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "nav", "aside", "header", "footer"]):
            tag.decompose()

        title = None
        for heading_tag in ["h1", "h2", "h3"]:
            h = soup.find(heading_tag)
            if h and h.get_text(strip=True):
                title = h.get_text(strip=True)
                break

        text = cls._clean_whitespace(soup.get_text(separator="\n"))
        return title, text

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Normalize line breaks and redundant spaces."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()

    @classmethod
    def sample_content(
        cls,
        target: Union[Path, str],
        max_chars: int = 4000,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Sample the beginning text, title page, or annotation of a book for metadata cleaning.
        Target can be a directory containing book files or a direct book file path.
        Returns a tuple of (extracted_text_sample, file_hint).
        """
        p = Path(target).expanduser().resolve()
        if not p.exists():
            return None, None

        if p.is_dir():
            files = [
                f for f in p.iterdir()
                if f.is_file()
                and f.suffix.lower() not in (".opf", ".jpg", ".jpeg", ".png", ".json")
                and not cls.is_graphical_format(f)
            ]
            ext_order = {
                ".epub": 1,
                ".fb2": 2,
                ".rtf": 3,
                ".pdf": 4,
                ".txt": 5,
                ".mobi": 6,
                ".azw3": 7,
                ".zip": 8,
            }
            files.sort(key=lambda f: ext_order.get(f.suffix.lower(), 99))
            if not files:
                return None, None
            target_file = files[0]
        else:
            if cls.is_graphical_format(p):
                return None, None
            target_file = p

        ext = target_file.suffix.lower()
        file_hint = target_file.name

        # 1. EPUB sampling
        if ext == ".epub":
            try:
                import zipfile
                with zipfile.ZipFile(target_file) as z:
                    sample_texts = []
                    html_files = [
                        n for n in z.namelist()
                        if n.lower().endswith((".html", ".xhtml", ".htm"))
                    ]
                    for name in html_files[:4]:
                        soup = BeautifulSoup(z.read(name), "html.parser")
                        for tag in soup(["script", "style", "nav", "aside"]):
                            tag.decompose()
                        t = cls._clean_whitespace(soup.get_text(separator=" "))
                        if len(t) > 30:
                            sample_texts.append(t)
                    if sample_texts:
                        return "\n\n".join(sample_texts)[:max_chars], file_hint
            except Exception as e:
                logger.debug(f"EPUB sampling error for {target_file}: {e}")

        # 2. RTF sampling
        elif ext == ".rtf":
            try:
                with open(target_file, "rb") as f:
                    raw = f.read(100000)
                enc = "cp1251" if b"ansicpg1251" in raw else "utf-8"
                # Strip embedded pict / jpeg hex blobs
                raw = re.sub(rb"\{[^{}]*\\\\pict[\s\S]*?\}", b"", raw)
                raw = re.sub(rb"\\\'([0-9a-fA-F]{2})", lambda m: bytes([int(m.group(1), 16)]), raw)
                text = raw.decode(enc, errors="replace")

                header_bits = []
                t = re.search(r"\\title\s+([^}\\\r\n]+)", text)
                if t and t.group(1).strip():
                    header_bits.append(f"Title: {t.group(1).strip()}")
                a = re.search(r"\\author\s+([^}\\\r\n]+)", text)
                if a and a.group(1).strip():
                    header_bits.append(f"Author: {a.group(1).strip()}")

                clean = re.sub(r"\\\*?[a-zA-Z]+(?:-?\d+)? ?", " ", text)
                clean = re.sub(r"[{}]", " ", clean)
                clean = re.sub(
                    r"\b(fonttbl|colortbl|stylesheet|fname|fswiss|froman|fmodern|Normal|heading|Annotation|FootNote|Paperw|margl|headery|footery|fcharset\d+)\b",
                    " ",
                    clean,
                    flags=re.I,
                )
                clean = re.sub(r"\s+", " ", clean).strip()

                first_coherent = re.search(r"[А-Яа-яЁёA-Za-z]{3,}", clean)
                if first_coherent:
                    clean = clean[first_coherent.start():]

                prefix = (" | ".join(header_bits) + "\n\n") if header_bits else ""
                final = (prefix + clean)[:max_chars]
                if len(final.strip()) > 5:
                    return final, file_hint
            except Exception as e:
                logger.debug(f"RTF sampling error for {target_file}: {e}")

        # 3. PDF sampling
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(target_file))
                pages_text = []
                for page in reader.pages[:3]:
                    txt = page.extract_text()
                    if txt:
                        pages_text.append(cls._clean_whitespace(txt))
                if pages_text:
                    return "\n\n".join(pages_text)[:max_chars], file_hint
            except Exception as e:
                logger.debug(f"PDF sampling error for {target_file}: {e}")

        # 4. FB2 sampling
        elif ext == ".fb2":
            try:
                with open(target_file, "rb") as f:
                    raw = f.read(50000)
                enc = "windows-1251" if b"windows-1251" in raw.lower() else "utf-8"
                soup = BeautifulSoup(raw.decode(enc, errors="replace"), "html.parser")
                fb2_texts = []
                bt = soup.find("book-title")
                if bt and bt.text:
                    fb2_texts.append(f"Title: {bt.text.strip()}")
                auth = soup.find("author")
                if auth and auth.text:
                    fb2_texts.append(f"Author: {auth.text.strip()}")
                ann = soup.find("annotation")
                if ann and ann.text:
                    fb2_texts.append(f"Annotation: {ann.text.strip()}")
                body = soup.find("body")
                if body and body.text:
                    fb2_texts.append(cls._clean_whitespace(body.text[:2000]))
                if fb2_texts:
                    return "\n\n".join(fb2_texts)[:max_chars], file_hint
            except Exception as e:
                logger.debug(f"FB2 sampling error for {target_file}: {e}")

        # 5. TXT sampling
        elif ext == ".txt":
            try:
                with open(target_file, "rb") as f:
                    raw = f.read(10000)
                for enc in ("utf-8", "cp1251", "latin-1"):
                    try:
                        text = raw.decode(enc)
                        return cls._clean_whitespace(text)[:max_chars], file_hint
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                logger.debug(f"TXT sampling error for {target_file}: {e}")

        # 6. ZIP archive inspection
        elif ext == ".zip":
            try:
                import zipfile
                with zipfile.ZipFile(target_file) as z:
                    names = z.namelist()
                    # Check if there is an inner readable text file
                    inner_candidates = [
                        n for n in names
                        if n.lower().endswith((".txt", ".fb2", ".rtf", ".html", ".htm"))
                    ]
                    if inner_candidates:
                        inner_name = inner_candidates[0]
                        inner_data = z.read(inner_name)[:10000]
                        for enc in ("utf-8", "cp1251", "latin-1"):
                            try:
                                return (
                                    cls._clean_whitespace(inner_data.decode(enc))[:max_chars],
                                    f"{file_hint} ({inner_name})",
                                )
                            except UnicodeDecodeError:
                                pass
                    inner_str = names[0] if names else ""
                    return (
                        f"ZIP archive containing: {', '.join(names[:10])}",
                        f"{file_hint} (contains {inner_str})",
                    )
            except Exception as e:
                logger.debug(f"ZIP sampling error for {target_file}: {e}")

        # 7. DJVU / other binary
        elif ext == ".djvu":
            return f"DjVu technical document: {file_hint}", file_hint

        return None, file_hint
