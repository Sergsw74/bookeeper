"""
EPUB and ebook parser with Table of Contents (TOC) extraction and spine fallback.
"""

import logging
import re
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Section(BaseModel):
    """Structured chapter or section extracted from an ebook."""

    title: str = Field(description="Title of the chapter or section.")
    chapter_idx: int = Field(description="1-based sequence index.")
    text: str = Field(description="Cleaned extracted plain text content.")


class BookParser:
    """Parser extracting structured sections from EPUB (and PDF) files."""

    GRAPHICAL_FORMATS = {"cbr", "cbz", "cbt", "cb7", "djvu"}

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
        - Windows-1251 decoded as Latin-1 / ISO-8859-1 (e.g. 'Ñóïåðìåí' -> 'Супермен')
        - UTF-8 decoded as Latin-1 (e.g. 'Ð¡ÑƒÐ¿ÐµÑ€Ð¼ÐµÐ½' -> 'Супермен')
        - UTF-8 decoded as CP1251 (e.g. 'Р .Р .Р .' -> 'Р.Р.Р.')
        """
        if not text or not isinstance(text, str):
            return text or ""

        s = text.strip()
        if not s:
            return text

        # 1. Check for Latin-1 -> CP1251 mojibake (e.g. 'Ñóïåðìåí' -> 'Супермен')
        try:
            cand = s.encode("latin1").decode("cp1251")
            cyrillic_cand = sum(1 for c in cand if "\u0400" <= c <= "\u04FF")
            cyrillic_orig = sum(1 for c in s if "\u0400" <= c <= "\u04FF")
            if cyrillic_cand > cyrillic_orig and cyrillic_cand > 0:
                s = cand
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        # 2. Check for Latin-1 -> UTF-8 mojibake (e.g. 'Ð¡ÑƒÐ¿ÐµÑ€Ð¼ÐµÐ½')
        try:
            cand = s.encode("latin1").decode("utf-8")
            cyrillic_cand = sum(1 for c in cand if "\u0400" <= c <= "\u04FF")
            cyrillic_orig = sum(1 for c in s if "\u0400" <= c <= "\u04FF")
            if cyrillic_cand > cyrillic_orig and cyrillic_cand > 0:
                s = cand
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        # 3. Check for CP1251 -> UTF-8 mojibake
        try:
            cand = s.encode("cp1251").decode("utf-8")
            if len(cand) < len(s) and sum(1 for c in cand if "\u0400" <= c <= "\u04FF") > 0:
                s = cand
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        return s

    @classmethod
    def parse(cls, file_path: Path | str) -> List[Section]:
        """Dispatch to appropriate parser by file extension."""
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Book file not found: {path}")

        ext = path.suffix.lower()
        if ext == ".epub":
            return cls.extract_epub_sections(path)
        elif ext == ".pdf":
            return cls.extract_pdf_sections(path)
        else:
            raise ValueError(f"Unsupported ebook extension: '{ext}'. Supported: .epub, .pdf")

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

        reader = PdfReader(str(path))
        num_pages = len(reader.pages)
        sections: List[Section] = []

        current_text: List[str] = []
        current_page_start = 1

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            cleaned = cls._clean_whitespace(page_text)
            if not cleaned:
                continue

            current_text.append(cleaned)
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
