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
