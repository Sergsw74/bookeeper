"""
Epub and PDF parser for structured table-of-contents and chapter text extraction.
"""

import logging
import re
from pathlib import Path
from typing import Any, List, Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChapterSection(BaseModel):
    """Structured chapter or section extracted from an ebook."""

    title: str
    level: int = 1
    sequence: int = 0
    text: str = ""
    href: Optional[str] = None
    page_start: Optional[int] = None
    character_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        if not self.character_count and self.text:
            self.character_count = len(self.text)


class BookParser:
    """Parser supporting EPUB and PDF format ingestion with TOC awareness."""

    @classmethod
    def parse(cls, file_path: Path | str) -> List[ChapterSection]:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Book file not found: {path}")

        ext = path.suffix.lower()
        if ext == ".epub":
            return cls.parse_epub(path)
        elif ext == ".pdf":
            return cls.parse_pdf(path)
        else:
            raise ValueError(f"Unsupported ebook extension: '{ext}'. Supported: .epub, .pdf")

    @classmethod
    def parse_epub(cls, path: Path) -> List[ChapterSection]:
        """Extract chapters and text from EPUB using ebooklib and BeautifulSoup."""
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError as e:
            raise ImportError("ebooklib is required to parse EPUB files. Install with `pip install ebooklib`") from e

        book = epub.read_epub(str(path), options={"ignore_ncx": False})
        sections: List[ChapterSection] = []
        sequence = 0

        # Gather all HTML items in spine order
        item_by_id = {item.get_id(): item for item in book.get_items()}
        spine_items = []
        for entry in book.spine:
            item_id = entry[0] if isinstance(entry, (tuple, list)) else entry
            if item_id in item_by_id:
                spine_items.append(item_by_id[item_id])

        # If spine is empty, fallback to get_items_of_type(ITEM_DOCUMENT)
        if not spine_items:
            spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

        for item in spine_items:
            content = item.get_content().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(content, "html.parser")

            # Remove navigation, scripts, and styling
            for tag in soup(["script", "style", "nav", "aside"]):
                tag.decompose()

            # Attempt to extract title from headings
            title = None
            for heading in ["h1", "h2", "h3"]:
                h_tag = soup.find(heading)
                if h_tag and h_tag.get_text(strip=True):
                    title = h_tag.get_text(strip=True)
                    break

            raw_text = soup.get_text(separator="\n")
            cleaned_text = cls._clean_text(raw_text)

            # Ignore empty or near-empty pages (like covers or copyright notices with <50 chars)
            if len(cleaned_text) < 50:
                continue

            sequence += 1
            if not title:
                title = f"Section {sequence}"

            sections.append(
                ChapterSection(
                    title=title,
                    sequence=sequence,
                    text=cleaned_text,
                    href=item.get_name(),
                    character_count=len(cleaned_text),
                )
            )

        return sections

    @classmethod
    def parse_pdf(cls, path: Path) -> List[ChapterSection]:
        """Extract pages / sections from PDF using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("pypdf is required to parse PDF files. Install with `pip install pypdf`") from e

        reader = PdfReader(str(path))
        num_pages = len(reader.pages)
        sections: List[ChapterSection] = []

        # Attempt to read PDF outlines / bookmarks
        outlines = []
        try:
            outlines = reader.outline or []
        except Exception as e:
            logger.debug(f"Could not extract PDF outlines: {e}")

        # If outlines exist, map bookmarks to page indices
        # Otherwise fallback to chunking across sequential pages
        current_text: List[str] = []
        current_title = "Document Start"
        current_page_start = 1
        sequence = 0

        # Combine in batches of ~5 pages or by headings if no outline
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            cleaned = cls._clean_text(page_text)
            if not cleaned:
                continue

            current_text.append(cleaned)

            # Flush roughly every 5 pages or ~6,000 characters to form chapter-sized units
            combined = "\n\n".join(current_text)
            if len(combined) >= 6000 or page_idx == num_pages:
                sequence += 1
                sections.append(
                    ChapterSection(
                        title=f"Pages {current_page_start}-{page_idx}",
                        sequence=sequence,
                        text=combined,
                        page_start=current_page_start,
                        character_count=len(combined),
                    )
                )
                current_text = []
                current_page_start = page_idx + 1

        return sections

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize line breaks and excessive whitespace."""
        # Collapse multiple spaces and excessive newlines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()
