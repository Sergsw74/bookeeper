"""
LangChain structured output schemas and Ollama extraction chains for book metadata and concepts.
"""

import logging
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from bookeeper.config import Settings

logger = logging.getLogger(__name__)


# ==============================================================================
# Strict Pydantic Output Schemas
# ==============================================================================


class BookMetadata(BaseModel):
    """Cleaned and normalized book catalog metadata."""

    title: str = Field(
        description="Clean, canonical book title (free of subtitles, file noise, or publisher tags)."
    )
    author: str = Field(
        description="Canonical primary author or comma-separated authors (e.g. 'Martin Kleppmann')."
    )
    summary: str = Field(
        description="Concise catalog blurb (2-4 sentences) summarizing the book's core subject and value proposition."
    )


class Concept(BaseModel):
    """Canonical domain idea, architectural pattern, or technical theme."""

    name: str = Field(
        description="Canonical concise 2-4 word concept name (e.g. 'Event Sourcing', 'Circuit Breaker', 'Two-Phase Commit')."
    )
    category: str = Field(
        description="Category classification (e.g. 'Architectural Pattern', 'Data Structure', 'System Design Principle', 'Tradeoff', 'Theme')."
    )
    summary: str = Field(
        description="Clear, 1-2 sentence explanation defining what this concept is and how it functions."
    )
    related_concepts: List[str] = Field(
        default_factory=list,
        description="List of related concept names (2-4 words each) discussed in connection with this concept.",
    )


class SectionExtraction(BaseModel):
    """Extraction output containing all concepts discovered in a section."""

    concepts: List[Concept] = Field(
        default_factory=list,
        description="List of technical concepts and architectural patterns discussed.",
    )


# ==============================================================================
# KnowledgeExtractor Chain
# ==============================================================================


class KnowledgeExtractor:
    """Extracts structured metadata and concept graphs using ChatOllama with structured output."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.temperature = temperature

        self.llm = ChatOllama(
            base_url=self.base_url,
            model=self.model_name,
            temperature=self.temperature,
        )

        # Create typed structured chains using with_structured_output
        self.metadata_chain = self.llm.with_structured_output(BookMetadata)
        self.section_chain = self.llm.with_structured_output(SectionExtraction)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> "KnowledgeExtractor":
        return cls(
            base_url=base_url or settings.ollama_base_url,
            model=model or settings.llm_model,
        )

    def clean_metadata(
        self,
        raw_title: str,
        raw_authors: List[str],
        raw_comments: Optional[str] = None,
    ) -> BookMetadata:
        """Clean and normalize title, authors, and summary blurb."""
        authors_str = ", ".join(raw_authors) if raw_authors else "Unknown"
        comments_str = (raw_comments or "").strip()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a master library cataloguer and metadata cleaning specialist. "
                    "Normalize the given book title, author, and description. Produce clean, formal "
                    "catalog metadata and an executive summary blurb without advertising fluff.",
                ),
                (
                    "human",
                    "Please clean and standardize the following book metadata:\n\n"
                    "Raw Title: {raw_title}\n"
                    "Raw Authors: {raw_authors}\n"
                    "Raw Comments / Blurb:\n{raw_comments}\n",
                ),
            ]
        )

        formatted_messages = prompt.format_messages(
            raw_title=raw_title,
            raw_authors=authors_str,
            raw_comments=comments_str or "(No comments provided)",
        )

        try:
            result = self.metadata_chain.invoke(formatted_messages)
            if isinstance(result, BookMetadata):
                return result
            return BookMetadata(**dict(result))
        except Exception as e:
            logger.warning(f"Metadata cleaning LLM call failed: {e}. Falling back to raw.")
            return BookMetadata(
                title=raw_title,
                author=authors_str,
                summary=comments_str[:300] if comments_str else "No summary available.",
            )

    def extract_section(
        self,
        text: str,
        book_title: str,
        section_title: str,
    ) -> SectionExtraction:
        """Extract atomic concepts and their relationships from a section chunk."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert technical knowledge graph extractor. "
                    "Extract canonical, specific technical concepts, patterns, algorithms, and design tradeoffs. "
                    "Ensure concept names are concise (2-4 words, e.g. 'Read-Copy-Update', 'Consistent Hashing'). "
                    "Identify how these concepts relate to one another.",
                ),
                (
                    "human",
                    "Book: '{book_title}'\n"
                    "Section: '{section_title}'\n\n"
                    "Text excerpt:\n\"\"\"\n{text}\n\"\"\"\n",
                ),
            ]
        )

        formatted_messages = prompt.format_messages(
            book_title=book_title,
            section_title=section_title,
            text=text[:4000],  # Ensure token safety
        )

        try:
            result = self.section_chain.invoke(formatted_messages)
            if isinstance(result, SectionExtraction):
                return result
            return SectionExtraction(**dict(result))
        except Exception as e:
            logger.warning(f"Section extraction LLM call failed for '{section_title}': {e}")
            return SectionExtraction(concepts=[])
