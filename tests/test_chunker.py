"""
Unit tests for SemanticChunker and ChapterSection parsing.
"""

from bookeeper.calibre.parser import ChapterSection
from bookeeper.config import ProcessingSettings
from bookeeper.processing.chunker import SemanticChunker


def test_chunker_basic_splitting():
    settings = ProcessingSettings(
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_size=100,
        toc_aware=True,
    )
    chunker = SemanticChunker(settings)

    # Sample paragraph text ~ 1200 characters
    p1 = "Event Sourcing is an architectural pattern where state changes are stored as an append-only sequence of immutable events. " * 4
    p2 = "CQRS (Command Query Responsibility Segregation) separates read models from write models to optimize queries and updates independently. " * 4

    section = ChapterSection(
        title="Chapter 1: Distributed Architectures",
        sequence=1,
        text=f"{p1}\n\n{p2}",
    )

    chunks = chunker.chunk_sections(
        sections=[section],
        book_id=42,
        book_title="Designing Distributed Systems",
    )

    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert c.book_id == 42
        assert c.book_title == "Designing Distributed Systems"
        assert c.chapter_title == "Chapter 1: Distributed Architectures"
        assert c.chapter_sequence == 1
        assert c.chunk_index == idx
        assert len(c.text) >= settings.min_chunk_size
        assert c.word_count > 0


def test_chunker_preserves_short_sections():
    settings = ProcessingSettings(
        chunk_size=1000,
        chunk_overlap=100,
        min_chunk_size=50,
    )
    chunker = SemanticChunker(settings)

    sec = ChapterSection(
        title="Introduction",
        sequence=1,
        text="This is a concise introductory section introducing software patterns and system design principles.",
    )

    chunks = chunker.chunk_sections([sec], book_id=1, book_title="Test Book")
    assert len(chunks) == 1
    assert chunks[0].chapter_title == "Introduction"
    assert chunks[0].chunk_index == 0


def test_chunker_skips_empty_or_subminimal_sections():
    settings = ProcessingSettings(min_chunk_size=100)
    chunker = SemanticChunker(settings)

    sec_empty = ChapterSection(title="Blank", sequence=1, text="")
    sec_tiny = ChapterSection(title="Too Short", sequence=2, text="Just ten words.")

    chunks = chunker.chunk_sections([sec_empty, sec_tiny], book_id=1, book_title="Test Book")
    assert len(chunks) == 0


def test_chunker_deterministic_ids():
    settings = ProcessingSettings(chunk_size=300, chunk_overlap=30, min_chunk_size=50)
    chunker = SemanticChunker(settings)

    sec = ChapterSection(
        title="Caching Patterns",
        sequence=3,
        text="Cache-Aside pattern loads data on demand into the cache from the underlying data store.\n\nWrite-Through cache writes data simultaneously into the cache and DB.",
    )

    chunks1 = chunker.chunk_sections([sec], book_id=7, book_title="High Performance")
    chunks2 = chunker.chunk_sections([sec], book_id=7, book_title="High Performance")

    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]
