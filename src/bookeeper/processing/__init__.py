"""
Processing pipeline: semantic/TOC chunking, structured LLM extraction, and entity deduplication.
"""

from bookeeper.processing.chunker import HierarchicalChunk, HierarchicalChunker
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import (
    BookMetadata,
    Concept,
    KnowledgeExtractor,
    SectionExtraction,
)

__all__ = [
    "HierarchicalChunk",
    "HierarchicalChunker",
    "BookMetadata",
    "Concept",
    "SectionExtraction",
    "KnowledgeExtractor",
    "EntityDeduplicator",
]
