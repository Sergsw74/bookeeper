"""
Processing pipeline: semantic/TOC chunking, structured LLM extraction, and entity deduplication.
"""

from bookeeper.processing.chunker import HierarchicalChunk, HierarchicalChunker
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import (
    BookMetadata,
    Concept,
    ExtractedIdea,
    KnowledgeExtractor,
    SectionExtraction,
)

from bookeeper.processing.rolling_semantic_chunker import RollingWindowSemanticChunker

__all__ = [
    "HierarchicalChunk",
    "HierarchicalChunker",
    "RollingWindowSemanticChunker",
    "BookMetadata",
    "Concept",
    "ExtractedIdea",
    "SectionExtraction",
    "KnowledgeExtractor",
    "EntityDeduplicator",
]
