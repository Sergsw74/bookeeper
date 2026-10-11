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
from bookeeper.processing.idea_postprocessor import (
    ConsolidatedIdeaOutput,
    IdeaCluster,
    IdeaObservation,
    IdeaPostProcessor,
    IdeaPostProcessorReport,
    normalize_idea_name,
)

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
    "IdeaPostProcessor",
    "IdeaPostProcessorReport",
    "IdeaCluster",
    "IdeaObservation",
    "ConsolidatedIdeaOutput",
    "normalize_idea_name",
]
