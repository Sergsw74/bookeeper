"""
Processing pipeline: semantic/TOC chunking, structured LLM extraction, and entity deduplication.
"""

from bookeeper.processing.chunker import HierarchicalChunk, HierarchicalChunker
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import (
    BatchSectionExtraction,
    BookMetadata,
    ChunkExtraction,
    Concept,
    ExtractedIdea,
    KnowledgeExtractor,
    SectionExtraction,
    batch_chunks_by_context_size,
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
from bookeeper.processing.recent_ideas import (
    BatchRoutingResult,
    ChunkPriority,
    IdeaEmbeddingIndex,
    RecentIdeaChunkRouter,
    RoutedChunk,
    normalize_sort_date,
    partition_books_into_batches,
    sort_books_chronologically,
)

__all__ = [
    "HierarchicalChunk",
    "HierarchicalChunker",
    "RollingWindowSemanticChunker",
    "BookMetadata",
    "Concept",
    "ExtractedIdea",
    "SectionExtraction",
    "ChunkExtraction",
    "BatchSectionExtraction",
    "batch_chunks_by_context_size",
    "KnowledgeExtractor",
    "EntityDeduplicator",
    "IdeaPostProcessor",
    "IdeaPostProcessorReport",
    "IdeaCluster",
    "IdeaObservation",
    "ConsolidatedIdeaOutput",
    "normalize_idea_name",
    "ChunkPriority",
    "RoutedChunk",
    "BatchRoutingResult",
    "IdeaEmbeddingIndex",
    "RecentIdeaChunkRouter",
    "normalize_sort_date",
    "sort_books_chronologically",
    "partition_books_into_batches",
]
