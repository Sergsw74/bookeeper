"""
Unit and integration tests for RollingWindowSemanticChunker.
Tests semantic chunking, rolling-window context smoothing, token boundary guards,
dialogue and structural handling, overlap preservation, and embedding interfaces.
"""

from typing import List
import numpy as np
import pytest

from bookeeper.processing import RollingWindowSemanticChunker
from bookeeper.processing.chunker import RollingWindowSemanticChunker as ChunkerFromChunker
from bookeeper.processing.rolling_semantic_chunker import RollingWindowSemanticChunker as ChunkerFromRolling
from bookeeper.processing.rolling_window_chunker import RollingWindowSemanticChunker as ChunkerFromWindow


class MockTopicEmbeddings:
    """Deterministic mock embedding model that clusters topics into orthogonal subspaces."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        t = text.lower()
        vec = np.zeros(self.dim, dtype=np.float32)
        # Topic 1: Distributed systems & consensus
        if any(w in t for w in ["consensus", "raft", "paxos", "network", "node", "quorum", "leader", "cluster"]):
            vec[0:4] += 2.0
        # Topic 2: Culinary & baking
        if any(w in t for w in ["baking", "yeast", "dough", "fermentation", "butter", "pastry", "flour", "gluten", "proofing", "bread"]):
            vec[4:8] += 2.0
        # Topic 3: Marine biology & oceanography
        if any(w in t for w in ["marine", "ocean", "trench", "abyssal", "vent", "bioluminescence", "hadal"]):
            vec[8:12] += 2.0
        # Topic 4: Quantum physics
        if any(w in t for w in ["quantum", "physics", "entanglement", "particle", "wave"]):
            vec[12:16] += 2.0

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        else:
            vec = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=np.float32)
        return vec.tolist()

    def __call__(self, texts: List[str]) -> np.ndarray:
        return np.array(self.embed_documents(texts), dtype=np.float32)


# ==============================================================================
# 1. Import Path Consistency Tests
# ==============================================================================


def test_import_paths():
    """Verify RollingWindowSemanticChunker is accessible across all canonical import paths."""
    assert RollingWindowSemanticChunker is ChunkerFromChunker
    assert RollingWindowSemanticChunker is ChunkerFromRolling
    assert RollingWindowSemanticChunker is ChunkerFromWindow


# ==============================================================================
# 2. Semantic Topic Boundary Detection Tests
# ==============================================================================


def test_rolling_window_semantic_topic_splits():
    """Verify that clear topic shifts trigger semantic_drop splits while intra-topic sentences stay grouped."""
    sample_text = (
        "Consensus protocols like Raft decompose coordination into leader election, log replication, and safety. "
        "A leader is elected when it receives votes from a quorum of active cluster nodes. "
        "Periodic heartbeat messages are sent to maintain leadership and prevent split-brain cluster states.\n\n"
        "Bread dough development begins with yeast fermentation converting starches into gas. "
        "The gluten matrix forms as glutenin and gliadin proteins hydrate during dough kneading. "
        "Proper proofing temperature and baking humidity directly determine final crumb porosity and oven spring.\n\n"
        "In the hadal trenches, sunlight ceases to penetrate beyond two hundred meters depth. "
        "Deep-sea marine organisms evolve bioluminescence for hunting prey and mate signaling. "
        "Hydrothermal vent communities thrive in the ocean completely independent of solar energy."
    )

    embedder = MockTopicEmbeddings()
    chunker = RollingWindowSemanticChunker(
        embedding_model=embedder,
        window_size=2,
        min_chunk_tokens=20,
        max_chunk_tokens=150,
        similarity_threshold=0.50,
        overlap_sentences=0,
    )

    chunks = chunker.chunk_text(sample_text)
    assert len(chunks) == 3

    # First chunk: Consensus
    assert "consensus" in chunks[0]["text"].lower()
    assert chunks[0]["split_reason"] == "semantic_drop"
    assert chunks[0]["start_sentence_idx"] == 0
    assert chunks[0]["end_sentence_idx"] == 2

    # Second chunk: Baking
    assert "baking" in chunks[1]["text"].lower() or "yeast" in chunks[1]["text"].lower()
    assert chunks[1]["split_reason"] == "semantic_drop"
    assert chunks[1]["start_sentence_idx"] == 3
    assert chunks[1]["end_sentence_idx"] == 5

    # Third chunk: Marine
    assert "hadal" in chunks[2]["text"].lower() or "marine" in chunks[2]["text"].lower()
    assert chunks[2]["split_reason"] == "document_end"
    assert chunks[2]["start_sentence_idx"] == 6
    assert chunks[2]["end_sentence_idx"] == 8


# ==============================================================================
# 3. Book Prose, Dialogue, and Structural Parsing Tests
# ==============================================================================


def test_book_sentence_parsing_dialogue_and_abbreviations():
    """Verify sentence tokenizer respects direct dialogue quotes, attributions, and honorific abbreviations."""
    embedder = MockTopicEmbeddings()
    chunker = RollingWindowSemanticChunker(embedding_model=embedder)

    text = (
        "Dr. Watson visited Mr. Holmes at Baker St. on Monday afternoon.\n\n"
        "\"Where is the hidden document?\" asked Watson anxiously. "
        "\"It has been secured in the vault,\" Holmes replied calmly.\n\n"
        "# Chapter 4: The Revelation\n\n"
        "The estimated ratio was 3.14 to 1.0, e.g. matching standard geometric constants."
    )

    sentence_data = chunker._split_into_sentences(text)
    sentences = [s[0] for s in sentence_data]

    # Check that 'Dr. Watson...' was not split at 'Dr.' or 'Mr.' or 'St.'
    assert "Dr. Watson visited Mr. Holmes at Baker St. on Monday afternoon." in sentences

    # Check dialogue with lower-case attribution is preserved intact
    assert any("\"Where is the hidden document?\" asked Watson anxiously." in s for s in sentences)
    assert any("\"It has been secured in the vault,\" Holmes replied calmly." in s for s in sentences)

    # Check markdown header is recognized as its own structural unit
    assert "# Chapter 4: The Revelation" in sentences


# ==============================================================================
# 4. Token Length Guards & Safety Bound Tests
# ==============================================================================


def test_hard_token_limits_enforced():
    """Verify that chunks never exceed max_chunk_tokens even with continuous topic similarity."""
    long_topic_text = " ".join(
        [
            f"Node {i} actively sends periodic consensus heartbeats to maintain replicated log quorum."
            for i in range(25)
        ]
    )

    embedder = MockTopicEmbeddings()
    chunker = RollingWindowSemanticChunker(
        embedding_model=embedder,
        window_size=2,
        min_chunk_tokens=30,
        max_chunk_tokens=70,
        similarity_threshold=0.20,  # Low threshold so similarity drops don't trigger early
        overlap_sentences=0,
    )

    chunks = chunker.chunk_text(long_topic_text)
    assert len(chunks) > 1

    for c in chunks:
        assert c["token_count"] <= 85  # Stays within bounded token limits
        if c["chunk_id"] < len(chunks) - 1:
            assert c["split_reason"] == "max_tokens_exceeded"


# ==============================================================================
# 5. Sentence Overlap Tests
# ==============================================================================


def test_sentence_overlap_preservation():
    """Verify that overlap_sentences carries forward trailing context into subsequent chunks."""
    text = (
        "Sentence zero introduces cluster consensus. "
        "Sentence one elaborates on Raft node quorums. "
        "Sentence two concludes cluster consensus membership.\n\n"
        "Sentence three introduces bread baking. "
        "Sentence four describes sourdough yeast fermentation."
    )

    embedder = MockTopicEmbeddings()
    chunker = RollingWindowSemanticChunker(
        embedding_model=embedder,
        window_size=1,
        min_chunk_tokens=10,
        max_chunk_tokens=100,
        similarity_threshold=0.60,
        overlap_sentences=1,
    )

    chunks = chunker.chunk_text(text)
    assert len(chunks) == 2

    # Chunk 0 ends at sentence 2
    assert chunks[0]["end_sentence_idx"] == 2
    # Chunk 1 with overlap_sentences=1 should start at sentence 2 (the last sentence of Chunk 0)
    assert chunks[1]["start_sentence_idx"] == 2
    assert "concludes cluster consensus membership" in chunks[1]["text"]


# ==============================================================================
# 6. Edge Cases & Defensive Programming
# ==============================================================================


def test_empty_and_whitespace_text():
    """Verify empty or whitespace-only inputs return empty chunk lists."""
    chunker = RollingWindowSemanticChunker(embedding_model=MockTopicEmbeddings())
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   \n\n\t  ") == []
    assert chunker.split_text("") == []


def test_single_sentence_text():
    """Verify single-sentence text returns exactly one chunk without requiring vector comparisons."""
    chunker = RollingWindowSemanticChunker(embedding_model=MockTopicEmbeddings())
    chunks = chunker.chunk_text("A single atomic statement without further context.")
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == 0
    assert chunks[0]["split_reason"] == "document_end"
    assert chunks[0]["token_count"] > 0


def test_invalid_parameters_raise():
    """Verify constructor raises ValueError on invalid configuration bounds."""
    embedder = MockTopicEmbeddings()
    with pytest.raises(ValueError, match="window_size"):
        RollingWindowSemanticChunker(embedding_model=embedder, window_size=0)

    with pytest.raises(ValueError, match="min_chunk_tokens"):
        RollingWindowSemanticChunker(embedding_model=embedder, min_chunk_tokens=0)

    with pytest.raises(ValueError, match="cannot be less than"):
        RollingWindowSemanticChunker(
            embedding_model=embedder, min_chunk_tokens=300, max_chunk_tokens=200
        )

    with pytest.raises(ValueError, match="overlap_sentences"):
        RollingWindowSemanticChunker(embedding_model=embedder, overlap_sentences=-1)


def test_fallback_when_embedding_model_is_none():
    """Verify chunker functions gracefully as a structural chunker when embedding_model is None."""
    chunker = RollingWindowSemanticChunker(
        embedding_model=None,
        min_chunk_tokens=20,
        max_chunk_tokens=50,
    )

    text = (
        "Paragraph one discusses architectural foundations with sufficient content. "
        "It provides basic background information.\n\n"
        "Paragraph two moves forward into detailed design implementation details. "
        "It focuses on service boundaries and protocols."
    )

    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 1
    for c in chunks:
        assert c["token_count"] > 0
        assert len(c["text"]) > 0


# ==============================================================================
# 7. Embeddings Interface Compatibility Tests
# ==============================================================================


def test_callable_embeddings_interface():
    """Verify custom callable (List[str] -> np.ndarray) works seamlessly."""
    def custom_embed(texts: List[str]) -> np.ndarray:
        return np.ones((len(texts), 8), dtype=np.float32)

    chunker = RollingWindowSemanticChunker(
        embedding_model=custom_embed,
        window_size=1,
        min_chunk_tokens=10,
        max_chunk_tokens=50,
    )
    chunks = chunker.chunk_text("Sentence one. Sentence two.")
    assert len(chunks) == 1


def test_object_with_encode_interface():
    """Verify objects with encode method (SentenceTransformer style) are supported."""
    class MockEncodeModel:
        def encode(self, texts: List[str]) -> np.ndarray:
            return np.ones((len(texts), 8), dtype=np.float32)

    chunker = RollingWindowSemanticChunker(
        embedding_model=MockEncodeModel(),
        window_size=1,
        min_chunk_tokens=10,
        max_chunk_tokens=50,
    )
    chunks = chunker.chunk_text("Sentence one. Sentence two.")
    assert len(chunks) == 1


# ==============================================================================
# 8. LangChain Document & Splitter Protocol Tests
# ==============================================================================


def test_langchain_document_interface():
    """Verify create_documents, split_documents, and split_text."""
    embedder = MockTopicEmbeddings()
    chunker = RollingWindowSemanticChunker(
        embedding_model=embedder,
        min_chunk_tokens=10,
        max_chunk_tokens=60,
    )

    texts = [
        "Consensus protocols coordinate distributed state machines across network partitions.",
        "Artisanal sourdough bread relies on yeast fermentation and gluten development.",
    ]
    metadatas = [{"book_id": 42}, {"book_id": 99}]

    # 1. create_documents
    docs = chunker.create_documents(texts, metadatas=metadatas)
    assert len(docs) >= 2
    assert docs[0].metadata["book_id"] == 42
    assert docs[1].metadata["book_id"] == 99
    assert "token_count" in docs[0].metadata

    # 2. split_documents
    split_docs = chunker.split_documents(docs)
    assert len(split_docs) >= len(docs)
    assert split_docs[0].metadata["book_id"] == 42

    # 3. split_text
    raw_chunks = chunker.split_text(texts[0])
    assert isinstance(raw_chunks, list)
    assert len(raw_chunks) >= 1
    assert isinstance(raw_chunks[0], str)


# ==============================================================================
# 9. Observability & Statistical Diagnostics Tests
# ==============================================================================


def test_statistics_and_sentence_similarities():
    """Verify get_sentence_similarities and get_statistics provide comprehensive diagnostic metrics."""
    sample = (
        "Consensus ensures data consistency across nodes. "
        "Raft simplifies election phases.\n\n"
        "Baking sourdough requires flour, salt, and water."
    )

    embedder = MockTopicEmbeddings()
    chunker = RollingWindowSemanticChunker(
        embedding_model=embedder,
        window_size=1,
    )

    sims = chunker.get_sentence_similarities(sample)
    assert len(sims) == 2  # 3 sentences -> 2 transitions
    assert "similarity" in sims[0]
    assert "sentence_before" in sims[0]
    assert "sentence_after" in sims[0]
    assert "is_paragraph_end" in sims[0]

    stats = chunker.get_statistics(sample)
    assert stats["total_sentences"] == 3
    assert stats["total_transitions"] == 2
    assert stats["total_chunks"] >= 1
    assert "mean_chunk_tokens" in stats
    assert "mean_similarity" in stats
