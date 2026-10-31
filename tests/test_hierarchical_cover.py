"""
Unit and Integration Tests for Hierarchical Semantic Cover & Section-Aware Deduplication.

Verifies:
1. Scientific reference and bibliography detection (is_reference_or_boilerplate_chunk).
2. Intra-section semantic cover with guaranteed coverage radius tau.
3. Section-stratified preservation (anchors preserved across all chapters/sections).
4. Reference section bypass without LLM extraction waste.
5. Missing embedding safe fallback behavior.
6. Integration with QuestionGenerator outline sampling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from bookeeper.config import Settings
from bookeeper.evaluation.question_generator import QuestionGenerator
from bookeeper.processing.hierarchical_cover import (
    CoverDecision,
    compute_cosine_similarity,
    hierarchical_semantic_cover,
    is_reference_or_boilerplate_chunk,
)


@dataclass
class DummyChunk:
    chunk_id: str
    text: str
    chapter_idx: int = 0
    section_title: str = ""
    chapter_title: str = ""
    breadcrumb: str = ""
    embedding: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# 1. Reference & Scientific Boilerplate Detection Tests
# ---------------------------------------------------------------------------


def test_reference_detection_by_section_title():
    """Verify title-based matching for references, bibliography, and acknowledgments."""
    titles = [
        "References",
        "References and Notes",
        "Bibliography",
        "Works Cited",
        "Literature Cited",
        "Acknowledgements",
        "Author Contributions",
        "Funding Information",
        "Competing Interests",
    ]
    for title in titles:
        chunk = DummyChunk(
            chunk_id="ref_1",
            text="Smith et al. (2020) demonstrated deep learning scaling laws.",
            section_title=title,
        )
        assert is_reference_or_boilerplate_chunk(chunk), f"Failed to match title: {title}"

    # Normal conceptual section should NOT match
    normal_chunk = DummyChunk(
        chunk_id="sec_1",
        text="We introduce a novel self-attention architecture for sparse graphs.",
        section_title="4. Architectural Details",
    )
    assert not is_reference_or_boilerplate_chunk(normal_chunk)


def test_reference_detection_by_citation_density():
    """Verify text-level citation bracket density triggers reference detection."""
    citation_heavy_text = """
    [1] Vaswani, A., et al. Attention is all you need. NeurIPS 2017.
    [2] Devlin, J., et al. BERT: Pre-training of deep bidirectional transformers. NAACL 2019.
    [3] Radford, A., et al. Language models are unsupervised multitask learners. 2019.
    [4] Brown, T., et al. Language models are few-shot learners. NeurIPS 2020.
    [5] Touvron, H., et al. Llama: Open and efficient foundation language models. 2023.
    """
    chunk = DummyChunk(
        chunk_id="ref_dense",
        text=citation_heavy_text,
        section_title="Appendix",
    )
    assert is_reference_or_boilerplate_chunk(chunk)


def test_reference_detection_non_reference_text_with_few_citations():
    """Verify conceptual chunk with 1-2 citations is NOT flagged as reference."""
    conceptual_text = """
    Recent breakthroughs in transformer scaling laws [1] demonstrate that compute
    is the primary determinant of model loss. In this section, we derive the
    optimal parameters-to-tokens ratio analytically from first principles.
    """
    chunk = DummyChunk(
        chunk_id="sec_math",
        text=conceptual_text,
        section_title="2. Theoretical Bounds",
    )
    assert not is_reference_or_boilerplate_chunk(chunk)


# ---------------------------------------------------------------------------
# 2. Intra-Section Semantic Cover Algorithm Tests
# ---------------------------------------------------------------------------


def test_hierarchical_semantic_cover_empty_and_single():
    """Verify edge cases: empty list and single chunk."""
    assert hierarchical_semantic_cover([]) == ([], [], [])

    single = DummyChunk(chunk_id="c1", text="Sole concept.", embedding=[1.0, 0.0])
    selected, skipped, decisions = hierarchical_semantic_cover([single])
    assert len(selected) == 1
    assert len(skipped) == 0
    assert len(decisions) == 1
    assert decisions[0].status == "selected_anchor"


def test_hierarchical_semantic_cover_within_section_pruning():
    """
    Verify intra-section pruning:
    Within a section, identical/near-duplicate chunk is skipped with guaranteed tau similarity,
    while distinct conceptual chunk is selected.
    """
    v_anchor = [1.0, 0.0, 0.0]
    v_redundant = [0.98, 0.1, 0.0]  # Cosine similarity > 0.95
    v_distinct = [0.0, 1.0, 0.0]   # Orthogonal: Cosine similarity ~ 0.0

    c1 = DummyChunk(chunk_id="c1", text="Anchor chunk", chapter_idx=1, section_title="Method", embedding=v_anchor)
    c2 = DummyChunk(chunk_id="c2", text="Elaboration chunk", chapter_idx=1, section_title="Method", embedding=v_redundant)
    c3 = DummyChunk(chunk_id="c3", text="Distinct equation", chapter_idx=1, section_title="Method", embedding=v_distinct)

    selected, skipped, decisions = hierarchical_semantic_cover(
        [c1, c2, c3],
        similarity_threshold=0.82,
        min_section_chunks=2,
    )

    # c1 is section anchor (selected)
    # c2 is redundant to c1 (skipped_redundant)
    # c3 is distinct (selected_distinct)
    assert len(selected) == 2
    assert len(skipped) == 1
    assert selected[0].chunk_id == "c1"
    assert selected[1].chunk_id == "c3"
    assert skipped[0].chunk_id == "c2"

    dec_map = {d.chunk.chunk_id: d for d in decisions}
    assert dec_map["c1"].status == "selected_anchor"
    assert dec_map["c2"].status == "skipped_redundant"
    assert dec_map["c2"].covered_by_chunk_id == "c1"
    assert dec_map["c2"].similarity_to_cover >= 0.82
    assert dec_map["c3"].status == "selected_distinct"


def test_hierarchical_cover_section_stratification():
    """
    Verify section stratification:
    Different chapters/sections each preserve their anchor, even if two sections
    happen to discuss similar high-level topics.
    """
    v_shared = [1.0, 0.0]

    # Chapter 1: Introduction
    ch1_c1 = DummyChunk(chunk_id="ch1_1", text="Intro concept", chapter_idx=1, section_title="Introduction", embedding=v_shared)
    # Chapter 2: Discussion (similar high-level vector)
    ch2_c1 = DummyChunk(chunk_id="ch2_1", text="Discussion concept", chapter_idx=2, section_title="Discussion", embedding=v_shared)

    selected, skipped, decisions = hierarchical_semantic_cover([ch1_c1, ch2_c1])

    # Both must be kept as anchors for their respective chapters
    assert len(selected) == 2
    assert len(skipped) == 0
    assert all(d.status == "selected_anchor" for d in decisions)


def test_hierarchical_cover_skips_reference_sections():
    """Verify entire reference sections are cleanly bypassed without embedding lookup."""
    c_method = DummyChunk(chunk_id="m1", text="Core method details", chapter_idx=1, section_title="Method", embedding=[1.0, 0.0])
    c_ref1 = DummyChunk(chunk_id="r1", text="[1] Author A, 2021", chapter_idx=2, section_title="References", embedding=[0.5, 0.5])
    c_ref2 = DummyChunk(chunk_id="r2", text="[2] Author B, 2022", chapter_idx=2, section_title="References", embedding=[0.5, 0.5])

    selected, skipped, decisions = hierarchical_semantic_cover(
        [c_method, c_ref1, c_ref2],
        skip_references=True,
    )

    assert len(selected) == 1
    assert selected[0].chunk_id == "m1"
    assert len(skipped) == 2
    assert skipped[0].chunk_id == "r1"
    assert skipped[1].chunk_id == "r2"

    ref_decs = [d for d in decisions if d.chunk.chunk_id in ("r1", "r2")]
    assert all(d.status == "skipped_reference" for d in ref_decs)


def test_hierarchical_cover_missing_embedding_fallback():
    """Verify chunks with missing embeddings are safely retained rather than discarded."""
    c1 = DummyChunk(chunk_id="c1", text="Anchor", chapter_idx=1, section_title="Intro", embedding=[1.0, 0.0])
    c_no_emb = DummyChunk(chunk_id="c2", text="Unembedded chunk", chapter_idx=1, section_title="Intro", embedding=None)

    selected, skipped, decisions = hierarchical_semantic_cover([c1, c_no_emb], min_section_chunks=2)

    assert len(selected) == 2
    assert len(skipped) == 0
    dec_map = {d.chunk.chunk_id: d for d in decisions}
    assert dec_map["c2"].status == "selected_distinct"


def test_hierarchical_cover_length_guard():
    """Verify that a short title/author snippet (<300 chars) cannot swallow a substantive paragraph (>=600 chars)."""
    short_header = DummyChunk(
        chunk_id="h1",
        text="Accurate but Not Humble: Evaluating Epistemic Humility in LLM Agents",  # ~70 chars
        chapter_idx=1,
        section_title="Pages 1-2",
        embedding=[1.0, 0.0],
    )
    long_intro = DummyChunk(
        chunk_id="i1",
        text="1 Introduction\n" + ("Large language models serve as backbones of agents. " * 30),  # >600 chars
        chapter_idx=1,
        section_title="Pages 1-2",
        embedding=[0.95, 0.05],  # Very close embedding (>0.94)
    )

    selected, skipped, decisions = hierarchical_semantic_cover(
        [short_header, long_intro],
        similarity_threshold=0.92,
        sliding_window_size=3,
    )

    # Even though similarity is > 0.92, long_intro must NOT be swallowed by the short header
    assert len(selected) == 2
    assert len(skipped) == 0
    assert selected[0].chunk_id == "h1"
    assert selected[1].chunk_id == "i1"
    assert decisions[1].status == "selected_distinct"


def test_cosine_similarity_helper():
    """Verify compute_cosine_similarity behavior with identical, orthogonal, and zero vectors."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    v_zero = [0.0, 0.0, 0.0]

    assert compute_cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-5)
    assert compute_cosine_similarity(v1, v3) == pytest.approx(0.0, abs=1e-5)
    assert compute_cosine_similarity(v1, v_zero) == 0.0


# ---------------------------------------------------------------------------
# 3. QuestionGenerator Outline Sampling Tests
# ---------------------------------------------------------------------------


def test_question_generator_hierarchical_cover_sampling(tmp_path: Path):
    """Verify QuestionGenerator uses hierarchical cover to sample diverse outline snippets."""
    fake_book = tmp_path / "paper.txt"
    fake_book.write_text("Title: Deep Representation Learning\n\nAbstract: We present an analysis.", encoding="utf-8")

    settings = Settings(
        enable_hierarchical_cover=True,
        cover_similarity_threshold=0.82,
        skip_references_sections=True,
    )

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [
        [1.0, 0.0],
        [0.99, 0.05],  # Very similar snippet
        [0.0, 1.0],   # Novel snippet
    ]

    generator = QuestionGenerator(
        settings=settings,
        embeddings=mock_embeddings,
        enable_hierarchical_cover=True,
        cover_threshold=0.82,
    )

    titles, sample = generator.extract_book_sample_context(
        file_path=fake_book,
        max_chars=2000,
    )

    assert titles == ["Full Text"]
    assert "Title: Deep Representation Learning" in sample


def test_question_generator_hierarchical_cover_disabled_switch(tmp_path: Path):
    """Verify that when enable_hierarchical_cover=False, cover filtering is completely bypassed."""
    fake_book = tmp_path / "paper.txt"
    fake_book.write_text("Quick test text.", encoding="utf-8")

    settings = Settings(enable_hierarchical_cover=False)
    mock_embeddings = MagicMock()

    generator = QuestionGenerator(
        settings=settings,
        embeddings=mock_embeddings,
        enable_hierarchical_cover=False,
    )

    titles, sample = generator.extract_book_sample_context(
        file_path=fake_book,
        max_chars=2000,
        enable_hierarchical_cover=False,
    )

    assert "Quick test text." in sample
    # embed_documents should not be called when cover is disabled
    mock_embeddings.embed_documents.assert_not_called()
