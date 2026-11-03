"""
Unit tests for Stateless Chunk Filtering Conveyor and ChunkFilter abstraction.
"""

from typing import Any
import pytest

from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.filters import (
    FilterDecision,
    ChunkFilter,
    BaseChunkFilter,
    ChunkFilterConveyor,
    FormulaDetectionFilter,
    ReferenceSectionFilter,
    BibliographicFilter,
    TechnicalBoilerplateFilter,
    DatasetStimulusFilter,
    DiagramOCRFilter,
    CaptionAndSignpostFilter,
    EmptyChunkFilter,
    ChunkSimilarityFilter,
    IdeaSimilarityFilter,
    build_chunk_filter_conveyor,
    get_default_filter_conveyor,
    has_mathematical_formula,
    get_chunk_attr,
    set_chunk_attr,
)


class DummyChunk:
    def __init__(
        self,
        chunk_id: str,
        text: str,
        section_title: str = "",
        chapter_title: str = "",
        breadcrumb: str = "",
        has_formula: bool = False,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.section_title = section_title
        self.chapter_title = chapter_title
        self.breadcrumb = breadcrumb
        self.has_formula = has_formula


class MockSpyFilter(BaseChunkFilter):
    """Spy filter for testing conveyor short-circuiting and invocation order."""

    def __init__(self, decision: FilterDecision, mutate_attr: str = "", mutate_val: Any = None):
        self.decision = decision
        self.mutate_attr = mutate_attr
        self.mutate_val = mutate_val
        self.call_count = 0

    def verify(self, chunk: Any) -> FilterDecision:
        self.call_count += 1
        if self.mutate_attr:
            set_chunk_attr(chunk, self.mutate_attr, self.mutate_val)
        return self.decision


def test_filter_decision_enum():
    """Verify FilterDecision values."""
    assert FilterDecision.SKIP.value == "skip"
    assert FilterDecision.RETAIN.value == "retain"
    assert FilterDecision.CONTINUE.value == "continue"


def test_chunk_attribute_helpers():
    """Verify get_chunk_attr and set_chunk_attr work for both objects and dicts."""
    # Test on dict
    d = {"chunk_id": "c1", "text": "hello"}
    assert get_chunk_attr(d, "text") == "hello"
    assert get_chunk_attr(d, "has_formula", False) is False
    set_chunk_attr(d, "has_formula", True)
    assert d["has_formula"] is True

    # Test on object
    obj = DummyChunk(chunk_id="c2", text="world")
    assert get_chunk_attr(obj, "text") == "world"
    assert get_chunk_attr(obj, "has_formula") is False
    set_chunk_attr(obj, "has_formula", True)
    assert obj.has_formula is True


def test_formula_detection_filter_mutates_chunk_and_continues():
    """Verify FormulaDetectionFilter sets has_formula = True and returns CONTINUE by default."""
    math_text = r"The Bellman loss satisfies LQh,ϕω = E_{(st,st+k,g) \sim D} [Qh(st,g,ϕω(st,st+k)) - Rh]^2"
    chunk = DummyChunk(chunk_id="m1", text=math_text)
    assert chunk.has_formula is False

    flt = FormulaDetectionFilter(retain_on_formula=False)
    decision = flt.verify(chunk)

    assert decision == FilterDecision.CONTINUE
    assert chunk.has_formula is True

    # Also test on dictionary chunk
    dict_chunk = {"chunk_id": "m2", "text": r"Using \chi_u(a+y) = \chi_u(a)\chi_u(y)"}
    dec_dict = flt.verify(dict_chunk)
    assert dec_dict == FilterDecision.CONTINUE
    assert dict_chunk["has_formula"] is True


def test_formula_detection_filter_default_retains_and_marks():
    """Verify FormulaDetectionFilter defaults to retain_on_formula=True, marking has_formula=True and returning RETAIN."""
    math_text = r"\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau) \right]"
    chunk = DummyChunk(chunk_id="m_grad", text=math_text)
    assert chunk.has_formula is False

    flt = FormulaDetectionFilter()  # defaults to retain_on_formula=True
    decision = flt.verify(chunk)

    assert decision == FilterDecision.RETAIN
    assert chunk.has_formula is True


def test_conveyor_retains_formula_and_shields_downstream_filters():
    """Verify conveyor immediately halts and retains formula chunk, shielding it from downstream skip filters."""
    math_text = r"\mathcal{L}_{total} = \alpha \mathcal{L}_{MSE} + \beta \mathcal{L}_{KL}"
    chunk = DummyChunk(chunk_id="m_loss", text=math_text)

    # Put a spy filter downstream that would SKIP if reached
    spy_downstream = MockSpyFilter(FilterDecision.SKIP)
    conveyor = ChunkFilterConveyor([FormulaDetectionFilter(), spy_downstream])

    decision = conveyor.verify(chunk)
    assert decision == FilterDecision.RETAIN
    assert chunk.has_formula is True
    assert spy_downstream.call_count == 0  # Downstream skip filter was short-circuited!
    assert conveyor.should_retain(chunk) is True
    assert conveyor.should_skip(chunk) is False


def test_formula_detection_filter_retain_on_formula_flag():
    """Verify FormulaDetectionFilter returns RETAIN when retain_on_formula=True."""
    math_text = r"LπBC_l = E_{(st,at) \sim D} [ ||vπl - (at-z)||^2 ]"
    chunk = DummyChunk(chunk_id="m3", text=math_text)

    flt = FormulaDetectionFilter(retain_on_formula=True)
    decision = flt.verify(chunk)

    assert decision == FilterDecision.RETAIN
    assert chunk.has_formula is True


def test_formula_detection_filter_non_math_text():
    """Verify non-math chunk is left with has_formula = False."""
    prose = "In this section, we analyze the conceptual differences between modular architectures."
    chunk = DummyChunk(chunk_id="p1", text=prose)

    flt = FormulaDetectionFilter()
    decision = flt.verify(chunk)

    assert decision == FilterDecision.CONTINUE
    assert chunk.has_formula is False


def test_empty_chunk_filter():
    """Verify EmptyChunkFilter skips empty or whitespace chunks."""
    flt = EmptyChunkFilter()
    assert flt.verify(DummyChunk("e1", "")) == FilterDecision.SKIP
    assert flt.verify(DummyChunk("e2", "   \n\t  ")) == FilterDecision.SKIP
    assert flt.verify(DummyChunk("e3", "Substantive prose")) == FilterDecision.CONTINUE


def test_reference_section_filter():
    """Verify ReferenceSectionFilter identifies reference sections by title and content."""
    flt = ReferenceSectionFilter()

    # Match by section title
    assert flt.verify(DummyChunk("r1", "Text here", section_title="References")) == FilterDecision.SKIP
    assert flt.verify(DummyChunk("r2", "Text here", section_title="Bibliography")) == FilterDecision.SKIP
    assert flt.verify(DummyChunk("r3", "Text here", section_title="Acknowledgements")) == FilterDecision.SKIP

    # Match by first line
    assert flt.verify(DummyChunk("r4", "References\n[1] Paper A\n[2] Paper B")) == FilterDecision.SKIP

    # Non-reference passes
    assert flt.verify(DummyChunk("n1", "Text here", section_title="3. Theoretical Framework")) == FilterDecision.CONTINUE


def test_bibliographic_filter():
    """Verify BibliographicFilter drops DOIs, arXiv headers, and reference slices."""
    flt = BibliographicFilter()

    # Short slice with arXiv
    arxiv_slice = "Single gluon tree amplitudes. arXiv preprint arXiv:2602.12176, 2026. Author List."
    assert flt.verify(DummyChunk("b1", arxiv_slice)) == FilterDecision.SKIP

    # Short slice with InProceedings
    proc_slice = "InProceedings of the 30th ACM SIGSOFT Symposium, pages 10-20."
    assert flt.verify(DummyChunk("b2", proc_slice)) == FilterDecision.SKIP

    # Normal conceptual chunk with 1 citation should CONTINUE
    normal_cited = "Recent breakthroughs in transformer scaling laws [1] demonstrate compute dominates loss."
    assert flt.verify(DummyChunk("n1", normal_cited)) == FilterDecision.CONTINUE


def test_technical_boilerplate_filter():
    """Verify TechnicalBoilerplateFilter skips code headers and license blocks."""
    flt = TechnicalBoilerplateFilter()

    code_headers = "#include <stdio.h>\n#include <stdlib.h>\n#ifndef HEADER_H\n#define HEADER_H\n#endif"
    assert flt.verify(DummyChunk("t1", code_headers)) == FilterDecision.SKIP

    license_block = "Permission is hereby granted, free of charge, to any person obtaining a copy. All rights reserved."
    assert flt.verify(DummyChunk("t2", license_block)) == FilterDecision.SKIP

    normal = "We implement our model in PyTorch using standard AdamW optimizer."
    assert flt.verify(DummyChunk("n2", normal)) == FilterDecision.CONTINUE


def test_dataset_stimulus_filter():
    """Verify DatasetStimulusFilter skips survey questionnaires and tokenizer noise."""
    flt = DatasetStimulusFilter()

    survey = "Question 1: What is the primary metric?\nOption A: Accuracy\nOption B: F1\nOption C: Loss"
    assert flt.verify(DummyChunk("s1", survey)) == FilterDecision.SKIP

    tokens = "The user was very low @-@ spirited and felt intimate @-@ connection."
    assert flt.verify(DummyChunk("s2", tokens)) == FilterDecision.SKIP

    normal = "Our evaluation dataset comprises 500 hand-annotated reasoning benchmarks."
    assert flt.verify(DummyChunk("n3", normal)) == FilterDecision.CONTINUE


def test_diagram_ocr_filter_respects_has_formula():
    """Verify DiagramOCRFilter drops table scraps but preserves math formulas."""
    flt = DiagramOCRFilter()

    diagram_lines = "\n".join([
        "Sym | Asym Perfect / Partial", "Game1", "Game2", "Generals", "Rollman",
        "2P / ●", "AntWar", "Loop", "Read", "Write", "Replay", "History", "Leaderboard"
    ])

    # Without has_formula: drops as diagram OCR
    chunk_diagram = DummyChunk("d1", diagram_lines, has_formula=False)
    assert flt.verify(chunk_diagram) == FilterDecision.SKIP

    # With has_formula = True: protected from diagram filter!
    chunk_math = DummyChunk("d2", diagram_lines, has_formula=True)
    assert flt.verify(chunk_math) == FilterDecision.CONTINUE


def test_conveyor_short_circuit_on_skip():
    """Verify conveyor immediately halts and drops chunk on first SKIP without executing subsequent filters."""
    spy_first = MockSpyFilter(FilterDecision.SKIP)
    spy_second = MockSpyFilter(FilterDecision.CONTINUE)

    conveyor = ChunkFilterConveyor([spy_first, spy_second])
    chunk = DummyChunk("c1", "Sample text")

    decision = conveyor.verify(chunk)
    assert decision == FilterDecision.SKIP
    assert spy_first.call_count == 1
    assert spy_second.call_count == 0  # Short-circuited!

    assert conveyor.should_skip(chunk) is True
    assert conveyor.should_retain(chunk) is False
    assert spy_first.call_count == 3
    assert spy_second.call_count == 0  # Still short-circuited!


def test_conveyor_short_circuit_on_retain():
    """Verify conveyor immediately halts and retains chunk on RETAIN without executing subsequent filters."""
    spy_first = MockSpyFilter(FilterDecision.RETAIN)
    spy_second = MockSpyFilter(FilterDecision.SKIP)

    conveyor = ChunkFilterConveyor([spy_first, spy_second])
    chunk = DummyChunk("c2", "Sample text")

    decision = conveyor.verify(chunk)
    assert decision == FilterDecision.RETAIN
    assert spy_first.call_count == 1
    assert spy_second.call_count == 0  # Short-circuited!

    assert conveyor.should_retain(chunk) is True
    assert conveyor.should_skip(chunk) is False
    assert spy_first.call_count == 3
    assert spy_second.call_count == 0  # Still short-circuited!


def test_conveyor_continue_to_end_retains_chunk():
    """Verify chunk is retained if all filters in queue return CONTINUE."""
    spy_1 = MockSpyFilter(FilterDecision.CONTINUE, mutate_attr="step1", mutate_val=True)
    spy_2 = MockSpyFilter(FilterDecision.CONTINUE, mutate_attr="step2", mutate_val=True)

    conveyor = ChunkFilterConveyor([spy_1, spy_2])
    chunk = DummyChunk("c3", "Sample text")

    decision = conveyor.verify(chunk)
    assert decision == FilterDecision.RETAIN
    assert spy_1.call_count == 1
    assert spy_2.call_count == 1
    assert getattr(chunk, "step1") is True
    assert getattr(chunk, "step2") is True

    assert conveyor.should_retain(chunk) is True
    assert spy_1.call_count == 2
    assert spy_2.call_count == 2


def test_empty_conveyor_processes_all_chunks():
    """Verify an empty conveyor retains 100% of chunks."""
    empty_conveyor = ChunkFilterConveyor([])
    chunk = DummyChunk("c4", "Any arbitrary text")

    assert empty_conveyor.verify(chunk) == FilterDecision.RETAIN
    assert empty_conveyor.should_retain(chunk) is True
    assert empty_conveyor.should_skip(chunk) is False

    chunks = [DummyChunk(f"c_{i}", f"Text {i}") for i in range(10)]
    assert len(empty_conveyor.filter_chunks(chunks)) == 10


def test_build_chunk_filter_conveyor_switchers():
    """Verify build_chunk_filter_conveyor respects on/off switches for each filter."""
    # All enabled: 8 filters in conveyor
    full_conv = build_chunk_filter_conveyor(
        enable_formula_detection=True,
        skip_empty=True,
        skip_reference_sections=True,
        skip_bibliographic=True,
        skip_technical_boilerplate=True,
        skip_dataset_stimulus=True,
        skip_diagram_ocr=True,
        skip_captions_and_signposts=True,
    )
    assert len(full_conv.filters) == 8

    # Disable formula detection and diagram ocr
    custom_conv = build_chunk_filter_conveyor(
        enable_formula_detection=False,
        skip_diagram_ocr=False,
    )
    assert len(custom_conv.filters) == 6
    assert not any(isinstance(f, FormulaDetectionFilter) for f in custom_conv.filters)
    assert not any(isinstance(f, DiagramOCRFilter) for f in custom_conv.filters)

    # All disabled -> empty conveyor (100% processed)
    none_conv = build_chunk_filter_conveyor(
        enable_formula_detection=False,
        skip_empty=False,
        skip_reference_sections=False,
        skip_bibliographic=False,
        skip_technical_boilerplate=False,
        skip_dataset_stimulus=False,
        skip_diagram_ocr=False,
        skip_captions_and_signposts=False,
    )
    assert len(none_conv.filters) == 0


def test_hierarchical_chunk_has_formula_attribute():
    """Verify HierarchicalChunk model initializes has_formula."""
    # Math text chunk
    math_chk = HierarchicalChunk(
        chunk_id="hc_1",
        book_id=1,
        book_title="Math Book",
        section_title="Formula Section",
        chapter_idx=1,
        chunk_idx=1,
        text=r"We minimize L = \sum_{i=1}^n (y_i - \hat{y}_i)^2.",
    )
    assert math_chk.has_formula is True

    # Prose chunk
    prose_chk = HierarchicalChunk(
        chunk_id="hc_2",
        book_id=1,
        book_title="History Book",
        section_title="Introduction",
        chapter_idx=1,
        chunk_idx=2,
        text="This book surveys the historical development of modern computation.",
    )
    assert prose_chk.has_formula is False


def test_filter_config_in_settings():
    """Verify FilterConfig parsing, alias support, and conveyor generation in Settings."""
    from bookeeper.config import Settings, FilterConfig

    # 1. Nested filter: config
    cfg1 = Settings(
        filter={
            "enabled": True,
            "skip_references_sections": True,
            "detect_formulas": True,
            "reference_sections": True,
            "bibliographic": False,
            "technical_boilerplate": True,
            "dataset_stimulus": True,
            "diagram_ocr": True,
        }
    )
    assert isinstance(cfg1.filter, FilterConfig)
    assert cfg1.filter.enabled is True
    assert cfg1.filter.bibliographic is False
    assert cfg1.filter_bibliographic is False
    conv1 = cfg1.get_chunk_filter_conveyor()
    assert not any(isinstance(f, BibliographicFilter) for f in conv1.filters)
    assert any(isinstance(f, FormulaDetectionFilter) for f in conv1.filters)

    # 2. Disabled master switch -> empty conveyor
    cfg_disabled = Settings(filter={"enabled": False})
    conv_disabled = cfg_disabled.get_chunk_filter_conveyor()
    assert len(conv_disabled.filters) == 0

    # 3. filters: alias support
    cfg_alias = Settings(filters={"reference_sections": False, "detect_formulas": True})
    assert cfg_alias.filter.reference_sections is False
    assert cfg_alias.filter_reference_sections is False

    # 4. Backward compatibility with flat settings
    cfg_legacy = Settings(
        filter_detect_formulas=False,
        filter_retain_formulas=False,
        filter_technical_boilerplate=False,
    )
    assert cfg_legacy.filter.detect_formulas is False
    assert cfg_legacy.filter.retain_formulas is False
    assert cfg_legacy.filter.technical_boilerplate is False


def test_chunk_similarity_filter():
    """Verify ChunkSimilarityFilter drops redundant chunks within sections."""
    csf = ChunkSimilarityFilter(similarity_threshold=0.92, sliding_window_size=3)

    # Chunk 1: Section anchor -> should CONTINUE
    c1 = DummyChunk(chunk_id="c1", text="Neural network optimization dynamics.")
    c1.embedding = [1.0, 0.0, 0.0]
    c1.chapter_idx = 1
    c1.section_title = "Optimization"
    assert csf.verify(c1) == FilterDecision.CONTINUE

    # Chunk 2: Redundant short chunk (sim=1.0 >= 0.92, len < 500) -> should SKIP
    c2 = DummyChunk(chunk_id="c2", text="Neural network optimization dynamics redundant.")
    c2.embedding = [1.0, 0.0, 0.0]
    c2.chapter_idx = 1
    c2.section_title = "Optimization"
    assert csf.verify(c2) == FilterDecision.SKIP
    assert c2.similarity_to_cover >= 0.92

    # Chunk 3: Distinct chunk (sim=0.0 < 0.92) -> should CONTINUE
    c3 = DummyChunk(chunk_id="c3", text="Gradient clipping thresholds and regularization.")
    c3.embedding = [0.0, 1.0, 0.0]
    c3.chapter_idx = 1
    c3.section_title = "Optimization"
    assert csf.verify(c3) == FilterDecision.CONTINUE

    # Chunk 4: Different section -> new section anchor -> should CONTINUE
    c4 = DummyChunk(chunk_id="c4", text="Neural network optimization dynamics.")
    c4.embedding = [1.0, 0.0, 0.0]
    c4.chapter_idx = 1
    c4.section_title = "Evaluation"
    assert csf.verify(c4) == FilterDecision.CONTINUE


def test_idea_similarity_filter():
    """Verify IdeaSimilarityFilter 3-tier routing: direct match (RETAIN), limbo (SKIP or CONTINUE), brand new (CONTINUE)."""
    class MockIdeaIndex:
        def find_nearest(self, emb):
            # emb[0] controls similarity returned
            sim = emb[0]
            if sim >= 0.87:
                return "Idea: Backprop", sim, "Summary of backprop"
            elif sim >= 0.72:
                return "Idea: Optimization", sim, "Summary of optimization"
            else:
                return "Idea: Low", sim, "Summary"

    index = MockIdeaIndex()
    isf = IdeaSimilarityFilter(
        idea_index=index,
        unassigned_threshold=0.72,
        high_confidence_threshold=0.87,
        skip_limbo_chunks=True,
        max_batch_chunk_size=1000,
    )

    # 1. High confidence (sim=0.90 >= 0.87) -> RETAIN
    c_high = DummyChunk(chunk_id="ch", text="Backpropagation algorithms and reverse mode differentiation.")
    c_high.embedding = [0.90, 0.10]
    assert isf.verify(c_high) == FilterDecision.RETAIN
    assert c_high.is_high_confidence is True
    assert c_high.matched_idea == "Idea: Backprop"

    # 2. Limbo chunk (sim=0.80, text len < 1000) -> SKIP
    c_limbo = DummyChunk(chunk_id="cl", text="Slightly overlapping concept discussion.")
    c_limbo.embedding = [0.80, 0.20]
    assert isf.verify(c_limbo) == FilterDecision.SKIP
    assert c_limbo.skipped_limbo is True

    # 3. Limbo chunk with recovery (sim=0.80, text len >= 1000) -> CONTINUE
    c_recovered = DummyChunk(chunk_id="cr", text="X" * 1200)
    c_recovered.embedding = [0.80, 0.20]
    assert isf.verify(c_recovered) == FilterDecision.CONTINUE
    assert c_recovered.recovered_from_limbo is True

    # 4. Brand new idea (sim=0.60 < 0.72) -> CONTINUE
    c_new = DummyChunk(chunk_id="cn", text="Completely unrelated quantum mechanics topic.")
    c_new.embedding = [0.60, 0.40]
    assert isf.verify(c_new) == FilterDecision.CONTINUE


def test_filter_config_chunk_and_idea_similarity_sync():
    """Verify bidirectional synchronization between nested filter config and legacy settings."""
    from bookeeper.config import Settings

    # Loading from nested config
    cfg = Settings(
        filter={
            "enabled": True,
            "chunk_similarity": {
                "enabled": True,
                "threshold": 0.95,
                "window_size": 4,
                "min_section_chunks": 3,
            },
            "idea_similarity": {
                "enabled": True,
                "unassigned_threshold": 0.70,
                "high_confidence_threshold": 0.85,
                "skip_limbo_chunks": False,
                "max_batch_chunk_size": 512,
            },
        }
    )
    # Check legacy synchronized attributes
    assert cfg.enable_hierarchical_cover is True
    assert cfg.cover_similarity_threshold == 0.95
    assert cfg.cover_window_size == 4
    assert cfg.cover_min_section_chunks == 3
    assert cfg.similarity_unassigned_threshold == 0.70
    assert cfg.similarity_high_confidence_threshold == 0.85
    assert cfg.skip_limbo_chunks is False
    assert cfg.max_batch_chunk_size == 512

    # Conveyor includes both filters
    conv = cfg.get_chunk_filter_conveyor(include_chunk_similarity=True, include_idea_similarity=True)
    assert any(isinstance(f, ChunkSimilarityFilter) for f in conv.filters)
    assert any(isinstance(f, IdeaSimilarityFilter) for f in conv.filters)

    # IdeaSimilarityFilter must appear BEFORE ChunkSimilarityFilter (Middle vs Least priority)
    idea_idx = next(i for i, f in enumerate(conv.filters) if isinstance(f, IdeaSimilarityFilter))
    chunk_idx = next(i for i, f in enumerate(conv.filters) if isinstance(f, ChunkSimilarityFilter))
    assert idea_idx < chunk_idx, "IdeaSimilarityFilter must precede ChunkSimilarityFilter in conveyor priority!"


def test_math_formula_pullback_and_shield_against_similarity_filters():
    """Verify that chunks containing mathematical formulas are never dropped by similarity filters."""
    # 1. ChunkSimilarityFilter: math formula chunk is never dropped even with 1.0 similarity to anchor
    csf = ChunkSimilarityFilter(similarity_threshold=0.80, sliding_window_size=3)
    c1 = DummyChunk(chunk_id="c1", text="Section anchor text.")
    c1.embedding = [1.0, 0.0]
    c1.chapter_idx = 1
    c1.section_title = "Math"
    assert csf.verify(c1) == FilterDecision.CONTINUE

    # Identical embedding, but contains math equation -> MUST CONTINUE (never SKIP)
    c_math = DummyChunk(chunk_id="c_math", text=r"The objective is \min_\theta \mathcal{L}(\theta) + \lambda \|\theta\|^2")
    c_math.embedding = [1.0, 0.0]
    c_math.chapter_idx = 1
    c_math.section_title = "Math"
    assert csf.verify(c_math) == FilterDecision.CONTINUE
    assert c_math.has_formula is True

    # 2. IdeaSimilarityFilter: math formula chunk is recovered from limbo
    class MockIndex:
        def find_nearest(self, emb):
            return "Prior Concept", 0.78, "Prior summary"

    isf = IdeaSimilarityFilter(
        idea_index=MockIndex(),
        unassigned_threshold=0.72,
        high_confidence_threshold=0.87,
        skip_limbo_chunks=True,
        max_batch_chunk_size=5000, # Large threshold so size recovery does not trigger
    )
    # Non-math chunk is skipped as limbo
    c_prose = DummyChunk(chunk_id="c_prose", text="Short prose text in limbo range.")
    c_prose.embedding = [0.78, 0.22]
    assert isf.verify(c_prose) == FilterDecision.SKIP

    # Math chunk in limbo range is PULLED BACK and recovered!
    c_math_limbo = DummyChunk(chunk_id="c_math_limbo", text=r"Define \rho_P := \max_{r \in \mathbb{Z}_P} \cos(2\pi r / P)")
    c_math_limbo.embedding = [0.78, 0.22]
    assert isf.verify(c_math_limbo) == FilterDecision.CONTINUE
    assert c_math_limbo.has_formula is True
    assert c_math_limbo.recovered_from_limbo is True


def test_master_filter_disabled_bypasses_all_filters_and_recovers_state(tmp_path):
    """
    Verify that when filter.enabled = false:
    1. Settings.from_yaml preserves filter.enabled = false and disables all child filters/switches.
    2. hierarchical_semantic_cover returns 100% chunks kept (0 skipped).
    3. BookProcessingState.recover_all_skipped_chunks unmarks all previously skipped chunks.
    """
    from bookeeper.config import Settings
    from bookeeper.processing.hierarchical_cover import hierarchical_semantic_cover
    from bookeeper.processing.state import BookProcessingState

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
filter:
  enabled: false
  chunk_similarity:
    enabled: true
    threshold: 0.90
  idea_similarity:
    enabled: true
    skip_limbo_chunks: true
  reference_sections: true
  skip_references_sections: true
""",
        encoding="utf-8",
    )

    settings = Settings.from_yaml(cfg_file)
    assert settings.filter.enabled is False
    assert settings.enable_hierarchical_cover is False
    assert settings.skip_references_sections is False
    assert settings.skip_limbo_chunks is False
    assert settings.filter.chunk_similarity.enabled is False
    assert settings.filter.idea_similarity.enabled is False

    # 2. hierarchical_semantic_cover with enable_chunk_similarity=False and skip_references=False
    conv = settings.get_chunk_filter_conveyor()
    assert len(conv.filters) == 0

    chunks = [
        DummyChunk(chunk_id=f"c{i}", text=f"Chunk text {i}", section_title="References", chapter_title="References")
        for i in range(5)
    ]
    for c in chunks:
        c.chapter_idx = 1
        c.embedding = [1.0, 0.0]  # Identical embeddings

    selected, skipped, decisions = hierarchical_semantic_cover(
        chunks=chunks,
        similarity_threshold=0.90,
        skip_references=settings.skip_references_sections,
        enable_chunk_similarity=settings.enable_hierarchical_cover,
        filter_conveyor=conv,
    )
    assert len(selected) == len(chunks)
    assert len(skipped) == 0, f"Expected 0 skipped chunks when filter.enabled=false, got {len(skipped)}"

    # 3. State recovery
    state_dir = tmp_path / ".book_states"
    state_dir.mkdir()
    bs = BookProcessingState(book_id=99, book_title="Test Book", state_dir=state_dir)
    # Simulate chunks previously marked as skipped in a prior run
    for c in chunks:
        bs.mark_chunk_processed(
            chunk_id=c.chunk_id,
            chunk_idx=1,
            section_title="References",
            extraction_dict={},
            duration=0.0,
            server_used="skipped_cover",
            auto_save=False,
        )
    bs.save()

    assert len(bs.data["processed_chunks"]) == 5
    recovered = bs.recover_all_skipped_chunks(chunks)
    assert len(recovered) == 5
    assert len(bs.data["processed_chunks"]) == 0


def test_verify_with_category_and_format_filter_stats():
    """Verify that verify_with_category identifies proper categories and format functions output concise summaries."""
    from bookeeper.processing.filters import (
        build_chunk_filter_conveyor,
        format_filter_stats,
        format_short_drop_stats,
    )

    conv = build_chunk_filter_conveyor(
        enable_formula_detection=True,
        retain_on_formula=True,
        skip_reference_sections=True,
        skip_bibliographic=True,
        skip_technical_boilerplate=True,
        skip_dataset_stimulus=True,
        skip_diagram_ocr=True,
    )

    # 1. Formula chunk -> RETAIN with "formulas"
    c_math = DummyChunk(chunk_id="c_math", text=r"Equation: \sum_{i=1}^N x_i = 1")
    dec, cat = conv.verify_with_category(c_math)
    assert dec == FilterDecision.RETAIN
    assert cat == "formulas"

    # 2. Reference section chunk -> SKIP with "ref_sections"
    c_ref = DummyChunk(chunk_id="c_ref", text="Normal text", section_title="References")
    dec, cat = conv.verify_with_category(c_ref)
    assert dec == FilterDecision.SKIP
    assert cat == "ref_sections"

    # 3. Bibliographic chunk -> SKIP with "biblio"
    c_bib = DummyChunk(chunk_id="c_bib", text="[1] J. Smith, arXiv:2104.12345, NeurIPS 2021.\n[2] A. Doe, In Proceedings of ICML 2022.")
    dec, cat = conv.verify_with_category(c_bib)
    assert dec == FilterDecision.SKIP
    assert cat == "biblio"

    # 4. Standard prose -> RETAIN with "passed"
    c_prose = DummyChunk(chunk_id="c_prose", text="This paper introduces a new framework for evaluation.")
    dec, cat = conv.verify_with_category(c_prose)
    assert dec == FilterDecision.RETAIN
    assert cat == "passed"

    # 5. Format stats test
    stats = {
        "formulas": 91,
        "biblio": 8,
        "ref_sections": 2,
        "diag_ocr": 1,
    }
    formatted = format_filter_stats(stats)
    assert "formulas: 91" in formatted
    assert "biblio: 8" in formatted
    assert "ref_sections: 2" in formatted
    assert "diag_ocr: 1" in formatted

    short_drop = format_short_drop_stats(stats)
    assert "biblio: 8" in short_drop
    assert "ref: 2" in short_drop
    assert "diag_ocr: 1" in short_drop
    assert "formulas" not in short_drop  # Only dropped categories in short drop


def test_stage1_conveyor_runs_even_when_chunk_similarity_disabled():
    """Verify that when chunk_similarity.enabled=False, the conveyor filter still drops boilerplate and retains formulas."""
    from bookeeper.config import Settings

    cfg = Settings(
        filter={
            "enabled": True,
            "detect_formulas": True,
            "retain_formulas": True,
            "reference_sections": True,
            "bibliographic": True,
            "diagram_ocr": True,
            "chunk_similarity": {
                "enabled": False,
            },
            "idea_similarity": {
                "enabled": False,
            },
        }
    )
    assert cfg.filter.enabled is True
    assert cfg.filter.chunk_similarity.enabled is False
    assert cfg.enable_hierarchical_cover is False

    conv = cfg.get_chunk_filter_conveyor()
    assert len(conv.filters) > 0
    # Must contain formula, ref_sections, biblio, etc.
    assert any(getattr(f, "name", "") == "formulas" for f in conv.filters)
    assert any(getattr(f, "name", "") == "ref_sections" for f in conv.filters)
    assert any(getattr(f, "name", "") == "biblio" for f in conv.filters)


def test_state_restoration_re_evaluates_against_active_conveyor():
    """Verify that cached chunks in state are re-evaluated when filter is enabled."""
    from bookeeper.processing.filters import (
        build_chunk_filter_conveyor,
        FilterDecision,
    )

    conv = build_chunk_filter_conveyor(
        enable_formula_detection=True,
        retain_on_formula=True,
        skip_reference_sections=True,
        skip_bibliographic=True,
    )

    # Simulate a chunk that was previously cached as extracted in a prior run
    c_bib = DummyChunk(chunk_id="c_bib", text="[1] J. Smith, arXiv:2104.12345, NeurIPS 2021.\n[2] A. Doe, In Proceedings of ICML 2022.")
    c_math = DummyChunk(chunk_id="c_math", text=r"Equation \nabla_\theta \mathcal{L} = 0")

    # In prior run when filter was disabled, both might have had server_used="cached"
    # Now when filter is active:
    dec_bib, cat_bib = conv.verify_with_category(c_bib)
    assert dec_bib == FilterDecision.SKIP
    assert cat_bib == "biblio"

    dec_math, cat_math = conv.verify_with_category(c_math)
    assert dec_math == FilterDecision.RETAIN
    assert cat_math == "formulas"


def test_categories_1_to_8_zero_idea_optimizations():
    """Verify Categories 1-8 structural filters and ensure they run before similarity filters."""
    from bookeeper.config import Settings
    from bookeeper.processing.hierarchical_cover import hierarchical_semantic_cover

    conv = build_chunk_filter_conveyor()

    # Category 1: Multi-line FirstName LastName bibliography & OpenReview/PMLR venues
    c_bib_multiline = DummyChunk(
        chunk_id="cat1_bib",
        text=(
            "Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec\n"
            "Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv\n"
            "preprint arXiv:2412.16720, 2024."
        ),
        section_title="Pages 11-12",
    )
    assert conv.verify_with_category(c_bib_multiline) == (FilterDecision.SKIP, "biblio")

    # Category 2: False-positive math in URLs (?id=H...) and currency ($0.10)
    url_bib_text = (
        "Angeliki Lazaridou, Alexander Peysakhovich, and Marco Baroni. Multi-agent cooperation and the\n"
        "emergence of (natural) language. In International Conference on Learning Representations, 2017.\n"
        "URL https://openreview.net/forum?id=HyzdRiR9Y7."
    )
    assert has_mathematical_formula(url_bib_text) is False
    c_url_bib = DummyChunk(chunk_id="cat2_url", text=url_bib_text, section_title="Pages 11-12")
    assert conv.verify_with_category(c_url_bib) == (FilterDecision.SKIP, "biblio")

    currency_text = "Each human annotator was paid $0.10 per HIT with a bonus of $1 for completing the survey."
    assert has_mathematical_formula(currency_text) is False

    # Category 3: Standalone captions, subfigure labels, and checkmark matrices
    c_caption = DummyChunk(
        chunk_id="cat3_cap",
        text="Table 1: Performance comparison of baseline models across five benchmark datasets. Best results are in bold.",
    )
    assert conv.verify_with_category(c_caption) == (FilterDecision.SKIP, "captions_signposts")

    c_subfig = DummyChunk(
        chunk_id="cat3_subfig",
        text="(a) Validation accuracy over training steps. (b) Token efficiency versus latency.",
    )
    assert conv.verify_with_category(c_subfig) == (FilterDecision.SKIP, "captions_signposts")

    c_matrix = DummyChunk(
        chunk_id="cat3_matrix",
        text="Method A ✓ ✓ × × ✓ ×\nMethod B × ✓ ✓ × × ✓",
    )
    assert conv.verify_with_category(c_matrix) == (FilterDecision.SKIP, "diag_ocr")

    # Category 4: Governance, AI disclosure, Reproducibility, Ethics, Funding, NeurIPS Checklist
    c_ai_use = DummyChunk(
        chunk_id="cat4_ai",
        text="AI USE STATEMENT\nLarge language models were used strictly for grammar polishing and formatting.",
        section_title="Pages 9-10",
    )
    assert conv.verify_with_category(c_ai_use) == (FilterDecision.SKIP, "ref_sections")

    c_checklist = DummyChunk(
        chunk_id="cat4_checklist",
        text="Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions?\nAnswer: [Yes]\nJustification: See Section 3 and Section 4.",
        section_title="Pages 15-16",
    )
    assert conv.verify_with_category(c_checklist) == (FilterDecision.SKIP, "ref_sections")

    # Category 5: Prompt wrappers, JSON agent schemas, VQA question dumps
    c_prompt = DummyChunk(
        chunk_id="cat5_prompt",
        text="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSolve the task.",
    )
    assert conv.verify_with_category(c_prompt) == (FilterDecision.SKIP, "dataset_stim")

    c_questions = DummyChunk(
        chunk_id="cat5_qdump",
        text="What color is the car? Where is the dog standing? How many people are visible in the background? Is it raining?",
    )
    assert conv.verify_with_category(c_questions) == (FilterDecision.SKIP, "dataset_stim")

    # Category 6: Title/Affiliation blocks and running headers
    c_affil = DummyChunk(
        chunk_id="cat6_affil",
        text="1Department of Computer Science, Stanford University, USA 2Google DeepMind\nCorrespondence to: Jane Doe <janedoe@stanford.edu>.",
    )
    assert conv.verify_with_category(c_affil) == (FilterDecision.SKIP, "tech_boiler")

    # Category 7: Short cross-reference signposts
    c_signpost = DummyChunk(
        chunk_id="cat7_signpost",
        text="Full experimental hyperparameters, ablation tables, and implementation details are provided in Appendix B.2.",
    )
    assert conv.verify_with_category(c_signpost) == (FilterDecision.SKIP, "captions_signposts")

    # Category 8: Table of contents leader dots
    c_toc = DummyChunk(
        chunk_id="cat8_toc",
        text="APPENDIX CONTENTS\nA Experimental Setup . . . . . . . . . . . . . . . . 14\nB Additional Results . . . . . . . . . . . . . . . . 18",
        section_title="Pages 13-14",
    )
    assert conv.verify_with_category(c_toc) == (FilterDecision.SKIP, "ref_sections")

    # Verify ordering: all structural filters (including CaptionAndSignpostFilter) precede IdeaSimilarityFilter & ChunkSimilarityFilter
    cfg = Settings(
        filter={
            "enabled": True,
            "captions_and_signposts": True,
            "idea_similarity": {"enabled": True},
            "chunk_similarity": {"enabled": True},
        }
    )
    full_conveyor = cfg.get_chunk_filter_conveyor(include_idea_similarity=True, include_chunk_similarity=True)
    names = [getattr(f, "name", "") for f in full_conveyor.filters]
    assert "captions_signposts" in names
    assert names.index("captions_signposts") < names.index("limbo")
    assert names.index("limbo") < names.index("chunk_sim")

    # Verify stateful document-order reference zone across "Pages X-Y" with Appendix exit
    doc_chunks = [
        DummyChunk("d1", "We conclude our study with future directions.\nReferences\n[1] First Ref, 2020.", section_title="Pages 9-10"),
        DummyChunk("d2", "Unnumbered multi-line reference entry without obvious venue token on page 11.", section_title="Pages 11-12"),
        DummyChunk("d3", "Appendix A. Proof of Theorem 1\nHere we prove the main convergence result in detail.", section_title="Pages 13-14"),
    ]
    for idx, ch in enumerate(doc_chunks):
        ch.chapter_idx = idx + 1
        ch.embedding = [1.0, 0.0]

    selected, skipped, decisions = hierarchical_semantic_cover(
        chunks=doc_chunks,
        skip_references=True,
        enable_chunk_similarity=False,
        filter_conveyor=conv,
    )
    dec_map = {d.chunk.chunk_id: d.status for d in decisions}
    # d1 contains prose before "References", d2 is inside reference zone (skipped), d3 exits reference zone at Appendix A
    assert dec_map["d2"] == "skipped_reference"
    assert dec_map["d3"] == "selected_anchor"


def test_chunk_node_filter_state_and_verifier_metrics():
    """Verify that filter_state ('skip', 'retain', 'continue') on Chunk nodes is correctly queried by KG verifier."""
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.graph_evaluator import get_chunk_filter_states, get_parent_chunk_ids

    store = ConceptGraphStore()
    c_skip = DummyChunk("c_skip", "Bibliographic reference chunk")
    c_retain = DummyChunk("c_retain", "Mathematical theorem equation chunk")
    c_continue = DummyChunk("c_continue", "Prose concept chunk extracted")

    store.add_chunk(c_skip, filter_state="skip")
    store.add_chunk(c_retain, filter_state="retain")
    store.add_chunk(c_continue, filter_state="continue")

    # Check node attributes directly in store
    node_skip = store.graph.nodes["chunk:c_skip"]
    node_retain = store.graph.nodes["chunk:c_retain"]
    node_continue = store.graph.nodes["chunk:c_continue"]

    assert node_skip.get("filter_state") == "skip"
    assert node_retain.get("filter_state") == "retain"
    assert node_continue.get("filter_state") == "continue"

    # Verify get_chunk_filter_states
    fstates = get_chunk_filter_states(store)
    assert fstates["c_skip"] == "skip"
    assert fstates["c_retain"] == "retain"
    assert fstates["c_continue"] == "continue"

    # Verify get_parent_chunk_ids uses filter_state:
    # parent chunks are retain + continue, while skip is excluded
    parents = get_parent_chunk_ids(store)
    assert "c_retain" in parents
    assert "c_continue" in parents
    assert "c_skip" not in parents





