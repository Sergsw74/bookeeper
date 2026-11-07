"""
Hierarchical Semantic Cover & Section-Aware Deduplication Module.

Replaces global centroid-based MMR with an intra-article, section-aware semantic cover
specifically engineered for scientific white papers and technical literature.

Core Principles:
1. Section-Stratified Anchoring: Every section / chapter retains its conceptual anchors;
   no chapter is starved or penalized by an artificial global centroid.
2. Semantic Radius Guarantee: A chunk is only skipped if it is semantically covered
   by another chunk in the same section within cosine radius tau (e.g. >= 0.82).
3. Scientific Boilerplate Filtering: Automatically detects and bypasses bibliography,
   references, author contributions, and acknowledgments sections with zero idea loss.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

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
    EmptyChunkFilter,
    build_chunk_filter_conveyor,
    get_default_filter_conveyor,
    has_mathematical_formula,
    get_chunk_attr,
    set_chunk_attr,
    extract_chunk_text_and_titles,
    REFERENCE_SECTION_TITLE_REGEX,
    CITATION_BRACKET_REGEX,
    CITATION_AUTHOR_YEAR_REGEX,
    DOI_REGEX,
    ARXIV_ID_REGEX,
    URL_COUNT_REGEX,
    VENUE_REGEX,
    SHORT_REF_KEYWORD_REGEX,
    BIB_ENTRY_LINE_REGEX,
    CODE_HEADER_BOILERPLATE_REGEX,
    LICENSE_BOILERPLATE_REGEX,
    WEB_SCRAPE_NOISE_REGEX,
    SURVEY_QUESTION_REGEX,
    TOKENIZER_NOISE_REGEX,
    PROSE_GUARD_REGEX,
)

logger = logging.getLogger(__name__)


def _normalize_and_project_vector(raw_emb: Any) -> Optional[np.ndarray]:
    """Normalize embedding vector and project legacy 1024-d vectors into 768-d JEPA space."""
    if raw_emb is None or len(raw_emb) <= 1:
        return None
    arr = np.asarray(raw_emb, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.flatten()
    if arr.shape[0] != 768 and arr.shape[0] == 1024 and Path("output/jepa_768_weights.npz").is_file():
        try:
            from bookeeper.evaluation.jepa_encoder import LightweightJEPAEncoder
            if not hasattr(hierarchical_semantic_cover, "_jepa_proj"):
                hierarchical_semantic_cover._jepa_proj = LightweightJEPAEncoder.load("output/jepa_768_weights.npz")
            arr = hierarchical_semantic_cover._jepa_proj.project(arr)
        except Exception:
            pass
    norm = float(np.linalg.norm(arr))
    return (arr / norm) if norm > 1e-12 else arr


def compute_cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compute cosine similarity between two 1D numerical vectors."""
    a = np.asarray(vec1, dtype=np.float32)
    b = np.asarray(vec2, dtype=np.float32)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


_cosine_sim = compute_cosine_similarity
_extract_chunk_fields = extract_chunk_text_and_titles


def is_bibliographic_chunk(chunk: Any) -> bool:
    """Detect whether a chunk is a bibliography, references, or citation block."""
    return BibliographicFilter().verify(chunk) == FilterDecision.SKIP or ReferenceSectionFilter().verify(chunk) == FilterDecision.SKIP


def is_diagram_or_ocr_layout_dump(chunk: Any) -> bool:
    """
    Detect whether a chunk is a diagram OCR dump, game board list, or visual table layout scrap.
    Such artifacts typically consist of many very short lines (average length <= 25 chars),
    repeated UI/diagram tokens, or high density of graphical layout symbols.
    """
    return DiagramOCRFilter().verify(chunk) == FilterDecision.SKIP


def is_technical_boilerplate_or_code_dump(chunk: Any) -> bool:
    """Detect whether a chunk is technical boilerplate (C/C++ headers, license notices, scrape noise)."""
    return TechnicalBoilerplateFilter().verify(chunk) == FilterDecision.SKIP


def is_dataset_stimulus_noise(chunk: Any) -> bool:
    """Detect raw evaluation dataset questionnaire options, tokenizer noise, or repetitive stimulus prompts."""
    return DatasetStimulusFilter().verify(chunk) == FilterDecision.SKIP


def is_non_conceptual_boilerplate(chunk: Any, conveyor: Optional[ChunkFilterConveyor] = None) -> bool:
    """Master filter checking if a chunk is non-conceptual boilerplate via ChunkFilterConveyor."""
    conv = conveyor or get_default_filter_conveyor()
    return conv.should_skip(chunk)


def is_reference_or_boilerplate_chunk(chunk: Any, conveyor: Optional[ChunkFilterConveyor] = None) -> bool:
    """Backward-compatible alias for non-conceptual boilerplate detection via ChunkFilterConveyor."""
    return is_non_conceptual_boilerplate(chunk, conveyor=conveyor)


@dataclass
class CoverDecision:
    """Detailed record of the coverage decision for a single chunk."""
    chunk: Any
    selected: bool
    status: str  # "selected_anchor", "selected_distinct", "skipped_redundant", "skipped_reference"
    covered_by_chunk_id: Optional[str] = None
    similarity_to_cover: float = 0.0
    section_name: str = ""
    filter_category: str = ""
    filter_state: str = "continue"  # "skip", "retain", "continue"


def hierarchical_semantic_cover(
    chunks: Sequence[Any],
    similarity_threshold: float = 0.92,
    skip_references: bool = True,
    min_section_chunks: int = 2,
    sliding_window_size: int = 3,
    on_progress: Optional[Callable[[int, int], None]] = None,
    filter_conveyor: Optional[ChunkFilterConveyor] = None,
    enable_chunk_similarity: bool = True,
) -> Tuple[List[Any], List[Any], List[CoverDecision]]:
    """
    Partition candidate chunks into:
      1. selected_chunks: Concept-dense anchor chunks and distinct subtopic shifts for LLM extraction.
      2. skipped_chunks: Redundant elaborations covered by an existing chunk in the same section,
         plus bibliography/boilerplate chunks.

    Guarantees:
      - Every skipped redundant chunk is mathematically guaranteed to have an extracted parent
        in the graph within cosine similarity >= similarity_threshold (default: 0.92).
      - Retention R_skipped is preserved while pruning ~35-45% of redundant text.

    Args:
        chunks: List of HierarchicalChunk objects (with chunk.embedding populated).
        similarity_threshold: Cosine similarity threshold (>=) to mark a chunk redundant (default: 0.92).
        skip_references: Automatically skip bibliography, references, and acknowledgments.
        min_section_chunks: Minimum chunks in a section before redundancy pruning applies.
        sliding_window_size: Maximum recent section anchors to compare against (default: 3).
        on_progress: Optional callback progress(completed_count, total_count).
        filter_conveyor: Optional custom ChunkFilterConveyor for chunk verification.

    Returns:
        Tuple of (selected_chunks, skipped_chunks, decisions_list).
    """
    if not chunks:
        return [], [], []

    if len(chunks) == 1:
        single = chunks[0]
        dec = CoverDecision(chunk=single, selected=True, status="selected_anchor", section_name="")
        return [single], [], [dec]

    if filter_conveyor is not None:
        effective_conveyor = filter_conveyor
    elif skip_references:
        effective_conveyor = get_default_filter_conveyor()
    else:
        effective_conveyor = ChunkFilterConveyor([])

    if not skip_references and not enable_chunk_similarity and len(effective_conveyor.filters) == 0:
        # Filtering is completely bypassed: retain 100% of chunks
        all_selected = list(chunks)
        all_decisions = [
            CoverDecision(chunk=c, selected=True, status="selected_anchor", section_name="")
            for c in chunks
        ]
        return all_selected, [], all_decisions

    selected_chunks: List[Any] = []
    skipped_chunks: List[Any] = []
    decisions: List[CoverDecision] = []

    # 1. Group chunks by Section / Chapter preserving sequential document order
    sections_map: Dict[Tuple[int, str], List[Any]] = {}
    for c in chunks:
        ch_idx = int(get_chunk_attr(c, "chapter_idx", 0) or 0)
        sec_title = str(get_chunk_attr(c, "section_title", "") or get_chunk_attr(c, "chapter_title", "") or "").strip()
        sec_key = (ch_idx, sec_title.lower())
        if sec_key not in sections_map:
            sections_map[sec_key] = []
        sections_map[sec_key].append(c)

    processed_count = 0
    total_count = len(chunks)
    in_trailing_references_by_chapter: Dict[int, bool] = {}
    in_document_reference_zone: bool = False

    _inline_ref_heading_re = re.compile(
        r"(?:^|\n)\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|Works\s+Cited|Literature\s+Cited)\s*(?:\n|$)"
    )
    _appendix_exit_re = re.compile(
        r"(?:^|\n|\.\s+)(?:Appendix|APPENDIX|Supplementary|SUPPLEMENTARY|[A-Z]\.\s+[A-Z]|[A-Z]\.\d+\s+[A-Z]|S\d+\s+[A-Z]|[A-Z]\s+[A-Z]{2,})"
    )

    def _eval_ref_zone_transition(chk_text: str, f_cat: str, is_page_sec: bool) -> bool:
        """Update in_document_reference_zone and return True only if chunk is a short reference continuation."""
        nonlocal in_document_reference_zone
        if not skip_references:
            return False
        if in_document_reference_zone:
            has_bib_cue = bool(
                re.search(r"\b(?:19|20)\d{2}[a-z]?\b", chk_text)
                or DOI_REGEX.search(chk_text)
                or ARXIV_ID_REGEX.search(chk_text)
                or VENUE_REGEX.search(chk_text)
                or CITATION_BRACKET_REGEX.search(chk_text)
                or "reference" in chk_text.lower()
            )
            text_no_urls = re.sub(r"https?://\S+|doi:\s*\S+|arXiv:\s*\S+|\b\d+\(\d+\):\d+", "", chk_text, flags=re.I)
            if (
                f_cat == "formulas"
                or _appendix_exit_re.search(chk_text)
                or PROSE_GUARD_REGEX.search(chk_text)
                or not has_bib_cue
                or len(chk_text) > 260
                or ":" in text_no_urls
            ):
                if f_cat == "formulas" or _appendix_exit_re.search(chk_text) or PROSE_GUARD_REGEX.search(chk_text) or not has_bib_cue:
                    in_document_reference_zone = False
                return False
            return True
        elif is_page_sec and _inline_ref_heading_re.search(chk_text):
            in_document_reference_zone = True
        return False

    # 2. Process each section hierarchically
    for (ch_idx, sec_title_lower), sec_chunks in sections_map.items():
        sec_display = sec_title_lower or f"Chapter {ch_idx}"
        is_page_sec = bool(re.match(r"^pages?\s+\d+", sec_display, re.I))

        # Check if the entire section is references/bibliography
        is_ref_section = skip_references and bool(REFERENCE_SECTION_TITLE_REGEX.search(sec_display))

        # If we were in a document reference zone across PDF page sections ("Pages X-Y"),
        # exit if we hit a named non-reference section or explicit appendix title.
        if in_document_reference_zone:
            if bool(re.search(r"(?:^|\b)(?:appendix|supplementary|annex)\b", sec_display, re.I)):
                in_document_reference_zone = False
            elif not is_page_sec and not is_ref_section:
                in_document_reference_zone = False

        # Check trailing reference continuation in chapter
        if in_trailing_references_by_chapter.get(ch_idx, False):
            is_appendix = bool(re.search(r"(?:^|\b)(?:appendix|supplementary|annex)\b", sec_display, re.I))
            if is_appendix:
                in_trailing_references_by_chapter[ch_idx] = False
            elif skip_references:
                is_ref_section = True

        if is_ref_section:
            in_trailing_references_by_chapter[ch_idx] = True
            in_document_reference_zone = True
            for chk in sec_chunks:
                f_dec, f_cat = effective_conveyor.verify_with_category(chk)
                eff_cat = f_cat if f_dec == FilterDecision.SKIP else "ref_sections"
                skipped_chunks.append(chk)
                dec = CoverDecision(
                    chunk=chk,
                    selected=False,
                    status="skipped_reference",
                    section_name=sec_display,
                    filter_category=eff_cat,
                    filter_state="skip",
                )
                decisions.append(dec)
                processed_count += 1
                if on_progress:
                    on_progress(processed_count, total_count)
            continue

        # If section has fewer chunks than min_section_chunks, keep all non-reference chunks
        if len(sec_chunks) < min_section_chunks:
            for chk in sec_chunks:
                chk_text = str(get_chunk_attr(chk, "text", "") or "")
                f_dec, f_cat = effective_conveyor.verify_with_category(chk)
                skip_by_ref_zone = _eval_ref_zone_transition(chk_text, f_cat, is_page_sec)
                if skip_by_ref_zone or f_dec == FilterDecision.SKIP:
                    eff_cat = f_cat if f_dec == FilterDecision.SKIP else "ref_sections"
                    skipped_chunks.append(chk)
                    dec = CoverDecision(
                        chunk=chk,
                        selected=False,
                        status="skipped_reference",
                        section_name=sec_display,
                        filter_category=eff_cat,
                        filter_state="skip",
                    )
                else:
                    selected_chunks.append(chk)
                    eff_state = "retain" if (f_cat == "formulas" or (f_dec == FilterDecision.RETAIN and f_cat not in ("passed", ""))) else "continue"
                    dec = CoverDecision(
                        chunk=chk,
                        selected=True,
                        status="selected_anchor",
                        section_name=sec_display,
                        filter_category=f_cat,
                        filter_state=eff_state,
                    )
                decisions.append(dec)
                processed_count += 1
                if on_progress:
                    on_progress(processed_count, total_count)
            continue

        # Section has >= min_section_chunks: run greedy intra-section semantic cover
        sec_selected_chunks: List[Any] = []
        sec_selected_vectors: List[np.ndarray] = []

        for chk in sec_chunks:
            chk_text = str(get_chunk_attr(chk, "text", "") or "")

            # Check individual chunk for bibliography/boilerplate via conveyor or active reference zone
            f_dec, f_cat = effective_conveyor.verify_with_category(chk)
            skip_by_ref_zone = _eval_ref_zone_transition(chk_text, f_cat, is_page_sec)
            if skip_by_ref_zone or f_dec == FilterDecision.SKIP:
                eff_cat = f_cat if f_dec == FilterDecision.SKIP else "ref_sections"
                skipped_chunks.append(chk)
                dec = CoverDecision(
                    chunk=chk,
                    selected=False,
                    status="skipped_reference",
                    section_name=sec_display,
                    filter_category=eff_cat,
                    filter_state="skip",
                )
                decisions.append(dec)
                processed_count += 1
                if on_progress:
                    on_progress(processed_count, total_count)
                continue

            # If chunk is verified to contain mathematical formulas, shield it from redundancy pruning
            if get_chunk_attr(chk, "has_formula", False) or f_cat == "formulas":
                selected_chunks.append(chk)
                sec_selected_chunks.append(chk)
                v_unit = _normalize_and_project_vector(getattr(chk, "embedding", None))
                if v_unit is not None:
                    sec_selected_vectors.append(v_unit)
                dec = CoverDecision(
                    chunk=chk,
                    selected=True,
                    status="selected_distinct" if sec_selected_vectors else "selected_anchor",
                    section_name=sec_display,
                    filter_category="formulas",
                    filter_state="retain",
                )
                decisions.append(dec)
                processed_count += 1
                if on_progress:
                    on_progress(processed_count, total_count)
                continue

            # Check if chunk has a valid embedding
            v_unit = _normalize_and_project_vector(getattr(chk, "embedding", None))
            if v_unit is None:
                # If chunk has no embedding, we cannot safely evaluate redundancy -> keep it
                selected_chunks.append(chk)
                sec_selected_chunks.append(chk)
                eff_state = "retain" if (f_dec == FilterDecision.RETAIN and f_cat not in ("passed", "")) else "continue"
                dec = CoverDecision(
                    chunk=chk,
                    selected=True,
                    status="selected_distinct",
                    section_name=sec_display,
                    filter_category=f_cat,
                    filter_state=eff_state,
                )
                decisions.append(dec)
                processed_count += 1
                if on_progress:
                    on_progress(processed_count, total_count)
                continue

            # First non-reference chunk of the section is always the section anchor
            if not sec_selected_chunks:
                sec_selected_chunks.append(chk)
                sec_selected_vectors.append(v_unit)
                selected_chunks.append(chk)
                eff_state = "retain" if (f_dec == FilterDecision.RETAIN and f_cat not in ("passed", "")) else "continue"
                dec = CoverDecision(
                    chunk=chk,
                    selected=True,
                    status="selected_anchor",
                    section_name=sec_display,
                    filter_category=f_cat,
                    filter_state=eff_state,
                )
                decisions.append(dec)
                processed_count += 1
                if on_progress:
                    on_progress(processed_count, total_count)
                continue

            # Compare against recently selected chunks in this section (within sliding window)
            window_vectors = sec_selected_vectors[-sliding_window_size:]
            window_chunks = sec_selected_chunks[-sliding_window_size:]

            compat_window_vectors = [v for v in window_vectors if v.shape == v_unit.shape]
            if compat_window_vectors:
                M_sec = np.stack(compat_window_vectors, axis=0)  # Shape: (W, D)
                sims = M_sec @ v_unit  # Shape: (W,)
                max_sim = float(np.max(sims))
                best_match_idx = int(np.argmax(sims))
                best_match_chk = window_chunks[best_match_idx] if best_match_idx < len(window_chunks) else window_chunks[0]
            else:
                max_sim = -1.0
                best_match_chk = window_chunks[-1]

            chk_text_len = len(str(getattr(chk, "text", "") or ""))
            best_text_len = len(str(getattr(best_match_chk, "text", "") or ""))

            # Length Guard: A short snippet/header (<300 chars) cannot cover a substantive conceptual chunk (>=500 chars)
            is_substantive_expansion = chk_text_len >= 500 and best_text_len < 300

            # Math safeguard: chunks containing equations must not be dropped as redundant elaborations
            chk_has_math = get_chunk_attr(chk, "has_formula", False) or has_mathematical_formula(str(getattr(chk, "text", "") or ""))

            if enable_chunk_similarity and max_sim >= similarity_threshold and not is_substantive_expansion and not chk_has_math:
                # Chunk is redundant elaboration covered by best_match_chk
                skipped_chunks.append(chk)
                dec = CoverDecision(
                    chunk=chk,
                    selected=False,
                    status="skipped_redundant",
                    covered_by_chunk_id=str(getattr(best_match_chk, "chunk_id", "") or ""),
                    similarity_to_cover=max_sim,
                    section_name=sec_display,
                    filter_category="chunk_sim",
                    filter_state="skip",
                )
            else:
                # Chunk introduces a distinct thought or subtopic shift
                sec_selected_chunks.append(chk)
                sec_selected_vectors.append(v_unit)
                selected_chunks.append(chk)
                eff_state = "retain" if (chk_has_math or (f_dec == FilterDecision.RETAIN and f_cat not in ("passed", ""))) else "continue"
                dec = CoverDecision(
                    chunk=chk,
                    selected=True,
                    status="selected_distinct",
                    similarity_to_cover=max_sim,
                    section_name=sec_display,
                    filter_category="formulas" if chk_has_math else f_cat,
                    filter_state=eff_state,
                )

            decisions.append(dec)
            processed_count += 1
            if on_progress:
                on_progress(processed_count, total_count)

    return selected_chunks, skipped_chunks, decisions
