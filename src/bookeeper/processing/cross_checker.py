"""
Cross-Check Verification Engine for A/B Testing Chunking Strategies.

Audits contiguous text blocks between two Knowledge Graph versions (Old / Baseline vs New / Candidate)
using the adversarial QA Probe Oracle on seamless reconstructed reference passages (W_raw).
Computes:
  - QA Recall for both systems and Delta Recall (New - Old)
  - Seam Integrity Rate (Cross-sentence / boundary-dependent probe pass rate)
  - Chunk Disproportion: intersect(block_A, block_B) / union(block_A, block_B)
  - Decision Gating Checklist
"""

import json
import logging
import random
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from bookeeper.calibre.parser import BookParser
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import Concept, KnowledgeExtractor
from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings, OllamaPool
from bookeeper.processing.qa_probe import (
    QAProbeEvaluation,
    QAProbeItem,
    QASystemBlockMetrics,
    evaluate_block_qa_probes,
)

logger = logging.getLogger(__name__)

AuditVerdict = Literal["VALID_DETAIL", "TRUNCATION_ARTIFACT", "HALLUCINATION"]


class CrossCheckAuditItem(BaseModel):
    """Classification record of an idea not present in the Oracle set."""

    idea: str
    verdict: AuditVerdict
    rationale: str = ""


class SystemBlockMetrics(BaseModel):
    """Performance metrics for a single system (old/baseline or new/candidate) on a text block."""

    system: Literal["old", "new"]
    raw_ideas_count: int = 0
    deduped_ideas_count: int = 0
    retained_ideas_count: int = 0
    dropped_ideas_count: int = 0
    fail_ratio: float = 0.0  # Dropped / Total_Block_IDEA_CNT
    oracle_recall: float = 0.0
    grounded_precision: float = 1.0
    truncation_rate: float = 0.0
    retained_oracle_ideas: List[str] = Field(default_factory=list)
    dropped_oracle_ideas: List[str] = Field(default_factory=list)
    audited_items: List[CrossCheckAuditItem] = Field(default_factory=list)

    # QA Probe Metrics
    qa_recall: float = 0.0
    seam_integrity_rate: float = 0.0
    passed_probes_count: int = 0
    total_probes_count: int = 0
    evaluations: List[QAProbeEvaluation] = Field(default_factory=list)


class CrossCheckBlockResult(BaseModel):
    """Audit evaluation result for a single sampled text window / block."""

    sample_index: int
    book_id: int
    book_title: str
    section_title: str
    old_chunk_ids: List[str]
    new_chunk_ids: List[str]
    passage_length_chars: int

    # Chunk lengths & Disproportion metric: intersect(block_A, block_B) / union(block_A, block_B)
    block_a_total_len: int
    block_b_total_len: int
    union_len: int
    chunk_disproportion: float

    oracle_ideas_count: int = 0
    old_system: SystemBlockMetrics
    new_system: SystemBlockMetrics
    delta_recall: float
    delta_fail_ratio: float = 0.0  # fail_ratio_B - fail_ratio_A
    duration_seconds: float = 0.0

    block_a_text: str = ""
    block_b_text: str = ""
    candidate_ideas_old: List[Union[Concept, Dict[str, Any], str]] = Field(default_factory=list)
    candidate_ideas_new: List[Union[Concept, Dict[str, Any], str]] = Field(default_factory=list)
    oracle_ideas: List[str] = Field(default_factory=list)

    # QA Probe Metrics
    probes: List[QAProbeItem] = Field(default_factory=list)
    delta_qa_recall: float = 0.0
    delta_seam_integrity: float = 0.0


class CrossCheckSummary(BaseModel):
    """Aggregated benchmark statistics and decision gating across all audited blocks."""

    total_samples: int
    mean_old_recall: float
    mean_new_recall: float
    mean_delta_recall: float
    mean_old_fail_ratio: float = 0.0
    mean_new_fail_ratio: float = 0.0
    mean_delta_fail_ratio: float = 0.0
    mean_old_precision: float = 1.0
    mean_new_precision: float = 1.0
    mean_old_truncation_rate: float = 0.0
    mean_new_truncation_rate: float = 0.0
    mean_chunk_disproportion: float
    mean_old_seam_integrity: float = 0.0
    mean_new_seam_integrity: float = 0.0
    mean_delta_seam_integrity: float = 0.0
    decision_pass: bool
    pass_reason: str = ""
    total_duration_seconds: float = 0.0


class CrossCheckReport(BaseModel):
    """Complete serialization report for Cross-Check verification mode."""

    mode: str = "cross-check"
    model_name: str
    baseline_branch: str = "Branch A"
    candidate_branch: str = "Branch B"
    summary: CrossCheckSummary
    samples: List[CrossCheckBlockResult]


# ==============================================================================
# 1. Content Stitching & Overlap Reconstruction
# ==============================================================================


def stitch_chunks_dedup_text(chunk_texts: List[str]) -> str:
    """
    Stitch consecutive chunks into a seamless contiguous reference passage W_raw,
    collapsing 1-2 buffer sentences or suffix-prefix overlap between adjacent chunks.
    """
    if not chunk_texts:
        return ""
    if len(chunk_texts) == 1:
        return chunk_texts[0].strip()

    result = chunk_texts[0].strip()
    for nxt in chunk_texts[1:]:
        nxt_clean = nxt.strip()
        if not nxt_clean:
            continue

        # Find longest common suffix of result that is prefix of nxt_clean
        # Search down from min(len(result), len(nxt_clean)) to 15 chars
        max_search = min(len(result), len(nxt_clean), 600)
        found_overlap = 0

        # Fast sentence-boundary scan
        for overlap_len in range(max_search, 14, -1):
            if result.endswith(nxt_clean[:overlap_len]):
                found_overlap = overlap_len
                break

        if found_overlap > 0:
            result = result + nxt_clean[found_overlap:]
        else:
            # Fallback sentence overlap check
            s_res = [s.strip() for s in re.split(r"(?<=[.!?])\s+", result) if s.strip()]
            s_nxt = [s.strip() for s in re.split(r"(?<=[.!?])\s+", nxt_clean) if s.strip()]
            if s_res and s_nxt and s_res[-1] == s_nxt[0]:
                overlap_text = s_nxt[0]
                idx = nxt_clean.find(overlap_text)
                if idx != -1:
                    result = result + nxt_clean[idx + len(overlap_text):]
                else:
                    result = result + " " + nxt_clean
            else:
                result = result + " " + nxt_clean

    return result.strip()


def find_overlapping_chunks(
    graph_b: ConceptGraphStore,
    book_id: int,
    w_raw: str,
    min_overlap: float = 0.80,
) -> List[Dict[str, Any]]:
    """
    Locate contiguous candidate chunks in Graph B matching W_raw.
    Selects BLOCK_B with min(dedup(CHUNK_B[n..n+k]), dedup(CHUNK_B[n..n+k+1]))
    that minimizes length deviation / disproportion gap relative to W_raw.
    """
    chunks = [
        dict(attrs)
        for _, attrs in graph_b.graph.nodes(data=True)
        if attrs.get("type") == "Chunk" and attrs.get("book_id") == book_id
    ]
    if not chunks or not w_raw.strip():
        return []

    chunks.sort(key=lambda x: (int(x.get("chapter_idx") or 0), int(x.get("chunk_idx") or 0)))

    w_raw_norm = " ".join(w_raw.lower().split())
    prefix = w_raw_norm[:350]

    # Find starting chunk in B with strongest match at the beginning of prefix
    start_candidates = []
    for idx, c in enumerate(chunks):
        t_norm = " ".join(c.get("text", "").lower().split())
        if not t_norm:
            continue
        sm = SequenceMatcher(None, prefix, t_norm, autojunk=False)
        m = sm.find_longest_match(0, len(prefix), 0, len(t_norm))
        if m.size >= 25:
            # Score prioritizing matches that start at the beginning of w_raw (m.a near 0)
            score = m.size - (m.a * 3)
            start_candidates.append((score, m.a, m.size, idx))

    if start_candidates:
        start_candidates.sort(key=lambda x: x[0], reverse=True)
        best_start = start_candidates[0][3]
    else:
        best_start = -1

    # Fallback to general sequence / substring matching if prefix match was weak
    if best_start == -1:
        for idx, c in enumerate(chunks):
            t_norm = " ".join(c.get("text", "").lower().split())
            if t_norm and (t_norm[:60] in w_raw_norm or w_raw_norm[:60] in t_norm):
                best_start = idx
                break

    if best_start == -1:
        # Fallback to independent overlap filtering
        fallback_candidates = []
        for c in chunks:
            t_norm = " ".join(c.get("text", "").lower().split())
            if not t_norm:
                continue
            sm = SequenceMatcher(None, w_raw_norm, t_norm, autojunk=False)
            match = sm.find_longest_match(0, len(w_raw_norm), 0, len(t_norm))
            ratio = match.size / max(1, len(t_norm))
            if ratio >= min_overlap:
                fallback_candidates.append(c)
        return fallback_candidates

    # Contiguous slice expansion: accumulate chunks that overlap w_raw
    max_k = min(50, len(chunks) - best_start)
    candidates = []
    for k in range(1, max_k + 1):
        c_k = chunks[best_start + k - 1]
        t_norm_k = " ".join(c_k.get("text", "").lower().split())
        # Check overlap of chunk k with w_raw to stop expanding when passage in B ends
        if k > 1:
            sm_k = SequenceMatcher(None, w_raw_norm, t_norm_k, autojunk=False)
            m_k = sm_k.find_longest_match(0, len(w_raw_norm), 0, len(t_norm_k))
            if m_k.size < 20:
                break
        slice_k = chunks[best_start : best_start + k]
        w_k = stitch_chunks_dedup_text([c.get("text", "") for c in slice_k])
        candidates.append((slice_k, w_k, len(w_k)))
        if len(w_k) >= len(w_raw):
            break

    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates[0][0]

    # Boundary comparison: min(dedup(CHUNK_B[n..n+k]), dedup(CHUNK_B[n..n+k+1]))
    cand_prev = candidates[-2]
    cand_curr = candidates[-1]

    diff_prev = abs(cand_prev[2] - len(w_raw))
    diff_curr = abs(cand_curr[2] - len(w_raw))

    chosen = cand_prev[0] if diff_prev <= diff_curr else cand_curr[0]
    return chosen


def compute_chunk_disproportion(
    chunks_a_texts: List[str],
    chunks_b_texts: List[str],
    w_raw: str,
) -> Tuple[int, int, int, float]:
    """
    Compute chunk lengths and chunk disproportion metric:
      intersect(block_A, block_B) / union(block_A, block_B)
    Using exact SequenceMatcher matching blocks across deduplicated passages.
    Returns:
      (sum_len_a, sum_len_b, union_len, chunk_disproportion)
    """
    sum_len_a = sum(len(t) for t in chunks_a_texts)
    sum_len_b = sum(len(t) for t in chunks_b_texts)

    if not chunks_a_texts and not chunks_b_texts:
        return 0, 0, 0, 0.0
    if not chunks_a_texts:
        return 0, sum_len_b, sum_len_b, 0.0
    if not chunks_b_texts:
        return sum_len_a, 0, sum_len_a, 0.0
    if chunks_a_texts == chunks_b_texts:
        return sum_len_a, sum_len_b, sum_len_a, 1.0

    w_b = stitch_chunks_dedup_text(chunks_b_texts)
    w_a_norm = " ".join(w_raw.lower().split())
    w_b_norm = " ".join(w_b.lower().split())

    sm = SequenceMatcher(None, w_a_norm, w_b_norm, autojunk=False)
    match_len = sum(b.size for b in sm.get_matching_blocks() if b.size > 0)
    union_norm = max(1, len(w_a_norm) + len(w_b_norm) - match_len)
    disproportion = round(match_len / union_norm, 4)

    # Scale union length back to raw character space
    scale = len(w_raw) / max(1, len(w_a_norm))
    union_len = max(1, round(union_norm * scale))

    return sum_len_a, sum_len_b, union_len, disproportion


# ==============================================================================
# 2. Tiered Deduplication & Concept Helpers
# ==============================================================================


def get_ideas_for_chunks(graph_store: ConceptGraphStore, chunk_node_ids: List[str]) -> List[Concept]:
    """Retrieve all Concept objects with complete metadata associated with a list of Chunk node IDs."""
    ideas: List[Concept] = []
    seen_names: Set[str] = set()

    for cid in chunk_node_ids:
        nid = cid if str(cid).startswith("chunk:") else f"chunk:{cid}"
        if not graph_store.graph.has_node(nid):
            continue

        # Look in predecessors and successors for Concept nodes
        for neighbor in list(graph_store.graph.neighbors(nid)) + list(graph_store.graph.predecessors(nid)):
            cattrs = graph_store.graph.nodes[neighbor]
            if cattrs.get("type") in ("Concept", "Idea"):
                cname = cattrs.get("name") or neighbor.replace("concept:", "")
                if cname not in seen_names:
                    seen_names.add(cname)
                    # Also inspect edge attributes between Chunk and Concept for supporting quotes and explanations
                    edge_fwd = graph_store.graph.get_edge_data(nid, neighbor) or {}
                    edge_rev = graph_store.graph.get_edge_data(neighbor, nid) or {}
                    edge_attrs = {**edge_rev, **edge_fwd}

                    brief = (
                        cattrs.get("brief_description")
                        or edge_attrs.get("brief_description")
                        or cattrs.get("summary")
                        or edge_attrs.get("summary")
                        or ""
                    )
                    detailed = (
                        cattrs.get("detailed_explanation")
                        or edge_attrs.get("detailed_explanation")
                        or ""
                    )
                    quote = (
                        cattrs.get("supporting_quote")
                        or edge_attrs.get("quote")
                        or edge_attrs.get("supporting_quote")
                        or None
                    )
                    category = cattrs.get("category") or edge_attrs.get("category") or "Idea"
                    weight = int(cattrs.get("weight") or edge_attrs.get("weight") or 5)

                    ideas.append(
                        Concept(
                            name=cname,
                            category=category,
                            brief_description=brief,
                            detailed_explanation=detailed,
                            supporting_quote=quote,
                            weight=weight,
                        )
                    )
    return ideas


def tiered_deduplicate_ideas(
    concepts: List[Concept],
    deduplicator: EntityDeduplicator,
) -> List[Concept]:
    """
    Deduplicate a list of concepts using tiered matching:
      - normalized string match / exact -> auto-merge
      - vector similarity >= 0.95 -> auto-merge
      - vector similarity in [0.80, 0.95) -> LLM judge
    Retains the longer, more informative concept definition on merge.
    """
    if not concepts:
        return []

    unique_concepts: List[Concept] = []

    for c in concepts:
        merged = False
        for idx, u in enumerate(unique_concepts):
            is_same, score, method, canon, reason = deduplicator.is_same_concept(u, c)
            if is_same:
                # Merge into existing: keep longer definition
                if len(c.name) + len(c.brief_description) > len(u.name) + len(u.brief_description):
                    unique_concepts[idx] = c
                merged = True
                break
        if not merged:
            unique_concepts.append(c)

    return unique_concepts


# ==============================================================================
# 3. Oracle Extraction & Boundary Seam Auditor
# ==============================================================================


def extract_oracle_ideas_from_passage(
    passage: str,
    pool: OllamaPool,
    model_name: str,
    extractor: Optional[KnowledgeExtractor] = None,
) -> List[Concept]:
    """
    Extract core standalone domain ideas and concepts directly from the unbroken
    passage W_raw using the Oracle model matching the production concept ontology.
    """
    prompt = (
        "You are an expert knowledge extractor, domain ontologist, and conceptual analyst.\n"
        "Your objective is to extract all canonical domain concepts, principles, strategic mechanisms, "
        "and standalone ideas discussed in the provided unbroken passage.\n\n"
        "EXTRACTION GUIDELINES:\n"
        "1. Extract ALL key technical, organizational, or domain concepts present (typically 3 to 10 concepts).\n"
        "2. 'name': Concise canonical title (2-4 words, capitalized noun phrase, e.g. 'eBook Conversion', 'Format Lock-In', 'EPUB Standard').\n"
        "3. 'brief_description': 1-2 sentence self-contained definition of the concept in context.\n"
        "4. 'category': Domain category (e.g. Technology, Architecture, Mechanism, Standard, Strategy).\n\n"
        f"Passage:\n\"\"\"\n{passage[:7000]}\n\"\"\"\n\n"
        "Return strictly valid JSON with this format:\n"
        "{\n"
        '  "concepts": [\n'
        '    {"name": "Canonical Title", "brief_description": "1-2 sentence definition", "category": "General"}\n'
        "  ]\n"
        "}\n"
    )

    try:
        raw_text, _ = pool.generate(model_name, prompt, temperature=0.0)
        clean_text = raw_text.strip()
        if "<think>" in clean_text and "</think>" in clean_text:
            clean_text = clean_text.split("</think>")[-1].strip()
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()

        data = json.loads(clean_text)
        concepts: List[Concept] = []
        raw_list = data.get("concepts", []) or data.get("ideas", [])
        for item in raw_list:
            if isinstance(item, dict) and item.get("name"):
                concepts.append(
                    Concept(
                        name=item["name"].strip(),
                        brief_description=item.get("brief_description", item.get("reasoning", "")).strip(),
                        category=item.get("category", "Concept"),
                    )
                )
            elif isinstance(item, str) and item.strip():
                concepts.append(Concept(name=item.strip(), brief_description="", category="Concept"))
        if concepts:
            return concepts
    except Exception as exc:
        logger.warning(f"Direct Oracle idea extraction failed: {exc}")

    # Fallback to extractor.extract_ideas if direct prompt failed
    if extractor is not None:
        try:
            return extractor.extract_ideas(passage, model=model_name)
        except Exception as e:
            logger.warning(f"KnowledgeExtractor.extract_ideas failed on passage: {e}")

    return []


def audit_unmatched_idea(
    idea_name: str,
    w_raw: str,
    pool: OllamaPool,
    model_name: str,
    extractor: Optional[KnowledgeExtractor] = None,
) -> CrossCheckAuditItem:
    """
    Classify an idea not matched to Oracle into:
      - VALID_DETAIL
      - TRUNCATION_ARTIFACT
      - HALLUCINATION
    """
    prompt = (
        "Context:\n"
        f"\"\"\"\n{w_raw[:4000]}\n\"\"\"\n\n"
        f"Candidate Statement: \"{idea_name}\"\n\n"
        "Classify this statement into exactly ONE category based on the context:\n"
        "- VALID_DETAIL: Factually accurate and supported, but represents a localized detail omitted from a high-level summary.\n"
        "- TRUNCATION_ARTIFACT: Misleading, incomplete, or false because a context boundary cut off an essential caveat, condition, or negation.\n"
        "- HALLUCINATION: Completely unsupported or fabricated.\n\n"
        "Return strictly valid JSON:\n"
        '{"verdict": "VALID_DETAIL" | "TRUNCATION_ARTIFACT" | "HALLUCINATION", "rationale": "one sentence explanation"}\n'
    )

    try:
        raw_text, _ = pool.generate(model_name, prompt, temperature=0.0)
        clean = raw_text.strip()
        if "<think>" in clean and "</think>" in clean:
            clean = clean.split("</think>")[-1].strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean = "\n".join(lines).strip()

        data = json.loads(clean)
        verdict_str = str(data.get("verdict", "VALID_DETAIL")).upper()
        if verdict_str not in ("VALID_DETAIL", "TRUNCATION_ARTIFACT", "HALLUCINATION"):
            verdict_str = "VALID_DETAIL"
        rationale_str = str(data.get("rationale", ""))
        return CrossCheckAuditItem(idea=idea_name, verdict=verdict_str, rationale=rationale_str)  # type: ignore
    except Exception as e:
        logger.warning(f"Audit LLM classification failed for '{idea_name}': {e}")
        return CrossCheckAuditItem(idea=idea_name, verdict="VALID_DETAIL", rationale="Fallback classification.")


def evaluate_system_against_oracle(
    candidate_ideas: List[Any],
    oracle_ideas: List[Any],
    w_raw: str,
    system_label: Literal["old", "new"],
    deduplicator: EntityDeduplicator,
    pool: OllamaPool,
    model_name: str,
    extractor: Optional[KnowledgeExtractor] = None,
) -> SystemBlockMetrics:
    """
    Compare candidate ideas against Oracle ground truth to compute Recall,
    Grounded Precision, and Truncation Artifact Rate.
    """
    if not oracle_ideas and not candidate_ideas:
        return SystemBlockMetrics(
            system=system_label,
            raw_ideas_count=0,
            deduped_ideas_count=0,
            oracle_recall=1.0,
            grounded_precision=1.0,
            truncation_rate=0.0,
        )

    matched_oracle = set()
    matched_candidate = set()

    for o_idx, o_idea in enumerate(oracle_ideas):
        for c_idx, c_idea in enumerate(candidate_ideas):
            if c_idx in matched_candidate:
                continue
            is_same, score, method, canon, reason = deduplicator.is_same_concept(o_idea, c_idea)
            if is_same:
                matched_oracle.add(o_idx)
                matched_candidate.add(c_idx)
                break

    retained = [getattr(oracle_ideas[i], "name", str(oracle_ideas[i])) for i in matched_oracle]
    dropped = [getattr(oracle_ideas[i], "name", str(oracle_ideas[i])) for i in range(len(oracle_ideas)) if i not in matched_oracle]

    recall = round(len(retained) / max(1, len(oracle_ideas)), 4) if oracle_ideas else 1.0

    # Audit candidate ideas not matched to Oracle (delta ideas)
    unmatched_candidates = [candidate_ideas[i] for i in range(len(candidate_ideas)) if i not in matched_candidate]
    audited_items: List[CrossCheckAuditItem] = []

    for u_cand in unmatched_candidates:
        u_name = getattr(u_cand, "name", str(u_cand))
        item = audit_unmatched_idea(u_name, w_raw, pool, model_name, extractor)
        audited_items.append(item)

    valid_detail_count = sum(1 for it in audited_items if it.verdict == "VALID_DETAIL")
    truncation_count = sum(1 for it in audited_items if it.verdict == "TRUNCATION_ARTIFACT")

    tot_cand = len(candidate_ideas)
    dropped_count = len(dropped)
    retained_count = len(retained)
    fail_ratio = round(dropped_count / tot_cand, 4) if tot_cand > 0 else (1.0 if dropped_count > 0 else 0.0)

    if tot_cand > 0:
        grounded_precision = round((len(matched_candidate) + valid_detail_count) / tot_cand, 4)
        truncation_rate = round(truncation_count / tot_cand, 4)
    else:
        grounded_precision = 1.0
        truncation_rate = 0.0

    return SystemBlockMetrics(
        system=system_label,
        raw_ideas_count=tot_cand,
        deduped_ideas_count=tot_cand,
        retained_ideas_count=retained_count,
        dropped_ideas_count=dropped_count,
        fail_ratio=fail_ratio,
        oracle_recall=recall,
        grounded_precision=grounded_precision,
        truncation_rate=truncation_rate,
        retained_oracle_ideas=retained,
        dropped_oracle_ideas=dropped,
        audited_items=audited_items,
    )


def print_block_comparison(
    sample_index: int,
    book_title: str,
    section_title: str,
    old_chunk_ids: List[str],
    new_chunk_ids: List[str],
    w_raw: str,
    w_b: str,
    disproportion: float,
    union_len: int,
    ideas_old: List[Any],
    ideas_new: List[Any],
    probes: Optional[List[QAProbeItem]] = None,
    eval_old: Optional[SystemBlockMetrics] = None,
    eval_new: Optional[SystemBlockMetrics] = None,
    console: Optional[Any] = None,
    oracle_ideas: Optional[List[Any]] = None,
) -> None:
    """Print formatted comparative analysis of a single block with QA Probes."""
    def _out(msg: str = "") -> None:
        if console is not None and hasattr(console, "print"):
            console.print(msg)
        else:
            print(msg)

    if probes is None:
        if oracle_ideas:
            probes = [
                QAProbeItem(
                    question_id=idx,
                    question=f"Concept: {getattr(o, 'name', str(o))}",
                    gold_answer=getattr(o, "brief_description", "") or str(o),
                    is_cross_sentence=(idx % 2 == 1),
                )
                for idx, o in enumerate(oracle_ideas, 1)
            ]
        else:
            probes = []

    if eval_old is None:
        eval_old = SystemBlockMetrics(system="old")
    if eval_new is None:
        eval_new = SystemBlockMetrics(system="new")

    sep = "=" * 80
    sub_sep = "-" * 80
    disp_status = "Boundary Discrepancy" if disproportion < 0.999 else "Aligned"
    _out(f"\n{sep}")
    _out(f"🔍 Block #{sample_index} Alignment Comparison & QA Probe Audit [Disproportion = {disproportion:.4f} ({disp_status})]")
    _out(f"📖 Book: {book_title} | Section: {section_title}")
    _out(sub_sep)
    _out(f"📦 BLOCK A ({len(old_chunk_ids)} chunks: {', '.join(old_chunk_ids)})")
    _out(f"   Total Chars: {len(w_raw)} | Text Preview:")
    excerpt_a = w_raw[:350].replace('\n', ' ')
    _out(f"   \"{excerpt_a}...\"")

    _out(f"\n📦 BLOCK B ({len(new_chunk_ids)} chunks: {', '.join(new_chunk_ids)})")
    _out(f"   Total Chars: {len(w_b)} | Text Preview:")
    excerpt_b = w_b[:350].replace('\n', ' ')
    _out(f"   \"{excerpt_b}...\"")

    _out(f"\n📏 ALIGNMENT: Disproportion = {disproportion:.4f} | Union = {union_len} chars")

    _out(f"\n💡 EXTRACTED CLAIMS CONTEXT:")
    _out(f"   • BLOCK A: {len(ideas_old)} unique ideas")
    for idx, c in enumerate(ideas_old[:5], 1):
        name = getattr(c, "name", str(c))
        desc = getattr(c, "brief_description", "") or getattr(c, "summary", "")
        desc_str = f" - {desc[:60]}..." if desc else ""
        _out(f"     {idx}. {name}{desc_str}")
    if len(ideas_old) > 5:
        _out(f"     ... and {len(ideas_old) - 5} more ideas")

    _out(f"   • BLOCK B: {len(ideas_new)} unique ideas")
    for idx, c in enumerate(ideas_new[:5], 1):
        name = getattr(c, "name", str(c))
        desc = getattr(c, "brief_description", "") or getattr(c, "summary", "")
        desc_str = f" - {desc[:60]}..." if desc else ""
        _out(f"     {idx}. {name}{desc_str}")
    if len(ideas_new) > 5:
        _out(f"     ... and {len(ideas_new) - 5} more ideas")

    _out(f"\n🎯 QA PROBES & SEAM INTEGRITY AUDIT ({len(probes)} Probes):")
    eval_old_map = {e.question_id: e for e in eval_old.evaluations}
    eval_new_map = {e.question_id: e for e in eval_new.evaluations}

    for p in probes:
        probe_tag = "[Cross-Sentence]" if p.is_cross_sentence else "[Single-Fact]"
        _out(f"   {p.question_id}. {probe_tag} {p.question}")
        _out(f"      • Gold: {p.gold_answer}")

        e_a = eval_old_map.get(p.question_id)
        if e_a:
            pred_a = e_a.predicted_answer[:70].replace("\n", " ")
            _out(f"      • Baseline (A): \"{pred_a}\" -> [{e_a.verdict}]")

        e_b = eval_new_map.get(p.question_id)
        if e_b:
            pred_b = e_b.predicted_answer[:70].replace("\n", " ")
            status_b = e_b.verdict
            if e_b.verdict == "FAIL" and p.is_cross_sentence and (e_a and e_a.verdict == "PASS"):
                status_b = "FAIL - Seam Lost"
            _out(f"      • Candidate (B): \"{pred_b}\" -> [{status_b}]")
        _out("")

    _out(f"📊 BLOCK QA METRICS:")
    _out(
        f"   • Baseline QA Recall:  {eval_old.qa_recall * 100:.1f}% ({eval_old.passed_probes_count}/{eval_old.total_probes_count} passed) | "
        f"Seam Integrity: {eval_old.seam_integrity_rate * 100:.1f}%"
    )
    _out(
        f"   • Candidate QA Recall: {eval_new.qa_recall * 100:.1f}% ({eval_new.passed_probes_count}/{eval_new.total_probes_count} passed) | "
        f"Seam Integrity: {eval_new.seam_integrity_rate * 100:.1f}%"
    )
    delta_pts = (eval_new.qa_recall - eval_old.qa_recall) * 100
    _out(f"   • Recall Delta:        {delta_pts:+.1f}% pts")
    _out(f"{sep}\n")


# ==============================================================================
# 4. Main Cross-Check Orchestrator
# ==============================================================================


def run_cross_check(
    store_a: ConceptGraphStore,
    store_b: ConceptGraphStore,
    num_blocks: int = 20,
    pool: Optional[OllamaPool] = None,
    model_name: str = "llama3.1:8b",
    embedding_model: str = "nomic-embed-text",
    branch_a_name: str = "Baseline",
    branch_b_name: str = "Candidate",
    seed: Optional[int] = None,
    print_chunks: int = 0,
    console: Optional[Any] = None,
    progress_callback: Optional[Callable[[int, int, str, Optional[CrossCheckBlockResult]], None]] = None,
) -> CrossCheckReport:
    """
    Execute full Cross-Check verification pipeline comparing Store A and Store B
    using the adversarial QA Probe Oracle directly on seamless passages.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if pool is None:
        pool = OllamaPool.from_urls(["http://localhost:11434"])

    embeddings: Optional[FailoverOllamaEmbeddings] = None
    try:
        embeddings = FailoverOllamaEmbeddings(
            model=embedding_model,
            pool=pool,
            request_timeout=60,
        )
    except Exception as exc:
        logger.warning(f"Could not initialize FailoverOllamaEmbeddings: {exc}")

    extractor = KnowledgeExtractor(pool=pool, model=model_name)
    deduplicator = EntityDeduplicator(
        embeddings=embeddings,
        extractor=extractor,
        similarity_threshold=0.80,
        high_similarity_threshold=0.95,
    )

    # 1. Discover common books between Graph A and Graph B
    books_a = {d.get("book_id"): d for n, d in store_a.graph.nodes(data=True) if d.get("type") == "Book"}
    books_b = {d.get("book_id"): d for n, d in store_b.graph.nodes(data=True) if d.get("type") == "Book"}
    common_book_ids = sorted(list(set(books_a.keys()) & set(books_b.keys())))

    if not common_book_ids:
        # Fallback to all books in A
        common_book_ids = sorted(list(books_a.keys()))

    # Collect chunks by book in Graph A
    chunks_by_book_a: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
    for nid, attrs in store_a.graph.nodes(data=True):
        if attrs.get("type") == "Chunk":
            b_id = attrs.get("book_id", 0)
            chunks_by_book_a.setdefault(b_id, []).append((nid, dict(attrs)))

    for b_id in chunks_by_book_a:
        chunks_by_book_a[b_id].sort(key=lambda x: (int(x[1].get("chapter_idx") or 0), int(x[1].get("chunk_idx") or 0)))

    # Filter books with at least 3 chunks
    valid_books = [bid for bid in common_book_ids if len(chunks_by_book_a.get(bid, [])) >= 3]
    if not valid_books:
        valid_books = [bid for bid in chunks_by_book_a if len(chunks_by_book_a[bid]) >= 1]

    samples: List[CrossCheckBlockResult] = []
    t_global_start = time.time()
    max_attempts = num_blocks * 5
    attempts = 0
    printed_chunks_count = 0

    while len(samples) < num_blocks and attempts < max_attempts:
        attempts += 1
        t0 = time.time()
        if not valid_books:
            break

        # 1. Select random book and 3 consecutive chunks from Graph A
        target_bid = random.choice(valid_books)
        c_list = chunks_by_book_a[target_bid]
        max_start = max(0, len(c_list) - 3)
        start_idx = random.randint(0, max_start)
        old_slice = c_list[start_idx : start_idx + 3]

        old_chunk_ids = [c[0] for c in old_slice]
        old_chunk_texts = [c[1].get("text", "") for c in old_slice]
        b_title = old_slice[0][1].get("book_title", f"Book #{target_bid}")
        sec_title = old_slice[0][1].get("section_title", "Section")

        # 2. Reconstruct seamless raw passage W_raw
        w_raw = stitch_chunks_dedup_text(old_chunk_texts)
        if not w_raw or len(w_raw) < 100:
            continue

        # 3. Locate matching new chunks in Graph B with optimal slice boundary
        new_chunk_dicts = find_overlapping_chunks(store_b, target_bid, w_raw, min_overlap=0.80)
        if not new_chunk_dicts:
            continue

        new_chunk_ids = [d.get("chunk_id") or d.get("id", "") for d in new_chunk_dicts]
        new_chunk_texts = [d.get("text", "") for d in new_chunk_dicts]
        w_b = stitch_chunks_dedup_text(new_chunk_texts)

        # 4. Compute chunk disproportion: intersect(block_A, block_B) / union(block_A, block_B)
        sum_len_a, sum_len_b, union_len, disproportion = compute_chunk_disproportion(
            old_chunk_texts, new_chunk_texts, w_raw
        )

        # 5. Union & Tiered Deduplication of ideas
        raw_ideas_old = get_ideas_for_chunks(store_a, old_chunk_ids)
        raw_ideas_new = get_ideas_for_chunks(store_b, new_chunk_ids)

        dedup_ideas_old = tiered_deduplicate_ideas(raw_ideas_old, deduplicator)
        dedup_ideas_new = tiered_deduplicate_ideas(raw_ideas_new, deduplicator)

        # 6. QA Probe Oracle Evaluation on contiguous passage
        probes, qa_m_old, qa_m_new = evaluate_block_qa_probes(
            w_raw=w_raw,
            ideas_old=dedup_ideas_old,
            ideas_new=dedup_ideas_new,
            pool=pool,
            model_name=model_name,
        )

        eval_old = SystemBlockMetrics(
            system="old",
            raw_ideas_count=len(raw_ideas_old),
            deduped_ideas_count=len(dedup_ideas_old),
            retained_ideas_count=qa_m_old.passed_probes,
            dropped_ideas_count=qa_m_old.failed_probes,
            fail_ratio=round(qa_m_old.failed_probes / max(1, qa_m_old.total_probes), 4),
            oracle_recall=qa_m_old.qa_recall,
            grounded_precision=1.0,
            truncation_rate=0.0,
            qa_recall=qa_m_old.qa_recall,
            seam_integrity_rate=qa_m_old.seam_integrity_rate,
            passed_probes_count=qa_m_old.passed_probes,
            total_probes_count=qa_m_old.total_probes,
            evaluations=qa_m_old.evaluations,
        )
        eval_new = SystemBlockMetrics(
            system="new",
            raw_ideas_count=len(raw_ideas_new),
            deduped_ideas_count=len(dedup_ideas_new),
            retained_ideas_count=qa_m_new.passed_probes,
            dropped_ideas_count=qa_m_new.failed_probes,
            fail_ratio=round(qa_m_new.failed_probes / max(1, qa_m_new.total_probes), 4),
            oracle_recall=qa_m_new.qa_recall,
            grounded_precision=1.0,
            truncation_rate=0.0,
            qa_recall=qa_m_new.qa_recall,
            seam_integrity_rate=qa_m_new.seam_integrity_rate,
            passed_probes_count=qa_m_new.passed_probes,
            total_probes_count=qa_m_new.total_probes,
            evaluations=qa_m_new.evaluations,
        )

        delta_rec = round(eval_new.qa_recall - eval_old.qa_recall, 4)
        delta_fail = round(eval_new.fail_ratio - eval_old.fail_ratio, 4)
        delta_seam = round(eval_new.seam_integrity_rate - eval_old.seam_integrity_rate, 4)
        dur = round(time.time() - t0, 2)
        curr_idx = len(samples) + 1

        block_result = CrossCheckBlockResult(
            sample_index=curr_idx,
            book_id=target_bid,
            book_title=b_title,
            section_title=sec_title,
            old_chunk_ids=old_chunk_ids,
            new_chunk_ids=new_chunk_ids,
            passage_length_chars=len(w_raw),
            block_a_total_len=sum_len_a,
            block_b_total_len=sum_len_b,
            union_len=union_len,
            chunk_disproportion=disproportion,
            oracle_ideas_count=len(probes),
            old_system=eval_old,
            new_system=eval_new,
            delta_recall=delta_rec,
            delta_fail_ratio=delta_fail,
            duration_seconds=dur,
            block_a_text=w_raw[:2000],
            block_b_text=w_b[:2000],
            candidate_ideas_old=dedup_ideas_old,
            candidate_ideas_new=dedup_ideas_new,
            oracle_ideas=[p.question for p in probes],
            probes=probes,
            delta_qa_recall=delta_rec,
            delta_seam_integrity=delta_seam,
        )
        samples.append(block_result)

        # Determine whether to print detailed block comparison:
        # 1. If print_chunks > 0, print up to print_chunks analyzed blocks.
        # 2. If print_chunks == 0, auto-print up to 3 blocks if disproportion < 1.0.
        should_print = False
        if print_chunks > 0 and printed_chunks_count < print_chunks:
            should_print = True
        elif print_chunks == 0 and disproportion < 0.999 and printed_chunks_count < 3:
            should_print = True

        if should_print:
            printed_chunks_count += 1
            print_block_comparison(
                sample_index=curr_idx,
                book_title=b_title,
                section_title=sec_title,
                old_chunk_ids=old_chunk_ids,
                new_chunk_ids=new_chunk_ids,
                w_raw=w_raw,
                w_b=w_b,
                disproportion=disproportion,
                union_len=union_len,
                ideas_old=dedup_ideas_old,
                ideas_new=dedup_ideas_new,
                probes=probes,
                eval_old=eval_old,
                eval_new=eval_new,
                console=console,
            )

        if progress_callback:
            progress_callback(curr_idx, num_blocks, f"Block #{curr_idx} ({b_title[:20]})", block_result)

    # 8. Compute Summary & Decision Checklist
    tot = len(samples)
    if tot > 0:
        mean_old_rec = round(float(np.mean([s.old_system.qa_recall for s in samples])), 4)
        mean_new_rec = round(float(np.mean([s.new_system.qa_recall for s in samples])), 4)
        mean_delta_rec = round(float(np.mean([s.delta_qa_recall for s in samples])), 4)

        mean_old_seam = round(float(np.mean([s.old_system.seam_integrity_rate for s in samples])), 4)
        mean_new_seam = round(float(np.mean([s.new_system.seam_integrity_rate for s in samples])), 4)
        mean_delta_seam = round(float(np.mean([s.delta_seam_integrity for s in samples])), 4)

        # Micro-aggregated fail ratios: sum(failed) / sum(total_block_probes)
        total_old_dropped = sum(s.old_system.dropped_ideas_count for s in samples)
        total_old_probes = sum(s.old_system.total_probes_count for s in samples)
        agg_old_fail = round(total_old_dropped / total_old_probes, 4) if total_old_probes > 0 else 0.0

        total_new_dropped = sum(s.new_system.dropped_ideas_count for s in samples)
        total_new_probes = sum(s.new_system.total_probes_count for s in samples)
        agg_new_fail = round(total_new_dropped / total_new_probes, 4) if total_new_probes > 0 else 0.0

        agg_delta_fail = round(agg_new_fail - agg_old_fail, 4)
        mean_old_prec = round(float(np.mean([s.old_system.grounded_precision for s in samples])), 4)
        mean_new_prec = round(float(np.mean([s.new_system.grounded_precision for s in samples])), 4)
        mean_old_trunc = round(float(np.mean([s.old_system.truncation_rate for s in samples])), 4)
        mean_new_trunc = round(float(np.mean([s.new_system.truncation_rate for s in samples])), 4)
        mean_disprop = round(float(np.mean([s.chunk_disproportion for s in samples])), 4)
    else:
        mean_old_rec = mean_new_rec = mean_delta_rec = mean_old_prec = mean_new_prec = 0.0
        agg_old_fail = agg_new_fail = agg_delta_fail = 0.0
        mean_old_trunc = mean_new_trunc = mean_disprop = 0.0
        mean_old_seam = mean_new_seam = mean_delta_seam = 0.0

    # Decision Gates:
    # 1. Mean Delta QA Recall >= -0.03
    # 2. Mean Candidate Seam Integrity Rate >= 0.80
    recall_pass = mean_delta_rec >= -0.03
    seam_pass = mean_new_seam >= 0.80
    decision_pass = recall_pass and seam_pass

    reasons = []
    if not recall_pass:
        reasons.append(f"Mean Delta QA Recall drop ({mean_delta_rec * 100:+.1f}% pts) exceeds 3% threshold")
    if not seam_pass:
        reasons.append(f"Candidate Seam Integrity ({mean_new_seam * 100:.1f}%) below 80% threshold")
    if decision_pass:
        reasons.append("All decision gates passed: QA Recall delta >= -3% and Seam Integrity >= 80%")

    summary = CrossCheckSummary(
        total_samples=tot,
        mean_old_recall=mean_old_rec,
        mean_new_recall=mean_new_rec,
        mean_delta_recall=mean_delta_rec,
        mean_old_fail_ratio=agg_old_fail,
        mean_new_fail_ratio=agg_new_fail,
        mean_delta_fail_ratio=agg_delta_fail,
        mean_old_precision=mean_old_prec,
        mean_new_precision=mean_new_prec,
        mean_old_truncation_rate=mean_old_trunc,
        mean_new_truncation_rate=mean_new_trunc,
        mean_chunk_disproportion=mean_disprop,
        mean_old_seam_integrity=mean_old_seam,
        mean_new_seam_integrity=mean_new_seam,
        mean_delta_seam_integrity=mean_delta_seam,
        decision_pass=decision_pass,
        pass_reason="; ".join(reasons),
        total_duration_seconds=round(time.time() - t_global_start, 2),
    )

    return CrossCheckReport(
        mode="cross-check",
        model_name=model_name,
        baseline_branch=branch_a_name,
        candidate_branch=branch_b_name,
        summary=summary,
        samples=samples,
    )
