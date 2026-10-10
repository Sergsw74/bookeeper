"""
Cross-Check Verification Engine for A/B Testing Chunking Strategies.

Audits contiguous text blocks between two Knowledge Graph versions (Old / Baseline vs New / Candidate)
against a high-tier Oracle extraction on seamless reconstructed reference passages (W_raw).
Computes:
  - Oracle Recall for both systems and Delta Recall (New - Old)
  - Grounded Precision & Boundary Seam Truncation Artifact Rates
  - Chunk Disproportion: sum(block_A.len) / sum(block_A ∪ block_B.len)
  - Decision Gating Checklist
"""

import json
import logging
import random
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

from bookeeper.calibre.parser import BookParser
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import Concept, KnowledgeExtractor
from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings, OllamaPool

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


class CrossCheckBlockResult(BaseModel):
    """Audit evaluation result for a single sampled text window / block."""

    sample_index: int
    book_id: int
    book_title: str
    section_title: str
    old_chunk_ids: List[str]
    new_chunk_ids: List[str]
    passage_length_chars: int

    # Chunk lengths & Disproportion metric: sum(block_A.len) / sum(block_A ∪ block_B.len)
    block_a_total_len: int
    block_b_total_len: int
    union_len: int
    chunk_disproportion: float

    oracle_ideas_count: int
    old_system: SystemBlockMetrics
    new_system: SystemBlockMetrics
    delta_recall: float
    delta_fail_ratio: float = 0.0  # fail_ratio_B - fail_ratio_A
    duration_seconds: float = 0.0


class CrossCheckSummary(BaseModel):
    """Aggregated benchmark statistics and decision gating across all audited blocks."""

    total_samples: int
    mean_old_recall: float
    mean_new_recall: float
    mean_delta_recall: float
    mean_old_fail_ratio: float = 0.0
    mean_new_fail_ratio: float = 0.0
    mean_delta_fail_ratio: float = 0.0
    mean_old_precision: float
    mean_new_precision: float
    mean_old_truncation_rate: float
    mean_new_truncation_rate: float
    mean_chunk_disproportion: float
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
    Locate all chunks in Graph B from the same book whose contiguous text overlaps with W_raw.
    Requires at least `min_overlap` (default: 0.80) contiguous character coverage.
    Returns sorted list of chunk attribute dicts.
    """
    w_raw_norm = " ".join(w_raw.lower().split())
    if not w_raw_norm:
        return []

    candidates: List[Tuple[int, int, Dict[str, Any], float]] = []

    for nid, attrs in graph_b.graph.nodes(data=True):
        if attrs.get("type") != "Chunk":
            continue
        if attrs.get("book_id") != book_id:
            continue

        t = attrs.get("text", "")
        if not t:
            continue

        t_norm = " ".join(t.lower().split())
        ch_idx = int(attrs.get("chapter_idx") or 0)
        c_idx = int(attrs.get("chunk_idx") or 0)

        # 1. Exact substring check (chunk text completely within W_raw)
        if t_norm in w_raw_norm:
            candidates.append((ch_idx, c_idx, dict(attrs), 1.0))
            continue

        # 2. W_raw completely within chunk text
        if w_raw_norm in t_norm:
            candidates.append((ch_idx, c_idx, dict(attrs), 1.0))
            continue

        # 3. Contiguous sequence overlap via SequenceMatcher
        sm = SequenceMatcher(None, w_raw_norm, t_norm)
        match = sm.find_longest_match(0, len(w_raw_norm), 0, len(t_norm))
        ratio = match.size / max(1, len(t_norm))
        if ratio >= min_overlap:
            candidates.append((ch_idx, c_idx, dict(attrs), ratio))

    # Sort candidates by chapter and chunk index
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [c[2] for c in candidates]


def compute_chunk_disproportion(
    chunks_a_texts: List[str],
    chunks_b_texts: List[str],
    w_raw: str,
) -> Tuple[int, int, int, float]:
    """
    Compute chunk lengths and chunk disproportion metric:
      sum(block_A.len) / sum(block_A ∪ block_B.len)
    Returns:
      (sum_len_a, sum_len_b, union_len, chunk_disproportion)
    """
    sum_len_a = sum(len(t) for t in chunks_a_texts)
    sum_len_b = sum(len(t) for t in chunks_b_texts)

    if not chunks_a_texts and not chunks_b_texts:
        return 0, 0, 0, 0.0
    if not chunks_a_texts:
        return 0, sum_len_b, sum_len_b, 0.0
    if chunks_a_texts == chunks_b_texts:
        return sum_len_a, sum_len_b, sum_len_a, 1.0

    w_b = stitch_chunks_dedup_text(chunks_b_texts)

    # Union length represents unique character span covered across both block sets
    # Max of stitched lengths plus non-overlapping boundary extensions
    w_a_norm = " ".join(w_raw.split())
    w_b_norm = " ".join(w_b.split())

    if w_b_norm in w_a_norm:
        union_len = len(w_raw)
    elif w_a_norm in w_b_norm:
        union_len = len(w_b)
    else:
        # Approximate union length by overlapping character set or lengths
        # Avoid double-counting mutual intersection
        overlap_est = max(0, min(len(w_raw), len(w_b)) * 0.75)
        union_len = max(len(w_raw), len(w_b), int(len(w_raw) + len(w_b) - overlap_est))

    union_len = max(1, union_len)
    disproportion = round(sum_len_a / union_len, 4)

    return sum_len_a, sum_len_b, union_len, disproportion


# ==============================================================================
# 2. Tiered Deduplication & Concept Helpers
# ==============================================================================


def get_ideas_for_chunks(graph_store: ConceptGraphStore, chunk_node_ids: List[str]) -> List[Concept]:
    """Retrieve all Concept objects associated with a list of Chunk node IDs."""
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
                    ideas.append(
                        Concept(
                            name=cname,
                            category=cattrs.get("category", "Concept"),
                            brief_description=cattrs.get("brief_description", "") or cattrs.get("summary", ""),
                            detailed_explanation=cattrs.get("detailed_explanation", ""),
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
) -> List[str]:
    """
    Extract core standalone ideas directly from the unbroken passage W_raw using the Oracle model.
    """
    if extractor is not None:
        try:
            concepts = extractor.extract_ideas(passage, model=model_name)
            if concepts:
                return [c.name for c in concepts]
        except Exception as e:
            logger.warning(f"KnowledgeExtractor.extract_ideas failed on passage: {e}")

    prompt = (
        "You are an expert knowledge extractor, domain ontologist, and conceptual analyst.\n"
        "Read the following unbroken passage and extract all core standalone ideas, facts, "
        "principles, conditions, and causal relations.\n"
        "Ensure each extracted idea is self-contained with its necessary conditions intact.\n\n"
        f"Passage:\n\"\"\"\n{passage[:6000]}\n\"\"\"\n\n"
        "Return strictly valid JSON with this format:\n"
        "{\n"
        '  "ideas": [\n'
        '    {"name": "Concise Idea Title", "reasoning": "Self-contained assertion or principle"}\n'
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
        ideas = []
        for item in data.get("ideas", []):
            if isinstance(item, dict) and item.get("name"):
                ideas.append(item["name"].strip())
            elif isinstance(item, str):
                ideas.append(item.strip())
        return ideas
    except Exception as exc:
        logger.warning(f"Direct Oracle idea extraction failed: {exc}")
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
    candidate_ideas: List[str],
    oracle_ideas: List[str],
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

    retained = [oracle_ideas[i] for i in matched_oracle]
    dropped = [oracle_ideas[i] for i in range(len(oracle_ideas)) if i not in matched_oracle]

    recall = round(len(retained) / max(1, len(oracle_ideas)), 4) if oracle_ideas else 1.0

    # Audit candidate ideas not matched to Oracle (delta ideas)
    unmatched_candidates = [candidate_ideas[i] for i in range(len(candidate_ideas)) if i not in matched_candidate]
    audited_items: List[CrossCheckAuditItem] = []

    for u_cand in unmatched_candidates:
        item = audit_unmatched_idea(u_cand, w_raw, pool, model_name, extractor)
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
    progress_callback: Optional[Callable[[int, int, str, Optional[CrossCheckBlockResult]], None]] = None,
) -> CrossCheckReport:
    """
    Execute full Cross-Check verification pipeline comparing Store A and Store B.
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

        # 3. Locate matching new chunks in Graph B with min_overlap >= 0.80
        new_chunk_dicts = find_overlapping_chunks(store_b, target_bid, w_raw, min_overlap=0.80)
        if not new_chunk_dicts:
            continue

        new_chunk_ids = [d.get("chunk_id") or d.get("id", "") for d in new_chunk_dicts]
        new_chunk_texts = [d.get("text", "") for d in new_chunk_dicts]

        # 4. Compute chunk disproportion: sum(block_A.len) / sum(block_A ∪ block_B.len)
        sum_len_a, sum_len_b, union_len, disproportion = compute_chunk_disproportion(
            old_chunk_texts, new_chunk_texts, w_raw
        )

        # 5. Union & Tiered Deduplication of ideas
        raw_ideas_old = get_ideas_for_chunks(store_a, old_chunk_ids)
        raw_ideas_new = get_ideas_for_chunks(store_b, new_chunk_ids)

        dedup_ideas_old = tiered_deduplicate_ideas(raw_ideas_old, deduplicator)
        dedup_ideas_new = tiered_deduplicate_ideas(raw_ideas_new, deduplicator)

        e_old = [c.name for c in dedup_ideas_old]
        e_new = [c.name for c in dedup_ideas_new]

        # 6. Oracle extraction on contiguous passage
        i_oracle = extract_oracle_ideas_from_passage(w_raw, pool, model_name, extractor)

        # 7. Evaluate both systems against Oracle
        eval_old = evaluate_system_against_oracle(e_old, i_oracle, w_raw, "old", deduplicator, pool, model_name, extractor)
        eval_new = evaluate_system_against_oracle(e_new, i_oracle, w_raw, "new", deduplicator, pool, model_name, extractor)

        delta_rec = round(eval_new.oracle_recall - eval_old.oracle_recall, 4)
        delta_fail = round(eval_new.fail_ratio - eval_old.fail_ratio, 4)
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
            oracle_ideas_count=len(i_oracle),
            old_system=eval_old,
            new_system=eval_new,
            delta_recall=delta_rec,
            delta_fail_ratio=delta_fail,
            duration_seconds=dur,
        )
        samples.append(block_result)

        if progress_callback:
            progress_callback(curr_idx, num_blocks, f"Block #{curr_idx} ({b_title[:20]})", block_result)

    # 8. Compute Summary & Decision Checklist
    tot = len(samples)
    if tot > 0:
        mean_old_rec = round(float(np.mean([s.old_system.oracle_recall for s in samples])), 4)
        mean_new_rec = round(float(np.mean([s.new_system.oracle_recall for s in samples])), 4)
        mean_delta_rec = round(float(np.mean([s.delta_recall for s in samples])), 4)

        # Micro-aggregated fail ratios: sum(failed) / sum(total_block_ideas)
        total_old_dropped = sum(s.old_system.dropped_ideas_count for s in samples)
        total_old_ideas = sum(s.old_system.deduped_ideas_count for s in samples)
        agg_old_fail = round(total_old_dropped / total_old_ideas, 4) if total_old_ideas > 0 else 0.0

        total_new_dropped = sum(s.new_system.dropped_ideas_count for s in samples)
        total_new_ideas = sum(s.new_system.deduped_ideas_count for s in samples)
        agg_new_fail = round(total_new_dropped / total_new_ideas, 4) if total_new_ideas > 0 else 0.0

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

    # Decision Gates:
    # 1. Mean Delta Recall >= -0.03
    # 2. Mean New Truncation Rate < 0.02
    recall_pass = mean_delta_rec >= -0.03
    trunc_pass = mean_new_trunc < 0.02
    decision_pass = recall_pass and trunc_pass

    reasons = []
    if not recall_pass:
        reasons.append(f"Mean Delta Recall drop ({mean_delta_rec * 100:.1f}%) exceeds 3% threshold")
    if not trunc_pass:
        reasons.append(f"Mean Truncation Rate ({mean_new_trunc * 100:.1f}%) exceeds 2% artifact threshold")
    if decision_pass:
        reasons.append("All decision gates passed: Recall delta >= -3% and Truncation Rate < 2%")

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
