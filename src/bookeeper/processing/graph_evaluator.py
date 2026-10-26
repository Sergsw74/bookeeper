"""
Comprehensive Metric Suite for Knowledge Graph Embedding Space Evaluation.

Evaluates how effectively a Knowledge Graph (idea nodes G) covers and disperses across
the semantic literature universe (raw chunks U) using cosine projection geometry,
distributional percentiles, Hausdorff blind spots, and Vendi spectral entropy diversity.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.memory_guard import (
    DEFAULT_MAX_MEMORY_BYTES,
    check_memory_limit,
    cleanup_memory,
    get_process_memory_gb,
)

logger = logging.getLogger(__name__)


def compute_max_cosine_sim_batched(
    U: np.ndarray,
    G: np.ndarray,
    max_batch_mem_bytes: int = 64 * 1024 * 1024,
) -> np.ndarray:
    """
    Computes max cosine similarity across G for each vector in U in memory-safe batches.
    Caps intermediate projection matrices strictly to max_batch_mem_bytes (default 64MB).
    """
    if len(U) == 0 or len(G) == 0:
        return np.zeros(len(U), dtype=np.float32)

    n_u = len(U)
    n_g = len(G)
    # Estimate batch size so (batch_size * n_g * 4 bytes) <= max_batch_mem_bytes
    item_bytes = max(1, n_g * 4)  # float32 = 4 bytes
    batch_size = max(50, min(5000, max_batch_mem_bytes // item_bytes))

    # Pre-cast to float32 contiguous array
    G_T = np.ascontiguousarray(G.T, dtype=np.float32)
    results = np.empty(n_u, dtype=np.float32)

    for i in range(0, n_u, batch_size):
        u_batch = np.ascontiguousarray(U[i : i + batch_size], dtype=np.float32)
        sim_batch = u_batch @ G_T
        results[i : i + batch_size] = np.max(sim_batch, axis=1)

    return results


def compute_weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """
    Computes weighted percentile (0-100) using cumulative weight distribution.
    Falls back to unweighted percentile if weights are empty or uniform zero.
    """
    if len(values) == 0:
        return 0.0
    if len(weights) != len(values) or np.sum(weights) <= 0:
        return float(np.percentile(values, percentile))
    idx = np.argsort(values)
    s_vals = values[idx]
    s_weights = weights[idx]
    cum_w = np.cumsum(s_weights)
    total_w = cum_w[-1]
    cutoff = (percentile / 100.0) * total_w
    p_idx = int(np.searchsorted(cum_w, cutoff))
    return float(s_vals[min(p_idx, len(s_vals) - 1)])


# ==============================================================================
# 1. Pydantic Models for Single Graph & Comparative Reports
# ==============================================================================


class GraphSpaceMetrics(BaseModel):
    """
    Evaluation metrics for a Knowledge Graph's dispersion and coverage over corpus space.
    """

    nodes_count: int = Field(..., description="Number of Idea/Concept nodes M in the graph.")
    universe_chunks_count: int = Field(0, description="Total number of corpus chunks N in the universe.")
    universe_total_chars: Optional[int] = Field(None, description="Total character weight of corpus chunks evaluated.")
    chamfer_mean: float = Field(..., description="Size-weighted mean Chamfer proximity across literature chunks.")
    chamfer_std: float = Field(..., description="Size-weighted regularity std dev.")
    median_p50: float = Field(..., description="Size-weighted median proximity (50th percentile).")
    worst_decile_p10: float = Field(..., description="Size-weighted worst-decile proximity (10th percentile).")
    coverage_at_tau_pct: float = Field(..., description="Threshold coverage: percentage of literature text volume reachable within cosine radius tau.")
    worst_case_gap: float = Field(..., description="Directed Hausdorff gap (1.0 - min_sim): absolute worst-case gap to the most isolated chunk.")
    vendi_diversity_score: float = Field(..., description="Vendi score: effective number of orthogonal, non-redundant concept modes spanned by G.")
    coverage_per_node_efficiency: float = Field(..., description="Compression efficiency: coverage percentage retained per graph node.")
    vendi_per_node: float = Field(0.0, description="Concept quality ratio: effective independent modes per node (lower indicates duplicate redundancy).")
    tau: float = Field(0.75, description="Semantic relevance threshold applied.")

    # Skipping Leverage Metrics
    parent_chunks_count: Optional[int] = Field(None, description="Number of chunks from which ideas were extracted.")
    skipped_chunks_count: Optional[int] = Field(None, description="Number of chunks bypassed/skipped by heuristic.")
    parent_chunk_ratio_pct: Optional[float] = Field(None, description="Extraction ratio rho_parent = W_parent / W_total (size-weighted percentage).")
    skipped_retention_pct: Optional[float] = Field(None, description="Skipped text retention rate R_skipped within tau radius (size-weighted).")
    generalization_surplus_pct: Optional[float] = Field(None, description="Generalization surplus Delta_cov = Cov_tau - rho_parent (size-weighted).")
    compression_leverage_factor: Optional[float] = Field(None, description="Compression leverage L_comp = Cov_tau / rho_parent (size-weighted).")


class GraphSpaceComparisonReport(BaseModel):
    """
    A/B Comparative Benchmark Report comparing Baseline (A) vs Candidate (B) over their chunk Universes.
    """

    tau: float = 0.75
    universe_chunks_count: int
    universe_chunks_count_a: Optional[int] = None
    universe_chunks_count_b: Optional[int] = None
    universe_total_chars: Optional[int] = None
    universe_total_chars_a: Optional[int] = None
    universe_total_chars_b: Optional[int] = None
    label_a: str = "Baseline (A)"
    label_b: str = "Candidate (B)"
    baseline_metrics: GraphSpaceMetrics
    candidate_metrics: GraphSpaceMetrics

    # Deltas (B - A)
    delta_nodes_count: int
    delta_chamfer_mean: float
    delta_chamfer_std: float
    delta_median_p50: float
    delta_worst_decile_p10: float
    delta_coverage_at_tau_pct: float
    delta_worst_case_gap: float
    delta_vendi_diversity_score: float
    delta_coverage_per_node_efficiency: float
    delta_vendi_per_node: float

    # Deltas for Skipping Leverage
    delta_parent_chunk_ratio_pct: Optional[float] = Field(None, description="Delta extraction ratio (B - A).")
    delta_skipped_retention_pct: Optional[float] = Field(None, description="Delta skipped retention rate (B - A).")
    delta_generalization_surplus_pct: Optional[float] = Field(None, description="Delta generalization surplus (B - A).")
    delta_compression_leverage_factor: Optional[float] = Field(None, description="Delta compression leverage (B - A).")

    # Decision Matrix Evaluation
    decision_pass: bool
    pass_reason: str
    gate_checklist: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


# ==============================================================================
# 2. Core Mathematical Graph Evaluator & Skipping Leverage Suite
# ==============================================================================


def evaluate_skipping_leverage(
    parent_chunk_embeddings: np.ndarray,   # (N_p, D) chunks where ideas were extracted
    skipped_chunk_embeddings: np.ndarray,  # (N_s, D) chunks skipped by heuristic
    graph_embeddings: np.ndarray,          # (M, D) extracted idea nodes in G
    tau: float = 0.75,
    parent_weights: Optional[np.ndarray] = None,   # (N_p,) chunk character weights
    skipped_weights: Optional[np.ndarray] = None,  # (N_s,) chunk character weights
) -> Dict[str, Any]:
    """
    Evaluates the compression leverage and semantic retention of the skipping mechanism,
    weighted by chunk size (character count) to avoid distortion from varied chunk sizes.

    Parameters:
        parent_chunk_embeddings: (N_p, D) vectors of chunks where ideas were extracted.
        skipped_chunk_embeddings: (N_s, D) vectors of chunks bypassed/skipped by heuristic.
        graph_embeddings: (M, D) vectors of extracted idea nodes in G.
        tau: Semantic relevance threshold (default: 0.75).
        parent_weights: Optional (N_p,) float array of parent chunk sizes in characters.
        skipped_weights: Optional (N_s,) float array of skipped chunk sizes in characters.

    Returns:
        Dict containing size-weighted leverage indicators:
            parent_chunk_ratio_pct: fraction of corpus text volume spent on compute (rho_parent).
            total_coverage_pct: percentage of combined text volume covered within tau radius.
            skipped_retention_pct: percentage of bypassed text volume safely retained (R_skipped).
            generalization_surplus_pct: excess text volume coverage gained over extraction ratio (Delta_cov).
            compression_leverage_factor: coverage multiplier per compute unit (L_comp).
    """
    def norm(x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.empty((0, 0), dtype=np.float32)
        arr = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    G = norm(graph_embeddings)
    U_parent = norm(parent_chunk_embeddings)
    U_skipped = norm(skipped_chunk_embeddings)

    n_p = len(U_parent)
    n_s = len(U_skipped)
    n_total = n_p + n_s

    # Size weights (in characters)
    w_p = np.asarray(parent_weights, dtype=np.float64) if parent_weights is not None and len(parent_weights) == n_p else np.ones(n_p, dtype=np.float64)
    w_s = np.asarray(skipped_weights, dtype=np.float64) if skipped_weights is not None and len(skipped_weights) == n_s else np.ones(n_s, dtype=np.float64)
    w_p = np.where(w_p > 0, w_p, 1.0)
    w_s = np.where(w_s > 0, w_s, 1.0)

    tot_w_p = float(np.sum(w_p)) if n_p > 0 else 0.0
    tot_w_s = float(np.sum(w_s)) if n_s > 0 else 0.0
    tot_w = tot_w_p + tot_w_s

    # Max similarity to G for both sets using memory-safe batched projection
    if len(G) > 0:
        sim_parent = compute_max_cosine_sim_batched(U_parent, G) if n_p > 0 else np.array([], dtype=np.float32)
        sim_skipped = compute_max_cosine_sim_batched(U_skipped, G) if n_s > 0 else np.array([], dtype=np.float32)
    else:
        sim_parent = np.zeros(n_p, dtype=np.float32)
        sim_skipped = np.zeros(n_s, dtype=np.float32)

    # Size-weighted mass meeting threshold tau
    cov_p_weight = float(np.sum(w_p[sim_parent >= tau])) if n_p > 0 else 0.0
    cov_s_weight = float(np.sum(w_s[sim_skipped >= tau])) if n_s > 0 else 0.0
    total_cov_weight = cov_p_weight + cov_s_weight

    # Size-weighted ratios
    parent_ratio = (tot_w_p / tot_w) if tot_w > 0 else 0.0
    total_coverage = (total_cov_weight / tot_w) if tot_w > 0 else 0.0
    skipped_retention = (cov_s_weight / tot_w_s) if tot_w_s > 0 else 1.0

    # Leverage Metrics
    generalization_surplus = total_coverage - parent_ratio
    compression_leverage = (total_coverage / parent_ratio) if parent_ratio > 0 else 0.0

    return {
        "parent_chunks_count": n_p,
        "skipped_chunks_count": n_s,
        "total_chunks_count": n_total,
        "parent_chunk_ratio_pct": round(parent_ratio * 100, 2),
        "total_coverage_pct": round(total_coverage * 100, 2),
        "skipped_retention_pct": round(skipped_retention * 100, 2),
        "generalization_surplus_pct": round(generalization_surplus * 100, 2),
        "compression_leverage_factor": round(compression_leverage, 2),
    }


def evaluate_graph_space(
    universe_embeddings: np.ndarray,  # (N, D) - all raw chunks from the corpus
    graph_embeddings: np.ndarray,     # (M, D) - extracted idea nodes in Graph G
    tau: float = 0.75,                # Semantic relevance threshold
    batch_size: int = 5000,
    parent_chunk_mask: Optional[np.ndarray] = None,  # (N,) bool array indicating parent chunks
    chunk_weights: Optional[np.ndarray] = None,      # (N,) float array of chunk sizes (chars)
) -> Dict[str, Any]:
    """
    Evaluates how effectively Graph G covers and disperses across the 
    semantic embedding space defined by Universe U, weighted by chunk character size.

    Parameters:
        universe_embeddings: (N, D) float array of chunk vectors.
        graph_embeddings: (M, D) float array of idea concept vectors.
        tau: Semantic proximity radius threshold (default: 0.75).
        batch_size: Chunk size for blockwise matrix multiplication to preserve RAM.
        parent_chunk_mask: Optional (N,) boolean mask marking chunks where ideas were extracted.
        chunk_weights: Optional (N,) float array of chunk sizes in characters.

    Returns:
        Dict matching GraphSpaceMetrics schema.
    """
    check_memory_limit(context="evaluate_graph_space_start")

    if len(graph_embeddings) == 0 or len(universe_embeddings) == 0:
        return {
            "nodes_count": len(graph_embeddings),
            "universe_chunks_count": len(universe_embeddings),
            "universe_total_chars": 0,
            "chamfer_mean": 0.0,
            "chamfer_std": 0.0,
            "median_p50": 0.0,
            "worst_decile_p10": 0.0,
            "coverage_at_tau_pct": 0.0,
            "worst_case_gap": 1.0,
            "vendi_diversity_score": 0.0,
            "coverage_per_node_efficiency": 0.0,
            "vendi_per_node": 0.0,
            "tau": float(tau),
            "parent_chunks_count": None,
            "skipped_chunks_count": None,
            "parent_chunk_ratio_pct": None,
            "skipped_retention_pct": None,
            "generalization_surplus_pct": None,
            "compression_leverage_factor": None,
        }

    # 1. Enforce L2 unit-norm for cosine dot products in float32
    def normalize(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    U = normalize(universe_embeddings)
    G = normalize(graph_embeddings)
    n_nodes = len(G)
    n_universe = len(U)

    # Enforce chunk weights (character size)
    if chunk_weights is not None and len(chunk_weights) == n_universe:
        W = np.asarray(chunk_weights, dtype=np.float64)
        W = np.where(W > 0, W, 1.0)
    else:
        W = np.ones(n_universe, dtype=np.float64)
    tot_w = float(np.sum(W))

    # 2. Pairwise Cosine Projection (memory-bounded in float32 batches)
    max_sims = compute_max_cosine_sim_batched(U, G, max_batch_mem_bytes=64 * 1024 * 1024)

    # 3. Size-Weighted Distributional Metrics (Chamfer, Variance, Percentiles)
    chamfer_mean = float(np.sum(W * max_sims) / tot_w) if tot_w > 0 else 0.0
    chamfer_std = float(np.sqrt(np.sum(W * (max_sims - chamfer_mean) ** 2) / tot_w)) if tot_w > 0 else 0.0
    p10 = compute_weighted_percentile(max_sims, W, 10.0)
    p50 = compute_weighted_percentile(max_sims, W, 50.0)
    coverage_tau = float(np.sum(W[max_sims >= tau]) / tot_w * 100.0) if tot_w > 0 else 0.0

    # 4. Extreme Blind Spot (Directed Hausdorff Distance in Cosine Distance)
    # Cosine distance = 1.0 - cosine_similarity
    worst_case_gap = float(1.0 - np.min(max_sims))

    # 5. Vendi Diversity Score (Effective Independent Concept Modes)
    # Kernel: linear dot-product shifted to [0, 1] PSD range: K_ij = (1 + G_i . G_j) / 2
    # Dual Spectral Formulation:
    # When n_nodes > D + 1, the non-zero eigenvalues of (1/M) K are IDENTICAL to the non-zero
    # eigenvalues of the dual Gram matrix C = (1/2) * [[1, mean_g^T], [mean_g, (1/M) G^T G]] of size (D+1, D+1).
    # This bounds memory strictly to min(M, D+1)^2 floats (< 5 MB for D=768) instead of allocating an
    # (M x M) matrix which requires tens of gigabytes when M >= 30,000.
    if n_nodes > 1:
        dim = G.shape[1] if G.ndim > 1 else 1
        if n_nodes > dim + 1:
            mean_g = np.mean(G, axis=0, dtype=np.float64)
            C = np.empty((dim + 1, dim + 1), dtype=np.float64)
            C[0, 0] = 1.0
            C[0, 1:] = mean_g
            C[1:, 0] = mean_g
            # Compute G^T G in float64
            C[1:, 1:] = (G.T.astype(np.float64) @ G.astype(np.float64)) / n_nodes
            C *= 0.5
            eigenvalues = np.linalg.eigvalsh(C)
        else:
            K = (1.0 + (G @ G.T)) / 2.0
            K_norm = K / n_nodes
            eigenvalues = np.linalg.eigvalsh(K_norm)

        # Numerical guard: zero out tiny or negative eigenvalues due to floating-point noise
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        vendi_score = float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))
    elif n_nodes == 1:
        vendi_score = 1.0
    else:
        vendi_score = 0.0

    # 6. Graph Efficiency
    efficiency = float(coverage_tau / n_nodes) if n_nodes > 0 else 0.0
    vendi_density = float(vendi_score / n_nodes) if n_nodes > 0 else 0.0

    # 7. Skipping Leverage & Generalization Analysis (Size-Weighted)
    if parent_chunk_mask is not None and len(parent_chunk_mask) == n_universe:
        p_mask = np.asarray(parent_chunk_mask, dtype=bool)
        n_p = int(np.sum(p_mask))
        n_s = n_universe - n_p

        W_p = W[p_mask]
        W_s = W[~p_mask]
        tot_w_p = float(np.sum(W_p))
        tot_w_s = float(np.sum(W_s))

        sim_parent = max_sims[p_mask] if n_p > 0 else np.array([], dtype=np.float32)
        sim_skipped = max_sims[~p_mask] if n_s > 0 else np.array([], dtype=np.float32)

        cov_p_weight = float(np.sum(W_p[sim_parent >= tau])) if n_p > 0 else 0.0
        cov_s_weight = float(np.sum(W_s[sim_skipped >= tau])) if n_s > 0 else 0.0
        tot_cov_weight = cov_p_weight + cov_s_weight

        p_ratio = (tot_w_p / tot_w) if tot_w > 0 else 0.0
        tot_cov = (tot_cov_weight / tot_w) if tot_w > 0 else 0.0
        s_retention = (cov_s_weight / tot_w_s) if tot_w_s > 0 else 1.0

        gen_surplus = tot_cov - p_ratio
        comp_lev = (tot_cov / p_ratio) if p_ratio > 0 else 0.0

        p_count_val: Optional[int] = n_p
        s_count_val: Optional[int] = n_s
        p_ratio_val: Optional[float] = round(p_ratio * 100, 2)
        s_retention_val: Optional[float] = round(s_retention * 100, 2)
        gen_surplus_val: Optional[float] = round(gen_surplus * 100, 2)
        comp_lev_val: Optional[float] = round(comp_lev, 2)
    else:
        p_count_val = None
        s_count_val = None
        p_ratio_val = None
        s_retention_val = None
        gen_surplus_val = None
        comp_lev_val = None

    check_memory_limit(context="evaluate_graph_space_finish")
    cleanup_memory()

    return {
        "nodes_count": n_nodes,
        "universe_chunks_count": n_universe,
        "universe_total_chars": int(tot_w),
        "chamfer_mean": round(chamfer_mean, 4),
        "chamfer_std": round(chamfer_std, 4),
        "median_p50": round(p50, 4),
        "worst_decile_p10": round(p10, 4),
        "coverage_at_tau_pct": round(coverage_tau, 2),
        "worst_case_gap": round(worst_case_gap, 4),
        "vendi_diversity_score": round(vendi_score, 2),
        "coverage_per_node_efficiency": round(efficiency, 4),
        "vendi_per_node": round(vendi_density, 4),
        "tau": round(float(tau), 4),
        "parent_chunks_count": p_count_val,
        "skipped_chunks_count": s_count_val,
        "parent_chunk_ratio_pct": p_ratio_val,
        "skipped_retention_pct": s_retention_val,
        "generalization_surplus_pct": gen_surplus_val,
        "compression_leverage_factor": comp_lev_val,
    }


# ==============================================================================
# 3. Practical A/B Decision Matrix Evaluator
# ==============================================================================


def compare_graph_spaces(
    universe_embeddings: np.ndarray,
    baseline_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    tau: float = 0.75,
    label_a: str = "Baseline (A)",
    label_b: str = "Candidate (B)",
    parent_mask_a: Optional[np.ndarray] = None,
    parent_mask_b: Optional[np.ndarray] = None,
    universe_embeddings_b: Optional[np.ndarray] = None,
    chunk_weights: Optional[np.ndarray] = None,
    chunk_weights_b: Optional[np.ndarray] = None,
) -> GraphSpaceComparisonReport:
    """
    Evaluates and compares Baseline (G1) vs Candidate (G2) against their respective chunk Universes
    applying the multi-factor Practical Decision Matrix, weighted by chunk size in characters.
    """
    u_b = universe_embeddings_b if universe_embeddings_b is not None else universe_embeddings
    w_b = chunk_weights_b if chunk_weights_b is not None else chunk_weights

    raw_m1 = evaluate_graph_space(
        universe_embeddings,
        baseline_embeddings,
        tau=tau,
        parent_chunk_mask=parent_mask_a,
        chunk_weights=chunk_weights,
    )
    raw_m2 = evaluate_graph_space(
        u_b,
        candidate_embeddings,
        tau=tau,
        parent_chunk_mask=parent_mask_b,
        chunk_weights=w_b,
    )

    m1 = GraphSpaceMetrics(**raw_m1)
    m2 = GraphSpaceMetrics(**raw_m2)

    d_nodes = m2.nodes_count - m1.nodes_count
    d_chamfer = round(m2.chamfer_mean - m1.chamfer_mean, 4)
    d_std = round(m2.chamfer_std - m1.chamfer_std, 4)
    d_p50 = round(m2.median_p50 - m1.median_p50, 4)
    d_p10 = round(m2.worst_decile_p10 - m1.worst_decile_p10, 4)
    d_cov = round(m2.coverage_at_tau_pct - m1.coverage_at_tau_pct, 2)
    d_gap = round(m2.worst_case_gap - m1.worst_case_gap, 4)
    d_vendi = round(m2.vendi_diversity_score - m1.vendi_diversity_score, 2)
    d_eff = round(m2.coverage_per_node_efficiency - m1.coverage_per_node_efficiency, 4)
    d_vdensity = round(m2.vendi_per_node - m1.vendi_per_node, 4)

    # Skipping leverage deltas
    d_pratio = (
        round(m2.parent_chunk_ratio_pct - m1.parent_chunk_ratio_pct, 2)
        if m1.parent_chunk_ratio_pct is not None and m2.parent_chunk_ratio_pct is not None
        else None
    )
    d_sret = (
        round(m2.skipped_retention_pct - m1.skipped_retention_pct, 2)
        if m1.skipped_retention_pct is not None and m2.skipped_retention_pct is not None
        else None
    )
    d_surplus = (
        round(m2.generalization_surplus_pct - m1.generalization_surplus_pct, 2)
        if m1.generalization_surplus_pct is not None and m2.generalization_surplus_pct is not None
        else None
    )
    d_leverage = (
        round(m2.compression_leverage_factor - m1.compression_leverage_factor, 2)
        if m1.compression_leverage_factor is not None and m2.compression_leverage_factor is not None
        else None
    )

    # Decision Matrix Checklist:
    # 1. Check Coverage Drop: Candidate coverage should not drop by more than 4%
    cov_pass = d_cov >= -4.0
    cov_reason = (
        f"Coverage drop {d_cov:+.2f}% is within acceptable 4.0% threshold"
        if cov_pass
        else f"Coverage drop {d_cov:+.2f}% exceeds acceptable 4.0% ceiling"
    )

    # 2. Check Regularity: chamfer_std shouldn't increase significantly (+0.03) and p10 >= 0.50 or delta >= -0.05
    reg_pass = (d_std <= 0.03) and (m2.worst_decile_p10 >= 0.50 or d_p10 >= -0.05)
    reg_reason = (
        f"Regularity preserved (std dev Δ {d_std:+.4f}, p10 {m2.worst_decile_p10:.4f})"
        if reg_pass
        else f"Coverage regularity degraded (std dev Δ {d_std:+.4f}, p10 {m2.worst_decile_p10:.4f})"
    )

    # 3. Check Maximum Blind Spot: Directed Hausdorff gap should not spike (> +0.15)
    blind_pass = d_gap <= 0.15
    blind_reason = (
        f"Worst-case gap Δ {d_gap:+.4f} within +0.15 limit (no topic severed)"
        if blind_pass
        else f"Severe blind spot detected: worst-case gap surged {d_gap:+.4f} (> +0.15)"
    )

    # 4. Check Concept Quality & Efficiency: either node efficiency or vendi density maintained/improved
    eff_pass = (d_eff >= -0.005) or (d_vdensity >= -0.02)
    eff_reason = (
        f"Concept efficiency verified (Efficiency Δ {d_eff:+.4f}, Vendi density Δ {d_vdensity:+.4f})"
        if eff_pass
        else f"Concept economy diluted (Efficiency Δ {d_eff:+.4f})"
    )

    all_pass = cov_pass and reg_pass and blind_pass and eff_pass

    checklist: Dict[str, Dict[str, Any]] = {
        "coverage_retention": {
            "name": "Corpus Coverage Retention",
            "passed": cov_pass,
            "threshold": "Δ Coverage ≥ -4.0% pts",
            "observed": f"{d_cov:+.2f}% pts",
            "details": cov_reason,
        },
        "coverage_regularity": {
            "name": "Topological Regularity (Std & p10)",
            "passed": reg_pass,
            "threshold": "Δ Std ≤ +0.03 and Candidate p10 ≥ 0.50 (or Δ ≥ -0.05)",
            "observed": f"Δ Std: {d_std:+.4f}, Candidate p10: {m2.worst_decile_p10:.4f}",
            "details": reg_reason,
        },
        "blind_spot_ceiling": {
            "name": "Maximum Blind Spot (Directed Hausdorff)",
            "passed": blind_pass,
            "threshold": "Δ Gap ≤ +0.15",
            "observed": f"Δ Gap: {d_gap:+.4f} (Candidate: {m2.worst_case_gap:.4f})",
            "details": blind_reason,
        },
        "concept_quality_economy": {
            "name": "Concept Economy & Vendi Diversity",
            "passed": eff_pass,
            "threshold": "Efficiency Δ ≥ -0.005 or Vendi/Node Δ ≥ -0.02",
            "observed": f"Efficiency Δ: {d_eff:+.4f}, Vendi/Node: {m2.vendi_per_node:.3f}",
            "details": eff_reason,
        },
    }

    # 5. Skipping Generalization & Retention Gate (when skipping metrics are available)
    if m2.skipped_retention_pct is not None and m2.generalization_surplus_pct is not None:
        # Check that skipped retention is >= 75.0% and generalization surplus >= 0.0% (or leverage >= 1.0)
        skip_pass = (m2.skipped_retention_pct >= 75.0) and (
            m2.generalization_surplus_pct >= 0.0 or (m2.compression_leverage_factor or 0.0) >= 1.0
        )
        skip_reason = (
            f"Skipping heuristic validated: {m2.skipped_retention_pct:.1f}% skipped chunks retained within radius τ, surplus {m2.generalization_surplus_pct:+.1f}% pts (leverage {m2.compression_leverage_factor:.2f}x)"
            if skip_pass
            else f"Skipping heuristic dropping information: retention {m2.skipped_retention_pct:.1f}% (< 75.0%) or surplus {m2.generalization_surplus_pct:+.1f}% pts (< 0%)"
        )
        checklist["skipping_generalization"] = {
            "name": "Skipping Generalization & Retention",
            "passed": skip_pass,
            "threshold": "Skipped Retention ≥ 75.0% and Generalization Surplus ≥ 0.0%",
            "observed": f"Retention: {m2.skipped_retention_pct:.1f}%, Surplus: {m2.generalization_surplus_pct:+.1f}% pts (Leverage: {m2.compression_leverage_factor:.2f}x)",
            "details": skip_reason,
        }
        all_pass = all_pass and skip_pass

    if all_pass:
        verdict = f"PASS: Candidate preserves semantic coverage ({d_cov:+.2f}%) with clean dispersion and no severe blind spots."
    else:
        failed_gates = [v["name"] for v in checklist.values() if not v["passed"]]
        verdict = f"FAIL: Candidate breached gating conditions on: {', '.join(failed_gates)}."

    return GraphSpaceComparisonReport(
        tau=tau,
        universe_chunks_count=len(universe_embeddings) if universe_embeddings_b is None or len(universe_embeddings) == len(u_b) else len(u_b),
        universe_chunks_count_a=len(universe_embeddings),
        universe_chunks_count_b=len(u_b),
        universe_total_chars=m2.universe_total_chars if universe_embeddings_b is not None else m1.universe_total_chars,
        universe_total_chars_a=m1.universe_total_chars,
        universe_total_chars_b=m2.universe_total_chars,
        label_a=label_a,
        label_b=label_b,
        baseline_metrics=m1,
        candidate_metrics=m2,
        delta_nodes_count=d_nodes,
        delta_chamfer_mean=d_chamfer,
        delta_chamfer_std=d_std,
        delta_median_p50=d_p50,
        delta_worst_decile_p10=d_p10,
        delta_coverage_at_tau_pct=d_cov,
        delta_worst_case_gap=d_gap,
        delta_vendi_diversity_score=d_vendi,
        delta_coverage_per_node_efficiency=d_eff,
        delta_vendi_per_node=d_vdensity,
        delta_parent_chunk_ratio_pct=d_pratio,
        delta_skipped_retention_pct=d_sret,
        delta_generalization_surplus_pct=d_surplus,
        delta_compression_leverage_factor=d_leverage,
        decision_pass=all_pass,
        pass_reason=verdict,
        gate_checklist=checklist,
    )


# ==============================================================================
# 4. Graph Embeddings Extraction & Preprocessing Helpers
# ==============================================================================


def get_parent_chunk_ids(store: ConceptGraphStore) -> Set[str]:
    """
    Returns the set of chunk IDs that directly support at least one Idea/Concept node in store.
    Identifies chunks connected via SUPPORTED_BY or SUPPORTS_IDEA relations,
    or flagged with is_parent/ideas metadata.
    """
    parent_cids: Set[str] = set()
    for u, v, data in store.graph.edges(data=True):
        rel = data.get("relation")
        if rel in ("SUPPORTED_BY", "SUPPORTS_IDEA"):
            if data.get("chunk_id"):
                parent_cids.add(str(data["chunk_id"]))
            for nid in (u, v):
                node_attrs = store.graph.nodes.get(nid, {})
                if str(nid).startswith("chunk:") or node_attrs.get("type") == "Chunk":
                    cid = node_attrs.get("chunk_id") or str(nid).replace("chunk:", "")
                    parent_cids.add(str(cid))
    # Also inspect node attributes for fallback metadata
    for nid, attrs in store.graph.nodes(data=True):
        if attrs.get("type") == "Chunk":
            if attrs.get("is_parent") or attrs.get("supports_ideas") or attrs.get("extracted_concepts"):
                cid = attrs.get("chunk_id") or str(nid).replace("chunk:", "")
                parent_cids.add(str(cid))
    return parent_cids


def extract_concept_embeddings(store: ConceptGraphStore) -> Tuple[List[str], np.ndarray]:
    """
    Extracts all Idea/Concept node IDs and their stored vector embeddings.
    """
    names: List[str] = []
    vec_arrays: List[np.ndarray] = []

    for node_id, attrs in store.graph.nodes(data=True):
        ntype = attrs.get("type", "")
        is_idea = bool(attrs.get("is_idea") or attrs.get("tag") == "idea" or ntype in ("Concept", "Idea"))
        if is_idea:
            emb = attrs.get("embedding")
            if isinstance(emb, (list, tuple, np.ndarray)) and len(emb) > 1:
                clean_name = attrs.get("name") or str(node_id).replace("concept:", "")
                names.append(str(clean_name))
                vec_arrays.append(np.asarray(emb, dtype=np.float32))

    if not vec_arrays:
        return names, np.empty((0, 0), dtype=np.float32)

    return names, np.vstack(vec_arrays)


def extract_chunk_embeddings_and_weights(
    store: ConceptGraphStore,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Extracts all Chunk node IDs, their stored vector embeddings, and chunk character weights.
    """
    chunk_ids: List[str] = []
    vec_arrays: List[np.ndarray] = []
    weights: List[float] = []

    for node_id, attrs in store.graph.nodes(data=True):
        if attrs.get("type") == "Chunk":
            emb = attrs.get("embedding")
            if isinstance(emb, (list, tuple, np.ndarray)) and len(emb) > 1:
                cid = attrs.get("chunk_id") or str(node_id).replace("chunk:", "")
                chunk_ids.append(str(cid))
                vec_arrays.append(np.asarray(emb, dtype=np.float32))
                c_len = attrs.get("char_count")
                if c_len is not None and int(c_len) > 0:
                    weights.append(float(c_len))
                else:
                    txt = attrs.get("text") or ""
                    weights.append(float(len(txt)) if len(txt) > 0 else 1.0)

    if not vec_arrays:
        return chunk_ids, np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float64)

    return chunk_ids, np.vstack(vec_arrays), np.asarray(weights, dtype=np.float64)


def extract_chunk_embeddings(store: ConceptGraphStore) -> Tuple[List[str], np.ndarray]:
    """
    Extracts all Chunk node IDs and their stored vector embeddings.
    """
    cids, mat, _ = extract_chunk_embeddings_and_weights(store)
    return cids, mat


def ensure_graph_embeddings(
    store: ConceptGraphStore,
    embeddings: Optional[Any] = None,
    chunks_dir: Optional[Union[Path, str]] = None,
    console: Optional[Any] = None,
) -> None:
    """
    Ensures that concept ideas and chunk nodes in store have vector embeddings,
    backfilling from chunks_dir or embedding model if necessary.
    """
    _, g_mat = extract_concept_embeddings(store)
    _, u_mat = extract_chunk_embeddings(store)

    needs_ideas = len(g_mat) == 0
    needs_chunks = len(u_mat) == 0

    if (needs_ideas or needs_chunks) and (embeddings is not None or chunks_dir is not None):
        msg = "[bold cyan]Populating missing embeddings for graph space evaluation...[/bold cyan]"
        if console:
            console.print(msg)
        else:
            logger.info(msg)
        store.backfill_missing_embeddings(
            embeddings=embeddings,
            include_ideas=needs_ideas,
            include_chunks=needs_chunks,
            chunks_dir=chunks_dir,
            console=console,
        )


def run_graph_space_evaluation(
    store: ConceptGraphStore,
    universe_store: Optional[ConceptGraphStore] = None,
    tau: float = 0.75,
    embeddings: Optional[Any] = None,
    chunks_dir: Optional[Union[Path, str]] = None,
    console: Optional[Any] = None,
) -> GraphSpaceMetrics:
    """
    Executes a single Graph Space evaluation on a Knowledge Graph.
    """
    ensure_graph_embeddings(store, embeddings=embeddings, chunks_dir=chunks_dir, console=console)
    _, g_mat = extract_concept_embeddings(store)

    u_target = universe_store if universe_store is not None else store
    if universe_store is not None:
        ensure_graph_embeddings(universe_store, embeddings=embeddings, chunks_dir=chunks_dir, console=console)
    u_ids, u_mat, u_weights = extract_chunk_embeddings_and_weights(u_target)

    if len(u_mat) == 0:
        raise ValueError("Universe chunk embeddings are missing. Provide a graph with stored chunks or configure chunks_dir.")
    if len(g_mat) == 0:
        raise ValueError("Concept/Idea node embeddings are missing in knowledge graph. Run backfill or verify embedding model.")

    parent_cids = get_parent_chunk_ids(store)
    parent_mask = np.array([cid in parent_cids for cid in u_ids], dtype=bool) if parent_cids else None

    raw = evaluate_graph_space(
        universe_embeddings=u_mat,
        graph_embeddings=g_mat,
        tau=tau,
        parent_chunk_mask=parent_mask,
        chunk_weights=u_weights,
    )
    return GraphSpaceMetrics(**raw)


def run_graph_space_comparison(
    store_a: ConceptGraphStore,
    store_b: ConceptGraphStore,
    tau: float = 0.75,
    label_a: str = "Baseline (A)",
    label_b: str = "Candidate (B)",
    embeddings: Optional[Any] = None,
    chunks_dir: Optional[Union[Path, str]] = None,
    console: Optional[Any] = None,
) -> GraphSpaceComparisonReport:
    """
    Runs end-to-end Graph Space Comparison between Store A and Store B over their combined Universe.
    """
    ensure_graph_embeddings(store_a, embeddings=embeddings, chunks_dir=chunks_dir, console=console)
    ensure_graph_embeddings(store_b, embeddings=embeddings, chunks_dir=chunks_dir, console=console)

    _, g1_mat = extract_concept_embeddings(store_a)
    _, g2_mat = extract_concept_embeddings(store_b)

    cids_a, u1_mat, w_a = extract_chunk_embeddings_and_weights(store_a)
    cids_b, u2_mat, w_b = extract_chunk_embeddings_and_weights(store_b)

    if len(u1_mat) == 0 and len(u2_mat) > 0:
        u1_mat = u2_mat
        cids_a = cids_b
        w_a = w_b
    elif len(u2_mat) == 0 and len(u1_mat) > 0:
        u2_mat = u1_mat
        cids_b = cids_a
        w_b = w_a

    if len(u1_mat) == 0:
        raise ValueError("No chunk embeddings found in either Graph A or Graph B to construct universe.")
    if len(g1_mat) == 0:
        raise ValueError(f"No concept embeddings found in Baseline ({label_a}).")
    if len(g2_mat) == 0:
        raise ValueError(f"No concept embeddings found in Candidate ({label_b}).")

    parent_cids_a = get_parent_chunk_ids(store_a)
    parent_cids_b = get_parent_chunk_ids(store_b)

    parent_mask_a = np.array([cid in parent_cids_a for cid in cids_a], dtype=bool) if parent_cids_a else None
    parent_mask_b = np.array([cid in parent_cids_b for cid in cids_b], dtype=bool) if parent_cids_b else None

    return compare_graph_spaces(
        universe_embeddings=u1_mat,
        baseline_embeddings=g1_mat,
        candidate_embeddings=g2_mat,
        tau=tau,
        label_a=label_a,
        label_b=label_b,
        parent_mask_a=parent_mask_a,
        parent_mask_b=parent_mask_b,
        universe_embeddings_b=u2_mat,
        chunk_weights=w_a,
        chunk_weights_b=w_b,
    )


# ==============================================================================
# 5. Rich Console Table and Markdown Generators
# ==============================================================================


def format_graph_comparison_table(report: GraphSpaceComparisonReport) -> Table:
    """
    Renders the Comprehensive Metric Suite and Decision Matrix as a Rich Table.
    """
    if (
        report.universe_chunks_count_a
        and report.universe_chunks_count_b
        and report.universe_chunks_count_a != report.universe_chunks_count_b
    ):
        title = f"Knowledge Graph Embedding Space Comparison (τ = {report.tau:.2f}, N_A = {report.universe_chunks_count_a:,}, N_B = {report.universe_chunks_count_b:,} Chunks)"
    else:
        title = f"Knowledge Graph Embedding Space Comparison (τ = {report.tau:.2f}, N = {report.universe_chunks_count:,} Chunks)"

    tbl = Table(
        title=title,
        border_style="cyan",
    )
    tbl.add_column("Metric", style="bold")
    tbl.add_column(f"Baseline ({report.label_a})", justify="right", style="cyan")
    tbl.add_column(f"Candidate ({report.label_b})", justify="right", style="magenta")
    tbl.add_column("Delta (B - A)", justify="right")
    tbl.add_column("Operational Purpose / Goal", style="dim")

    m1 = report.baseline_metrics
    m2 = report.candidate_metrics

    def _fmt(val: Optional[float], is_pct: bool = False, decimals: int = 2) -> str:
        if val is None:
            return "-"
        return f"{val:.{decimals}f}%" if is_pct else f"{val:.{decimals}f}"

    def _delta_badge(
        delta: Optional[float],
        is_pct: bool = False,
        higher_is_better: bool = True,
        decimals: int = 2,
    ) -> str:
        if delta is None:
            return "[dim]-[/dim]"
        txt = f"{delta:+.{decimals}f}" + ("% pts" if is_pct else "")
        if abs(delta) < 1e-4:
            return f"[dim]{txt}[/dim]"
        color = "green" if (delta > 0 if higher_is_better else delta < 0) else "red"
        return f"[{color}]{txt}[/{color}]"

    # Core Metric Rows
    if m1.universe_chunks_count != m2.universe_chunks_count or (m1.universe_total_chars and m2.universe_total_chars and m1.universe_total_chars != m2.universe_total_chars):
        a_u_str = f"{m1.universe_chunks_count:,} ({m1.universe_total_chars/1_000_000:.1f}M chars)" if m1.universe_total_chars else f"{m1.universe_chunks_count:,}"
        b_u_str = f"{m2.universe_chunks_count:,} ({m2.universe_total_chars/1_000_000:.1f}M chars)" if m2.universe_total_chars else f"{m2.universe_chunks_count:,}"
        tbl.add_row(
            "Corpus Chunks Evaluated (|U|)",
            a_u_str,
            b_u_str,
            _delta_badge(m2.universe_chunks_count - m1.universe_chunks_count, decimals=0, higher_is_better=False),
            "Total chunks & character mass in corpus space",
        )
    tbl.add_row(
        "Idea / Concept Nodes (|G|)",
        f"{m1.nodes_count:,}",
        f"{m2.nodes_count:,}",
        _delta_badge(report.delta_nodes_count, decimals=0, higher_is_better=False),
        "Total extracted concept vertices",
    )
    tbl.add_row(
        f"Threshold Coverage (Cov_τ ≥ {report.tau:.2f})",
        _fmt(m1.coverage_at_tau_pct, True),
        _fmt(m2.coverage_at_tau_pct, True),
        _delta_badge(report.delta_coverage_at_tau_pct, is_pct=True, higher_is_better=True),
        "Text volume reachable within radius τ (Macro-coverage)",
    )
    tbl.add_row(
        "Chamfer Mean Coverage (μ)",
        _fmt(m1.chamfer_mean, decimals=4),
        _fmt(m2.chamfer_mean, decimals=4),
        _delta_badge(report.delta_chamfer_mean, decimals=4, higher_is_better=True),
        "Size-weighted expected semantic proximity",
    )
    tbl.add_row(
        "Coverage Std Dev (σ)",
        _fmt(m1.chamfer_std, decimals=4),
        _fmt(m2.chamfer_std, decimals=4),
        _delta_badge(report.delta_chamfer_std, decimals=4, higher_is_better=False),
        "Size-weighted regularity across text volume",
    )
    tbl.add_row(
        "Worst-Decile Proximity (p10)",
        _fmt(m1.worst_decile_p10, decimals=4),
        _fmt(m2.worst_decile_p10, decimals=4),
        _delta_badge(report.delta_worst_decile_p10, decimals=4, higher_is_better=True),
        "Tail risk: proximity depth of 10% hardest text mass",
    )
    tbl.add_row(
        "Median Proximity (p50)",
        _fmt(m1.median_p50, decimals=4),
        _fmt(m2.median_p50, decimals=4),
        _delta_badge(report.delta_median_p50, decimals=4, higher_is_better=True),
        "Size-weighted median proximity of corpus text",
    )
    tbl.add_row(
        "Directed Hausdorff Gap (dH)",
        _fmt(m1.worst_case_gap, decimals=4),
        _fmt(m2.worst_case_gap, decimals=4),
        _delta_badge(report.delta_worst_case_gap, decimals=4, higher_is_better=False),
        "Maximum blind spot (1 - min cos): isolated voids",
    )
    tbl.add_row(
        "Vendi Diversity Score (VS)",
        _fmt(m1.vendi_diversity_score, decimals=1),
        _fmt(m2.vendi_diversity_score, decimals=1),
        _delta_badge(report.delta_vendi_diversity_score, decimals=1, higher_is_better=True),
        "Effective non-redundant orthogonal concept modes",
    )
    tbl.add_row(
        "Compression Efficiency (Cov/Node)",
        _fmt(m1.coverage_per_node_efficiency, decimals=4),
        _fmt(m2.coverage_per_node_efficiency, decimals=4),
        _delta_badge(report.delta_coverage_per_node_efficiency, decimals=4, higher_is_better=True),
        "Corpus space covered per graph node retained",
    )
    tbl.add_row(
        "Concept Quality (Vendi/Node)",
        _fmt(m1.vendi_per_node, decimals=3),
        _fmt(m2.vendi_per_node, decimals=3),
        _delta_badge(report.delta_vendi_per_node, decimals=3, higher_is_better=True),
        "Information density: higher implies less duplicate redundancy",
    )

    # Skipping Leverage Metrics Section
    if m1.parent_chunk_ratio_pct is not None or m2.parent_chunk_ratio_pct is not None:
        tbl.add_section()
        tbl.add_row(
            "Extraction Ratio (ρ_parent)",
            _fmt(m1.parent_chunk_ratio_pct, True),
            _fmt(m2.parent_chunk_ratio_pct, True),
            _delta_badge(report.delta_parent_chunk_ratio_pct, is_pct=True, higher_is_better=False),
            "Fraction of corpus text volume compute was spent on",
        )
        tbl.add_row(
            "Skipped Retention (R_skipped)",
            _fmt(m1.skipped_retention_pct, True),
            _fmt(m2.skipped_retention_pct, True),
            _delta_badge(report.delta_skipped_retention_pct, is_pct=True, higher_is_better=True),
            "Bypassed text volume retained within radius τ",
        )
        tbl.add_row(
            "Generalization Surplus (Δ_cov)",
            _fmt(m1.generalization_surplus_pct, True),
            _fmt(m2.generalization_surplus_pct, True),
            _delta_badge(report.delta_generalization_surplus_pct, is_pct=True, higher_is_better=True),
            "Excess text coverage gained over extraction ratio",
        )
        tbl.add_row(
            "Compression Leverage (L_comp)",
            f"{m1.compression_leverage_factor:.2f}x" if m1.compression_leverage_factor is not None else "-",
            f"{m2.compression_leverage_factor:.2f}x" if m2.compression_leverage_factor is not None else "-",
            _delta_badge(report.delta_compression_leverage_factor, decimals=2, higher_is_better=True),
            "Coverage multiplier per unit compute (Cov_τ / ρ_parent)",
        )

    tbl.add_section()
    # Decision Matrix Checklist Rows
    badge = "[bold green]PASS[/bold green]" if report.decision_pass else "[bold red]FAIL[/bold red]"
    tbl.add_row("Practical Decision Checklist", "-", badge, f"[dim]{report.pass_reason}[/dim]", "")
    for k, gate in report.gate_checklist.items():
        g_badge = "[green]✓ PASS[/green]" if gate["passed"] else "[red]✗ FAIL[/red]"
        tbl.add_row(f"  • {gate['name']}", "-", g_badge, f"[dim]{gate['details']}[/dim]", gate["threshold"])

    return tbl


def format_graph_space_table(metrics: GraphSpaceMetrics, title: str = "Knowledge Graph Embedding Space Metrics") -> Table:
    """
    Renders single graph metrics as a Rich Table.
    """
    tbl = Table(title=title, border_style="cyan")
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right", style="cyan")
    tbl.add_column("Operational Meaning / Purpose", style="dim")

    tbl.add_row("Idea Nodes Count (|G|)", f"{metrics.nodes_count:,}", "Total concept vertices in knowledge graph")
    tbl.add_row("Universe Chunks Count (|U|)", f"{metrics.universe_chunks_count:,}", "Total raw literature chunks evaluated")
    tbl.add_row(f"Threshold Coverage (Cov_τ ≥ {metrics.tau:.2f})", f"{metrics.coverage_at_tau_pct:.2f}%", "Percentage of corpus reachable within semantic radius τ")
    tbl.add_row("Chamfer Mean Coverage (μ)", f"{metrics.chamfer_mean:.4f}", "Expected cosine proximity across arbitrary chunks")
    tbl.add_row("Coverage Std Dev (σ)", f"{metrics.chamfer_std:.4f}", "Regularity: uniformity of coverage across topics")
    tbl.add_row("Worst-Decile Proximity (p10)", f"{metrics.worst_decile_p10:.4f}", "Tail risk: proximity of bottom 10% hardest chunks")
    tbl.add_row("Median Proximity (p50)", f"{metrics.median_p50:.4f}", "Robust central tendency of graph coverage")
    tbl.add_row("Directed Hausdorff Gap (dH)", f"{metrics.worst_case_gap:.4f}", "Maximum blind spot distance to most isolated chunk")
    tbl.add_row("Vendi Diversity Score (VS)", f"{metrics.vendi_diversity_score:.2f}", "Effective orthogonal concept directions spanned")
    tbl.add_row("Compression Efficiency (Cov/Node)", f"{metrics.coverage_per_node_efficiency:.4f}", "Corpus space covered per graph node retained")
    tbl.add_row("Concept Quality (Vendi/Node)", f"{metrics.vendi_per_node:.4f}", "Information density per node")

    if metrics.parent_chunk_ratio_pct is not None:
        tbl.add_section()
        if metrics.parent_chunks_count is not None:
            tbl.add_row("Parent Chunks Count (Np)", f"{metrics.parent_chunks_count:,}", "Chunks from which ideas were extracted")
        if metrics.skipped_chunks_count is not None:
            tbl.add_row("Skipped Chunks Count (Ns)", f"{metrics.skipped_chunks_count:,}", "Chunks bypassed by skipping heuristic")
        tbl.add_row("Extraction Ratio (ρ_parent)", f"{metrics.parent_chunk_ratio_pct:.2f}%", "Fraction of corpus compute spent on (Np / N)")
        if metrics.skipped_retention_pct is not None:
            tbl.add_row("Skipped Retention (R_skipped)", f"{metrics.skipped_retention_pct:.2f}%", "Percentage of bypassed chunks reachable within radius τ")
        if metrics.generalization_surplus_pct is not None:
            tbl.add_row("Generalization Surplus (Δ_cov)", f"{metrics.generalization_surplus_pct:+.2f}% pts", "Coverage gained beyond parent ratio (Cov_τ - ρ_parent)")
        if metrics.compression_leverage_factor is not None:
            tbl.add_row("Compression Leverage (L_comp)", f"{metrics.compression_leverage_factor:.2f}x", "Multiplier of concept coverage per compute spent (Cov_τ / ρ_parent)")

    return tbl
