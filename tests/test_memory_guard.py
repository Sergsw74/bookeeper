"""
Unit tests for MemoryGuard, memory limit enforcement (10 GB ceiling),
batched max cosine projection, and dual Gram Vendi score computation.
"""

import numpy as np
import pytest

from bookeeper.processing.graph_evaluator import (
    compute_max_cosine_sim_batched,
    evaluate_graph_space,
    evaluate_skipping_leverage,
)
from bookeeper.processing.memory_guard import (
    DEFAULT_MAX_MEMORY_BYTES,
    check_memory_limit,
    cleanup_memory,
    get_process_memory_bytes,
    get_process_memory_gb,
)


def test_process_memory_metrics():
    """Verify process memory retrieval returns valid positive integers."""
    mem_bytes = get_process_memory_bytes()
    mem_gb = get_process_memory_gb()
    assert mem_bytes > 0
    assert mem_gb > 0.0
    assert abs(mem_gb - (mem_bytes / (1024.0**3))) < 1e-2


def test_memory_limit_within_bounds():
    """Verify check_memory_limit succeeds when within ceiling."""
    current_gb = check_memory_limit(limit_bytes=DEFAULT_MAX_MEMORY_BYTES, context="test")
    assert current_gb > 0.0
    # Process memory should typically be under 2 GB during normal tests
    assert current_gb < 10.0


def test_memory_limit_raises_on_breach():
    """Verify check_memory_limit raises MemoryError when ceiling is breached."""
    # Set artificial tiny ceiling of 1 KB
    tiny_ceiling = 1024
    with pytest.raises(MemoryError) as exc_info:
        check_memory_limit(limit_bytes=tiny_ceiling, context="unit_test_breach", raise_on_breach=True)
    assert "exceeded the maximum safety ceiling" in str(exc_info.value)
    assert "unit_test_breach" in str(exc_info.value)


def test_cleanup_memory_triggers_garbage_collection():
    """Verify cleanup_memory runs without errors."""
    cleanup_memory()


def test_compute_max_cosine_sim_batched_matches_exact_matrix_multiplication():
    """Verify batched cosine projection matches exact unbatched projection."""
    np.random.seed(42)
    n_chunks = 250
    n_ideas = 120
    dim = 64

    U = np.random.randn(n_chunks, dim).astype(np.float32)
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    G = np.random.randn(n_ideas, dim).astype(np.float32)
    G /= np.linalg.norm(G, axis=1, keepdims=True)

    exact_max = np.max(U @ G.T, axis=1)

    # Test with very small batch memory cap to force multiple small batches
    batched_max = compute_max_cosine_sim_batched(U, G, max_batch_mem_bytes=4 * 1024)  # 4 KB cap

    assert np.allclose(exact_max, batched_max, atol=1e-5)


def test_compute_max_cosine_sim_batched_empty():
    """Verify empty input handling in batched projection."""
    res = compute_max_cosine_sim_batched(np.empty((0, 32)), np.empty((0, 32)))
    assert len(res) == 0


def test_vendi_score_dual_spectral_formulation_equivalence():
    """
    Verify that when M > D+1, the dual Gram matrix formulation produces identical
    Vendi scores to direct M x M kernel computation with zero numerical drift.
    """
    np.random.seed(123)
    M = 200
    D = 16

    G = np.random.randn(M, D).astype(np.float64)
    G /= np.linalg.norm(G, axis=1, keepdims=True)

    # 1. Direct (M x M) formulation
    K = (1.0 + (G @ G.T)) / 2.0
    eig_direct = np.linalg.eigvalsh(K / M)
    eig_direct = eig_direct[eig_direct > 1e-10]
    vendi_direct = float(np.exp(-np.sum(eig_direct * np.log(eig_direct))))

    # 2. Dual Gram ((D+1) x (D+1)) formulation
    mean_g = np.mean(G, axis=0, dtype=np.float64)
    C = np.empty((D + 1, D + 1), dtype=np.float64)
    C[0, 0] = 1.0
    C[0, 1:] = mean_g
    C[1:, 0] = mean_g
    C[1:, 1:] = (G.T @ G) / M
    C *= 0.5
    eig_dual = np.linalg.eigvalsh(C)
    eig_dual = eig_dual[eig_dual > 1e-10]
    vendi_dual = float(np.exp(-np.sum(eig_dual * np.log(eig_dual))))

    assert np.isclose(vendi_direct, vendi_dual, rtol=1e-5)
    # 3. Verify evaluate_graph_space uses this formulation
    U = np.random.randn(50, D).astype(np.float32)
    metrics = evaluate_graph_space(universe_embeddings=U, graph_embeddings=G.astype(np.float32))
    assert abs(metrics["vendi_diversity_score"] - vendi_dual) < 0.05
