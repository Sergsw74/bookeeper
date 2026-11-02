"""
Unit tests for the hybrid math formula feature vector and block-normalized
composite embedding architecture (Option A: Pure Standard Cosine).
"""

import numpy as np
import pytest

from bookeeper.processing.formula_embedding import (
    CANONICAL_OPERATORS,
    OPERATOR_IDF_WEIGHTS,
    ENGLISH_MATH_KEYWORD_MAP,
    build_formula_feature_vector,
    build_composite_embedding,
    build_composite_embeddings_matrix,
    compute_adaptive_cosine,
)
from bookeeper.graph.store import ConceptGraphStore, Concept
from bookeeper.processing.graph_evaluator import (
    extract_concept_embeddings,
    extract_chunk_embeddings_and_weights,
    extract_chunk_embeddings,
    evaluate_graph_space,
    compute_max_cosine_sim_batched,
)


def test_formula_feature_vector_dimensions_and_norm():
    """Verify formula feature vector has dimension m and non-zero entries."""
    m = 128
    formula_text = r"\mathcal{L}(\theta) = \mathbb{E}_{x \sim p}[ \log f_\theta(x) ] + \frac{\lambda}{2} \|\theta\|^2"
    vec = build_formula_feature_vector(formula_text, m=m)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (m,)
    assert vec.dtype == np.float32
    # Verify non-zero norm and normalize
    norm = float(np.linalg.norm(vec))
    assert norm > 0.0
    vec_norm = vec / norm
    assert pytest.approx(float(np.linalg.norm(vec_norm)), abs=1e-5) == 1.0


def test_formula_feature_vector_empty_input():
    """Verify empty text produces all-zero vector."""
    m = 128
    vec = build_formula_feature_vector("", m=m)
    assert vec.shape == (m,)
    assert np.all(vec == 0.0)


def test_structural_operator_idf_weighting():
    """Verify dims 0..31 reflect canonical operator occurrences with IDF weights."""
    m = 128
    # Formula with expectation \mathbb{E} and gradient \nabla
    formula = r"\nabla_\theta \mathbb{E}[R(\tau)] = 0"
    vec = build_formula_feature_vector(formula, m=m)

    # Check that canonical operator indices for \nabla and \mathbb{E} are active
    idx_nabla = CANONICAL_OPERATORS.index(r"\nabla")
    idx_exp = CANONICAL_OPERATORS.index(r"\mathbb{E}")

    assert vec[idx_nabla] > 0.0
    assert vec[idx_exp] > 0.0

    # Ensure rare operators carry their respective IDF scale
    assert idx_nabla < 32
    assert idx_exp < 32


def test_ast_transition_order_discrimination():
    """
    Verify dims 32..63 (AST 2-gram transition buckets) differentiate
    log sum exp from sum log exp.
    """
    m = 128
    # Two expressions with the exact same bag of operators, but inverted order
    expr1 = r"\log \sum \exp(x_i)"
    expr2 = r"\sum \log \exp(x_i)"

    vec1 = build_formula_feature_vector(expr1, m=m)
    vec2 = build_formula_feature_vector(expr2, m=m)

    # Dims 0..31 (bag-of-operators) should be identical because same operators are used
    np.testing.assert_allclose(vec1[:32], vec2[:32], atol=1e-5)

    # Dims 32..63 (2-gram transitions \log->\sum vs \sum->\log) must differ
    assert not np.allclose(vec1[32:64], vec2[32:64], atol=1e-4)


def test_variable_and_greek_hashing():
    """Verify dims 64..127 contain deterministic hash bucket activations for variables."""
    m = 128
    expr = r"\theta_{t+1} = \theta_t + \alpha \nabla J(\theta_t)"
    vec = build_formula_feature_vector(expr, m=m)

    # Greek letter \theta and \alpha must activate hashed buckets in dims 64..127
    var_subspace = vec[64:128]
    assert np.any(var_subspace > 0.0)


def test_bilingual_english_latex_alignment():
    """
    Verify English mathematical keywords map to the same vector subspaces
    as their canonical LaTeX representations.
    """
    m = 128
    # English description
    english_desc = "Compute the expectation of gradient loss with respect to parameter theta and summation"
    # Corresponding raw LaTeX
    latex_formula = r"\mathbb{E}[ \nabla_\theta \mathcal{L} ] + \sum"

    v_en = build_formula_feature_vector(english_desc, m=m)
    v_lt = build_formula_feature_vector(latex_formula, m=m)

    # The cosine similarity between the English description's math vector
    # and the LaTeX formula's math vector should be strongly positive
    cos_sim = float(np.dot(v_en, v_lt))
    assert cos_sim > 0.45, f"Expected bilingual similarity > 0.45, got {cos_sim:.3f}"


def test_composite_embedding_unit_norm_invariant():
    """
    Verify Option A block-normalized composite vector has exact unit L2 norm
    for any valid alpha in [0, 1].
    """
    d = 768
    m = 128
    rng = np.random.default_rng(42)

    text_vec = rng.standard_normal(d, dtype=np.float32)
    math_vec = rng.standard_normal(m, dtype=np.float32)

    for alpha in [0.0, 0.1, 0.35, 0.5, 0.8, 1.0]:
        z = build_composite_embedding(
            text_vec=text_vec,
            math_vec=math_vec,
            has_formula=True,
            alpha=alpha,
            m=m,
        )
        assert z.shape == (d + m,)
        norm_z = float(np.linalg.norm(z))
        assert pytest.approx(norm_z, abs=1e-5) == 1.0


def test_composite_embedding_graceful_degradation():
    """
    When has_formula=False, math subspace is all zeros and composite vector
    degrades cleanly to pure text embedding.
    """
    d = 384
    m = 128
    rng = np.random.default_rng(123)

    text_vec = rng.standard_normal(d, dtype=np.float32)
    math_vec = rng.standard_normal(m, dtype=np.float32)

    z = build_composite_embedding(
        text_vec=text_vec,
        math_vec=math_vec,
        has_formula=False,
        alpha=0.35,
        m=m,
    )

    # First d dimensions must equal L2-normalized text_vec
    norm_text = text_vec / np.linalg.norm(text_vec)
    np.testing.assert_allclose(z[:d], norm_text, atol=1e-5)

    # Remaining m dimensions must be strictly 0
    assert np.all(z[d:] == 0.0)

    # Norm must remain exactly 1.0
    assert pytest.approx(float(np.linalg.norm(z)), abs=1e-5) == 1.0


def test_composite_matrix_builder():
    """Verify matrix builder creates correctly shaped (N, d+m) unit-norm array."""
    N = 10
    d = 64
    m = 32
    rng = np.random.default_rng(99)

    text_mat = rng.standard_normal((N, d), dtype=np.float32)
    math_mat = rng.standard_normal((N, m), dtype=np.float32)
    has_formula_flags = np.array([bool(i % 2 == 0) for i in range(N)], dtype=bool)

    composite_mat = build_composite_embeddings_matrix(
        text_matrix=text_mat,
        has_formula_mask=has_formula_flags,
        math_vectors=math_mat,
        alpha=0.35,
        m=m,
    )

    assert composite_mat.shape == (N, d + m)
    # Every row must be unit-norm
    row_norms = np.linalg.norm(composite_mat, axis=1)
    np.testing.assert_allclose(row_norms, np.ones(N), atol=1e-5)

    # Odd indices had has_formula=False -> math subspace should be 0
    for i in range(N):
        if not has_formula_flags[i]:
            assert np.all(composite_mat[i, d:] == 0.0)


def test_standard_cosine_dot_product_equivalence():
    """
    Verify that standard dot product of two composite vectors equals
    (1 - alpha) * cos(text1, text2) + alpha * cos(math1, math2).
    """
    d = 128
    m = 64
    alpha = 0.35
    rng = np.random.default_rng(7)

    t1 = rng.standard_normal(d, dtype=np.float32)
    t2 = rng.standard_normal(d, dtype=np.float32)
    m1 = rng.standard_normal(m, dtype=np.float32)
    m2 = rng.standard_normal(m, dtype=np.float32)

    # Normalization
    t1_norm = t1 / np.linalg.norm(t1)
    t2_norm = t2 / np.linalg.norm(t2)
    m1_norm = m1 / np.linalg.norm(m1)
    m2_norm = m2 / np.linalg.norm(m2)

    z1 = build_composite_embedding(t1, m1, has_formula=True, alpha=alpha, m=m)
    z2 = build_composite_embedding(t2, m2, has_formula=True, alpha=alpha, m=m)

    dot_direct = float(np.dot(z1, z2))
    expected_dot = float((1 - alpha) * np.dot(t1_norm, t2_norm) + alpha * np.dot(m1_norm, m2_norm))

    assert pytest.approx(dot_direct, abs=1e-5) == expected_dot


def test_hybrid_graph_evaluation_integration():
    """Verify hybrid_math evaluation in graph space with Concept and Formula nodes."""
    store = ConceptGraphStore()
    d = 64
    rng = np.random.default_rng(42)

    # Add a normal prose concept (unit-normalized)
    raw_emb1 = rng.standard_normal(d, dtype=np.float32)
    emb1 = (raw_emb1 / np.linalg.norm(raw_emb1)).tolist()
    store.add_concept(
        Concept(
            name="Attention Mechanism",
            category="Architecture",
            brief_description="Transformer self-attention maps query and key pairs to attention weights.",
            has_formula=False,
        ),
        embedding=emb1,
    )

    # Add a math concept with has_formula=True (unit-normalized)
    raw_emb2 = rng.standard_normal(d, dtype=np.float32)
    emb2 = (raw_emb2 / np.linalg.norm(raw_emb2)).tolist()
    store.add_concept(
        Concept(
            name="Policy Gradient Theorem",
            category="Theory",
            brief_description=r"The analytical gradient \nabla_\theta J(\theta) = \mathbb{E}[ \nabla \log \pi_\theta(a|s) Q(s,a) ]",
            has_formula=True,
        ),
        embedding=emb2,
    )

    # Add a dedicated Formula node and connect to the Policy Gradient concept
    formula_id = store.add_formula(
        {
            "raw_latex": r"\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[ \sum_t \nabla \log \pi_\theta(a_t|s_t) G_t ]",
            "label": "REINFORCE gradient formula",
            "supporting_quote": "REINFORCE objective gradient expectation formula",
        },
        concept_name="Policy Gradient Theorem",
    )
    assert formula_id.startswith("formula:")

    # Extract concept embeddings with hybrid_math=True
    names_hybrid, mat_hybrid = extract_concept_embeddings(store, hybrid_math=True, alpha=0.35, formula_dim=128)
    assert len(names_hybrid) == 2
    assert mat_hybrid.shape == (2, d + 128)
    for c_vec in mat_hybrid:
        assert pytest.approx(float(np.linalg.norm(c_vec)), abs=1e-5) == 1.0

    # Extract concept embeddings with hybrid_math=False
    names_pure, mat_pure = extract_concept_embeddings(store, hybrid_math=False)
    assert len(names_pure) == 2
    assert mat_pure.shape == (2, d)
    for c_vec in mat_pure:
        assert pytest.approx(float(np.linalg.norm(c_vec)), abs=1e-5) == 1.0


def test_adaptive_cosine_unpenalized_when_concept_lacks_math():
    """
    Verify that when a chunk has a formula (z1 with active math subspace)
    and an idea concept lacks a formula (z2 with zero math subspace),
    compute_adaptive_cosine returns the unpenalized pure text similarity,
    completely eliminating the artificial sqrt(1 - alpha) penalty.
    """
    d = 128
    m = 64
    alpha = 0.35
    rng = np.random.default_rng(101)

    t1 = rng.standard_normal(d, dtype=np.float32)
    t2 = t1 + 0.1 * rng.standard_normal(d, dtype=np.float32)  # Highly similar text
    m1 = rng.standard_normal(m, dtype=np.float32)

    # Unit normalization of text
    t1_hat = t1 / np.linalg.norm(t1)
    t2_hat = t2 / np.linalg.norm(t2)
    true_text_sim = float(np.dot(t1_hat, t2_hat))

    # Chunk z1 has formula, Idea z2 has NO formula
    z1 = build_composite_embedding(t1, m1, has_formula=True, alpha=alpha, m=m)
    z2 = build_composite_embedding(t2, None, has_formula=False, alpha=alpha, m=m)

    # Naive dot product suffers from sqrt(1 - alpha) ~ 0.806 penalty
    naive_dot = float(np.dot(z1, z2))
    assert naive_dot < true_text_sim
    assert pytest.approx(naive_dot, abs=1e-3) == true_text_sim * np.sqrt(1.0 - alpha)

    # Adaptive similarity recovers the full unpenalized text similarity
    adapt_sim = compute_adaptive_cosine(z1, z2, formula_dim=m, alpha=alpha)
    assert pytest.approx(adapt_sim, abs=1e-5) == true_text_sim


def test_adaptive_batched_projection_preserves_text_and_boosts_math():
    """
    Verify compute_max_cosine_sim_batched with formula_dim > 0 matches adaptive similarity
    and guarantees no formula-bearing chunk is penalized against text-only concepts.
    """
    N = 5
    M = 8
    d = 64
    m = 32
    alpha = 0.35
    rng = np.random.default_rng(202)

    t_chunks = rng.standard_normal((N, d), dtype=np.float32)
    m_chunks = rng.standard_normal((N, m), dtype=np.float32)
    t_concepts = rng.standard_normal((M, d), dtype=np.float32)
    m_concepts = rng.standard_normal((M, m), dtype=np.float32)

    # Normalize texts
    t_chunks_hat = t_chunks / np.linalg.norm(t_chunks, axis=1, keepdims=True)
    t_concepts_hat = t_concepts / np.linalg.norm(t_concepts, axis=1, keepdims=True)

    # Chunks: rows 0, 1, 2 have formulas; rows 3, 4 do not
    u_mask = np.array([True, True, True, False, False], dtype=bool)
    # Concepts: rows 0, 1 have formulas; rows 2..7 do not
    g_mask = np.array([True, True, False, False, False, False, False, False], dtype=bool)

    U = build_composite_embeddings_matrix(t_chunks, u_mask, m_chunks, alpha=alpha, m=m)
    G = build_composite_embeddings_matrix(t_concepts, g_mask, m_concepts, alpha=alpha, m=m)

    sims_batched = compute_max_cosine_sim_batched(U, G, formula_dim=m, alpha=alpha)
    assert len(sims_batched) == N

    # Calculate expected max adaptive sim individually
    for i in range(N):
        best_expected = max(compute_adaptive_cosine(U[i], G[j], formula_dim=m, alpha=alpha) for j in range(M))
        assert pytest.approx(sims_batched[i], abs=1e-5) == best_expected
