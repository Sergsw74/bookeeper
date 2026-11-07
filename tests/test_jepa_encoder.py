"""
Unit tests for LightweightJEPAEncoder.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from bookeeper.evaluation.jepa_encoder import LightweightJEPAEncoder


def test_jepa_encoder_initialization():
    encoder = LightweightJEPAEncoder(latent_dim=128, epsilon=1e-3, align_alpha=0.5)
    assert encoder.latent_dim == 128
    assert encoder.epsilon == 1e-3
    assert encoder.align_alpha == 0.5
    assert not encoder.is_fitted


def test_jepa_encoder_fit_and_project():
    rng = np.random.RandomState(42)
    # Generate anisotropic synthetic embeddings (e.g. mean offset)
    X = rng.randn(100, 64).astype(np.float32) + 3.0

    encoder = LightweightJEPAEncoder(latent_dim=32)
    encoder.fit_projector(X)
    assert encoder.is_fitted

    Z = encoder.project(X)
    assert Z.shape == (100, 32)

    # Check unit-sphere normalization: ||z||_2 == 1.0
    norms = np.linalg.norm(Z, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # Check that unrelated pairs have near-zero cosine similarity
    # Covariance of Z should be approximately diagonal
    Z_cov = (Z.T @ Z) / len(Z)
    off_diag = Z_cov - np.diag(np.diag(Z_cov))
    assert np.mean(np.abs(off_diag)) < 0.1

    # Pairwise cosine between non-identical pairs should center around 0.0
    sim_matrix = Z @ Z.T
    mask = ~np.eye(len(Z), dtype=bool)
    assert abs(np.mean(sim_matrix[mask])) < 0.05


def test_jepa_encoder_predictive_alignment():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 32).astype(np.float32) + 2.0
    # Construct synthetic predictive pairs with shared features
    pairs = [(i, i + 1) for i in range(0, 30, 2)]
    for i, j in pairs:
        shared_signal = rng.randn(12).astype(np.float32) * 2.5
        X[i, :12] += shared_signal
        X[j, :12] += shared_signal

    encoder = LightweightJEPAEncoder(latent_dim=16, align_alpha=0.6)
    encoder.fit_projector(X, positive_pairs=pairs)

    Z = encoder.project(X)
    paired_sims = [float(np.dot(Z[i], Z[j])) for i, j in pairs]
    unpaired_sims = [float(np.dot(Z[i], Z[j])) for i in range(20) for j in range(20, 40) if abs(i - j) > 1]

    assert np.mean(paired_sims) > np.mean(unpaired_sims)
    assert np.mean(paired_sims) > 0.5


def test_jepa_encoder_save_and_load(tmp_path: Path):
    rng = np.random.RandomState(42)
    X = rng.randn(40, 32).astype(np.float32)

    encoder = LightweightJEPAEncoder(latent_dim=16)
    encoder.fit_projector(X)

    save_path = tmp_path / "jepa_test.npz"
    encoder.save(save_path)
    assert save_path.is_file()

    loaded = LightweightJEPAEncoder.load(save_path)
    assert loaded.is_fitted
    assert loaded.latent_dim == 16

    z_orig = encoder.project(X[:5])
    z_loaded = loaded.project(X[:5])
    np.testing.assert_allclose(z_orig, z_loaded, atol=1e-6)


def test_jepa_encoder_text_encoding():
    encoder = LightweightJEPAEncoder(latent_dim=64)
    v1 = encoder.encode_text("Joint Embedding Predictive Architecture")
    v2 = encoder.encode_text("Joint Embedding Predictive Architecture")
    v3 = encoder.encode_text("Completely unrelated astrophysical phenomenon")

    assert v1.shape == (64,)
    # Self similarity is 1.0
    sim_same = float(np.dot(v1, v2))
    assert pytest.approx(sim_same, abs=1e-5) == 1.0

    # Unrelated similarity is lower
    sim_diff = float(np.dot(v1, v3))
    assert sim_diff < 0.5


def test_jepa_encoder_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    c = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert pytest.approx(float(LightweightJEPAEncoder.cosine_similarity(a, b)), abs=1e-6) == 0.0
    assert pytest.approx(float(LightweightJEPAEncoder.cosine_similarity(a, c)), abs=1e-6) == 1.0
