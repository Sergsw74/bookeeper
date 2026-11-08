"""
Tests for JEPAEmbeddings drop-in replacement and 768-d configuration.
"""

from pathlib import Path
import numpy as np
import pytest

from bookeeper.config import Settings, JEPAConfig
from bookeeper.evaluation.jepa_encoder import LightweightJEPAEncoder, LightweightJEPANetwork
from bookeeper.evaluation.jepa_embeddings import JEPAEmbeddings
from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings
from bookeeper.graph.embedding_codec import decode_embedding, encode_embedding


def test_jepa_config_defaults():
    cfg = Settings(
        embedding_provider="jepa",
        embedding_model="jepa-768",
        embedding_dim=768,
        jepa=JEPAConfig(enabled=True, dimension=768),
        similarity_threshold=0.45,
        high_similarity_threshold=0.70,
    )
    assert cfg.embedding_provider == "jepa"
    assert cfg.embedding_model == "jepa-768"
    assert cfg.embedding_dim == 768
    assert cfg.jepa.enabled is True
    assert cfg.jepa.dimension == 768
    assert cfg.jepa.device == "gpu"
    assert cfg.similarity_threshold == 0.45
    assert cfg.high_similarity_threshold == 0.70


def test_jepa_embeddings_direct_network():
    net = LightweightJEPANetwork(vocab_size=1000, hidden_dim=64, latent_dim=768)
    jepa_emb = JEPAEmbeddings(
        dimension=768,
        model_type="direct_network",
        neural_net=net,
    )
    texts = [
        "First document text discussing zero-isotropic embeddings.",
        "Second document text regarding knowledge graph relations.",
    ]
    vectors = jepa_emb.embed_documents(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 768
    assert len(vectors[1]) == 768

    # Check unit normalization
    v0 = np.array(vectors[0])
    assert np.isclose(np.linalg.norm(v0), 1.0, atol=1e-4)

    # Test embed_query
    q_vec = jepa_emb.embed_query("Search query")
    assert len(q_vec) == 768
    assert np.isclose(np.linalg.norm(q_vec), 1.0, atol=1e-4)

    # Test stats
    stats = jepa_emb.embedding_stats
    assert stats["total_texts"] == 3
    assert stats["dimension"] == 768
    assert "device" in stats


def test_jepa_embeddings_projector(tmp_path):
    # Create synthetic base embeddings and fit encoder
    np.random.seed(42)
    base_data = np.random.randn(50, 1024).astype(np.float32)
    encoder = LightweightJEPAEncoder(latent_dim=768, epsilon=1e-3)
    encoder.fit_projector(base_data)

    weights_file = tmp_path / "test_jepa_768.npz"
    encoder.save(weights_file)

    jepa_emb = JEPAEmbeddings(
        dimension=768,
        model_type="whitened_projector",
        weights_path=weights_file,
        encoder=encoder,
    )

    # Project raw vectors
    proj_vecs = jepa_emb.project_vectors(base_data[:5])
    assert proj_vecs.shape == (5, 768)
    norms = np.linalg.norm(proj_vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_failover_ollama_embeddings_factory_routing():
    cfg = Settings(
        embedding_provider="jepa",
        embedding_model="jepa-768",
        embedding_dim=768,
        jepa=JEPAConfig(enabled=True, dimension=768, model_type="direct_network"),
    )
    emb = FailoverOllamaEmbeddings.from_settings(cfg)
    assert isinstance(emb, JEPAEmbeddings)
    assert emb.dimension == 768


def test_decode_embedding_with_list_and_array():
    raw_list = [0.1, 0.2, 0.3, 0.4]
    decoded = decode_embedding(raw_list)
    assert isinstance(decoded, np.ndarray)
    assert np.allclose(decoded, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

    raw_arr = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    decoded_arr = decode_embedding(raw_arr)
    assert isinstance(decoded_arr, np.ndarray)
    assert np.allclose(decoded_arr, raw_arr)
