"""
Unit tests for embedding compression and Base64 binary codec.
"""

import numpy as np
import pytest

from bookeeper.graph.embedding_codec import (
    compare_vector_fidelity,
    compress_node_embedding,
    decode_embedding,
    decompress_node_embedding,
    encode_embedding,
    get_node_embedding,
)


def test_encode_and_decode_float16_roundtrip():
    # 768-dimensional random unit vector
    rng = np.random.default_rng(42)
    original = rng.standard_normal(768).astype(np.float32)
    original /= np.linalg.norm(original)

    # Encode with float16
    encoded = encode_embedding(original, dtype="float16")
    assert "embedding_b64" in encoded
    assert encoded["embedding_dim"] == 768
    assert encoded["embedding_dtype"] == "float16"

    # Decode
    decoded = decode_embedding(encoded, return_numpy=True)
    assert decoded.shape == (768,)
    assert decoded.dtype == np.float32

    # Verify high fidelity (> 0.9999 cosine similarity)
    cos_sim, max_err = compare_vector_fidelity(original, decoded)
    assert cos_sim > 0.9999
    assert max_err < 0.005  # Float16 precision


def test_encode_and_decode_float32_exact():
    rng = np.random.default_rng(123)
    original = rng.standard_normal(256).astype(np.float32)

    encoded = encode_embedding(original, dtype="float32")
    assert encoded["embedding_dtype"] == "float32"

    decoded = decode_embedding(encoded, return_numpy=True)
    np.testing.assert_allclose(original, decoded, rtol=1e-6, atol=1e-6)


def test_compress_and_decompress_node_embedding():
    raw_vec = [0.1, -0.25, 0.75, -0.99]
    node = {
        "id": "concept:ai",
        "type": "Concept",
        "name": "Artificial Intelligence",
        "embedding": raw_vec,
    }

    compressed = compress_node_embedding(node, dtype="float16")
    assert "embedding" not in compressed
    assert "embedding_b64" in compressed
    assert compressed["embedding_dim"] == 4
    assert compressed["embedding_dtype"] == "float16"

    # Access via universal getter
    got_vec = get_node_embedding(compressed, return_numpy=False)
    assert len(got_vec) == 4
    np.testing.assert_allclose(got_vec, raw_vec, atol=1e-3)

    # Decompress back to list
    decompressed = decompress_node_embedding(compressed, as_list=True)
    assert "embedding" in decompressed
    assert isinstance(decompressed["embedding"], list)
    np.testing.assert_allclose(decompressed["embedding"], raw_vec, atol=1e-3)


def test_get_node_embedding_handles_legacy_and_empty():
    # Legacy list
    n1 = {"embedding": [0.5, 0.5]}
    v1 = get_node_embedding(n1, return_numpy=True)
    assert isinstance(v1, np.ndarray)
    np.testing.assert_allclose(v1, [0.5, 0.5])

    # Pre-existing numpy array
    n2 = {"embedding": np.array([1.0, 2.0], dtype=np.float32)}
    v2 = get_node_embedding(n2, return_numpy=False)
    assert v2 == [1.0, 2.0]

    # Empty / None
    assert get_node_embedding({}) is None
    assert get_node_embedding({"embedding": None}) is None
    assert get_node_embedding({"embedding": []}) is None
    assert decode_embedding("") is None
