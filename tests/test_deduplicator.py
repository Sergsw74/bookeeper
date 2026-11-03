"""
Tests for EntityDeduplicator optimizations:
- O(1) canonical index mapping
- Lexical fast-path disambiguation
- Compressed embedding seeding from graph
- orjson fast save/load in ConceptGraphStore
"""

import numpy as np
import pytest

from bookeeper.graph.embedding_codec import encode_embedding
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.deduplicator import EntityDeduplicator
from bookeeper.processing.extractor import Concept


def test_lexical_disambiguation_heuristics():
    """Verify deterministic lexical resolution avoids unnecessary LLM calls."""
    # 1. Exact case-insensitive match
    c1 = Concept(name="Error Propagation Modeling", brief_description="Original definition")
    c2 = Concept(name="error propagation modeling", brief_description="New incoming definition")
    dec = EntityDeduplicator._try_lexical_disambiguation(c1, c2, similarity=0.90)
    assert dec is not None
    assert dec.is_same_concept is True

    # 2. Reordered core tokens (ignoring stop words)
    c3 = Concept(name="Modeling of Error Propagation")
    dec = EntityDeduplicator._try_lexical_disambiguation(c1, c3, similarity=0.91)
    assert dec is not None
    assert dec.is_same_concept is True

    # 3. Singular vs plural morphology
    c_sing = Concept(name="Failure Mode")
    c_plur = Concept(name="Failure Modes")
    dec = EntityDeduplicator._try_lexical_disambiguation(c_sing, c_plur, similarity=0.92)
    assert dec is not None
    assert dec.is_same_concept is True

    # 4. Opposing / antonym concepts
    c_att = Concept(name="Adversarial Attack Vector")
    c_def = Concept(name="Adversarial Defense Vector")
    dec = EntityDeduplicator._try_lexical_disambiguation(c_att, c_def, similarity=0.88)
    assert dec is not None
    assert dec.is_same_concept is False

    # 5. High similarity (>=0.93) + high Jaccard (>=0.75)
    c_hi1 = Concept(name="Multi-Agent Attribution System")
    c_hi2 = Concept(name="Multi-Agent Attribution Systems Architecture")
    dec = EntityDeduplicator._try_lexical_disambiguation(c_hi1, c_hi2, similarity=0.94)
    assert dec is not None
    assert dec.is_same_concept is True

    # 6. Low similarity (<0.83) + low Jaccard (<0.20)
    c_lo1 = Concept(name="Markov Decision Process")
    c_lo2 = Concept(name="Reinforcement Learning Agent")
    dec = EntityDeduplicator._try_lexical_disambiguation(c_lo1, c_lo2, similarity=0.81)
    assert dec is not None
    assert dec.is_same_concept is False

    # 7. Ambiguous nuance falls back to LLM (returns None)
    c_amb1 = Concept(name="Dynamic Prompting Strategy")
    c_amb2 = Concept(name="Adaptive Context Selection")
    dec = EntityDeduplicator._try_lexical_disambiguation(c_amb1, c_amb2, similarity=0.87)
    assert dec is None, "Ambiguous conceptual nuance must return None to trigger LLM disambiguation"


def test_seed_from_graph_compressed_embeddings(tmp_path):
    """Verify that seed_from_graph correctly restores Base64 compressed embeddings into cache."""
    store = ConceptGraphStore()
    vec = np.random.randn(128).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    c = Concept(
        name="Failure Attribution",
        category="Idea",
        brief_description="Tracing failures across agents.",
        embedding=vec.tolist(),
    )
    store.add_concept(c, embedding=vec.tolist())

    # Compress node embedding (as done when saved)
    node_data = store.graph.nodes["concept:Failure Attribution"]
    b64 = encode_embedding(vec, dtype="float16")
    node_data["embedding_b64"] = b64
    node_data["embedding_dtype"] = "float16"
    del node_data["embedding"]

    dedup = EntityDeduplicator(similarity_threshold=0.80, high_similarity_threshold=0.95)
    seeded = dedup.seed_from_graph(store)

    assert seeded == 1
    assert "Failure Attribution" in dedup._cache
    stored_vec, stored_concept = dedup._cache["Failure Attribution"]
    assert stored_vec is not None
    assert stored_vec.shape == (128,)
    # Verify vector was placed in embedding cache
    assert "Failure Attribution" in dedup._embed_cache


def test_store_orjson_fast_save_and_load(tmp_path):
    """Verify that ConceptGraphStore fast serialization preserves nodes, edges, and attributes."""
    store = ConceptGraphStore()
    store.add_book(41, title="Test Book", author="Author A")
    c = Concept(name="Test Concept", brief_description="A test idea")
    store.add_concept(c)

    file_path = tmp_path / "test_kg.json"
    saved = store.save(file_path)
    assert saved.is_file()

    loaded_store = ConceptGraphStore()
    loaded_store.load(file_path)

    assert loaded_store.has_node("book:41")
    assert loaded_store.has_node("concept:Test Concept")
    concept_data = loaded_store.get_node_data("concept:Test Concept")
    assert concept_data is not None
    assert concept_data.get("name") == "Test Concept"
