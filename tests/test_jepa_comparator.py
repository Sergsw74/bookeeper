"""
Unit tests for JEPA Comparator Engine and RankScaleCalibrator.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from bookeeper.evaluation.jepa_comparator import (
    DistributionSummary,
    JEPAComparatorEngine,
    RankScaleCalibrator,
    ReferenceChunk,
    ReferenceDataset,
    ReferenceIdea,
    compute_roc_auc,
    generate_markdown_report,
    load_reference_data,
)


def test_distribution_summary():
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    summary = DistributionSummary.from_array(data)
    assert summary.count == 5
    assert pytest.approx(summary.mean, abs=1e-5) == 0.3
    assert pytest.approx(summary.min, abs=1e-5) == 0.1
    assert pytest.approx(summary.max, abs=1e-5) == 0.5


def test_compute_roc_auc():
    pos = np.array([0.8, 0.9, 0.95])
    neg = np.array([0.1, 0.2, 0.3])
    auc = compute_roc_auc(pos, neg)
    assert pytest.approx(auc, abs=1e-5) == 1.0

    pos_overlap = np.array([0.5, 0.6])
    neg_overlap = np.array([0.4, 0.7])
    auc_overlap = compute_roc_auc(pos_overlap, neg_overlap)
    assert 0.0 <= auc_overlap <= 1.0


def test_rank_scale_calibrator():
    scores = np.array([0.5, 0.75, 1.0])
    aligned = RankScaleCalibrator.align_zero_anchor(scores, baseline_neg_mean=0.5)
    np.testing.assert_allclose(aligned, [0.0, 0.5, 1.0], atol=1e-5)

    min_max = RankScaleCalibrator.min_max_scale(scores)
    np.testing.assert_allclose(min_max, [0.0, 0.5, 1.0], atol=1e-5)

    ref = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    ranks = RankScaleCalibrator.percentile_rank(np.array([0.1, 0.5, 0.9]), ref)
    assert ranks[0] == 0.0
    assert ranks[1] == 0.4  # 2 out of 5 are <= 0.5
    assert ranks[2] == 0.8  # 4 out of 5 are <= 0.9


def test_load_reference_data_from_mock_state(tmp_path: Path):
    book_state = {
        "book_id": "99",
        "chunks": [
            {
                "chunk_id": "b99_c1_p1",
                "section_title": "Chapter 1",
                "chunk_idx": 1,
                "text": "Discussion of predictive representations.",
                "embedding": [0.1] * 64,
            },
            {
                "chunk_id": "b99_c1_p2",
                "section_title": "Chapter 1",
                "chunk_idx": 2,
                "text": "Further details on latent dynamics.",
                "embedding": [0.2] * 64,
            },
        ],
        "processed_chunks": {
            "b99_c1_p1": {
                "extraction": {
                    "concepts": [
                        {
                            "name": "Latent Dynamics",
                            "brief_description": "Dynamics in latent space",
                            "related_concepts": ["World Model"],
                        }
                    ],
                    "extracted_ideas": [
                        {
                            "idea_statement": "Predictive world modeling avoids pixel reconstruction.",
                            "key_entities": ["World Model"],
                        }
                    ],
                }
            }
        },
    }

    state_file = tmp_path / "book_99_state.json"
    state_file.write_text(json.dumps(book_state), encoding="utf-8")

    dataset = load_reference_data(state_dir=tmp_path, max_books=1)
    assert len(dataset.chunks) == 2
    assert "Latent Dynamics" in dataset.ideas
    assert ("Latent Dynamics", "World Model") in dataset.idea_relations


def test_jepa_comparator_engine_full(tmp_path: Path):
    rng = np.random.RandomState(42)
    dataset = ReferenceDataset(book_ids=["1", "2"])

    # Create synthetic chunks for book 1 and 2
    for b_id in ["1", "2"]:
        for i in range(10):
            cid = f"b{b_id}_c1_p{i}"
            emb = rng.randn(32).astype(np.float32) + (2.0 if b_id == "1" else -2.0)
            dataset.chunks.append(
                ReferenceChunk(
                    chunk_id=cid,
                    book_id=b_id,
                    section_title="Sec 1",
                    chunk_idx=i,
                    text=f"Text for chunk {cid}",
                    embedding=emb,
                )
            )

    # Attach ideas
    c0 = dataset.chunks[0]
    dataset.ideas["Concept Alpha"] = ReferenceIdea(
        name="Concept Alpha",
        book_id="1",
        chunk_id=c0.chunk_id,
        desc="Description Alpha",
        embedding=c0.embedding.copy(),
        related_names=["Concept Beta"],
    )
    dataset.ideas["Concept Beta"] = ReferenceIdea(
        name="Concept Beta",
        book_id="1",
        chunk_id=dataset.chunks[1].chunk_id,
        desc="Description Beta",
        embedding=dataset.chunks[1].embedding.copy(),
        related_names=[],
    )
    dataset.idea_relations.append(("Concept Alpha", "Concept Beta"))
    dataset.chunk_to_ideas[c0.chunk_id] = ["Concept Alpha"]

    engine = JEPAComparatorEngine(dataset, latent_dim=16, max_pairs_per_regime=50)
    report = engine.evaluate_all()

    assert report.books_sampled == 2
    assert report.chunks_total == 20
    assert report.chunk_vs_chunk is not None
    assert report.idea_vs_chunk is not None
    assert report.idea_vs_idea is not None

    # Check that JEPA negative mean is near 0
    assert abs(report.chunk_vs_chunk.jepa_neg.mean) < 0.2

    # Check report generation
    report_file = tmp_path / "report.md"
    md = generate_markdown_report(report, report_file)
    assert report_file.is_file()
    assert "JEPA Encoder vs. Current Embedding Model Performance Comparison Report" in md
    assert "xychart-beta" in md
