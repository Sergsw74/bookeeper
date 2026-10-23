"""
Unit tests for maximum context size and simultaneous multi-chunk LLM processing.
"""

from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from bookeeper.config import Settings
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.extractor import (
    BatchSectionExtraction,
    ChunkExtraction,
    Concept,
    ExtractedIdea,
    KnowledgeExtractor,
    SectionExtraction,
    batch_chunks_by_context_size,
)
from bookeeper.cli import app, _get_effective_settings

runner = CliRunner()


def test_max_context_size_config_defaults_and_parsing():
    """Verify max_context_size defaults to 5120 (5KB) and parses string units."""
    # 1. Default value is 5120 bytes (5 KB)
    cfg = Settings()
    assert cfg.max_context_size == 5120

    # 2. Integer configuration
    cfg2 = Settings(max_context_size=8192)
    assert cfg2.max_context_size == 8192

    # 3. String units (5kb, 5KB, 5k)
    cfg_kb = Settings(max_context_size="5kb")
    assert cfg_kb.max_context_size == 5120

    cfg_upper_kb = Settings(max_context_size="5KB")
    assert cfg_upper_kb.max_context_size == 5120

    cfg_k = Settings(max_context_size="5k")
    assert cfg_k.max_context_size == 5120

    cfg_mb = Settings(max_context_size="1MB")
    assert cfg_mb.max_context_size == 1024 * 1024


def test_get_effective_settings_cli_override():
    """Verify CLI --max-context-size overrides config file settings."""
    cfg = _get_effective_settings(max_context_size=10240)
    assert cfg.max_context_size == 10240


def test_batch_chunks_by_context_size_combining_three_chunks():
    """
    Verify chunks are combined up to max_context_size (default 5KB / 5120 bytes).
    Specifically tests joining three ~1.2KB chunks into one batch, while a 4th chunk starts a new batch.
    """
    # Create 4 chunks with ~1200 bytes of text each
    # Each chunk with 150B tag overhead is ~1350 bytes.
    # 3 chunks = ~4050 bytes <= 5120 bytes.
    # 4 chunks = ~5400 bytes > 5120 bytes -> 4th chunk goes into batch 2.
    text_1200 = "Sentence about systems theory. " * 40  # ~1240 bytes
    c1 = HierarchicalChunk(chunk_id="b1_c1", book_id=1, book_title="Book A", section_title="Sec 1", chapter_idx=1, chunk_idx=1, text=text_1200)
    c2 = HierarchicalChunk(chunk_id="b1_c2", book_id=1, book_title="Book A", section_title="Sec 1", chapter_idx=1, chunk_idx=2, text=text_1200)
    c3 = HierarchicalChunk(chunk_id="b1_c3", book_id=1, book_title="Book A", section_title="Sec 1", chapter_idx=1, chunk_idx=3, text=text_1200)
    c4 = HierarchicalChunk(chunk_id="b1_c4", book_id=1, book_title="Book A", section_title="Sec 2", chapter_idx=1, chunk_idx=4, text=text_1200)

    batches = batch_chunks_by_context_size([c1, c2, c3, c4], max_context_size=5120)

    assert len(batches) == 2
    # First batch contains three chunks joined together
    assert len(batches[0]) == 3
    assert [c.chunk_id for c in batches[0]] == ["b1_c1", "b1_c2", "b1_c3"]
    # Second batch contains the 4th chunk
    assert len(batches[1]) == 1
    assert batches[1][0].chunk_id == "b1_c4"


def test_batch_chunks_large_single_chunk():
    """Verify a chunk exceeding max_context_size by itself is placed in its own single-item batch."""
    huge_text = "Data intensive architectures. " * 300  # ~9000 bytes > 5120
    small_text = "Small intro."
    c1 = HierarchicalChunk(chunk_id="huge_1", book_id=1, book_title="Book A", section_title="Sec 1", chapter_idx=1, chunk_idx=1, text=huge_text)
    c2 = HierarchicalChunk(chunk_id="small_2", book_id=1, book_title="Book A", section_title="Sec 1", chapter_idx=1, chunk_idx=2, text=small_text)

    batches = batch_chunks_by_context_size([c1, c2], max_context_size=5120)
    assert len(batches) == 2
    assert len(batches[0]) == 1
    assert batches[0][0].chunk_id == "huge_1"
    assert len(batches[1]) == 1
    assert batches[1][0].chunk_id == "small_2"


def test_batch_chunks_disabled_when_zero_or_negative():
    """Verify batching is disabled (each chunk independent) when max_context_size <= 0."""
    c1 = HierarchicalChunk(chunk_id="c1", book_id=1, book_title="B", section_title="S", chapter_idx=1, chunk_idx=1, text="Text 1")
    c2 = HierarchicalChunk(chunk_id="c2", book_id=1, book_title="B", section_title="S", chapter_idx=1, chunk_idx=2, text="Text 2")

    batches = batch_chunks_by_context_size([c1, c2], max_context_size=0)
    assert len(batches) == 2
    assert len(batches[0]) == 1
    assert len(batches[1]) == 1


def test_batch_section_extraction_schema_validation():
    """Verify BatchSectionExtraction supports standard list, dict keyed by chunk_id, and legacy forms."""
    # 1. Standard format: {"chunks": [{"chunk_id": "c1", ...}, {"chunk_id": "c2", ...}]}
    data1 = {
        "chunks": [
            {
                "chunk_id": "c1",
                "extracted_ideas": [
                    {
                        "source_quote": "Quote one",
                        "idea_statement": "Idea one",
                        "key_entities": ["Concept1"],
                        "idea_type": "Mechanism",
                    }
                ],
            },
            {
                "chunk_id": "c2",
                "extracted_ideas": [
                    {
                        "source_quote": "Quote two",
                        "idea_statement": "Idea two",
                        "key_entities": ["Concept2"],
                        "idea_type": "Definition",
                    }
                ],
            },
        ]
    }
    batch1 = BatchSectionExtraction.model_validate(data1)
    assert len(batch1.chunks) == 2
    assert batch1.chunks[0].chunk_id == "c1"
    assert len(batch1.chunks[0].extracted_ideas) == 1
    assert len(batch1.chunks[0].concepts) == 1  # auto-synthesized
    assert batch1.chunks[1].chunk_id == "c2"

    # 2. Dict keyed by chunk_id: {"chunks": {"c1": {...}, "c2": {...}}}
    data2 = {
        "chunks": {
            "c1": {
                "extracted_ideas": [
                    {
                        "source_quote": "Quote one",
                        "idea_statement": "Idea one",
                        "key_entities": ["Concept1"],
                        "idea_type": "Argument",
                    }
                ]
            },
            "c2": {
                "extracted_ideas": []
            },
        }
    }
    batch2 = BatchSectionExtraction.model_validate(data2)
    assert len(batch2.chunks) == 2
    c_ids = {c.chunk_id for c in batch2.chunks}
    assert c_ids == {"c1", "c2"}


def test_knowledge_extractor_extract_chunks_with_tags():
    """
    Verify extract_chunks generates proper <chunk id="..." section="..." index="..."> XML tags
    and returns independent SectionExtraction objects for each chunk.
    """
    c1 = HierarchicalChunk(
        chunk_id="chk_alpha",
        book_id=1,
        book_title="Distributed Systems",
        section_title="Consensus Protocols",
        subtitle="Paxos",
        chapter_idx=1,
        chunk_idx=1,
        text="Paxos guarantees consensus even under network partitions. Each proposer broadcasts accept requests.",
    )
    c2 = HierarchicalChunk(
        chunk_id="chk_beta",
        book_id=1,
        book_title="Distributed Systems",
        section_title="Consensus Protocols",
        subtitle="Raft",
        chapter_idx=1,
        chunk_idx=2,
        text="Raft decomposes consensus into leader election and log replication. Heartbeats maintain leadership.",
    )

    extractor = KnowledgeExtractor(model="test_model", max_context_size=5120)

    # Mock _execute_structured_invoke to inspect messages and return structured batch response
    captured_messages = []

    def mock_invoke(schema_cls, messages, **kwargs):
        captured_messages.extend(messages)
        return BatchSectionExtraction(
            chunks=[
                ChunkExtraction(
                    chunk_id="chk_alpha",
                    extracted_ideas=[
                        ExtractedIdea(
                            source_quote="Paxos guarantees consensus even under network partitions.",
                            idea_statement="Paxos provides robust consensus under network partitions.",
                            key_entities=["Paxos", "Consensus"],
                            idea_type="Mechanism",
                        )
                    ],
                ),
                ChunkExtraction(
                    chunk_id="chk_beta",
                    extracted_ideas=[
                        ExtractedIdea(
                            source_quote="Raft decomposes consensus into leader election and log replication.",
                            idea_statement="Raft simplifies consensus via distinct leader election and log replication stages.",
                            key_entities=["Raft", "Log Replication"],
                            idea_type="Definition",
                        )
                    ],
                ),
            ]
        )

    with patch.object(extractor, "_execute_structured_invoke", side_effect=mock_invoke):
        results = extractor.extract_chunks([c1, c2], book_title="Distributed Systems")

        # 1. Output must provide independent results for each chunk
        assert "chk_alpha" in results
        assert "chk_beta" in results

        alpha_ext = results["chk_alpha"]
        assert len(alpha_ext.extracted_ideas) == 1
        assert "Paxos" in alpha_ext.extracted_ideas[0].idea_statement
        assert len(alpha_ext.concepts) == 1
        assert alpha_ext.concepts[0].name == "Paxos"

        beta_ext = results["chk_beta"]
        assert len(beta_ext.extracted_ideas) == 1
        assert "Raft" in beta_ext.extracted_ideas[0].idea_statement
        assert len(beta_ext.concepts) == 1
        assert beta_ext.concepts[0].name == "Raft"

        # 2. Verify XML chunk tags in prompt
        human_msg = captured_messages[1].content
        assert '<chunk id="chk_alpha" section="Consensus Protocols" subtitle="Paxos" index="1">' in human_msg
        assert '<chunk id="chk_beta" section="Consensus Protocols" subtitle="Raft" index="2">' in human_msg
        assert "</chunk>" in human_msg


def test_extract_chunks_quote_grounding_isolated_per_chunk():
    """
    Verify that ideas extracted for a chunk must have quotes grounded in that specific chunk,
    not leaking from another chunk in the batch.
    """
    c1 = HierarchicalChunk(
        chunk_id="c1",
        book_id=1,
        book_title="Test",
        section_title="S1",
        chapter_idx=1,
        chunk_idx=1,
        text="Apples grow on trees and are sweet fruits.",
    )
    c2 = HierarchicalChunk(
        chunk_id="c2",
        book_id=1,
        book_title="Test",
        section_title="S1",
        chapter_idx=1,
        chunk_idx=2,
        text="Carrots grow under the ground and are rich in vitamin A.",
    )

    extractor = KnowledgeExtractor(model="test_model")

    # Suppose model mistakenly attributes the Carrot quote to chunk c1 (Apples)
    mock_batch = BatchSectionExtraction(
        chunks=[
            ChunkExtraction(
                chunk_id="c1",
                extracted_ideas=[
                    ExtractedIdea(
                        source_quote="Carrots grow under the ground",  # Absent in c1!
                        idea_statement="Carrots are root vegetables.",
                        key_entities=["Carrots"],
                        idea_type="Definition",
                    )
                ],
            ),
            ChunkExtraction(
                chunk_id="c2",
                extracted_ideas=[
                    ExtractedIdea(
                        source_quote="Carrots grow under the ground",  # Present in c2!
                        idea_statement="Carrots are root vegetables.",
                        key_entities=["Carrots"],
                        idea_type="Definition",
                    )
                ],
            ),
        ]
    )

    with patch.object(extractor, "_execute_structured_invoke", return_value=mock_batch):
        results = extractor.extract_chunks([c1, c2])

        # c1 ideas must NOT be grounded because quote and concepts are absent from c1
        assert len(results["c1"].extracted_ideas) == 0

        # c2 ideas must be grounded because quote matches c2
        assert len(results["c2"].extracted_ideas) == 1
        assert results["c2"].extracted_ideas[0].source_quote == "Carrots grow under the ground"


def test_build_graph_with_chunk_batching(tmp_path):
    """Verify build-graph CLI combines chunks up to max_context_size during indexing."""
    from pathlib import Path
    from bookeeper.calibre.parser import Section

    out_dir = tmp_path / "output"
    chunks_dir = tmp_path / "custom_chunks"
    book_file = tmp_path / "batch_test.epub"
    book_file.write_text("Test content", encoding="utf-8")

    # Create isolated test config pointing to tmp_path
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(
        f"output_dir: '{out_dir}'\nchunks_dir: '{chunks_dir}'\nclean_export: false\nneo4j:\n  enabled: false\n",
        encoding="utf-8",
    )

    # 3 sections of ~200-300 bytes each (>= min_chunk_chars 120)
    mock_sections = [
        Section(
            title="Chapter 1",
            chapter_idx=1,
            text="Distributed consensus protocols allow a group of networked computers to agree on a state value or log sequence even in the presence of unreliable network connections or node failures.",
        ),
        Section(
            title="Chapter 2",
            chapter_idx=2,
            text="Leader election algorithms establish a single authoritative coordinator responsible for ordering client mutations and replicating them across all follower nodes.",
        ),
        Section(
            title="Chapter 3",
            chapter_idx=3,
            text="Replication logs synchronize state transitions sequentially to ensure deterministic recovery and consistency across distributed replicas.",
        ),
    ]

    captured_batches = []

    def mock_extract_chunks(chunks, **kwargs):
        captured_batches.append(list(chunks))
        res = {}
        for c in chunks:
            res[c.chunk_id] = SectionExtraction(
                extracted_ideas=[
                    ExtractedIdea(
                        source_quote=c.text,
                        idea_statement=f"Idea for {c.section_title}",
                        key_entities=[c.section_title],
                        idea_type="Mechanism",
                    )
                ]
            )
        return res

    with patch("bookeeper.calibre.parser.BookParser.parse", return_value=mock_sections), \
         patch("bookeeper.calibre.parser.BookParser.is_graphical_format", return_value=False), \
         patch.object(KnowledgeExtractor, "extract_chunks", side_effect=mock_extract_chunks), \
         patch.object(KnowledgeExtractor, "from_settings") as mock_from_settings, \
         patch("bookeeper.processing.deduplicator.EntityDeduplicator.from_settings"):

        mock_ext_instance = MagicMock()
        mock_ext_instance.extract_chunks.side_effect = mock_extract_chunks
        mock_from_settings.return_value = mock_ext_instance

        res = runner.invoke(
            app,
            [
                "build-graph",
                "--file", str(book_file),
                "--config", str(cfg_file),
                "--skip-warmup",
                "--max-context-size", "5120",
                "--max-tasks", "1",
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        assert "Knowledge Graph Build Complete" in res.stdout

        # Verify that all 3 chunks were combined into a single batch (total size ~900B < 5120B)
        assert len(captured_batches) >= 1
        assert len(captured_batches[0]) == 3

