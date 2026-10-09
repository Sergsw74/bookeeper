"""
Unit tests for Knowledge Graph verification engine (IdeaVerifier, verify_graph, CLI verify command).
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.extractor import Concept
from bookeeper.processing.ollama_pool import OllamaPool, OllamaServerNode
from bookeeper.processing.verifier import (
    IdeaVerifier,
    VerificationItem,
    VerificationReport,
    VerificationResult,
    _parse_verification_text,
    verify_graph,
)


@pytest.fixture
def mock_pool():
    return OllamaPool(
        servers=[OllamaServerNode(url="http://localhost:11434", priority=1)],
        cooldown_seconds=60,
    )


@pytest.fixture
def populated_store():
    store = ConceptGraphStore()
    store.add_book(1, title="Test Book 1", author="Author A")

    for i in range(1, 11):
        concept = Concept(
            name=f"Concept {i}",
            category="Technology",
            summary=f"Summary for concept {i}",
            weight=i % 10,
            related_concepts=[],
        )
        chk = HierarchicalChunk(
            chunk_id=f"chk_{i}",
            chunk_idx=i,
            book_id=1,
            book_title="Test Book 1",
            chapter_idx=1,
            chapter_title="Chapter 1",
            section_title="Section 1",
            subtitle="Subsection A",
            breadcrumb="Test Book 1 > Chapter 1",
            text=f"This is chunk {i} discussing concept {i} in great detail.",
            char_count=50,
            word_count=10,
        )
        store.add_concept(concept)
        store.add_idea_support_link(
            concept_name=f"Concept {i}",
            chunk=chk,
            quote=f"discussing concept {i}",
            brief_description=f"Brief {i}",
            detailed_explanation=f"Detailed {i}",
        )
    return store


def test_idea_verifier_mocked(mock_pool):
    verifier = IdeaVerifier(pool=mock_pool, model_name="llama3.1:8b")

    mock_res = VerificationResult(
        is_supported=True,
        confidence=0.95,
        explanation="The chunk explicitly defines the concept.",
    )

    with patch.object(verifier, "_execute_structured_invoke", return_value=mock_res):
        res = verifier.verify_idea_chunk(
            idea_name="Event Sourcing",
            idea_category="Architecture",
            idea_summary="Append-only log architecture",
            quote="State changes are stored as events.",
            book_title="Distributed Systems",
            section_title="Data Consistency",
            chunk_text="In Event Sourcing, state changes are stored as events in an append-only log.",
        )
        assert res.is_supported is True
        assert res.confidence == 0.95
        assert "explicitly defines" in res.explanation


def test_verify_graph_workflow(mock_pool, populated_store):
    """Verify sampling, stats computation, and discrepancy tracking."""
    # Mock verifier: even index = supported, odd index = discrepancy
    def _mock_invoke(messages):
        user_content = messages[-1].content
        # If concept contains an odd number -> discrepancy
        is_supported = not any(f"Concept {odd}" in user_content for odd in [1, 3, 5, 7, 9])
        return VerificationResult(
            is_supported=is_supported,
            confidence=0.9,
            explanation="Supported by text" if is_supported else "Idea not found in chunk",
        )

    with patch.object(IdeaVerifier, "_execute_structured_invoke", side_effect=_mock_invoke):
        # Sample 50% of ideas (5 out of 10)
        report = verify_graph(
            store=populated_store,
            pool=mock_pool,
            model_name="llama3.1:8b",
            percent=50.0,
            mode="ideas",
            max_examples=20,
            seed=42,
            concurrency=2,
        )

        assert isinstance(report, VerificationReport)
        st = report.stats
        assert st.total_ideas_in_graph == 10
        assert st.candidate_ideas_with_chunks == 10
        assert st.sampled_ideas == 5
        assert st.total_evaluations == 5
        assert st.verified_count + st.discrepancy_count == 5
        assert 0.0 <= st.verified_percentage <= 100.0
        assert 0.0 <= st.discrepancy_percentage <= 100.0
        assert st.mode == "ideas"

        # Check discrepancy records
        for d in report.discrepancies:
            assert d.is_supported is False
            assert "not found" in d.explanation.lower()


def test_verify_cli_command(tmp_path, populated_store):
    """Test running 'bookeeper verify' CLI command with output file generation."""
    graph_file = tmp_path / "knowledge_graph.json"
    populated_store.save(graph_file)

    cfg_file = tmp_path / "config.yaml"
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    report_file = out_dir / "custom_report.json"

    cfg_file.write_text(f"""
output_dir: "{out_dir}"
verification:
  model: "llama3.1:8b"
  percent: 20.0
  mode: "ideas"
""", encoding="utf-8")

    mock_res = VerificationResult(
        is_supported=True,
        confidence=1.0,
        explanation="Chunk discusses the idea directly.",
    )

    runner = CliRunner()
    with patch.object(IdeaVerifier, "_execute_structured_invoke", return_value=mock_res):
        res = runner.invoke(
            app,
            [
                "verify",
                "--graph-file", str(graph_file),
                "--config", str(cfg_file),
                "--percent", "20.0",
                "--output", str(report_file),
                "--seed", "123",
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        assert "Knowledge Graph Verification Plan" in res.stdout
        assert "Verification Results Summary" in res.stdout
        assert "Verified / Supported (Good)" in res.stdout
        assert report_file.is_file()

        # Verify saved JSON report structure
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "stats" in data
        assert data["stats"]["mode"] == "ideas"
        assert data["stats"]["total_ideas_in_graph"] == 10
        assert data["stats"]["sampled_ideas"] == 2  # 20% of 10
        assert data["stats"]["verified_count"] == 2
        assert data["stats"]["discrepancy_count"] == 0


def test_cosine_distance_and_sentence_splitter():
    from bookeeper.processing.verifier import cosine_distance, split_sentences

    # 1. Cosine distance properties
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert abs(cosine_distance([1.0, 0.0], [0.0, 1.0]) - 1.0) < 1e-6
    assert abs(cosine_distance([1.0, 0.0], [-1.0, 0.0]) - 2.0) < 1e-6
    assert cosine_distance([0.0, 0.0], [1.0, 1.0]) == 1.0

    # 2. Sentence splitter
    text = "First sentence here. Second sentence follows! Third one? Yes, indeed."
    sents = split_sentences(text)
    assert len(sents) == 4
    assert sents[0] == "First sentence here."
    assert sents[1] == "Second sentence follows!"
    assert sents[2] == "Third one?"
    assert sents[3] == "Yes, indeed."
    assert split_sentences("") == []
    assert split_sentences("   ") == []


class MockDeterministicEmbeddings:
    """Mock embeddings generating deterministic vectors based on word content."""

    model: str = "mock-embed"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        t = text.lower()
        vec = [0.1] * 5
        if "apple" in t:
            vec[0] += 2.0
            vec[1] += 1.0
        if "banana" in t:
            vec[0] += 1.8
            vec[1] += 1.2
        if "quantum" in t or "physics" in t:
            vec[3] += 5.0
            vec[4] += 5.0
        return vec


def test_smart_sentence_chunker_expansion():
    from bookeeper.processing.verifier import SmartSentenceChunker

    mock_emb = MockDeterministicEmbeddings()
    chunker = SmartSentenceChunker(
        embeddings=mock_emb,
        distance_threshold=0.10,
        max_chunk_chars=500,
        min_chunk_chars=20,
    )

    # Coherent text staying on apples and bananas
    coherent_text = (
        "Apples are sweet red fruits that grow on trees. "
        "Bananas are also delicious yellow fruits popular everywhere. "
        "More apples and bananas make a great fruit salad."
    )
    chunks = chunker.chunk_text(coherent_text)
    assert len(chunks) == 1

    # Mixed text shifting abruptly from fruit to quantum physics
    mixed_text = (
        "Apples are sweet red fruits that grow on trees. "
        "Bananas are also delicious yellow fruits popular everywhere. "
        "Quantum physics calculates wave function collapse in subatomic particles."
    )
    mixed_chunks = chunker.chunk_text(mixed_text)
    assert len(mixed_chunks) == 2
    assert "Apples" in mixed_chunks[0]
    assert "Quantum" in mixed_chunks[1]


def test_chunking_verifier_distribution_and_stats(tmp_path):
    from bookeeper.processing.chunker import ChunkStore, HierarchicalChunk
    from bookeeper.processing.verifier import ChunkingVerifier, verify_chunking

    mock_emb = MockDeterministicEmbeddings()
    store = ChunkStore(tmp_path / "chunks")

    # Book 1: 2 coherent chunks, 1 divergent chunk
    c1 = HierarchicalChunk(
        chunk_id="b1_c1_p1",
        book_id=1,
        book_title="Fruit Book",
        section_title="Chapter 1",
        chapter_idx=1,
        chunk_idx=1,
        text="Apples are fresh fruits. Bananas are sweet yellow fruits.",
    )
    c2 = HierarchicalChunk(
        chunk_id="b1_c1_p2",
        book_id=1,
        book_title="Fruit Book",
        section_title="Chapter 1",
        chapter_idx=1,
        chunk_idx=2,
        text="Apples and bananas make healthy smoothies for breakfast.",
    )
    # Divergent chunk with sudden semantic jump
    c3 = HierarchicalChunk(
        chunk_id="b1_c2_p1",
        book_id=1,
        book_title="Fruit Book",
        section_title="Chapter 2",
        chapter_idx=2,
        chunk_idx=1,
        text="Apples are tasty fruits. Quantum physics describes particle entanglement.",
    )
    store.save_chunks(1, "Fruit Book", [c1, c2, c3])

    verifier = ChunkingVerifier(embeddings=mock_emb, distance_threshold=0.10)
    stats, discrepancies = verifier.verify_book(1, [c1, c2, c3], "Fruit Book")

    assert stats.total_chunks == 3
    assert stats.total_sentences == 5
    assert stats.total_transitions == 2
    assert stats.coherent_chunks == 2
    assert stats.divergent_chunks == 1
    assert stats.coherence_rate == round(2 / 3 * 100.0, 2)
    assert len(stats.distribution_buckets) == 5
    assert len(discrepancies) >= 1

    # verify_chunking workflow
    report = verify_chunking(
        chunk_store=store,
        embeddings=mock_emb,
        distance_threshold=0.10,
        book_id=1,
    )
    assert report.mode == "chunking"
    assert report.stats.book_id == 1
    assert len(report.discrepancies) >= 1


def test_verify_cli_chunking_mode(tmp_path):
    from typer.testing import CliRunner
    from bookeeper.cli import app
    from bookeeper.processing.chunker import ChunkStore, HierarchicalChunk

    mock_emb = MockDeterministicEmbeddings()
    chunks_dir = tmp_path / "chunks"
    store = ChunkStore(chunks_dir)

    c1 = HierarchicalChunk(
        chunk_id="b2_c1_p1",
        book_id=2,
        book_title="Science Book",
        section_title="Chapter 1",
        chapter_idx=1,
        chunk_idx=1,
        text="Quantum physics studies atoms. Quantum entanglement links states.",
    )
    store.save_chunks(2, "Science Book", [c1])

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    report_file = out_dir / "chunk_report.json"

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"""
output_dir: "{out_dir}"
chunks_dir: "{chunks_dir}"
verification:
  distance_threshold: 0.25
  mode: "chunking"
""", encoding="utf-8")

    runner = CliRunner()
    with patch("bookeeper.cli.FailoverOllamaEmbeddings") as mock_foe:
        mock_foe.return_value = mock_emb
        mock_foe.from_settings.return_value = mock_emb
        res = runner.invoke(
            app,
            [
                "verify",
                "--mode", "chunking",
                "--config", str(cfg_file),
                "--chunks-dir", str(chunks_dir),
                "--output", str(report_file),
            ],
            catch_exceptions=False,
        )

        assert res.exit_code == 0
        assert "Smart Chunking Verification Plan" in res.stdout
        assert "Chunking Verification Results" in res.stdout
        assert "Semantic Distance Distribution" in res.stdout
        assert report_file.is_file()

        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["mode"] == "chunking"
        assert data["stats"]["total_chunks"] == 1
        assert data["stats"]["coherent_chunks"] == 1


def test_parse_verification_text():
    """Verify JSON extraction, fence stripping, and heuristic text parsing."""
    # 1. Clean JSON
    res1 = _parse_verification_text('{"is_supported": true, "confidence": 0.95, "explanation": "Direct match"}')
    assert res1 is not None
    assert res1.is_supported is True
    assert res1.confidence == 0.95
    assert "Direct match" in res1.explanation

    # 2. Markdown wrapped JSON with thinking tags
    raw_md = (
        "<think>Let me analyze...</think>\n"
        "```json\n"
        '{"is_supported": false, "confidence": 0.8, "explanation": "Not mentioned in chunk."}\n'
        "```"
    )
    res2 = _parse_verification_text(raw_md)
    assert res2 is not None
    assert res2.is_supported is False
    assert res2.confidence == 0.8

    # 3. Freeform text matching user's real-world error log:
    # '* Target Idea: "Rampart" *does* substantiate the premise...'
    res3 = _parse_verification_text('* Target Idea: "Rampart" *does* substantiate the premise of fortress construction.')
    assert res3 is not None
    assert res3.is_supported is True
    assert "Rampart" in res3.explanation

    # 4. Negative freeform text
    res4 = _parse_verification_text('* Target Idea: "Data Streaming" does not substantiate the passage about cooking.')
    assert res4 is not None
    assert res4.is_supported is False
    assert "cooking" in res4.explanation

    # 5. Empty or whitespace
    assert _parse_verification_text("") is None
    assert _parse_verification_text("   ") is None


def test_execute_structured_invoke_recovers_validation_error(mock_pool):
    """Test that when LangChain raises a ValidationError on bullet text, it is recovered without cooldown."""
    verifier = IdeaVerifier(pool=mock_pool, model_name="llama3.1:8b")

    # Simulate ChatOllama raising ValidationError with input_value
    from langchain_core.messages import HumanMessage
    from pydantic_core import ValidationError

    def _mock_execute_failover(op, **kwargs):
        # Verify quarantine_server is disabled so server does not cool down for 600s
        assert kwargs.get("quarantine_server") is False
        assert kwargs.get("max_task_duration") >= 180.0
        return op("http://localhost:11434")

    bullet_text = '* Target Idea: "Rampart" *does* substantiate the fortress defense.'
    with patch.object(mock_pool, "execute_with_failover", side_effect=_mock_execute_failover):
        with patch("bookeeper.processing.verifier.ChatOllama") as mock_chat:
            mock_inst = MagicMock()
            mock_chat.return_value = mock_inst
            mock_struct = MagicMock()
            mock_inst.with_structured_output.return_value = mock_struct

            # Mock structured_llm.invoke raising ValueError / string matching validation error
            mock_struct.invoke.side_effect = ValueError(
                f"1 validation error for VerificationResult\n"
                f"  Invalid JSON: expected value [type=json_invalid, input_value='{bullet_text}', input_type=str]"
            )

            res = verifier._execute_structured_invoke([HumanMessage(content="test")])
            assert res.is_supported is True
            assert "fortress defense" in res.explanation


def test_verify_graph_passes_item_to_progress_callback(mock_pool, populated_store):
    """Verify that progress_callback receives the item including its server_node."""
    received_items = []

    def _callback(completed, total, idea_name, item=None):
        if item is not None:
            received_items.append(item)

    mock_res = VerificationResult(
        is_supported=True,
        confidence=0.9,
        explanation="Concept matches chunk",
    )

    with patch.object(IdeaVerifier, "_execute_structured_invoke", return_value=mock_res):
        report = verify_graph(
            store=populated_store,
            pool=mock_pool,
            model_name="llama3.1:8b",
            percent=30.0,
            seed=42,
            progress_callback=_callback,
        )

        assert len(received_items) == report.stats.total_evaluations
        for item in received_items:
            assert isinstance(item, VerificationItem)
            assert item.is_supported is True
            assert item.idea_name


