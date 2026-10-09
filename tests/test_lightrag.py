"""Unit tests for LightRAGEngine integration and CLI query command."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.config import OllamaServerConfig, Settings
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.extractor import OllamaPool
from bookeeper.rag.lightrag_engine import LightRAGEngine

runner = CliRunner()


@pytest.fixture
def mock_pool():
    servers = [OllamaServerConfig(url="http://mock-ollama:11434", priority=1, name="mock")]
    return OllamaPool(servers=servers)


def test_lightrag_engine_init(tmp_path, mock_pool):
    """Verify LightRAGEngine initializes with settings and pool."""
    engine = LightRAGEngine(
        working_dir=tmp_path / "lightrag_store",
        pool=mock_pool,
        llm_model="llama3.1:8b",
        embedding_model="nomic-embed-text",
        embedding_dim=768,
    )
    assert engine.working_dir.is_dir()
    assert engine.llm_model == "llama3.1:8b"
    assert engine.embedding_dim == 768


@pytest.mark.anyio
async def test_lightrag_llm_and_embedding_funcs(tmp_path, mock_pool):
    """Verify async LLM completion and embedding functions format requests properly."""
    engine = LightRAGEngine(
        working_dir=tmp_path,
        pool=mock_pool,
        embedding_dim=768,
    )

    with patch.object(engine, "_call_ollama") as mock_call:
        # Mock chat response
        mock_call.side_effect = [
            {"message": {"content": "LightRAG consensus response"}},
            {"embeddings": [[0.1] * 768, [0.2] * 768]},
        ]

        # Test LLM
        res = await engine._llm_model_func("What is Raft?", system_prompt="You are an expert.")
        assert res == "LightRAG consensus response"

        # Test Embed
        vecs = await engine._embedding_func(["text 1", "text 2"])
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (2, 768)


@pytest.mark.anyio
async def test_lightrag_insert_chunks(tmp_path, mock_pool):
    """Verify HierarchicalChunk objects format with breadcrumbs into LightRAG documents."""
    engine = LightRAGEngine(
        working_dir=tmp_path,
        pool=mock_pool,
    )

    mock_rag = MagicMock()
    # ainsert is an async coroutine
    async def mock_ainsert(docs, ids=None, **kwargs):
        return "ok"

    mock_rag.ainsert.side_effect = mock_ainsert
    engine._rag = mock_rag
    engine._initialized = True

    chunks = [
        HierarchicalChunk(
            book_id=101,
            book_title="Distributed Systems",
            section_title="Chapter 3: Consensus",
            chapter_idx=3,
            chunk_idx=1,
            text="Leader election happens when a follower receives no heartbeat.",
            chunk_id="chunk_1",
            chapter_title="Chapter 3: Consensus",
            subtitle="Leader Election Mechanism",
        )
    ]

    count = await engine.ainsert_chunks(chunks, book_title="Distributed Systems", book_id=101)
    assert count == 1
    assert mock_rag.ainsert.called
    called_docs = mock_rag.ainsert.call_args[0][0]
    called_ids = mock_rag.ainsert.call_args[1]["ids"]
    assert "Book: Distributed Systems" in called_docs[0]
    assert "Chapter: Chapter 3: Consensus" in called_docs[0]
    assert "Section: Leader Election Mechanism" in called_docs[0]
    assert called_ids[0] == "book_101_chunk_chunk_1"


@pytest.mark.anyio
async def test_lightrag_query_modes(tmp_path, mock_pool):
    """Verify LightRAG query delegates to underlying engine with validated modes."""
    engine = LightRAGEngine(
        working_dir=tmp_path,
        pool=mock_pool,
    )

    mock_rag = MagicMock()
    async def mock_aquery(query, param=None, **kwargs):
        return f"Synthesized answer in mode: {param.mode}"

    mock_rag.aquery.side_effect = mock_aquery
    engine._rag = mock_rag
    engine._initialized = True

    ans = await engine.aquery("How does Paxos work?", mode="global")
    assert "Synthesized answer in mode: global" in ans

    # Invalid mode check
    with pytest.raises(ValueError, match="Invalid mode 'unknown'"):
        await engine.aquery("Test?", mode="unknown")


def test_cli_query_empty_directory_exits():
    """Verify CLI exits cleanly with instructions when query is called before index is built."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_dir = Path(tmpdir) / "empty_rag"
        result = runner.invoke(app, ["query", "What is Raft?", "--lightrag-dir", str(empty_dir)])
        assert result.exit_code != 0
        assert "is empty or does not exist" in result.output
        assert "bookeeper build-graph --lightrag" in result.output
