"""
Unit tests for LightRAG Raw Indexer, Question Generator, Benchmark Judge, and A/B Benchmark Runner.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.config import Settings
from bookeeper.evaluation.benchmark import (
    ABBenchmarkRunner,
    BenchmarkJudge,
    BenchmarkReport,
    JudgeEvaluation,
)
from bookeeper.evaluation.question_generator import (
    QuestionGenerator,
    QuestionItem,
    QuestionLibrary,
)
from bookeeper.rag.lightrag_service import LightRAGRawIndexer, LightRAGService

runner = CliRunner()


@pytest.fixture
def sample_settings(tmp_path):
    cfg = Settings(output_dir=str(tmp_path / "output"))
    return cfg


def test_question_library_roundtrip_persistence(tmp_path):
    """Verify QuestionLibrary serializes to JSON and deserializes cleanly."""
    ql = QuestionLibrary(
        model="llama3.1:8b",
        book_titles=["Designing Data-Intensive Applications"],
        questions=[
            QuestionItem(
                id="q_001",
                book_id=1,
                book_title="Designing Data-Intensive Applications",
                category="concept",
                question="What is the difference between leader-based and leaderless replication?",
                context_hint="Chapter 5: Replication",
                expected_aspects=["Quorums", "Split-brain mitigation", "Failover"],
            ),
            QuestionItem(
                id="q_002",
                book_id=1,
                book_title="Designing Data-Intensive Applications",
                category="relational",
                question="How does two-phase commit differ from Raft consensus?",
                context_hint="Chapter 9: Consistency and Consensus",
                expected_aspects=["Blocking nature of 2PC", "Fault-tolerant progress in Raft"],
            ),
        ],
    )

    save_path = tmp_path / "test_questions.json"
    saved = ql.save(save_path)
    assert saved.is_file()

    loaded = QuestionLibrary.load(save_path)
    assert len(loaded.questions) == 2
    assert loaded.model == "llama3.1:8b"
    assert loaded.questions[0].id == "q_001"
    assert loaded.questions[0].category == "concept"
    assert "Quorums" in loaded.questions[0].expected_aspects
    assert loaded.questions[1].id == "q_002"


def test_question_generator_with_mock_llm(sample_settings, tmp_path):
    """Verify QuestionGenerator parses LLM responses into structured QuestionLibrary."""
    generator = QuestionGenerator(settings=sample_settings, model_name="llama3.1:8b")

    mock_llm_json = {
        "questions": [
            {
                "category": "concept",
                "question": "What is log-structured merge-tree (LSM-tree)?",
                "context_hint": "Storage engines chapter",
                "expected_aspects": ["Memtable", "SSTables", "Compaction"],
            },
            {
                "category": "factual",
                "question": "Under what condition does read repair occur in Dynamo-style datastores?",
                "context_hint": "Leaderless replication section",
                "expected_aspects": ["Version vectors", "Stale replica update"],
            },
        ]
    }

    dummy_book = tmp_path / "book.txt"
    dummy_book.write_text("Chapter 1: Storage and Retrieval\nLSM-trees write sequentially to SSTables.", encoding="utf-8")

    books = [{"id": 10, "title": "Database Internals", "path": str(dummy_book)}]

    with patch.object(generator, "_call_llm_json", return_value=mock_llm_json):
        ql = generator.generate_library(books=books, questions_per_book=2)
        assert len(ql.questions) == 2
        assert ql.questions[0].category == "concept"
        assert "LSM-tree" in ql.questions[0].question
        assert ql.questions[1].category == "factual"


def test_lightrag_raw_indexer_text_extraction(sample_settings, tmp_path):
    """Verify LightRAGRawIndexer extracts clean text from text/markdown and parses correctly."""
    indexer = LightRAGRawIndexer(settings=sample_settings, working_dir=tmp_path / "lightrag")

    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Hello World! This is raw input text for natural chunking.", encoding="utf-8")

    text, char_count = indexer.extract_raw_book_text(sample_file)
    assert "Hello World!" in text
    assert char_count == len(text)


def test_lightrag_raw_indexer_index_books_mocked(sample_settings, tmp_path):
    """Verify LightRAGRawIndexer invokes insert_text per book."""
    indexer = LightRAGRawIndexer(settings=sample_settings, working_dir=tmp_path / "lightrag")

    mock_engine = MagicMock()
    indexer.engine = mock_engine

    sample_book = tmp_path / "book1.txt"
    sample_book.write_text("Raw text of book 1 for indexing.", encoding="utf-8")

    books = [{"id": 1, "title": "Book 1", "path": str(sample_book)}]
    stats = indexer.index_books(books=books, clean=False, show_progress=False)

    assert stats["total_books_indexed"] == 1
    assert mock_engine.insert_text.called
    assert mock_engine.insert_text.call_args[0][0] == "Raw text of book 1 for indexing."
    assert mock_engine.insert_text.call_args[1]["doc_id"] == "book_1"


def test_lightrag_service_query_mocked(sample_settings, tmp_path):
    """Verify LightRAGService queries engine and measures latency."""
    service = LightRAGService(settings=sample_settings, working_dir=tmp_path / "lightrag")

    mock_engine = MagicMock()
    mock_engine.query.return_value = "LightRAG synthesized consensus answer."
    service._engine = mock_engine

    res = service.query("What is Raft?", mode="hybrid")
    assert res["query"] == "What is Raft?"
    assert res["response"] == "LightRAG synthesized consensus answer."
    assert res["engine"] == "lightrag"
    assert "response_time" in res


from bookeeper.evaluation.benchmark import (
    ABBenchmarkRunner,
    BenchmarkJudge,
    BenchmarkReport,
    ChecklistCheckResult,
    JudgeEvaluation,
)
from bookeeper.evaluation.question_generator import (
    ChecklistItem,
    QuestionGenerator,
    QuestionItem,
    QuestionLibrary,
)


def test_benchmark_judge_evaluation_mocked(sample_settings):
    """Verify BenchmarkJudge evaluates responses against binary checklists."""
    judge = BenchmarkJudge(settings=sample_settings, model_name="llama3.1:8b")

    checklist = [
        ChecklistItem(criterion="Affirms term numbers are strictly monotonic", expected=True, aspect_label="Term numbers"),
        ChecklistItem(criterion="Denies asynchronous leaderless quorums", expected=False, aspect_label="Leader-based"),
    ]

    q_item = QuestionItem(
        id="q_1",
        book_id=1,
        book_title="Distributed Systems",
        category="concept",
        query="How does leader election operate in Raft?",
        checklist=checklist,
    )

    bk_res = {
        "response": "Leader election is initiated when a follower's election timer expires. Term numbers strictly increase.",
        "response_time": 0.05,
        "references": [{"reference_id": "b1", "file_path": "Distributed Systems"}],
    }

    lr_res = {
        "response": "Raft uses randomized timers.",
        "response_time": 1.25,
        "references": [],
    }

    # Side effects for 1st call (Bookkeeper) and 2nd call (LightRAG)
    bk_check_response = {
        "results": [
            {"check_index": 1, "verdict": True, "reason": "Term numbers strictly increase."},
            {"check_index": 2, "verdict": False, "reason": "Does not claim leaderless quorums."},
        ]
    }
    lr_check_response = {
        "results": [
            {"check_index": 1, "verdict": False, "reason": "Did not mention term numbers."},
            {"check_index": 2, "verdict": True, "reason": "Confused Raft with leaderless quorums."},
        ]
    }

    with patch.object(judge, "_call_llm_json", side_effect=[bk_check_response, lr_check_response]):
        ev = judge.evaluate(q_item, bk_res, lr_res)
        assert ev.question_id == "q_1"
        assert ev.winner == "bookkeeper"
        assert ev.bookkeeper_passed_count == 2
        assert ev.lightrag_passed_count == 0
        assert ev.bookkeeper_pass_rate == 100.0
        assert ev.lightrag_pass_rate == 0.0
        assert ev.bookkeeper_total_score == 5.0
        assert ev.lightrag_total_score == 1.0
        assert ev.bookkeeper_latency == 0.05
        assert ev.lightrag_latency == 1.25


def test_benchmark_report_markdown_generation(tmp_path):
    """Verify BenchmarkReport generates aligned markdown summary with checklist breakdown."""
    ev = JudgeEvaluation(
        question_id="q_001",
        query="Explain Raft consensus.",
        category="concept",
        book_title="Distributed Systems",
        bookkeeper_latency=0.04,
        lightrag_latency=2.10,
        bookkeeper_checks=[
            ChecklistCheckResult(criterion="Affirms monotonic terms", expected=True, actual=True, passed=True, reason="Terms increase monotonically"),
        ],
        lightrag_checks=[
            ChecklistCheckResult(criterion="Affirms monotonic terms", expected=True, actual=False, passed=False, reason="Omitted terms"),
        ],
        bookkeeper_passed_count=1,
        lightrag_passed_count=0,
        total_checks=1,
        bookkeeper_pass_rate=100.0,
        lightrag_pass_rate=0.0,
        grounding_bookkeeper=5.0,
        grounding_lightrag=1.0,
        completeness_bookkeeper=5.0,
        completeness_lightrag=1.0,
        clarity_bookkeeper=5.0,
        clarity_lightrag=1.0,
        relevance_bookkeeper=5.0,
        relevance_lightrag=1.0,
        bookkeeper_total_score=5.0,
        lightrag_total_score=1.0,
        winner="bookkeeper",
        rationale="Bookkeeper passed all invariant criteria.",
    )

    report = BenchmarkReport(
        judge_model="llama3.1:8b",
        book_titles=["Distributed Systems"],
        total_questions=1,
        total_checklist_items=1,
        bookkeeper_wins=1,
        lightrag_wins=0,
        ties=0,
        bookkeeper_avg_latency=0.04,
        lightrag_avg_latency=2.10,
        bookkeeper_total_passed_checks=1,
        lightrag_total_passed_checks=0,
        bookkeeper_overall_pass_rate=100.0,
        lightrag_overall_pass_rate=0.0,
        bookkeeper_avg_score=5.0,
        lightrag_avg_score=1.0,
        avg_grounding={"bookkeeper": 5.0, "lightrag": 1.0},
        avg_completeness={"bookkeeper": 5.0, "lightrag": 1.0},
        avg_clarity={"bookkeeper": 5.0, "lightrag": 1.0},
        avg_relevance={"bookkeeper": 5.0, "lightrag": 1.0},
        bookkeeper_nodes_count=4000,
        bookkeeper_edges_count=12000,
        lightrag_nodes_count=100,
        lightrag_edges_count=80,
        bookkeeper_chunks_count=1000,
        lightrag_chunks_count=20,
        evaluations=[ev],
    )

    md_file = tmp_path / "ab_test_summary.md"
    md_content = report.generate_markdown_summary(output_path=md_file)

    assert "A/B Benchmark Summary: Bookkeeper vs. LightRAG" in md_content
    assert "Checklist Invariant Pass Rate" in md_content
    assert "Knowledge Graph Structural Topology Comparison" in md_content
    assert "Total Graph Nodes" in md_content
    assert "Total Graph Edges" in md_content
    assert "4,000" in md_content
    assert "12,000" in md_content
    assert "Bookkeeper (A)" in md_content
    assert "LightRAG (B)" in md_content
    assert "Explain Raft consensus." in md_content
    assert "Affirms monotonic terms" in md_content
    assert md_file.is_file()


def test_cli_ab_benchmark_help():
    """Verify Typer CLI help output for new benchmark and question generation commands."""
    res_ab = runner.invoke(app, ["ab-benchmark", "--help"])
    assert res_ab.exit_code == 0
    assert "Run an end-to-end comparative A/B benchmark between Bookkeeper and LightRAG" in res_ab.stdout
    assert "--questions" in res_ab.stdout
    assert "--lightrag-mode" in res_ab.stdout
    assert "--judge-model" in res_ab.stdout

    res_q = runner.invoke(app, ["generate-questions", "--help"])
    assert res_q.exit_code == 0
    assert "Generate an evaluation question library for N books" in res_q.stdout

    res_lr = runner.invoke(app, ["lightrag-index", "--help"])
    assert res_lr.exit_code == 0
    assert "Index N books into LightRAG directly using raw book text" in res_lr.stdout


def test_clean_json_str_variations():
    """Verify _clean_json_str handles markdown fences, think tags, and preamble noise."""
    from bookeeper.evaluation.question_generator import _clean_json_str

    # 1. Clean JSON string
    assert _clean_json_str('{"key": "value"}') == '{"key": "value"}'

    # 2. Markdown fenced JSON
    fenced = '```json\n{"key": "value"}\n```'
    assert _clean_json_str(fenced) == '{"key": "value"}'

    # 3. Thinking tags from reasoning models
    thinking = '<think>Let me formulate the question</think>\n```json\n{"questions": [{"question": "test"}]}\n```'
    assert _clean_json_str(thinking) == '{"questions": [{"question": "test"}]}'

    # 4. JSON Array
    arr = 'Here are your questions:\n[{"question": "test"}]\nHope this helps!'
    assert _clean_json_str(arr) == '[{"question": "test"}]'


def test_question_generator_call_llm_json_pool_integration(sample_settings):
    """Verify QuestionGenerator._call_llm_json delegates via OllamaPool.execute_with_failover."""
    from bookeeper.processing.ollama_pool import OllamaPool

    mock_pool = MagicMock(spec=OllamaPool)
    mock_pool.execute_with_failover.return_value = {"questions": [{"question": "Q1"}]}

    gen = QuestionGenerator(settings=sample_settings, pool=mock_pool)
    res = gen._call_llm_json("Sample prompt", system_prompt="System instructions")

    assert res == {"questions": [{"question": "Q1"}]}
    assert mock_pool.execute_with_failover.called


def test_ab_benchmark_runner_checkpoint_safe_points(sample_settings, tmp_path):
    """Verify checkpoint recording and safe-point queries."""
    runner = ABBenchmarkRunner(settings=sample_settings, output_dir=tmp_path / "session_1")

    assert not runner.is_phase1_complete()
    assert not runner.is_phase2_complete()

    # Record Safe Point 1
    dummy_kg = tmp_path / "knowledge_graph.json"
    dummy_kg.write_text('{"nodes": [1, 2, 3]}', encoding="utf-8")
    runner.record_safe_point_1(dummy_kg, 1, books=[{"id": 1, "title": "Test Book"}])

    assert runner.is_phase1_complete()
    assert not runner.is_phase2_complete()

    cp = runner.load_checkpoint()
    assert cp.get("phase1_bookkeeper_completed") is True
    assert cp.get("bookkeeper_books_count") == 1

    # Record Safe Point 2
    dummy_storage = tmp_path / "lightrag_storage"
    dummy_storage.mkdir()
    runner.record_safe_point_2(dummy_storage, books=[{"id": 1, "title": "Test Book"}])

    assert runner.is_phase2_complete()
    cp = runner.load_checkpoint()
    assert cp.get("phase2_lightrag_completed") is True

    # Record Safe Point 3
    dummy_report = tmp_path / "benchmark_report.json"
    dummy_report.write_text("{}", encoding="utf-8")
    runner.record_safe_point_3(dummy_report)

    cp = runner.load_checkpoint()
    assert cp.get("phase3_queries_completed") is True


def test_lightrag_raw_indexer_continuation_skips_indexed(sample_settings, tmp_path):
    """Verify that LightRAGRawIndexer reuses already-indexed books from indexed_books.json."""
    indexer = LightRAGRawIndexer(settings=sample_settings, working_dir=tmp_path / "storage")

    # Mock get_engine to prevent Ollama network calls
    mock_engine = MagicMock()
    indexer.get_engine = MagicMock(return_value=mock_engine)

    book_file = tmp_path / "book1.txt"
    book_file.write_text("Chapter 1: Deep learning architectures.", encoding="utf-8")
    books = [{"id": 1, "title": "Deep Learning", "path": str(book_file)}]

    # First pass: clean indexing
    res1 = indexer.index_books(books=books, clean=True, show_progress=False)
    assert res1["total_books_indexed"] == 1
    assert mock_engine.insert_text.call_count == 1

    # Second pass: continuation (clean=False)
    mock_engine.insert_text.reset_mock()
    res2 = indexer.index_books(books=books, clean=False, show_progress=False)
    assert res2["total_books_indexed"] == 1
    # Book was already indexed, so insert_text should NOT be called again
    assert mock_engine.insert_text.call_count == 0


def test_search_engine_backend_local_kg_fallback(sample_settings, tmp_path):
    """Verify SearchEngineBackend falls back to querying knowledge_graph.json when Neo4j is offline."""
    from bookeeper.ui.backend import SearchEngineBackend

    # Create dummy local knowledge_graph.json
    kg_dir = tmp_path / "output"
    kg_dir.mkdir(parents=True, exist_ok=True)
    kg_path = kg_dir / "knowledge_graph.json"

    dummy_kg = {
        "nodes": [
            {
                "id": "concept:TimeHorizon",
                "name": "Time Horizon",
                "type": "Concept",
                "tag": "idea",
                "weight": 8,
                "brief_description": "Task completion threshold duration in AI evaluation.",
            },
            {
                "id": "book:1",
                "name": "AI Evaluation Methodology",
                "type": "Book",
                "tag": "book",
            },
        ],
        "edges": [
            {
                "source": "concept:TimeHorizon",
                "target": "book:1",
                "relation": "FEATURED_IN",
            }
        ],
    }
    kg_path.write_text(json.dumps(dummy_kg), encoding="utf-8")

    sample_settings.output_dir = str(kg_dir)
    backend = SearchEngineBackend(config=sample_settings)

    # Force neo4j driver to raise error (simulating offline Neo4j server)
    mock_driver = MagicMock()
    mock_driver.session.side_effect = Exception("Neo4j offline connection refused")
    backend.get_neo4j_driver = MagicMock(return_value=mock_driver)

    result = backend.query_ideas("What is a time horizon?")

    assert "Time Horizon" in result["identified_concepts"]
    assert "Task completion threshold duration" in result["answer"]
    assert len(result["graph"]["nodes"]) >= 1


