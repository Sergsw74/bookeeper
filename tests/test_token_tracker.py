"""
Unit tests for the Token Usage Statistics and Multi-Tier Pricing Subsystem.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from bookeeper.config import Settings, TokenPricingConfig, SectionPricing
from bookeeper.processing.token_tracker import (
    TokenCounts,
    TokenTracker,
    estimate_token_count,
)


def test_token_counts_and_segregation():
    """Verify TokenCounts maintains the four segregated sections and computes total tokens."""
    counts = TokenCounts(
        embedding_tokens=150,
        input_tokens=1000,
        output_tokens=300,
        cache_tokens=50,
    )

    assert counts.embedding_tokens == 150
    assert counts.input_tokens == 1000
    assert counts.output_tokens == 300
    assert counts.cache_tokens == 50
    assert counts.total_tokens == 1500

    d = counts.to_dict()
    assert d["embedding_tokens"] == 150
    assert d["input_tokens"] == 1000
    assert d["output_tokens"] == 300
    assert d["cache_tokens"] == 50
    assert d["total_tokens"] == 1500


def test_estimate_token_count():
    """Verify estimate_token_count handles strings accurately."""
    assert estimate_token_count("") == 0
    text = "The quick brown fox jumps over the lazy dog."
    cnt = estimate_token_count(text)
    assert cnt >= 8 and cnt <= 12


def test_token_tracker_recording():
    """Verify TokenTracker records tokens across all four categories."""
    tracker = TokenTracker.reset_instance()

    tracker.record_embedding_tokens(500)
    tracker.record_llm_tokens(input_tokens=2000, output_tokens=400, cache_tokens=100)

    snap = tracker.get_summary_snapshot()
    session = snap["session"]
    assert session["embedding_tokens"] == 500
    assert session["input_tokens"] == 2000
    assert session["output_tokens"] == 400
    assert session["cache_tokens"] == 100
    assert session["total_tokens"] == 3000


def test_token_tracker_record_response_data():
    """Verify recording from Ollama / OpenAI response dictionaries."""
    tracker = TokenTracker.reset_instance()

    # 1. Ollama embedding response
    tracker.record_response_data(
        {"prompt_eval_count": 256},
        task_type="embedding",
    )

    # 2. Ollama chat response with prompt caching
    tracker.record_response_data(
        {
            "prompt_eval_count": 1024,
            "eval_count": 128,
            "cache_tokens": 64,
        },
        task_type="llm",
    )

    snap = tracker.get_summary_snapshot()
    session = snap["session"]
    assert session["embedding_tokens"] == 256
    assert session["input_tokens"] == 1024
    assert session["output_tokens"] == 128
    assert session["cache_tokens"] == 64


def test_token_tracker_persistence_only_tokens(tmp_path: Path):
    """
    Verify that token statistics are stored ONLY in token terms,
    accumulating across sessions over time without saving monetary values.
    """
    stats_file = tmp_path / "token_usage_stats.json"
    tracker = TokenTracker.reset_instance()

    # Session 1: Process 2 books
    tracker.record_embedding_tokens(100_000)
    tracker.record_llm_tokens(input_tokens=500_000, output_tokens=50_000, cache_tokens=20_000)
    saved_path = tracker.save_cumulative(stats_file, books_processed=2)

    assert saved_path.is_file()
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check that file stores ONLY token numbers
    assert data["sessions_count"] == 1
    assert data["cumulative"]["embedding_tokens"] == 100_000
    assert data["cumulative"]["input_tokens"] == 500_000
    assert data["cumulative"]["output_tokens"] == 50_000
    assert data["cumulative"]["cache_tokens"] == 20_000
    assert data["cumulative"]["total_tokens"] == 670_000

    # Ensure no currency or prices are stored in JSON
    json_str = json.dumps(data)
    assert "$" not in json_str
    assert "cost" not in json_str
    assert "price" not in json_str

    # Session 2: Fresh session, load cumulative
    tracker2 = TokenTracker.reset_instance()
    tracker2.load_cumulative(stats_file)
    assert tracker2.cumulative_counts.total_tokens == 670_000

    tracker2.reset_session()
    tracker2.record_embedding_tokens(50_000)
    tracker2.record_llm_tokens(input_tokens=200_000, output_tokens=30_000, cache_tokens=10_000)
    tracker2.save_cumulative(stats_file, books_processed=1)

    with open(stats_file, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    assert data2["sessions_count"] == 2
    assert data2["cumulative"]["embedding_tokens"] == 150_000
    assert data2["cumulative"]["input_tokens"] == 700_000
    assert data2["cumulative"]["output_tokens"] == 80_000
    assert data2["cumulative"]["cache_tokens"] == 30_000
    assert data2["cumulative"]["total_tokens"] == 960_000


def test_token_tracker_price_calculations_and_summary_table():
    """Verify price calculation across three types of prices and rich table rendering."""
    tracker = TokenTracker.reset_instance()
    tracker.record_embedding_tokens(1_000_000)
    tracker.record_llm_tokens(input_tokens=2_000_000, output_tokens=500_000, cache_tokens=200_000)

    pricing = TokenPricingConfig(
        embedding_price_per_1m=0.10,
        input_price_per_1m=0.20,
        output_price_per_1m=0.80,
        cache_price_per_1m=0.05,
    )

    # 1. Price Type 1: Unit Rate ($/1M)
    assert pricing.embedding_price_per_1m == 0.10

    # 2. Price Type 2: Session Cost
    s_emb_cost = tracker.calculate_prices_for_section(1_000_000, pricing.embedding_price_per_1m)
    s_inp_cost = tracker.calculate_prices_for_section(2_000_000, pricing.input_price_per_1m)
    s_out_cost = tracker.calculate_prices_for_section(500_000, pricing.output_price_per_1m)
    s_cac_cost = tracker.calculate_prices_for_section(200_000, pricing.cache_price_per_1m)

    assert pytest.approx(s_emb_cost, 0.0001) == 0.10
    assert pytest.approx(s_inp_cost, 0.0001) == 0.40
    assert pytest.approx(s_out_cost, 0.0001) == 0.40
    assert pytest.approx(s_cac_cost, 0.0001) == 0.01

    # Print summary table using mock or rich console
    mock_console = MagicMock()
    tracker.print_summary_table(console=mock_console, pricing_config=pricing)
    assert mock_console.print.call_count >= 2


def test_settings_pricing_yaml_parsing():
    """Verify Settings parses pricing configuration from dictionary / YAML."""
    cfg = Settings(
        pricing={
            "embedding": 0.05,
            "input": 0.25,
            "output": 1.00,
            "cache": 0.05,
        }
    )

    assert cfg.pricing.embedding_price_per_1m == 0.05
    assert cfg.pricing.input_price_per_1m == 0.25
    assert cfg.pricing.output_price_per_1m == 1.00
    assert cfg.pricing.cache_price_per_1m == 0.05


def test_extractor_structured_invoke_records_tokens():
    """
    Verify KnowledgeExtractor._execute_structured_invoke actually records
    input and output tokens without crashing on undefined variables.
    """
    from bookeeper.processing.extractor import KnowledgeExtractor
    from langchain_core.messages import SystemMessage, HumanMessage
    from pydantic import BaseModel

    class DummyExtraction(BaseModel):
        summary: str = "Test"

    tracker = TokenTracker.reset_instance()
    mock_pool = MagicMock()
    # Mock pool.execute_with_failover to invoke the inner operation
    mock_pool.execute_with_failover.side_effect = lambda op, **kw: op("http://localhost:11434")

    extractor = KnowledgeExtractor(
        model="dummy:model",
        pool=mock_pool,
    )

    mock_ollama_resp = {
        "message": {"content": json.dumps({"summary": "Successfully extracted concept"})},
        "prompt_eval_count": 450,
        "eval_count": 85,
    }

    mock_run_result = MagicMock()
    mock_run_result.returncode = 0
    mock_run_result.stdout = json.dumps(mock_ollama_resp)

    from unittest.mock import patch
    with patch("subprocess.run", return_value=mock_run_result):
        messages = [
            SystemMessage(content="You are an expert extractor."),
            HumanMessage(content="Here is a book chunk about systems architecture."),
        ]
        res = extractor._execute_structured_invoke(DummyExtraction, messages)
        assert res.summary == "Successfully extracted concept"

    snap = tracker.get_summary_snapshot()
    session = snap["session"]
    assert session["input_tokens"] == 450
    assert session["output_tokens"] == 85
    assert session["total_tokens"] >= 535

