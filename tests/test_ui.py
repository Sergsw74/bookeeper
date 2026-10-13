"""
Tests for Bookeeper UI Web Server, REST API, and Search Engine Backend.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.config import Settings
from bookeeper.ui.backend import SearchEngineBackend
from bookeeper.ui.server import BookeeperUIRequestHandler, run_server

runner = CliRunner()


@pytest.fixture
def mock_backend(tmp_path):
    cfg = Settings()
    backend = SearchEngineBackend(config=cfg, lightrag_dir=tmp_path)
    return backend


def test_ui_backend_candidate_terms_extraction(mock_backend):
    sample_text = """
    Here is an analysis:
    - **Proper Scoring Rules**: crucial for evaluation
    - "Time Horizon": measured in minutes or days
    - [[Complexity Classes]]: mathematical foundation
    """
    candidates = mock_backend._extract_candidate_terms(sample_text)
    assert "Proper Scoring Rules" in candidates
    assert "Time Horizon" in candidates
    assert "Complexity Classes" in candidates


def test_ui_backend_query_ideas_with_mocks(mock_backend):
    # Mock Neo4j session and record returns
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Mock Neo4j node
    mock_node = MagicMock()
    mock_node.labels = ["Concept", "Idea"]
    mock_node.get.side_effect = lambda k, d=None: {
        "id": "concept:TimeHorizon",
        "name": "Time Horizon",
        "category": "Evaluation",
        "weight": 8,
    }.get(k, d)
    mock_node.keys.return_value = ["id", "name", "category", "weight"]
    mock_node.__iter__.return_value = ["id", "name", "category", "weight"]
    mock_node.__getitem__.side_effect = lambda k: {
        "id": "concept:TimeHorizon",
        "name": "Time Horizon",
        "category": "Evaluation",
        "weight": 8,
    }[k]

    mock_record = {
        "c": mock_node,
        "rel": None,
        "neighbor": None,
    }
    mock_session.run.return_value = [mock_record]

    with patch.object(mock_backend, "get_neo4j_driver", return_value=mock_driver):
        with patch.object(mock_backend, "get_lightrag_engine", return_value=None):
            res = mock_backend.query_ideas("What is a time horizon?")
            assert "graph" in res
            assert len(res["graph"]["nodes"]) == 1
            assert res["graph"]["nodes"][0]["name"] == "Time Horizon"
            assert "telemetry" in res
            assert res["telemetry"]["neo4j"]["status"] == "success"
            assert "exact_cypher" in res["telemetry"]["neo4j"]


def test_ui_backend_node_details_with_mocks(mock_backend):
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_node = MagicMock()
    mock_node.labels = ["Concept"]
    mock_node.get.side_effect = lambda k, d=None: {
        "id": "concept:TimeHorizon",
        "name": "Time Horizon",
        "summary": "Measures task time",
    }.get(k, d)
    mock_node.keys.return_value = ["id", "name", "summary"]
    mock_node.__iter__.return_value = ["id", "name", "summary"]
    mock_node.__getitem__.side_effect = lambda k: {
        "id": "concept:TimeHorizon",
        "name": "Time Horizon",
        "summary": "Measures task time",
    }[k]

    mock_rec = MagicMock()
    mock_rec.__getitem__.side_effect = lambda k: {
        "n": mock_node,
        "relations": [
            {
                "type": "FEATURED_IN",
                "direction": "outgoing",
                "other_id": "book:1",
                "other_name": "Sample AI Book",
                "other_labels": ["Book"],
                "other_category": "Book",
                "properties": {},
            }
        ],
    }[k]
    mock_session.run.return_value.single.return_value = mock_rec

    with patch.object(mock_backend, "get_neo4j_driver", return_value=mock_driver):
        details = mock_backend.get_node_details("concept:TimeHorizon")
        assert "node" in details
        assert details["node"]["name"] == "Time Horizon"
        assert len(details["books"]) == 1
        assert details["books"][0]["target_name"] == "Sample AI Book"
        assert "telemetry" in details


def test_ui_http_server_endpoints(mock_backend, tmp_path):
    import urllib.request
    from threading import Thread

    # Set up server on port 0 (dynamic port)
    server = run_server(host="127.0.0.1", port=0, backend=mock_backend)
    port = server.server_port
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Health check
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"

        # 2. Index HTML
        with urllib.request.urlopen(f"{base_url}/", timeout=3) as resp:
            html = resp.read().decode("utf-8")
            assert "Bookeeper | Idea-Based Search Engine" in html
            assert "Cytoscape" in html
            assert "Debug & Observability" in html

        # 3. Search POST API
        post_data = json.dumps({"query": "time horizon", "mode": "hybrid"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/search",
            data=post_data,
            headers={"Content-Type": "application/json"},
        )
        with patch.object(mock_backend, "query_ideas", return_value={
            "query": "time horizon",
            "answer": "Synthesized test answer",
            "graph": {"nodes": [], "edges": []},
            "telemetry": {"lightrag": {}, "neo4j": {}}
        }):
            with urllib.request.urlopen(req, timeout=3) as resp:
                search_res = json.loads(resp.read().decode("utf-8"))
                assert search_res["query"] == "time horizon"
                assert "Synthesized test answer" in search_res["answer"]
    finally:
        server.shutdown()
        server.server_close()


def test_ui_cli_help():
    res = runner.invoke(app, ["ui", "--help"])
    assert res.exit_code == 0
    assert "Launch the interactive Bookeeper Idea-Based Search Engine Web UI" in res.stdout
    assert "--host" in res.stdout
    assert "--port" in res.stdout
    assert "--lightrag-server" in res.stdout


def test_lightrag_ui_cli_help():
    res = runner.invoke(app, ["lightrag-ui", "--help"])
    assert res.exit_code == 0
    assert "Launch the official LightRAG FastAPI server with embedded WebUI" in res.stdout
    assert "--host" in res.stdout
    assert "--port" in res.stdout


def test_lightrag_server_manager_initialization(tmp_path):
    from bookeeper.ui.lightrag_server import LightRAGServerManager

    cfg = Settings()
    mgr = LightRAGServerManager(config=cfg, working_dir=tmp_path, host="127.0.0.1", port=9655)
    assert mgr.host == "127.0.0.1"
    assert mgr.port == 9655
    assert "http://127.0.0.1:9655/webui/" == mgr.webui_url
    assert "http://127.0.0.1:9655/workspace/" == mgr.workspace_url
    assert not mgr.is_alive()

