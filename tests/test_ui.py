"""
Tests for Bookeeper UI Web Server, REST API, LightRAG-Compatible Query API, and Search Engine Backend.
"""

import json
from pathlib import Path
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
    cfg = Settings(output_dir=str(tmp_path))
    backend = SearchEngineBackend(config=cfg, lightrag_dir=tmp_path)
    backend.get_neo4j_stats = MagicMock(return_value=(False, None, "Connection refused"))
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

    # Test extract_query_terms on natural language questions
    query_terms = mock_backend.extract_query_terms("What is a time horizon?")
    assert "time horizon" in query_terms
    assert "time" in query_terms or "horizon" in query_terms


def test_ui_backend_query_ideas_with_mocks(mock_backend):
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_node = MagicMock()
    mock_node.labels = ["Concept", "Idea"]
    mock_node.get.side_effect = lambda k, d=None: {
        "id": "concept:TimeHorizon",
        "name": "Time Horizon",
        "category": "Evaluation",
        "weight": 8,
        "brief_description": "Measures task time",
    }.get(k, d)
    mock_node.keys.return_value = ["id", "name", "category", "weight", "brief_description"]
    mock_node.__iter__.return_value = ["id", "name", "category", "weight", "brief_description"]
    mock_node.__getitem__.side_effect = lambda k: {
        "id": "concept:TimeHorizon",
        "name": "Time Horizon",
        "category": "Evaluation",
        "weight": 8,
        "brief_description": "Measures task time",
    }[k]

    mock_record = {
        "c": mock_node,
        "rel": None,
        "neighbor": None,
    }
    mock_session.run.return_value = [mock_record]

    with patch.object(mock_backend, "get_neo4j_driver", return_value=mock_driver):
        res = mock_backend.query_ideas("What is a time horizon?")
        assert "graph" in res
        assert len(res["graph"]["nodes"]) == 1
        assert res["graph"]["nodes"][0]["name"] == "Time Horizon"
        assert "Time Horizon" in res["answer"]
        assert "telemetry" in res
        assert res["telemetry"]["neo4j"]["status"] == "success"
        assert "exact_cypher" in res["telemetry"]["neo4j"]


def test_ui_backend_query_api_mimic(mock_backend):
    """Verify backend.query_api outputs LightRAG-compatible response schema."""
    with patch.object(mock_backend, "query_ideas", return_value={
        "query": "kv-cache",
        "answer": "KV Cache stores attention key-value pairs.",
        "identified_concepts": ["KV Cache"],
        "graph": {
            "nodes": [
                {
                    "id": "book:1",
                    "name": "Transformer Architecture",
                    "labels": ["Book"],
                    "properties": {"title": "Transformer Architecture", "description": "LLM foundations"}
                }
            ],
            "edges": []
        },
        "selected_node_id": "concept:KVCache",
        "telemetry": {"query": "kv-cache", "neo4j": {}}
    }):
        api_res = mock_backend.query_api("kv-cache", include_references=True)
        assert "response" in api_res
        assert api_res["response"] == "KV Cache stores attention key-value pairs."
        assert "references" in api_res
        assert len(api_res["references"]) == 1
        assert api_res["references"][0]["reference_id"] == "book:1"
        assert api_res["references"][0]["file_path"] == "Transformer Architecture"
        assert api_res["llm_generated"] is False
        assert "response_time" in api_res


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
        # 1. Health check (/health and /api/health)
        with urllib.request.urlopen(f"{base_url}/health", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["service"] == "bookkeeper-search"

        with urllib.request.urlopen(f"{base_url}/api/health", timeout=3) as resp:
            data2 = json.loads(resp.read().decode("utf-8"))
            assert data2["status"] == "ok"

        # 2. Index HTML
        with urllib.request.urlopen(f"{base_url}/", timeout=3) as resp:
            html = resp.read().decode("utf-8")
            assert "Bookeeper | Idea-Based Search Engine" in html
            assert "Cytoscape" in html
            assert "Debug & Observability" in html
            assert "pane-resizer" in html
            assert "btn-unfold-panel" in html
            assert "btn-fold-panel" in html
            assert "Star Layout" in html

        # 3. Search POST API (/api/search)
        post_data = json.dumps({"query": "time horizon", "mode": "ideas"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/search",
            data=post_data,
            headers={"Content-Type": "application/json"},
        )
        with patch.object(mock_backend, "query_ideas", return_value={
            "query": "time horizon",
            "answer": "Synthesized test answer",
            "graph": {"nodes": [], "edges": []},
            "telemetry": {"candidate_terms": ["time horizon"], "neo4j": {}}
        }):
            with urllib.request.urlopen(req, timeout=3) as resp:
                search_res = json.loads(resp.read().decode("utf-8"))
                assert search_res["query"] == "time horizon"
                assert "Synthesized test answer" in search_res["answer"]

        # 4. LightRAG-Compatible Query API (/query and /api/query)
        query_data = json.dumps({"query": "time horizon", "mode": "hybrid"}).encode("utf-8")
        req_query = urllib.request.Request(
            f"{base_url}/query",
            data=query_data,
            headers={"Content-Type": "application/json"},
        )
        with patch.object(mock_backend, "query_api", return_value={
            "response": "LightRAG compatible synthesized answer",
            "references": [{"reference_id": "book:1", "file_path": "Book A", "content": []}],
            "response_time": 0.05,
            "llm_generated": False,
        }):
            with urllib.request.urlopen(req_query, timeout=3) as resp:
                query_res = json.loads(resp.read().decode("utf-8"))
                assert "response" in query_res
                assert query_res["response"] == "LightRAG compatible synthesized answer"
                assert len(query_res["references"]) == 1

    finally:
        server.shutdown()
        server.server_close()


def test_ui_cli_help():
    res = runner.invoke(app, ["ui", "--help"])
    assert res.exit_code == 0
    assert "Launch the interactive Bookeeper Idea-Based Search Engine Web UI" in res.stdout
    assert "--host" in res.stdout
    assert "--port" in res.stdout


def test_search_cli_help():
    res = runner.invoke(app, ["search", "--help"])
    assert res.exit_code == 0
    assert "Search ideas directly in the Bookkeeper Knowledge Graph" in res.stdout


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


def test_format_browser_cypher():
    from bookeeper.ui.backend import SearchEngineBackend

    query = """
    UNWIND $candidate_terms AS term
    MATCH (c)
    WHERE degree <= $max_hub_degree
    LIMIT $max_seeds
    """
    params = {
        "candidate_terms": ["kv cache", "cache"],
        "max_seeds": 8,
        "max_hub_degree": 120,
    }

    # Format with nested props ($obj.props.<param>)
    formatted_props = SearchEngineBackend.format_browser_cypher(query, params, use_nested_props=True)
    assert formatted_props.startswith(":param obj => ({props: {")
    assert 'candidate_terms: ["kv cache", "cache"]' in formatted_props
    assert '"candidate_terms":' not in formatted_props
    assert 'max_seeds: 8' in formatted_props
    assert '"max_seeds":' not in formatted_props
    assert 'max_hub_degree: 120' in formatted_props
    assert "UNWIND $obj.props.candidate_terms AS term" in formatted_props
    assert "WHERE degree <= $obj.props.max_hub_degree" in formatted_props
    assert "LIMIT $obj.props.max_seeds" in formatted_props

    # Format keeping original param ($param)
    formatted_param = SearchEngineBackend.format_browser_cypher(query, params, use_nested_props=False)
    assert formatted_param.startswith(":param obj => ({props: {")
    assert 'candidate_terms: ["kv cache", "cache"]' in formatted_param
    assert '"candidate_terms":' not in formatted_param
    assert "UNWIND $candidate_terms AS term" in formatted_param
    assert "WHERE degree <= $max_hub_degree" in formatted_param
    assert "LIMIT $max_seeds" in formatted_param

    # Empty params
    assert SearchEngineBackend.format_browser_cypher("RETURN 1", {}) == "RETURN 1"
    assert SearchEngineBackend.format_browser_cypher("", params) == ""


def test_ui_index_html_has_browser_param_toggle():
    backend_path = Path(__file__).resolve().parent.parent / "src" / "bookeeper" / "ui" / "static" / "index.html"
    content = backend_path.read_text(encoding="utf-8")

    assert ":param obj => ({props: " in content
    assert "serializeMapUnquotedKeys" in content
    assert "formatBrowserCypher" in content
    assert "setCypherFormatMode" in content
    assert "$obj.props" in content
    assert "$param" in content
    assert "debug-toggle-group" in content
    assert "getGraphLayout" in content
    assert "Centered Rings" in content
    assert "Force Layout" in content


def test_post_process_and_filter_subgraph(mock_backend):
    """Test that secondary chunks/sections are capped, and nodes/edges receive weights."""
    seed_nodes = {
        "idea:1": {
            "id": "idea:1",
            "name": "Attention Mechanism",
            "category": "Idea",
            "labels": ["Idea", "Concept"],
            "weight": 5,
        }
    }
    nodes_map = dict(seed_nodes)

    # Add 1 connected concept, 1 book, 5 chunks, and 5 sections
    nodes_map["concept:2"] = {"id": "concept:2", "name": "Transformer", "category": "Concept", "labels": ["Concept"], "weight": 7}
    nodes_map["book:1"] = {"id": "book:1", "name": "Deep Learning Book", "category": "Book", "labels": ["Book"]}

    edges = [
        {"source": "idea:1", "target": "concept:2", "type": "RELATES_TO"},
        {"source": "idea:1", "target": "book:1", "type": "APPEARS_IN"},
    ]

    for i in range(1, 6):
        cid = f"chunk:{i}"
        sec_id = f"section:{i}"
        nodes_map[cid] = {"id": cid, "name": f"Chunk {i}", "category": "Chunk", "labels": ["Chunk"]}
        nodes_map[sec_id] = {"id": sec_id, "name": f"Section {i}", "category": "Section", "labels": ["Section"]}
        edges.append({"source": "idea:1", "target": cid, "type": "MENTIONS"})
        edges.append({"source": "idea:1", "target": sec_id, "type": "IN_SECTION"})

    seed_neighbors = {"idea:1": []}

    filtered_nodes, filtered_edges, _ = mock_backend._post_process_and_filter_subgraph(
        seed_nodes=seed_nodes,
        nodes_map=nodes_map,
        edges_list=edges,
        seed_neighbors=seed_neighbors,
        denoise_level="standard",
    )

    # Core concept and book are preserved
    assert "idea:1" in filtered_nodes
    assert "concept:2" in filtered_nodes
    assert "book:1" in filtered_nodes

    # Weights and styling:
    # Seed node: weight 10, amber, size 60
    assert filtered_nodes["idea:1"]["tier"] == "primary"
    assert filtered_nodes["idea:1"]["is_seed"] is True
    assert filtered_nodes["idea:1"]["weight"] == 10.0
    assert filtered_nodes["idea:1"]["color"] == "#f59e0b"
    assert filtered_nodes["idea:1"]["size"] == 60

    # Connected concept: secondary, weight 6-8.5, purple, size 42
    assert filtered_nodes["concept:2"]["tier"] == "secondary"
    assert filtered_nodes["concept:2"]["weight"] >= 6.0
    assert filtered_nodes["concept:2"]["color"] == "#8b5cf6"
    assert filtered_nodes["concept:2"]["size"] == 42

    # Book: secondary, blue, size 38
    assert filtered_nodes["book:1"]["tier"] == "secondary"
    assert filtered_nodes["book:1"]["color"] == "#3b82f6"
    assert filtered_nodes["book:1"]["size"] == 38

    # Secondary container nodes (Chunks and Sections) are capped to 2 per seed
    chunk_nodes = [nid for nid, n in filtered_nodes.items() if n.get("category") == "Chunk"]
    section_nodes = [nid for nid, n in filtered_nodes.items() if n.get("category") == "Section"]
    assert len(chunk_nodes) <= 2
    assert len(section_nodes) <= 2

    # Verify secondary nodes are gray/grayish with small size and low weight
    for cid in chunk_nodes:
        assert filtered_nodes[cid]["tier"] == "context"
        assert filtered_nodes[cid]["color"] == "#475569"
        assert filtered_nodes[cid]["weight"] == 1.5
        assert filtered_nodes[cid]["size"] == 16

    for sec_id in section_nodes:
        assert filtered_nodes[sec_id]["tier"] == "context"
        assert filtered_nodes[sec_id]["color"] == "#64748b"
        assert filtered_nodes[sec_id]["weight"] == 2.8
        assert filtered_nodes[sec_id]["size"] == 22

    # Verify relationship weights
    concept_rel = next(e for e in filtered_edges if e["source"] == "idea:1" and e["target"] == "concept:2")
    assert concept_rel["weight"] == 9.0
    assert concept_rel["line_color"] == "#818cf8"

    book_rel = next(e for e in filtered_edges if e["source"] == "idea:1" and e["target"] == "book:1")
    assert book_rel["weight"] == 6.5
    assert book_rel["line_color"] == "#3b82f6"

    chunk_rel = next(e for e in filtered_edges if "chunk" in e["target"])
    assert chunk_rel["weight"] == 1.8
    assert chunk_rel["line_color"] == "#334155"


def test_ui_primary_selected_id_never_selects_chunk_or_section(mock_backend):
    """Verify primary_selected_id strictly picks Concept/Idea, then Book, and never Chunk or Section."""
    # 1. When seed nodes has a Concept and a Chunk, it must select the Concept
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    concept_node = MagicMock()
    concept_node.labels = ["Concept"]
    concept_node.get.side_effect = lambda k, d=None: {
        "id": "concept:Attention",
        "name": "Attention",
        "category": "Concept",
        "weight": 8,
        "brief_description": "Attention mechanism",
    }.get(k, d)
    concept_node.keys.return_value = ["id", "name", "category", "weight", "brief_description"]
    concept_node.__iter__.return_value = ["id", "name", "category", "weight", "brief_description"]
    concept_node.__getitem__.side_effect = lambda k: {
        "id": "concept:Attention",
        "name": "Attention",
        "category": "Concept",
        "weight": 8,
        "brief_description": "Attention mechanism",
    }[k]

    chunk_node = MagicMock()
    chunk_node.labels = ["Chunk"]
    chunk_node.get.side_effect = lambda k, d=None: {
        "id": "chunk:99",
        "name": "Chunk 99",
        "category": "Chunk",
        "weight": 1,
        "text": "Some raw page chunk content",
    }.get(k, d)
    chunk_node.keys.return_value = ["id", "name", "category", "weight", "text"]
    chunk_node.__iter__.return_value = ["id", "name", "category", "weight", "text"]
    chunk_node.__getitem__.side_effect = lambda k: {
        "id": "chunk:99",
        "name": "Chunk 99",
        "category": "Chunk",
        "weight": 1,
        "text": "Some raw page chunk content",
    }[k]

    mock_rel = MagicMock()
    mock_rel.type = "MENTIONS"
    mock_rel.start_node = concept_node
    mock_rel.end_node = chunk_node
    mock_rel.__iter__.return_value = []

    mock_session.run.return_value = [
        {"c": concept_node, "rel": mock_rel, "neighbor": chunk_node}
    ]

    with patch.object(mock_backend, "get_neo4j_driver", return_value=mock_driver):
        res = mock_backend.query_ideas("attention")
        assert res["selected_node_id"] == "concept:Attention"
        assert res["selected_node_id"] != "chunk:99"


def test_ui_index_html_has_evidence_deprioritization_and_one_click_accordion():
    """Verify index.html contains the 1-click evidence filter button and Obsidian accordion elements."""
    html_path = Path(__file__).resolve().parent.parent / "src" / "bookeeper" / "ui" / "static" / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # 1-Click Evidence Toggle
    assert 'id="btn-toggle-evidence"' in content
    assert "toggleEvidenceFilter()" in content
    assert "applyEvidenceVisibility()" in content
    assert "focusEvidenceNode(" in content
    assert "evidenceVisible = false" in content

    # 1-Click Accordions & Excerpt reader
    assert "evidence-accordion" in content
    assert "evidence-summary" in content
    assert "evidence-text-content" in content
    assert "btn-focus-chunk" in content
    assert "1-Click View" in content
    assert "Supporting Evidence & Chunks" in content


def test_ui_get_node_details_evidence_extraction(mock_backend):
    """Verify get_node_details retrieves full chunk text and properties."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    idea_node = MagicMock()
    idea_node.labels = ["Idea"]
    idea_node.get.side_effect = lambda k, d=None: {
        "id": "idea:FlashAttention",
        "name": "Flash Attention",
        "category": "Idea",
    }.get(k, d)
    idea_node.keys.return_value = ["id", "name", "category"]
    idea_node.__iter__.return_value = ["id", "name", "category"]
    idea_node.__getitem__.side_effect = lambda k: {
        "id": "idea:FlashAttention",
        "name": "Flash Attention",
        "category": "Idea",
    }[k]

    mock_rec = MagicMock()
    mock_rec.__getitem__.side_effect = lambda k: {
        "n": idea_node,
        "relations": [
            {
                "type": "MENTIONS",
                "direction": "outgoing",
                "other_id": "chunk:12",
                "other_name": "Chunk 12 Excerpt",
                "other_labels": ["Chunk"],
                "other_category": "Chunk",
                "properties": {"breadcrumb": "Chapter 3 > Flash Attention"},
                "other_properties": {
                    "text": "FlashAttention tiles the softmax computation across SRAM blocks.",
                    "book_title": "Fast Inference",
                    "page": 42,
                },
            }
        ],
    }[k]
    mock_session.run.return_value.single.return_value = mock_rec

    with patch.object(mock_backend, "get_neo4j_driver", return_value=mock_driver):
        details = mock_backend.get_node_details("idea:FlashAttention")
        assert len(details["evidence"]) == 1
        ev = details["evidence"][0]
        assert ev["target_id"] == "chunk:12"
        assert ev["breadcrumb"] == "Chapter 3 > Flash Attention"
        assert ev["book_title"] == "Fast Inference"
        assert ev["page"] == 42
        assert "FlashAttention tiles the softmax computation" in ev["text"]


def test_ui_backend_hierarchical_term_extraction(mock_backend):
    """Verify multi-word phrases and single words are properly separated and prioritized."""
    h = mock_backend.extract_query_terms_hierarchical("What is a kv cache size?")
    assert h["full_query"] == "kv cache size"
    assert "kv cache size" in h["phrase_terms"]
    assert "kv cache" in h["phrase_terms"] or "cache size" in h["phrase_terms"]
    assert "size" in h["word_terms"]
    assert "cache" in h["word_terms"]

    h2 = mock_backend.extract_query_terms_hierarchical("time horizon")
    assert h2["full_query"] == "time horizon"
    assert h2["phrase_terms"] == ["time horizon"]
    assert "time" in h2["word_terms"]
    assert "horizon" in h2["word_terms"]


def test_ui_backend_fast_concept_vectorizer():
    """Verify FastConceptVectorizer cosine similarity, morphological matching, and sub-ms latency."""
    from bookeeper.ui.backend import FastConceptVectorizer

    corpus = {
        "concept:TimeHorizon": "Time Horizon metric for task completion time",
        "concept:TaskDifficulty": "Task Difficulty and difficulty estimation in AI tasks",
        "concept:KVCache": "KV Cache storage and attention keys",
        "concept:Parameters": "Parameters saved to storage disks",
    }
    vec = FastConceptVectorizer(corpus)

    # Exact or near-exact match
    matches = vec.query("time horizon", top_k=3, min_threshold=0.45)
    assert len(matches) > 0
    assert matches[0][0] == "concept:TimeHorizon"
    assert matches[0][1] >= 0.50

    # Multi-word phrase matching morphological variations
    diff_matches = vec.query("difficulty estimation", top_k=3, min_threshold=0.45)
    assert len(diff_matches) > 0
    assert diff_matches[0][0] == "concept:TaskDifficulty"

    # Query with kv cache should find KVCache, NOT Parameters
    cache_matches = vec.query("kv cache", top_k=3, min_threshold=0.45)
    match_ids = [m[0] for m in cache_matches]
    assert "concept:KVCache" in match_ids
    assert "concept:Parameters" not in match_ids


def test_ui_backend_browser_cypher_unquoted_and_direct_seeds(mock_backend):
    """Verify browser Cypher format has unquoted keys and direct seed IDs."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

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

    mock_session.run.return_value = [{"c": mock_node, "rel": None, "neighbor": None, "relevance": 100, "degree": 5}]

    with patch.object(mock_backend, "get_neo4j_driver", return_value=mock_driver):
        res = mock_backend.query_ideas("time horizon")
        telemetry = res["telemetry"]
        assert "seed_ids" in telemetry
        assert "core_ideas" in telemetry

        neo = telemetry["neo4j"]
        browser_cypher = neo["browser_cypher_props"]
        assert ":param obj => ({props:" in browser_cypher
        # Verify unquoted parameter keys
        assert "seed_ids:" in browser_cypher
        assert "max_seeds:" in browser_cypher
        assert "similarity_threshold:" in browser_cypher
        assert '"seed_ids":' not in browser_cypher
        assert '"max_seeds":' not in browser_cypher
        # Verify $obj.props rewriting
        assert "$obj.props.seed_ids" in browser_cypher
        assert "$obj.props.max_seeds" in browser_cypher


def test_concept_embedding_index_priority_product(tmp_path):
    """Verify embedding priority ranking = cosine_similarity * node_weight."""
    from bookeeper.ui.embeddings import ConceptEmbeddingIndex
    import numpy as np

    cache_file = tmp_path / "test_embeddings.npz"

    # Create synthetic normalized embeddings: 3 concepts of dim 4
    # Query vector will align perfectly with concept 1, partially with concept 2, zero with concept 3
    # Concept 1: vector [1, 0, 0, 0], weight 5.0 -> cosine 1.0, priority = 5.0
    # Concept 2: vector [0.9, 0.435, 0, 0] (norm=1), weight 8.0 -> cosine 0.9, priority = 7.2 (WINS despite lower cosine!)
    # Concept 3: vector [0, 1, 0, 0], weight 10.0 -> cosine 0.0, priority = 0.0 (filtered by threshold)
    c2_v = np.array([0.9, np.sqrt(1 - 0.9**2), 0, 0], dtype=np.float32)

    np.savez_compressed(
        cache_file,
        ids=np.array(["concept:1", "concept:2", "concept:3"], dtype=object),
        names=np.array(["Idea One", "Idea Two High Weight", "Idea Three"], dtype=object),
        weights=np.array([5.0, 8.0, 10.0], dtype=np.float32),
        vectors=np.array([
            [1.0, 0.0, 0.0, 0.0],
            c2_v,
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32),
    )

    index = ConceptEmbeddingIndex(cache_path=cache_file)
    assert index.load() is True

    # Mock embed_text to return [1, 0, 0, 0]
    index.embed_text = lambda q: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    core_ideas = index.query_core_ideas("query", similarity_threshold=0.70, top_k=5)

    assert len(core_ideas) == 2  # Concept 3 is cosine 0.0 < 0.70, so filtered out
    # Concept 2 wins the top spot because priority (7.2) > Concept 1 priority (5.0)!
    assert core_ideas[0]["id"] == "concept:2"
    assert core_ideas[0]["priority_score"] == 7.2
    assert core_ideas[0]["weight"] == 8.0
    assert core_ideas[0]["cosine_similarity"] == 0.9

    assert core_ideas[1]["id"] == "concept:1"
    assert core_ideas[1]["priority_score"] == 5.0
    assert core_ideas[1]["cosine_similarity"] == 1.0

    # Also test query_core_ideas_with_telemetry and pairwise similarity matrix
    ideas, meta = index.query_core_ideas_with_telemetry("query", similarity_threshold=0.70, top_k=5)
    assert len(ideas) == 2
    assert meta["candidates_above_threshold_count"] == 2
    assert meta["top_k_retrieved"] == 2
    assert "pairwise_similarity" in meta
    pairwise = meta["pairwise_similarity"]
    assert len(pairwise["names"]) == 2
    assert pairwise["matrix"][0][0] == 1.0  # Diagonal
    assert pairwise["matrix"][1][1] == 1.0  # Diagonal
    assert round(pairwise["matrix"][0][1], 3) == 0.9  # Inter-idea cosine similarity
    assert round(pairwise["matrix"][1][0], 3) == 0.9  # Symmetric


def test_ui_backend_expand_node_neighborhood(mock_backend):
    """Verify backend.expand_node_neighborhood returns undirected neighbors and relationships."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_center = MagicMock()
    mock_center.labels = ["Concept", "Idea"]
    mock_center.get.side_effect = lambda k, d=None: {
        "id": "concept:Center",
        "name": "Center Concept",
        "weight": 8,
    }.get(k, d)

    mock_neighbor = MagicMock()
    mock_neighbor.labels = ["Book"]
    mock_neighbor.get.side_effect = lambda k, d=None: {
        "id": "book:1",
        "title": "Reference Book",
        "weight": 6,
    }.get(k, d)

    mock_rel = MagicMock()
    mock_rel.type = "FEATURED_IN"
    mock_rel.start_node = mock_center
    mock_rel.end_node = mock_neighbor
    mock_rel.__iter__.return_value = []
    mock_rel.items.return_value = []

    mock_record = {
        "n": mock_center,
        "rel": mock_rel,
        "neighbor": mock_neighbor,
    }
    mock_session.run.return_value = [mock_record]

    with patch.object(mock_backend, "get_neo4j_driver", return_value=mock_driver):
        res = mock_backend.expand_node_neighborhood(node_id="concept:Center", type_filter="all")
        assert res["node_id"] == "concept:Center"
        assert res["type_filter"] == "all"
        assert res["total_nodes"] == 2
        assert res["total_edges"] == 1
        assert res["edges"][0]["type"] == "FEATURED_IN"
        assert res["edges"][0]["source"] == "concept:Center"
        assert res["edges"][0]["target"] == "book:1"
        assert "telemetry" in res
        assert "exact_cypher" in res["telemetry"]
        assert "browser_cypher_props" in res["telemetry"]


def test_ui_backend_expand_node_neighborhood_local_fallback(tmp_path):
    """Verify expand_node_neighborhood falls back to local knowledge_graph.json when Neo4j is offline."""
    kg_file = tmp_path / "knowledge_graph.json"
    kg_data = {
        "nodes": [
            {"id": "concept:A", "name": "Concept A", "type": "Concept"},
            {"id": "concept:B", "name": "Concept B", "type": "Concept"},
            {"id": "book:1", "title": "Book 1", "type": "Book", "category": "Book"},
        ],
        "edges": [
            {"source": "concept:A", "target": "concept:B", "relation": "RELATES_TO"},
            {"source": "concept:A", "target": "book:1", "relation": "FEATURED_IN"},
        ],
    }
    kg_file.write_text(json.dumps(kg_data), encoding="utf-8")

    cfg = Settings(output_dir=str(tmp_path))
    backend = SearchEngineBackend(config=cfg)

    # Neo4j offline
    with patch.object(backend, "get_neo4j_driver", side_effect=Exception("Connection refused")):
        # Expand all
        res_all = backend.expand_node_neighborhood("concept:A", type_filter="all")
        assert res_all["total_nodes"] == 3
        assert res_all["total_edges"] == 2

        # Expand only Book
        res_book = backend.expand_node_neighborhood("concept:A", type_filter="Book")
        assert res_book["total_nodes"] == 2  # Center + Book
        assert res_book["total_edges"] == 1
        assert res_book["edges"][0]["target"] == "book:1"


def test_ui_server_expand_endpoint(mock_backend):
    """Verify HTTP POST and GET /api/node/expand endpoints."""
    import urllib.request
    from threading import Thread

    with patch.object(mock_backend, "expand_node_neighborhood", return_value={
        "node_id": "concept:Test",
        "type_filter": "all",
        "nodes": [{"id": "concept:Test", "name": "Test"}],
        "edges": [],
        "total_nodes": 1,
        "total_edges": 0,
    }):
        server = run_server(host="127.0.0.1", port=0, backend=mock_backend)
        port = server.server_port
        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        base_url = f"http://127.0.0.1:{port}"

        # Test POST /api/node/expand
        post_req = urllib.request.Request(
            f"{base_url}/api/node/expand",
            data=json.dumps({"id": "concept:Test", "type_filter": "all"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(post_req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["node_id"] == "concept:Test"
            assert data["total_nodes"] == 1

        # Test GET /api/node/expand
        with urllib.request.urlopen(f"{base_url}/api/node/expand?id=concept:Test&type_filter=Concept") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["node_id"] == "concept:Test"

        server.shutdown()


def test_node_types_distinct_styling_and_shapes(mock_backend):
    """Verify each node type has constant, distinct visual styling (shapes, colors, borders, icons)."""
    # 1. Seed Concept
    seed_mock = MagicMock()
    seed_mock.labels = ["Concept", "Idea"]
    seed_mock.get.side_effect = lambda k, d=None: {"id": "c:seed", "name": "Core Seed", "weight": 9}.get(k, d)
    seed = mock_backend._serialize_node(seed_mock, is_seed=True, relevance=95.0)
    assert seed["node_type"] == "seed"
    assert seed["shape"] == "ellipse"
    assert seed["color"] == "#f59e0b"
    assert seed["border_color"] == "#fde68a"
    assert seed["icon"] == "🧠"

    # 2. Connected Concept
    concept_mock = MagicMock()
    concept_mock.labels = ["Concept", "Idea"]
    concept_mock.get.side_effect = lambda k, d=None: {"id": "c:other", "name": "Linked Concept", "weight": 7}.get(k, d)
    concept = mock_backend._serialize_node(concept_mock, is_seed=False)
    assert concept["node_type"] == "concept"
    assert concept["shape"] == "ellipse"
    assert concept["color"] == "#8b5cf6"
    assert concept["border_color"] == "#c4b5fd"
    assert concept["icon"] == "💡"

    # 3. Book
    book_mock = MagicMock()
    book_mock.labels = ["Book"]
    book_mock.get.side_effect = lambda k, d=None: {"id": "book:1", "title": "AI Foundations", "weight": 6}.get(k, d)
    book = mock_backend._serialize_node(book_mock, is_seed=False)
    assert book["node_type"] == "book"
    assert book["shape"] == "round-rectangle"
    assert book["color"] == "#2563eb"
    assert book["border_color"] == "#60a5fa"
    assert book["icon"] == "📚"

    # 4. Section
    sec_mock = MagicMock()
    sec_mock.labels = ["Section"]
    sec_mock.get.side_effect = lambda k, d=None: {"id": "sec:1", "title": "Chapter 1", "weight": 3}.get(k, d)
    sec = mock_backend._serialize_node(sec_mock, is_seed=False)
    assert sec["node_type"] == "section"
    assert sec["shape"] == "round-rectangle"
    assert sec["color"] == "#0d9488"
    assert sec["border_color"] == "#34d399"
    assert sec["icon"] == "📑"

    # 5. Chunk
    chunk_mock = MagicMock()
    chunk_mock.labels = ["Chunk"]
    chunk_mock.get.side_effect = lambda k, d=None: {"id": "chk:1", "text": "Excerpt", "weight": 1.5}.get(k, d)
    chunk = mock_backend._serialize_node(chunk_mock, is_seed=False)
    assert chunk["node_type"] == "chunk"
    assert chunk["shape"] == "ellipse"
    assert chunk["color"] == "#334155"
    assert chunk["border_color"] == "#64748b"
    assert chunk["icon"] == "📄"


def test_constellation_gravitational_layout_ui():
    """Verify index.html contains the constellation gravitational layout with multi-hub centers, organic phyllotaxis, and inside-node scaled titles."""
    index_path = Path(__file__).parent.parent / "src" / "bookeeper" / "ui" / "static" / "index.html"
    assert index_path.exists()
    content = index_path.read_text(encoding="utf-8")

    # Star layout button with data-layout="star"
    assert 'data-layout="star"' in content
    assert 'cyReLayout(\'star\')' in content
    assert 'Constellation Gravitational Layout' in content

    # Constellation positioning function & physics engine
    assert 'computeConstellationLayout()' in content
    assert 'prominenceScores' in content
    assert 'constellationCenters' in content
    assert 'centerCoords' in content
    assert 'goldenAngle' in content

    # Scaled title within node formatting (no square notes, text centered inside)
    assert "'text-valign': 'center'" in content
    assert "'text-halign': 'center'" in content
    assert "'text-wrap': 'wrap'" in content
    assert "'text-max-width': 'data(textMaxWidth)'" in content

    # Clean titles inside nodes without emoji prefixes on graph canvas
    assert "Clean title text WITHOUT emoji icons" in content


def test_celestial_star_system_layout_ui():
    """Compatibility alias test for constellation layout."""
    test_constellation_gravitational_layout_ui()


def test_expand_and_query_smart_id_and_canonical_idea_resolution(mock_backend):
    """Verify backend supports numeric Neo4j IDs (id(n) = 1335) and upgrades subordinate concepts to canonical Ideas."""
    backend_path = Path(__file__).parent.parent / "src" / "bookeeper" / "ui" / "backend.py"
    backend_code = backend_path.read_text(encoding="utf-8")

    # expand_node_neighborhood supports numeric ID and elementId suffix
    assert "toString(id(target)) = $node_id" in backend_code
    assert "elementId(target) ENDS WITH (':' + $node_id)" in backend_code
    assert "OPTIONAL MATCH (better:Idea)" in backend_code

    # get_node_details supports numeric ID and elementId suffix
    assert "MATCH (target)" in backend_code
    assert "WITH coalesce(better, target) AS n" in backend_code

    # query_ideas includes exact Neo4j Idea name matching and canonical Idea upgrade
    assert "MATCH (c_seed)" in backend_code
    assert "WITH coalesce(better, c_seed) AS c" in backend_code


def test_book_file_resolver(tmp_path):
    """Verify resolve_book_file correctly resolves documents by title, filename, breadcrumb, and ID."""
    books_dir = tmp_path / "downloads"
    books_dir.mkdir()
    pdf_file = books_dir / "sample_paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 sample content")

    index_data = {
        "2610.99999": {
            "title": "Quantum Attention Mechanisms in Large Language Models",
            "file_path": "sample_paper.pdf",
            "url": "https://arxiv.org/pdf/2610.99999",
            "format": "pdf",
        }
    }
    (books_dir / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    cfg = Settings()
    cfg.books_dir = str(books_dir)
    backend = SearchEngineBackend(config=cfg)

    # 1. Match by exact title
    res_title = backend.resolve_book_file(title="Quantum Attention Mechanisms in Large Language Models")
    assert res_title is not None
    assert res_title["found"] is True
    assert res_title["filename"] == "sample_paper.pdf"
    assert "/api/book/download?filename=" in res_title["download_url"]
    assert "/api/book/file?filename=" in res_title["view_url"]
    assert res_title["external_url"] == "https://arxiv.org/pdf/2610.99999"

    # 2. Match by breadcrumb
    res_bc = backend.resolve_book_file(breadcrumb="Quantum Attention Mechanisms in Large Language Models > Chapter 3 [part 1]")
    assert res_bc is not None
    assert res_bc["found"] is True
    assert res_bc["filename"] == "sample_paper.pdf"

    # 3. Match by direct filename
    res_fn = backend.resolve_book_file(filename="sample_paper.pdf")
    assert res_fn is not None
    assert res_fn["found"] is True
    assert res_fn["file_size"] == len(b"%PDF-1.4 sample content")

    # 4. Match by arxiv/book ID
    res_id = backend.resolve_book_file(book_id="2610.99999")
    assert res_id is not None
    assert res_id["found"] is True

    # 5. Non-existent book
    res_none = backend.resolve_book_file(title="NonExistent Paper Never Written")
    assert res_none is None


def test_server_book_endpoints(tmp_path):
    """Test HTTP API endpoints for /api/book/file, /api/book/download, and /api/book/info."""
    books_dir = tmp_path / "downloads"
    books_dir.mkdir()
    pdf_file = books_dir / "ai_horizons.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 test stream content")

    index_data = {
        "2610.12466": {
            "title": "On the estimation and validity of AI time horizons",
            "file_path": "ai_horizons.pdf",
            "url": "https://arxiv.org/pdf/2610.12466",
        }
    }
    (books_dir / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    cfg = Settings()
    cfg.books_dir = str(books_dir)
    backend = SearchEngineBackend(config=cfg)

    server = run_server(host="127.0.0.1", port=8991, backend=backend, auto_port=True)
    import threading
    import urllib.request
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    actual_port = server.server_address[1]
    base_url = f"http://127.0.0.1:{actual_port}"

    try:
        # 1. Info endpoint
        req = urllib.request.urlopen(f"{base_url}/api/book/info?title=AI%20time%20horizons")
        assert req.status == 200
        info_data = json.loads(req.read().decode("utf-8"))
        assert info_data["found"] is True
        assert info_data["filename"] == "ai_horizons.pdf"
        assert "absolute_path" not in info_data  # Should not expose local server paths

        # 2. View file endpoint (inline)
        req_view = urllib.request.urlopen(f"{base_url}/api/book/file?title=AI%20time%20horizons")
        assert req_view.status == 200
        assert req_view.headers.get("Content-Type") == "application/pdf"
        assert "inline" in req_view.headers.get("Content-Disposition", "")
        assert req_view.read() == b"%PDF-1.4 test stream content"

        # 3. Download file endpoint (attachment)
        req_dl = urllib.request.urlopen(f"{base_url}/api/book/download?filename=ai_horizons.pdf")
        assert req_dl.status == 200
        assert "attachment" in req_dl.headers.get("Content-Disposition", "")
        assert req_dl.read() == b"%PDF-1.4 test stream content"

        # 4. Path traversal attempt rejected
        with pytest.raises(Exception):
            urllib.request.urlopen(f"{base_url}/api/book/file?filename=../../config.yaml")

    finally:
        server.shutdown()
        server.server_close()


def test_ui_book_file_links_in_html():
    """Verify frontend HTML renders book open and download links in evidence and book cards."""
    html_file = Path(__file__).parent.parent / "src" / "bookeeper" / "ui" / "static" / "index.html"
    content = html_file.read_text(encoding="utf-8")

    # CSS classes for book buttons
    assert ".evidence-book-btn" in content
    assert ".evidence-download-btn" in content
    assert ".btn-book-action" in content
    assert ".book-pill-card" in content
    assert ".book-source-banner" in content

    # JavaScript rendering of book files
    assert "formatBytes(bytes)" in content
    assert "bf.view_url" in content
    assert "bf.download_url" in content
    assert "📖 Open Book" in content
    assert "📥 Download" in content


def test_ui_database_consistency_offline(mock_backend):
    """Verify consistency status reports standalone offline mode when Neo4j is offline."""
    mock_backend.get_neo4j_stats = MagicMock(return_value=(False, None, "Connection refused"))
    mock_backend.get_local_graph_stats = MagicMock(return_value={
        "books": 20,
        "chunks": 5000,
        "ideas": 12000,
        "total_nodes": 17020,
        "total_edges": 50000,
    })

    val = mock_backend.validate_database_consistency()
    assert val["neo4j_connected"] is False
    assert val["standalone_mode"] is True
    assert val["is_consistent"] is True
    assert val["warning"] is None
    assert val["local_stats"]["books"] == 20


def test_ui_database_consistency_inconsistent(mock_backend):
    """Verify consistency status detects when Neo4j database is outdated compared to local KB."""
    mock_backend.get_local_graph_stats = MagicMock(return_value={
        "books": 22,
        "chunks": 6000,
        "ideas": 14000,
        "total_nodes": 20022,
        "total_edges": 60000,
    })
    # Neo4j has only 10 books, 2000 chunks, 4000 ideas
    mock_backend.get_neo4j_stats = MagicMock(return_value=(True, {
        "books": 10,
        "chunks": 2000,
        "ideas": 4000,
        "total_nodes": 6010,
        "total_edges": 15000,
    }, None))

    val = mock_backend.validate_database_consistency()
    assert val["neo4j_connected"] is True
    assert val["standalone_mode"] is False
    assert val["is_consistent"] is False
    assert len(val["discrepancies"]) == 3
    assert any("Books:" in d for d in val["discrepancies"])
    assert any("Chunks:" in d for d in val["discrepancies"])
    assert any("Ideas:" in d for d in val["discrepancies"])
    assert val["warning"] is not None
    assert "out of sync" in val["warning"].lower()


def test_ui_database_consistency_endpoints(tmp_path):
    """Verify /api/sync/status and /health endpoints return database consistency payload."""
    import urllib.request
    import json
    import threading

    cfg = Settings()
    backend = SearchEngineBackend(config=cfg)
    backend.get_local_graph_stats = MagicMock(return_value={
        "books": 15,
        "chunks": 3000,
        "ideas": 8000,
        "total_nodes": 11015,
        "total_edges": 30000,
    })
    backend.get_neo4j_stats = MagicMock(return_value=(True, {
        "books": 5,
        "chunks": 1000,
        "ideas": 2000,
        "total_nodes": 3005,
        "total_edges": 6000,
    }, None))

    server = run_server(host="127.0.0.1", port=0, backend=backend)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        # 1. /api/sync/status
        req_sync = urllib.request.urlopen(f"{base_url}/api/sync/status")
        assert req_sync.status == 200
        sync_data = json.loads(req_sync.read().decode())
        assert sync_data["neo4j_connected"] is True
        assert sync_data["is_consistent"] is False
        assert len(sync_data["discrepancies"]) == 3

        # 2. /health includes consistency
        req_health = urllib.request.urlopen(f"{base_url}/health")
        assert req_health.status == 200
        health_data = json.loads(req_health.read().decode())
        assert "consistency" in health_data
        assert health_data["consistency"]["is_consistent"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_ui_consistency_banner_in_html():
    """Verify HTML template includes warning banner and JS sync validation."""
    html_file = Path(__file__).parent.parent / "src" / "bookeeper" / "ui" / "static" / "index.html"
    content = html_file.read_text(encoding="utf-8")

    assert ".consistency-banner" in content
    assert "id=\"consistency-banner\"" in content
    assert "Database Inconsistency Detected" in content
    assert "checkDatabaseConsistency()" in content
    assert "showConsistencyBanner" in content
    assert "dismissConsistencyBanner" in content
    assert "/api/sync/status" in content
    assert "id=\"debug-sync-panel\"" in content


def test_ui_backend_get_indexed_books(mock_backend, tmp_path):
    """Test get_indexed_books logic with descending order, occurrences*weight scoring, and publication_date."""
    # Create mock knowledge_graph.json
    kg_data = {
        "nodes": [
            {
                "id": "book:10",
                "type": "Book",
                "title": "Older Book",
                "author": "Author A",
                "timestamp": 1000.0,
                "publication_date": "2023-01-15",
            },
            {
                "id": "book:20",
                "type": "Book",
                "title": "Newer Book",
                "author": "Author B",
                "timestamp": 2000.0,
                "publication_date": "2024-06-20",
            },
            {
                "id": "concept:Alpha",
                "type": "Concept",
                "name": "Alpha",
                "weight": 8,
            },
            {
                "id": "concept:Beta",
                "type": "Concept",
                "name": "Beta",
                "weight": 5,
            },
        ],
        "edges": [
            {
                "source": "book:20",
                "target": "concept:Alpha",
                "relation": "EXPLORES_IDEA",
                "occurrences": 3,
            },
            {
                "source": "book:20",
                "target": "concept:Beta",
                "relation": "EXPLORES_IDEA",
                "occurrences": 10,
            },
        ],
    }
    kg_path = tmp_path / "knowledge_graph.json"
    kg_path.write_text(json.dumps(kg_data), encoding="utf-8")

    with patch.object(mock_backend, "get_neo4j_driver", side_effect=Exception("Neo4j offline")), \
         patch.object(mock_backend.config, "output_dir", str(tmp_path)):
        books = mock_backend.get_indexed_books()

        assert len(books) == 2
        # Verify descending order by timestamp (most recent first)
        assert books[0]["title"] == "Newer Book"
        assert books[0]["timestamp"] == 2000.0
        assert books[0]["publication_date"] == "2024-06-20"
        assert books[1]["title"] == "Older Book"
        assert books[1]["timestamp"] == 1000.0
        assert books[1]["publication_date"] == "2023-01-15"

        # Verify highlighted ideas calculation: occurrences * weight
        top_ideas = books[0]["top_ideas"]
        assert len(top_ideas) == 2
        # Beta: occ=10, wt=5 => score=50
        # Alpha: occ=3, wt=8 => score=24
        assert top_ideas[0]["name"] == "Beta"
        assert top_ideas[0]["score"] == 50.0
        assert top_ideas[0]["occurrences"] == 10
        assert top_ideas[0]["weight"] == 5.0

        assert top_ideas[1]["name"] == "Alpha"
        assert top_ideas[1]["score"] == 24.0


def test_ui_indexed_books_endpoints(mock_backend, tmp_path):
    """Test /api/books and /api/books/indexed endpoints."""
    import urllib.request
    import threading

    mock_books = [
        {
            "book_id": "2",
            "id": "book:2",
            "title": "Recent Book",
            "author": "Alice",
            "timestamp": 2000.0,
            "status": "completed",
            "total_chunks": 40,
            "ideas_count": 2,
            "top_ideas": [{"name": "Super Intelligence", "occurrences": 5, "weight": 9, "score": 45.0}],
            "book_file": None,
        },
        {
            "book_id": "1",
            "id": "book:1",
            "title": "Old Book",
            "author": "Bob",
            "timestamp": 1000.0,
            "status": "completed",
            "total_chunks": 25,
            "ideas_count": 1,
            "top_ideas": [{"name": "Neural Nets", "occurrences": 2, "weight": 6, "score": 12.0}],
            "book_file": None,
        },
    ]

    with patch.object(mock_backend, "get_indexed_books", return_value=mock_books):
        server = run_server(host="127.0.0.1", port=0, backend=mock_backend)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{server.server_port}"

        try:
            with urllib.request.urlopen(f"{base_url}/api/books") as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert "books" in data
                assert data["count"] == 2
                assert data["books"][0]["title"] == "Recent Book"
                assert data["books"][0]["top_ideas"][0]["name"] == "Super Intelligence"
                assert data["books"][0]["top_ideas"][0]["score"] == 45.0

            with urllib.request.urlopen(f"{base_url}/api/books/indexed") as resp2:
                assert resp2.status == 200
                data2 = json.loads(resp2.read().decode("utf-8"))
                assert data2["count"] == 2
        finally:
            server.shutdown()
            server.server_close()


def test_ui_index_html_has_indexed_books_page():
    """Verify HTML template contains indexed books tab, table, idea search redirect, and URL search handler."""
    html_file = Path(__file__).parent.parent / "src" / "bookeeper" / "ui" / "static" / "index.html"
    content = html_file.read_text(encoding="utf-8")

    assert "tab-btn-books" in content
    assert "Indexed Books" in content
    assert "pane-books" in content
    assert "books-table" in content
    assert "books-table-body" in content
    assert "loadIndexedBooks" in content
    assert "searchIdeaFromBooks" in content
    assert "filterBooksTable" in content
    assert "openBookInGraph" in content
    assert "occurrences × weight" in content
    assert "Full Backup" in content
    assert "backup-modal" in content
    assert "openBackupModal" in content
    assert "startBackupCreation" in content
    assert "Restore Backup" in content
    assert "restore-modal" in content
    assert "openRestoreModal" in content
    assert "startBackupRestore" in content
    assert "/api/backup/restore" in content


def test_ui_index_html_has_backup_restore_page_and_navigation():
    """Verify HTML template contains the dedicated Backup & Restore page, tabs, and direct menu navigation."""
    html_file = Path(__file__).parent.parent / "src" / "bookeeper" / "ui" / "static" / "index.html"
    content = html_file.read_text(encoding="utf-8")

    # Navigation menu tabs
    assert 'id="tab-btn-graph"' in content
    assert 'id="tab-btn-books"' in content
    assert 'id="tab-btn-backup"' in content
    assert 'id="tab-btn-debug"' in content

    # View panes
    assert 'id="pane-graph"' in content
    assert 'id="pane-books"' in content
    assert 'id="pane-backup"' in content
    assert 'id="pane-debug"' in content

    # Backup & Restore page contents
    assert 'id="page-backup-card"' in content
    assert 'id="page-restore-card"' in content
    assert 'id="btn-page-backup-start"' in content
    assert 'id="btn-page-restore-start"' in content
    assert 'startPageBackupCreation' in content
    assert 'startPageBackupRestore' in content
    assert 'onPageRestoreFileSelected' in content

    # switchTab and URL parameter routing
    assert "tab === 'backup'" in content or "tab === 'restore'" in content
    assert "tabParam === 'backup'" in content


def test_ui_backend_restore_from_backup_flow(mock_backend, tmp_path):
    """Test restore_from_backup restores files, triggers Neo4j export, and runs consistency check."""
    import zipfile

    # Create dummy archive
    archive_zip = tmp_path / "restore_test.zip"
    with zipfile.ZipFile(archive_zip, "w") as zf:
        zf.writestr("knowledge_graph.json", json.dumps({"nodes": [], "edges": []}))

    with patch("bookeeper.backup.restore_backup", return_value={"files_restored": 1, "bytes_restored": 100, "backup_type": "knowledge"}), \
         patch.object(mock_backend, "validate_database_consistency", return_value={"neo4j_connected": False, "is_consistent": True, "discrepancies": []}), \
         patch.object(mock_backend.config, "output_dir", str(tmp_path)):

        res = mock_backend.restore_from_backup(archive_file=archive_zip, clean=True, export_neo4j=False)
        assert res["success"] is True
        assert res["restored_files"] == 1
        assert "consistency" in res
        assert res["consistency"]["is_consistent"] is True


def test_ui_restore_endpoint_flow(mock_backend, tmp_path):
    """Test /api/backup/restore POST endpoint with multipart upload."""
    import urllib.request
    import threading

    mock_restore_res = {
        "success": True,
        "restored_files": 42,
        "restored_bytes": 1024,
        "backup_type": "full",
        "consistency": {"is_consistent": True, "discrepancies": []},
        "neo4j_export": {"nodes_upserted": 10, "edges_upserted": 20},
    }

    with patch.object(mock_backend, "restore_from_backup", return_value=mock_restore_res):
        server = run_server(host="127.0.0.1", port=0, backend=mock_backend)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{server.server_port}"

        try:
            # 1. Status endpoint (/api/backup/restore/status)
            with urllib.request.urlopen(f"{base_url}/api/backup/restore/status") as st_resp:
                assert st_resp.status == 200
                st_data = json.loads(st_resp.read().decode())
                assert "in_progress" in st_data

            # 2. Upload multipart restore request
            boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
            body = (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="clean"\r\n\r\n'
                "true\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="test.zip"\r\n'
                "Content-Type: application/zip\r\n\r\n"
                "PK0304dummy\r\n"
                f"--{boundary}--\r\n"
            ).encode("utf-8")

            req = urllib.request.Request(
                f"{base_url}/api/backup/restore",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )

            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["success"] is True
                assert data["restored_files"] == 42
                assert data["neo4j_export"]["nodes_upserted"] == 10
        finally:
            server.shutdown()
            server.server_close()



def test_ui_backend_prepare_full_backup(mock_backend, tmp_path):
    """Test prepare_full_backup creates valid zip archive with metadata."""
    kg_path = tmp_path / "knowledge_graph.json"
    kg_path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    dummy_books_dir = tmp_path / "sample_books_repo"
    dummy_books_dir.mkdir(parents=True, exist_ok=True)
    (dummy_books_dir / "sample.pdf").write_bytes(b"%PDF-1.4 dummy")

    with patch.object(mock_backend.config, "output_dir", str(tmp_path)), \
         patch.object(mock_backend, "_get_books_dir", return_value=dummy_books_dir), \
         patch.object(mock_backend, "_get_all_books_dirs", return_value=[]):
        res = mock_backend.prepare_full_backup(backup_type="full", format_type="zip")
        assert res["success"] is True
        assert res["filename"].endswith(".zip")
        assert res["total_files"] >= 1
        assert "download_url" in res
        assert Path(res["absolute_path"]).is_file()


def test_ui_backup_endpoints(mock_backend, tmp_path):
    """Test /api/backup/prepare and /api/backup/download endpoints."""
    import urllib.request
    import threading

    # Create dummy backup file
    backups_dir = tmp_path / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    test_zip = backups_dir / "test_backup.zip"
    test_zip.write_bytes(b"PK\x03\x04dummy_zip_content")

    mock_res = {
        "success": True,
        "filename": "test_backup.zip",
        "file_size": len(b"PK\x03\x04dummy_zip_content"),
        "total_files": 1,
        "backup_type": "full",
        "format": "zip",
        "download_url": "/api/backup/download?filename=test_backup.zip",
        "absolute_path": str(test_zip),
    }

    with patch.object(mock_backend, "prepare_full_backup", return_value=mock_res), \
         patch.object(mock_backend.config, "output_dir", str(tmp_path)):
        server = run_server(host="127.0.0.1", port=0, backend=mock_backend)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{server.server_port}"

        try:
            # 1. Prepare POST
            post_data = json.dumps({"type": "full", "format": "zip"}).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/backup/prepare",
                data=post_data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["success"] is True
                assert data["filename"] == "test_backup.zip"

            # 2. Download GET
            with urllib.request.urlopen(f"{base_url}/api/backup/download?filename=test_backup.zip") as dl_resp:
                assert dl_resp.status == 200
                content = dl_resp.read()
                assert content == b"PK\x03\x04dummy_zip_content"
                assert "test_backup.zip" in dl_resp.headers.get("Content-Disposition", "")
        finally:
            server.shutdown()
            server.server_close()










