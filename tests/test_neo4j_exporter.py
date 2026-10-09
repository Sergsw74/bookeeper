"""
Tests for Neo4j knowledge graph exporter with upsert semantics.
"""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from bookeeper.cli import app
from bookeeper.config import Neo4jConfig
from bookeeper.graph.neo4j_exporter import (
    Neo4jExporter,
    clean_neo4j_properties,
    sanitize_label,
    sanitize_rel_type,
)
from bookeeper.graph.store import ConceptGraphStore
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.extractor import Concept


def test_sanitize_helpers():
    assert sanitize_label("Book") == "Book"
    assert sanitize_label("Concept Name!") == "Concept_Name"
    assert sanitize_label("123_invalid") == "123_invalid"
    assert sanitize_label("   ") == "Entity"

    assert sanitize_rel_type("HAS_SECTION") == "HAS_SECTION"
    assert sanitize_rel_type("relates-to") == "RELATES_TO"
    assert sanitize_rel_type("explores idea!") == "EXPLORES_IDEA"
    assert sanitize_rel_type("   ") == "RELATES_TO"


def test_clean_neo4j_properties():
    raw = {
        "title": "Clean Code",
        "pages": 464,
        "rating": 4.8,
        "is_active": True,
        "tags": ["programming", "software-engineering"],
        "metadata": {"key": "val", "nested": 123},
        "none_val": None,
    }
    cleaned = clean_neo4j_properties(raw, exclude_keys={"pages"})
    assert cleaned["title"] == "Clean Code"
    assert "pages" not in cleaned
    assert "none_val" not in cleaned
    assert cleaned["rating"] == 4.8
    assert cleaned["is_active"] is True
    assert cleaned["tags"] == ["programming", "software-engineering"]
    assert '"key": "val"' in cleaned["metadata"]


def test_neo4j_config_and_exporter_init():
    cfg = Neo4jConfig(
        uri="bolt://192.168.1.100:7687",
        user="custom_user",
        password="secret_password",
        database="library",
        batch_size=250,
    )
    exporter = Neo4jExporter.from_config(cfg)
    assert exporter.uri == "bolt://192.168.1.100:7687"
    assert exporter.user == "custom_user"
    assert exporter.password == "secret_password"
    assert exporter.database == "library"
    assert exporter.batch_size == 250


def test_neo4j_export_upsert_flow():
    # Setup test graph store
    store = ConceptGraphStore()
    store.add_book(book_id=1, title="Test Book", author="Test Author", summary="A test book")
    store.add_section(book_id=1, chapter_idx=0, title="Chapter 1", text="Intro text")

    chunk = HierarchicalChunk(
        chunk_id="chk-101",
        chunk_idx=0,
        book_id=1,
        book_title="Test Book",
        chapter_idx=0,
        section_title="Chapter 1",
        breadcrumb="Test Book > Chapter 1",
        text="Sample paragraph about architecture.",
        char_count=35,
    )
    store.add_chunk(chunk)

    c1 = Concept(name="Software Architecture", category="Engineering", summary="System design patterns")
    c2 = Concept(name="Microservices", category="Architecture", summary="Decoupled services")
    store.add_concept(c1)
    store.add_concept(c2)

    store.add_book_idea_link(book_id=1, concept_name="Software Architecture")
    store.add_idea_support_link(concept_name="Software Architecture", chunk=chunk, quote="design patterns")
    store.add_concept_relation("Software Architecture", "Microservices", "ENABLES")

    # Mock Neo4j driver and session
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j", batch_size=100)
        result = exporter.export(store)

    assert result["status"] == "success"
    assert result["nodes_upserted"] >= 4
    assert result["edges_upserted"] >= 4
    assert "Book" in result["node_breakdown"]
    assert "Concept" in result["node_breakdown"]

    # Verify Cypher queries executed in session
    executed_queries = [call.args[0] for call in mock_session.run.call_args_list]

    # Verify constraints were checked/created
    assert any("CREATE CONSTRAINT" in q or "CREATE INDEX" in q for q in executed_queries)

    # Verify upsert queries: MERGE with ON CREATE SET and ON MATCH SET
    node_upserts = [q for q in executed_queries if "MERGE (n:" in q]
    assert len(node_upserts) > 0
    for q in node_upserts:
        assert "ON CREATE SET n += row.properties" in q
        assert "ON MATCH SET n += row.properties" in q

    edge_upserts = [q for q in executed_queries if "MERGE (source)-[r:" in q]
    assert len(edge_upserts) > 0
    for q in edge_upserts:
        assert "ON CREATE SET r += row.properties" in q
        assert "ON MATCH SET r += row.properties" in q


def test_cli_export_neo4j_command(tmp_path):
    runner = CliRunner()

    # Create dummy graph file
    graph_file = tmp_path / "knowledge_graph.json"
    store = ConceptGraphStore()
    store.add_book(book_id=42, title="Guide to the Galaxy", author="Douglas Adams")
    store.save(graph_file)

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        result = runner.invoke(
            app,
            [
                "export-neo4j",
                "--graph-file",
                str(graph_file),
                "--uri",
                "bolt://localhost:7687",
                "--user",
                "neo4j",
                "--password",
                "testpass",
            ],
        )

    assert result.exit_code == 0
    assert "Neo4j Export Summary" in result.output
    assert "Total Nodes" in result.output
    assert "Successfully exported Knowledge Graph to Neo4j" in result.output


def test_cli_export_neo4j_missing_file(tmp_path):
    runner = CliRunner()
    missing_file = tmp_path / "does_not_exist.json"

    result = runner.invoke(app, ["export-neo4j", "--graph-file", str(missing_file)])
    assert result.exit_code == 1
    assert "Knowledge graph file not found" in result.output


def test_neo4j_export_clean_start():
    """Verify clean export executes DETACH DELETE before upsert."""
    store = ConceptGraphStore()
    store.add_book(book_id=1, title="Test Clean Export Book")

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Mock count query returning 15 existing nodes
    mock_result_cnt = MagicMock()
    mock_result_cnt.single.return_value = {"cnt": 15}

    def mock_run(query, *args, **kwargs):
        if "RETURN count(n) AS cnt" in query:
            return mock_result_cnt
        return MagicMock()

    mock_session.run.side_effect = mock_run

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j", clean_export=True)
        result = exporter.export(store, clean=True)

    assert result["clean_start"] is True
    assert result["cleaned_nodes"] == 15

    executed_queries = [call.args[0] for call in mock_session.run.call_args_list]
    assert any("DETACH DELETE" in q for q in executed_queries)


def test_cli_export_neo4j_clean_option(tmp_path):
    runner = CliRunner()

    graph_file = tmp_path / "knowledge_graph.json"
    store = ConceptGraphStore()
    store.add_book(book_id=10, title="Clean Start Book")
    store.save(graph_file)

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_result_cnt = MagicMock()
    mock_result_cnt.single.return_value = {"cnt": 5}

    def mock_run(query, *args, **kwargs):
        if "RETURN count(n) AS cnt" in query:
            return mock_result_cnt
        return MagicMock()

    mock_session.run.side_effect = mock_run

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        result = runner.invoke(
            app,
            [
                "export-neo4j",
                "--graph-file",
                str(graph_file),
                "--clean",
            ],
        )

    assert result.exit_code == 0
    assert "Clean Start" in result.output
    assert "clean start + upsert" in result.output


def test_neo4j_export_progress_callback():
    """Verify that progress_callback receives updates for nodes and relationships."""
    store = ConceptGraphStore()
    store.add_book(book_id=1, title="Test Book", author="Test Author")
    store.add_section(book_id=1, chapter_idx=0, title="Chapter 1", text="Intro text")

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    events = []

    def callback(stage, completed, total, description):
        events.append((stage, completed, total, description))

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j", batch_size=1)
        result = exporter.export(store, show_progress=False, progress_callback=callback)

    assert result["status"] == "success"
    stages = {e[0] for e in events}
    assert "nodes" in stages
    assert "relationships" in stages
    node_events = [e for e in events if e[0] == "nodes"]
    assert len(node_events) > 0
    assert node_events[-1][1] == node_events[-1][2]

