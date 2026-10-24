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


def test_chunk_next_prev_relationships_in_store():
    """Verify ConceptGraphStore correctly links consecutive chunks with NEXT and PREV edges."""
    store = ConceptGraphStore()
    store.add_book(book_id=1, title="Chronicles of Narnia")

    # Add three sequential chunks
    c1 = HierarchicalChunk(
        chunk_id="b1_c1_p1_aaa",
        book_id=1,
        book_title="Chronicles of Narnia",
        chapter_idx=1,
        chunk_idx=1,
        section_title="Chapter 1",
        text="Once there were four children...",
        next_chunk_id="b1_c1_p2_bbb",
    )
    c2 = HierarchicalChunk(
        chunk_id="b1_c1_p2_bbb",
        book_id=1,
        book_title="Chronicles of Narnia",
        chapter_idx=1,
        chunk_idx=2,
        section_title="Chapter 1",
        text="They explored the wardrobe...",
        prev_chunk_id="b1_c1_p1_aaa",
        next_chunk_id="b1_c1_p3_ccc",
    )
    c3 = HierarchicalChunk(
        chunk_id="b1_c1_p3_ccc",
        book_id=1,
        book_title="Chronicles of Narnia",
        chapter_idx=1,
        chunk_idx=3,
        section_title="Chapter 1",
        text="They entered Narnia...",
        prev_chunk_id="b1_c1_p2_bbb",
    )

    store.add_chunk(c1)
    store.add_chunk(c2)
    store.add_chunk(c3)

    g = store.graph
    # Verify nodes exist
    assert g.has_node("chunk:b1_c1_p1_aaa")
    assert g.has_node("chunk:b1_c1_p2_bbb")
    assert g.has_node("chunk:b1_c1_p3_ccc")

    # Verify NEXT edges
    assert g.has_edge("chunk:b1_c1_p1_aaa", "chunk:b1_c1_p2_bbb")
    assert g.edges["chunk:b1_c1_p1_aaa", "chunk:b1_c1_p2_bbb"]["relation"] == "NEXT"
    assert g.has_edge("chunk:b1_c1_p2_bbb", "chunk:b1_c1_p3_ccc")
    assert g.edges["chunk:b1_c1_p2_bbb", "chunk:b1_c1_p3_ccc"]["relation"] == "NEXT"

    # Verify PREV edges
    assert g.has_edge("chunk:b1_c1_p2_bbb", "chunk:b1_c1_p1_aaa")
    assert g.edges["chunk:b1_c1_p2_bbb", "chunk:b1_c1_p1_aaa"]["relation"] == "PREV"
    assert g.has_edge("chunk:b1_c1_p3_ccc", "chunk:b1_c1_p2_bbb")
    assert g.edges["chunk:b1_c1_p3_ccc", "chunk:b1_c1_p2_bbb"]["relation"] == "PREV"

    # Verify edge stats
    stats = store.stats()
    assert stats["edge_types"]["NEXT"] == 2
    assert stats["edge_types"]["PREV"] == 2


def test_neo4j_export_chunk_next_prev_relationships():
    """Verify Neo4jExporter exports (:Chunk)-[:NEXT]->(:Chunk) and (:Chunk)-[:PREV]->(:Chunk)."""
    store = ConceptGraphStore()
    store.add_book(book_id=2, title="Lord of the Rings")

    c1 = HierarchicalChunk(
        chunk_id="b2_c1_p1_111",
        book_id=2,
        book_title="Lord of the Rings",
        chapter_idx=1,
        chunk_idx=1,
        section_title="Concerning Hobbits",
        text="Hobbits are an unobtrusive people...",
    )
    c2 = HierarchicalChunk(
        chunk_id="b2_c1_p2_222",
        book_id=2,
        book_title="Lord of the Rings",
        chapter_idx=1,
        chunk_idx=2,
        section_title="Concerning Hobbits",
        text="They love peace and quiet...",
    )

    store.add_chunk(c1)
    store.add_chunk(c2)
    # Ensure link_sequential_chunks creates the links even if chunks lacked explicit prev/next at creation
    store.link_sequential_chunks(book_id=2)

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j")
        result = exporter.export(store)

    assert result["status"] == "success"
    assert "NEXT" in result["edge_breakdown"]
    assert "PREV" in result["edge_breakdown"]
    assert result["edge_breakdown"]["NEXT"] == 1
    assert result["edge_breakdown"]["PREV"] == 1

    executed_queries = [call.args[0] for call in mock_session.run.call_args_list]

    # Verify NEXT Cypher query
    next_queries = [q for q in executed_queries if "MERGE (source)-[r:`NEXT`]->(target)" in q]
    assert len(next_queries) > 0
    assert "MATCH (source:`Chunk`" in next_queries[0]
    assert "MATCH (target:`Chunk`" in next_queries[0]

    # Verify PREV Cypher query
    prev_queries = [q for q in executed_queries if "MERGE (source)-[r:`PREV`]->(target)" in q]
    assert len(prev_queries) > 0
    assert "MATCH (source:`Chunk`" in prev_queries[0]
    assert "MATCH (target:`Chunk`" in prev_queries[0]


def test_neo4j_routing_error_fallback_to_bolt():
    """Verify that routing failure on neo4j:// protocol automatically switches to bolt://."""
    mock_routing_driver = MagicMock()
    mock_routing_driver.verify_connectivity.side_effect = RuntimeError("Unable to retrieve routing information")

    mock_bolt_driver = MagicMock()
    mock_bolt_driver.verify_connectivity.return_value = None

    drivers = [mock_routing_driver, mock_bolt_driver]

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", side_effect=drivers):
        exporter = Neo4jExporter(uri="neo4j://192.168.50.20:7687")
        assert exporter.uri == "neo4j://192.168.50.20:7687"
        success = exporter.verify_connectivity()
        assert success is True
        assert exporter.uri == "bolt://192.168.50.20:7687"
        assert mock_routing_driver.verify_connectivity.called
        assert mock_bolt_driver.verify_connectivity.called


def test_neo4j_search_index_suspension_and_bulk_recreation_during_export():
    """
    Verify that during export:
    1. capture_search_indexes captures only VECTOR and FULLTEXT indexes (excluding LOOKUP, RANGE, POINT, constraints).
    2. drop_search_indexes drops VECTOR and FULLTEXT indexes BEFORE node MERGE upserts.
    3. recreate_search_indexes and wait_for_indexes_online recreate them and poll until ONLINE AFTER upserts.
    """
    import pytest

    store = ConceptGraphStore()
    store.add_book(book_id=1, title="High Throughput Neo4j")
    chunk = HierarchicalChunk(
        chunk_id="chk-bulk-1",
        chunk_idx=1,
        book_id=1,
        book_title="High Throughput Neo4j",
        chapter_idx=1,
        section_title="Vectors",
        breadcrumb="Book > Vectors",
        text="Dense vector embeddings in HNSW.",
        char_count=32,
        embedding=[0.1, 0.2, 0.3],
    )
    store.add_chunk(chunk)
    store.add_concept(
        Concept(name="HNSW Index", category="Database", summary="Hierarchical Navigable Small World"),
        embedding=[0.4, 0.5, 0.6],
    )

    show_indexes_catalog = [
        {
            "name": "constraint_chunk_id",
            "type": "RANGE",
            "entityType": "NODE",
            "labelsOrTypes": ["Chunk"],
            "properties": ["id"],
            "owningConstraint": "constraint_chunk_id",
            "createStatement": "CREATE RANGE INDEX constraint_chunk_id FOR (n:Chunk) ON (n.id)",
        },
        {
            "name": "token_lookup_idx",
            "type": "LOOKUP",
            "entityType": "NODE",
            "labelsOrTypes": None,
            "properties": None,
            "createStatement": "CREATE LOOKUP INDEX token_lookup_idx FOR (n) ON EACH labels(n)",
        },
        {
            "name": "idea_idx",
            "type": "VECTOR",
            "entityType": "NODE",
            "labelsOrTypes": ["Idea"],
            "properties": ["embedding"],
            "createStatement": "CREATE VECTOR INDEX idea_idx FOR (n:Idea) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 3, `vector.similarity_function`: 'cosine'}}",
        },
        {
            "name": "idea_fulltext_idx",
            "type": "FULLTEXT",
            "entityType": "NODE",
            "labelsOrTypes": ["Idea"],
            "properties": ["name", "brief_description"],
            "createStatement": "CREATE FULLTEXT INDEX idea_fulltext_idx FOR (n:Idea) ON EACH [n.name, n.brief_description]",
        },
    ]

    poll_Counter = {"count": 0}
    executed_queries: list[str] = []

    def _mock_run(query: str, **params):
        executed_queries.append(query)
        if "SHOW INDEXES YIELD name, type, entityType" in query:
            return show_indexes_catalog
        if "SHOW INDEXES YIELD name, state, populationPercent" in query:
            poll_Counter["count"] += 1
            if poll_Counter["count"] == 1:
                return [
                    {"name": "idea_idx", "state": "POPULATING", "populationPercent": 45.0},
                    {"name": "idea_fulltext_idx", "state": "ONLINE", "populationPercent": 100.0},
                    {"name": "chunk_idx", "state": "POPULATING", "populationPercent": 60.0},
                ]
            return [
                {"name": "idea_idx", "state": "ONLINE", "populationPercent": 100.0},
                {"name": "idea_fulltext_idx", "state": "ONLINE", "populationPercent": 100.0},
                {"name": "chunk_idx", "state": "ONLINE", "populationPercent": 100.0},
            ]
        return MagicMock()

    mock_session = MagicMock()
    mock_session.run.side_effect = _mock_run
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j")
        exporter.export(store, index_poll_interval=0.01)

    # 1. Verify RANGE and LOOKUP indexes were NEVER dropped
    assert not any("DROP INDEX constraint_chunk_id" in q for q in executed_queries)
    assert not any("DROP INDEX token_lookup_idx" in q for q in executed_queries)

    # 2. Verify VECTOR and FULLTEXT indexes WERE dropped before MERGE node upserts,
    #    and recreated AFTER MERGE node upserts
    drop_idea_idx_pos = next(i for i, q in enumerate(executed_queries) if "DROP INDEX idea_idx IF EXISTS" in q)
    drop_ft_idx_pos = next(i for i, q in enumerate(executed_queries) if "DROP INDEX idea_fulltext_idx IF EXISTS" in q)
    first_merge_pos = next(i for i, q in enumerate(executed_queries) if "MERGE (n:" in q)
    last_merge_pos = max(i for i, q in enumerate(executed_queries) if "MERGE (" in q)
    recreate_idea_pos = next(
        i for i, q in enumerate(executed_queries)
        if "CREATE VECTOR INDEX idea_idx IF NOT EXISTS FOR (n:Idea)" in q
    )
    recreate_ft_pos = next(
        i for i, q in enumerate(executed_queries)
        if "CREATE FULLTEXT INDEX idea_fulltext_idx IF NOT EXISTS FOR (n:Idea)" in q
    )

    assert drop_idea_idx_pos < first_merge_pos
    assert drop_ft_idx_pos < first_merge_pos
    assert recreate_idea_pos > last_merge_pos
    assert recreate_ft_pos > last_merge_pos
    assert poll_Counter["count"] == 2


def test_neo4j_search_index_suspension_recreates_on_exception_and_timeout():
    """
    Verify that:
    1. If an exception occurs during node ingestion inside search_index_suspension,
       captured indexes are still safely recreated in the finally block.
    2. wait_for_indexes_online raises TimeoutError when an index stays POPULATING past timeout_sec.
    """
    import pytest

    mock_session = MagicMock()
    executed_queries: list[str] = []

    def _mock_run(query: str, **params):
        executed_queries.append(query)
        if "SHOW INDEXES YIELD name, type, entityType" in query:
            return [
                {
                    "name": "chunk_idx",
                    "type": "VECTOR",
                    "entityType": "NODE",
                    "labelsOrTypes": ["Chunk"],
                    "properties": ["embedding"],
                    "createStatement": "CREATE VECTOR INDEX chunk_idx FOR (n:Chunk) ON (n.embedding)",
                }
            ]
        if "SHOW INDEXES YIELD name, state, populationPercent" in query:
            return [{"name": "chunk_idx", "state": "POPULATING", "populationPercent": 12.5}]
        return MagicMock()

    mock_session.run.side_effect = _mock_run
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j")

        # 1. Exception during ingestion still triggers index recreation in finally block
        with pytest.raises(ValueError, match="Simulated write failure"):
            with exporter.search_index_suspension(session=mock_session, wait_online=False):
                raise ValueError("Simulated write failure")

        assert any("DROP INDEX chunk_idx IF EXISTS" in q for q in executed_queries)
        assert any("CREATE VECTOR INDEX chunk_idx IF NOT EXISTS FOR (n:Chunk)" in q for q in executed_queries)

        # 2. Timeout when index stays POPULATING past timeout_sec
        with pytest.raises(TimeoutError, match="Timed out"):
            exporter.wait_for_indexes_online(
                session=mock_session,
                index_names=["chunk_idx"],
                timeout_sec=0,
                poll_interval=0.01,
            )




