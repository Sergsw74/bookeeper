"""
Tests for Neo4j knowledge graph exporter with upsert semantics.
"""

import json
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

    # Verify upsert queries: MERGE with ON CREATE SET and ON MATCH SET (with Strategy B content_hash guard)
    node_upserts = [q for q in executed_queries if "MERGE (n:" in q]
    assert len(node_upserts) > 0
    for q in node_upserts:
        assert "ON CREATE SET n += row.properties" in q
        assert "ON MATCH SET n += CASE WHEN" in q or "ON MATCH SET n += row.properties" in q

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
        if "RETURN count(n) AS cnt" in query or "RETURN count(r) AS cnt" in query:
            return mock_result_cnt
        return MagicMock()

    mock_session.run.side_effect = mock_run

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j", clean_export=True)
        result = exporter.export(store, clean=True)

    assert result["clean_start"] is True
    assert result["cleaned_nodes"] == 15

    executed_queries = [call.args[0] for call in mock_session.run.call_args_list]
    assert any("DELETE" in q for q in executed_queries)


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
        if "RETURN count(n) AS cnt" in query or "RETURN count(r) AS cnt" in query:
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


def test_strategy_a_per_book_cluster_sync():
    """
    Verify Strategy A (Per-Book Cluster Synchronization):
    - Book 1 has identical hash in Neo4j -> its sections and chunks are skipped.
    - Book 2 is new/modified -> its sections and chunks are upserted.
    """
    store = ConceptGraphStore()
    store.add_book(book_id=1, title="Book One", author="Author 1")
    store.add_section(book_id=1, chapter_idx=1, title="Chapter 1.1")
    c1 = HierarchicalChunk(
        chunk_id="b1_c1",
        book_id=1,
        book_title="Book One",
        chapter_idx=1,
        chunk_idx=1,
        section_title="Chapter 1.1",
        text="Book 1 chunk text",
    )
    store.add_chunk(c1)

    store.add_book(book_id=2, title="Book Two", author="Author 2")
    store.add_section(book_id=2, chapter_idx=1, title="Chapter 2.1")
    c2 = HierarchicalChunk(
        chunk_id="b2_c1",
        book_id=2,
        book_title="Book Two",
        chapter_idx=1,
        chunk_idx=1,
        section_title="Chapter 2.1",
        text="Book 2 chunk text",
    )
    store.add_chunk(c2)

    # Compute expected hash for Book 1
    b1_hash = Neo4jExporter.compute_book_content_hash(1, store=store)

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    executed_queries: list[str] = []
    executed_params: list[dict] = []

    def _mock_run(query: str, **params):
        executed_queries.append(query)
        executed_params.append(params)
        # 1. Book hashes query: Neo4j already has Book 1 with identical hash, but not Book 2
        if "MATCH (b:Book) RETURN b.id AS id, b.content_hash AS content_hash" in query:
            return [{"id": "book:1", "content_hash": b1_hash}]
        # 2. Sync metadata query: return empty or outdated
        if "MATCH (s:BookeeperSync {id: 'sync_metadata'})" in query:
            mock_res = MagicMock()
            mock_res.single.return_value = {"graph_hash": "outdated_hash"}
            return mock_res
        return MagicMock()

    mock_session.run.side_effect = _mock_run

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j")
        result = exporter.export(store, incremental_sync=True)

    assert result["status"] == "success"
    assert result["books_evaluated"] == 2
    assert result["books_unchanged"] == 1
    assert result["books_modified"] == 1
    assert result["chunks_skipped"] >= 1

    # Verify that Book 1's chunk 'b1_c1' was NEVER included in the upsert batch!
    upserted_ids = set()
    for p in executed_params:
        if "batch" in p:
            for item in p["batch"]:
                if "id" in item:
                    upserted_ids.add(item["id"])
                elif "source_id" in item:
                    upserted_ids.add(item["source_id"])
                    upserted_ids.add(item["target_id"])

    # Book 2's chunk MUST be upserted
    assert "chunk:b2_c1" in upserted_ids
    # Book 1's chunk MUST NOT be upserted (skipped!)
    assert "chunk:b1_c1" not in upserted_ids
    assert "section:1:1" not in upserted_ids


def test_strategy_b_node_content_hash_guard():
    """
    Verify Strategy B (Node-Level content_hash Guard):
    - Tests determinism of compute_node_content_hash.
    - Tests that Cypher UNWIND query contains the CASE WHEN content_hash check.
    """
    # 1. Determinism
    props_a = {"name": "Attention", "category": "Deep Learning", "weight": 8}
    props_b = {"name": "Attention", "category": "Deep Learning", "weight": 8}
    props_modified = {"name": "Attention", "category": "Deep Learning", "weight": 9}

    hash_a = Neo4jExporter.compute_node_content_hash("concept:Attention", "Concept", props_a)
    hash_b = Neo4jExporter.compute_node_content_hash("concept:Attention", "Concept", props_b)
    hash_mod = Neo4jExporter.compute_node_content_hash("concept:Attention", "Concept", props_modified)

    assert hash_a == hash_b
    assert hash_a != hash_mod

    # 2. Query structure check
    store = ConceptGraphStore()
    store.add_concept(Concept(name="Attention", category="Deep Learning", summary="Self attention mechanism"))

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    executed_queries: list[str] = []

    def _mock_run(query: str, **params):
        executed_queries.append(query)
        return MagicMock()

    mock_session.run.side_effect = _mock_run

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j")
        exporter.export(store)

    concept_upserts = [q for q in executed_queries if "MERGE (n:`Concept`" in q]
    assert len(concept_upserts) > 0
    query = concept_upserts[0]
    assert "n.content_hash = row.content_hash" in query
    assert "CASE WHEN n.content_hash IS NULL OR n.content_hash <> row.content_hash THEN row.properties ELSE {} END" in query


def test_strategy_c_preserve_vector_indexes_during_incremental_sync():
    """
    Verify Strategy C: When unchanged books are present (incremental update),
    vector indexes are preserved online (search_indexes_suspended = False).
    """
    store = ConceptGraphStore()
    store.add_book(book_id=1, title="Book One")
    store.add_book(book_id=2, title="Book Two")
    b1_hash = Neo4jExporter.compute_book_content_hash(1, store=store)

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    def _mock_run(query: str, **params):
        if "MATCH (b:Book) RETURN b.id AS id, b.content_hash AS content_hash" in query:
            return [{"id": "book:1", "content_hash": b1_hash}]
        return MagicMock()

    mock_session.run.side_effect = _mock_run

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j")
        result = exporter.export(store, incremental_sync=True)

    # Strategy C: Vector indexes were kept online!
    assert result["search_indexes_suspended"] is False


def test_sync_metadata_skip_when_already_synced():
    """
    Verify Technical Sync Metadata Node:
    When Neo4j has matching graph_hash, export returns immediately with 0ms overhead.
    """
    store = ConceptGraphStore()
    store.add_book(book_id=42, title="Hitchhiker's Guide", author="Douglas Adams")
    store.add_concept(Concept(name="42", category="Philosophy", summary="Answer to life"))

    b_hash = Neo4jExporter.compute_book_content_hash(42, store=store)
    expected_graph_hash = Neo4jExporter.compute_graph_hash(store, book_hashes={"book:42": b_hash})

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    executed_queries: list[str] = []

    def _mock_run(query: str, **params):
        executed_queries.append(query)
        if "MATCH (s:BookeeperSync {id: 'sync_metadata'})" in query:
            mock_res = MagicMock()
            mock_res.single.return_value = {
                "graph_hash": expected_graph_hash,
                "synced_at_iso": "2026-11-03T20:00:00Z",
                "total_books": 1,
                "total_nodes": 2,
                "total_edges": 0,
                "book_hashes": '{"book:42": "' + b_hash + '"}',
            }
            return mock_res
        return MagicMock()

    mock_session.run.side_effect = _mock_run

    with patch("bookeeper.graph.neo4j_exporter.GraphDatabase.driver", return_value=mock_driver):
        exporter = Neo4jExporter(uri="bolt://localhost:7687", database="neo4j")
        result = exporter.export(store, incremental_sync=True)

    assert result["status"] == "already_synchronized"
    assert result["nodes_upserted"] == 0
    assert result["edges_upserted"] == 0
    assert result["graph_hash"] == expected_graph_hash

    # Verify no node MERGE statements were executed!
    assert not any("MERGE (n:" in q for q in executed_queries)


def test_compute_book_content_hash_with_files(tmp_path):
    """Verify compute_book_content_hash hashes chunks.json and state.json files from disk."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir(parents=True)
    chunks_file = chunks_dir / "book_77_chunks.json"
    chunks_file.write_text(json.dumps([{"chunk_id": "c77_1", "text": "Sample text"}]))

    states_dir = tmp_path / ".book_states"
    states_dir.mkdir(parents=True)
    state_file = states_dir / "book_77_state.json"
    state_file.write_text(json.dumps({"book_id": 77, "status": "indexed"}))

    hash1 = Neo4jExporter.compute_book_content_hash(77, output_dir=tmp_path)
    assert len(hash1) == 64  # SHA-256 hex string

    # Modifying state changes the hash
    state_file.write_text(json.dumps({"book_id": 77, "status": "reindexed", "extra": 123}))
    hash2 = Neo4jExporter.compute_book_content_hash(77, output_dir=tmp_path)
    assert hash1 != hash2


def test_neo4j_and_ollama_environment_variable_overrides(monkeypatch):
    """Verify that BOOKEEPER_NEO4J_* and BOOKEEPER_OLLAMA_BASE_URL env vars override config values."""
    from bookeeper.config import Settings
    from bookeeper.ui.backend import SearchEngineBackend

    base_yaml = {
        "ollama_servers": [
            {"url": "http://192.168.1.100:11434", "priority": 1, "name": "yaml_server"}
        ],
        "neo4j": {
            "enabled": False,
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "old_password",
        }
    }

    # 1. Without env vars: disabled and points to localhost
    cfg = Settings.from_dict(base_yaml)
    assert cfg.neo4j.enabled is False
    assert cfg.neo4j.uri == "bolt://localhost:7687"
    backend = SearchEngineBackend(config=cfg)
    assert backend.is_neo4j_enabled is False

    # 2. With BOOKEEPER_NEO4J_URI: auto-enabled and uses new URI + credentials
    monkeypatch.setenv("BOOKEEPER_NEO4J_URI", "bolt://192.168.50.20:7687")
    monkeypatch.setenv("BOOKEEPER_NEO4J_PASSWORD", "cThuTQ83")
    cfg2 = Settings.from_dict(base_yaml)
    assert cfg2.neo4j.enabled is True
    assert cfg2.neo4j.uri == "bolt://192.168.50.20:7687"
    assert cfg2.neo4j.password == "cThuTQ83"
    backend2 = SearchEngineBackend(config=cfg2)
    assert backend2.is_neo4j_enabled is True

    # 3. With explicit BOOKEEPER_NEO4J_ENABLED=false: respects disabled setting
    monkeypatch.setenv("BOOKEEPER_NEO4J_ENABLED", "false")
    cfg3 = Settings.from_dict(base_yaml)
    assert cfg3.neo4j.enabled is False
    backend3 = SearchEngineBackend(config=cfg3)
    assert backend3.is_neo4j_enabled is False

    # 4. Ollama base URL override
    monkeypatch.delenv("BOOKEEPER_NEO4J_ENABLED", raising=False)
    monkeypatch.setenv("BOOKEEPER_OLLAMA_BASE_URL", "http://192.168.50.20:11434")
    cfg4 = Settings.from_dict(base_yaml)
    assert cfg4.ollama_base_url == "http://192.168.50.20:11434"
    assert len(cfg4.resolved_ollama_servers) == 1
    assert cfg4.resolved_ollama_servers[0].url == "http://192.168.50.20:11434"
    assert cfg4.resolved_llm_servers[0].url == "http://192.168.50.20:11434"
    assert cfg4.resolved_embedding_servers[0].url == "http://192.168.50.20:11434"





