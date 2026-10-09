"""
Neo4j Graph Database Exporter for Concept Knowledge Graph.
Performs high-throughput upsert (insert new, update existing) for nodes and relationships via Cypher.
"""

import json
import logging
import re
from contextlib import nullcontext
from typing import Any, Callable, Optional

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:
    Progress = None
    Console = None

try:
    import neo4j
    from neo4j import GraphDatabase
except ImportError:
    neo4j = None
    GraphDatabase = None

from bookeeper.calibre.parser import BookParser
from bookeeper.config import Neo4jConfig
from bookeeper.graph.store import ConceptGraphStore

logger = logging.getLogger(__name__)


def sanitize_label(label: str) -> str:
    """Sanitize string to valid Neo4j Node label."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(label).strip())
    clean = clean.strip("_")
    return clean or "Entity"


def sanitize_rel_type(rel: str) -> str:
    """Sanitize string to valid Neo4j Relationship type."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(rel).strip().upper())
    clean = clean.strip("_")
    return clean or "RELATES_TO"


def clean_neo4j_properties(attrs: dict[str, Any], exclude_keys: set[str] | None = None) -> dict[str, Any]:
    """
    Ensure property values conform to Neo4j supported primitive types:
    int, float, bool, str, and uniform lists thereof.
    Dicts or non-primitive objects are serialized to JSON strings.
    """
    exclude = exclude_keys or set()
    cleaned = {}
    for k, v in attrs.items():
        if k in exclude or v is None:
            continue

        if isinstance(v, str):
            cleaned[k] = BookParser.repair_mojibake(v)
        elif isinstance(v, (int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, list):
            # Neo4j supports homogeneous lists of primitives
            if not v:
                cleaned[k] = []
            elif all(isinstance(x, str) for x in v):
                cleaned[k] = [BookParser.repair_mojibake(str(x)) for x in v]
            elif all(isinstance(x, bool) for x in v):
                cleaned[k] = [bool(x) for x in v]
            elif all(isinstance(x, int) and not isinstance(x, bool) for x in v):
                cleaned[k] = [int(x) for x in v]
            elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                cleaned[k] = [float(x) for x in v]
            else:
                cleaned[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            cleaned[k] = json.dumps(v, ensure_ascii=False)
        else:
            cleaned[k] = BookParser.repair_mojibake(str(v))
    return cleaned


class Neo4jExporter:
    """
    Exports ConceptGraphStore into Neo4j using upsert semantics (MERGE).
    Inserts new nodes and edges, updates existing nodes and edges with latest attributes.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        batch_size: int = 500,
        connection_timeout: float = 10.0,
        clean_export: bool = False,
    ):
        if GraphDatabase is None:
            raise ImportError(
                "The 'neo4j' driver package is required for Neo4j export. "
                "Install it with 'pip install neo4j' or 'uv add neo4j'."
            )
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database or "neo4j"
        self.batch_size = max(1, batch_size)
        self.connection_timeout = connection_timeout
        self.clean_export = clean_export
        self._driver = None

    @classmethod
    def from_config(cls, config: Neo4jConfig) -> "Neo4jExporter":
        """Instantiate Neo4jExporter from Neo4jConfig."""
        return cls(
            uri=config.uri,
            user=config.user,
            password=config.password,
            database=config.database,
            batch_size=config.batch_size,
            connection_timeout=config.connection_timeout,
            clean_export=config.clean_export,
        )

    def get_driver(self):
        """Get or create Neo4j driver instance."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=self.connection_timeout,
            )
        return self._driver

    def close(self):
        """Close Neo4j driver connection."""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def verify_connectivity(self) -> bool:
        """Verify driver connectivity to Neo4j database."""
        driver = self.get_driver()
        driver.verify_connectivity()
        return True

    def ensure_constraints(self, session, labels: list[str]):
        """
        Create uniqueness constraints on node 'id' for each label to guarantee
        fast O(1) index lookups during MERGE statements.
        """
        for label in labels:
            clean_lbl = sanitize_label(label)
            # Try Neo4j 4.4+ constraint syntax
            constraint_name = f"constraint_{clean_lbl.lower()}_id"
            try:
                query = f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:`{clean_lbl}`) REQUIRE n.id IS UNIQUE"
                session.run(query)
            except Exception as e:
                logger.debug(f"Could not create unique constraint for :{clean_lbl} ({e}), trying index fallback.")
                try:
                    index_name = f"index_{clean_lbl.lower()}_id"
                    index_query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:`{clean_lbl}`) ON (n.id)"
                    session.run(index_query)
                except Exception as ex:
                    logger.debug(f"Could not create index for :{clean_lbl}: {ex}")

    def clean_database(self) -> int:
        """
        Delete all nodes and relationships in the target Neo4j database (clean start).
        Uses iterative or batched detach delete to handle graphs cleanly.
        """
        driver = self.get_driver()
        self.verify_connectivity()
        deleted_count = 0
        with driver.session(database=self.database) as session:
            result = session.run("MATCH (n) RETURN count(n) AS cnt")
            record = result.single()
            total_nodes = record["cnt"] if record else 0
            if total_nodes > 0:
                logger.info(
                    f"Performing clean start: deleting {total_nodes} existing nodes and all relationships in Neo4j (db: {self.database})..."
                )
                try:
                    session.run("MATCH (n) DETACH DELETE n")
                except Exception as e:
                    logger.debug(f"Direct DETACH DELETE failed ({e}), retrying with transaction batch...")
                    while True:
                        res = session.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS c")
                        rec = res.single()
                        c = rec["c"] if rec else 0
                        if c == 0:
                            break
                deleted_count = total_nodes
        return deleted_count

    def export(
        self,
        store: ConceptGraphStore,
        clean: Optional[bool] = None,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        console: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Export full ConceptGraphStore into Neo4j via batched upsert with live progress display.

        Semantics:
        - clean=True: Performs a clean start (deletes all content in database before export)
        - Nodes: MERGE on (id), ON CREATE SET props, ON MATCH SET props
        - Edges: MERGE (source)-[r:REL]->(target), ON CREATE SET props, ON MATCH SET props
        - show_progress=True: Displays rich real-time progress bars for nodes and relationships
        """
        driver = self.get_driver()
        self.verify_connectivity()

        use_progress = show_progress and Progress is not None
        target_console = console or (Console() if Console is not None else None)

        progress_columns = [
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=28),
            TaskProgressColumn(),
            TextColumn("•"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

        progress_cm = (
            Progress(*progress_columns, console=target_console, transient=False)
            if use_progress
            else nullcontext()
        )

        with progress_cm as progress:
            effective_clean = self.clean_export if clean is None else clean
            deleted_before_export = 0
            if effective_clean:
                clean_task = (
                    progress.add_task("[bold yellow]Clean Start: Purging previous Neo4j database content...[/bold yellow]", total=None)
                    if progress
                    else None
                )
                deleted_before_export = self.clean_database()
                if progress and clean_task is not None:
                    progress.update(
                        clean_task,
                        completed=1,
                        total=1,
                        description=f"[bold green]✓ Clean Start Complete[/bold green] (purged {deleted_before_export:,} nodes)",
                    )
                if progress_callback:
                    progress_callback("clean", deleted_before_export, deleted_before_export, "Clean start complete")

            # Ensure sequential NEXT and PREV relationships between chunks exist in graph
            store.link_sequential_chunks()

            g = store.graph
            total_nodes = g.number_of_nodes()
            total_edges = g.number_of_edges()

            node_counts: dict[str, int] = {}
            edge_counts: dict[str, int] = {}

            # 1. Prepare Nodes Grouped by (PrimaryLabel, is_idea)
            # Grouping allows using specific label indices for maximum Cypher performance
            nodes_by_group: dict[tuple[str, bool], list[dict[str, Any]]] = {}
            distinct_labels: set[str] = set()

            for node_id, attrs in g.nodes(data=True):
                primary_label = sanitize_label(attrs.get("type", "Entity"))
                is_idea = bool(attrs.get("is_idea", False) or attrs.get("tag") == "idea")
                distinct_labels.add(primary_label)
                if is_idea:
                    distinct_labels.add("Idea")

                props = clean_neo4j_properties(attrs)
                props["id"] = node_id  # Guarantee primary key property

                group_key = (primary_label, is_idea)
                if group_key not in nodes_by_group:
                    nodes_by_group[group_key] = []
                nodes_by_group[group_key].append({"id": node_id, "properties": props})

            # 2. Prepare Edges Grouped by (SourceLabel, RelType, TargetLabel)
            edges_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

            for src_id, tgt_id, attrs in g.edges(data=True):
                src_node = g.nodes.get(src_id, {})
                tgt_node = g.nodes.get(tgt_id, {})

                src_label = sanitize_label(src_node.get("type", "Entity"))
                tgt_label = sanitize_label(tgt_node.get("type", "Entity"))
                rel_type = sanitize_rel_type(attrs.get("relation", "RELATES_TO"))

                props = clean_neo4j_properties(attrs, exclude_keys={"relation"})

                edge_key = (src_label, rel_type, tgt_label)
                if edge_key not in edges_by_group:
                    edges_by_group[edge_key] = []
                edges_by_group[edge_key].append({
                    "source_id": src_id,
                    "target_id": tgt_id,
                    "properties": props,
                })

            # Execute Upserts inside Session
            with driver.session(database=self.database) as session:
                # Step A: Ensure Schema Constraints & Indexes
                schema_task = (
                    progress.add_task(f"Verifying Schema Constraints ({len(distinct_labels)} labels)...", total=len(distinct_labels) or 1)
                    if progress
                    else None
                )
                self.ensure_constraints(session, list(distinct_labels))
                if progress and schema_task is not None:
                    progress.update(
                        schema_task,
                        completed=len(distinct_labels) or 1,
                        description=f"[bold green]✓ Schema Constraints Verified[/bold green] ({len(distinct_labels)} labels)",
                    )
                if progress_callback:
                    progress_callback("schema", len(distinct_labels), len(distinct_labels), "Constraints verified")

                # Step B: Upsert Nodes in Batches
                node_task = (
                    progress.add_task("Upserting Nodes...", total=total_nodes)
                    if progress
                    else None
                )
                nodes_done = 0
                for (primary_label, is_idea), node_batch_list in nodes_by_group.items():
                    label_query = (
                        f"UNWIND $batch AS row\n"
                        f"MERGE (n:`{primary_label}` {{id: row.id}})\n"
                        f"ON CREATE SET n += row.properties"
                        + (", n:`Idea`" if is_idea and primary_label != "Idea" else "")
                        + "\n"
                        "ON MATCH SET n += row.properties"
                        + (", n:`Idea`" if is_idea and primary_label != "Idea" else "")
                    )

                    for i in range(0, len(node_batch_list), self.batch_size):
                        batch = node_batch_list[i : i + self.batch_size]
                        session.run(label_query, batch=batch)
                        nodes_done += len(batch)
                        if progress and node_task is not None:
                            progress.update(
                                node_task,
                                completed=nodes_done,
                                description=f"Upserting Nodes (:{primary_label})",
                            )
                        if progress_callback:
                            progress_callback("nodes", nodes_done, total_nodes, f":{primary_label}")

                    node_counts[primary_label] = node_counts.get(primary_label, 0) + len(node_batch_list)

                if progress and node_task is not None:
                    progress.update(
                        node_task,
                        completed=total_nodes,
                        description=f"[bold green]✓ Nodes Complete[/bold green] ({total_nodes:,} nodes)",
                    )

                # Step C: Upsert Relationships in Batches
                edge_task = (
                    progress.add_task("Upserting Relationships...", total=total_edges)
                    if progress
                    else None
                )
                edges_done = 0
                for (src_label, rel_type, tgt_label), edge_batch_list in edges_by_group.items():
                    rel_query = (
                        f"UNWIND $batch AS row\n"
                        f"MATCH (source:`{src_label}` {{id: row.source_id}})\n"
                        f"MATCH (target:`{tgt_label}` {{id: row.target_id}})\n"
                        f"MERGE (source)-[r:`{rel_type}`]->(target)\n"
                        f"ON CREATE SET r += row.properties\n"
                        f"ON MATCH SET r += row.properties"
                    )

                    for i in range(0, len(edge_batch_list), self.batch_size):
                        batch = edge_batch_list[i : i + self.batch_size]
                        session.run(rel_query, batch=batch)
                        edges_done += len(batch)
                        if progress and edge_task is not None:
                            progress.update(
                                edge_task,
                                completed=edges_done,
                                description=f"Upserting Relationships (:{rel_type})",
                            )
                        if progress_callback:
                            progress_callback("relationships", edges_done, total_edges, f":{rel_type}")

                    edge_counts[rel_type] = edge_counts.get(rel_type, 0) + len(edge_batch_list)

                if progress and edge_task is not None:
                    progress.update(
                        edge_task,
                        completed=total_edges,
                        description=f"[bold green]✓ Relationships Complete[/bold green] ({total_edges:,} relationships)",
                    )

        logger.info(
            f"Successfully upserted {total_nodes} nodes and {total_edges} edges into Neo4j ({self.uri}, db={self.database})"
        )

        return {
            "status": "success",
            "uri": self.uri,
            "database": self.database,
            "clean_start": effective_clean,
            "cleaned_nodes": deleted_before_export,
            "nodes_upserted": total_nodes,
            "node_breakdown": node_counts,
            "edges_upserted": total_edges,
            "edge_breakdown": edge_counts,
        }
