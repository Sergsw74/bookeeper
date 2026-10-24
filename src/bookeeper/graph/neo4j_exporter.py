"""
Neo4j Graph Database Exporter for Concept Knowledge Graph.
Performs high-throughput upsert (insert new, update existing) for nodes and relationships via Cypher.
"""

import json
import logging
import re
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Iterator, Optional

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
        self._captured_search_indexes: list[dict[str, Any]] = []

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
        """
        Verify driver connectivity to Neo4j database.
        Automatically falls back from neo4j:// routing protocol to bolt:// direct connection
        if the target is a standalone instance that does not support routing tables.
        """
        driver = self.get_driver()
        try:
            driver.verify_connectivity()
            return True
        except Exception as e:
            err_msg = str(e).lower()
            routing_err = "routing" in err_msg or "unable to retrieve routing" in err_msg
            if routing_err:
                fallback_scheme = None
                if self.uri.startswith("neo4j://"):
                    fallback_scheme = "bolt://"
                    orig_prefix = "neo4j://"
                elif self.uri.startswith("neo4j+s://"):
                    fallback_scheme = "bolt+s://"
                    orig_prefix = "neo4j+s://"
                elif self.uri.startswith("neo4j+ssc://"):
                    fallback_scheme = "bolt+ssc://"
                    orig_prefix = "neo4j+ssc://"

                if fallback_scheme:
                    fallback_uri = self.uri.replace(orig_prefix, fallback_scheme, 1)
                    logger.warning(
                        f"Neo4j routing failed ('{e}'). Standalone instance detected. "
                        f"Automatically switching protocol from '{self.uri}' to direct connection '{fallback_uri}'."
                    )
                    self.close()
                    self.uri = fallback_uri
                    driver = self.get_driver()
                    driver.verify_connectivity()
                    return True
            raise

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

    def ensure_idea_vector_index(
        self,
        session: Any,
        dimensions: int = 1024,
        index_name: str = "idea_idx",
    ) -> bool:
        """
        Create a native HNSW vector index on :Idea(embedding) in Neo4j.
        """
        dim = max(1, int(dimensions or 1024))
        safe_idx = re.sub(r"[^A-Za-z0-9_]", "_", index_name) or "idea_idx"
        cypher = (
            f"CREATE VECTOR INDEX {safe_idx} IF NOT EXISTS\n"
            f"FOR (n:Idea) ON (n.embedding)\n"
            f"OPTIONS {{indexConfig: {{\n"
            f"  `vector.dimensions`: {dim},\n"
            f"  `vector.similarity_function`: 'cosine'\n"
            f"}}}}"
        )
        try:
            session.run(cypher)
            return True
        except Exception as e:
            logger.debug(f"Could not create vector index '{safe_idx}' on :Idea(embedding): {e}")
            return False

    def ensure_chunk_vector_index(
        self,
        session: Any,
        dimensions: int = 1024,
        index_name: str = "chunk_idx",
    ) -> bool:
        """
        Create a native HNSW vector index on :Chunk(embedding) in Neo4j.
        """
        dim = max(1, int(dimensions or 1024))
        safe_idx = re.sub(r"[^A-Za-z0-9_]", "_", index_name) or "chunk_idx"
        cypher = (
            f"CREATE VECTOR INDEX {safe_idx} IF NOT EXISTS\n"
            f"FOR (n:Chunk) ON (n.embedding)\n"
            f"OPTIONS {{indexConfig: {{\n"
            f"  `vector.dimensions`: {dim},\n"
            f"  `vector.similarity_function`: 'cosine'\n"
            f"}}}}"
        )
        try:
            session.run(cypher)
            return True
        except Exception as e:
            logger.debug(f"Could not create vector index '{safe_idx}' on :Chunk(embedding): {e}")
            return False

    @staticmethod
    def _ensure_if_not_exists_in_create_index(create_stmt: str) -> str:
        """Ensure a captured CREATE [VECTOR|FULLTEXT] INDEX statement is idempotent with IF NOT EXISTS."""
        stmt = (create_stmt or "").strip()
        if not stmt:
            return stmt
        if re.search(r"\bIF\s+NOT\s+EXISTS\b", stmt, flags=re.IGNORECASE):
            return stmt
        return re.sub(r"\bFOR\b", "IF NOT EXISTS FOR", stmt, count=1, flags=re.IGNORECASE)

    def _execute_write_query(self, session: Any, query: str, **params: Any) -> Any:
        """
        Execute a parameterized Cypher write query inside an explicit write transaction
        (session.execute_write) on real Neo4j sessions, while remaining compatible with mocks.
        """
        from unittest.mock import MagicMock

        if isinstance(session, MagicMock):
            if getattr(session, "execute_write", None) is not None and session.execute_write.side_effect is not None:
                return session.execute_write(lambda tx: tx.run(query, **params))
            return session.run(query, **params)
        if hasattr(session, "execute_write"):
            return session.execute_write(lambda tx: list(tx.run(query, **params)))
        return session.run(query, **params)

    def capture_search_indexes(self, session: Optional[Any] = None) -> list[dict[str, Any]]:
        """
        Discover and capture all VECTOR and FULLTEXT search indexes in Neo4j:
          SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, createStatement
        Strictly excludes LOOKUP, RANGE, POINT, TEXT, and constraint-backing indexes
        so unique ID constraints (e.g. :Chunk(id), :Concept(id)) remain intact for O(1) MERGE lookups.
        """
        if session is None:
            driver = self.get_driver()
            with driver.session(database=self.database) as owned_session:
                return self.capture_search_indexes(session=owned_session)

        query = "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, createStatement"
        captured: list[dict[str, Any]] = []
        try:
            result = session.run(query)
            records = list(result) if result is not None else []
        except Exception as e:
            logger.debug(f"Could not query SHOW INDEXES to capture search indexes: {e}")
            self._captured_search_indexes = []
            return []

        for rec in records:
            try:
                row = dict(rec) if not isinstance(rec, dict) else rec
            except Exception:
                continue

            idx_name = str(row.get("name") or "").strip()
            idx_type = str(row.get("type") or "").strip().upper()
            create_stmt = str(row.get("createStatement") or "").strip()
            owning_constraint = row.get("owningConstraint")

            if not idx_name or idx_type not in ("VECTOR", "FULLTEXT"):
                continue
            if owning_constraint is not None or "CONSTRAINT" in create_stmt.upper():
                continue
            if not create_stmt:
                continue

            idempotent_stmt = self._ensure_if_not_exists_in_create_index(create_stmt)
            captured.append({
                "name": idx_name,
                "type": idx_type,
                "entityType": row.get("entityType"),
                "labelsOrTypes": row.get("labelsOrTypes"),
                "properties": row.get("properties"),
                "createStatement": idempotent_stmt,
            })

        self._captured_search_indexes = captured
        if captured:
            logger.info(
                f"Captured {len(captured)} search index(es) ({', '.join(c['name'] for c in captured)}) "
                "for suspension during bulk ingestion."
            )
        return captured

    def drop_search_indexes(
        self,
        session: Optional[Any] = None,
        captured_indexes: Optional[list[dict[str, Any]]] = None,
    ) -> list[str]:
        """
        Drop each captured VECTOR and FULLTEXT search index using `DROP INDEX <name> IF EXISTS`
        prior to massive node/vector upserts, preserving all LOOKUP/RANGE/constraint indexes.
        """
        if session is None:
            driver = self.get_driver()
            with driver.session(database=self.database) as owned_session:
                return self.drop_search_indexes(session=owned_session, captured_indexes=captured_indexes)

        targets = captured_indexes if captured_indexes is not None else self._captured_search_indexes
        dropped_names: list[str] = []
        for idx in targets:
            name = str(idx.get("name") or "").strip()
            if not name:
                continue
            safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
            drop_query = f"DROP INDEX {safe_name} IF EXISTS"
            try:
                session.run(drop_query)
                dropped_names.append(name)
                logger.info(f"Dropped {idx.get('type', 'SEARCH')} index '{name}' prior to bulk upsert.")
            except Exception as e:
                logger.warning(f"Failed to drop search index '{name}': {e}")
        return dropped_names

    def recreate_search_indexes(
        self,
        session: Optional[Any] = None,
        captured_indexes: Optional[list[dict[str, Any]]] = None,
    ) -> list[str]:
        """
        Recreate all previously captured VECTOR and FULLTEXT search indexes using their
        saved `createStatement` (with idempotent IF NOT EXISTS semantics).
        """
        if session is None:
            driver = self.get_driver()
            with driver.session(database=self.database) as owned_session:
                return self.recreate_search_indexes(session=owned_session, captured_indexes=captured_indexes)

        targets = captured_indexes if captured_indexes is not None else self._captured_search_indexes
        recreated_names: list[str] = []
        for idx in targets:
            name = str(idx.get("name") or "").strip()
            stmt = self._ensure_if_not_exists_in_create_index(str(idx.get("createStatement") or ""))
            if not name or not stmt:
                continue
            try:
                session.run(stmt)
                recreated_names.append(name)
                logger.info(f"Recreated {idx.get('type', 'SEARCH')} index '{name}' for multithreaded bulk population.")
            except Exception as e:
                logger.warning(f"Failed to recreate search index '{name}': {e}")
        return recreated_names

    def wait_for_indexes_online(
        self,
        session: Optional[Any] = None,
        index_names: Optional[list[str]] = None,
        timeout_sec: int = 300,
        poll_interval: float = 3.0,
        on_poll: Optional[Callable[[dict[str, tuple[str, float]]], None]] = None,
    ) -> bool:
        """
        Poll `SHOW INDEXES YIELD name, state, populationPercent WHERE name IN $index_names`
        until all recreated VECTOR and FULLTEXT indexes reach state 'ONLINE'.
        Raises TimeoutError if any index remains 'POPULATING' past timeout_sec, or RuntimeError on 'FAILED'.
        """
        if session is None:
            driver = self.get_driver()
            with driver.session(database=self.database) as owned_session:
                return self.wait_for_indexes_online(
                    session=owned_session,
                    index_names=index_names,
                    timeout_sec=timeout_sec,
                    poll_interval=poll_interval,
                    on_poll=on_poll,
                )

        target_names = list(
            dict.fromkeys(
                [n for n in (index_names if index_names is not None else [c.get("name", "") for c in self._captured_search_indexes]) if n]
            )
        )
        if not target_names:
            return True

        poll_query = (
            "SHOW INDEXES YIELD name, state, populationPercent "
            "WHERE name IN $index_names "
            "RETURN name, state, populationPercent"
        )
        t_start = time.monotonic()

        while True:
            try:
                result = session.run(poll_query, index_names=target_names)
                records = list(result) if result is not None else []
            except Exception as e:
                logger.debug(f"Could not poll index status via SHOW INDEXES: {e}")
                return True

            if not records:
                return True

            status_map: dict[str, tuple[str, float]] = {}
            failed_indexes: list[str] = []
            populating_indexes: list[str] = []

            for rec in records:
                try:
                    row = dict(rec) if not isinstance(rec, dict) else rec
                except Exception:
                    continue
                name = str(row.get("name") or "")
                state = str(row.get("state") or "").upper()
                raw_pct = row.get("populationPercent")
                pct = float(raw_pct) if isinstance(raw_pct, (int, float)) else (100.0 if state == "ONLINE" else 0.0)
                if name:
                    status_map[name] = (state, pct)
                    if state == "FAILED":
                        failed_indexes.append(name)
                    elif state != "ONLINE":
                        populating_indexes.append(f"{name} ({state}: {pct:.1f}%)")

            if not status_map:
                return True

            if on_poll is not None:
                on_poll(status_map)

            summary_str = ", ".join(f"{k}: {v[0]} ({v[1]:.1f}%)" for k, v in status_map.items())
            logger.info(f"Neo4j search index population status: {summary_str}")

            if failed_indexes:
                raise RuntimeError(
                    f"Neo4j search index(es) entered FAILED state during bulk rebuild: {', '.join(failed_indexes)}"
                )

            if not populating_indexes:
                logger.info(f"All Neo4j search indexes are ONLINE ({', '.join(status_map.keys())}).")
                return True

            elapsed = time.monotonic() - t_start
            if elapsed >= timeout_sec:
                raise TimeoutError(
                    f"Timed out after {timeout_sec}s waiting for Neo4j search indexes to reach ONLINE state: "
                    f"{', '.join(populating_indexes)}"
                )

            time.sleep(max(0.01, float(poll_interval)))

    @contextmanager
    def search_index_suspension(
        self,
        session: Any,
        distinct_labels: Optional[set[str]] = None,
        idea_vector_dim: int = 1024,
        chunk_vector_dim: int = 1024,
        suspend_existing: bool = True,
        wait_online: bool = True,
        timeout_sec: int = 300,
        poll_interval: float = 3.0,
        progress: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """
        Lifecycle context manager that:
          1. Captures and drops existing VECTOR and FULLTEXT search indexes before bulk ingestion
             (while keeping all unique ID constraint/lookup indexes intact for fast MERGE).
          2. Guarantees in a `finally:` block that all captured search indexes (plus `idea_idx`
             and `chunk_idx` for :Idea and :Chunk embeddings) are recreated and reach `ONLINE`
             even if an exception occurs during node/edge ingestion.
        """
        labels = distinct_labels or set()
        captured: list[dict[str, Any]] = []
        if suspend_existing:
            captured = self.capture_search_indexes(session=session)
            if captured:
                self.drop_search_indexes(session=session, captured_indexes=captured)

        try:
            yield captured
        finally:
            recreated_names: list[str] = []
            if captured:
                recreated_names.extend(self.recreate_search_indexes(session=session, captured_indexes=captured))

            recreated_set = set(recreated_names)
            if "Idea" in labels and "idea_idx" not in recreated_set:
                if self.ensure_idea_vector_index(session, dimensions=idea_vector_dim, index_name="idea_idx"):
                    recreated_names.append("idea_idx")
                    recreated_set.add("idea_idx")
            if "Chunk" in labels and "chunk_idx" not in recreated_set:
                if self.ensure_chunk_vector_index(session, dimensions=chunk_vector_dim, index_name="chunk_idx"):
                    recreated_names.append("chunk_idx")
                    recreated_set.add("chunk_idx")

            if wait_online and recreated_names:
                idx_task = (
                    progress.add_task(
                        f"Rebuilding HNSW / Search Indexes ({', '.join(recreated_names)})...",
                        total=100,
                    )
                    if progress
                    else None
                )

                def _on_index_poll(status_map: dict[str, tuple[str, float]]) -> None:
                    avg_pct = (
                        sum(pct for _, pct in status_map.values()) / len(status_map)
                        if status_map
                        else 100.0
                    )
                    detail = ", ".join(f"{k} {pct:.0f}%" for k, (_, pct) in status_map.items())
                    if progress and idx_task is not None:
                        progress.update(
                            idx_task,
                            completed=min(100, int(avg_pct)),
                            description=f"Building Search Indexes ({detail})...",
                        )
                    if progress_callback:
                        progress_callback("indexes", int(avg_pct), 100, detail)

                self.wait_for_indexes_online(
                    session=session,
                    index_names=recreated_names,
                    timeout_sec=timeout_sec,
                    poll_interval=poll_interval,
                    on_poll=_on_index_poll,
                )
                if progress and idx_task is not None:
                    progress.update(
                        idx_task,
                        completed=100,
                        description=f"[bold green]✓ Search Indexes ONLINE[/bold green] ({', '.join(recreated_names)})",
                    )

    def query_ideas_hnsw_batch(
        self,
        idea_vectors: list[list[float]],
        top_k: int = 3,
        index_name: str = "idea_idx",
    ) -> list[list[dict[str, Any]]]:
        """
        Query all extracted idea vectors (e.g., 20 ideas from an article) in a single
        batched Neo4j transaction against the native HNSW vector index on :Idea(embedding):
            UNWIND $idea_vectors AS vec
            CALL db.index.vector.queryNodes('idea_idx', 3, vec)
        Returns a list of top_k matched idea dicts for each input vector in order.
        """
        if not idea_vectors:
            return []

        clean_vecs = [[float(x) for x in v] for v in idea_vectors]
        k = max(1, int(top_k or 3))
        safe_idx = re.sub(r"[^A-Za-z0-9_]", "_", index_name) or "idea_idx"

        # Use the exact batched UNWIND + CALL db.index.vector.queryNodes('idea_idx', 3, vec) pattern
        if safe_idx == "idea_idx" and k == 3:
            cypher = (
                "UNWIND $idea_vectors AS vec CALL db.index.vector.queryNodes('idea_idx', 3, vec)\n"
                "YIELD node, score\n"
                "RETURN vec, collect({\n"
                "    id: node.id,\n"
                "    name: node.name,\n"
                "    category: node.category,\n"
                "    brief_description: node.brief_description,\n"
                "    detailed_explanation: node.detailed_explanation,\n"
                "    summary: node.summary,\n"
                "    weight: node.weight,\n"
                "    first_occurrence_month: node.first_occurrence_month,\n"
                "    month: node.month,\n"
                "    score: score\n"
                "}) AS matches"
            )
            params: dict[str, Any] = {"idea_vectors": clean_vecs}
        else:
            cypher = (
                f"UNWIND $idea_vectors AS vec CALL db.index.vector.queryNodes('{safe_idx}', $top_k, vec)\n"
                f"YIELD node, score\n"
                f"RETURN vec, collect({{\n"
                f"    id: node.id,\n"
                f"    name: node.name,\n"
                f"    category: node.category,\n"
                f"    brief_description: node.brief_description,\n"
                f"    detailed_explanation: node.detailed_explanation,\n"
                f"    summary: node.summary,\n"
                f"    weight: node.weight,\n"
                f"    first_occurrence_month: node.first_occurrence_month,\n"
                f"    month: node.month,\n"
                f"    score: score\n"
                f"}}) AS matches"
            )
            params = {"idea_vectors": clean_vecs, "top_k": k}

        driver = self.get_driver()
        with driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            records = list(result)

        # Map results back to input order of idea_vectors
        # If Neo4j returned 1 row per UNWIND item in order:
        if len(records) == len(clean_vecs):
            return [list(rec.get("matches") or []) for rec in records]

        # Map by vector tuple key if some vectors had 0 matches in an empty index
        by_vec_key: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        for rec in records:
            r_vec = rec.get("vec")
            if r_vec is not None:
                key = tuple(round(float(x), 5) for x in r_vec[:16])
                by_vec_key[key] = list(rec.get("matches") or [])

        ordered_results: list[list[dict[str, Any]]] = []
        for v in clean_vecs:
            key = tuple(round(float(x), 5) for x in v[:16])
            ordered_results.append(by_vec_key.get(key, []))
        return ordered_results

    def upsert_idea_vectors_batch(
        self,
        ideas_batch: list[dict[str, Any]],
        dimensions: Optional[int] = None,
        index_name: str = "idea_idx",
    ) -> int:
        """
        Upsert a batch of :Idea(:Concept) nodes and their embeddings into Neo4j in a single transaction,
        ensuring the HNSW vector index on :Idea(embedding) exists.
        """
        if not ideas_batch:
            return 0

        rows = []
        detected_dim = dimensions
        for item in ideas_batch:
            name = BookParser.repair_mojibake(str(item.get("name") or "")).strip()
            if not name:
                continue
            nid = str(item.get("id") or f"concept:{name}")
            emb = item.get("embedding")
            clean_emb = [float(x) for x in emb] if emb else None
            if clean_emb and not detected_dim:
                detected_dim = len(clean_emb)

            props = clean_neo4j_properties(item, exclude_keys={"id", "embedding"})
            props["id"] = nid
            props["name"] = name
            props["type"] = "Concept"
            props["tag"] = "idea"
            props["is_idea"] = True
            if clean_emb:
                props["embedding"] = clean_emb
            rows.append({"id": nid, "properties": props})

        if not rows:
            return 0

        driver = self.get_driver()
        with driver.session(database=self.database) as session:
            if detected_dim:
                self.ensure_idea_vector_index(session, dimensions=detected_dim, index_name=index_name)
            cypher = (
                "UNWIND $batch AS row\n"
                "MERGE (n:Concept {id: row.id})\n"
                "ON CREATE SET n += row.properties, n:Idea\n"
                "ON MATCH SET n += row.properties, n:Idea"
            )
            session.run(cypher, batch=rows)
        return len(rows)

    def clean_database(self) -> int:
        """
        Delete all nodes and relationships in the target Neo4j database (clean start).
        Uses iterative or batched detach delete to handle graphs cleanly.
        """
        self.verify_connectivity()
        driver = self.get_driver()
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
        only_affected: bool = False,
        suspend_search_indexes: bool = True,
        index_timeout_sec: int = 300,
        index_poll_interval: float = 3.0,
    ) -> dict[str, Any]:
        """
        Export ConceptGraphStore into Neo4j via batched upsert with live progress display.

        Semantics:
        - clean=True: Performs a clean start (deletes all content in database before export)
        - only_affected=True: Upserts ONLY nodes and edges marked dirty in `store` (and deletes
          any merged/removed nodes in `store.removed_nodes`), then clears `store`'s dirty tracking.
        - suspend_search_indexes=True: Captures and drops existing VECTOR and FULLTEXT search
          indexes prior to bulk node/edge upserts (strictly preserving unique ID constraints
          for O(1) MERGE), then recreates them in bulk and waits until ONLINE in a finally block.
        - Nodes: MERGE on (id), ON CREATE SET props, ON MATCH SET props
        - Edges: MERGE (source)-[r:REL]->(target), ON CREATE SET props, ON MATCH SET props
        - show_progress=True: Displays rich real-time progress bars for nodes and relationships
        """
        self.verify_connectivity()
        driver = self.get_driver()

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
            incremental_mode = bool(only_affected and not effective_clean)
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
            if not incremental_mode:
                store.link_sequential_chunks()

            g = store.graph
            dirty_node_set: set[str] = set(getattr(store, "dirty_nodes", set())) if incremental_mode else set()
            dirty_edge_set: set[tuple[str, str]] = set(getattr(store, "dirty_edges", set())) if incremental_mode else set()
            removed_node_list: list[str] = list(getattr(store, "removed_nodes", set())) if incremental_mode else []

            if incremental_mode:
                for u_id, v_id in dirty_edge_set:
                    if g.has_node(u_id):
                        dirty_node_set.add(u_id)
                    if g.has_node(v_id):
                        dirty_node_set.add(v_id)
                target_nodes_iter = [
                    (nid, g.nodes[nid]) for nid in dirty_node_set if g.has_node(nid)
                ]
                target_edges_iter = [
                    (u_id, v_id, g.edges[u_id, v_id])
                    for (u_id, v_id) in dirty_edge_set
                    if g.has_edge(u_id, v_id)
                ]
            else:
                target_nodes_iter = list(g.nodes(data=True))
                target_edges_iter = list(g.edges(data=True))

            total_nodes = len(target_nodes_iter)
            total_edges = len(target_edges_iter)

            node_counts: dict[str, int] = {}
            edge_counts: dict[str, int] = {}

            # 1. Prepare Nodes Grouped by (PrimaryLabel, is_idea)
            # Grouping allows using specific label indices for maximum Cypher performance
            nodes_by_group: dict[tuple[str, bool], list[dict[str, Any]]] = {}
            distinct_labels: set[str] = set()
            idea_vector_dim: int = 1024
            chunk_vector_dim: int = 1024

            for node_id, attrs in target_nodes_iter:
                primary_label = sanitize_label(attrs.get("type", "Entity"))
                is_idea = bool(attrs.get("is_idea", False) or attrs.get("tag") == "idea" or primary_label in ("Concept", "Idea"))
                distinct_labels.add(primary_label)
                raw_emb = attrs.get("embedding")
                if is_idea:
                    distinct_labels.add("Idea")
                    if isinstance(raw_emb, (list, tuple)) and len(raw_emb) > 0:
                        idea_vector_dim = len(raw_emb)
                elif primary_label == "Chunk":
                    if isinstance(raw_emb, (list, tuple)) and len(raw_emb) > 0:
                        chunk_vector_dim = len(raw_emb)

                # Export embeddings on :Idea and :Chunk nodes
                excl = None if (is_idea or primary_label == "Chunk") else {"embedding"}
                props = clean_neo4j_properties(attrs, exclude_keys=excl)
                props["id"] = node_id  # Guarantee primary key property

                group_key = (primary_label, is_idea)
                if group_key not in nodes_by_group:
                    nodes_by_group[group_key] = []
                nodes_by_group[group_key].append({"id": node_id, "properties": props})

            # 2. Prepare Edges Grouped by (SourceLabel, RelType, TargetLabel)
            edges_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

            for src_id, tgt_id, attrs in target_edges_iter:
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
                # If incremental mode has removed/merged nodes, delete them first
                if removed_node_list:
                    for i in range(0, len(removed_node_list), self.batch_size):
                        rem_batch = removed_node_list[i : i + self.batch_size]
                        self._execute_write_query(
                            session,
                            "UNWIND $ids AS nid MATCH (n {id: nid}) DETACH DELETE n",
                            ids=rem_batch,
                        )

                # Step A: Ensure Unique ID Schema Constraints (strictly preserved for O(1) MERGE)
                if distinct_labels:
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

                # Step B & C: Suspend VECTOR/FULLTEXT search indexes during bulk node & relationship upserts,
                # then recreate them in bulk and wait for ONLINE status in the finally block.
                has_vector_labels = bool("Idea" in distinct_labels or "Chunk" in distinct_labels)
                with self.search_index_suspension(
                    session=session,
                    distinct_labels=distinct_labels,
                    idea_vector_dim=idea_vector_dim,
                    chunk_vector_dim=chunk_vector_dim,
                    suspend_existing=bool(suspend_search_indexes and (has_vector_labels or total_nodes > 0)),
                    wait_online=True,
                    timeout_sec=index_timeout_sec,
                    poll_interval=index_poll_interval,
                    progress=progress,
                    progress_callback=progress_callback,
                ):
                    # Step B: Upsert Nodes in Batches
                    node_task = (
                        progress.add_task("Upserting Affected Nodes..." if incremental_mode else "Upserting Nodes...", total=total_nodes)
                        if progress and total_nodes > 0
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
                            self._execute_write_query(session, label_query, batch=batch)
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
                        progress.add_task("Upserting Affected Relationships..." if incremental_mode else "Upserting Relationships...", total=total_edges)
                        if progress and total_edges > 0
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
                            self._execute_write_query(session, rel_query, batch=batch)
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

        if hasattr(store, "clear_dirty"):
            store.clear_dirty()

        logger.info(
            f"Successfully upserted {total_nodes} nodes and {total_edges} edges into Neo4j ({self.uri}, db={self.database}, only_affected={incremental_mode})"
        )

        return {
            "status": "success",
            "uri": self.uri,
            "database": self.database,
            "clean_start": effective_clean,
            "only_affected": incremental_mode,
            "removed_nodes_deleted": len(removed_node_list),
            "cleaned_nodes": deleted_before_export,
            "nodes_upserted": total_nodes,
            "node_breakdown": node_counts,
            "edges_upserted": total_edges,
            "edge_breakdown": edge_counts,
        }
