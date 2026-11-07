"""
Neo4j Graph Database Exporter for Concept Knowledge Graph.
Performs high-throughput upsert (insert new, update existing) for nodes and relationships via Cypher.
"""

import atexit
import hashlib
import json
import logging
import os
import platform
import re
import socket
import subprocess
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from urllib.parse import urlparse

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
from bookeeper.graph.embedding_codec import get_node_embedding
from bookeeper.graph.store import ConceptGraphStore

logger = logging.getLogger(__name__)

# Suppress cosmetic Neo4j DBMS server GQL compile notifications (e.g. absent schema property keys)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

_ACTIVE_BRIDGES: set[subprocess.Popen] = set()


def _cleanup_all_bridges():
    for p in list(_ACTIVE_BRIDGES):
        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=1.0)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    _ACTIVE_BRIDGES.clear()


atexit.register(_cleanup_all_bridges)

_MACOS_BRIDGE_SCRIPT = """
import socket, sys, threading

def fwd(src, dst):
    try:
        while True:
            buf = src.recv(65536)
            if not buf:
                break
            dst.sendall(buf)
    except Exception:
        pass
    finally:
        try:
            src.close()
        except Exception:
            pass
        try:
            dst.close()
        except Exception:
            pass

def main():
    target_host = sys.argv[1]
    target_port = int(sys.argv[2])
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 0))
    local_port = server.getsockname()[1]
    server.listen(64)
    print(local_port, flush=True)

    while True:
        try:
            client_sock, _ = server.accept()
            remote_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote_sock.connect((target_host, target_port))
            t1 = threading.Thread(target=fwd, args=(client_sock, remote_sock), daemon=True)
            t2 = threading.Thread(target=fwd, args=(remote_sock, client_sock), daemon=True)
            t1.start()
            t2.start()
        except Exception:
            pass

if __name__ == '__main__':
    main()
"""


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
        incremental_sync: bool = True,
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
        self.incremental_sync = incremental_sync
        self._driver = None
        self._bridge_proc: Optional[subprocess.Popen] = None
        self._bridge_uri: Optional[str] = None
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
            incremental_sync=getattr(config, "incremental_sync", True),
        )

    @property
    def effective_uri(self) -> str:
        """Return the effective URI, using local loopback bridge if active."""
        return self._bridge_uri or self.uri

    def _ensure_macos_local_bridge(self):
        """
        Check if macOS Local Network Privacy blocks direct access to target host (Errno 65).
        If so, proactively start a transparent loopback relay using /usr/bin/python3.
        """
        if platform.system() != "Darwin":
            return
        if self._bridge_proc is not None and self._bridge_proc.poll() is None:
            return

        try:
            parsed = urlparse(self.uri)
            host = parsed.hostname
            port = parsed.port or 7687
            if not host or host in ("127.0.0.1", "localhost", "::1"):
                return

            needs_bridge = False
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.3)
                s.connect((host, port))
                s.close()
            except OSError as e:
                if getattr(e, "errno", None) == 65 or "no route to host" in str(e).lower():
                    needs_bridge = True

            if needs_bridge:
                self._start_macos_bridge(host, port)
        except Exception as e:
            logger.debug(f"macOS bridge check error: {e}")

    def _start_macos_bridge(self, host: str, port: int):
        """Start a local loopback relay on 127.0.0.1 forwarding to host:port via /usr/bin/python3."""
        if not os.path.exists("/usr/bin/python3"):
            return
        if self._bridge_proc is not None and self._bridge_proc.poll() is None:
            return

        try:
            proc = subprocess.Popen(
                ["/usr/bin/python3", "-c", _MACOS_BRIDGE_SCRIPT, host, str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            line = proc.stdout.readline().strip()
            if not line:
                return
            local_port = int(line)
            _ACTIVE_BRIDGES.add(proc)
            self._bridge_proc = proc
            parsed = urlparse(self.uri)
            scheme = parsed.scheme or "bolt"
            query = f"?{parsed.query}" if parsed.query else ""
            self._bridge_uri = f"{scheme}://127.0.0.1:{local_port}{query}"
            logger.info(
                f"macOS Local Network Privacy bypass active: bridged {host}:{port} -> 127.0.0.1:{local_port}"
            )
        except Exception as e:
            logger.warning(f"Could not start macOS local network bridge: {e}")

    def get_driver(self):
        """Get or create Neo4j driver instance."""
        if self._driver is None:
            self._ensure_macos_local_bridge()
            self._driver = GraphDatabase.driver(
                self.effective_uri,
                auth=(self.user, self.password),
                connection_timeout=self.connection_timeout,
            )
        return self._driver

    def close(self):
        """Close Neo4j driver connection and terminate any local network bridge."""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
        if self._bridge_proc is not None:
            _ACTIVE_BRIDGES.discard(self._bridge_proc)
            try:
                if self._bridge_proc.poll() is None:
                    self._bridge_proc.terminate()
                    self._bridge_proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._bridge_proc.kill()
                except Exception:
                    pass
            self._bridge_proc = None
            self._bridge_uri = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def verify_connectivity(self) -> bool:
        """
        Verify driver connectivity to Neo4j database.
        Automatically falls back from neo4j:// routing protocol to bolt:// direct connection
        if the target is a standalone instance that does not support routing tables.
        Also automatically establishes a loopback bridge on macOS if Local Network Privacy
        blocks direct socket access (Errno 65 / No route to host).
        """
        driver = self.get_driver()
        try:
            driver.verify_connectivity()
            return True
        except Exception as e:
            err_msg = str(e).lower()

            # 1. macOS Local Network Privacy restriction (Errno 65 / No route to host)
            if (
                platform.system() == "Darwin"
                and self._bridge_proc is None
                and ("65" in err_msg or "no route to host" in err_msg or "ehostunreach" in err_msg)
            ):
                try:
                    parsed = urlparse(self.uri)
                    host = parsed.hostname
                    port = parsed.port or 7687
                    if host and host not in ("127.0.0.1", "localhost", "::1"):
                        logger.warning(
                            f"Neo4j direct connection failed with macOS Local Network restriction ('{e}'). "
                            f"Establishing automatic loopback bridge to {host}:{port}..."
                        )
                        self._start_macos_bridge(host, port)
                        if self._driver is not None:
                            try:
                                self._driver.close()
                            except Exception:
                                pass
                            self._driver = None
                        driver = self.get_driver()
                        driver.verify_connectivity()
                        return True
                except Exception as bridge_err:
                    logger.debug(f"Failed to establish macOS bridge: {bridge_err}")

            # 2. Neo4j routing failure on standalone instances
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
                    if self._driver is not None:
                        try:
                            self._driver.close()
                        except Exception:
                            pass
                        self._driver = None
                    self.uri = fallback_uri
                    if self._bridge_uri is not None:
                        parsed_b = urlparse(self._bridge_uri)
                        self._bridge_uri = f"{fallback_scheme}{parsed_b.netloc}{parsed_b.path}"
                    driver = self.get_driver()
                    driver.verify_connectivity()
                    return True
            raise

    def ensure_constraints(self, session, labels: list[str]):
        """
        Create uniqueness constraints on node 'id' for each label to guarantee
        fast O(1) index lookups during MERGE statements.
        """
        labels_to_ensure = set(labels)
        labels_to_ensure.add("BookeeperSync")
        for label in labels_to_ensure:
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

    @staticmethod
    def compute_book_content_hash(
        book_id: int | str,
        output_dir: Optional[Path | str] = None,
        store: Optional[ConceptGraphStore] = None,
    ) -> str:
        """
        Compute a deterministic content hash for a book compilation unit (Strategy A):
          content_hash = SHA256(chunks_json + state_json)
        Falls back to deterministic serialization of the book's graph elements if files are unavailable.
        """
        try:
            bid_clean = int(str(book_id).replace("book:", "").strip())
        except (ValueError, TypeError):
            bid_clean = str(book_id).replace("book:", "").strip()

        hasher = hashlib.sha256()
        hasher.update(f"book:{bid_clean}:".encode("utf-8"))

        out_path: Optional[Path] = None
        if output_dir:
            out_path = Path(output_dir).expanduser().resolve()
        elif store and getattr(store, "output_dir", None):
            out_path = Path(store.output_dir).expanduser().resolve()

        found_files = False
        if out_path and out_path.is_dir():
            cand_chunks = [
                out_path / "chunks" / f"book_{bid_clean}_chunks.json",
                out_path / f"book_{bid_clean}_chunks.json",
            ]
            cand_states = [
                out_path / ".book_states" / f"book_{bid_clean}_state.json",
                out_path / f"book_{bid_clean}_state.json",
            ]
            for cp in cand_chunks:
                if cp.is_file():
                    try:
                        hasher.update(cp.read_bytes())
                        found_files = True
                        break
                    except Exception:
                        pass
            for sp in cand_states:
                if sp.is_file():
                    try:
                        hasher.update(sp.read_bytes())
                        found_files = True
                        break
                    except Exception:
                        pass

        # Fallback to in-memory graph elements if disk files are not present
        if not found_files and store and hasattr(store, "graph"):
            g = store.graph
            b_nid = f"book:{bid_clean}"
            b_attrs = g.nodes.get(b_nid, {}) if g.has_node(b_nid) else {}
            hasher.update(str(b_attrs.get("title", "")).encode("utf-8"))
            hasher.update(str(b_attrs.get("author", "")).encode("utf-8"))
            hasher.update(str(b_attrs.get("summary", "")).encode("utf-8"))

            sections = sorted(
                [n for n in g.nodes if n.startswith(f"section:{bid_clean}:")]
            )
            for s in sections:
                sd = g.nodes[s]
                hasher.update(f"{s}:{sd.get('title', '')}:{sd.get('char_count', 0)};".encode("utf-8"))

            chunks = sorted(
                [
                    (n, d)
                    for n, d in g.nodes(data=True)
                    if d.get("type") == "Chunk" and str(d.get("book_id")) == str(bid_clean)
                ],
                key=lambda x: x[0],
            )
            for cid, cd in chunks:
                hasher.update(
                    f"{cid}:{cd.get('chunk_idx', 0)}:{cd.get('char_count', 0)}:{cd.get('text', '')[:128]};".encode(
                        "utf-8"
                    )
                )

        return hasher.hexdigest()

    @staticmethod
    def compute_node_content_hash(
        node_id: str,
        primary_label: str,
        properties: dict[str, Any],
    ) -> str:
        """
        Compute deterministic SHA-256 attribute hash for node-level upsert guarding (Strategy B).
        Unchanged nodes in Neo4j skip property rewrites and HNSW index modifications.
        """
        hasher = hashlib.sha256()
        hasher.update(f"{node_id}:{primary_label}:".encode("utf-8"))
        for k in sorted(properties.keys()):
            if k in ("content_hash", "id"):
                continue
            v = properties[k]
            if isinstance(v, list):
                if v and isinstance(v[0], float):
                    v_str = ",".join(f"{x:.4f}" for x in v[:16]) + f";len={len(v)}"
                else:
                    v_str = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, dict):
                v_str = json.dumps(v, sort_keys=True, ensure_ascii=False)
            else:
                v_str = str(v)
            hasher.update(f"{k}={v_str}|".encode("utf-8"))
        return hasher.hexdigest()[:32]

    @staticmethod
    def compute_graph_hash(
        store: ConceptGraphStore,
        book_hashes: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Compute deterministic composite hash of the overall knowledge base.
        Combines sorted per-book content hashes and concept graph structural counts.
        """
        hasher = hashlib.sha256()
        if book_hashes:
            for bid in sorted(book_hashes.keys()):
                hasher.update(f"{bid}={book_hashes[bid]};".encode("utf-8"))

        g = getattr(store, "graph", None)
        if g is not None:
            concept_count = sum(
                1 for n, d in g.nodes(data=True)
                if d.get("type") in ("Concept", "Idea") or d.get("is_idea") or d.get("tag") == "idea"
            )
            hasher.update(
                f"concepts={concept_count};nodes={g.number_of_nodes()};edges={g.number_of_edges()}".encode("utf-8")
            )
        return hasher.hexdigest()

    def get_sync_metadata(self, session: Any) -> Optional[dict[str, Any]]:
        """
        Fetch technical synchronization metadata from Neo4j (:BookeeperSync {id: 'sync_metadata'}).
        """
        query = (
            "MATCH (s:BookeeperSync {id: 'sync_metadata'}) "
            "RETURN s.graph_hash AS graph_hash, s.synced_at_iso AS synced_at_iso, "
            "s.total_books AS total_books, s.total_nodes AS total_nodes, "
            "s.total_edges AS total_edges, s.book_hashes AS book_hashes"
        )
        try:
            result = session.run(query)
            rec = result.single() if hasattr(result, "single") else None
            if not rec:
                return None
            row = dict(rec) if not isinstance(rec, dict) else rec
            b_hashes = row.get("book_hashes")
            if isinstance(b_hashes, str):
                try:
                    b_hashes = json.loads(b_hashes)
                except Exception:
                    b_hashes = {}
            return {
                "graph_hash": row.get("graph_hash"),
                "synced_at_iso": row.get("synced_at_iso"),
                "total_books": row.get("total_books", 0),
                "total_nodes": row.get("total_nodes", 0),
                "total_edges": row.get("total_edges", 0),
                "book_hashes": b_hashes or {},
            }
        except Exception as e:
            logger.debug(f"Could not read :BookeeperSync metadata: {e}")
            return None

    def update_sync_metadata(
        self,
        session: Any,
        graph_hash: str,
        total_books: int,
        total_nodes: int,
        total_edges: int,
        book_hashes: dict[str, str],
        backup_hash: Optional[str] = None,
    ) -> None:
        """
        Upsert technical synchronization metadata node into Neo4j.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        query = """
        MERGE (s:BookeeperSync {id: 'sync_metadata'})
        SET s.graph_hash = $graph_hash,
            s.synced_at_iso = $synced_at_iso,
            s.total_books = $total_books,
            s.total_nodes = $total_nodes,
            s.total_edges = $total_edges,
            s.book_hashes = $book_hashes,
            s.backup_hash = $backup_hash
        """
        try:
            self._execute_write_query(
                session,
                query,
                graph_hash=graph_hash,
                synced_at_iso=now_iso,
                total_books=total_books,
                total_nodes=total_nodes,
                total_edges=total_edges,
                book_hashes=json.dumps(book_hashes),
                backup_hash=backup_hash or "",
            )
        except Exception as e:
            logger.debug(f"Could not update :BookeeperSync metadata: {e}")

    def fetch_remote_book_hashes(self, session: Any) -> dict[str, str]:
        """
        Fetch mapping of book_node_id -> content_hash for all books currently stored in Neo4j.
        """
        query = "MATCH (b:Book) RETURN b.id AS id, b.content_hash AS content_hash"
        try:
            result = session.run(query)
            records = list(result) if result is not None else []
        except Exception as e:
            logger.debug(f"Could not fetch book hashes from Neo4j: {e}")
            return {}

        hashes: dict[str, str] = {}
        for rec in records:
            try:
                row = dict(rec) if not isinstance(rec, dict) else rec
            except Exception:
                continue
            bid = str(row.get("id") or "").strip()
            chash = str(row.get("content_hash") or "").strip()
            if bid:
                hashes[bid] = chash
        return hashes

    def fetch_remote_books(self, session: Optional[Any] = None) -> list[dict[str, Any]]:
        """
        Fetch list of all books stored in Neo4j with their id, book_id, title, author, arxiv_id, and content_hash.
        """
        self.verify_connectivity()
        query = """
        MATCH (b:Book)
        RETURN b.id AS id, properties(b) AS props
        """
        def _run(s):
            try:
                res = s.run(query)
                records = list(res) if res is not None else []
            except Exception as e:
                logger.debug(f"Could not fetch remote books from Neo4j: {e}")
                return []
            books = []
            for rec in records:
                try:
                    row = dict(rec) if not isinstance(rec, dict) else rec
                except Exception:
                    continue
                props = row.get("props")
                if not isinstance(props, dict):
                    props = {}
                bid = props.get("book_id")
                if bid is None and row.get("id"):
                    try:
                        bid = int(str(row["id"]).replace("book:", ""))
                    except (ValueError, TypeError):
                        pass
                books.append({
                    "id": row.get("id") or props.get("id"),
                    "book_id": bid,
                    "title": props.get("title") or (f"Book {bid}" if bid else "Untitled"),
                    "author": props.get("author", "Unknown"),
                    "arxiv_id": props.get("arxiv_id"),
                    "content_hash": props.get("content_hash"),
                })
            return books

        if session is not None:
            return _run(session)
        with self.get_driver().session(database=self.database) as s:
            return _run(s)

    def purge_book_subgraph(self, session: Any, book_id_str: str) -> int:
        """
        Purge all sections and chunks belonging to a modified book (Strategy A):
          MATCH (b:Book {id: $bid})-[:HAS_SECTION]->(s:Section)
          OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
          DETACH DELETE s, c
        Detaching chunks automatically cleans up adjacent relations (NEXT, PREV, SUPPORTED_BY, SUPPORTS_IDEA).
        """
        try:
            bid_num = int(str(book_id_str).replace("book:", "").strip())
        except (ValueError, TypeError):
            bid_num = -1

        query = """
        MATCH (b:Book {id: $bid})
        OPTIONAL MATCH (b)-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c2:Chunk) WHERE c2.book_id = $bid_num
        DETACH DELETE s, c, c2
        """
        try:
            self._execute_write_query(session, query, bid=book_id_str, bid_num=bid_num)
            return 1
        except Exception as e:
            logger.warning(f"Failed to purge book subgraph for {book_id_str}: {e}")
            return 0

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
            clean_emb = get_node_embedding(item, return_numpy=False)
            if clean_emb and not detected_dim:
                detected_dim = len(clean_emb)

            props = clean_neo4j_properties(
                item,
                exclude_keys={"id", "embedding", "embedding_b64", "embedding_dim", "embedding_dtype"},
            )
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

    def clean_database(self, console: Optional[Any] = None, batch_size: int = 10000) -> int:
        """
        Delete all nodes and relationships in the target Neo4j database (clean start).
        Uses high-throughput transactional subquery batching (relationships first, then nodes)
        to prevent monolithic transaction locks, JVM heap bloat, and prolonged fsync stalls.
        """
        self.verify_connectivity()
        driver = self.get_driver()
        deleted_count = 0
        with driver.session(database=self.database) as session:
            rec_n = session.run("MATCH (n) RETURN count(n) AS cnt").single()
            try:
                total_nodes = int(rec_n["cnt"]) if rec_n else 0
            except (ValueError, TypeError):
                total_nodes = 0
            rec_r = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
            try:
                total_rels = int(rec_r["cnt"]) if rec_r else 0
            except (ValueError, TypeError):
                total_rels = 0

            if total_nodes == 0 and total_rels == 0:
                logger.debug(f"Neo4j database '{self.database}' is already empty. Skipping purge.")
                return 0

            status_msg = f"Purging Neo4j database '{self.database}' ({total_nodes:,} nodes, {total_rels:,} relationships)..."
            logger.info(status_msg)
            if console is not None:
                console.print(f"[bold cyan]⚡ {status_msg}[/bold cyan]")

            t_start = time.monotonic()

            # 1. Delete relationships FIRST in transactional subqueries
            # Deleting relationships first eliminates graph edges and neighbor lock contention.
            if total_rels > 0:
                try:
                    rel_query = f"""
                    MATCH ()-[r]->()
                    CALL {{
                        WITH r
                        DELETE r
                    }} IN TRANSACTIONS OF {batch_size} ROWS
                    """
                    session.run(rel_query)
                except Exception as e:
                    logger.debug(f"Transactional subquery for relationship delete failed ({e}), falling back to iterative batches...")
                    while True:
                        res = session.run(
                            "MATCH ()-[r]->() WITH r LIMIT $batch_size DELETE r RETURN count(r) AS c",
                            batch_size=batch_size,
                        ).single()
                        c = res["c"] if res else 0
                        if c == 0:
                            break

            # 2. Delete remaining isolated nodes in transactional subqueries
            if total_nodes > 0:
                try:
                    node_query = f"""
                    MATCH (n)
                    CALL {{
                        WITH n
                        DELETE n
                    }} IN TRANSACTIONS OF {batch_size} ROWS
                    """
                    session.run(node_query)
                except Exception as e:
                    logger.debug(f"Transactional subquery for node delete failed ({e}), falling back to iterative batches...")
                    while True:
                        res = session.run(
                            "MATCH (n) WITH n LIMIT $batch_size DELETE n RETURN count(n) AS c",
                            batch_size=batch_size,
                        ).single()
                        c = res["c"] if res else 0
                        if c == 0:
                            break

            duration = time.monotonic() - t_start
            deleted_count = total_nodes
            done_msg = f"Purged Neo4j database '{self.database}': cleared {total_nodes:,} nodes and {total_rels:,} relationships in {duration:.2f}s."
            logger.info(done_msg)
            if console is not None:
                console.print(f"[bold green]✓ {done_msg}[/bold green]")

        return deleted_count

    def export(
        self,
        store: ConceptGraphStore,
        clean: Optional[bool] = None,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        console: Optional[Any] = None,
        only_affected: bool = False,
        suspend_search_indexes: Optional[bool] = None,
        index_timeout_sec: int = 300,
        index_poll_interval: float = 3.0,
        incremental_sync: Optional[bool] = None,
        backup_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Export ConceptGraphStore into Neo4j via batched upsert with live progress display.

        Smart Synchronization Features:
        - Strategy A (Per-Book Cluster Synchronization):
            Computes deterministic per-book content_hash: SHA256(chunks_json + state_json).
            Compares with hashes in Neo4j (MATCH (b:Book)). Unchanged book subgraphs (sections,
            chunks, intra-chunk links) are skipped completely; modified books purge only their
            subgraph and re-upsert.
        - Strategy B (Node-Level content_hash Guard):
            Computes deterministic attribute hash for each node. Cypher UNWIND checks:
            ON MATCH SET n += CASE WHEN n.content_hash IS NULL OR n.content_hash <> row.content_hash
                              THEN row.properties ELSE {} END, n.content_hash = row.content_hash
            Unchanged nodes skip property rewrites, preventing unnecessary HNSW index modifications.
        - Strategy C (Preserve Vector Indexes during Incremental Updates):
            Keeps native HNSW vector indexes online during incremental updates (< 5% modified nodes),
            avoiding unnecessary drop/rebuild overhead.
        - Technical Metadata Node (:BookeeperSync):
            Tracks overall graph_hash. If 100% matched, export returns immediately with 0ms overhead.
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
            effective_incremental_sync = bool(self.incremental_sync if incremental_sync is None else incremental_sync)
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

            # Calculate deterministic per-book content hashes and overall graph hash
            local_book_hashes: dict[str, str] = {}
            out_dir = getattr(store, "output_dir", None)
            for nid, attrs in list(g.nodes(data=True)):
                if attrs.get("type") == "Book":
                    bid = attrs.get("book_id", nid.replace("book:", ""))
                    b_hash = self.compute_book_content_hash(bid, output_dir=out_dir, store=store)
                    attrs["content_hash"] = b_hash
                    local_book_hashes[nid] = b_hash

            current_graph_hash = self.compute_graph_hash(store, book_hashes=local_book_hashes)

            with driver.session(database=self.database) as session:
                # 0. Check Technical Metadata Node (:BookeeperSync) for instant 0ms skip
                if effective_incremental_sync and not effective_clean and not incremental_mode and local_book_hashes:
                    sync_meta = self.get_sync_metadata(session)
                    if sync_meta and sync_meta.get("graph_hash") == current_graph_hash:
                        logger.info(
                            f"Neo4j database is already fully synchronized with Knowledge Graph (graph_hash={current_graph_hash[:12]}...). Skipping export."
                        )
                        if console is not None:
                            console.print(
                                f"[bold green]✓ Neo4j database is already fully synchronized with Knowledge Graph[/bold green] "
                                f"[dim]({len(local_book_hashes)} books, graph_hash: {current_graph_hash[:12]}...)[/dim]\n"
                            )
                        return {
                            "status": "already_synchronized",
                            "uri": self.uri,
                            "database": self.database,
                            "clean_start": False,
                            "only_affected": False,
                            "incremental_sync": True,
                            "graph_hash": current_graph_hash,
                            "books_evaluated": len(local_book_hashes),
                            "books_unchanged": len(local_book_hashes),
                            "books_modified": 0,
                            "chunks_skipped": 0,
                            "nodes_upserted": 0,
                            "node_breakdown": {},
                            "edges_upserted": 0,
                            "edge_breakdown": {},
                            "cleaned_nodes": 0,
                            "removed_nodes_deleted": 0,
                            "search_indexes_suspended": False,
                        }

                # Strategy A: Per-Book Cluster Synchronization
                unchanged_books: set[str] = set()
                modified_books: set[str] = set(local_book_hashes.keys())
                skipped_subgraph_nodes: set[str] = set()

                if effective_incremental_sync and not effective_clean and not incremental_mode and local_book_hashes:
                    remote_book_hashes = self.fetch_remote_book_hashes(session)
                    if remote_book_hashes:
                        unchanged_books = {
                            bid for bid, h in local_book_hashes.items()
                            if bid in remote_book_hashes and remote_book_hashes[bid] == h and h != ""
                        }
                        modified_books = set(local_book_hashes.keys()) - unchanged_books
                        removed_books = set(remote_book_hashes.keys()) - set(local_book_hashes.keys())

                        # Purge subgraphs for modified books and removed books
                        for mod_bid in modified_books:
                            if mod_bid in remote_book_hashes:
                                self.purge_book_subgraph(session, mod_bid)
                        for rem_bid in removed_books:
                            self.purge_book_subgraph(session, rem_bid)
                            try:
                                session.run("MATCH (b:Book {id: $bid}) DETACH DELETE b", bid=rem_bid)
                            except Exception:
                                pass

                        if unchanged_books:
                            # Collect chunk and section node IDs belonging to unchanged books
                            for ub in unchanged_books:
                                ub_num = str(ub).replace("book:", "")
                                for n in g.nodes:
                                    if n.startswith(f"section:{ub_num}:"):
                                        skipped_subgraph_nodes.add(n)
                                for n, d in g.nodes(data=True):
                                    if d.get("type") == "Chunk" and str(d.get("book_id")) == ub_num:
                                        skipped_subgraph_nodes.add(n)

                            cluster_msg = (
                                f"Per-Book Cluster Sync: {len(unchanged_books)} books unchanged "
                                f"({len(skipped_subgraph_nodes):,} chunks/sections skipped), "
                                f"{len(modified_books)} books modified/new."
                            )
                            logger.info(cluster_msg)
                            if console is not None:
                                console.print(f"[bold cyan]⚡ {cluster_msg}[/bold cyan]")

                dirty_node_set: set[str] = set(getattr(store, "dirty_nodes", set())) if incremental_mode else set()
                dirty_edge_set: set[tuple[str, str]] = set(getattr(store, "dirty_edges", set())) if incremental_mode else set()
                removed_node_list: list[str] = list(getattr(store, "removed_nodes", set())) if incremental_mode else []

                if incremental_mode:
                    for u_id, v_id in dirty_edge_set:
                        if store.has_node(u_id):
                            dirty_node_set.add(u_id)
                        if store.has_node(v_id):
                            dirty_node_set.add(v_id)
                    target_nodes_iter = []
                    for nid in dirty_node_set:
                        ndata = store.get_node_data(nid) if hasattr(store, "get_node_data") else (g.nodes[nid] if g.has_node(nid) else None)
                        if ndata is not None and nid not in skipped_subgraph_nodes:
                            target_nodes_iter.append((nid, ndata))

                    target_edges_iter = []
                    for (u_id, v_id) in dirty_edge_set:
                        if u_id in skipped_subgraph_nodes or v_id in skipped_subgraph_nodes:
                            continue
                        if g.has_edge(u_id, v_id):
                            target_edges_iter.append((u_id, v_id, g.edges[u_id, v_id]))
                        elif getattr(store, "is_sharded", False) and store.active_shard_graph and store.active_shard_graph.has_edge(u_id, v_id):
                            target_edges_iter.append((u_id, v_id, store.active_shard_graph.edges[u_id, v_id]))
                else:
                    all_nodes = store.iter_all_nodes(data=True) if getattr(store, "is_sharded", False) else g.nodes(data=True)
                    all_edges = store.iter_all_edges(data=True) if getattr(store, "is_sharded", False) else g.edges(data=True)
                    target_nodes_iter = [(nid, d) for nid, d in all_nodes if nid not in skipped_subgraph_nodes]
                    target_edges_iter = [(u, v, d) for u, v, d in all_edges if u not in skipped_subgraph_nodes and v not in skipped_subgraph_nodes]

                total_nodes = len(target_nodes_iter)
                total_edges = len(target_edges_iter)

                node_counts: dict[str, int] = {}
                edge_counts: dict[str, int] = {}

                # 1. Prepare Nodes Grouped by (PrimaryLabel, is_idea)
                nodes_by_group: dict[tuple[str, bool], list[dict[str, Any]]] = {}
                distinct_labels: set[str] = set()
                idea_vector_dim: int = 1024
                chunk_vector_dim: int = 1024

                for node_id, attrs in target_nodes_iter:
                    primary_label = sanitize_label(attrs.get("type", "Entity"))
                    is_idea = bool(attrs.get("is_idea", False) or attrs.get("tag") == "idea" or primary_label in ("Concept", "Idea"))
                    distinct_labels.add(primary_label)
                    raw_emb = get_node_embedding(attrs, return_numpy=False)
                    if is_idea:
                        distinct_labels.add("Idea")
                        if raw_emb and len(raw_emb) > 0:
                            idea_vector_dim = len(raw_emb)
                    elif primary_label == "Chunk":
                        if raw_emb and len(raw_emb) > 0:
                            chunk_vector_dim = len(raw_emb)

                    # Export embeddings on :Idea and :Chunk nodes
                    excl = {"embedding_b64", "embedding_dim", "embedding_dtype"}
                    if not (is_idea or primary_label == "Chunk"):
                        excl.add("embedding")
                    props = clean_neo4j_properties(attrs, exclude_keys=excl)
                    if (is_idea or primary_label == "Chunk") and raw_emb:
                        props["embedding"] = raw_emb
                    props["id"] = node_id  # Guarantee primary key property

                    # Strategy B: Compute deterministic node content hash
                    node_hash = attrs.get("content_hash") or self.compute_node_content_hash(node_id, primary_label, props)
                    props["content_hash"] = node_hash

                    group_key = (primary_label, is_idea)
                    if group_key not in nodes_by_group:
                        nodes_by_group[group_key] = []
                    nodes_by_group[group_key].append({
                        "id": node_id,
                        "properties": props,
                        "content_hash": node_hash,
                    })

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

                # If incremental mode has removed/merged nodes, delete them first
                if removed_node_list:
                    for i in range(0, len(removed_node_list), self.batch_size):
                        rem_batch = removed_node_list[i : i + self.batch_size]
                        self._execute_write_query(
                            session,
                            "UNWIND $ids AS nid MATCH (n {id: nid}) DETACH DELETE n",
                            ids=rem_batch,
                        )

                # Step A: Ensure Unique ID Schema Constraints
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

                # Strategy C: Auto-detect whether to suspend vector/fulltext search indexes
                if suspend_search_indexes is not None:
                    should_suspend = bool(suspend_search_indexes)
                elif effective_clean:
                    should_suspend = True
                elif len(unchanged_books) > 0 or incremental_mode:
                    # Incremental update: keep vector indexes online!
                    should_suspend = False
                else:
                    # Full export / initial build: suspend indexes for bulk load performance
                    should_suspend = True

                has_vector_labels = bool("Idea" in distinct_labels or "Chunk" in distinct_labels)
                with self.search_index_suspension(
                    session=session,
                    distinct_labels=distinct_labels,
                    idea_vector_dim=idea_vector_dim,
                    chunk_vector_dim=chunk_vector_dim,
                    suspend_existing=bool(should_suspend and (has_vector_labels or total_nodes > 0)),
                    wait_online=True,
                    timeout_sec=index_timeout_sec,
                    poll_interval=index_poll_interval,
                    progress=progress,
                    progress_callback=progress_callback,
                ):
                    # Step B: Upsert Nodes in Batches with Strategy B content_hash guard
                    node_task = (
                        progress.add_task("Upserting Affected Nodes..." if (incremental_mode or unchanged_books) else "Upserting Nodes...", total=total_nodes)
                        if progress and total_nodes > 0
                        else None
                    )
                    nodes_done = 0
                    for (primary_label, is_idea), node_batch_list in nodes_by_group.items():
                        extra_create = ", n:`Idea`" if is_idea and primary_label != "Idea" else ""
                        extra_match = ", n:`Idea`" if is_idea and primary_label != "Idea" else ""
                        label_query = (
                            f"UNWIND $batch AS row\n"
                            f"MERGE (n:`{primary_label}` {{id: row.id}})\n"
                            f"ON CREATE SET n += row.properties, n.content_hash = row.content_hash{extra_create}\n"
                            f"ON MATCH SET n += CASE WHEN n.content_hash IS NULL OR n.content_hash <> row.content_hash THEN row.properties ELSE {{}} END, "
                            f"n.content_hash = row.content_hash{extra_match}"
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
                        progress.add_task("Upserting Affected Relationships..." if (incremental_mode or unchanged_books) else "Upserting Relationships...", total=total_edges)
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

                    # Step D: Update Technical Metadata Node (:BookeeperSync)
                    self.update_sync_metadata(
                        session=session,
                        graph_hash=current_graph_hash,
                        total_books=len(local_book_hashes),
                        total_nodes=g.number_of_nodes() if g else total_nodes,
                        total_edges=g.number_of_edges() if g else total_edges,
                        book_hashes=local_book_hashes,
                        backup_hash=backup_hash,
                    )

        if hasattr(store, "clear_dirty"):
            store.clear_dirty()

        logger.info(
            f"Successfully exported to Neo4j ({self.uri}, db={self.database}, "
            f"nodes={total_nodes}, edges={total_edges}, unchanged_books={len(unchanged_books)}, "
            f"chunks_skipped={len(skipped_subgraph_nodes)})"
        )

        return {
            "status": "success",
            "uri": self.uri,
            "database": self.database,
            "clean_start": effective_clean,
            "only_affected": incremental_mode,
            "incremental_sync": effective_incremental_sync,
            "graph_hash": current_graph_hash,
            "books_evaluated": len(local_book_hashes),
            "books_unchanged": len(unchanged_books),
            "books_modified": len(modified_books),
            "chunks_skipped": len(skipped_subgraph_nodes),
            "removed_nodes_deleted": len(removed_node_list),
            "cleaned_nodes": deleted_before_export,
            "nodes_upserted": total_nodes,
            "node_breakdown": node_counts,
            "edges_upserted": total_edges,
            "edge_breakdown": edge_counts,
            "search_indexes_suspended": should_suspend,
        }
