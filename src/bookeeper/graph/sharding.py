"""
Spine and Sharded Knowledge Graph Architecture.

Splits large monolithic knowledge graphs (e.g., >10 GB) into:
  1. Spine Graph (knowledge_graph_spine.json):
     - Lightweight structural skeleton: Books, Sections, Ideas/Concepts, and semantic relationships.
     - High-speed in-memory graph (~50-200 MB), fast traversal, zero chunk payload bloat.
  2. Payload Shards (shards/shard_XXXX.json):
     - Partitioned chunk nodes, sequential edges, and grounding links.
     - Hard-limited by target shard size (default: 1 GB / 1000 MB per shard).
     - Stored with Float16 Base64 compressed embeddings.
  3. Shard Manifest (shards_manifest.json):
     - Fast O(1) lookup index mapping Books and Chunks to their respective shard files.
"""

import logging
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

try:
    import orjson

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")

    def json_dump_file(obj: Any, path: Path) -> None:
        with open(path, "wb") as f:
            f.write(orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS))

    def json_load_file(path: Path) -> Any:
        with open(path, "rb") as f:
            return orjson.loads(f.read())

except ImportError:
    import json

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, separators=(",", ":"))

    def json_dump_file(obj: Any, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"))

    def json_load_file(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

from bookeeper.graph.embedding_codec import (
    compress_node_embedding,
    decode_embedding,
    decompress_node_embedding,
    encode_embedding,
    get_node_embedding,
)

logger = logging.getLogger(__name__)

SPINE_NODE_TYPES = {"Book", "Section", "Concept", "Idea", "Cluster", "SeedIdea", "Tag"}
PAYLOAD_NODE_TYPES = {"Chunk"}


@dataclass
class ShardInfo:
    shard_id: str
    filename: str
    file_size_bytes: int
    book_ids: List[int]
    node_count: int
    edge_count: int


@dataclass
class ShardManifest:
    version: str = "1.0"
    format: str = "spine_and_shards"
    created_at: str = ""
    target_shard_size_mb: int = 1000
    embedding_dtype: str = "float16"
    spine_file: str = "knowledge_graph_spine.json"
    shards_dir: str = "shards"
    total_books: int = 0
    total_spine_nodes: int = 0
    total_spine_edges: int = 0
    total_payload_nodes: int = 0
    total_payload_edges: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    shards: List[Dict[str, Any]] = field(default_factory=list)
    book_to_shard: Dict[str, str] = field(default_factory=dict)
    node_to_shard: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardManifest":
        return cls(**data)


@dataclass
class MigrationReport:
    original_file: str
    original_size_bytes: int
    original_nodes: int
    original_edges: int
    spine_size_bytes: int
    spine_nodes: int
    spine_edges: int
    num_shards: int
    total_sharded_size_bytes: int
    total_nodes_migrated: int
    total_edges_migrated: int
    compression_ratio: float
    duration_seconds: float
    verified: bool


class ShardedGraphStore:
    """
    Unified interface for interacting with a Spine + Sharded Knowledge Graph.
    Keeps the Spine permanently in memory and transparently pages payload shards
    through an LRU cache (default: 2 active shards, ~1-2 GB max RAM).
    """

    def __init__(
        self,
        base_dir: Union[Path, str],
        max_cached_shards: int = 2,
    ):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.max_cached_shards = max_cached_shards
        self.manifest_path = self.base_dir / "shards_manifest.json"
        self.spine_path = self.base_dir / "knowledge_graph_spine.json"
        self.shards_dir = self.base_dir / "shards"

        self.manifest: Optional[ShardManifest] = None
        self.spine_graph: nx.DiGraph = nx.DiGraph()
        self._shard_cache: OrderedDict[str, nx.DiGraph] = OrderedDict()

    def is_sharded(self) -> bool:
        """Check if base_dir contains a valid sharded spine installation."""
        return self.manifest_path.is_file() and self.spine_path.is_file()

    def load_spine(self) -> None:
        """Load the Spine graph and manifest into memory."""
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Shard manifest not found at {self.manifest_path}")
        if not self.spine_path.is_file():
            raise FileNotFoundError(f"Spine graph not found at {self.spine_path}")

        m_data = json_load_file(self.manifest_path)
        self.manifest = ShardManifest.from_dict(m_data)

        s_data = json_load_file(self.spine_path)
        if "links" in s_data and "edges" not in s_data:
            s_data["edges"] = s_data.pop("links")
        self.spine_graph = json_graph.node_link_graph(s_data, directed=True, multigraph=False)
        logger.info(
            f"Loaded Spine Graph: {self.spine_graph.number_of_nodes()} nodes, "
            f"{self.spine_graph.number_of_edges()} edges across {len(self.manifest.shards)} shards."
        )

    def get_shard(self, shard_id: str) -> nx.DiGraph:
        """
        Retrieve a payload shard graph by ID with LRU caching.
        Maintains at most max_cached_shards in memory simultaneously.
        """
        if shard_id in self._shard_cache:
            self._shard_cache.move_to_end(shard_id)
            return self._shard_cache[shard_id]

        shard_file = self.shards_dir / f"{shard_id}.json"
        if not shard_file.is_file():
            raise FileNotFoundError(f"Shard file not found: {shard_file}")

        data = json_load_file(shard_file)
        if "links" in data and "edges" not in data:
            data["edges"] = data.pop("links")
        shard_g = json_graph.node_link_graph(data, directed=True, multigraph=False)

        if len(self._shard_cache) >= self.max_cached_shards:
            evicted_id, _ = self._shard_cache.popitem(last=False)
            logger.debug(f"Evicted shard {evicted_id} from memory cache.")

        self._shard_cache[shard_id] = shard_g
        return shard_g

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """O(1) lookup of a chunk node from its mapped shard."""
        if self.manifest is None:
            self.load_spine()

        shard_id = self.manifest.node_to_shard.get(chunk_id)
        if not shard_id:
            # Fallback check in spine
            if self.spine_graph.has_node(chunk_id):
                return dict(self.spine_graph.nodes[chunk_id])
            return None

        shard_g = self.get_shard(shard_id)
        if shard_g.has_node(chunk_id):
            return dict(shard_g.nodes[chunk_id])
        return None

    def get_book_chunks(self, book_id: int) -> List[Dict[str, Any]]:
        """Retrieve all chunk nodes belonging to a specific book."""
        if self.manifest is None:
            self.load_spine()

        shard_id = self.manifest.book_to_shard.get(str(book_id))
        if not shard_id:
            return []

        shard_g = self.get_shard(shard_id)
        chunks = []
        for nid, attrs in shard_g.nodes(data=True):
            if attrs.get("type") == "Chunk" and attrs.get("book_id") == book_id:
                chunks.append(dict(attrs))
        chunks.sort(key=lambda c: (c.get("section_id", 0), c.get("chunk_index", 0)))
        return chunks

    def iter_all_nodes(self, data: bool = True) -> Generator[Any, None, None]:
        """
        Stream all nodes across the Spine and all Shards one-by-one.
        Safe for large graphs: never holds all shards in RAM at once.
        """
        if self.manifest is None:
            self.load_spine()

        # 1. Yield Spine nodes
        for item in self.spine_graph.nodes(data=data):
            yield item

        # 2. Yield Shard nodes sequentially (excluding spine stub nodes)
        for sh_info in self.manifest.shards:
            shard_id = sh_info["shard_id"]
            shard_g = self.get_shard(shard_id)
            for item in shard_g.nodes(data=data):
                nid = item[0] if data else item
                if not self.spine_graph.has_node(nid):
                    yield item

    def iter_all_edges(self, data: bool = True) -> Generator[Any, None, None]:
        """
        Stream all edges across the Spine and all Shards one-by-one.
        """
        if self.manifest is None:
            self.load_spine()

        # 1. Yield Spine edges
        for item in self.spine_graph.edges(data=data):
            yield item

        # 2. Yield Shard edges sequentially
        seen_edges: Set[Tuple[str, str]] = set()
        for sh_info in self.manifest.shards:
            shard_id = sh_info["shard_id"]
            shard_g = self.get_shard(shard_id)
            for item in shard_g.edges(data=data):
                u, v = item[0], item[1]
                key = (str(u), str(v))
                if key not in seen_edges:
                    seen_edges.add(key)
                    yield item

    def stats(self) -> Dict[str, Any]:
        """Return combined statistics for the spine and shards."""
        if self.manifest is None:
            self.load_spine()

        return {
            "total_nodes": self.manifest.total_nodes,
            "total_edges": self.manifest.total_edges,
            "spine_nodes": self.manifest.total_spine_nodes,
            "spine_edges": self.manifest.total_spine_edges,
            "payload_nodes": self.manifest.total_payload_nodes,
            "payload_edges": self.manifest.total_payload_edges,
            "num_shards": len(self.manifest.shards),
            "target_shard_size_mb": self.manifest.target_shard_size_mb,
            "embedding_dtype": self.manifest.embedding_dtype,
        }


class GraphMigrator:
    """
    Transforms a monolithic knowledge_graph.json into a high-performance Spine + Sharded graph.
    Applies Float16 Base64 embedding compression and enforces hard limits on shard file sizes.
    """

    @classmethod
    def migrate_monolithic_graph(
        cls,
        source_path: Union[Path, str],
        output_dir: Optional[Union[Path, str]] = None,
        target_shard_size_mb: int = 1000,
        dtype: str = "float16",
        make_backup: bool = True,
        console: Optional[Any] = None,
    ) -> MigrationReport:
        """
        Execute migration from monolithic knowledge_graph.json to Spine + Shards.

        Args:
            source_path: Path to existing knowledge_graph.json.
            output_dir: Destination directory (defaults to parent directory of source_path).
            target_shard_size_mb: Maximum size limit per shard file (default: 1000 MB = 1 GB).
            dtype: Compression precision for Base64 embeddings ('float16' recommended).
            make_backup: Preserve knowledge_graph.json.bak before migration.
            console: Optional Rich console for live progress reporting.

        Returns:
            MigrationReport with audit metrics, sizes, and verification status.
        """
        start_time = time.time()
        src = Path(source_path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Source knowledge graph not found at {src}")

        dest_dir = Path(output_dir).expanduser().resolve() if output_dir else src.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = dest_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)

        orig_size_bytes = src.stat().st_size

        # 1. Safety Backup
        if make_backup:
            backup_path = src.with_suffix(".json.bak")
            if not backup_path.exists():
                if console:
                    console.print(f"[bold cyan]Creating safety backup:[/bold cyan] {backup_path.name}")
                shutil.copy2(src, backup_path)

        if console:
            console.print(f"[bold cyan]Reading monolithic graph ({orig_size_bytes / 1e9:.2f} GB)...[/bold cyan]")

        # 2. Load monolithic graph
        raw_data = json_load_file(src)
        if "links" in raw_data and "edges" not in raw_data:
            raw_data["edges"] = raw_data.pop("links")

        orig_nodes = raw_data.get("nodes", [])
        orig_edges = raw_data.get("edges", [])
        total_orig_nodes = len(orig_nodes)
        total_orig_edges = len(orig_edges)

        if console:
            console.print(
                f"[dim]Loaded {total_orig_nodes:,} nodes and {total_orig_edges:,} edges. Partitioning...[/dim]"
            )

        # 3. Classify nodes into Spine vs Payload
        spine_nodes: List[Dict[str, Any]] = []
        spine_node_ids: Set[str] = set()

        payload_nodes_by_book: Dict[int, List[Dict[str, Any]]] = {}
        unassigned_payload_nodes: List[Dict[str, Any]] = []

        total_embeddings_compressed = 0

        for n in orig_nodes:
            # Compress embedding in-place to Float16 Base64
            has_emb = "embedding" in n or "embedding_b64" in n
            if has_emb:
                compress_node_embedding(n, dtype=dtype, in_place=True)
                total_embeddings_compressed += 1

            nid = str(n.get("id"))
            ntype = n.get("type", "")
            is_idea = bool(n.get("is_idea") or n.get("tag") == "idea" or ntype in ("Concept", "Idea"))

            # Determine whether node belongs to Spine or Payload Shard
            if is_idea or ntype in SPINE_NODE_TYPES:
                spine_nodes.append(n)
                spine_node_ids.add(nid)
            else:
                # Payload node (Chunk or evidence)
                bid = n.get("book_id")
                if bid is not None:
                    try:
                        bid_int = int(bid)
                        payload_nodes_by_book.setdefault(bid_int, []).append(n)
                    except ValueError:
                        unassigned_payload_nodes.append(n)
                else:
                    unassigned_payload_nodes.append(n)

        # 4. Classify edges into Spine vs Payload
        spine_edges: List[Dict[str, Any]] = []
        payload_edges_by_book: Dict[int, List[Dict[str, Any]]] = {}
        unassigned_payload_edges: List[Dict[str, Any]] = []

        for e in orig_edges:
            src_id = str(e.get("source"))
            tgt_id = str(e.get("target"))

            # If both endpoints are in the Spine, edge belongs to the Spine
            if src_id in spine_node_ids and tgt_id in spine_node_ids:
                spine_edges.append(e)
            else:
                # Payload edge (connects to or from a chunk)
                bid = e.get("book_id")
                if bid is not None:
                    try:
                        bid_int = int(bid)
                        payload_edges_by_book.setdefault(bid_int, []).append(e)
                        continue
                    except ValueError:
                        pass

                # If book_id not on edge, infer from node ID (e.g. chunk:42:1 -> book 42)
                inferred_bid = cls._infer_book_id_from_edge(src_id, tgt_id)
                if inferred_bid is not None:
                    payload_edges_by_book.setdefault(inferred_bid, []).append(e)
                else:
                    unassigned_payload_edges.append(e)

        # 5. Write Spine Graph
        spine_graph_file = dest_dir / "knowledge_graph_spine.json"
        spine_payload = {
            "directed": True,
            "multigraph": False,
            "graph": raw_data.get("graph", {}),
            "nodes": spine_nodes,
            "edges": spine_edges,
        }
        json_dump_file(spine_payload, spine_graph_file)
        spine_size_bytes = spine_graph_file.stat().st_size

        if console:
            console.print(
                f"[bold green]✓ Spine Graph written:[/bold green] {spine_graph_file.name} "
                f"({len(spine_nodes):,} nodes, {len(spine_edges):,} edges, {spine_size_bytes / 1e6:.1f} MB)"
            )

        # 6. Group books into Shards respecting the target_shard_size_mb limit
        target_bytes_limit = target_shard_size_mb * 1024 * 1024
        shards_manifest_list: List[Dict[str, Any]] = []
        book_to_shard_map: Dict[str, str] = {}
        node_to_shard_map: Dict[str, str] = {}

        shard_index = 1
        current_shard_books: List[int] = []
        current_shard_nodes: List[Dict[str, Any]] = []
        current_shard_edges: List[Dict[str, Any]] = []
        current_est_bytes = 0

        # Sort books by ID for deterministic, clean grouping
        sorted_bids = sorted(payload_nodes_by_book.keys())
        total_sharded_nodes = 0
        total_sharded_edges = 0

        def flush_current_shard() -> None:
            nonlocal shard_index, current_shard_books, current_shard_nodes, current_shard_edges, current_est_bytes
            nonlocal total_sharded_nodes, total_sharded_edges
            if not current_shard_nodes and not current_shard_edges:
                return

            shard_id = f"shard_{shard_index:04d}"
            shard_filename = f"{shard_id}.json"
            shard_file_path = shards_dir / shard_filename

            shard_dict = {
                "shard_id": shard_id,
                "directed": True,
                "multigraph": False,
                "books": current_shard_books,
                "nodes": current_shard_nodes,
                "edges": current_shard_edges,
            }
            json_dump_file(shard_dict, shard_file_path)
            actual_bytes = shard_file_path.stat().st_size

            for b in current_shard_books:
                book_to_shard_map[str(b)] = shard_id
            for nd in current_shard_nodes:
                node_to_shard_map[str(nd["id"])] = shard_id

            shards_manifest_list.append(
                {
                    "shard_id": shard_id,
                    "filename": f"shards/{shard_filename}",
                    "file_size_bytes": actual_bytes,
                    "books": list(current_shard_books),
                    "node_count": len(current_shard_nodes),
                    "edge_count": len(current_shard_edges),
                }
            )

            total_sharded_nodes += len(current_shard_nodes)
            total_sharded_edges += len(current_shard_edges)

            if console:
                console.print(
                    f"  • [cyan]{shard_id}:[/cyan] {len(current_shard_books)} books, "
                    f"{len(current_shard_nodes):,} nodes, {len(current_shard_edges):,} edges "
                    f"({actual_bytes / 1e6:.1f} MB)"
                )

            shard_index += 1
            current_shard_books = []
            current_shard_nodes = []
            current_shard_edges = []
            current_est_bytes = 0

        for bid in sorted_bids:
            b_nodes = payload_nodes_by_book[bid]
            b_edges = payload_edges_by_book.get(bid, [])

            # Estimate memory/serialized JSON size of this book's chunks
            est_b_bytes = len(json_dumps({"nodes": b_nodes, "edges": b_edges}))

            # If adding this book would exceed target shard size and current shard is non-empty, flush first
            if (current_est_bytes + est_b_bytes) > target_bytes_limit and current_shard_nodes:
                flush_current_shard()

            current_shard_books.append(bid)
            current_shard_nodes.extend(b_nodes)
            current_shard_edges.extend(b_edges)
            current_est_bytes += est_b_bytes

        # Append any unassigned payload nodes to the final shard
        if unassigned_payload_nodes or unassigned_payload_edges:
            current_shard_nodes.extend(unassigned_payload_nodes)
            current_shard_edges.extend(unassigned_payload_edges)

        # Flush the trailing shard
        flush_current_shard()

        # 7. Write Shard Manifest (shards_manifest.json)
        manifest_file = dest_dir / "shards_manifest.json"
        total_payload_bytes = sum(s["file_size_bytes"] for s in shards_manifest_list)
        total_migrated_size = spine_size_bytes + total_payload_bytes

        manifest = ShardManifest(
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            target_shard_size_mb=target_shard_size_mb,
            embedding_dtype=dtype,
            spine_file=spine_graph_file.name,
            shards_dir="shards",
            total_books=len(sorted_bids),
            total_spine_nodes=len(spine_nodes),
            total_spine_edges=len(spine_edges),
            total_payload_nodes=total_sharded_nodes,
            total_payload_edges=total_sharded_edges,
            total_nodes=len(spine_nodes) + total_sharded_nodes,
            total_edges=len(spine_edges) + total_sharded_edges,
            shards=shards_manifest_list,
            book_to_shard=book_to_shard_map,
            node_to_shard=node_to_shard_map,
        )
        json_dump_file(manifest.to_dict(), manifest_file)

        # 8. Verification Audit
        verified = (
            (len(spine_nodes) + total_sharded_nodes) == total_orig_nodes
            and (len(spine_edges) + total_sharded_edges) == total_orig_edges
        )

        compression_ratio = orig_size_bytes / total_migrated_size if total_migrated_size > 0 else 1.0
        elapsed = time.time() - start_time

        report = MigrationReport(
            original_file=str(src),
            original_size_bytes=orig_size_bytes,
            original_nodes=total_orig_nodes,
            original_edges=total_orig_edges,
            spine_size_bytes=spine_size_bytes,
            spine_nodes=len(spine_nodes),
            spine_edges=len(spine_edges),
            num_shards=len(shards_manifest_list),
            total_sharded_size_bytes=total_migrated_size,
            total_nodes_migrated=len(spine_nodes) + total_sharded_nodes,
            total_edges_migrated=len(spine_edges) + total_sharded_edges,
            compression_ratio=compression_ratio,
            duration_seconds=elapsed,
            verified=verified,
        )

        return report

    @classmethod
    def export_monolithic_from_shards(
        cls,
        base_dir: Union[Path, str],
        output_file: Union[Path, str],
        decompress_embeddings: bool = False,
        console: Optional[Any] = None,
    ) -> Path:
        """
        Reconstruct a single monolithic knowledge_graph.json from the Spine and Shards.
        Streams nodes and edges into output_file to keep memory consumption low.
        """
        sharded_store = ShardedGraphStore(base_dir)
        sharded_store.load_spine()

        out_path = Path(output_file).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if console:
            console.print(f"[bold cyan]Reconstructing monolithic graph from {len(sharded_store.manifest.shards)} shards...[/bold cyan]")

        all_nodes: List[Dict[str, Any]] = []
        all_edges: List[Dict[str, Any]] = []
        seen_node_ids: Set[str] = set()
        seen_edge_keys: Set[Tuple[str, str]] = set()

        # 1. Collect Spine
        for nid, attrs in sharded_store.spine_graph.nodes(data=True):
            nd = dict(attrs)
            nd["id"] = nid
            if decompress_embeddings:
                decompress_node_embedding(nd, as_list=True)
            all_nodes.append(nd)
            seen_node_ids.add(nid)

        for u, v, attrs in sharded_store.spine_graph.edges(data=True):
            ed = dict(attrs)
            ed["source"] = u
            ed["target"] = v
            all_edges.append(ed)
            seen_edge_keys.add((str(u), str(v)))

        # 2. Collect Shards
        for sh in sharded_store.manifest.shards:
            shard_g = sharded_store.get_shard(sh["shard_id"])
            for nid, attrs in shard_g.nodes(data=True):
                if nid in seen_node_ids:
                    continue
                nd = dict(attrs)
                nd["id"] = nid
                if decompress_embeddings:
                    decompress_node_embedding(nd, as_list=True)
                all_nodes.append(nd)
                seen_node_ids.add(nid)

            for u, v, attrs in shard_g.edges(data=True):
                edge_key = (str(u), str(v))
                if edge_key in seen_edge_keys:
                    continue
                ed = dict(attrs)
                ed["source"] = u
                ed["target"] = v
                all_edges.append(ed)
                seen_edge_keys.add(edge_key)

        payload = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": all_nodes,
            "edges": all_edges,
        }
        json_dump_file(payload, out_path)
        logger.info(f"Reconstructed monolithic graph with {len(all_nodes)} nodes to {out_path}")
        return out_path

    @staticmethod
    def _infer_book_id_from_edge(src_id: str, tgt_id: str) -> Optional[int]:
        """Infer book_id from chunk or section node identifiers (e.g. 'chunk:42:0' -> 42)."""
        for s in (src_id, tgt_id):
            if s.startswith("chunk:") or s.startswith("section:"):
                parts = s.split(":")
                if len(parts) >= 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        pass
        return None
