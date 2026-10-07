"""
Concept Knowledge Graph storage abstraction backed by NetworkX.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from networkx.readwrite import json_graph

from bookeeper.config import GraphSettings
from bookeeper.processing.chunker import TextChunk
from bookeeper.processing.extractor import ConceptNode, RelationshipEdge

logger = logging.getLogger(__name__)


class ConceptGraphStore:
    """Manages the in-memory NetworkX MultiDiGraph and persistence."""

    def __init__(self, settings: Optional[GraphSettings] = None):
        self.settings = settings or GraphSettings()
        self.graph = nx.MultiDiGraph()

    def add_book_and_chunk(self, chunk: TextChunk) -> None:
        """Ensure Book and Chapter nodes exist in the graph and are linked."""
        book_node_id = f"book:{chunk.book_id}"
        if not self.graph.has_node(book_node_id):
            self.graph.add_node(
                book_node_id,
                type="Book",
                book_id=chunk.book_id,
                title=chunk.book_title,
            )

        chap_node_id = f"chapter:{chunk.book_id}:{chunk.chapter_sequence}"
        if not self.graph.has_node(chap_node_id):
            self.graph.add_node(
                chap_node_id,
                type="Chapter",
                book_id=chunk.book_id,
                sequence=chunk.chapter_sequence,
                title=chunk.chapter_title,
            )
            # Edge: Book -> CONTAINS -> Chapter
            self.graph.add_edge(book_node_id, chap_node_id, relation="CONTAINS")

    def add_concept(self, concept: ConceptNode, source_chunk: Optional[TextChunk] = None) -> str:
        """Add or update a concept node, with optional provenance link to chapter."""
        node_id = f"concept:{concept.name}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                type="Concept",
                name=concept.name,
                category=concept.category,
                description=concept.description,
                aliases=concept.aliases,
                occurrences=1,
            )
        else:
            self.graph.nodes[node_id]["occurrences"] += 1
            # Merge aliases
            existing_aliases: List[str] = self.graph.nodes[node_id].get("aliases", [])
            for a in concept.aliases:
                if a not in existing_aliases:
                    existing_aliases.append(a)
            self.graph.nodes[node_id]["aliases"] = existing_aliases

        if source_chunk:
            self.add_book_and_chunk(source_chunk)
            chap_node_id = f"chapter:{source_chunk.book_id}:{source_chunk.chapter_sequence}"
            # Edge: Chapter -> MENTIONS -> Concept
            self.graph.add_edge(
                chap_node_id,
                node_id,
                relation="MENTIONS",
                chunk_id=source_chunk.chunk_id,
            )

        return node_id

    def add_relationship(
        self,
        edge: RelationshipEdge,
        source_chunk: Optional[TextChunk] = None,
    ) -> None:
        """Connect two concept nodes with a semantic relation."""
        src_id = f"concept:{edge.source}"
        tgt_id = f"concept:{edge.target}"

        # Ensure both nodes exist (at least as placeholder concepts)
        if not self.graph.has_node(src_id):
            self.graph.add_node(
                src_id,
                type="Concept",
                name=edge.source,
                category="Concept",
                description="",
                aliases=[],
                occurrences=1,
            )
        if not self.graph.has_node(tgt_id):
            self.graph.add_node(
                tgt_id,
                type="Concept",
                name=edge.target,
                category="Concept",
                description="",
                aliases=[],
                occurrences=1,
            )

        self.graph.add_edge(
            src_id,
            tgt_id,
            relation=edge.relation_type,
            explanation=edge.explanation,
            chunk_id=source_chunk.chunk_id if source_chunk else None,
        )

    def stats(self) -> Dict[str, Any]:
        """Summary statistics of the graph."""
        type_counts: Dict[str, int] = {}
        for _, attrs in self.graph.nodes(data=True):
            ntype = attrs.get("type", "Unknown")
            type_counts[ntype] = type_counts.get(ntype, 0) + 1

        rel_counts: Dict[str, int] = {}
        for _, _, attrs in self.graph.edges(data=True):
            rel = attrs.get("relation", "Unknown")
            rel_counts[rel] = rel_counts.get(rel, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": type_counts,
            "edge_types": rel_counts,
        }

    def save(self, file_path: Optional[Path | str] = None) -> Path:
        """Persist graph to JSON node-link format."""
        path = Path(file_path or self.settings.storage_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = json_graph.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved knowledge graph with {len(self.graph)} nodes to {path}")
        return path

    def load(self, file_path: Optional[Path | str] = None) -> None:
        """Load graph from JSON node-link format."""
        path = Path(file_path or self.settings.storage_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Knowledge graph file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph = json_graph.node_link_graph(data, directed=True, multigraph=True)
        logger.info(f"Loaded knowledge graph with {len(self.graph)} nodes from {path}")
