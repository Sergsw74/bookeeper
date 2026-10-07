"""
Directed Concept Knowledge Graph store implemented with NetworkX DiGraph.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from networkx.readwrite import json_graph

from bookeeper.processing.extractor import Concept

logger = logging.getLogger(__name__)


class ConceptGraphStore:
    """
    Manages a directed graph (networkx.DiGraph) modeling:
      (:Book) -[:HAS_SECTION]-> (:Section)
      (:Section) -[:DISCUSSES {summary, quote}]-> (:Concept)
      (:Concept) -[:RELATES_TO]-> (:Concept)
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph: nx.DiGraph = graph if graph is not None else nx.DiGraph()

    def add_book(
        self,
        book_id: int,
        title: str,
        author: str = "Unknown",
        summary: str = "",
    ) -> str:
        """Add or update a Book node."""
        node_id = f"book:{book_id}"
        self.graph.add_node(
            node_id,
            type="Book",
            book_id=book_id,
            title=title,
            author=author,
            summary=summary,
        )
        return node_id

    def add_section(
        self,
        book_id: int,
        chapter_idx: int,
        title: str,
        text: str = "",
    ) -> str:
        """Add a Section node and connect (:Book) -[:HAS_SECTION]-> (:Section)."""
        book_node_id = f"book:{book_id}"
        if not self.graph.has_node(book_node_id):
            self.add_book(book_id, title=f"Book {book_id}")

        section_node_id = f"section:{book_id}:{chapter_idx}"
        self.graph.add_node(
            section_node_id,
            type="Section",
            book_id=book_id,
            chapter_idx=chapter_idx,
            title=title,
            char_count=len(text),
        )

        # Edge: (:Book) -[:HAS_SECTION]-> (:Section)
        self.graph.add_edge(
            book_node_id,
            section_node_id,
            relation="HAS_SECTION",
        )
        return section_node_id

    def add_concept(self, concept: Concept) -> str:
        """Add or update a canonical Concept node."""
        node_id = f"concept:{concept.name}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                type="Concept",
                name=concept.name,
                category=concept.category,
                summary=concept.summary,
                occurrences=1,
            )
        else:
            self.graph.nodes[node_id]["occurrences"] += 1
            if len(concept.summary) > len(self.graph.nodes[node_id].get("summary", "")):
                self.graph.nodes[node_id]["summary"] = concept.summary

        return node_id

    def add_section_concept_link(
        self,
        section_node_id: str,
        concept_name: str,
        summary: str = "",
        quote: str = "",
    ) -> None:
        """Connect (:Section) -[:DISCUSSES {summary, quote}]-> (:Concept)."""
        concept_node_id = f"concept:{concept_name}"
        if not self.graph.has_node(concept_node_id):
            self.graph.add_node(
                concept_node_id,
                type="Concept",
                name=concept_name,
                category="Concept",
                summary=summary,
                occurrences=1,
            )

        self.graph.add_edge(
            section_node_id,
            concept_node_id,
            relation="DISCUSSES",
            summary=summary,
            quote=quote,
        )

    def add_concept_relation(
        self,
        src_concept_name: str,
        tgt_concept_name: str,
        relation_type: str = "RELATES_TO",
    ) -> None:
        """Connect (:Concept) -[:RELATES_TO]-> (:Concept)."""
        if src_concept_name == tgt_concept_name:
            return

        src_id = f"concept:{src_concept_name}"
        tgt_id = f"concept:{tgt_concept_name}"

        for nid, name in [(src_id, src_concept_name), (tgt_id, tgt_concept_name)]:
            if not self.graph.has_node(nid):
                self.graph.add_node(
                    nid,
                    type="Concept",
                    name=name,
                    category="Concept",
                    summary="",
                    occurrences=1,
                )

        self.graph.add_edge(
            src_id,
            tgt_id,
            relation=relation_type,
        )

    def stats(self) -> Dict[str, Any]:
        """Return counts of node types and edge relations."""
        node_types: Dict[str, int] = {}
        for _, attrs in self.graph.nodes(data=True):
            ntype = attrs.get("type", "Unknown")
            node_types[ntype] = node_types.get(ntype, 0) + 1

        edge_types: Dict[str, int] = {}
        for _, _, attrs in self.graph.edges(data=True):
            rel = attrs.get("relation", "Unknown")
            edge_types[rel] = edge_types.get(rel, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
        }

    def save(self, file_path: Path | str) -> Path:
        """Persist graph to JSON node-link format."""
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = json_graph.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved graph with {len(self.graph)} nodes to {path}")
        return path

    def load(self, file_path: Path | str) -> None:
        """Load graph from JSON node-link format."""
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Knowledge graph file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph = json_graph.node_link_graph(data, directed=True, multigraph=False)
        logger.info(f"Loaded graph with {len(self.graph)} nodes from {path}")
