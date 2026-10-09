"""
Directed Concept Knowledge Graph store implemented with NetworkX DiGraph.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from networkx.readwrite import json_graph

from bookeeper.calibre.parser import BookParser
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
        tags: Optional[List[str]] = None,
    ) -> str:
        """Add or update a Book node enriched with research literature note tags and metadata."""
        title = BookParser.repair_mojibake(title)
        author = BookParser.repair_mojibake(author) if author else "Unknown"
        summary = BookParser.repair_mojibake(summary) if summary else ""

        node_id = f"book:{book_id}"
        book_tags = ["book", "literature-note", "research-source"]
        if tags:
            for t in tags:
                if t not in book_tags:
                    book_tags.append(t)

        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                type="Book",
                tag="book",
                tags=book_tags,
                book_id=book_id,
                title=title,
                author=author,
                summary=summary,
            )
        else:
            nd = self.graph.nodes[node_id]
            nd["title"] = title
            if author and author != "Unknown":
                nd["author"] = author
            if summary and len(summary) > len(nd.get("summary", "")):
                nd["summary"] = summary
            if "tags" not in nd:
                nd["tags"] = book_tags
        return node_id

    def add_book_idea_link(
        self,
        book_id: int,
        concept_name: str,
        occurrences: int = 1,
    ) -> None:
        """
        Connect (:Book) -[:EXPLORES_IDEA]-> (:Concept/Idea)
        and (:Concept/Idea) -[:FEATURED_IN]-> (:Book)
        to anchor ideas to books for navigation and research synthesis.
        """
        concept_name = BookParser.repair_mojibake(concept_name)
        book_node_id = f"book:{book_id}"
        concept_node_id = f"concept:{concept_name}"

        if not self.graph.has_node(book_node_id):
            self.add_book(book_id, title=f"Book {book_id}")

        if not self.graph.has_node(concept_node_id):
            self.graph.add_node(
                concept_node_id,
                type="Concept",
                tag="idea",
                tags=["idea"],
                is_idea=True,
                name=concept_name,
                category="Idea",
                brief_description="",
                detailed_explanation="",
                summary="",
                weight=5,
                occurrences=1,
            )

        if not self.graph.has_edge(book_node_id, concept_node_id):
            self.graph.add_edge(
                book_node_id,
                concept_node_id,
                relation="EXPLORES_IDEA",
                occurrences=occurrences,
            )
        else:
            self.graph[book_node_id][concept_node_id]["occurrences"] += occurrences

        if not self.graph.has_edge(concept_node_id, book_node_id):
            self.graph.add_edge(
                concept_node_id,
                book_node_id,
                relation="FEATURED_IN",
            )

    def add_section(
        self,
        book_id: int,
        chapter_idx: int,
        title: str,
        text: str = "",
    ) -> str:
        """Add a Section node and connect (:Book) -[:HAS_SECTION]-> (:Section)."""
        title = BookParser.repair_mojibake(title)
        text = BookParser.repair_mojibake(text) if text else ""

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
        """Add or update an Idea / Concept node with brief description, detailed explanation, and significance weight."""
        name = BookParser.repair_mojibake(concept.name)
        category = BookParser.repair_mojibake(concept.category)
        node_id = f"concept:{name}"
        brief = BookParser.repair_mojibake(getattr(concept, "brief_description", "") or concept.summary)
        detailed = BookParser.repair_mojibake(getattr(concept, "detailed_explanation", "") or concept.summary)
        weight = getattr(concept, "weight", 5)

        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                type="Concept",
                tag="idea",
                tags=["idea", category.lower().replace(" ", "-")],
                is_idea=True,
                name=name,
                category=category,
                brief_description=brief,
                detailed_explanation=detailed,
                summary=brief or detailed,
                weight=weight,
                occurrences=1,
            )
        else:
            node_data = self.graph.nodes[node_id]
            node_data["occurrences"] += 1
            node_data["is_idea"] = True
            node_data["tag"] = "idea"
            # Update weight to the highest observed significance
            node_data["weight"] = max(node_data.get("weight", 0), weight)
            if brief and len(brief) > len(node_data.get("brief_description", "")):
                node_data["brief_description"] = brief
            if detailed and len(detailed) > len(node_data.get("detailed_explanation", "")):
                node_data["detailed_explanation"] = detailed
            summary_cand = brief or detailed
            if len(summary_cand) > len(node_data.get("summary", "")):
                node_data["summary"] = summary_cand

        return node_id

    def add_chunk(self, chunk: Any) -> str:
        """Add a Chunk node and connect (:Section) -[:HAS_CHUNK]-> (:Chunk)."""
        chunk_node_id = f"chunk:{chunk.chunk_id}"
        section_node_id = f"section:{chunk.book_id}:{chunk.chapter_idx}"

        # Ensure Section exists in graph
        if not self.graph.has_node(section_node_id):
            self.add_section(
                book_id=chunk.book_id,
                chapter_idx=chunk.chapter_idx,
                title=chunk.section_title,
            )

        self.graph.add_node(
            chunk_node_id,
            type="Chunk",
            chunk_id=chunk.chunk_id,
            book_id=chunk.book_id,
            book_title=chunk.book_title,
            chapter_idx=chunk.chapter_idx,
            chapter_title=getattr(chunk, "chapter_title", chunk.section_title),
            section_title=chunk.section_title,
            subtitle=getattr(chunk, "subtitle", chunk.section_title),
            breadcrumb=chunk.breadcrumb,
            text=chunk.text,
            char_count=chunk.char_count,
            parent_chunk_id=getattr(chunk, "parent_chunk_id", None),
            hierarchy_level=getattr(chunk, "hierarchy_level", "child"),
        )

        # Edge: (:Section) -[:HAS_CHUNK]-> (:Chunk)
        self.graph.add_edge(
            section_node_id,
            chunk_node_id,
            relation="HAS_CHUNK",
        )
        return chunk_node_id

    def add_idea_support_link(
        self,
        concept_name: str,
        chunk: Any,
        quote: str = "",
        brief_description: str = "",
        detailed_explanation: str = "",
    ) -> None:
        """
        Link an Idea node directly to the supporting Chunk, Chapter, and Subtitle:
          (:Concept/Idea) -[:SUPPORTED_BY {quote, breadcrumb, chapter_title, subtitle, chunk_id}]-> (:Chunk)
          (:Chunk) -[:SUPPORTS_IDEA]-> (:Concept/Idea)
        """
        concept_name = BookParser.repair_mojibake(concept_name)
        quote = BookParser.repair_mojibake(quote) if quote else ""
        brief_description = BookParser.repair_mojibake(brief_description) if brief_description else ""
        detailed_explanation = BookParser.repair_mojibake(detailed_explanation) if detailed_explanation else ""

        concept_node_id = f"concept:{concept_name}"
        chunk_node_id = self.add_chunk(chunk)

        if not self.graph.has_node(concept_node_id):
            self.graph.add_node(
                concept_node_id,
                type="Concept",
                tag="idea",
                tags=["idea"],
                is_idea=True,
                name=concept_name,
                category="Idea",
                brief_description=brief_description,
                detailed_explanation=detailed_explanation,
                summary=brief_description or detailed_explanation,
                weight=5,
                occurrences=1,
            )

        # Edge: (:Concept/Idea) -[:SUPPORTED_BY]-> (:Chunk)
        self.graph.add_edge(
            concept_node_id,
            chunk_node_id,
            relation="SUPPORTED_BY",
            quote=quote,
            brief_description=brief_description,
            detailed_explanation=detailed_explanation,
            breadcrumb=chunk.breadcrumb,
            chapter_title=getattr(chunk, "chapter_title", chunk.section_title),
            subtitle=getattr(chunk, "subtitle", chunk.section_title),
            chunk_id=chunk.chunk_id,
        )

        # Edge: (:Chunk) -[:SUPPORTS_IDEA]-> (:Concept/Idea)
        self.graph.add_edge(
            chunk_node_id,
            concept_node_id,
            relation="SUPPORTS_IDEA",
            quote=quote,
            brief_description=brief_description,
            detailed_explanation=detailed_explanation,
        )

    def add_section_concept_link(
        self,
        section_node_id: str,
        concept_name: str,
        summary: str = "",
        quote: str = "",
        weight: int = 5,
    ) -> None:
        """Connect (:Section) -[:DISCUSSES {summary, quote}]-> (:Concept)."""
        concept_name = BookParser.repair_mojibake(concept_name)
        summary = BookParser.repair_mojibake(summary) if summary else ""
        quote = BookParser.repair_mojibake(quote) if quote else ""

        concept_node_id = f"concept:{concept_name}"
        if not self.graph.has_node(concept_node_id):
            self.graph.add_node(
                concept_node_id,
                type="Concept",
                tag="idea",
                tags=["idea"],
                is_idea=True,
                name=concept_name,
                category="Idea",
                brief_description=summary,
                detailed_explanation=summary,
                summary=summary,
                weight=weight,
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
        src_concept_name = BookParser.repair_mojibake(src_concept_name)
        tgt_concept_name = BookParser.repair_mojibake(tgt_concept_name)

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
                    weight=5,
                    occurrences=1,
                )

        self.graph.add_edge(
            src_id,
            tgt_id,
            relation=relation_type,
        )

    def remove_book(self, book_id: int) -> Dict[str, int]:
        """
        Atomically remove a Book and all its associated sections, chunks, and orphan concepts from the graph.
        Returns a dict summarizing the count of removed nodes.
        """
        book_node_id = f"book:{book_id}"
        removed_nodes: Set[str] = set()

        # 1. Identify all chunks belonging to this book
        chunk_nodes = [
            n for n, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "Chunk" and attrs.get("book_id") == book_id
        ]
        removed_nodes.update(chunk_nodes)

        # 2. Identify all sections belonging to this book
        section_nodes = [
            n for n, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "Section" and (
                attrs.get("book_id") == book_id or n.startswith(f"section:{book_id}:")
            )
        ]
        removed_nodes.update(section_nodes)

        # 3. Add the book node itself if present
        if self.graph.has_node(book_node_id):
            removed_nodes.add(book_node_id)

        # 4. Check for affected concepts (concepts connected to removed books/sections/chunks)
        affected_concepts = set()
        for node in removed_nodes:
            for neighbor in self.graph.neighbors(node):
                if self.graph.nodes[neighbor].get("type") in ("Concept", "Idea"):
                    affected_concepts.add(neighbor)
            for predecessor in self.graph.predecessors(node):
                if self.graph.nodes[predecessor].get("type") in ("Concept", "Idea"):
                    affected_concepts.add(predecessor)

        # Remove the identified book, section, and chunk nodes
        self.graph.remove_nodes_from(removed_nodes)

        # 5. For affected concepts, check if they have any remaining connections to other books or chunks
        orphan_concepts = []
        for c_node in affected_concepts:
            if not self.graph.has_node(c_node):
                continue
            has_other_connections = False
            for neighbor in list(self.graph.neighbors(c_node)) + list(self.graph.predecessors(c_node)):
                nb_type = self.graph.nodes[neighbor].get("type")
                if nb_type in ("Book", "Section", "Chunk"):
                    has_other_connections = True
                    break
            if not has_other_connections:
                orphan_concepts.append(c_node)

        if orphan_concepts:
            self.graph.remove_nodes_from(orphan_concepts)

        stats = {
            "book_removed": 1 if book_node_id in removed_nodes else 0,
            "sections_removed": len(section_nodes),
            "chunks_removed": len(chunk_nodes),
            "orphan_concepts_removed": len(orphan_concepts),
            "total_nodes_removed": len(removed_nodes) + len(orphan_concepts),
        }
        logger.info(f"Removed book #{book_id} from graph: {stats}")
        return stats

    def stats(self) -> Dict[str, Any]:
        """Return counts of node types, edge relations, and concept weight statistics."""
        node_types: Dict[str, int] = {}
        concept_weights: List[int] = []
        for _, attrs in self.graph.nodes(data=True):
            ntype = attrs.get("type", "Unknown")
            node_types[ntype] = node_types.get(ntype, 0) + 1
            if ntype in ["Concept", "Idea"] and "weight" in attrs:
                concept_weights.append(attrs["weight"])

        edge_types: Dict[str, int] = {}
        for _, _, attrs in self.graph.edges(data=True):
            rel = attrs.get("relation", "Unknown")
            edge_types[rel] = edge_types.get(rel, 0) + 1

        avg_weight = round(sum(concept_weights) / len(concept_weights), 1) if concept_weights else 0.0

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "average_concept_weight": avg_weight,
            "concept_weight_count": len(concept_weights),
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
        """Load graph from JSON node-link format and repair any mojibake in nodes/edges."""
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Knowledge graph file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph = json_graph.node_link_graph(data, directed=True, multigraph=False)

        # 1. Sanitize all node attributes
        for _, attrs in self.graph.nodes(data=True):
            for k, v in list(attrs.items()):
                if isinstance(v, str):
                    attrs[k] = BookParser.repair_mojibake(v)
                elif isinstance(v, list):
                    attrs[k] = [BookParser.repair_mojibake(x) if isinstance(x, str) else x for x in v]

        # 2. Sanitize all edge attributes
        for _, _, attrs in self.graph.edges(data=True):
            for k, val in list(attrs.items()):
                if isinstance(val, str):
                    attrs[k] = BookParser.repair_mojibake(val)
                elif isinstance(val, list):
                    attrs[k] = [BookParser.repair_mojibake(x) if isinstance(x, str) else x for x in val]

        # 3. Relabel any node keys whose identifiers had mojibake
        mapping = {}
        for nid in list(self.graph.nodes()):
            clean_nid = BookParser.repair_mojibake(nid)
            if clean_nid != nid:
                mapping[nid] = clean_nid
        if mapping:
            nx.relabel_nodes(self.graph, mapping, copy=False)

        logger.info(f"Loaded graph with {len(self.graph)} nodes from {path}")
