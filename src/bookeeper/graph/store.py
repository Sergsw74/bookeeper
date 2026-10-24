"""
Directed Concept Knowledge Graph store implemented with NetworkX DiGraph.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from networkx.readwrite import json_graph

from bookeeper.calibre.parser import BookParser, format_idea_month
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
        self.dirty_nodes: Set[str] = set()
        self.dirty_edges: Set[tuple[str, str]] = set()
        self.removed_nodes: Set[str] = set()

    def mark_node_dirty(self, node_id: str) -> None:
        """Record a node as added/modified since the last incremental export."""
        self.dirty_nodes.add(node_id)
        self.removed_nodes.discard(node_id)

    def mark_edge_dirty(self, src_id: str, tgt_id: str) -> None:
        """Record a directed edge as added/modified since the last incremental export."""
        self.dirty_edges.add((src_id, tgt_id))
        self.dirty_nodes.add(src_id)
        self.dirty_nodes.add(tgt_id)

    def mark_node_removed(self, node_id: str) -> None:
        """Record a node as deleted/merged so incremental export removes it from Neo4j."""
        self.dirty_nodes.discard(node_id)
        self.removed_nodes.add(node_id)
        self.dirty_edges = {
            (u, v) for (u, v) in self.dirty_edges if u != node_id and v != node_id
        }

    def has_dirty_changes(self) -> bool:
        """Return True if any nodes or edges were modified or removed since the last incremental export."""
        return bool(self.dirty_nodes or self.dirty_edges or self.removed_nodes)

    def clear_dirty(self) -> None:
        """Clear dirty node/edge tracking sets after a successful incremental export."""
        self.dirty_nodes.clear()
        self.dirty_edges.clear()
        self.removed_nodes.clear()

    def add_book(
        self,
        book_id: int,
        title: str,
        author: str = "Unknown",
        summary: str = "",
        tags: Optional[List[str]] = None,
        publication_date: Optional[str] = None,
    ) -> str:
        """Add or update a Book node enriched with research literature note tags and metadata."""
        title = BookParser.repair_mojibake(title)
        author = BookParser.repair_mojibake(author) if author else "Unknown"
        summary = BookParser.repair_mojibake(summary) if summary else ""
        pub_date = BookParser.parse_publication_date(publication_date)

        node_id = f"book:{book_id}"
        book_tags = ["book", "literature-note", "research-source"]
        if tags:
            for t in tags:
                if t not in book_tags:
                    book_tags.append(t)

        if not self.graph.has_node(node_id):
            node_attrs = {
                "type": "Book",
                "tag": "book",
                "tags": book_tags,
                "book_id": book_id,
                "title": title,
                "author": author,
                "summary": summary,
            }
            if pub_date:
                node_attrs["publication_date"] = pub_date
            self.graph.add_node(node_id, **node_attrs)
        else:
            nd = self.graph.nodes[node_id]
            nd["title"] = title
            if author and author != "Unknown":
                nd["author"] = author
            if summary and len(summary) > len(nd.get("summary", "")):
                nd["summary"] = summary
            if pub_date:
                nd["publication_date"] = pub_date
            if "tags" not in nd:
                nd["tags"] = book_tags
        self.mark_node_dirty(node_id)
        return node_id

    def update_idea_occurrence_month(
        self,
        concept_name: str,
        date_val: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record or update the first occurrence month of an idea.
        If the idea already has an occurrence month, the EARLIER date wins!
        Returns the effective first_occurrence_month (YYYY-MM).
        """
        concept_name = BookParser.repair_mojibake(concept_name)
        concept_node_id = f"concept:{concept_name}"
        month = format_idea_month(date_val)
        if not month:
            return None

        if self.graph.has_node(concept_node_id):
            nd = self.graph.nodes[concept_node_id]
            curr_month = nd.get("first_occurrence_month") or nd.get("month")
            if not curr_month or month < curr_month:
                nd["first_occurrence_month"] = month
                nd["month"] = month
                self.mark_node_dirty(concept_node_id)
                return month
            return curr_month
        return None

    def add_book_idea_link(
        self,
        book_id: int,
        concept_name: str,
        occurrences: int = 1,
        publication_date: Optional[str] = None,
    ) -> None:
        """
        Connect (:Book) -[:EXPLORES_IDEA]-> (:Concept/Idea)
        and (:Concept/Idea) -[:FEATURED_IN]-> (:Book)
        to anchor ideas to books for navigation and research synthesis.
        Tracks first_occurrence_month where the earlier date wins.
        """
        concept_name = BookParser.repair_mojibake(concept_name)
        book_node_id = f"book:{book_id}"
        concept_node_id = f"concept:{concept_name}"

        if not self.graph.has_node(book_node_id):
            self.add_book(book_id, title=f"Book {book_id}")

        pub_date = publication_date
        if not pub_date and self.graph.has_node(book_node_id):
            pub_date = self.graph.nodes[book_node_id].get("publication_date")
        month = format_idea_month(pub_date)

        if not self.graph.has_node(concept_node_id):
            node_attrs = {
                "type": "Concept",
                "tag": "idea",
                "tags": ["idea"],
                "is_idea": True,
                "name": concept_name,
                "category": "Idea",
                "brief_description": "",
                "detailed_explanation": "",
                "summary": "",
                "weight": 5,
                "occurrences": 1,
            }
            if month:
                node_attrs["first_occurrence_month"] = month
                node_attrs["month"] = month
            self.graph.add_node(concept_node_id, **node_attrs)
        elif month:
            self.update_idea_occurrence_month(concept_name, month)

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
        self.mark_edge_dirty(book_node_id, concept_node_id)
        self.mark_edge_dirty(concept_node_id, book_node_id)

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
        self.mark_edge_dirty(book_node_id, section_node_id)
        return section_node_id

    def add_concept(self, concept: Concept, embedding: Optional[List[float]] = None) -> str:
        """Add or update an Idea / Concept node with brief description, detailed explanation, significance weight, and optional embedding."""
        name = BookParser.repair_mojibake(concept.name)
        category = BookParser.repair_mojibake(concept.category)
        node_id = f"concept:{name}"
        brief = BookParser.repair_mojibake(getattr(concept, "brief_description", "") or concept.summary)
        detailed = BookParser.repair_mojibake(getattr(concept, "detailed_explanation", "") or concept.summary)
        weight = getattr(concept, "weight", 5)
        emb = embedding if embedding is not None else getattr(concept, "embedding", None)

        if not self.graph.has_node(node_id):
            node_attrs: Dict[str, Any] = {
                "type": "Concept",
                "tag": "idea",
                "tags": ["idea", category.lower().replace(" ", "-")],
                "is_idea": True,
                "name": name,
                "category": category,
                "brief_description": brief,
                "detailed_explanation=detailed" if False else "detailed_explanation": detailed,
                "summary": brief or detailed,
                "weight": weight,
                "occurrences": 1,
            }
            if emb:
                node_attrs["embedding"] = [round(float(x), 6) for x in emb]
            self.graph.add_node(node_id, **node_attrs)
        else:
            node_data = self.graph.nodes[node_id]
            node_data["occurrences"] += 1
            node_data["is_idea"] = True
            node_data["tag"] = "idea"
            # Update weight to the highest observed significance
            node_data["weight"] = max(node_data.get("weight", 0), weight)
            if not node_data.get("refined"):
                if brief and len(brief) > len(node_data.get("brief_description", "")):
                    node_data["brief_description"] = brief
                if detailed and len(detailed) > len(node_data.get("detailed_explanation", "")):
                    node_data["detailed_explanation"] = detailed
                summary_cand = brief or detailed
                if len(summary_cand) > len(node_data.get("summary", "")):
                    node_data["summary"] = summary_cand
            if emb and not node_data.get("embedding"):
                node_data["embedding"] = [round(float(x), 6) for x in emb]

        self.mark_node_dirty(node_id)
        return node_id

    def add_chunk(self, chunk: Any) -> str:
        """Add a Chunk node (preserving its embedding) and connect (:Section) -[:HAS_CHUNK]-> (:Chunk)."""
        chunk_node_id = f"chunk:{chunk.chunk_id}"
        section_node_id = f"section:{chunk.book_id}:{chunk.chapter_idx}"

        # Ensure Section exists in graph
        if not self.graph.has_node(section_node_id):
            self.add_section(
                book_id=chunk.book_id,
                chapter_idx=chunk.chapter_idx,
                title=chunk.section_title,
            )

        chunk_idx = getattr(chunk, "chunk_idx", 0)
        prev_chunk_id = getattr(chunk, "prev_chunk_id", None)
        next_chunk_id = getattr(chunk, "next_chunk_id", None)
        emb = getattr(chunk, "embedding", None)
        existing_emb = (
            self.graph.nodes[chunk_node_id].get("embedding")
            if self.graph.has_node(chunk_node_id)
            else None
        )

        chunk_attrs: Dict[str, Any] = {
            "type": "Chunk",
            "chunk_id": chunk.chunk_id,
            "book_id": chunk.book_id,
            "book_title": chunk.book_title,
            "chapter_idx": chunk.chapter_idx,
            "chapter_title": getattr(chunk, "chapter_title", chunk.section_title),
            "section_title": chunk.section_title,
            "subtitle": getattr(chunk, "subtitle", chunk.section_title),
            "breadcrumb": chunk.breadcrumb,
            "text": chunk.text,
            "char_count": chunk.char_count,
            "parent_chunk_id": getattr(chunk, "parent_chunk_id", None),
            "hierarchy_level": getattr(chunk, "hierarchy_level", "child"),
            "chunk_idx": chunk_idx,
            "prev_chunk_id": prev_chunk_id,
            "next_chunk_id": next_chunk_id,
        }
        if emb and len(emb) > 1:
            chunk_attrs["embedding"] = [round(float(x), 6) for x in emb]
        elif existing_emb and len(existing_emb) > 1:
            chunk_attrs["embedding"] = existing_emb

        self.graph.add_node(chunk_node_id, **chunk_attrs)

        # Edge: (:Section) -[:HAS_CHUNK]-> (:Chunk)
        self.graph.add_edge(
            section_node_id,
            chunk_node_id,
            relation="HAS_CHUNK",
        )
        self.mark_edge_dirty(section_node_id, chunk_node_id)

        # Sequential edges between chunks: (:Chunk A) -[:NEXT]-> (:Chunk B), (:Chunk B) -[:PREV]-> (:Chunk A)
        if prev_chunk_id:
            p_nid = f"chunk:{prev_chunk_id}" if not str(prev_chunk_id).startswith("chunk:") else str(prev_chunk_id)
            if self.graph.has_node(p_nid):
                self.graph.add_edge(p_nid, chunk_node_id, relation="NEXT")
                self.graph.add_edge(chunk_node_id, p_nid, relation="PREV")
                self.mark_edge_dirty(p_nid, chunk_node_id)
                self.mark_edge_dirty(chunk_node_id, p_nid)

        if next_chunk_id:
            n_nid = f"chunk:{next_chunk_id}" if not str(next_chunk_id).startswith("chunk:") else str(next_chunk_id)
            if self.graph.has_node(n_nid):
                self.graph.add_edge(chunk_node_id, n_nid, relation="NEXT")
                self.graph.add_edge(n_nid, chunk_node_id, relation="PREV")
                self.mark_edge_dirty(chunk_node_id, n_nid)
                self.mark_edge_dirty(n_nid, chunk_node_id)

        return chunk_node_id

    def link_sequential_chunks(self, book_id: Optional[int] = None) -> int:
        """
        Link consecutive Chunk nodes within each book with :NEXT and :PREV edges:
          (:Chunk A) -[:NEXT]-> (:Chunk B)
          (:Chunk B) -[:PREV]-> (:Chunk A)

        Uses explicit prev_chunk_id / next_chunk_id properties or orders chunks
        by (chapter_idx, chunk_idx / global_chunk_idx).
        Returns the number of sequential chunk pairs linked.
        """
        chunks_by_book: Dict[int, List[tuple[str, Dict[str, Any]]]] = {}
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "Chunk":
                b_id = attrs.get("book_id", 0)
                if book_id is not None and b_id != book_id:
                    continue
                chunks_by_book.setdefault(b_id, []).append((node_id, attrs))

        def get_chunk_sort_key(item: tuple[str, Dict[str, Any]]) -> tuple[int, int, str]:
            node_id, attrs = item
            ch_idx = int(attrs.get("chapter_idx") or 0)
            c_idx = int(attrs.get("chunk_idx") or 0)
            if c_idx > 0:
                return (ch_idx, c_idx, node_id)
            cid = str(attrs.get("chunk_id", node_id))
            m_p = re.search(r"_p(\d+)_", cid)
            if m_p:
                return (ch_idx, int(m_p.group(1)), node_id)
            bc = str(attrs.get("breadcrumb", ""))
            m_bc = re.search(r"\[part\s+(\d+)\.(\d+)\]", bc)
            if m_bc:
                return (int(m_bc.group(1)), int(m_bc.group(2)), node_id)
            return (ch_idx, 0, node_id)

        linked_pairs = 0
        for b_id, chunk_list in chunks_by_book.items():
            if len(chunk_list) < 2:
                continue

            sorted_chunks = sorted(chunk_list, key=get_chunk_sort_key)
            for i in range(len(sorted_chunks) - 1):
                curr_id, curr_attrs = sorted_chunks[i]
                next_id, next_attrs = sorted_chunks[i + 1]

                # Ensure edges exist
                self.graph.add_edge(curr_id, next_id, relation="NEXT")
                self.graph.add_edge(next_id, curr_id, relation="PREV")
                if book_id is not None:
                    self.mark_edge_dirty(curr_id, next_id)
                    self.mark_edge_dirty(next_id, curr_id)

                curr_cid = curr_attrs.get("chunk_id") or curr_id.replace("chunk:", "")
                next_cid = next_attrs.get("chunk_id") or next_id.replace("chunk:", "")
                curr_attrs.setdefault("next_chunk_id", next_cid)
                next_attrs.setdefault("prev_chunk_id", curr_cid)
                if not curr_attrs.get("chunk_idx"):
                    curr_attrs["chunk_idx"] = i + 1
                if not next_attrs.get("chunk_idx"):
                    next_attrs["chunk_idx"] = i + 2

                linked_pairs += 1

        # Also connect any explicit prev_chunk_id / next_chunk_id references if present
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "Chunk":
                b_id = attrs.get("book_id", 0)
                if book_id is not None and b_id != book_id:
                    continue
                prev_cid = attrs.get("prev_chunk_id")
                if prev_cid:
                    p_nid = f"chunk:{prev_cid}" if not str(prev_cid).startswith("chunk:") else str(prev_cid)
                    if self.graph.has_node(p_nid):
                        self.graph.add_edge(p_nid, node_id, relation="NEXT")
                        self.graph.add_edge(node_id, p_nid, relation="PREV")
                        if book_id is not None:
                            self.mark_edge_dirty(p_nid, node_id)
                            self.mark_edge_dirty(node_id, p_nid)
                next_cid = attrs.get("next_chunk_id")
                if next_cid:
                    n_nid = f"chunk:{next_cid}" if not str(next_cid).startswith("chunk:") else str(next_cid)
                    if self.graph.has_node(n_nid):
                        self.graph.add_edge(node_id, n_nid, relation="NEXT")
                        self.graph.add_edge(n_nid, node_id, relation="PREV")
                        if book_id is not None:
                            self.mark_edge_dirty(node_id, n_nid)
                            self.mark_edge_dirty(n_nid, node_id)

        return linked_pairs

    def add_idea_support_link(
        self,
        concept_name: str,
        chunk: Any,
        quote: str = "",
        brief_description: str = "",
        detailed_explanation: str = "",
        publication_date: Optional[str] = None,
    ) -> None:
        """
        Link an Idea node directly to the supporting Chunk, Chapter, and Subtitle:
          (:Concept/Idea) -[:SUPPORTED_BY {quote, breadcrumb, chapter_title, subtitle, chunk_id}]-> (:Chunk)
          (:Chunk) -[:SUPPORTS_IDEA]-> (:Concept/Idea)
        Tracks first_occurrence_month where the earlier date wins.
        """
        concept_name = BookParser.repair_mojibake(concept_name)
        quote = BookParser.repair_mojibake(quote) if quote else ""
        brief_description = BookParser.repair_mojibake(brief_description) if brief_description else ""
        detailed_explanation = BookParser.repair_mojibake(detailed_explanation) if detailed_explanation else ""

        concept_node_id = f"concept:{concept_name}"
        chunk_node_id = self.add_chunk(chunk)

        book_nid = f"book:{getattr(chunk, 'book_id', 0)}"
        pub_date = publication_date or getattr(chunk, "publication_date", None)
        if not pub_date and self.graph.has_node(book_nid):
            pub_date = self.graph.nodes[book_nid].get("publication_date")
        month = format_idea_month(pub_date)

        if not self.graph.has_node(concept_node_id):
            node_attrs = {
                "type": "Concept",
                "tag": "idea",
                "tags": ["idea"],
                "is_idea": True,
                "name": concept_name,
                "category": "Idea",
                "brief_description": brief_description,
                "detailed_explanation": detailed_explanation,
                "summary": brief_description or detailed_explanation,
                "weight": 5,
                "occurrences": 1,
            }
            if month:
                node_attrs["first_occurrence_month"] = month
                node_attrs["month"] = month
            self.graph.add_node(concept_node_id, **node_attrs)
        elif month:
            self.update_idea_occurrence_month(concept_name, month)

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
        self.mark_edge_dirty(concept_node_id, chunk_node_id)
        self.mark_edge_dirty(chunk_node_id, concept_node_id)

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
        self.mark_edge_dirty(section_node_id, concept_node_id)

    def add_concept_relation(
        self,
        src_concept_name: str,
        tgt_concept_name: str,
        relation_type: str = "RELATES_TO",
    ) -> bool:
        """
        Connect (:Concept) -[:RELATES_TO]-> (:Concept).
        Returns True if a new edge (or node) was created/updated, False if it already existed unchanged.
        """
        src_concept_name = BookParser.repair_mojibake(src_concept_name)
        tgt_concept_name = BookParser.repair_mojibake(tgt_concept_name)

        if src_concept_name == tgt_concept_name:
            return False

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
                self.mark_node_dirty(nid)

        if self.graph.has_edge(src_id, tgt_id) and self.graph[src_id][tgt_id].get("relation") == relation_type:
            return False

        self.graph.add_edge(
            src_id,
            tgt_id,
            relation=relation_type,
        )
        self.mark_edge_dirty(src_id, tgt_id)
        return True

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

        for rn in removed_nodes:
            self.mark_node_removed(rn)
        for oc in orphan_concepts:
            self.mark_node_removed(oc)

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

    def save(self, file_path: Path | str, link_chunks: bool = True) -> Path:
        """
        Persist graph to JSON node-link format.
        Uses compact JSON formatting when the graph contains vector embeddings or >100 nodes
        to avoid writing millions of indented float lines on every book checkpoint.
        """
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        if link_chunks:
            self.link_sequential_chunks()

        data = json_graph.node_link_data(self.graph)
        use_compact = self.graph.number_of_nodes() > 100 or any(
            "embedding" in attrs for _, attrs in self.graph.nodes(data=True)
        )
        with open(path, "w", encoding="utf-8") as f:
            if use_compact:
                json.dump(data, f, separators=(",", ":"))
            else:
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

    def backfill_missing_embeddings(
        self,
        embeddings: Any,
        include_ideas: bool = True,
        include_chunks: bool = True,
        batch_size: int = 64,
        console: Optional[Any] = None,
        chunks_dir: Optional[Path | str] = None,
    ) -> Dict[str, int]:
        """
        Backfill missing 'embedding' vectors on existing :Idea (:Concept) and :Chunk nodes
        in the knowledge graph using size-aware batching (and reusing cached chunk embeddings
        from chunks_dir when available), without needing to rebuild the knowledge graph from scratch.
        """
        import numpy as np
        from bookeeper.processing.ollama_pool import (
            FailoverOllamaEmbeddings,
            format_idea_for_embedding,
            partition_texts_for_embedding,
        )

        # First, if chunks_dir is provided, reuse any chunk embeddings already stored in book_{id}.json
        chunks_reused_from_disk = 0
        if include_chunks and chunks_dir is not None:
            cdir = Path(chunks_dir).expanduser().resolve()
            if cdir.is_dir():
                books_needing_chunks: Set[int] = set()
                for _, attrs in self.graph.nodes(data=True):
                    if attrs.get("type") == "Chunk" and not attrs.get("embedding"):
                        bid = attrs.get("book_id")
                        if bid is not None:
                            books_needing_chunks.add(int(bid))
                for bid in books_needing_chunks:
                    cfile = cdir / f"book_{bid}.json"
                    if cfile.is_file():
                        try:
                            with open(cfile, "r", encoding="utf-8") as f:
                                cdata = json.load(f)
                            for citem in cdata.get("chunks", []):
                                cid = citem.get("chunk_id")
                                cemb = citem.get("embedding")
                                cnid = f"chunk:{cid}"
                                if cid and cemb and len(cemb) > 1 and self.graph.has_node(cnid):
                                    self.graph.nodes[cnid]["embedding"] = [round(float(x), 6) for x in cemb]
                                    self.mark_node_dirty(cnid)
                                    chunks_reused_from_disk += 1
                        except Exception:
                            pass

        if embeddings is None:
            return {"ideas_embedded": 0, "chunks_embedded": chunks_reused_from_disk}

        missing_idea_nodes: List[tuple[str, str]] = []
        missing_chunk_nodes: List[tuple[str, str]] = []

        for node_id, attrs in self.graph.nodes(data=True):
            existing_emb = attrs.get("embedding")
            if isinstance(existing_emb, (list, tuple, np.ndarray)) and len(existing_emb) > 1:
                continue

            ntype = attrs.get("type", "")
            is_idea = bool(attrs.get("is_idea") or attrs.get("tag") == "idea" or ntype in ("Concept", "Idea"))
            if is_idea and include_ideas:
                raw_name = attrs.get("name") or str(node_id).replace("concept:", "")
                clean_name = BookParser.repair_mojibake(raw_name).strip()
                if not clean_name:
                    continue
                brief = attrs.get("brief_description") or attrs.get("summary") or ""
                detailed = attrs.get("detailed_explanation") or ""
                summary = attrs.get("summary") or ""
                text = format_idea_for_embedding(
                    clean_name,
                    brief_description=brief,
                    detailed_explanation=detailed,
                    summary=summary,
                )
                missing_idea_nodes.append((node_id, text))
            elif ntype == "Chunk" and include_chunks:
                chunk_text = (attrs.get("text") or "").strip()
                if chunk_text:
                    missing_chunk_nodes.append((node_id, chunk_text))

        ideas_embedded = 0
        chunks_embedded = chunks_reused_from_disk

        if missing_idea_nodes:
            idea_texts = [t for _, t in missing_idea_nodes]
            idea_batches = partition_texts_for_embedding(
                idea_texts,
                max_batch_chars=12000,
                max_batch_items=batch_size,
            )
            if console is not None:
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    SpinnerColumn,
                    TaskProgressColumn,
                    TextColumn,
                    TimeElapsedColumn,
                )
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]Backfilling Idea Embeddings:[/bold cyan]"),
                    BarColumn(bar_width=25),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as prog:
                    t_id = prog.add_task("ideas", total=len(missing_idea_nodes))
                    pooled_done = False
                    if isinstance(embeddings, FailoverOllamaEmbeddings):
                        try:
                            vecs = embeddings.embed_documents(
                                idea_texts,
                                batch_size=batch_size,
                                max_batch_chars=12000,
                                on_progress=lambda n: prog.update(t_id, advance=n),
                            )
                            for (nid, _), v in zip(missing_idea_nodes, vecs):
                                if v and len(v) > 1:
                                    arr = np.array(v, dtype=np.float32)
                                    norm = float(np.linalg.norm(arr))
                                    unit = (arr / norm) if norm > 0.0 else arr
                                    self.graph.nodes[nid]["embedding"] = [round(float(x), 6) for x in unit.tolist()]
                                    self.mark_node_dirty(nid)
                                    ideas_embedded += 1
                            pooled_done = True
                        except Exception as e:
                            logger.warning(f"Failed pooled backfill of idea embeddings, falling back to batches: {e}")
                    if not pooled_done:
                        offset = 0
                        for sub in idea_batches:
                            try:
                                vecs = (
                                    embeddings.embed_documents(sub)
                                    if hasattr(embeddings, "embed_documents")
                                    else [embeddings.embed_query(x) for x in sub]
                                )
                                for (nid, _), v in zip(missing_idea_nodes[offset : offset + len(sub)], vecs):
                                    if v and len(v) > 1:
                                        arr = np.array(v, dtype=np.float32)
                                        norm = float(np.linalg.norm(arr))
                                        unit = (arr / norm) if norm > 0.0 else arr
                                        self.graph.nodes[nid]["embedding"] = [round(float(x), 6) for x in unit.tolist()]
                                        self.mark_node_dirty(nid)
                                        ideas_embedded += 1
                            except Exception as e:
                                logger.warning(f"Failed backfilling idea embeddings batch at {offset}: {e}")
                            offset += len(sub)
                            prog.update(t_id, advance=len(sub))
            else:
                pooled_done = False
                if isinstance(embeddings, FailoverOllamaEmbeddings):
                    try:
                        vecs = embeddings.embed_documents(
                            idea_texts,
                            batch_size=batch_size,
                            max_batch_chars=12000,
                        )
                        for (nid, _), v in zip(missing_idea_nodes, vecs):
                            if v and len(v) > 1:
                                arr = np.array(v, dtype=np.float32)
                                norm = float(np.linalg.norm(arr))
                                unit = (arr / norm) if norm > 0.0 else arr
                                self.graph.nodes[nid]["embedding"] = [round(float(x), 6) for x in unit.tolist()]
                                self.mark_node_dirty(nid)
                                ideas_embedded += 1
                        pooled_done = True
                    except Exception as e:
                        logger.warning(f"Failed pooled backfill of idea embeddings, falling back to batches: {e}")
                if not pooled_done:
                    offset = 0
                    for sub in idea_batches:
                        try:
                            vecs = (
                                embeddings.embed_documents(sub)
                                if hasattr(embeddings, "embed_documents")
                                else [embeddings.embed_query(x) for x in sub]
                            )
                            for (nid, _), v in zip(missing_idea_nodes[offset : offset + len(sub)], vecs):
                                if v and len(v) > 1:
                                    arr = np.array(v, dtype=np.float32)
                                    norm = float(np.linalg.norm(arr))
                                    unit = (arr / norm) if norm > 0.0 else arr
                                    self.graph.nodes[nid]["embedding"] = [round(float(x), 6) for x in unit.tolist()]
                                    self.mark_node_dirty(nid)
                                    ideas_embedded += 1
                        except Exception as e:
                            logger.warning(f"Failed backfilling idea embeddings batch at {offset}: {e}")
                        offset += len(sub)

        if missing_chunk_nodes:
            chunk_texts = [t for _, t in missing_chunk_nodes]
            chunk_batches = partition_texts_for_embedding(
                chunk_texts,
                max_batch_chars=12000,
                max_batch_items=min(batch_size, 16),
            )
            if console is not None:
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    SpinnerColumn,
                    TaskProgressColumn,
                    TextColumn,
                    TimeElapsedColumn,
                )
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]Backfilling Chunk Embeddings:[/bold cyan]"),
                    BarColumn(bar_width=25),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as prog:
                    t_id = prog.add_task("chunks", total=len(missing_chunk_nodes))
                    pooled_done = False
                    if isinstance(embeddings, FailoverOllamaEmbeddings):
                        try:
                            vecs = embeddings.embed_documents(
                                chunk_texts,
                                batch_size=min(batch_size, 16),
                                max_batch_chars=12000,
                                on_progress=lambda n: prog.update(t_id, advance=n),
                            )
                            for (nid, _), v in zip(missing_chunk_nodes, vecs):
                                if v and len(v) > 1:
                                    arr = np.array(v, dtype=np.float32)
                                    norm = float(np.linalg.norm(arr))
                                    unit = (arr / norm) if norm > 0.0 else arr
                                    self.graph.nodes[nid]["embedding"] = [round(float(x), 6) for x in unit.tolist()]
                                    self.mark_node_dirty(nid)
                                    chunks_embedded += 1
                            pooled_done = True
                        except Exception as e:
                            logger.warning(f"Failed pooled backfill of chunk embeddings, falling back to batches: {e}")
                    if not pooled_done:
                        offset = 0
                        for sub in chunk_batches:
                            try:
                                vecs = (
                                    embeddings.embed_documents(sub)
                                    if hasattr(embeddings, "embed_documents")
                                    else [embeddings.embed_query(x) for x in sub]
                                )
                                for (nid, _), v in zip(missing_chunk_nodes[offset : offset + len(sub)], vecs):
                                    if v and len(v) > 1:
                                        arr = np.array(v, dtype=np.float32)
                                        norm = float(np.linalg.norm(arr))
                                        unit = (arr / norm) if norm > 0.0 else arr
                                        self.graph.nodes[nid]["embedding"] = [round(float(x), 6) for x in unit.tolist()]
                                        self.mark_node_dirty(nid)
                                        chunks_embedded += 1
                            except Exception as e:
                                logger.warning(f"Failed backfilling chunk embeddings batch at {offset}: {e}")
                            offset += len(sub)
                            prog.update(t_id, advance=len(sub))
            else:
                pooled_done = False
                if isinstance(embeddings, FailoverOllamaEmbeddings):
                    try:
                        vecs = embeddings.embed_documents(
                            chunk_texts,
                            batch_size=min(batch_size, 16),
                            max_batch_chars=12000,
                        )
                        for (nid, _), v in zip(missing_chunk_nodes, vecs):
                            if v and len(v) > 1:
                                arr = np.array(v, dtype=np.float32)
                                norm = float(np.linalg.norm(arr))
                                unit = (arr / norm) if norm > 0.0 else arr
                                self.graph.nodes[nid]["embedding"] = [round(float(x), 6) for x in unit.tolist()]
                                self.mark_node_dirty(nid)
                                chunks_embedded += 1
                        pooled_done = True
                    except Exception as e:
                        logger.warning(f"Failed pooled backfill of chunk embeddings, falling back to batches: {e}")
                if not pooled_done:
                    offset = 0
                    for sub in chunk_batches:
                        try:
                            vecs = (
                                embeddings.embed_documents(sub)
                                if hasattr(embeddings, "embed_documents")
                                else [embeddings.embed_query(x) for x in sub]
                            )
                            for (nid, _), v in zip(missing_chunk_nodes[offset : offset + len(sub)], vecs):
                                if v and len(v) > 1:
                                    arr = np.array(v, dtype=np.float32)
                                    norm = float(np.linalg.norm(arr))
                                    unit = (arr / norm) if norm > 0.0 else arr
                                    self.graph.nodes[nid]["embedding"] = [round(float(x), 6) for x in unit.tolist()]
                                    self.mark_node_dirty(nid)
                                    chunks_embedded += 1
                        except Exception as e:
                            logger.warning(f"Failed backfilling chunk embeddings batch at {offset}: {e}")
                        offset += len(sub)

        return {"ideas_embedded": ideas_embedded, "chunks_embedded": chunks_embedded}
