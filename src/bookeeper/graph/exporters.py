"""
Exporters for Concept Knowledge Graph: Obsidian Markdown Vault with wikilinks, GraphML, and GEXF.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from bookeeper.graph.store import ConceptGraphStore

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Sanitize title or concept name for safe filenames across OS filesystems."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return clean or "Untitled"


class ObsidianExporter:
    """
    Exports ConceptGraphStore into an interconnected Obsidian markdown vault
    with YAML frontmatter and [[wiki-links]] between Books, Sections, and Concepts.
    """

    def __init__(self, output_dir: Path | str, create_moc: bool = True):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.create_moc = create_moc

    def export(self, store: ConceptGraphStore) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        books_dir = self.output_dir / "Books"
        sections_dir = self.output_dir / "Sections"
        concepts_dir = self.output_dir / "Concepts"

        books_dir.mkdir(exist_ok=True)
        sections_dir.mkdir(exist_ok=True)
        concepts_dir.mkdir(exist_ok=True)

        g = store.graph

        # 1. Export Book notes
        for node_id, attrs in g.nodes(data=True):
            if attrs.get("type") != "Book":
                continue

            title = attrs.get("title", "Untitled Book")
            author = attrs.get("author", "Unknown Author")
            summary = attrs.get("summary", "")
            filename = sanitize_filename(title) + ".md"

            # Discover sections and concepts for this book
            child_sections = []
            discussed_concepts = set()

            for _, sec_id, edge in g.out_edges(node_id, data=True):
                if edge.get("relation") == "HAS_SECTION":
                    sec_attrs = g.nodes.get(sec_id, {})
                    sec_title = sec_attrs.get("title", sec_id)
                    child_sections.append(sec_title)

                    # Concepts discussed in this section
                    for _, c_id, c_edge in g.out_edges(sec_id, data=True):
                        if c_edge.get("relation") == "DISCUSSES":
                            c_name = g.nodes.get(c_id, {}).get("name", c_id.replace("concept:", ""))
                            discussed_concepts.add(c_name)

            content = [
                "---",
                f"title: \"{title}\"",
                "type: book",
                f"author: \"{author}\"",
                "---",
                "",
                f"# {title}",
                f"*Author: {author}*",
                "",
            ]
            if summary:
                content.extend(["> " + summary.replace("\n", " "), ""])

            if child_sections:
                content.append("## Sections")
                for s in child_sections:
                    sec_note_name = f"{title} - {s}"
                    content.append(f"- [[{sec_note_name}|{s}]]")
                content.append("")

            if discussed_concepts:
                content.append("## Core Concepts Discovered")
                for c in sorted(discussed_concepts):
                    content.append(f"- [[{c}]]")
                content.append("")

            with open(books_dir / filename, "w", encoding="utf-8") as f:
                f.write("\n".join(content))

        # 2. Export Section notes
        for node_id, attrs in g.nodes(data=True):
            if attrs.get("type") != "Section":
                continue

            sec_title = attrs.get("title", "Section")
            book_id = attrs.get("book_id")
            book_title = "Unknown Book"
            if book_id and g.has_node(f"book:{book_id}"):
                book_title = g.nodes[f"book:{book_id}"].get("title", book_title)

            sec_note_name = sanitize_filename(f"{book_title} - {sec_title}")
            filename = sec_note_name + ".md"

            discussed_concepts = []
            for _, c_id, edge in g.out_edges(node_id, data=True):
                if edge.get("relation") == "DISCUSSES":
                    c_name = g.nodes.get(c_id, {}).get("name", c_id.replace("concept:", ""))
                    edge_sum = edge.get("summary", "")
                    quote = edge.get("quote", "")
                    entry = f"- [[{c_name}]]"
                    if edge_sum:
                        entry += f": {edge_sum}"
                    if quote:
                        entry += f"\n  > \"{quote}\""
                    discussed_concepts.append(entry)

            sec_content = [
                "---",
                f"title: \"{sec_title}\"",
                "type: section",
                f"book: \"{book_title}\"",
                "---",
                "",
                f"# {sec_title}",
                f"*From book: [[{sanitize_filename(book_title)}|{book_title}]]*",
                "",
            ]
            if discussed_concepts:
                sec_content.append("## Concepts Discussed")
                sec_content.extend(discussed_concepts)
                sec_content.append("")

            with open(sections_dir / filename, "w", encoding="utf-8") as f:
                f.write("\n".join(sec_content))

        # 3. Export Concept / Idea notes
        categories: Dict[str, List[str]] = {}
        for node_id, attrs in g.nodes(data=True):
            if attrs.get("type") not in ["Concept", "Idea"]:
                continue

            name = attrs.get("name", node_id.replace("concept:", ""))
            category = attrs.get("category", "Idea")
            brief_desc = attrs.get("brief_description", "")
            detailed_exp = attrs.get("detailed_explanation", "")
            summary = attrs.get("summary", "") or brief_desc or detailed_exp
            occurrences = attrs.get("occurrences", 1)

            categories.setdefault(category, []).append(name)
            filename = sanitize_filename(name) + ".md"

            # 3a. Supporting chunks & book locations (SUPPORTED_BY edges)
            supporting_evidence = []
            for _, chunk_id, edge in g.out_edges(node_id, data=True):
                if edge.get("relation") == "SUPPORTED_BY":
                    quote = edge.get("quote", "")
                    breadcrumb = edge.get("breadcrumb", "")
                    cid = edge.get("chunk_id", chunk_id.replace("chunk:", ""))
                    sec_attrs = g.nodes.get(chunk_id, {})
                    stitle = sec_attrs.get("section_title", "Section")
                    bid = sec_attrs.get("book_id")
                    btitle = "Book"
                    if bid and g.has_node(f"book:{bid}"):
                        btitle = g.nodes[f"book:{bid}"].get("title", btitle)
                    sec_note = sanitize_filename(f"{btitle} - {stitle}")

                    display_loc = breadcrumb if breadcrumb else f"{btitle} > {stitle}"
                    line = f"- **[[{sec_note}|{display_loc}]]** `[Chunk {cid}]`"
                    if quote:
                        line += f"\n  > \"{quote}\""
                    supporting_evidence.append(line)

            # 3b. Mentioned in sections (DISCUSSES in-edges)
            mentioned_in = []
            for src_id, _, edge in g.in_edges(node_id, data=True):
                if edge.get("relation") == "DISCUSSES":
                    src_attrs = g.nodes.get(src_id, {})
                    stitle = src_attrs.get("title", "Section")
                    bid = src_attrs.get("book_id")
                    btitle = "Book"
                    if bid and g.has_node(f"book:{bid}"):
                        btitle = g.nodes[f"book:{bid}"].get("title", btitle)
                    sec_note = sanitize_filename(f"{btitle} - {stitle}")
                    quote = edge.get("quote", "")
                    entry = f"- [[{sec_note}|{btitle} > {stitle}]]"
                    if quote:
                        entry += f"\n  > \"{quote}\""
                    mentioned_in.append(entry)

            # 3c. Outgoing relations to other concepts
            related_out = []
            for _, tgt_id, edge in g.out_edges(node_id, data=True):
                if edge.get("relation") not in ["DISCUSSES", "SUPPORTED_BY", "HAS_SECTION", "HAS_CHUNK"]:
                    tgt_name = g.nodes.get(tgt_id, {}).get("name", tgt_id.replace("concept:", ""))
                    rel = edge.get("relation", "RELATES_TO")
                    related_out.append(f"- **{rel}** -> [[{tgt_name}]]")

            # 3d. Incoming relations from other concepts
            related_in = []
            for src_id, _, edge in g.in_edges(node_id, data=True):
                if edge.get("relation") not in ["DISCUSSES", "HAS_SECTION", "HAS_CHUNK", "SUPPORTS_IDEA"]:
                    src_name = g.nodes.get(src_id, {}).get("name", src_id.replace("concept:", ""))
                    rel = edge.get("relation", "RELATES_TO")
                    related_in.append(f"- [[{src_name}]] -> **{rel}**")

            cat_tag = re.sub(r"[^\w-]", "", category.lower().replace(" ", "-")) or "idea"
            c_content = [
                "---",
                f"title: \"{name}\"",
                "type: idea",
                "tags:",
                "  - idea",
                f"  - {cat_tag}",
                f"category: \"{category}\"",
                f"occurrences: {occurrences}",
                "---",
                "",
                f"# {name}",
                f"*Category: {category}*",
                "",
            ]

            # Brief Description / Summary
            if brief_desc or summary:
                c_content.extend([
                    "## Summary",
                    f"> {brief_desc or summary}",
                    "",
                ])

            # Detailed Explanation
            if detailed_exp and detailed_exp != brief_desc:
                c_content.extend([
                    "## Detailed Explanation",
                    detailed_exp,
                    "",
                ])

            # Supporting Evidence from Chunks
            if supporting_evidence:
                c_content.append("## Supporting Evidence & Book Locations")
                c_content.extend(supporting_evidence)
                c_content.append("")
            elif mentioned_in:
                c_content.append("## Mentioned In")
                c_content.extend(sorted(set(mentioned_in)))
                c_content.append("")

            # Related Ideas
            if related_out or related_in:
                c_content.append("## Related Ideas")
                if related_out:
                    c_content.extend(related_out)
                if related_in:
                    c_content.extend(related_in)
                c_content.append("")

            with open(concepts_dir / filename, "w", encoding="utf-8") as f:
                f.write("\n".join(c_content))

        # 4. Map of Contents (Index)
        if self.create_moc:
            index_path = self.output_dir / "00_Index.md"
            moc = [
                "# 🧠 Concept Knowledge Graph Index",
                "",
                "Welcome to your extracted book knowledge graph. Click on any concept or book below or explore via Obsidian's **Graph View**.",
                "",
            ]
            for cat, names in sorted(categories.items()):
                moc.append(f"## {cat}")
                for n in sorted(names):
                    moc.append(f"- [[{n}]]")
                moc.append("")

            with open(index_path, "w", encoding="utf-8") as f:
                f.write("\n".join(moc))

        logger.info(f"Exported Obsidian vault to {self.output_dir}")
        return self.output_dir


class GraphMLExporter:
    """Exports graph in standard GraphML format for Gephi, Cytoscape Desktop, and yEd."""

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path).expanduser().resolve()

    def export(self, store: ConceptGraphStore) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure all attribute values are primitive strings/ints for GraphML writer
        g_copy = nx.DiGraph()
        for n, attrs in store.graph.nodes(data=True):
            clean_attrs = {}
            for k, v in attrs.items():
                clean_attrs[k] = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            g_copy.add_node(n, **clean_attrs)

        for u, v, attrs in store.graph.edges(data=True):
            clean_attrs = {}
            for k, val in attrs.items():
                clean_attrs[k] = json.dumps(val) if isinstance(val, (list, dict)) else str(val)
            g_copy.add_edge(u, v, **clean_attrs)

        nx.write_graphml(g_copy, str(self.output_path))
        logger.info(f"Exported GraphML to {self.output_path}")
        return self.output_path


class GEXFExporter:
    """Exports graph in standard GEXF format for Gephi."""

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path).expanduser().resolve()

    def export(self, store: ConceptGraphStore) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        g_copy = nx.DiGraph()
        for n, attrs in store.graph.nodes(data=True):
            clean_attrs = {}
            for k, v in attrs.items():
                clean_attrs[k] = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            g_copy.add_node(n, **clean_attrs)

        for u, v, attrs in store.graph.edges(data=True):
            clean_attrs = {}
            for k, val in attrs.items():
                clean_attrs[k] = json.dumps(val) if isinstance(val, (list, dict)) else str(val)
            g_copy.add_edge(u, v, **clean_attrs)

        nx.write_gexf(g_copy, str(self.output_path))
        logger.info(f"Exported GEXF to {self.output_path}")
        return self.output_path
