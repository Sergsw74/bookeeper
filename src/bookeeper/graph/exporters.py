"""
Exporters for Concept Knowledge Graph: Obsidian Markdown Vault, GraphML, and Cytoscape JSON.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from bookeeper.config import ExportSettings
from bookeeper.graph.store import ConceptGraphStore

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Sanitize concept or book titles for filesystem and Obsidian note names."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return sanitized or "Untitled"


class ObsidianExporter:
    """Exports graph nodes and edges into an interconnected Obsidian markdown vault."""

    def __init__(self, output_dir: Path | str, create_moc: bool = True):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.create_moc = create_moc

    def export(self, store: ConceptGraphStore) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        concepts_dir = self.output_dir / "Concepts"
        books_dir = self.output_dir / "Books"
        concepts_dir.mkdir(exist_ok=True)
        books_dir.mkdir(exist_ok=True)

        g = store.graph
        categories: Dict[str, List[str]] = {}

        # 1. Export Concept notes
        for node_id, attrs in g.nodes(data=True):
            if attrs.get("type") != "Concept":
                continue

            name = attrs.get("name", node_id.replace("concept:", ""))
            category = attrs.get("category", "Concept")
            desc = attrs.get("description", "")
            aliases = attrs.get("aliases", [])
            occurrences = attrs.get("occurrences", 1)

            categories.setdefault(category, []).append(name)
            filename = sanitize_filename(name) + ".md"
            filepath = concepts_dir / filename

            # Build outgoing semantic relationships
            out_edges_text = []
            for _, tgt_id, edge_attrs in g.out_edges(node_id, data=True):
                rel = edge_attrs.get("relation", "RELATES_TO")
                if rel in ["MENTIONS", "CONTAINS"]:
                    continue
                tgt_attrs = g.nodes.get(tgt_id, {})
                tgt_name = tgt_attrs.get("name", tgt_id.replace("concept:", ""))
                expl = edge_attrs.get("explanation", "")
                expl_str = f" - *{expl}*" if expl else ""
                out_edges_text.append(f"- **{rel}** -> [[{tgt_name}]]{expl_str}")

            # Build incoming semantic relationships
            in_edges_text = []
            for src_id, _, edge_attrs in g.in_edges(node_id, data=True):
                rel = edge_attrs.get("relation", "")
                if rel in ["MENTIONS", "CONTAINS"]:
                    continue
                src_attrs = g.nodes.get(src_id, {})
                src_name = src_attrs.get("name", src_id.replace("concept:", ""))
                expl = edge_attrs.get("explanation", "")
                expl_str = f" - *{expl}*" if expl else ""
                in_edges_text.append(f"- [[{src_name}]] -> **{rel}**{expl_str}")

            # Build source chapter mentions
            mentions_text = []
            for src_id, _, edge_attrs in g.in_edges(node_id, data=True):
                if edge_attrs.get("relation") == "MENTIONS":
                    src_attrs = g.nodes.get(src_id, {})
                    chap_title = src_attrs.get("title", "Unknown Section")
                    book_id = src_attrs.get("book_id")
                    book_title = "Unknown Book"
                    if book_id and g.has_node(f"book:{book_id}"):
                        book_title = g.nodes[f"book:{book_id}"].get("title", book_title)
                    mentions_text.append(f"- [[{book_title}]] > *{chap_title}*")

            # Markdown Content
            content = ["---"]
            content.append(f"title: \"{name}\"")
            content.append("type: concept")
            content.append(f"category: \"{category}\"")
            if aliases:
                content.append(f"aliases: {json.dumps(aliases)}")
            content.append(f"occurrences: {occurrences}")
            content.append("---\n")

            content.append(f"# {name}\n")
            if desc:
                content.append(f"> {desc}\n")

            if out_edges_text or in_edges_text:
                content.append("## Relationships\n")
                if out_edges_text:
                    content.extend(out_edges_text)
                if in_edges_text:
                    content.extend(in_edges_text)
                content.append("")

            if mentions_text:
                content.append("## Mentioned In\n")
                content.extend(sorted(set(mentions_text)))
                content.append("")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(content))

        # 2. Export Book notes
        for node_id, attrs in g.nodes(data=True):
            if attrs.get("type") != "Book":
                continue

            title = attrs.get("title", "Untitled Book")
            filename = sanitize_filename(title) + ".md"
            filepath = books_dir / filename

            # Gather concepts mentioned in this book
            mentioned_concepts = set()
            for _, chap_id, edge in g.out_edges(node_id, data=True):
                if edge.get("relation") == "CONTAINS":
                    for _, c_id, c_edge in g.out_edges(chap_id, data=True):
                        if c_edge.get("relation") == "MENTIONS":
                            c_name = g.nodes.get(c_id, {}).get("name", c_id.replace("concept:", ""))
                            mentioned_concepts.add(c_name)

            b_content = ["---"]
            b_content.append(f"title: \"{title}\"")
            b_content.append("type: book")
            b_content.append("---\n")
            b_content.append(f"# {title}\n")

            if mentioned_concepts:
                b_content.append("## Key Concepts\n")
                for c in sorted(mentioned_concepts):
                    b_content.append(f"- [[{c}]]")
                b_content.append("")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(b_content))

        # 3. Create Map of Contents (Index)
        if self.create_moc:
            index_path = self.output_dir / "00_Index.md"
            moc = ["# Concept Knowledge Graph Index\n"]
            for cat, names in sorted(categories.items()):
                moc.append(f"## {cat}\n")
                for n in sorted(names):
                    moc.append(f"- [[{n}]]")
                moc.append("")
            with open(index_path, "w", encoding="utf-8") as f:
                f.write("\n".join(moc))

        logger.info(f"Exported Obsidian vault to {self.output_dir}")
        return self.output_dir


class GraphMLExporter:
    """Exports graph in standard GraphML format for Gephi, Cytoscape, and NetworkX."""

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path).expanduser().resolve()

    def export(self, store: ConceptGraphStore) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert lists to strings for GraphML compatibility
        g_copy = nx.MultiDiGraph()
        for n, attrs in store.graph.nodes(data=True):
            clean_attrs = {}
            for k, v in attrs.items():
                clean_attrs[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
            g_copy.add_node(n, **clean_attrs)

        for u, v, k, attrs in store.graph.edges(keys=True, data=True):
            clean_attrs = {}
            for key, val in attrs.items():
                clean_attrs[key] = json.dumps(val) if isinstance(val, (list, dict)) else val
            g_copy.add_edge(u, v, key=k, **clean_attrs)

        nx.write_graphml(g_copy, str(self.output_path))
        logger.info(f"Exported GraphML graph to {self.output_path}")
        return self.output_path


class CytoscapeExporter:
    """Exports graph elements in Cytoscape.js compatible JSON format."""

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path).expanduser().resolve()

    def export(self, store: ConceptGraphStore) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        for node_id, attrs in store.graph.nodes(data=True):
            nodes.append(
                {
                    "data": {
                        "id": node_id,
                        "label": attrs.get("name") or attrs.get("title") or node_id,
                        **attrs,
                    }
                }
            )

        for idx, (src, tgt, edge_attrs) in enumerate(store.graph.edges(data=True)):
            edges.append(
                {
                    "data": {
                        "id": f"e_{idx}_{src}_{tgt}",
                        "source": src,
                        "target": tgt,
                        "relation": edge_attrs.get("relation", "RELATES_TO"),
                        **edge_attrs,
                    }
                }
            )

        payload = {"elements": {"nodes": nodes, "edges": edges}}
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Exported Cytoscape elements to {self.output_path}")
        return self.output_path
