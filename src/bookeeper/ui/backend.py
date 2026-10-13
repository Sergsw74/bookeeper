"""
Bookeeper Idea-Based Search Engine Backend Service.

Coordinates:
1. Natural language query handling and concept identification via LightRAG (and vector/entity index fallback).
2. Knowledge Graph traversal, entity expansion, and relation fetching via Neo4j.
3. Observability telemetry capturing exact prompts, Cypher statements, and raw responses for the UI Debug Tab.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import neo4j

from bookeeper.config import Neo4jConfig, Settings
from bookeeper.rag.lightrag_engine import LightRAGEngine

logger = logging.getLogger(__name__)


# Generic / stopword concepts that should be penalized or filtered out unless directly queried
GENERIC_CONCEPT_DENOISE_SET = {
    "definition", "definitions", "example", "examples", "concept", "concepts",
    "description", "plot details", "plot", "plots", "interpretation", "interpretations",
    "specifics", "comparison", "comparisons", "details", "role", "roles",
    "title", "titles", "author", "authors", "book reference", "reference document",
    "summary", "overview", "introduction", "conclusion", "method", "methodology",
    "principle", "principles", "pattern", "patterns", "model", "models",
    "figure", "table", "section", "chapter", "page", "result", "results",
    "performance", "metric", "metrics", "measurement", "measurements",
    "analysis", "evaluation", "evaluations", "approach", "framework",
    "study", "paper", "article", "task", "tasks", "data", "dataset",
}


class SearchEngineBackend:
    """
    Coordinates LightRAG and Neo4j for idea-based semantic search and graph visualization.
    """

    def __init__(
        self,
        config: Settings,
        lightrag_dir: Optional[Path] = None,
        neo4j_config: Optional[Neo4jConfig] = None,
    ):
        self.config = config
        self.lightrag_dir = lightrag_dir or config.resolved_lightrag_dir
        self.neo4j_cfg = neo4j_config or config.neo4j
        self._lightrag_engine: Optional[LightRAGEngine] = None
        self._neo4j_driver: Optional[neo4j.Driver] = None
        self._vdb_entities: Optional[List[Dict[str, Any]]] = None

    def get_neo4j_driver(self) -> neo4j.Driver:
        """Get or initialize the Neo4j driver."""
        if self._neo4j_driver is None:
            self._neo4j_driver = neo4j.GraphDatabase.driver(
                self.neo4j_cfg.uri,
                auth=(self.neo4j_cfg.user, self.neo4j_cfg.password),
                connection_timeout=self.neo4j_cfg.connection_timeout,
            )
        return self._neo4j_driver

    def close(self):
        """Clean up connections."""
        if self._neo4j_driver is not None:
            try:
                self._neo4j_driver.close()
            except Exception:
                pass
            self._neo4j_driver = None

    def get_lightrag_engine(self) -> Optional[LightRAGEngine]:
        """Get or initialize LightRAGEngine if storage is available."""
        if self._lightrag_engine is None and self.lightrag_dir.exists():
            try:
                self._lightrag_engine = LightRAGEngine.from_settings(
                    self.config,
                    working_dir=self.lightrag_dir,
                )
                self._lightrag_engine.initialize()
            except Exception as e:
                logger.warning(f"LightRAGEngine could not be initialized: {e}")
                self._lightrag_engine = None
        return self._lightrag_engine

    def _load_vdb_entities(self) -> List[Dict[str, Any]]:
        """Load entity names and descriptions from LightRAG storage for fast keyword/semantic fallback."""
        if self._vdb_entities is not None:
            return self._vdb_entities
        vdb_file = self.lightrag_dir / "vdb_entities.json"
        if vdb_file.is_file():
            try:
                with open(vdb_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._vdb_entities = data.get("data", [])
                    return self._vdb_entities
            except Exception as e:
                logger.debug(f"Failed to load vdb_entities.json: {e}")
        self._vdb_entities = []
        return self._vdb_entities

    def _extract_candidate_terms(self, text: str, user_query: str = "") -> List[str]:
        """
        Extract candidate concept terms from free-form text or LightRAG responses,
        filtering out generic boilerplate headings (Definition, Example, Overview, etc.).
        """
        raw_candidates = []
        # Markdown bold / headers / quotes
        for match in re.findall(r"\*\*([^*]+)\*\*", text):
            cleaned = match.strip()
            if 2 <= len(cleaned) <= 60 and not cleaned.startswith("http"):
                raw_candidates.append(cleaned)
        # Quoted terms
        for match in re.findall(r"\"([^\"]+)\"", text):
            cleaned = match.strip()
            if 2 <= len(cleaned) <= 60:
                raw_candidates.append(cleaned)
        # Bullet list items
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(("-", "*", "•")):
                item = line.lstrip("-*•").split(":", 1)[0].split(" - ", 1)[0].strip()
                item = re.sub(r"\[\[(.*?)\]\]", r"\1", item)  # unwrap wikilinks
                if 2 <= len(item) <= 60:
                    raw_candidates.append(item)

        user_query_clean = user_query.lower().strip()
        user_words = set(re.findall(r"\w+", user_query_clean))

        # Denoise candidates: remove generic structural labels unless user explicitly asked for them
        filtered = []
        seen = set()
        for c in raw_candidates:
            c_norm = c.lower().strip(" *\"'[]").strip()
            if not c_norm or len(c_norm) < 2:
                continue

            # If it's a generic word and not part of the user query, discard
            if c_norm in GENERIC_CONCEPT_DENOISE_SET and c_norm not in user_query_clean:
                continue

            # Strip document citation artifacts (e.g. "Reference Document 1] Book")
            if "reference document" in c_norm or "document chunk" in c_norm:
                continue

            if c_norm not in seen:
                seen.add(c_norm)
                # Compute relevance boost: does it share keywords with the user query?
                c_words = set(re.findall(r"\w+", c_norm))
                overlap = len(c_words.intersection(user_words))
                filtered.append((c, overlap))

        # Sort by keyword overlap with user query descending, preserving discovery order
        filtered.sort(key=lambda x: -x[1])
        return [term for term, _ in filtered]

    def query_ideas(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 40,
        chunk_top_k: int = 20,
        graph_limit: int = 50,
        denoise_level: str = "standard",
    ) -> Dict[str, Any]:
        """
        Execute the Idea-Based Search pipeline with smart noise reduction:
        1. Query LightRAG to synthesize an answer and extract relevant concept entities.
        2. Denoise and prioritize concept candidates based on relevance to user query.
        3. Execute Cypher queries on Neo4j using relevance ranking and hub degree limits.
        4. Capture full telemetry for the UI Debug Tab.
        """
        telemetry: Dict[str, Any] = {
            "query": query,
            "mode": mode,
            "denoise_level": denoise_level,
            "timestamp": time.time(),
            "lightrag": {},
            "neo4j": {},
        }

        # Step 1: Semantic Retrieval via LightRAG
        lightrag_response = ""
        retrieved_entity_names: List[str] = []
        lightrag_error = None
        t0_rag = time.perf_counter()

        engine = self.get_lightrag_engine()
        telemetry["lightrag"]["exact_query"] = query
        telemetry["lightrag"]["mode"] = mode
        telemetry["lightrag"]["top_k"] = top_k
        telemetry["lightrag"]["chunk_top_k"] = chunk_top_k

        if engine is not None:
            try:
                lightrag_response = engine.query(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                )
                telemetry["lightrag"]["status"] = "success"
                telemetry["lightrag"]["raw_response"] = lightrag_response
            except Exception as e:
                lightrag_error = str(e)
                logger.warning(f"LightRAG query invocation failed: {e}")
                telemetry["lightrag"]["status"] = "error"
                telemetry["lightrag"]["error"] = lightrag_error
        else:
            telemetry["lightrag"]["status"] = "skipped"
            telemetry["lightrag"]["note"] = "LightRAG storage not initialized or offline."

        rag_duration = time.perf_counter() - t0_rag
        telemetry["lightrag"]["duration_seconds"] = round(rag_duration, 4)

        # Step 2: Denoise & Rank Candidate Terms
        candidate_terms = []
        # User query itself is always prioritized at rank 1
        candidate_terms.append(query.strip())

        # Substrings / ngrams of query if query has multiple words (e.g. "kv-cache" -> ["kv-cache", "kv", "cache"])
        query_words = [w for w in re.findall(r"[\w-]+", query) if len(w) >= 2]
        for w in query_words:
            if w.lower() not in GENERIC_CONCEPT_DENOISE_SET:
                candidate_terms.append(w)

        # Add denoised terms extracted from LightRAG response
        if lightrag_response:
            candidate_terms.extend(self._extract_candidate_terms(lightrag_response, user_query=query))

        # Check pre-indexed entities in LightRAG vdb for fuzzy / keyword match against user prompt
        vdb_ents = self._load_vdb_entities()
        query_kw_set = {w.lower() for w in query_words if len(w) > 2}
        for ent in vdb_ents:
            ename = ent.get("entity_name", "")
            if not ename:
                continue
            ename_lower = ename.lower()
            if ename_lower in query.lower() or any(w in ename_lower for w in query_kw_set):
                if ename_lower not in GENERIC_CONCEPT_DENOISE_SET:
                    candidate_terms.append(ename)

        # Deduplicate while preserving relevance rank
        dedup_terms = []
        seen = set()
        for t in candidate_terms:
            t_clean = t.strip(" *\"'[]").strip()
            if t_clean and t_clean.lower() not in seen and len(t_clean) >= 2:
                seen.add(t_clean.lower())
                dedup_terms.append(t_clean)

        retrieved_entity_names = dedup_terms[:25]
        telemetry["lightrag"]["identified_concepts"] = retrieved_entity_names

        # Step 3: Graph Construction via Neo4j with Hub-Degree Penalty & Relevance
        t0_neo = time.perf_counter()
        driver = self.get_neo4j_driver()

        # Maximum degree threshold to filter out flooding generic super-hubs
        max_hub_degree = 250 if denoise_level == "relaxed" else (120 if denoise_level == "standard" else 60)

        # Build Cypher query matching identified ideas and their neighbor subgraph
        cypher_query = """
        // Find matching Concept/Idea nodes with relevance weighting
        UNWIND $candidate_terms AS term
        MATCH (c:Concept)
        WHERE toLower(c.name) = toLower(term)
           OR toLower(c.name) STARTS WITH toLower(term)
           OR (size(term) >= 3 AND toLower(c.name) CONTAINS toLower(term))
        WITH c,
             CASE
               WHEN toLower(c.name) = toLower(term) THEN 120
               WHEN toLower(c.name) STARTS WITH toLower(term) THEN 70
               ELSE 25
             END AS relevance

        // Compute degree to penalize over-connected super-hubs
        OPTIONAL MATCH (c)-[deg_rel]-()
        WITH c, max(relevance) AS relevance, count(deg_rel) AS degree
        WHERE degree <= $max_hub_degree OR relevance >= 100
        WITH c, relevance, degree
        ORDER BY relevance DESC, c.weight DESC, degree ASC
        LIMIT $max_seeds

        // Fetch surrounding 1-hop subgraph (Books, Sections, and tightly-linked Concepts)
        OPTIONAL MATCH (c)-[rel]-(neighbor)
        WHERE (neighbor:Concept OR neighbor:Book OR neighbor:Section OR neighbor:Chunk)
        RETURN c, rel, neighbor, relevance, degree
        LIMIT $graph_limit
        """

        cypher_params = {
            "candidate_terms": retrieved_entity_names,
            "max_seeds": 8,
            "max_hub_degree": max_hub_degree,
            "graph_limit": graph_limit,
        }

        telemetry["neo4j"]["exact_cypher"] = cypher_query.strip()
        telemetry["neo4j"]["parameters"] = cypher_params

        nodes_map: Dict[str, Dict[str, Any]] = {}
        edges_list: List[Dict[str, Any]] = []
        neo4j_error = None

        try:
            with driver.session(database=self.neo4j_cfg.database) as session:
                result = session.run(cypher_query, cypher_params)
                for record in result:
                    c_node = record["c"]
                    rel = record["rel"]
                    neighbor = record["neighbor"]

                    if c_node:
                        cid = c_node.get("id", str(c_node.element_id))
                        if cid not in nodes_map:
                            nodes_map[cid] = self._serialize_node(c_node)

                    if neighbor:
                        nid = neighbor.get("id", str(neighbor.element_id))
                        if nid not in nodes_map:
                            nodes_map[nid] = self._serialize_node(neighbor)

                    if rel and c_node and neighbor:
                        start_id = rel.start_node.get("id", str(rel.start_node.element_id))
                        end_id = rel.end_node.get("id", str(rel.end_node.element_id))
                        edge_obj = {
                            "id": f"{start_id}->{rel.type}->{end_id}",
                            "source": start_id,
                            "target": end_id,
                            "type": rel.type,
                            "properties": dict(rel),
                        }
                        edges_list.append(edge_obj)

            # If no nodes were found via candidate terms, do a broad text search fallback
            if not nodes_map:
                fallback_cypher = """
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($query)
                   OR toLower(c.brief_description) CONTAINS toLower($query)
                WITH c LIMIT 5
                OPTIONAL MATCH (c)-[rel]-(neighbor)
                RETURN c, rel, neighbor LIMIT $graph_limit
                """
                telemetry["neo4j"]["fallback_cypher"] = fallback_cypher.strip()
                with driver.session(database=self.neo4j_cfg.database) as session:
                    res2 = session.run(fallback_cypher, {"query": query.strip(), "graph_limit": graph_limit})
                    for record in res2:
                        c_node = record["c"]
                        rel = record["rel"]
                        neighbor = record["neighbor"]
                        if c_node:
                            cid = c_node.get("id", str(c_node.element_id))
                            if cid not in nodes_map:
                                nodes_map[cid] = self._serialize_node(c_node)
                        if neighbor:
                            nid = neighbor.get("id", str(neighbor.element_id))
                            if nid not in nodes_map:
                                nodes_map[nid] = self._serialize_node(neighbor)
                        if rel and c_node and neighbor:
                            start_id = rel.start_node.get("id", str(rel.start_node.element_id))
                            end_id = rel.end_node.get("id", str(rel.end_node.element_id))
                            edges_list.append({
                                "id": f"{start_id}->{rel.type}->{end_id}",
                                "source": start_id,
                                "target": end_id,
                                "type": rel.type,
                                "properties": dict(rel),
                            })

            telemetry["neo4j"]["status"] = "success"
            telemetry["neo4j"]["nodes_count"] = len(nodes_map)
            telemetry["neo4j"]["edges_count"] = len(edges_list)

        except Exception as e:
            neo4j_error = str(e)
            logger.error(f"Neo4j Cypher query failed: {e}")
            telemetry["neo4j"]["status"] = "error"
            telemetry["neo4j"]["error"] = neo4j_error

        neo_duration = time.perf_counter() - t0_neo
        telemetry["neo4j"]["duration_seconds"] = round(neo_duration, 4)

        # Step 4: Data Formatting
        # Pick the most relevant primary node to automatically select in Detail Panel
        primary_selected_id = None
        for nid, nprops in nodes_map.items():
            if "Concept" in nprops.get("labels", []):
                primary_selected_id = nid
                break
        if not primary_selected_id and nodes_map:
            primary_selected_id = next(iter(nodes_map.keys()))

        return {
            "query": query,
            "answer": lightrag_response or (f"Retrieved {len(nodes_map)} nodes from Neo4j." if nodes_map else "No results found."),
            "identified_concepts": retrieved_entity_names,
            "graph": {
                "nodes": list(nodes_map.values()),
                "edges": edges_list,
            },
            "selected_node_id": primary_selected_id,
            "telemetry": telemetry,
        }

    def get_node_details(self, node_id: str) -> Dict[str, Any]:
        """
        Fetch full details for a selected node, including its properties and
        categorized relations for rendering like an Obsidian page.
        """
        driver = self.get_neo4j_driver()
        cypher = """
        MATCH (n {id: $node_id})
        OPTIONAL MATCH (n)-[rel]-(other)
        RETURN n,
               collect({
                   type: type(rel),
                   direction: CASE WHEN startNode(rel) = n THEN 'outgoing' ELSE 'incoming' END,
                   properties: properties(rel),
                   other_id: other.id,
                   other_name: other.name,
                   other_title: other.title,
                   other_labels: labels(other),
                   other_category: other.category
               }) AS relations
        """
        telemetry = {
            "exact_cypher": cypher.strip(),
            "node_id": node_id,
        }

        with driver.session(database=self.neo4j_cfg.database) as session:
            result = session.run(cypher, {"node_id": node_id})
            record = result.single()
            if not record or not record["n"]:
                return {"error": f"Node '{node_id}' not found.", "telemetry": telemetry}

            n_node = record["n"]
            raw_relations = record["relations"] or []

            # Categorize relations (Books, Related Concepts, Sections/Evidence)
            books = []
            related_concepts = []
            evidence = []
            other_rels = []

            for r in raw_relations:
                if not r.get("other_id"):
                    continue
                labels = r.get("other_labels", [])
                name = r.get("other_name") or r.get("other_title") or r.get("other_id")
                rel_entry = {
                    "rel_type": r.get("type"),
                    "direction": r.get("direction"),
                    "target_id": r.get("other_id"),
                    "target_name": name,
                    "target_labels": labels,
                    "target_category": r.get("other_category"),
                    "quote": r.get("properties", {}).get("quote", ""),
                    "breadcrumb": r.get("properties", {}).get("breadcrumb", ""),
                }

                if "Book" in labels:
                    books.append(rel_entry)
                elif "Concept" in labels:
                    related_concepts.append(rel_entry)
                elif "Chunk" in labels or "Section" in labels:
                    evidence.append(rel_entry)
                else:
                    other_rels.append(rel_entry)

            node_data = self._serialize_node(n_node)
            return {
                "node": node_data,
                "books": books,
                "related_concepts": related_concepts,
                "evidence": evidence,
                "other_relations": other_rels,
                "telemetry": telemetry,
            }

    def _serialize_node(self, node: Any) -> Dict[str, Any]:
        """Convert a Neo4j Node into a JSON-serializable dictionary with display formatting."""
        labels = list(node.labels) if hasattr(node, "labels") else []
        props = dict(node)
        nid = props.get("id", str(getattr(node, "element_id", "unknown")))
        name = props.get("name") or props.get("title") or nid

        # Color & Icon categorization
        category = props.get("category", "General")
        color = "#8b5cf6"  # violet default for concepts
        icon = "💡"

        if "Book" in labels:
            color = "#3b82f6"  # blue
            icon = "📚"
            category = "Book"
        elif "Section" in labels:
            color = "#10b981"  # emerald green
            icon = "📑"
            category = "Section"
        elif "Chunk" in labels:
            color = "#64748b"  # slate gray
            icon = "📄"
            category = "Chunk"
        elif "Idea" in labels or "Concept" in labels:
            color = "#f59e0b" if props.get("weight", 0) >= 7 else "#8b5cf6"
            icon = "🧠" if props.get("weight", 0) >= 8 else "💡"

        return {
            "id": nid,
            "name": name,
            "labels": labels,
            "category": category,
            "weight": props.get("weight", 5),
            "occurrences": props.get("occurrences", 1),
            "color": color,
            "icon": icon,
            "properties": props,
        }
