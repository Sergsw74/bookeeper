"""
Bookeeper Idea-Based Search Engine Backend Service.

Coordinates:
1. Pure Knowledge Graph idea retrieval and concept identification via Neo4j.
2. Neighborhood subgraph expansion (connected concepts, books, sections, chunks).
3. Structured idea synthesis summarizing matched concepts, relations, and supporting evidence.
4. LightRAG-compatible API service interface (/query) for drop-in interoperability.
5. Observability telemetry capturing candidate terms, exact Cypher statements, and execution timing.
"""

import datetime
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import neo4j
import numpy as np

from bookeeper.config import Neo4jConfig, Settings
from bookeeper.ui.embeddings import ConceptEmbeddingIndex

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

# English question stop phrases
QUESTION_PREFIXES = [
    r"^what\s+is\s+a\s+",
    r"^what\s+is\s+an\s+",
    r"^what\s+is\s+the\s+",
    r"^what\s+is\s+",
    r"^what\s+are\s+the\s+",
    r"^what\s+are\s+",
    r"^how\s+does\s+",
    r"^how\s+do\s+",
    r"^tell\s+me\s+about\s+",
    r"^explain\s+",
    r"^describe\s+",
    r"^definition\s+of\s+",
    r"^define\s+",
    r"^overview\s+of\s+",
]


class FastConceptVectorizer:
    """
    In-memory sublinear TF-IDF + Character N-Gram & Token Cosine Similarity Index.
    Runs with sub-millisecond latency (<1ms) and zero network overhead.
    Allows fuzzy matching, morphological variations, and semantic relevance for graph concepts.
    """

    def __init__(self, corpus: Dict[str, str]):
        self.doc_ids: List[str] = list(corpus.keys())
        self.docs: List[str] = list(corpus.values())
        df: Counter = Counter()
        self.doc_features: List[List[str]] = []
        for doc in self.docs:
            feats = self._extract_features(doc)
            self.doc_features.append(feats)
            for f in set(feats):
                df[f] += 1
        n_docs = len(self.docs)
        self.idf: Dict[str, float] = {
            f: float(np.log((n_docs + 1) / (cnt + 1)) + 1.0)
            for f, cnt in df.items()
        }
        self.matrix: List[Dict[str, float]] = [self._vectorize(feats) for feats in self.doc_features]

    @staticmethod
    def _extract_features(text: str) -> List[str]:
        t = text.lower()
        words = [w for w in re.findall(r"\w+", t) if len(w) >= 2]
        ngrams = [t[i:i+3] for i in range(len(t) - 2)]
        return words + ngrams

    def _vectorize(self, feats: List[str]) -> Dict[str, float]:
        counts = Counter(feats)
        vec = {f: float((1.0 + np.log(c)) * self.idf.get(f, 1.0)) for f, c in counts.items()}
        norm = np.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {f: float(v / norm) for f, v in vec.items()}

    def query(self, query_text: str, top_k: int = 10, min_threshold: float = 0.45) -> List[Tuple[str, float]]:
        q_feats = self._extract_features(query_text)
        q_vec = self._vectorize(q_feats)
        if not q_vec:
            return []
        scores: List[Tuple[str, float]] = []
        for i, doc_vec in enumerate(self.matrix):
            score = sum(val * doc_vec[f] for f, val in q_vec.items() if f in doc_vec)
            if score >= min_threshold:
                scores.append((self.doc_ids[i], float(score)))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class SearchEngineBackend:
    """
    Coordinates Neo4j idea retrieval, neighborhood expansion, and LightRAG API emulation.
    """

    def __init__(
        self,
        config: Settings,
        lightrag_dir: Optional[Path] = None,
        neo4j_config: Optional[Neo4jConfig] = None,
    ):
        self.config = config
        self.lightrag_dir = lightrag_dir or getattr(config, "resolved_lightrag_dir", None)
        self.neo4j_cfg = neo4j_config or config.neo4j
        self._neo4j_driver: Optional[neo4j.Driver] = None
        self._concept_vectorizer: Optional[FastConceptVectorizer] = None
        self._concept_embedding_index: Optional[ConceptEmbeddingIndex] = None
        self._books_index_cache: Optional[Dict[str, Any]] = None
        self._books_index_mtime: float = 0.0
        self._local_kg_stats_cache: Optional[Dict[str, int]] = None
        self._local_kg_mtime: float = 0.0
        self._restore_lock = threading.Lock()
        self._is_restoring: bool = False
        self._restore_status: Dict[str, Any] = {
            "in_progress": False,
            "stage": "idle",
            "message": "",
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }

    @property
    def is_restoring(self) -> bool:
        return self._is_restoring

    def get_restore_status(self) -> Dict[str, Any]:
        with self._restore_lock:
            return dict(self._restore_status)

    def _get_all_books_dirs(self) -> List[Path]:
        """
        Discover and return all candidate directories containing book files or index.json.
        Preserves precedence while eliminating duplicates and empty folders.
        """
        raw_candidates: List[Any] = []

        # 1. Environment variables
        raw_candidates.append(os.environ.get("BOOKEEPER_BOOKS_DIR"))
        raw_candidates.append(os.environ.get("BOOKS_DIR"))

        # 2. Configured resolved_books_dir / books_dir
        raw_candidates.append(getattr(self.config, "resolved_books_dir", None))
        raw_candidates.append(getattr(self.config, "books_dir", None))

        # 3. Locations relative to output directory (where full backups extract books)
        out_dir = getattr(self.config, "resolved_output_dir", None) or getattr(self.config, "output_dir", None)
        if out_dir:
            try:
                out_p = Path(out_dir).expanduser().resolve()
                raw_candidates.extend([
                    out_p / "archive_downloader" / "downloads",
                    out_p / "downloads",
                    out_p / "books",
                    out_p.parent / "archive_downloader" / "downloads",
                    out_p.parent / "downloads",
                    out_p.parent / "books",
                ])
            except Exception:
                pass

        # 4. Standard container mount locations
        raw_candidates.extend([
            Path("/app/output/archive_downloader/downloads"),
            Path("/app/output/downloads"),
            Path("/app/output/books"),
            Path("/app/archive_downloader/downloads"),
            Path("/app/downloads"),
            Path("/app/books"),
        ])

        # 5. Common server filesystem paths (e.g., Ubuntu server restore targets)
        raw_candidates.extend([
            Path("/srv/dev-disk-by-uuid-1d5527f6-4c4f-419b-aee6-896ae783c213/bookeeper/archive_downloader/downloads"),
            Path("/srv/dev-disk-by-uuid-1d5527f6-4c4f-419b-aee6-896ae783c213/bookeeper/downloads"),
            Path("/srv/dev-disk-by-uuid-1d5527f6-4c4f-419b-aee6-896ae783c213/bookeeper/books"),
            Path("/srv/dev-disk-by-uuid-1d5527f6-4c4f-419b-aee6-896ae783c213/bookeeper"),
            Path("/home/ubuntu/repos/bookeeper/archive_downloader/downloads"),
            Path("/home/ubuntu/repos/bookeeper/downloads"),
            Path("/home/ubuntu/repos/bookeeper/books"),
        ])

        # 6. Current working directory relatives
        raw_candidates.extend([
            Path("archive_downloader/downloads").resolve(),
            Path("./downloads").resolve(),
            Path("./books").resolve(),
            Path("../archive_downloader/downloads").resolve(),
        ])

        def _dir_has_books(p: Path) -> bool:
            try:
                if not p.is_dir():
                    return False
                idx = p / "index.json"
                if idx.is_file() and idx.stat().st_size > 2:
                    return True
                for ext in ("*.pdf", "*.epub", "*.mobi", "*.azw3", "*.djvu", "*.txt", "*.opf"):
                    if next(p.glob(ext), None) is not None:
                        return True
            except Exception:
                pass
            return False

        valid_dirs: List[Path] = []
        seen = set()

        # First pass: only return directories that actually contain books or index.json
        for c in raw_candidates:
            if not c:
                continue
            try:
                p = Path(c).expanduser().resolve()
                if p not in seen:
                    seen.add(p)
                    if _dir_has_books(p):
                        valid_dirs.append(p)
            except Exception:
                continue

        # Second pass fallback: if no directory had books, pick any candidate that exists on disk
        if not valid_dirs:
            for c in raw_candidates:
                if not c:
                    continue
                try:
                    p = Path(c).expanduser().resolve()
                    if p.is_dir() and p not in valid_dirs:
                        valid_dirs.append(p)
                except Exception:
                    continue

        return valid_dirs

    def _get_books_dir(self) -> Optional[Path]:
        """Resolve the primary directory containing book files."""
        dirs = self._get_all_books_dirs()
        return dirs[0] if dirs else None

    def _get_book_title_by_id(self, book_id: Any) -> Optional[str]:
        """
        Resolve real book title from numeric/string book_id using offline artifacts:
        .book_states, chunks, knowledge_graph.json, or Neo4j if available.
        """
        if book_id is None:
            return None
        bid_str = str(book_id).replace("book:", "").strip()
        if not bid_str or bid_str.lower() in ("sample", "unknown", "untitled", "none"):
            return None

        if not hasattr(self, "_book_id_title_map") or self._book_id_title_map is None:
            self._book_id_title_map = {}

        if bid_str in self._book_id_title_map:
            return self._book_id_title_map[bid_str]

        out_dir = getattr(self.config, "resolved_output_dir", None) or Path("output")
        try:
            out_p = Path(out_dir).expanduser().resolve()
        except Exception:
            out_p = Path("output").resolve()

        # 1. Check .book_states/
        try:
            state_file = out_p / ".book_states" / f"book_{bid_str}_state.json"
            if state_file.is_file():
                with open(state_file, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                    bt = sdata.get("book_title")
                    if bt and bt.strip() and bt.lower() != "sample":
                        self._book_id_title_map[bid_str] = bt.strip()
                        return bt.strip()
        except Exception:
            pass

        # 2. Check chunks/
        try:
            chunk_file = out_p / "chunks" / f"book_{bid_str}_chunks.json"
            if chunk_file.is_file():
                with open(chunk_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    bt = cdata.get("book_title") if isinstance(cdata, dict) else None
                    if bt and bt.strip() and bt.lower() != "sample":
                        self._book_id_title_map[bid_str] = bt.strip()
                        return bt.strip()
        except Exception:
            pass

        # 3. Check knowledge_graph.json
        try:
            kg_path = out_p / "knowledge_graph.json"
            if kg_path.is_file():
                with open(kg_path, "r", encoding="utf-8") as f:
                    kg_data = json.load(f)
                for node in kg_data.get("nodes", []):
                    n_bid = str(node.get("book_id", "")).replace("book:", "").strip()
                    n_bt = node.get("book_title")
                    if n_bid and n_bt and n_bt.strip() and n_bt.lower() != "sample":
                        self._book_id_title_map[n_bid] = n_bt.strip()
                if bid_str in self._book_id_title_map:
                    return self._book_id_title_map[bid_str]
        except Exception:
            pass

        # 4. Check Neo4j if available
        try:
            bid_num = int(bid_str)
            driver = self.get_neo4j_driver()
            with driver.session(database=self.neo4j_cfg.database) as s:
                rec = s.run(
                    "MATCH (c:Chunk) WHERE c.book_id = $bid AND c.book_title IS NOT NULL AND toLower(c.book_title) <> 'sample' RETURN c.book_title AS bt LIMIT 1",
                    {"bid": bid_num},
                ).single()
                if rec and rec["bt"]:
                    res_bt = rec["bt"].strip()
                    self._book_id_title_map[bid_str] = res_bt
                    return res_bt
        except Exception:
            pass

        return None

    def _load_books_index(self) -> Dict[str, Any]:
        """
        Load index.json mapping from all available books directories, caching it in memory.
        Returns a dict containing:
        - 'by_key': raw dict from index.json
        - 'by_norm_title': normalized_title -> entry
        - 'by_file_name': lowercase filename -> entry
        - 'entries': list of entries
        """
        books_dirs = self._get_all_books_dirs()
        if not books_dirs:
            b_dir = self._get_books_dir()
            if b_dir:
                books_dirs = [b_dir]
            else:
                return {"by_key": {}, "by_norm_title": {}, "by_file_name": {}, "entries": []}

        mtimes: List[Tuple[str, float]] = []
        for b in books_dirs:
            idx_file = b / "index.json"
            if idx_file.is_file():
                try:
                    mtimes.append((str(idx_file), idx_file.stat().st_mtime))
                except Exception:
                    pass
        current_mtimes = tuple(mtimes)
        if self._books_index_cache is not None and getattr(self, "_books_index_mtimes", None) == current_mtimes:
            return self._books_index_cache

        by_key: Dict[str, Any] = {}
        by_norm_title: Dict[str, Any] = {}
        by_file_name: Dict[str, Any] = {}
        entries: List[Dict[str, Any]] = []

        for b in books_dirs:
            idx_path = b / "index.json"
            if not idx_path.is_file():
                continue
            try:
                with open(idx_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                if not isinstance(raw_data, dict):
                    continue
                for key, val in raw_data.items():
                    if not isinstance(val, dict):
                        continue
                    entry = dict(val)
                    entry.setdefault("key", key)
                    entry["_dir"] = str(b)
                    by_key[key] = entry
                    title = entry.get("title", "")
                    if title:
                        norm_t = re.sub(r"[^a-z0-9]", "", title.lower())
                        if norm_t:
                            by_norm_title[norm_t] = entry
                    fpath = entry.get("file_path", "")
                    if fpath:
                        by_file_name[Path(fpath).name.lower()] = entry
                    entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to load books index.json from {idx_path}: {e}")

        cache = {
            "by_key": by_key,
            "by_norm_title": by_norm_title,
            "by_file_name": by_file_name,
            "entries": entries,
        }
        self._books_index_cache = cache
        self._books_index_mtimes = current_mtimes
        return cache

    def resolve_book_file(
        self,
        title: str = "",
        book_id: Optional[Any] = None,
        breadcrumb: str = "",
        filename: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a book's physical file on disk from books_dir (config.books_dir) or index.json.
        Matches by filename, exact/fuzzy book title, breadcrumb header, or book ID.
        Returns a dict with metadata, download/view URLs, or None if not found.
        """
        books_dirs = self._get_all_books_dirs()
        if not books_dirs:
            primary = self._get_books_dir()
            if primary:
                books_dirs = [primary]
            else:
                return None

        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower()) if s else ""

        idx = self._load_books_index()
        by_norm_title = idx.get("by_norm_title", {})
        by_file_name = idx.get("by_file_name", {})
        by_key = idx.get("by_key", {})
        entries = idx.get("entries", [])

        target_file: Optional[Path] = None
        matched_entry: Optional[Dict[str, Any]] = None

        # 1. Direct filename match across any discovered books directory
        if filename:
            clean_name = Path(filename).name
            for b_dir in books_dirs:
                cand = (b_dir / clean_name).resolve()
                if cand.is_file():
                    target_file = cand
                    matched_entry = by_file_name.get(cand.name.lower())
                    break

        # 2. Match by key / arxiv_id / book_id if provided in index.json
        if not target_file and book_id is not None:
            bid_str = str(book_id).strip()
            bid_clean = bid_str.replace("book:", "").strip()
            if bid_clean in by_key:
                matched_entry = by_key[bid_clean]
                fp = matched_entry.get("file_path")
                if fp:
                    entry_dir = matched_entry.get("_dir")
                    dirs_to_check = [Path(entry_dir)] if entry_dir else books_dirs
                    for b_dir in dirs_to_check:
                        cand = (b_dir / fp).resolve()
                        if cand.is_file():
                            target_file = cand
                            break

        # 3. Determine effective title (substituting real title if 'sample' or empty)
        effective_title = (title or "").strip()
        if not effective_title or effective_title.lower() in ("sample", "unknown", "untitled", "none"):
            if book_id is not None:
                real_title = self._get_book_title_by_id(book_id)
                if real_title:
                    effective_title = real_title

        # Check breadcrumb if still empty or sample
        if (not effective_title or effective_title.lower() in ("sample", "unknown", "untitled", "none")) and breadcrumb:
            parts = breadcrumb.split(" > ")
            if parts and parts[0].strip() and parts[0].strip().lower() not in ("sample", "unknown", "untitled", "none"):
                effective_title = parts[0].strip()

        # 4. Match by Title (only if effective_title is non-empty and not generic)
        if not target_file and effective_title and effective_title.lower() not in ("sample", "unknown", "untitled", "none"):
            norm_q = _norm(effective_title)
            # 4a. Exact normalized match in index.json
            if norm_q in by_norm_title:
                matched_entry = by_norm_title[norm_q]
                fp = matched_entry.get("file_path")
                if fp:
                    entry_dir = matched_entry.get("_dir")
                    dirs_to_check = [Path(entry_dir)] if entry_dir else books_dirs
                    for b_dir in dirs_to_check:
                        cand = (b_dir / fp).resolve()
                        if cand.is_file():
                            target_file = cand
                            break

            # 4b. Containment/substring match in index.json
            if not target_file and len(norm_q) >= 4:
                for entry in entries:
                    ent_norm = _norm(entry.get("title", ""))
                    if ent_norm and (norm_q in ent_norm or ent_norm in norm_q):
                        fp = entry.get("file_path")
                        if fp:
                            entry_dir = entry.get("_dir")
                            dirs_to_check = [Path(entry_dir)] if entry_dir else books_dirs
                            for b_dir in dirs_to_check:
                                cand = (b_dir / fp).resolve()
                                if cand.is_file():
                                    target_file = cand
                                    matched_entry = entry
                                    break
                        if target_file:
                            break

            # 4c. Direct file scan across all books_dirs if not found in index.json
            if not target_file and len(norm_q) >= 4:
                for b_dir in books_dirs:
                    for ext in ("*.pdf", "*.epub", "*.mobi", "*.azw3", "*.djvu", "*.txt"):
                        for f in b_dir.glob(ext):
                            norm_f = _norm(f.stem)
                            if norm_f and (norm_q in norm_f or norm_f in norm_q):
                                target_file = f.resolve()
                                matched_entry = by_file_name.get(f.name.lower())
                                break
                        if target_file:
                            break
                    if target_file:
                        break

        # 5. Online Neo4j fallback if still not found
        if not target_file and book_id is not None:
            try:
                bid_num = int(str(book_id).replace("book:", ""))
                driver = self.get_neo4j_driver()
                with driver.session(database=self.neo4j_cfg.database) as s:
                    rec = s.run(
                        "MATCH (c:Chunk) WHERE c.book_id = $bid AND c.book_title IS NOT NULL AND toLower(c.book_title) <> 'sample' RETURN c.book_title AS bt LIMIT 1",
                        {"bid": bid_num},
                    ).single()
                    if rec and rec["bt"] and rec["bt"] != effective_title:
                        return self.resolve_book_file(title=rec["bt"], filename=filename)
            except Exception:
                pass

        if not target_file or not target_file.is_file():
            return None

        rel_name = target_file.name
        safe_rel = quote(rel_name)
        fmt = target_file.suffix.lstrip(".").lower()
        file_size = target_file.stat().st_size
        book_title = (matched_entry.get("title") if matched_entry else None) or effective_title or target_file.stem
        external_url = matched_entry.get("url") if matched_entry else None
        arxiv_id = matched_entry.get("arxiv_id") if matched_entry else None

        return {
            "found": True,
            "filename": rel_name,
            "title": book_title,
            "format": fmt,
            "file_size": file_size,
            "download_url": f"/api/book/download?filename={safe_rel}",
            "view_url": f"/api/book/file?filename={safe_rel}",
            "external_url": external_url,
            "arxiv_id": arxiv_id,
            "absolute_path": str(target_file),
        }

    def _get_concept_embedding_index(self) -> ConceptEmbeddingIndex:
        """Lazy-load and cache the neural ConceptEmbeddingIndex."""
        if self._concept_embedding_index is not None:
            return self._concept_embedding_index

        cache_path = getattr(self.config, "resolved_output_dir", Path("output")) / "concept_embeddings.npz"

        # Determine Ollama endpoint from config
        ollama_url = getattr(self.config, "embedding_base_url", None)
        if not ollama_url or ollama_url == "pool":
            for s in getattr(self.config, "ollama_servers", []):
                if s.supports_embedding:
                    ollama_url = s.url
                    break
        if not ollama_url or ollama_url == "pool":
            ollama_url = getattr(self.config, "ollama_base_url", "http://192.168.50.15:11434")

        model_name = getattr(self.config, "embedding_model", "bge-m3:latest")

        index = ConceptEmbeddingIndex(cache_path=cache_path, ollama_url=ollama_url, model_name=model_name)
        if not index.load():
            logger.info("Concept embeddings cache not found on disk yet.")
        self._concept_embedding_index = index
        return self._concept_embedding_index

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

    def get_local_graph_stats(self, force_refresh: bool = False) -> Dict[str, int]:
        """
        Return high-level counts for the local knowledge base:
        books, chunks, ideas/concepts, total_nodes, total_edges.
        Caches results based on file modification time.
        """
        kg_path = getattr(self.config, "resolved_output_dir", None) or Path("output")
        kg_file = Path(kg_path).expanduser().resolve() / "knowledge_graph.json"

        if not kg_file.is_file():
            out_p = Path(kg_path).expanduser().resolve()
            book_states = list((out_p / ".book_states").glob("book_*_state.json")) if (out_p / ".book_states").is_dir() else []
            chunk_files = list((out_p / "chunks").glob("book_*_chunks.json")) if (out_p / "chunks").is_dir() else []
            return {
                "books": len(book_states),
                "chunks": len(chunk_files),
                "ideas": 0,
                "total_nodes": len(book_states) + len(chunk_files),
                "total_edges": 0,
            }

        try:
            mtime = kg_file.stat().st_mtime
            if not force_refresh and self._local_kg_stats_cache is not None and self._local_kg_mtime == mtime:
                return self._local_kg_stats_cache

            with open(kg_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            books = 0
            chunks = 0
            ideas = 0

            for n in nodes:
                ntype = n.get("type") or n.get("category") or ""
                if ntype == "Book" or n.get("tag") == "book":
                    books += 1
                elif ntype == "Chunk" or n.get("tag") == "chunk":
                    chunks += 1
                elif ntype in ("Concept", "Idea") or n.get("is_idea") or n.get("tag") == "idea":
                    ideas += 1

            stats = {
                "books": books,
                "chunks": chunks,
                "ideas": ideas,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            }
            self._local_kg_stats_cache = stats
            self._local_kg_mtime = mtime
            return stats
        except Exception as e:
            logger.warning(f"Error computing local knowledge graph stats from {kg_file}: {e}")
            return {"books": 0, "chunks": 0, "ideas": 0, "total_nodes": 0, "total_edges": 0}

    def get_neo4j_stats(self) -> Tuple[bool, Optional[Dict[str, int]], Optional[str]]:
        """
        Return (connected, stats_dict, error_message).
        Queries high-level node and relationship counts from Neo4j.
        """
        try:
            driver = self.get_neo4j_driver()
            with driver.session(database=self.neo4j_cfg.database) as session:
                q_nodes = """
                MATCH (n)
                RETURN
                    count(n) AS total_nodes,
                    count(CASE WHEN n:Book THEN 1 END) AS books,
                    count(CASE WHEN n:Chunk THEN 1 END) AS chunks,
                    count(CASE WHEN n:Idea OR n:Concept THEN 1 END) AS ideas
                """
                rec_nodes = session.run(q_nodes).single()
                if not rec_nodes:
                    return True, {"books": 0, "chunks": 0, "ideas": 0, "total_nodes": 0, "total_edges": 0}, None

                q_edges = "MATCH ()-[r]->() RETURN count(r) AS total_edges"
                rec_edges = session.run(q_edges).single()
                total_edges = rec_edges["total_edges"] if rec_edges else 0

                stats = {
                    "books": int(rec_nodes["books"] or 0),
                    "chunks": int(rec_nodes["chunks"] or 0),
                    "ideas": int(rec_nodes["ideas"] or 0),
                    "total_nodes": int(rec_nodes["total_nodes"] or 0),
                    "total_edges": int(total_edges or 0),
                }
                return True, stats, None
        except Exception as e:
            return False, None, str(e)

    def validate_database_consistency(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Compare high-level statistics between local knowledge base and connected Neo4j database.
        Returns consistency evaluation, discrepancies, and warnings.
        """
        local_stats = self.get_local_graph_stats(force_refresh=force_refresh)
        connected, neo4j_stats, error_msg = self.get_neo4j_stats()

        if not connected:
            return {
                "neo4j_connected": False,
                "is_consistent": True,  # Consistent in standalone offline mode
                "standalone_mode": True,
                "local_stats": local_stats,
                "neo4j_stats": None,
                "discrepancies": [],
                "warning": None,
                "error": error_msg,
            }

        discrepancies: List[str] = []

        # Check books
        if local_stats["books"] != neo4j_stats["books"]:
            diff = neo4j_stats["books"] - local_stats["books"]
            discrepancies.append(f"Books: local={local_stats['books']}, neo4j={neo4j_stats['books']} (diff: {diff:+d})")

        # Check chunks
        if local_stats["chunks"] != neo4j_stats["chunks"]:
            diff = neo4j_stats["chunks"] - local_stats["chunks"]
            discrepancies.append(f"Chunks: local={local_stats['chunks']:,}, neo4j={neo4j_stats['chunks']:,} (diff: {diff:+d})")

        # Check ideas / concepts
        if local_stats["ideas"] != neo4j_stats["ideas"]:
            diff = neo4j_stats["ideas"] - local_stats["ideas"]
            discrepancies.append(f"Ideas: local={local_stats['ideas']:,}, neo4j={neo4j_stats['ideas']:,} (diff: {diff:+d})")

        # Check total nodes if not already flagged
        if not discrepancies and local_stats["total_nodes"] != neo4j_stats["total_nodes"]:
            diff = neo4j_stats["total_nodes"] - local_stats["total_nodes"]
            discrepancies.append(f"Total Nodes: local={local_stats['total_nodes']:,}, neo4j={neo4j_stats['total_nodes']:,} (diff: {diff:+d})")

        is_consistent = len(discrepancies) == 0
        warning = None
        if not is_consistent:
            warning = (
                f"Neo4j database is out of sync with local knowledge base. "
                + "; ".join(discrepancies)
                + ". Run 'bookeeper build-graph --export-neo4j' or 'bookeeper rebuild-graph --export-neo4j' to synchronize."
            )

        return {
            "neo4j_connected": True,
            "is_consistent": is_consistent,
            "standalone_mode": False,
            "local_stats": local_stats,
            "neo4j_stats": neo4j_stats,
            "discrepancies": discrepancies,
            "warning": warning,
            "error": None,
        }

    def _get_concept_vectorizer(self) -> Optional[FastConceptVectorizer]:
        """Lazy-load and cache the FastConceptVectorizer using Neo4j or local knowledge_graph.json."""
        if self._concept_vectorizer is not None:
            return self._concept_vectorizer

        corpus: Dict[str, str] = {}
        # Try Neo4j first if available
        try:
            driver = self.get_neo4j_driver()
            with driver.session(database=self.neo4j_cfg.database) as session:
                res = session.run(
                    "MATCH (c) WHERE c:Concept OR c:Idea RETURN c.id AS id, coalesce(c.name, c.id) AS name, coalesce(c.brief_description, '') AS brief"
                )
                for r in res:
                    cid = r.get("id")
                    if cid:
                        name = r.get("name") or cid
                        brief = r.get("brief") or ""
                        corpus[cid] = f"{name} {brief}".strip()
        except Exception:
            pass

        # Fallback to local knowledge_graph.json
        if not corpus:
            kg_path = getattr(self.config, "resolved_output_dir", Path("output")) / "knowledge_graph.json"
            if kg_path.is_file():
                try:
                    with open(kg_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for n in data.get("nodes", []):
                        if n.get("type") in ("Concept", "Idea") or n.get("tag") == "idea" or "idea" in n.get("tags", []):
                            nid = n.get("id")
                            if nid:
                                name = n.get("name") or nid
                                brief = n.get("brief_description") or n.get("description") or ""
                                corpus[nid] = f"{name} {brief}".strip()
                except Exception as e:
                    logger.warning(f"Failed to read concepts for vectorizer from {kg_path}: {e}")

        if corpus:
            self._concept_vectorizer = FastConceptVectorizer(corpus)
            logger.info(f"Initialized FastConceptVectorizer with {len(corpus)} concepts.")
        return self._concept_vectorizer

    def extract_query_terms_hierarchical(self, query: str) -> Dict[str, Any]:
        """
        Derive hierarchically ranked candidate search terms from a user query:
        1. full_query: Normalized full multi-word query (question words stripped).
        2. phrase_terms: 2+ word phrases (quoted substrings, wikilinks, bigrams/trigrams).
        3. word_terms: Individual keywords (excluding generic stopwords).
        """
        raw_query = query.strip()
        if not raw_query:
            return {"full_query": "", "phrase_terms": [], "word_terms": []}

        clean_q = raw_query
        for pat in QUESTION_PREFIXES:
            clean_q = re.sub(pat, "", clean_q, flags=re.IGNORECASE).strip()
        clean_q = clean_q.rstrip("?.!").strip()
        full_query = clean_q or raw_query

        phrase_terms = []
        for match in re.findall(r"\"([^\"]+)\"", raw_query):
            m = match.strip()
            if len(m.split()) >= 2:
                phrase_terms.append(m)

        for match in re.findall(r"\[\[(.*?)\]\]", raw_query):
            m = match.strip()
            if len(m.split()) >= 2:
                phrase_terms.append(m)

        words = [w for w in re.findall(r"[\w-]+", full_query) if len(w) >= 2]
        if len(words) >= 2 and full_query.lower() not in [p.lower() for p in phrase_terms]:
            phrase_terms.insert(0, full_query)

        if len(words) >= 3:
            for i in range(len(words) - 2):
                tri = f"{words[i]} {words[i+1]} {words[i+2]}"
                if tri.lower() not in GENERIC_CONCEPT_DENOISE_SET:
                    phrase_terms.append(tri)
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bi = f"{words[i]} {words[i+1]}"
                if bi.lower() not in GENERIC_CONCEPT_DENOISE_SET:
                    phrase_terms.append(bi)

        seen_phrases = set()
        deduped_phrases = []
        for p in phrase_terms:
            p_norm = p.strip(" *\"'[]").strip()
            if p_norm and p_norm.lower() not in seen_phrases and len(p_norm) >= 3:
                seen_phrases.add(p_norm.lower())
                deduped_phrases.append(p_norm)

        seen_words = set()
        deduped_words = []
        for w in words:
            w_norm = w.strip(" *\"'[]").strip()
            if (
                w_norm
                and w_norm.lower() not in seen_words
                and w_norm.lower() not in GENERIC_CONCEPT_DENOISE_SET
                and len(w_norm) >= 2
            ):
                seen_words.add(w_norm.lower())
                deduped_words.append(w_norm)

        return {
            "full_query": full_query,
            "phrase_terms": deduped_phrases[:15],
            "word_terms": deduped_words[:15],
        }

    def extract_query_terms(self, query: str) -> List[str]:
        """
        Derive ranked candidate search terms from a user query:
        1. Exact query (stripped)
        2. Query with question words removed (e.g. 'What is a time horizon?' -> 'time horizon')
        3. Quoted substrings or wikilinks
        4. Meaningful n-grams and keywords (excluding generic stopwords)
        """
        raw_query = query.strip()
        if not raw_query:
            return []

        candidates = [raw_query]

        # 1. Clean question words
        clean_q = raw_query
        for pat in QUESTION_PREFIXES:
            clean_q = re.sub(pat, "", clean_q, flags=re.IGNORECASE).strip()
        clean_q = clean_q.rstrip("?.!").strip()
        if clean_q and clean_q.lower() != raw_query.lower():
            candidates.append(clean_q)

        # 2. Quoted phrases and wikilinks
        for match in re.findall(r"\"([^\"]+)\"", raw_query):
            m = match.strip()
            if len(m) >= 2:
                candidates.append(m)

        for match in re.findall(r"\[\[(.*?)\]\]", raw_query):
            m = match.strip()
            if len(m) >= 2:
                candidates.append(m)

        # 3. N-grams and words (2+ words if query has >= 3 words)
        words = [w for w in re.findall(r"[\w-]+", clean_q or raw_query) if len(w) >= 2]
        if len(words) > 1:
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if phrase.lower() not in GENERIC_CONCEPT_DENOISE_SET:
                    candidates.append(phrase)

        for w in words:
            if w.lower() not in GENERIC_CONCEPT_DENOISE_SET and len(w) >= 3:
                candidates.append(w)

        # Deduplicate while preserving rank
        seen = set()
        deduped = []
        for c in candidates:
            c_norm = c.strip(" *\"'[]").strip()
            if c_norm and c_norm.lower() not in seen and len(c_norm) >= 2:
                seen.add(c_norm.lower())
                deduped.append(c_norm)

        return deduped

    def _extract_candidate_terms(self, text: str, user_query: str = "") -> List[str]:
        """
        Extract candidate concept terms from free-form text or Markdown notes,
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

        filtered = []
        seen = set()
        for c in raw_candidates:
            c_norm = c.lower().strip(" *\"'[]").strip()
            if not c_norm or len(c_norm) < 2:
                continue

            if c_norm in GENERIC_CONCEPT_DENOISE_SET and c_norm not in user_query_clean:
                continue

            if "reference document" in c_norm or "document chunk" in c_norm:
                continue

            if c_norm not in seen:
                seen.add(c_norm)
                c_words = set(re.findall(r"\w+", c_norm))
                overlap = len(c_words.intersection(user_words))
                filtered.append((c, overlap))

        filtered.sort(key=lambda x: -x[1])
        return [term for term, _ in filtered]

    def _serialize_local_node(self, node: Dict[str, Any], is_seed: bool = False, relevance: float = 0.0) -> Dict[str, Any]:
        nid = node.get("id", "")
        ntype = node.get("type", "Concept")
        name = node.get("name") or node.get("title") or nid
        cat = node.get("category") or node.get("tag") or "General"
        labels = [ntype] + ([node.get("tag")] if node.get("tag") else [])

        if is_seed:
            node_type = "seed"
            tier = "primary"
            color = "#f59e0b"
            border_color = "#fde68a"
            border_width = 4
            shape = "ellipse"
            icon = "🧠"
            weight = 10.0
            size = int(max(52, min(68, 52 + (relevance / 100.0) * 16))) if relevance > 0 else 60
        elif "Book" in labels or cat == "Book":
            node_type = "book"
            tier = "secondary"
            color = "#2563eb"
            border_color = "#60a5fa"
            border_width = 2.5
            shape = "round-rectangle"
            icon = "📚"
            cat = "Book"
            raw_w = float(node.get("weight", 6.5))
            weight = round(min(8.0, max(5.0, raw_w)), 1)
            size = int(max(34, min(48, 28 + weight * 2.5)))
        elif "Section" in labels or cat == "Section":
            node_type = "section"
            tier = "context"
            color = "#0d9488"
            border_color = "#34d399"
            border_width = 2
            shape = "round-rectangle"
            icon = "📑"
            cat = "Section"
            raw_w = float(node.get("weight", 2.8))
            weight = round(min(4.5, max(2.0, raw_w)), 1)
            size = int(max(30, min(42, 24 + weight * 3.0)))
        elif "Chunk" in labels or cat == "Chunk":
            node_type = "chunk"
            tier = "context"
            color = "#334155"
            border_color = "#64748b"
            border_width = 1.5
            shape = "ellipse"
            icon = "📄"
            cat = "Chunk"
            raw_w = float(node.get("weight", 1.5))
            weight = round(min(3.0, max(1.0, raw_w)), 1)
            size = int(max(24, min(36, 20 + weight * 3.0)))
        elif "Idea" in labels or "Concept" in labels or cat in ("Idea", "Concept"):
            node_type = "concept"
            tier = "secondary"
            color = "#8b5cf6"
            border_color = "#c4b5fd"
            border_width = 2.5
            shape = "ellipse"
            icon = "💡"
            raw_w = float(node.get("weight", 6.0))
            weight = round(min(8.5, max(5.0, raw_w)), 1)
            size = int(max(42, min(58, 32 + weight * 3.0)))
        else:
            node_type = "other"
            tier = "context"
            color = "#475569"
            border_color = "#94a3b8"
            border_width = 2
            shape = "ellipse"
            icon = "🔹"
            weight = 2.0
            size = 20

        return {
            "id": nid,
            "name": name,
            "category": cat,
            "node_type": node_type,
            "shape": shape,
            "labels": labels,
            "tier": tier,
            "is_seed": is_seed,
            "weight": weight,
            "size": size,
            "color": color,
            "border_color": border_color,
            "border_width": border_width,
            "icon": icon,
            "relevance": relevance,
            "properties": node,
        }

    @staticmethod
    def _serialize_map_unquoted_keys(obj: Any) -> str:
        """Serialize a dict/structure into Cypher map syntax with unquoted keys."""
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                k_str = str(k)
                key_part = k_str if re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", k_str) else json.dumps(k_str, ensure_ascii=False)
                val_part = SearchEngineBackend._serialize_map_unquoted_keys(v)
                parts.append(f"{key_part}: {val_part}")
            return "{" + ", ".join(parts) + "}"
        elif isinstance(obj, list):
            return "[" + ", ".join(SearchEngineBackend._serialize_map_unquoted_keys(x) for x in obj) + "]"
        else:
            return json.dumps(obj, ensure_ascii=False)

    @staticmethod
    def format_browser_cypher(
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        use_nested_props: bool = True,
    ) -> str:
        """
        Format Cypher query and parameters for direct execution in Neo4j / Cypher browsers.

        Prepends parameter initialization before query with unquoted parameter names:
        :param obj => ({props: {key: ...}})

        If use_nested_props is True, rewrites $param occurrences to $obj.props.param.
        If use_nested_props is False, keeps $param occurrences untouched.
        """
        if not cypher:
            return ""
        query = cypher.strip()
        params = parameters or {}
        if use_nested_props:
            for k in params.keys():
                query = re.sub(rf"\${re.escape(k)}\b", f"$obj.props.{k}", query)
        if params:
            param_str = SearchEngineBackend._serialize_map_unquoted_keys(params)
            return f":param obj => ({{props: {param_str}}})\n\n{query}"
        return query

    def _query_local_knowledge_graph(
        self,
        query: str,
        candidate_terms: Optional[List[str]] = None,
        top_k: int = 40,
        graph_limit: int = 60,
        hierarchical_terms: Optional[Dict[str, Any]] = None,
        semantic_seed_ids: Optional[List[str]] = None,
        seed_ids: Optional[List[str]] = None,
        core_ideas: Optional[List[Dict[str, Any]]] = None,
        mode: str = "ideas",
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Query knowledge_graph.json locally when Neo4j is offline or returns empty results.
        Uses direct embedding seed ideas or hierarchical scoring fallback.
        Returns (nodes_map, edges_list, seed_nodes, seed_neighbors).
        """
        kg_path = getattr(self.config, "resolved_output_dir", Path("output")) / "knowledge_graph.json"
        if not kg_path.is_file():
            return {}, [], {}, {}

        try:
            with open(kg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read local knowledge_graph.json: {e}")
            return {}, [], {}, {}

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Index all nodes by id
        nodes_by_id: Dict[str, Dict[str, Any]] = {n.get("id"): n for n in nodes if n.get("id")}

        # Build adjacency
        adj: Dict[str, List[Dict[str, Any]]] = {}
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src and tgt:
                adj.setdefault(src, []).append(edge)
                adj.setdefault(tgt, []).append(edge)

        selected_seed_nodes = []
        seed_scores = {}
        if core_ideas:
            seed_scores = {c["id"]: c.get("priority_score", 100.0) for c in core_ideas}

        # 1. Direct embedding seed ideas if provided
        if seed_ids:
            for sid in seed_ids:
                if sid in nodes_by_id:
                    selected_seed_nodes.append(nodes_by_id[sid])

        # 2. Fallback to candidate term scoring if seed_ids produced no seeds
        if not selected_seed_nodes:
            h_terms = hierarchical_terms or self.extract_query_terms_hierarchical(query)
            full_q = h_terms.get("full_query", "").lower().strip()
            phrases = [p.lower().strip() for p in h_terms.get("phrase_terms", []) if p.strip()]
            words = [w.lower().strip() for w in h_terms.get("word_terms", []) if w.strip()]

            if semantic_seed_ids is None:
                vec = self._get_concept_vectorizer()
                if vec is not None:
                    try:
                        semantic_matches = vec.query(query, top_k=10, min_threshold=0.45)
                        semantic_seed_ids = [m[0] for m in semantic_matches]
                    except Exception:
                        semantic_seed_ids = []
                else:
                    semantic_seed_ids = []
            semantic_set = set(semantic_seed_ids)

            scored_seeds: List[Tuple[float, Dict[str, Any]]] = []
            for n in nodes:
                ntype = n.get("type", "")
                ntag = n.get("tag", "")
                tags = n.get("tags", [])
                is_concept = ntype in ("Concept", "Idea") or ntag == "idea" or "idea" in tags
                if not is_concept:
                    continue

                nid = (n.get("id") or "").lower()
                name = (n.get("name") or nid).lower()
                brief = (n.get("brief_description") or n.get("description") or "").lower()
                weight = float(n.get("weight", 5))

                full_score = 0.0
                if full_q and len(full_q) >= 2:
                    if name == full_q or nid == full_q:
                        full_score = 15000.0
                    elif name.startswith(full_q):
                        full_score = 9000.0
                    elif len(full_q) >= 3 and full_q in name:
                        full_score = 5000.0

                phrase_score = 0.0
                for p in phrases:
                    if name == p or nid == p:
                        phrase_score = max(phrase_score, 2500.0)
                    elif name.startswith(p):
                        phrase_score = max(phrase_score, 1200.0)
                    elif p in name:
                        phrase_score = max(phrase_score, 600.0)

                semantic_score = 4000.0 if n.get("id") in semantic_set else 0.0

                word_score = 0.0
                for w in words:
                    if name == w:
                        word_score = max(word_score, 50.0)
                    elif name.startswith(w):
                        word_score = max(word_score, 25.0)
                    elif w in name:
                        word_score = max(word_score, 15.0)

                match_score = full_score + phrase_score + semantic_score + word_score
                if match_score == 0.0 and len(full_q) >= 3 and (full_q in name or full_q in brief):
                    match_score = 30.0

                if match_score > 0.0:
                    scored_seeds.append((match_score + weight, n))

            scored_seeds.sort(key=lambda x: x[0], reverse=True)
            max_seeds = min(8, top_k)
            selected_seed_nodes = [item[1] for item in scored_seeds[:max_seeds]]

        nodes_map: Dict[str, Dict[str, Any]] = {}
        seed_nodes: Dict[str, Dict[str, Any]] = {}
        seed_neighbors: Dict[str, List[Dict[str, Any]]] = {}
        edges_list: List[Dict[str, Any]] = []

        for sn in selected_seed_nodes:
            sid = sn.get("id")
            rel_score = float(seed_scores.get(sid, 100.0))
            serialized_seed = self._serialize_local_node(sn, is_seed=True, relevance=rel_score)
            nodes_map[sid] = serialized_seed
            seed_nodes[sid] = serialized_seed
            seed_neighbors[sid] = []

            for edge in adj.get(sid, []):
                if len(edges_list) >= graph_limit:
                    break
                src_id = edge.get("source")
                tgt_id = edge.get("target")
                rel_type = edge.get("relation") or edge.get("type") or "RELATED_TO"
                is_out = (src_id == sid)
                nbr_id = tgt_id if is_out else src_id

                if nbr_id in nodes_by_id:
                    nbr_raw = nodes_by_id[nbr_id]
                    if nbr_id not in nodes_map:
                        nodes_map[nbr_id] = self._serialize_local_node(nbr_raw, is_seed=False)

                    edge_entry = {
                        "id": f"{src_id}->{rel_type}->{tgt_id}",
                        "source": src_id,
                        "target": tgt_id,
                        "type": rel_type,
                        "properties": {k: v for k, v in edge.items() if k not in ("source", "target", "relation", "type")},
                    }
                    edges_list.append(edge_entry)
                    seed_neighbors[sid].append({
                        "rel": rel_type,
                        "direction": "outgoing" if is_out else "incoming",
                        "neighbor": nodes_map[nbr_id],
                        "properties": edge_entry["properties"],
                    })

        # Deep / 2-hop mode expansion in local graph
        if mode in ("deep", "2hop"):
            first_hop_ids = [nid for nid in list(nodes_map.keys()) if nid not in seed_nodes]
            for fid in first_hop_ids:
                if len(edges_list) >= graph_limit:
                    break
                f_node = nodes_map.get(fid, {})
                f_labels = f_node.get("labels", [])
                if "Concept" in f_labels or "Idea" in f_labels:
                    for edge in adj.get(fid, []):
                        if len(edges_list) >= graph_limit:
                            break
                        src_id = edge.get("source")
                        tgt_id = edge.get("target")
                        rel_type = edge.get("relation") or edge.get("type") or "RELATED_TO"
                        is_out = (src_id == fid)
                        nbr_id = tgt_id if is_out else src_id
                        if nbr_id in nodes_by_id and nbr_id not in nodes_map:
                            nodes_map[nbr_id] = self._serialize_local_node(nodes_by_id[nbr_id], is_seed=False)
                            edges_list.append({
                                "id": f"{src_id}->{rel_type}->{tgt_id}",
                                "source": src_id,
                                "target": tgt_id,
                                "type": rel_type,
                                "properties": {k: v for k, v in edge.items() if k not in ("source", "target", "relation", "type")},
                            })

        filtered_nodes_map, filtered_edges_list, filtered_seed_neighbors = self._post_process_and_filter_subgraph(
            seed_nodes=seed_nodes,
            nodes_map=nodes_map,
            edges_list=edges_list,
            seed_neighbors=seed_neighbors,
            denoise_level="standard",
        )
        return filtered_nodes_map, filtered_edges_list, seed_nodes, filtered_seed_neighbors

    def _post_process_and_filter_subgraph(
        self,
        seed_nodes: Dict[str, Dict[str, Any]],
        nodes_map: Dict[str, Dict[str, Any]],
        edges_list: List[Dict[str, Any]],
        seed_neighbors: Dict[str, List[Dict[str, Any]]],
        denoise_level: str = "standard",
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Post-filter and evaluate weights for nodes and relationships.

        Ensures:
        1. Query-matched seed concepts are prioritized as Tier 1 (center of the graph, weight 10.0, size 60, amber).
        2. Connected Concepts and Books are preserved as Tier 2 (weight 6.0-8.5, size 36-42, distinct color).
        3. Excessive secondary container nodes (Chunks and Sections/Pages) are capped and post-filtered
           so they do not overload or hijack canvas space.
        4. Secondary/contextual nodes are assigned low weights (1.0-2.8), smaller sizes, and grayish colors.
        5. Every relationship receives an explicit evaluated weight (1.0-10.0) based on endpoint semantics.
        """
        if denoise_level in ("focused", "strict"):
            max_chunks_per_seed = 0
            max_sections_per_seed = 1
            max_total_chunks = 0
            max_total_sections = 2
        elif denoise_level in ("detailed", "relaxed"):
            max_chunks_per_seed = 4
            max_sections_per_seed = 4
            max_total_chunks = 15
            max_total_sections = 10
        else:  # standard
            max_chunks_per_seed = 2
            max_sections_per_seed = 2
            max_total_chunks = 6
            max_total_sections = 5

        # Build adjacency mapping for neighbors
        seed_to_chunks: Dict[str, List[str]] = {sid: [] for sid in seed_nodes}
        seed_to_sections: Dict[str, List[str]] = {sid: [] for sid in seed_nodes}
        core_allowed_nodes: Set[str] = set(seed_nodes.keys())

        # All Concepts, Ideas, and Books connected to seeds are kept
        for nid, node in nodes_map.items():
            if nid in seed_nodes:
                continue
            cat = node.get("category", "")
            labels = node.get("labels", [])
            if "Book" in labels or cat == "Book":
                core_allowed_nodes.add(nid)
            elif "Concept" in labels or "Idea" in labels or cat in ("Concept", "Idea"):
                core_allowed_nodes.add(nid)

        # Map edges to seed connections
        for edge in edges_list:
            src = edge["source"]
            tgt = edge["target"]
            for sid in seed_nodes:
                other_id = tgt if src == sid else (src if tgt == sid else None)
                if not other_id or other_id not in nodes_map:
                    continue
                other_node = nodes_map[other_id]
                other_cat = other_node.get("category", "")
                other_labels = other_node.get("labels", [])

                if "Chunk" in other_labels or other_cat == "Chunk":
                    if other_id not in seed_to_chunks[sid]:
                        seed_to_chunks[sid].append(other_id)
                elif "Section" in other_labels or other_cat == "Section":
                    if other_id not in seed_to_sections[sid]:
                        seed_to_sections[sid].append(other_id)

        # Select prioritized chunks and sections within limits
        kept_chunks: Set[str] = set()
        kept_sections: Set[str] = set()

        for sid in seed_nodes:
            chunks = seed_to_chunks.get(sid, [])
            for cid in chunks:
                if len([c for c in seed_to_chunks[sid] if c in kept_chunks]) < max_chunks_per_seed and len(kept_chunks) < max_total_chunks:
                    kept_chunks.add(cid)

            sections = seed_to_sections.get(sid, [])
            for sec_id in sections:
                if len([s for s in seed_to_sections[sid] if s in kept_sections]) < max_sections_per_seed and len(kept_sections) < max_total_sections:
                    kept_sections.add(sec_id)

        allowed_nodes: Set[str] = core_allowed_nodes | kept_chunks | kept_sections
        if not allowed_nodes and nodes_map:
            allowed_nodes = set(nodes_map.keys())

        # Filter nodes_map and ensure tier/weight/size/color are set
        filtered_nodes_map: Dict[str, Dict[str, Any]] = {}
        for nid, node in nodes_map.items():
            if nid not in allowed_nodes:
                continue

            labels = node.get("labels", [])
            cat = node.get("category", "")
            is_seed = nid in seed_nodes

            if is_seed:
                tier = "primary"
                weight = 10.0
                size = 60
                color = "#f59e0b"
                icon = "🧠"
            elif "Book" in labels or cat == "Book":
                tier = "secondary"
                weight = 6.5
                size = 38
                color = "#3b82f6"
                icon = "📚"
            elif "Concept" in labels or "Idea" in labels or cat in ("Concept", "Idea"):
                tier = "secondary"
                weight = round(min(8.5, max(6.0, float(node.get("properties", {}).get("weight", 6.5)))), 1)
                size = 42
                color = "#8b5cf6"
                icon = "💡"
            elif "Section" in labels or cat == "Section":
                tier = "context"
                weight = 2.8
                size = 22
                color = "#64748b"
                icon = "📑"
            elif "Chunk" in labels or cat == "Chunk":
                tier = "context"
                weight = 1.5
                size = 16
                color = "#475569"
                icon = "📄"
            else:
                tier = "context"
                weight = 2.0
                size = 20
                color = "#475569"
                icon = "🔹"

            node["tier"] = tier
            node["is_seed"] = is_seed
            node["weight"] = weight
            node["size"] = size
            node["color"] = color
            node["icon"] = icon
            filtered_nodes_map[nid] = node

        # Filter edges_list and assign edge weights
        filtered_edges_list: List[Dict[str, Any]] = []
        for edge in edges_list:
            src = edge["source"]
            tgt = edge["target"]
            if src not in filtered_nodes_map or tgt not in filtered_nodes_map:
                continue

            src_node = filtered_nodes_map[src]
            tgt_node = filtered_nodes_map[tgt]
            src_tier = src_node.get("tier", "context")
            tgt_tier = tgt_node.get("tier", "context")
            src_cat = src_node.get("category", "")
            tgt_cat = tgt_node.get("category", "")

            # Weight relationship based on connection semantics
            if src_tier in ("primary", "secondary") and tgt_tier in ("primary", "secondary") and "Book" not in (src_cat, tgt_cat):
                edge_weight = 9.0
                line_color = "#818cf8"
                opacity = 0.9
            elif "Book" in (src_cat, tgt_cat):
                edge_weight = 6.5
                line_color = "#3b82f6"
                opacity = 0.75
            elif "Section" in (src_cat, tgt_cat):
                edge_weight = 3.5
                line_color = "#475569"
                opacity = 0.55
            elif "Chunk" in (src_cat, tgt_cat):
                edge_weight = 1.8
                line_color = "#334155"
                opacity = 0.4
            else:
                edge_weight = 3.0
                line_color = "#475569"
                opacity = 0.5

            edge["weight"] = edge_weight
            edge["line_color"] = line_color
            edge["opacity"] = opacity
            if "properties" not in edge:
                edge["properties"] = {}
            edge["properties"]["weight"] = edge_weight

            filtered_edges_list.append(edge)

        # Update seed_neighbors to only reference filtered nodes
        filtered_seed_neighbors: Dict[str, List[Dict[str, Any]]] = {}
        for sid, neighbors in seed_neighbors.items():
            filtered_seed_neighbors[sid] = [
                n for n in neighbors
                if n.get("neighbor", {}).get("id") in filtered_nodes_map
            ]

        return filtered_nodes_map, filtered_edges_list, filtered_seed_neighbors



    def query_ideas(
        self,
        query: str,
        mode: str = "ideas",
        top_k: int = 40,
        chunk_top_k: int = 20,
        graph_limit: int = 60,
        denoise_level: str = "standard",
        similarity_threshold: float = 0.70,
        top_k_seeds: int = 6,
    ) -> Dict[str, Any]:
        """
        Execute Idea-Based Search directly against Neo4j Knowledge Graph:
        1. Extract candidate terms from query without external RAG dependencies.
        2. Match seed Idea/Concept nodes by relevance, significance weight, and degree penalty.
        3. Retrieve 1-hop (or 2-hop for deep mode) neighborhood (connected ideas, books, evidence).
        4. Synthesize structured markdown overview and capture complete observability telemetry.
        """
        t0 = time.perf_counter()
        if self._is_restoring:
            raise RuntimeError("Search is temporarily disabled while a backup restore operation is in progress.")
        candidate_terms = self.extract_query_terms(query)

        # 1. Extract core ideas using embedding cosine similarity * node weight
        emb_index = self._get_concept_embedding_index()
        core_ideas: List[Dict[str, Any]] = []
        embedding_telemetry: Dict[str, Any] = {}
        if emb_index.is_loaded():
            try:
                core_ideas, embedding_telemetry = emb_index.query_core_ideas_with_telemetry(
                    query=query,
                    similarity_threshold=similarity_threshold,
                    top_k=top_k_seeds,
                )
            except Exception as e:
                logger.warning(f"Neural embedding query failed: {e}")

        # Check direct Neo4j exact/prefix name and ID matches for Idea / Concept nodes
        try:
            driver = self.get_neo4j_driver()
            with driver.session(database=self.neo4j_cfg.database) as session:
                exact_res = session.run(
                    """
                    MATCH (n)
                    WHERE (n:Idea OR n:Concept) AND (
                        toLower(n.name) = toLower($query)
                        OR n.id = $query
                        OR toString(id(n)) = $query
                        OR elementId(n) ENDS WITH (':' + $query)
                    )
                    OPTIONAL MATCH (better:Idea)
                    WHERE toLower(better.name) = toLower(n.name)
                    WITH coalesce(better, n) AS best
                    ORDER BY (CASE WHEN best:Idea THEN 1 ELSE 2 END) ASC, coalesce(best.weight, 5) DESC
                    LIMIT 3
                    RETURN best.id AS id, best.name AS name, coalesce(best.weight, 8.0) AS weight
                    """,
                    {"query": query.strip()},
                )
                for r in exact_res:
                    xid = r.get("id")
                    if xid:
                        xw = float(r.get("weight") or 8.0)
                        exact_idea = {
                            "id": xid,
                            "name": r.get("name") or xid,
                            "weight": xw,
                            "cosine_similarity": 1.0,
                            "priority_score": round(1.0 * xw, 4),
                            "threshold_met": True,
                        }
                        core_ideas = [exact_idea] + [i for i in core_ideas if i["id"] != xid]
        except Exception as e:
            logger.debug(f"Direct Neo4j exact idea query failed: {e}")

        # Fallback to fast vectorizer if neural embeddings produced no matches
        if not core_ideas:
            vec = self._get_concept_vectorizer()
            if vec is not None:
                try:
                    fuzzy_matches = vec.query(query, top_k=top_k_seeds, min_threshold=0.40)
                    core_ideas = [
                        {
                            "id": m[0],
                            "name": m[0].split(":", 1)[-1],
                            "weight": 5.0,
                            "cosine_similarity": round(m[1], 4),
                            "priority_score": round(m[1] * 5.0, 4),
                            "threshold_met": bool(m[1] >= similarity_threshold),
                        }
                        for m in fuzzy_matches
                    ]
                except Exception as e:
                    logger.warning(f"Fuzzy vectorizer query failed: {e}")

        seed_ids = [idea["id"] for idea in core_ideas]
        seed_scores = {idea["id"]: idea.get("priority_score", 100.0) for idea in core_ideas}
        seed_cosines = {idea["id"]: idea.get("cosine_similarity", 1.0) for idea in core_ideas}
        max_seeds = min(top_k_seeds, len(seed_ids) or top_k_seeds)

        # 2. Build direct Cypher query targeting core seed ideas with canonical Idea upgrade
        if mode in ("deep", "2hop"):
            cypher_query = """
            // Step 1: Match core seed ideas directly by ID and upgrade to canonical :Idea node
            UNWIND $seed_ids AS sid
            MATCH (c_seed)
            WHERE c_seed.id = sid 
               OR elementId(c_seed) = sid 
               OR toString(id(c_seed)) = sid 
               OR elementId(c_seed) ENDS WITH (':' + sid)
               OR toLower(c_seed.name) = toLower(sid)
            OPTIONAL MATCH (better:Idea)
            WHERE toLower(better.name) = toLower(c_seed.name)
            WITH coalesce(better, c_seed) AS c
            ORDER BY (CASE WHEN c:Idea THEN 1 ELSE 2 END) ASC, coalesce(c.weight, 5) DESC
            WITH DISTINCT c
            LIMIT $max_seeds

            // Step 2: Fetch 1-hop and 2-hop neighborhood (prioritize concepts and books)
            OPTIONAL MATCH (c)-[rel1]-(neighbor)
            WHERE (neighbor:Concept OR neighbor:Idea OR neighbor:Book OR neighbor:Section OR neighbor:Chunk)
            WITH c, rel1, neighbor,
                 CASE
                   WHEN neighbor:Concept OR neighbor:Idea THEN 1
                   WHEN neighbor:Book THEN 2
                   WHEN neighbor:Section THEN 3
                   ELSE 4
                 END AS p1
            ORDER BY p1 ASC, coalesce(neighbor.weight, 5) DESC

            OPTIONAL MATCH (neighbor)-[rel2]-(hop2)
            WHERE (neighbor:Concept OR neighbor:Idea)
              AND (hop2:Concept OR hop2:Idea OR hop2:Book OR hop2:Section OR hop2:Chunk)
            RETURN c, rel1, neighbor, rel2, hop2
            LIMIT $graph_limit
            """
        else:
            cypher_query = """
            // Step 1: Match core seed ideas directly by ID and upgrade to canonical :Idea node
            UNWIND $seed_ids AS sid
            MATCH (c_seed)
            WHERE c_seed.id = sid 
               OR elementId(c_seed) = sid 
               OR toString(id(c_seed)) = sid 
               OR elementId(c_seed) ENDS WITH (':' + sid)
               OR toLower(c_seed.name) = toLower(sid)
            OPTIONAL MATCH (better:Idea)
            WHERE toLower(better.name) = toLower(c_seed.name)
            WITH coalesce(better, c_seed) AS c
            ORDER BY (CASE WHEN c:Idea THEN 1 ELSE 2 END) ASC, coalesce(c.weight, 5) DESC
            WITH DISTINCT c
            LIMIT $max_seeds

            // Step 2: Fetch surrounding 1-hop neighborhood (prioritize concepts and books)
            OPTIONAL MATCH (c)-[rel]-(neighbor)
            WHERE (neighbor:Concept OR neighbor:Idea OR neighbor:Book OR neighbor:Section OR neighbor:Chunk)
            WITH c, rel, neighbor,
                 CASE
                   WHEN neighbor:Concept OR neighbor:Idea THEN 1
                   WHEN neighbor:Book THEN 2
                   WHEN neighbor:Section THEN 3
                   ELSE 4
                 END AS neighbor_priority
            ORDER BY neighbor_priority ASC, coalesce(neighbor.weight, 5) DESC
            RETURN c, rel, neighbor
            LIMIT $graph_limit
            """

        cypher_params = {
            "seed_ids": seed_ids,
            "max_seeds": max_seeds,
            "similarity_threshold": similarity_threshold,
            "graph_limit": graph_limit,
        }

        telemetry: Dict[str, Any] = {
            "query": query,
            "mode": mode,
            "denoise_level": denoise_level,
            "similarity_threshold": similarity_threshold,
            "top_k_seeds": top_k_seeds,
            "candidate_terms": candidate_terms,
            "core_ideas": core_ideas,
            "embedding_debug": embedding_telemetry,
            "seed_ids": seed_ids,
            "timestamp": time.time(),
            "neo4j": {
                "exact_cypher": cypher_query.strip(),
                "parameters": cypher_params,
                "browser_cypher_props": self.format_browser_cypher(cypher_query, cypher_params, use_nested_props=True),
                "browser_cypher_param": self.format_browser_cypher(cypher_query, cypher_params, use_nested_props=False),
            },
        }

        driver = self.get_neo4j_driver()
        nodes_map: Dict[str, Dict[str, Any]] = {}
        edges_list: List[Dict[str, Any]] = []
        seed_nodes: Dict[str, Dict[str, Any]] = {}
        seed_neighbors: Dict[str, List[Dict[str, Any]]] = {}

        try:
            with driver.session(database=self.neo4j_cfg.database) as session:
                result = session.run(cypher_query, cypher_params)
                for record in result:
                    c_node = record.get("c")
                    rel = record.get("rel1") if record.get("rel1") is not None else record.get("rel")
                    neighbor = record.get("neighbor")
                    hop2_node = record.get("hop2")
                    rel2 = record.get("rel2")

                    cid = None
                    if c_node is not None:
                        cid = c_node.get("id", str(getattr(c_node, "element_id", "")))
                        if cid not in nodes_map:
                            rel_score = float(seed_scores.get(cid, 100.0))
                            serialized = self._serialize_node(c_node, is_seed=True, relevance=rel_score)
                            serialized["cosine_similarity"] = seed_cosines.get(cid, 1.0)
                            nodes_map[cid] = serialized
                            seed_nodes[cid] = serialized
                            seed_neighbors[cid] = []

                    nid = None
                    if neighbor is not None:
                        nid = neighbor.get("id", str(getattr(neighbor, "element_id", "")))
                        if nid not in nodes_map:
                            nodes_map[nid] = self._serialize_node(neighbor, is_seed=False)

                    if hop2_node is not None:
                        h2_id = hop2_node.get("id", str(getattr(hop2_node, "element_id", "")))
                        if h2_id not in nodes_map:
                            nodes_map[h2_id] = self._serialize_node(hop2_node, is_seed=False)

                    if rel is not None and c_node is not None and neighbor is not None and cid:
                        start_id = rel.start_node.get("id", str(getattr(rel.start_node, "element_id", "")))
                        end_id = rel.end_node.get("id", str(getattr(rel.end_node, "element_id", "")))
                        edge_obj = {
                            "id": f"{start_id}->{rel.type}->{end_id}",
                            "source": start_id,
                            "target": end_id,
                            "type": rel.type,
                            "properties": dict(rel),
                        }
                        edges_list.append(edge_obj)
                        seed_neighbors[cid].append({
                            "rel": rel.type,
                            "direction": "outgoing" if start_id == cid else "incoming",
                            "neighbor": nodes_map[nid],
                            "properties": dict(rel),
                        })

                    if rel2 is not None and neighbor is not None and hop2_node is not None:
                        start_id2 = rel2.start_node.get("id", str(getattr(rel2.start_node, "element_id", "")))
                        end_id2 = rel2.end_node.get("id", str(getattr(rel2.end_node, "element_id", "")))
                        edges_list.append({
                            "id": f"{start_id2}->{rel2.type}->{end_id2}",
                            "source": start_id2,
                            "target": end_id2,
                            "type": rel2.type,
                            "properties": dict(rel2),
                        })

            # Fallback search if candidate terms produced no seeds
            if not nodes_map:
                fallback_cypher = """
                MATCH (c)
                WHERE (c:Concept OR c:Idea)
                  AND (
                    toLower(coalesce(c.name, '')) CONTAINS toLower($query)
                    OR toLower(coalesce(c.brief_description, '')) CONTAINS toLower($query)
                    OR toLower(coalesce(c.description, '')) CONTAINS toLower($query)
                  )
                WITH c LIMIT 5
                OPTIONAL MATCH (c)-[rel]-(neighbor)
                WHERE (neighbor:Concept OR neighbor:Idea OR neighbor:Book OR neighbor:Section OR neighbor:Chunk)
                RETURN c, rel, neighbor LIMIT $graph_limit
                """
                fallback_params = {"query": query.strip(), "graph_limit": graph_limit}
                telemetry["neo4j"]["fallback_cypher"] = fallback_cypher.strip()
                telemetry["neo4j"]["fallback_parameters"] = fallback_params
                telemetry["neo4j"]["fallback_browser_cypher_props"] = self.format_browser_cypher(fallback_cypher, fallback_params, use_nested_props=True)
                telemetry["neo4j"]["fallback_browser_cypher_param"] = self.format_browser_cypher(fallback_cypher, fallback_params, use_nested_props=False)
                with driver.session(database=self.neo4j_cfg.database) as session:
                    res2 = session.run(fallback_cypher, fallback_params)
                    for record in res2:
                        c_node = record.get("c")
                        rel = record.get("rel")
                        neighbor = record.get("neighbor")
                        cid = None
                        if c_node is not None:
                            cid = c_node.get("id", str(getattr(c_node, "element_id", "")))
                            if cid not in nodes_map:
                                serialized = self._serialize_node(c_node, is_seed=True, relevance=100)
                                nodes_map[cid] = serialized
                                seed_nodes[cid] = serialized
                                seed_neighbors[cid] = []
                        if neighbor is not None:
                            nid = neighbor.get("id", str(getattr(neighbor, "element_id", "")))
                            if nid not in nodes_map:
                                nodes_map[nid] = self._serialize_node(neighbor, is_seed=False)
                        if rel is not None and c_node is not None and neighbor is not None and cid:
                            start_id = rel.start_node.get("id", str(getattr(rel.start_node, "element_id", "")))
                            end_id = rel.end_node.get("id", str(getattr(rel.end_node, "element_id", "")))
                            edges_list.append({
                                "id": f"{start_id}->{rel.type}->{end_id}",
                                "source": start_id,
                                "target": end_id,
                                "type": rel.type,
                                "properties": dict(rel),
                            })
                            seed_neighbors[cid].append({
                                "rel": rel.type,
                                "direction": "outgoing" if start_id == cid else "incoming",
                                "neighbor": nodes_map[nid],
                                "properties": dict(rel),
                            })

            # Post-filter and evaluate weights for nodes and relationships
            nodes_map, edges_list, seed_neighbors = self._post_process_and_filter_subgraph(
                seed_nodes=seed_nodes,
                nodes_map=nodes_map,
                edges_list=edges_list,
                seed_neighbors=seed_neighbors,
                denoise_level=denoise_level,
            )

            telemetry["denoise_summary"] = {
                "level": denoise_level,
                "description": "Pure Ideas & Books (Chunks stripped)" if denoise_level in ("focused", "strict") else ("Balanced Ideas + Evidence" if denoise_level == "standard" else "Full Evidence Provenance"),
                "chunks_count": len([n for n in nodes_map.values() if n.get("category") == "Chunk"]),
                "sections_count": len([n for n in nodes_map.values() if n.get("category") == "Section"]),
                "ideas_count": len([n for n in nodes_map.values() if n.get("category") not in ("Chunk", "Section", "Book")]),
                "books_count": len([n for n in nodes_map.values() if n.get("category") == "Book"]),
            }

            telemetry["neo4j"]["status"] = "success"
            telemetry["neo4j"]["nodes_count"] = len(nodes_map)
            telemetry["neo4j"]["edges_count"] = len(edges_list)
            telemetry["neo4j"]["seed_ideas"] = [n["name"] for n in seed_nodes.values()]

        except Exception as e:
            logger.info(f"Neo4j is unavailable ({e}). Falling back to local Knowledge Graph.")
            telemetry["neo4j"]["status"] = "offline"
            telemetry["neo4j"]["error"] = str(e)

        # Fallback to local knowledge_graph.json if Neo4j is offline or produced no nodes
        if not nodes_map:
            local_nodes_map, local_edges_list, local_seed_nodes, local_seed_neighbors = self._query_local_knowledge_graph(
                query=query,
                candidate_terms=candidate_terms,
                top_k=top_k,
                graph_limit=graph_limit,
                seed_ids=seed_ids,
                core_ideas=core_ideas,
                mode=mode,
            )
            if local_nodes_map:
                nodes_map = local_nodes_map
                edges_list = local_edges_list
                seed_nodes = local_seed_nodes
                seed_neighbors = local_seed_neighbors
                telemetry["local_graph_fallback"] = {
                    "status": "success",
                    "nodes_count": len(nodes_map),
                    "edges_count": len(edges_list),
                    "seed_ideas": [n["name"] for n in seed_nodes.values()],
                }

        elapsed = time.perf_counter() - t0
        telemetry["neo4j"]["duration_seconds"] = round(elapsed, 4)

        # Build structured markdown synthesis answer
        identified_concepts = [n["name"] for n in seed_nodes.values()]
        answer = self._generate_idea_synthesis(query, seed_nodes, seed_neighbors, nodes_map)

        # Determine primary selected node:
        # Strictly prioritize Concept / Idea nodes, falling back to Book, and NEVER select a Chunk or Section
        primary_selected_id = None
        for sid, sdata in seed_nodes.items():
            cat = sdata.get("category", "")
            labels = sdata.get("labels", [])
            if cat in ("Concept", "Idea") or any(l in ("Concept", "Idea") for l in labels):
                primary_selected_id = sid
                break
        if not primary_selected_id and seed_nodes:
            primary_selected_id = next(iter(seed_nodes.keys()))

        if not primary_selected_id and nodes_map:
            concept_candidates = [
                nid for nid, n in nodes_map.items()
                if n.get("category") in ("Concept", "Idea") or any(l in ("Concept", "Idea") for l in n.get("labels", []))
            ]
            if concept_candidates:
                primary_selected_id = max(concept_candidates, key=lambda nid: float(nodes_map[nid].get("weight", 0.0)))
            else:
                book_candidates = [
                    nid for nid, n in nodes_map.items()
                    if n.get("category") == "Book" or "Book" in n.get("labels", [])
                ]
                if book_candidates:
                    primary_selected_id = max(book_candidates, key=lambda nid: float(nodes_map[nid].get("weight", 0.0)))
                else:
                    primary_selected_id = next(iter(nodes_map.keys()))

        return {
            "query": query,
            "answer": answer,
            "identified_concepts": identified_concepts,
            "graph": {
                "nodes": list(nodes_map.values()),
                "edges": edges_list,
            },
            "selected_node_id": primary_selected_id,
            "telemetry": telemetry,
        }

    def _generate_idea_synthesis(
        self,
        query: str,
        seed_nodes: Dict[str, Dict[str, Any]],
        seed_neighbors: Dict[str, List[Dict[str, Any]]],
        nodes_map: Dict[str, Dict[str, Any]],
    ) -> str:
        """Construct structured markdown summary of retrieved ideas and their neighborhood notes."""
        if not seed_nodes:
            if nodes_map:
                return f"Retrieved {len(nodes_map)} nodes from the Knowledge Graph matching '{query}'."
            return f"No concepts matching '{query}' were found in the Knowledge Graph."

        lines = []
        for i, (cid, sdata) in enumerate(seed_nodes.items()):
            name = sdata.get("name", cid)
            cat = sdata.get("category", "General")
            props = sdata.get("properties", {})
            weight = props.get("weight") or sdata.get("weight", 5)
            desc = (
                props.get("brief_description")
                or props.get("description")
                or props.get("summary")
                or f"Core concept in category '{cat}'."
            )

            lines.append(f"### 💡 Idea: **{name}**")
            lines.append(f"*Category: {cat} | Significance Weight: {weight}/10*")
            lines.append(f"\n{desc}\n")

            # Extract neighborhood connections
            nbrs = seed_neighbors.get(cid, [])
            books = []
            concepts = []
            quotes = []

            for item in nbrs:
                nb = item.get("neighbor", {})
                labels = nb.get("labels", [])
                rel_type = item.get("rel", "")
                direction = item.get("direction", "outgoing")
                arrow = "→" if direction == "outgoing" else "←"
                nb_name = nb.get("name", nb.get("id", ""))
                rel_props = item.get("properties", {})

                if "Book" in labels:
                    books.append(f"**{nb_name}**")
                elif "Concept" in labels or "Idea" in labels:
                    concepts.append(f"{arrow} **{nb_name}** (`{rel_type}`)")
                elif "Chunk" in labels or "Section" in labels:
                    quote = rel_props.get("quote") or nb.get("properties", {}).get("text", "")
                    breadcrumb = rel_props.get("breadcrumb") or nb.get("properties", {}).get("breadcrumb", "")
                    if quote:
                        quotes.append((quote, breadcrumb))

            if books:
                lines.append(f"**Featured In Books:** {', '.join(sorted(set(books)))}")

            if concepts:
                unique_concepts = list(dict.fromkeys(concepts))[:6]
                lines.append(f"**Related Concepts:** {', '.join(unique_concepts)}")

            if quotes:
                lines.append("\n**Neighborhood Evidence & Notes:**")
                for q_text, q_bc in quotes[:2]:
                    clean_q = q_text.strip().replace("\n", " ")
                    if len(clean_q) > 280:
                        clean_q = clean_q[:280] + "..."
                    lines.append(f"> \"{clean_q}\"")
                    if q_bc:
                        lines.append(f"> *(Context: {q_bc})*")

            lines.append("\n---\n")

        return "\n".join(lines).strip().rstrip("---").strip()

    def query_api(
        self,
        query: str,
        mode: str = "ideas",
        top_k: int = 40,
        chunk_top_k: int = 20,
        graph_limit: int = 60,
        denoise_level: str = "standard",
        include_references: bool = True,
        include_chunk_content: bool = True,
    ) -> Dict[str, Any]:
        """
        LightRAG-compatible Query API endpoint.
        Returns the exact schema expected by LightRAG clients (/query):
        {
            "response": str,
            "references": [{"reference_id": str, "file_path": str, "content": [str]}],
            "response_time": float,
            "llm_generated": bool,
            "identified_concepts": [...],
            "graph": {"nodes": [...], "edges": [...]}
        }
        """
        t0 = time.perf_counter()
        search_res = self.query_ideas(
            query=query,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            graph_limit=graph_limit,
            denoise_level=denoise_level,
        )
        elapsed = time.perf_counter() - t0

        references: List[Dict[str, Any]] = []
        if include_references:
            seen_refs = set()
            for node in search_res["graph"]["nodes"]:
                labels = node.get("labels", [])
                props = node.get("properties", {})
                nid = node.get("id", "")

                if "Book" in labels and nid not in seen_refs:
                    seen_refs.add(nid)
                    title = props.get("title") or props.get("name") or nid
                    references.append({
                        "reference_id": nid,
                        "file_path": title,
                        "content": [props.get("description", "")] if (include_chunk_content and props.get("description")) else [],
                    })
                elif ("Chunk" in labels or "Section" in labels) and nid not in seen_refs:
                    seen_refs.add(nid)
                    src = props.get("book_title") or props.get("source") or props.get("title") or nid
                    txt = props.get("text") or props.get("quote") or props.get("description") or ""
                    references.append({
                        "reference_id": nid,
                        "file_path": src,
                        "content": [txt] if (include_chunk_content and txt) else [],
                    })

        return {
            "response": search_res["answer"],
            "references": references,
            "response_time": round(elapsed, 4),
            "llm_generated": False,
            "identified_concepts": search_res["identified_concepts"],
            "graph": search_res["graph"],
            "selected_node_id": search_res["selected_node_id"],
            "telemetry": search_res["telemetry"],
        }

    def get_indexed_books(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve list of indexed books in descending chronological order (most recent first).
        For each book, highlights its most important ideas based on significance score:
        score = occurrences in this book * weight.
        Falls back to local knowledge base / state checkpoints if Neo4j is offline or empty.
        """
        books: List[Dict[str, Any]] = []

        # 1. Try querying Neo4j if available
        try:
            driver = self.get_neo4j_driver()
            with driver.session(database=self.neo4j_cfg.database) as session:
                cypher = """
                MATCH (b:Book)
                OPTIONAL MATCH (b)-[r:EXPLORES_IDEA]->(c:Concept)
                WITH b, c, r,
                     coalesce(r.occurrences, 1) AS occ,
                     coalesce(c.weight, 5) AS wt
                WITH b,
                     collect(CASE WHEN c IS NOT NULL THEN {
                         name: coalesce(c.name, c.id),
                         id: c.id,
                         occurrences: occ,
                         weight: wt,
                         score: occ * wt,
                         category: coalesce(c.category, 'Idea')
                     } ELSE null END) AS raw_ideas
                RETURN b,
                       [i IN raw_ideas WHERE i IS NOT NULL] AS ideas
                """
                results = session.run(cypher)
                neo4j_records = list(results)

                if neo4j_records:
                    # Also load local state for timestamps and extra metadata if present
                    out_p = Path(getattr(self.config, "resolved_output_dir", Path("output"))).expanduser().resolve()
                    state_books: Dict[str, Dict[str, Any]] = {}
                    state_file = out_p / ".bookeeper_state.json"
                    if state_file.is_file():
                        try:
                            with open(state_file, "r", encoding="utf-8") as sf:
                                sdata = json.load(sf)
                            bg = sdata.get("operations", {}).get("build_graph", {})
                            for k, v in bg.items():
                                if isinstance(v, dict):
                                    state_books[str(k)] = v
                        except Exception:
                            pass

                    for record in neo4j_records:
                        b_node = record["b"]
                        b_props = dict(b_node.items()) if hasattr(b_node, "items") else {}
                        bid = str(b_props.get("book_id") or b_props.get("id", "")).replace("book:", "").strip()
                        title = b_props.get("title") or b_props.get("name") or f"Book {bid}"
                        author = b_props.get("author") or "Unknown"
                        summary = b_props.get("summary") or ""

                        # Resolve timestamp
                        ts = float(b_props.get("timestamp") or b_props.get("indexed_at") or 0.0)
                        status = "completed"
                        total_chunks = b_props.get("total_chunks", 0)

                        if bid in state_books:
                            sb = state_books[bid]
                            if not ts and sb.get("timestamp"):
                                ts = float(sb["timestamp"])
                            status = sb.get("status", status)
                            if sb.get("metadata", {}).get("total_chunks"):
                                total_chunks = sb["metadata"]["total_chunks"]

                        # Check per-book state file for exact timestamp if still 0
                        if not ts:
                            b_state_file = out_p / ".book_states" / f"book_{bid}_state.json"
                            if b_state_file.is_file():
                                try:
                                    with open(b_state_file, "r", encoding="utf-8") as bsf:
                                        bs_data = json.load(bsf)
                                    ts = float(bs_data.get("updated_at") or bs_data.get("created_at") or b_state_file.stat().st_mtime)
                                    if not total_chunks and bs_data.get("total_chunks"):
                                        total_chunks = bs_data.get("total_chunks")
                                except Exception:
                                    ts = b_state_file.stat().st_mtime

                        raw_ideas = record["ideas"] or []
                        # Deduplicate ideas by name
                        ideas_by_name: Dict[str, Dict[str, Any]] = {}
                        for item in raw_ideas:
                            name = item.get("name") or ""
                            if not name:
                                continue
                            if name not in ideas_by_name or item.get("score", 0) > ideas_by_name[name].get("score", 0):
                                ideas_by_name[name] = item

                        sorted_ideas = sorted(ideas_by_name.values(), key=lambda x: x.get("score", 0), reverse=True)

                        # Try resolving book file link
                        book_file = self.resolve_book_file(title=title, book_id=bid)
                        safe_book_file = {k: v for k, v in book_file.items() if k != "absolute_path"} if book_file else None

                        books.append({
                            "book_id": bid,
                            "id": f"book:{bid}",
                            "title": title,
                            "author": author,
                            "summary": summary,
                            "timestamp": ts,
                            "status": status,
                            "total_chunks": total_chunks,
                            "ideas_count": len(sorted_ideas),
                            "top_ideas": sorted_ideas[:10],
                            "book_file": safe_book_file,
                        })
        except Exception as e:
            logger.info(f"Neo4j get_indexed_books query skipped or failed ({e}). Checking local graph.")

        # 2. Local fallback if Neo4j returned no books or is offline
        if not books:
            out_p = Path(getattr(self.config, "resolved_output_dir", Path("output"))).expanduser().resolve()
            kg_path = out_p / "knowledge_graph.json"

            # Check state file
            state_books: Dict[str, Dict[str, Any]] = {}
            state_file = out_p / ".bookeeper_state.json"
            if state_file.is_file():
                try:
                    with open(state_file, "r", encoding="utf-8") as sf:
                        sdata = json.load(sf)
                    bg = sdata.get("operations", {}).get("build_graph", {})
                    for k, v in bg.items():
                        if isinstance(v, dict):
                            state_books[str(k)] = v
                except Exception:
                    pass

            nodes_by_id: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []
            if kg_path.is_file():
                try:
                    with open(kg_path, "r", encoding="utf-8") as f:
                        kg_data = json.load(f)
                    nodes_by_id = {n.get("id"): n for n in kg_data.get("nodes", []) if n.get("id")}
                    edges = kg_data.get("edges", [])
                except Exception as e:
                    logger.warning(f"Failed to read local knowledge_graph.json: {e}")

            # Identify Book nodes
            local_book_nodes = [
                n for n in nodes_by_id.values()
                if n.get("type") == "Book" or n.get("tag") == "book"
            ]

            # If no Book nodes in KG, see if we have books from state_books or .book_states
            book_ids_seen = set()
            for b_node in local_book_nodes:
                nid = b_node.get("id", "")
                bid = str(b_node.get("book_id") or nid).replace("book:", "").strip()
                book_ids_seen.add(bid)

                title = b_node.get("title") or b_node.get("name") or f"Book {bid}"
                author = b_node.get("author") or "Unknown"
                summary = b_node.get("summary") or ""

                # Ideas connected to this book
                ideas_map: Dict[str, Dict[str, Any]] = {}
                for edge in edges:
                    if edge.get("source") == nid and edge.get("relation") == "EXPLORES_IDEA":
                        cid = edge.get("target")
                        cnode = nodes_by_id.get(cid, {})
                        cname = cnode.get("name") or (cid.replace("concept:", "") if cid else "")
                        if not cname:
                            continue
                        occ = int(edge.get("occurrences", 1))
                        wt = float(cnode.get("weight", 5))
                        score = occ * wt
                        ideas_map[cname] = {
                            "name": cname,
                            "id": cid,
                            "occurrences": occ,
                            "weight": wt,
                            "score": score,
                            "category": cnode.get("category", "Idea"),
                        }

                # Also check chunk evidence linkages if EXPLORES_IDEA is sparse
                if not ideas_map:
                    for edge in edges:
                        if edge.get("relation") in ("SUPPORTED_BY", "SUPPORTS_IDEA"):
                            cid = edge.get("source") if edge.get("relation") == "SUPPORTED_BY" else edge.get("target")
                            chid = edge.get("target") if edge.get("relation") == "SUPPORTED_BY" else edge.get("source")
                            cnode = nodes_by_id.get(cid, {})
                            chnode = nodes_by_id.get(chid, {})
                            if str(chnode.get("book_id", "")).replace("book:", "").strip() == bid:
                                cname = cnode.get("name") or (cid.replace("concept:", "") if cid else "")
                                if cname:
                                    cur = ideas_map.get(cname, {
                                        "name": cname,
                                        "id": cid,
                                        "occurrences": 0,
                                        "weight": float(cnode.get("weight", 5)),
                                        "category": cnode.get("category", "Idea"),
                                    })
                                    cur["occurrences"] += 1
                                    cur["score"] = cur["occurrences"] * cur["weight"]
                                    ideas_map[cname] = cur

                sorted_ideas = sorted(ideas_map.values(), key=lambda x: x.get("score", 0), reverse=True)

                # Determine timestamp & status
                ts = float(b_node.get("timestamp") or 0.0)
                status = "completed"
                total_chunks = b_node.get("total_chunks", 0)

                if bid in state_books:
                    sb = state_books[bid]
                    if not ts and sb.get("timestamp"):
                        ts = float(sb["timestamp"])
                    status = sb.get("status", status)
                    if sb.get("metadata", {}).get("total_chunks"):
                        total_chunks = sb["metadata"]["total_chunks"]

                if not ts:
                    b_state_file = out_p / ".book_states" / f"book_{bid}_state.json"
                    if b_state_file.is_file():
                        try:
                            with open(b_state_file, "r", encoding="utf-8") as bsf:
                                bs_data = json.load(bsf)
                            ts = float(bs_data.get("updated_at") or bs_data.get("created_at") or b_state_file.stat().st_mtime)
                            if not total_chunks and bs_data.get("total_chunks"):
                                total_chunks = bs_data.get("total_chunks")
                        except Exception:
                            ts = b_state_file.stat().st_mtime

                book_file = self.resolve_book_file(title=title, book_id=bid)
                safe_book_file = {k: v for k, v in book_file.items() if k != "absolute_path"} if book_file else None

                books.append({
                    "book_id": bid,
                    "id": nid,
                    "title": title,
                    "author": author,
                    "summary": summary,
                    "timestamp": ts,
                    "status": status,
                    "total_chunks": total_chunks,
                    "ideas_count": len(sorted_ideas),
                    "top_ideas": sorted_ideas[:10],
                    "book_file": safe_book_file,
                })

            # Check if there are other completed books in state_books not in KG nodes
            for bid, sb in state_books.items():
                if bid in book_ids_seen:
                    continue
                if sb.get("status") not in ("completed", "partially_indexed"):
                    continue
                book_ids_seen.add(bid)
                title = sb.get("title") or f"Book {bid}"
                ts = float(sb.get("timestamp") or 0.0)
                total_chunks = sb.get("metadata", {}).get("total_chunks", 0)

                # Check per-book state for ideas
                ideas_map = {}
                b_state_file = out_p / ".book_states" / f"book_{bid}_state.json"
                if b_state_file.is_file():
                    try:
                        with open(b_state_file, "r", encoding="utf-8") as bsf:
                            bs_data = json.load(bsf)
                        if not ts:
                            ts = float(bs_data.get("updated_at") or bs_data.get("created_at") or b_state_file.stat().st_mtime)
                        if not total_chunks:
                            total_chunks = bs_data.get("total_chunks", 0)

                        for chunk in bs_data.get("processed_chunks", {}).values():
                            for c in chunk.get("extraction", {}).get("concepts", []):
                                cname = c.get("name")
                                if cname:
                                    cur = ideas_map.get(cname, {
                                        "name": cname,
                                        "id": f"concept:{cname}",
                                        "occurrences": 0,
                                        "weight": float(c.get("weight", 5)),
                                        "category": c.get("category", "Idea"),
                                    })
                                    cur["occurrences"] += 1
                                    cur["score"] = cur["occurrences"] * cur["weight"]
                                    ideas_map[cname] = cur
                    except Exception:
                        pass

                sorted_ideas = sorted(ideas_map.values(), key=lambda x: x.get("score", 0), reverse=True)
                book_file = self.resolve_book_file(title=title, book_id=bid)
                safe_book_file = {k: v for k, v in book_file.items() if k != "absolute_path"} if book_file else None

                books.append({
                    "book_id": bid,
                    "id": f"book:{bid}",
                    "title": title,
                    "author": "Unknown",
                    "summary": "",
                    "timestamp": ts,
                    "status": sb.get("status", "completed"),
                    "total_chunks": total_chunks,
                    "ideas_count": len(sorted_ideas),
                    "top_ideas": sorted_ideas[:10],
                    "book_file": safe_book_file,
                })

        # Sort descending by timestamp (most recent first)
        books.sort(key=lambda b: float(b.get("timestamp") or 0.0), reverse=True)

        if limit is not None and limit > 0:
            books = books[:limit]

        return books

    def prepare_full_backup(
        self,
        format_type: str = "zip",
        backup_type: str = "full",
        destination_dir: Optional[Path] = None,
        indexed_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Create and package a full or knowledge backup archive (.zip or .7z)
        containing graph knowledge artifacts, state files, and indexed source books.
        Returns backup metadata and download resolution info.
        """
        from bookeeper.backup import create_backup

        out_dir = Path(getattr(self.config, "resolved_output_dir", Path("output"))).expanduser().resolve()
        
        # Target backup storage folder (e.g. output/backups)
        if destination_dir is None:
            dest_dir = out_dir / "backups"
        else:
            dest_dir = Path(destination_dir).expanduser().resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Source books directory
        b_dir = self._get_books_dir()
        if not b_dir:
            candidate_dirs = self._get_all_books_dirs()
            b_dir = candidate_dirs[0] if candidate_dirs else None

        result = create_backup(
            source_dir=out_dir,
            destination=dest_dir,
            format_type=format_type,
            backup_type=backup_type,
            books_dir=b_dir,
            indexed_only=indexed_only,
            include_chunks=True,
            include_obsidian=True,
            include_states=True,
            include_lightrag=True,
        )

        dest_file = Path(result["destination"])
        download_url = f"/api/backup/download?filename={dest_file.name}"

        return {
            "success": True,
            "filename": dest_file.name,
            "file_size": result.get("compressed_bytes", dest_file.stat().st_size if dest_file.is_file() else 0),
            "uncompressed_bytes": result.get("uncompressed_bytes", 0),
            "total_files": result.get("total_files", 0),
            "backup_type": result.get("backup_type", backup_type),
            "format": result.get("format", format_type),
            "download_url": download_url,
            "absolute_path": str(dest_file),
            "created_at": datetime.datetime.now().isoformat(),
        }

    def restore_from_backup(
        self,
        archive_file: Path,
        clean: bool = False,
        export_neo4j: bool = True,
    ) -> Dict[str, Any]:
        """
        Unpack a backup archive, restore knowledge artifacts and source books,
        export/update the connected Neo4j database with the restored ConceptGraphStore,
        and re-run validation between local files and Neo4j.
        Blocks search queries during execution via self._is_restoring flag.
        """
        from bookeeper.backup import restore_backup
        from bookeeper.graph.neo4j_exporter import Neo4jExporter
        from bookeeper.graph.store import ConceptGraphStore

        with self._restore_lock:
            if self._is_restoring:
                raise RuntimeError("A restore operation is already in progress.")
            self._is_restoring = True
            self._restore_status = {
                "in_progress": True,
                "stage": "unpacking",
                "message": "Extracting backup archive...",
                "started_at": datetime.datetime.now().isoformat(),
                "completed_at": None,
                "result": None,
                "error": None,
            }

        try:
            out_dir = Path(getattr(self.config, "resolved_output_dir", Path("output"))).expanduser().resolve()
            b_dir = self._get_books_dir()
            if not b_dir:
                candidate_dirs = self._get_all_books_dirs()
                b_dir = candidate_dirs[0] if candidate_dirs else (out_dir / "downloads")

            # 1. Unpack backup files
            restore_result = restore_backup(
                archive_path=archive_file,
                target_dir=out_dir,
                books_dir=b_dir,
                clean=clean,
                dry_run=False,
            )

            # Invalidate all caches
            self._local_kg_stats_cache = None
            self._local_kg_mtime = 0.0
            self._books_index_cache = None
            self._books_index_mtime = 0.0
            self._concept_vectorizer = None
            self._concept_embedding_index = None

            # 2. Export / update Neo4j database with restored graph
            neo4j_export_result = None
            kg_path = out_dir / "knowledge_graph.json"
            if export_neo4j and kg_path.is_file():
                with self._restore_lock:
                    self._restore_status["stage"] = "exporting_neo4j"
                    self._restore_status["message"] = "Updating Neo4j database with restored knowledge graph..."

                try:
                    store = ConceptGraphStore()
                    store.load(kg_path)

                    neo_exp = Neo4jExporter.from_config(self.neo4j_cfg)
                    # Clean previous content if requested so Neo4j mirrors restored snapshot
                    neo4j_export_result = neo_exp.export(
                        store,
                        clean=clean,
                        show_progress=False,
                    )
                except Exception as e:
                    logger.warning(f"Neo4j export after restore encountered error: {e}")
                    neo4j_export_result = {"error": str(e), "success": False}

            # 3. Re-run validation: Neo4j vs local files
            with self._restore_lock:
                self._restore_status["stage"] = "validating"
                self._restore_status["message"] = "Validating database consistency (Neo4j vs local files)..."

            consistency_result = self.validate_database_consistency(force_refresh=True)

            final_result = {
                "success": True,
                "restored_files": restore_result.get("files_restored", 0),
                "restored_bytes": restore_result.get("bytes_restored", 0),
                "backup_type": restore_result.get("backup_type", "knowledge"),
                "target_dir": str(out_dir),
                "books_dir": str(b_dir) if b_dir else None,
                "neo4j_export": neo4j_export_result,
                "consistency": consistency_result,
                "completed_at": datetime.datetime.now().isoformat(),
            }

            with self._restore_lock:
                self._restore_status["in_progress"] = False
                self._restore_status["stage"] = "completed"
                self._restore_status["message"] = "Restore completed successfully."
                self._restore_status["completed_at"] = final_result["completed_at"]
                self._restore_status["result"] = final_result

            return final_result

        except Exception as e:
            logger.error(f"Restore failed: {e}", exc_info=True)
            with self._restore_lock:
                self._restore_status["in_progress"] = False
                self._restore_status["stage"] = "failed"
                self._restore_status["message"] = f"Restore failed: {e}"
                self._restore_status["error"] = str(e)
            raise
        finally:
            self._is_restoring = False

    def get_node_details(self, node_id: str) -> Dict[str, Any]:
        """
        Fetch full details for a selected node, including its properties and
        categorized relations for rendering like an Obsidian page.
        """
        cypher = """
        MATCH (target)
        WHERE target.id = $node_id 
           OR elementId(target) = $node_id 
           OR toString(id(target)) = $node_id 
           OR elementId(target) ENDS WITH (':' + $node_id)
           OR toLower(target.name) = toLower($node_id)
        WITH target
        ORDER BY (CASE WHEN target:Idea THEN 1 ELSE 2 END) ASC, coalesce(target.weight, 5) DESC
        LIMIT 1
        OPTIONAL MATCH (better:Idea)
        WHERE toLower(better.name) = toLower(target.name)
        WITH coalesce(better, target) AS n
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
                   other_category: other.category,
                   other_properties: properties(other)
               }) AS relations
        """
        node_params = {"node_id": node_id}
        telemetry = {
            "exact_cypher": cypher.strip(),
            "node_id": node_id,
            "parameters": node_params,
            "browser_cypher_props": self.format_browser_cypher(cypher, node_params, use_nested_props=True),
            "browser_cypher_param": self.format_browser_cypher(cypher, node_params, use_nested_props=False),
        }

        try:
            driver = self.get_neo4j_driver()
            with driver.session(database=self.neo4j_cfg.database) as session:
                result = session.run(cypher, node_params)
                record = result.single()
                if record and record["n"]:
                    n_node = record["n"]
                    raw_relations = record["relations"] or []

                    books = []
                    related_concepts = []
                    evidence = []
                    other_rels = []

                    for r in raw_relations:
                        if not r.get("other_id"):
                            continue
                        labels = r.get("other_labels", [])
                        name = r.get("other_name") or r.get("other_title") or r.get("other_id")
                        other_props = r.get("other_properties") or {}
                        rel_props = r.get("properties") or {}

                        # Full evidence quote / chunk text
                        chunk_text = (
                            other_props.get("text")
                            or other_props.get("content")
                            or other_props.get("description")
                            or rel_props.get("quote")
                            or other_props.get("quote")
                            or ""
                        )
                        chunk_quote = rel_props.get("quote") or other_props.get("quote") or chunk_text

                        breadcrumb_str = rel_props.get("breadcrumb") or other_props.get("breadcrumb") or ""
                        book_title_str = other_props.get("book_title") or other_props.get("source") or ""
                        page_val = other_props.get("page") or other_props.get("page_num") or rel_props.get("page")
                        other_bid = other_props.get("book_id") or r.get("other_id")

                        rel_entry = {
                            "rel_type": r.get("type"),
                            "direction": r.get("direction"),
                            "target_id": r.get("other_id"),
                            "target_name": name,
                            "target_labels": labels,
                            "target_category": r.get("other_category"),
                            "quote": chunk_quote,
                            "text": chunk_text,
                            "breadcrumb": breadcrumb_str,
                            "book_title": book_title_str,
                            "page": page_val,
                            "other_properties": other_props,
                        }

                        if "Book" in labels:
                            bf = self.resolve_book_file(
                                title=name or other_props.get("title") or book_title_str,
                                book_id=other_bid,
                            )
                            rel_entry["book_file"] = {k: v for k, v in bf.items() if k != "absolute_path"} if bf else None
                            books.append(rel_entry)
                        elif "Concept" in labels or "Idea" in labels:
                            related_concepts.append(rel_entry)
                        elif "Chunk" in labels or "Section" in labels:
                            bf = self.resolve_book_file(
                                title=book_title_str,
                                book_id=other_bid,
                                breadcrumb=breadcrumb_str,
                            )
                            rel_entry["book_file"] = {k: v for k, v in bf.items() if k != "absolute_path"} if bf else None
                            evidence.append(rel_entry)
                        else:
                            other_rels.append(rel_entry)

                    node_data = self._serialize_node(n_node)
                    node_labels = node_data.get("labels", [])
                    node_cat = node_data.get("category", "")
                    node_props = node_data.get("properties", {})
                    if "Book" in node_labels or node_cat == "Book":
                        bf = self.resolve_book_file(
                            title=node_data.get("name") or node_props.get("title"),
                            book_id=node_props.get("book_id") or node_data.get("id"),
                        )
                        node_data["book_file"] = {k: v for k, v in bf.items() if k != "absolute_path"} if bf else None
                    elif "Chunk" in node_labels or node_cat == "Chunk" or "Section" in node_labels:
                        bf = self.resolve_book_file(
                            title=node_props.get("book_title"),
                            book_id=node_props.get("book_id"),
                            breadcrumb=node_props.get("breadcrumb"),
                        )
                        node_data["book_file"] = {k: v for k, v in bf.items() if k != "absolute_path"} if bf else None

                    return {
                        "node": node_data,
                        "books": books,
                        "related_concepts": related_concepts,
                        "evidence": evidence,
                        "other_relations": other_rels,
                        "telemetry": telemetry,
                    }
        except Exception as e:
            logger.info(f"Neo4j node details lookup failed or offline ({e}). Checking local graph.")

        # Local fallback if Neo4j is offline or returns not found
        kg_path = getattr(self.config, "resolved_output_dir", Path("output")) / "knowledge_graph.json"
        if kg_path.is_file():
            try:
                with open(kg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                nodes = {n.get("id"): n for n in data.get("nodes", []) if n.get("id")}
                if node_id in nodes:
                    n_raw = nodes[node_id]
                    books = []
                    related_concepts = []
                    evidence = []
                    other_rels = []
                    for edge in data.get("edges", []):
                        src = edge.get("source")
                        tgt = edge.get("target")
                        if src == node_id or tgt == node_id:
                            other_id = tgt if src == node_id else src
                            direction = "outgoing" if src == node_id else "incoming"
                            other_node = nodes.get(other_id, {})
                            labels = [other_node.get("type", "Concept")] + ([other_node.get("tag")] if other_node.get("tag") else [])
                            name = other_node.get("name") or other_node.get("title") or other_id
                            rel_type = edge.get("relation") or edge.get("type") or "RELATED_TO"
                            rel_props = {k: v for k, v in edge.items() if k not in ("source", "target", "relation", "type")}
                            chunk_text = other_node.get("text") or other_node.get("content") or other_node.get("description") or rel_props.get("quote", "")

                            breadcrumb_str = rel_props.get("breadcrumb") or other_node.get("breadcrumb", "")
                            book_title_str = other_node.get("book_title") or other_node.get("source", "")
                            other_bid = other_node.get("book_id") or other_id

                            rel_entry = {
                                "rel_type": rel_type,
                                "direction": direction,
                                "target_id": other_id,
                                "target_name": name,
                                "target_labels": labels,
                                "target_category": other_node.get("category") or other_node.get("tag"),
                                "quote": rel_props.get("quote") or chunk_text,
                                "text": chunk_text,
                                "breadcrumb": breadcrumb_str,
                                "book_title": book_title_str,
                                "page": other_node.get("page"),
                                "other_properties": other_node,
                            }
                            if "Book" in labels or other_node.get("category") == "Book":
                                bf = self.resolve_book_file(
                                    title=name or other_node.get("title") or book_title_str,
                                    book_id=other_bid,
                                )
                                rel_entry["book_file"] = {k: v for k, v in bf.items() if k != "absolute_path"} if bf else None
                                books.append(rel_entry)
                            elif "Concept" in labels or "Idea" in labels or other_node.get("category") in ("Concept", "Idea"):
                                related_concepts.append(rel_entry)
                            elif "Chunk" in labels or "Section" in labels or other_node.get("category") in ("Chunk", "Section"):
                                bf = self.resolve_book_file(
                                    title=book_title_str,
                                    book_id=other_bid,
                                    breadcrumb=breadcrumb_str,
                                )
                                rel_entry["book_file"] = {k: v for k, v in bf.items() if k != "absolute_path"} if bf else None
                                evidence.append(rel_entry)
                            else:
                                other_rels.append(rel_entry)

                    local_node = self._serialize_local_node(n_raw)
                    l_labels = local_node.get("labels", [])
                    l_cat = local_node.get("category", "")
                    l_props = local_node.get("properties", {})
                    if "Book" in l_labels or l_cat == "Book":
                        bf = self.resolve_book_file(
                            title=local_node.get("name") or l_props.get("title"),
                            book_id=l_props.get("book_id") or local_node.get("id"),
                        )
                        local_node["book_file"] = {k: v for k, v in bf.items() if k != "absolute_path"} if bf else None
                    elif "Chunk" in l_labels or l_cat == "Chunk" or "Section" in l_labels:
                        bf = self.resolve_book_file(
                            title=l_props.get("book_title"),
                            book_id=l_props.get("book_id"),
                            breadcrumb=l_props.get("breadcrumb"),
                        )
                        local_node["book_file"] = {k: v for k, v in bf.items() if k != "absolute_path"} if bf else None

                    return {
                        "node": local_node,
                        "books": books,
                        "related_concepts": related_concepts,
                        "evidence": evidence,
                        "other_relations": other_rels,
                        "telemetry": telemetry,
                    }
            except Exception as e:
                logger.warning(f"Local node details lookup error: {e}")

        return {"error": f"Node '{node_id}' not found.", "telemetry": telemetry}

    def expand_node_neighborhood(
        self,
        node_id: str,
        type_filter: str = "all",
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Pull undirected 1-hop neighborhood for a specific node on demand (Neo4j Browser-like expansion).
        Supported type filters: 'all', 'Concept', 'Book', 'Chunk', 'Section'.
        Returns serialized nodes and edges in undirected order.
        """
        cypher_query = """
        MATCH (target)
        WHERE target.id = $node_id 
           OR elementId(target) = $node_id 
           OR toString(id(target)) = $node_id 
           OR elementId(target) ENDS WITH (':' + $node_id)
           OR toLower(target.name) = toLower($node_id)
        WITH target
        ORDER BY (CASE WHEN target:Idea THEN 1 ELSE 2 END) ASC, coalesce(target.weight, 5) DESC
        LIMIT 1
        OPTIONAL MATCH (better:Idea)
        WHERE toLower(better.name) = toLower(target.name)
        WITH coalesce(better, target) AS n
        OPTIONAL MATCH (n)-[rel]-(neighbor)
        WHERE neighbor IS NOT NULL AND (
           $type_filter = 'all'
           OR $type_filter IN labels(neighbor)
           OR ($type_filter = 'Concept' AND (neighbor:Concept OR neighbor:Idea))
           OR ($type_filter = 'Book' AND neighbor:Book)
           OR ($type_filter = 'Chunk' AND neighbor:Chunk)
           OR ($type_filter = 'Section' AND neighbor:Section)
        )
        WITH n, rel, neighbor,
             CASE
               WHEN neighbor:Concept OR neighbor:Idea THEN 1
               WHEN neighbor:Book THEN 2
               WHEN neighbor:Section THEN 3
               WHEN neighbor:Chunk THEN 4
               ELSE 5
             END AS neighbor_priority
        ORDER BY neighbor_priority ASC, coalesce(neighbor.weight, 5) DESC
        RETURN n, rel, neighbor
        LIMIT $limit
        """
        cypher_params = {
            "node_id": node_id,
            "type_filter": type_filter,
            "limit": limit,
        }

        telemetry = {
            "node_id": node_id,
            "type_filter": type_filter,
            "limit": limit,
            "exact_cypher": cypher_query.strip(),
            "parameters": cypher_params,
            "browser_cypher_props": self.format_browser_cypher(cypher_query, cypher_params, use_nested_props=True),
            "browser_cypher_param": self.format_browser_cypher(cypher_query, cypher_params, use_nested_props=False),
        }

        nodes_map: Dict[str, Dict[str, Any]] = {}
        edges_list: List[Dict[str, Any]] = []

        try:
            driver = self.get_neo4j_driver()
            with driver.session(database=self.neo4j_cfg.database) as session:
                result = session.run(cypher_query, cypher_params)
                for record in result:
                    n_node = record.get("n")
                    rel = record.get("rel")
                    neighbor = record.get("neighbor")

                    if n_node is not None:
                        nid = n_node.get("id", str(getattr(n_node, "element_id", "")))
                        if nid and nid not in nodes_map:
                            nodes_map[nid] = self._serialize_node(n_node, is_seed=False)

                    if neighbor is not None:
                        nbr_id = neighbor.get("id", str(getattr(neighbor, "element_id", "")))
                        if nbr_id and nbr_id not in nodes_map:
                            nodes_map[nbr_id] = self._serialize_node(neighbor, is_seed=False)

                    if rel is not None and n_node is not None and neighbor is not None:
                        start_id = rel.start_node.get("id", str(getattr(rel.start_node, "element_id", "")))
                        end_id = rel.end_node.get("id", str(getattr(rel.end_node, "element_id", "")))
                        edges_list.append({
                            "id": f"{start_id}->{rel.type}->{end_id}",
                            "source": start_id,
                            "target": end_id,
                            "type": rel.type,
                            "properties": dict(rel),
                        })

            if nodes_map:
                return {
                    "node_id": node_id,
                    "type_filter": type_filter,
                    "nodes": list(nodes_map.values()),
                    "edges": edges_list,
                    "total_nodes": len(nodes_map),
                    "total_edges": len(edges_list),
                    "telemetry": telemetry,
                }
        except Exception as e:
            logger.info(f"Neo4j expansion failed or offline ({e}). Checking local graph.")

        # Local fallback if Neo4j is offline or returns not found
        kg_path = getattr(self.config, "resolved_output_dir", Path("output")) / "knowledge_graph.json"
        if kg_path.is_file():
            try:
                with open(kg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_nodes = {n.get("id"): n for n in data.get("nodes", []) if n.get("id")}
                if node_id in all_nodes:
                    center_node = all_nodes[node_id]
                    nodes_map[node_id] = self._serialize_local_node(center_node)

                    for edge in data.get("edges", []):
                        src = edge.get("source")
                        tgt = edge.get("target")
                        if src == node_id or tgt == node_id:
                            other_id = tgt if src == node_id else src
                            other_node = all_nodes.get(other_id)
                            if not other_node:
                                continue

                            labels = [other_node.get("type", "Concept")]
                            if other_node.get("tag"):
                                labels.append(other_node.get("tag"))

                            # Check type_filter
                            matches = False
                            if type_filter == "all":
                                matches = True
                            elif type_filter == "Concept" and any(l in ("Concept", "Idea") for l in labels):
                                matches = True
                            elif type_filter == "Book" and ("Book" in labels or other_node.get("category") == "Book"):
                                matches = True
                            elif type_filter == "Chunk" and ("Chunk" in labels or other_node.get("category") == "Chunk"):
                                matches = True
                            elif type_filter == "Section" and ("Section" in labels or other_node.get("category") == "Section"):
                                matches = True
                            elif type_filter in labels:
                                matches = True

                            if matches:
                                if other_id not in nodes_map:
                                    nodes_map[other_id] = self._serialize_local_node(other_node)
                                rel_type = edge.get("relation") or edge.get("type") or "RELATED_TO"
                                edges_list.append({
                                    "id": f"{src}->{rel_type}->{tgt}",
                                    "source": src,
                                    "target": tgt,
                                    "type": rel_type,
                                    "properties": {k: v for k, v in edge.items() if k not in ("source", "target", "relation", "type")},
                                })
                                if len(nodes_map) >= limit + 1:
                                    break

                    return {
                        "node_id": node_id,
                        "type_filter": type_filter,
                        "nodes": list(nodes_map.values()),
                        "edges": edges_list,
                        "total_nodes": len(nodes_map),
                        "total_edges": len(edges_list),
                        "telemetry": telemetry,
                    }
            except Exception as e:
                logger.warning(f"Local expansion fallback error: {e}")

        return {
            "node_id": node_id,
            "type_filter": type_filter,
            "nodes": [],
            "edges": [],
            "total_nodes": 0,
            "total_edges": 0,
            "telemetry": telemetry,
        }

    def _serialize_node(self, node: Any, is_seed: bool = False, relevance: float = 0.0) -> Dict[str, Any]:
        """Convert a Neo4j Node into a JSON-serializable dictionary with display formatting and weight system."""
        labels = list(node.labels) if hasattr(node, "labels") else []
        props = dict(node)
        nid = props.get("id", str(getattr(node, "element_id", "unknown")))
        name = props.get("name") or props.get("title") or nid

        category = props.get("category", "General")

        if is_seed:
            node_type = "seed"
            tier = "primary"
            color = "#f59e0b"
            border_color = "#fde68a"
            border_width = 4
            shape = "ellipse"
            icon = "🧠"
            weight = 10.0
            size = int(max(52, min(68, 52 + (relevance / 100.0) * 16))) if relevance > 0 else 60
        elif "Book" in labels:
            node_type = "book"
            tier = "secondary"
            color = "#2563eb"
            border_color = "#60a5fa"
            border_width = 2.5
            shape = "round-rectangle"
            icon = "📚"
            category = "Book"
            raw_w = float(props.get("weight", 6.5))
            weight = round(min(8.0, max(5.0, raw_w)), 1)
            size = int(max(34, min(48, 28 + weight * 2.5)))
        elif "Section" in labels:
            node_type = "section"
            tier = "context"
            color = "#0d9488"
            border_color = "#34d399"
            border_width = 2
            shape = "round-rectangle"
            icon = "📑"
            category = "Section"
            raw_w = float(props.get("weight", 2.8))
            weight = round(min(4.5, max(2.0, raw_w)), 1)
            size = int(max(30, min(42, 24 + weight * 3.0)))
        elif "Chunk" in labels:
            node_type = "chunk"
            tier = "context"
            color = "#334155"
            border_color = "#64748b"
            border_width = 1.5
            shape = "ellipse"
            icon = "📄"
            category = "Chunk"
            raw_w = float(props.get("weight", 1.5))
            weight = round(min(3.0, max(1.0, raw_w)), 1)
            size = int(max(24, min(36, 20 + weight * 3.0)))
        elif "Idea" in labels or "Concept" in labels:
            node_type = "concept"
            tier = "secondary"
            color = "#8b5cf6"
            border_color = "#c4b5fd"
            border_width = 2.5
            shape = "ellipse"
            icon = "💡"
            raw_w = float(props.get("weight", 6.0))
            weight = round(min(8.5, max(5.0, raw_w)), 1)
            size = int(max(42, min(58, 32 + weight * 3.0)))
        else:
            node_type = "other"
            tier = "context"
            color = "#475569"
            border_color = "#94a3b8"
            border_width = 2
            shape = "ellipse"
            icon = "🔹"
            weight = 2.0
            size = 20

        return {
            "id": nid,
            "name": name,
            "labels": labels,
            "category": category,
            "node_type": node_type,
            "shape": shape,
            "tier": tier,
            "is_seed": is_seed,
            "weight": weight,
            "size": size,
            "color": color,
            "border_color": border_color,
            "border_width": border_width,
            "icon": icon,
            "relevance": relevance,
            "occurrences": props.get("occurrences", 1),
            "properties": props,
        }
