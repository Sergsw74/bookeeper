"""LightRAG engine integration for fast graph-enhanced indexing and dual-level retrieval."""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from bookeeper.config import Settings
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.extractor import OllamaPool

logger = logging.getLogger(__name__)


class LightRAGEngine:
    """
    LightRAG engine wrapper configured with distributed Ollama pool
    for fast incremental graph indexing and dual-level (local/global/hybrid) retrieval.
    """

    def __init__(
        self,
        working_dir: Union[Path, str],
        pool: OllamaPool,
        llm_model: str = "llama3.1:8b",
        embedding_model: str = "nomic-embed-text",
        embedding_dim: int = 768,
        max_token_size: int = 8192,
        embedding_pool: Optional[OllamaPool] = None,
        max_parallel_insert: int = 6,
        llm_max_async: int = 16,
        embedding_batch_num: int = 64,
        embedding_max_async: int = 32,
        keep_alive: str = "-1",
    ):
        self.working_dir = Path(working_dir).expanduser().resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.pool = pool
        self.embedding_pool = embedding_pool or pool
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.max_token_size = max_token_size
        self.max_parallel_insert = max_parallel_insert
        self.llm_max_async = llm_max_async
        self.embedding_batch_num = embedding_batch_num
        self.embedding_max_async = embedding_max_async
        self.keep_alive = keep_alive

        self._rag: Optional[LightRAG] = None
        self._initialized = False

    def clean(self) -> None:
        """Purge all stored LightRAG graph, vector, and index files in working_dir."""
        if self.working_dir.exists():
            for item in self.working_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    try:
                        item.unlink(missing_ok=True)
                    except Exception:
                        pass
        self._rag = None
        self._initialized = False

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        pool: Optional[OllamaPool] = None,
        working_dir: Optional[Union[Path, str]] = None,
        embedding_pool: Optional[OllamaPool] = None,
    ) -> "LightRAGEngine":
        """Instantiate LightRAGEngine from application Settings."""
        target_dir = working_dir or settings.resolved_lightrag_dir
        cd = getattr(settings, "resolved_analysis_cooldown_seconds", getattr(settings, "failover_cooldown_seconds", 600))
        if pool is None:
            pool = OllamaPool(
                servers=settings.resolved_ollama_servers,
                cooldown_seconds=cd,
                max_tasks_per_server=4,
            )
        if embedding_pool is None:
            # Raise concurrency limit for embedding workers to match OLLAMA_NUM_PARALLEL
            embedding_pool = OllamaPool(
                servers=settings.resolved_embedding_servers,
                cooldown_seconds=cd,
                max_tasks_per_server=8,
            )
        return cls(
            working_dir=target_dir,
            pool=pool,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.lightrag_embedding_dim,
            embedding_pool=embedding_pool,
            max_parallel_insert=getattr(settings, "lightrag_max_parallel_insert", 6),
            llm_max_async=getattr(settings, "lightrag_llm_max_async", 16),
            embedding_batch_num=getattr(settings, "lightrag_embedding_batch_num", 64),
            embedding_max_async=getattr(settings, "lightrag_embedding_max_async", 32),
            keep_alive=getattr(settings, "lightrag_ollama_keep_alive", "-1"),
        )

    def _call_ollama(
        self,
        endpoint: str,
        data: Dict[str, Any],
        timeout: int = 120,
        use_embedding_pool: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute request against Ollama pool with automatic failover and curl fallback
        to ensure network resilience on macOS Sequoia Local Network Privacy.
        """
        def _invoke(base_url: str) -> Dict[str, Any]:
            url = f"{base_url.rstrip('/')}{endpoint}"
            post_bytes = json.dumps(data).encode("utf-8")

            # 1. Try standard Python socket connection
            try:
                req = urllib.request.Request(
                    url,
                    data=post_bytes,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception:
                # 2. Resilient fallback to curl for macOS Sequoia LAN routing
                if shutil.which("curl"):
                    cmd = [
                        "curl",
                        "-s",
                        "--max-time",
                        str(timeout),
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        json.dumps(data),
                        url,
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if res.returncode == 0 and res.stdout.strip():
                        return json.loads(res.stdout)
                raise

        target_pool = self.embedding_pool if use_embedding_pool else self.pool
        req_chars = len(data.get("prompt", "")) or (sum(len(t) for t in data.get("input", [])) if isinstance(data.get("input"), list) else len(str(data.get("input", ""))))
        return target_pool.execute_with_failover(
            _invoke,
            retries=3,
            input_chars=req_chars,
            task_type="embedding" if use_embedding_pool else "rag",
        )

    async def _llm_model_func(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        """Async LLM completion function bound to multi-server Ollama pool."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
        }
        res = await asyncio.to_thread(self._call_ollama, "/api/chat", data, timeout=240, use_embedding_pool=False)
        return res.get("message", {}).get("content", "")

    async def _embedding_func(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """Async embedding function generating vectors via Ollama pool."""
        data = {
            "model": self.embedding_model,
            "input": texts,
        }
        res = await asyncio.to_thread(self._call_ollama, "/api/embed", data, timeout=120, use_embedding_pool=True)
        embeddings = res.get("embeddings", [])
        if not embeddings or len(embeddings) != len(texts):
            # Fallback per-text if bulk embed returns unexpected dimensions
            embeddings = []
            for t in texts:
                single_res = await asyncio.to_thread(
                    self._call_ollama,
                    "/api/embed",
                    {"model": self.embedding_model, "input": [t]},
                    timeout=60,
                    use_embedding_pool=True,
                )
                em_list = single_res.get("embeddings", [[]])
                if em_list and len(em_list[0]) == self.embedding_dim:
                    embeddings.append(em_list[0])
                else:
                    embeddings.append([0.0] * self.embedding_dim)

        return np.array(embeddings, dtype=np.float32)

    def detect_embedding_dim(self) -> int:
        """Probe embedding model to detect output vector dimension."""
        try:
            res = self._call_ollama(
                "/api/embed",
                {"model": self.embedding_model, "input": ["dimension probe"]},
                timeout=30,
                use_embedding_pool=True,
            )
            embeddings = res.get("embeddings", [])
            if embeddings and len(embeddings[0]) > 0:
                detected_dim = len(embeddings[0])
                return detected_dim
        except Exception as e:
            logger.warning(
                f"Failed to auto-detect embedding dimension for '{self.embedding_model}': {e}. "
                f"Falling back to configured dimension {self.embedding_dim}."
            )
        return self.embedding_dim

    async def ainitialize(self) -> None:
        """Initialize the LightRAG instance and persistent storages."""
        if self._rag is not None and self._initialized:
            return

        # Auto-detect or sync actual embedding dimension from Ollama model
        detected_dim = await asyncio.to_thread(self.detect_embedding_dim)
        if detected_dim and detected_dim != self.embedding_dim:
            logger.warning(
                f"LightRAG embedding dimension mismatch detected: configured={self.embedding_dim}, "
                f"detected={detected_dim} for model '{self.embedding_model}'. "
                f"Automatically updating embedding_dim to {detected_dim}."
            )
            self.embedding_dim = detected_dim

        # Use closure functions rather than bound methods (self._llm_model_func)
        # because LightRAG's __post_init__ runs dataclasses.asdict(self) which
        # deepcopies bound methods and fails with TypeError on unpicklable threading.RLock in pool.
        call_ollama = self._call_ollama
        llm_model = self.llm_model
        embedding_model = self.embedding_model
        embedding_dim = self.embedding_dim
        keep_alive = self.keep_alive

        async def _llm_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[Dict[str, Any]]] = None,
            **kwargs: Any,
        ) -> str:
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": llm_model,
                "messages": messages,
                "stream": False,
                "keep_alive": keep_alive,
            }
            res = await asyncio.to_thread(call_ollama, "/api/chat", data, timeout=240, use_embedding_pool=False)
            return res.get("message", {}).get("content", "")

        async def _embed_func(
            texts: List[str],
            **kwargs: Any,
        ) -> np.ndarray:
            data = {
                "model": embedding_model,
                "input": texts,
                "keep_alive": keep_alive,
            }
            res = await asyncio.to_thread(call_ollama, "/api/embed", data, timeout=120, use_embedding_pool=True)
            embeddings = res.get("embeddings", [])
            if not embeddings or len(embeddings) != len(texts):
                embeddings = []
                for t in texts:
                    single_res = await asyncio.to_thread(
                        call_ollama,
                        "/api/embed",
                        {"model": embedding_model, "input": [t], "keep_alive": keep_alive},
                        timeout=60,
                        use_embedding_pool=True,
                    )
                    em_list = single_res.get("embeddings", [[]])
                    if em_list and len(em_list[0]) == embedding_dim:
                        embeddings.append(em_list[0])
                    else:
                        embeddings.append([0.0] * embedding_dim)

            return np.array(embeddings, dtype=np.float32)

        self._rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.max_token_size,
                func=_embed_func,
            ),
            # Concurrency & Throughput parameters
            max_parallel_insert=self.max_parallel_insert,
            llm_model_max_async=self.llm_max_async,
            embedding_batch_num=self.embedding_batch_num,
            embedding_func_max_async=self.embedding_max_async,
            enable_llm_cache=True,
            enable_llm_cache_for_entity_extract=True,
        )
        await self._rag.initialize_storages()
        self._initialized = True
        logger.info(f"Initialized LightRAG engine in {self.working_dir}")

    def initialize(self) -> None:
        """Synchronous initialization of storages."""
        asyncio.run(self.ainitialize())

    async def ainsert_text(
        self,
        text: str,
        doc_id: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> str:
        """Insert raw text document into LightRAG incrementally."""
        await self.ainitialize()
        assert self._rag is not None
        ids = [doc_id] if doc_id else None
        fps = [file_path] if file_path else None
        return await self._rag.ainsert(text, ids=ids, file_paths=fps)

    def insert_text(
        self,
        text: str,
        doc_id: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> str:
        """Synchronous document insertion."""
        return asyncio.run(self.ainsert_text(text, doc_id=doc_id, file_path=file_path))

    async def ainsert_chunks(
        self,
        chunks: List[HierarchicalChunk],
        book_title: str,
        book_id: int,
        batch_size: int = 12,
        on_progress: Optional[Any] = None,
    ) -> int:
        """
        Incrementally index structured hierarchical chunks into LightRAG.
        Each chunk is formatted with its macro chapter/subtitle context header.
        Submits in parallel batches to allow live progress updates and avoid
        giant memory spikes on multi-hundred chunk books.
        """
        await self.ainitialize()
        assert self._rag is not None

        if not chunks:
            return 0

        documents: List[str] = []
        doc_ids: List[str] = []
        for c in chunks:
            header = f"# Book: {book_title}\n"
            if c.chapter_title:
                header += f"## Chapter: {c.chapter_title}\n"
            if c.subtitle and c.subtitle != c.chapter_title:
                header += f"### Section: {c.subtitle}\n"

            content = f"{header}\n{c.text.strip()}"
            documents.append(content)
            doc_ids.append(f"book_{book_id}_chunk_{c.chunk_id}")

        total_docs = len(documents)
        completed_docs = 0

        # Subdivide into batches for responsive progress reporting
        effective_batch_size = max(1, batch_size)
        for i in range(0, total_docs, effective_batch_size):
            batch_docs = documents[i : i + effective_batch_size]
            batch_ids = doc_ids[i : i + effective_batch_size]
            await self._rag.ainsert(batch_docs, ids=batch_ids)
            completed_docs += len(batch_docs)
            if on_progress:
                try:
                    on_progress(completed_docs, total_docs)
                except Exception:
                    pass

        return total_docs

    def insert_chunks(
        self,
        chunks: List[HierarchicalChunk],
        book_title: str,
        book_id: int,
        batch_size: int = 12,
        on_progress: Optional[Any] = None,
    ) -> int:
        """Synchronous chunk indexing with optional progress tracking."""
        return asyncio.run(
            self.ainsert_chunks(
                chunks,
                book_title=book_title,
                book_id=book_id,
                batch_size=batch_size,
                on_progress=on_progress,
            )
        )

    async def aquery(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 40,
        chunk_top_k: int = 20,
    ) -> str:
        """
        Execute dual-level query via LightRAG.
        Modes:
          - 'local': focuses on specific entities and direct relations
          - 'global': focuses on broader themes, high-level summaries, and relationships
          - 'hybrid': blends local subgraphs and global relationship context
          - 'naive': traditional flat vector search across chunks
          - 'mix': combines knowledge graph structure with vector ranking
        """
        await self.ainitialize()
        assert self._rag is not None

        valid_modes = {"local", "global", "hybrid", "naive", "mix"}
        m = mode.lower().strip()
        if m not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {sorted(valid_modes)}")

        param = QueryParam(mode=m, top_k=top_k, chunk_top_k=chunk_top_k)
        result = await self._rag.aquery(query, param=param)
        return str(result)

    def query(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 40,
        chunk_top_k: int = 20,
    ) -> str:
        """Synchronous dual-level query."""
        return asyncio.run(self.aquery(query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k))

    def get_stats(self) -> Dict[str, Any]:
        """Inspect LightRAG graph and storage stats."""
        graph_file = self.working_dir / "graph_chunk_entity_relation.graphml"
        nodes_count = 0
        edges_count = 0
        if graph_file.is_file():
            try:
                import networkx as nx
                g = nx.read_graphml(str(graph_file))
                nodes_count = g.number_of_nodes()
                edges_count = g.number_of_edges()
            except Exception:
                pass

        return {
            "working_dir": str(self.working_dir),
            "graphml_exists": graph_file.is_file(),
            "nodes_count": nodes_count,
            "edges_count": edges_count,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
        }
