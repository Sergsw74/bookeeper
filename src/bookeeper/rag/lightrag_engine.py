"""LightRAG engine integration for fast graph-enhanced indexing and dual-level retrieval."""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, TypeVar, Union

import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from bookeeper.config import Settings
from bookeeper.processing.chunker import HierarchicalChunk
from bookeeper.processing.extractor import OllamaPool

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _LoopThread:
    """A dedicated daemon thread running a persistent asyncio event loop for LightRAG operations."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True, name="LightRAGEventLoop")
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run_sync(self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
        """Run coroutine in the dedicated event loop and return its result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self, timeout: float = 10.0) -> None:
        """Cleanly cancel all pending tasks, drain workers, and shut down the event loop."""
        if not self._loop.is_running():
            return

        async def _cleanup():
            current = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self._loop.shutdown_asyncgens()
            except Exception:
                pass
            self._loop.stop()

        try:
            future = asyncio.run_coroutine_threadsafe(_cleanup(), self._loop)
            future.result(timeout=timeout)
        except Exception:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

        if not self._loop.is_running() and not self._loop.is_closed():
            try:
                self._loop.close()
            except Exception:
                pass


def _normalize_keep_alive(val: Any) -> Any:
    """
    Normalize keep_alive value for Ollama API.
    Ollama requires duration strings (e.g. '5m', '1h') or integers (e.g. -1 for indefinite, 0 to unload).
    String "-1" causes Ollama to fail with 'time: missing unit in duration "-1"'.
    """
    if val is None:
        return -1
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        v = val.strip()
        if v in ("-1", "0") or v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
            return int(v)
        return v
    return -1


def _clean_json_str(text: str) -> str:
    """Clean model response by removing thinking tags, markdown code blocks, and extra text."""
    if not text:
        return ""
    text = text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start_brace = text.find("{")
    if start_brace != -1:
        end_brace = text.rfind("}")
        if end_brace > start_brace:
            return text[start_brace : end_brace + 1]
    return text


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
        keep_alive: Union[str, int] = -1,
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
        self.keep_alive = _normalize_keep_alive(keep_alive)

        self._rag: Optional[LightRAG] = None
        self._initialized = False
        self._loop_thread: Optional[_LoopThread] = None

    def close(self) -> None:
        """Clean up resources, finalize storages, shut down all worker queues, and terminate event loop."""
        if self._rag is not None and self._loop_thread is not None:
            async def _finalize_rag():
                rag = self._rag
                if rag is None:
                    return

                # 1. Finalize storages
                if hasattr(rag, "finalize_storages"):
                    try:
                        await rag.finalize_storages()
                    except Exception:
                        pass

                # 2. Shutdown all role LLM worker queues
                states = getattr(rag, "_role_llm_states", {}) or {}
                for state in states.values():
                    if hasattr(state, "wrapped") and hasattr(state.wrapped, "shutdown"):
                        try:
                            await state.wrapped.shutdown(graceful=False)
                        except Exception:
                            pass

                # 3. Shutdown embedding worker queue
                if hasattr(rag, "embedding_func") and rag.embedding_func:
                    fn = getattr(rag.embedding_func, "func", None)
                    if hasattr(fn, "shutdown"):
                        try:
                            await fn.shutdown(graceful=False)
                        except Exception:
                            pass

            try:
                self._run_sync(_finalize_rag(), timeout=30)
            except Exception:
                pass

        if self._loop_thread is not None:
            self._loop_thread.stop()
            self._loop_thread = None
        self._rag = None
        self._initialized = False

    def clean(self) -> None:
        """Purge all stored LightRAG graph, vector, and index files in working_dir."""
        self.close()
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

    def _run_sync(self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
        """Execute a coroutine synchronously inside the persistent LightRAG event loop."""
        if self._loop_thread is None:
            self._loop_thread = _LoopThread()
        return self._loop_thread.run_sync(coro, timeout=timeout)

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
                    parsed = json.loads(resp.read().decode("utf-8"))
                    if isinstance(parsed, dict) and "error" in parsed:
                        raise RuntimeError(f"Ollama error from {url}: {parsed['error']}")
                    return parsed
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
                        parsed = json.loads(res.stdout)
                        if isinstance(parsed, dict) and "error" in parsed:
                            raise RuntimeError(f"Ollama error from {url}: {parsed['error']}")
                        return parsed
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

            data: Dict[str, Any] = {
                "model": llm_model,
                "messages": messages,
                "stream": False,
                "keep_alive": keep_alive,
            }

            # Forward JSON mode to Ollama when structured output is requested (e.g. LightRAG keyword extraction)
            response_format = kwargs.get("response_format")
            is_json_requested = bool(
                response_format
                or kwargs.get("format") == "json"
                or "high_level_keywords" in prompt
                or "json_object" in str(response_format)
            )
            if is_json_requested:
                data["format"] = "json"

            res = await asyncio.to_thread(call_ollama, "/api/chat", data, timeout=240, use_embedding_pool=False)
            content = res.get("message", {}).get("content", "")

            if is_json_requested:
                content = _clean_json_str(content)
                parsed: Dict[str, Any] = {}
                try:
                    if content:
                        parsed = json.loads(content)
                        if not isinstance(parsed, dict):
                            parsed = {}
                except Exception:
                    parsed = {}

                # Guarantee valid high-signal keywords for LightRAG dual-level retrieval
                if "high_level_keywords" in prompt:
                    hl = parsed.get("high_level_keywords") or []
                    ll = parsed.get("low_level_keywords") or []
                    if not isinstance(hl, list):
                        hl = [str(hl)] if hl else []
                    if not isinstance(ll, list):
                        ll = [str(ll)] if ll else []

                    # If one list is empty, balance with the other
                    if not hl and ll:
                        hl = list(ll)
                    elif not ll and hl:
                        ll = list(hl)
                    elif not hl and not ll:
                        # Fallback: extract key subject terms directly from prompt's User Query
                        query_part = prompt.split("User Query:")[-1].strip() if "User Query:" in prompt else prompt
                        stopwords = {
                            "what", "which", "where", "when", "how", "why", "who", "whom", "whose",
                            "user", "query", "language", "english", "please", "identify", "extract",
                            "from", "with", "that", "this", "these", "those", "about", "into", "over",
                        }
                        words = [
                            w for w in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", query_part)
                            if w.lower() not in stopwords
                        ]
                        fallback_term = " ".join(words[:4]) if words else "query"
                        hl = [fallback_term]
                        ll = words[:4] if words else [fallback_term]

                    parsed["high_level_keywords"] = hl
                    parsed["low_level_keywords"] = ll
                    content = json.dumps(parsed)

            return content

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
        self._run_sync(self.ainitialize())

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
        return self._run_sync(self.ainsert_text(text, doc_id=doc_id, file_path=file_path))

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
        return self._run_sync(
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
        if hasattr(result, "content") and result.content is not None:
            return str(result.content)
        return str(result)

    def query(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 40,
        chunk_top_k: int = 20,
    ) -> str:
        """Synchronous dual-level query."""
        return self._run_sync(self.aquery(query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k))

    def get_stats(self) -> Dict[str, Any]:
        """Inspect LightRAG graph and storage stats."""
        graph_file = self.working_dir / "graph_chunk_entity_relation.graphml"
        nodes_count = 0
        edges_count = 0
        node_types: Dict[str, int] = {}
        if graph_file.is_file():
            try:
                import networkx as nx
                from collections import Counter
                g = nx.read_graphml(str(graph_file))
                nodes_count = g.number_of_nodes()
                edges_count = g.number_of_edges()
                node_types = dict(Counter(data.get("entity_type") or "unknown" for _, data in g.nodes(data=True)))
            except Exception:
                pass

        chunks_file = self.working_dir / "kv_store_text_chunks.json"
        chunks_count = 0
        if chunks_file.is_file():
            try:
                with open(chunks_file, "r", encoding="utf-8") as f:
                    chunks_count = len(json.load(f))
            except Exception:
                pass

        return {
            "working_dir": str(self.working_dir),
            "graphml_exists": graph_file.is_file(),
            "nodes_count": nodes_count,
            "edges_count": edges_count,
            "chunks_count": chunks_count,
            "node_types": node_types,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
        }
