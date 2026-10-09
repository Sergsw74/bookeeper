"""
Multi-server smart proxy and failover client for Ollama with priority routing and automatic recovery.
"""

import json
import logging
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from bookeeper.config import DEFAULT_CAPABILITIES, OllamaServerConfig

logger = logging.getLogger(__name__)


@dataclass
class OllamaServerNode:
    """Tracks dynamic health, priority, capabilities, active tasks, and cooldown for an individual Ollama endpoint."""

    url: str
    priority: int = 1
    name: Optional[str] = None
    capabilities: List[str] = field(default_factory=lambda: list(DEFAULT_CAPABILITIES))
    failed_at: Optional[float] = None
    failure_count: int = 0
    last_error: Optional[str] = None
    active_tasks: int = 0
    last_task_start: Optional[float] = None

    @property
    def label(self) -> str:
        return self.name or self.url

    @property
    def consecutive_failures(self) -> int:
        return self.failure_count

    @property
    def capability(self) -> List[str]:
        return self.capabilities

    def has_capability(self, cap: str) -> bool:
        """Check if server node supports the requested capability (llm, embedding, verification)."""
        cap_clean = cap.strip().lower()
        if cap_clean in ("embedding", "embeddings", "embed", "embeds"):
            target = "embedding"
        elif cap_clean in ("verification", "verify", "verifier"):
            target = "verification"
        elif cap_clean in ("llm", "llms", "chat", "generate"):
            target = "llm"
        else:
            target = cap_clean
        return target in [c.lower() for c in self.capabilities]

    def is_eligible(self, cooldown_seconds: int = 600, now: Optional[float] = None) -> bool:
        """Check if server is healthy or its failover cooldown has elapsed."""
        if self.failed_at is None:
            return True
        current_time = time.time() if now is None else now
        return (current_time - self.failed_at) >= cooldown_seconds

    def cooldown_remaining(self, cooldown_seconds: int = 600, now: Optional[float] = None) -> float:
        """Seconds remaining before this server becomes eligible for probe/retry."""
        if self.failed_at is None:
            return 0.0
        current_time = time.time() if now is None else now
        elapsed = current_time - self.failed_at
        return max(0.0, cooldown_seconds - elapsed)

    def mark_failure(self, error: str) -> None:
        """Record failure timestamp and error."""
        self.failed_at = time.time()
        self.failure_count += 1
        self.last_error = error

    def mark_success(self) -> None:
        """Clear failure state and mark healthy."""
        if self.failed_at is not None:
            logger.info(f"Ollama server '{self.url}' recovered and is now active.")
        self.failed_at = None
        self.failure_count = 0
        self.last_error = None

    def release_without_cooldown(self, error: str) -> None:
        """Release server from current task without placing into cooldown."""
        self.failed_at = None
        self.last_error = error


_thread_local = threading.local()


def is_connection_error(exc: Exception) -> bool:
    """
    Return True if exception represents a network connection refusal, host unreachable,
    or server down / offline (as opposed to a model timeout or prompt formatting issue).
    """
    if isinstance(exc, (ConnectionRefusedError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return True

    # Check urllib URLError and underlying reason
    reason = getattr(exc, "reason", None)
    if isinstance(reason, Exception):
        if is_connection_error(reason):
            return True
    elif isinstance(reason, str):
        r_lower = reason.lower()
        if any(p in r_lower for p in [
            "connection refused", "couldn't connect", "failed to connect",
            "errno 61", "errno 111", "winerror 10061",
            "network is unreachable", "host is down", "no route to host",
        ]):
            return True

    err_str = str(exc).lower()
    patterns = [
        "connection refused",
        "errno 61",          # macOS Connection refused
        "errno 111",         # Linux Connection refused
        "winerror 10061",    # Windows Connection refused
        "failed to connect",
        "couldn't connect",
        "network is unreachable",
        "host is down",
        "no route to host",
        "remotedisconnected",
        "remote end closed connection",
        "connection reset by peer",
    ]
    return any(p in err_str for p in patterns)


class OllamaPool:
    """
    Manages N Ollama servers with priority-based routing, automated failover,
    and concurrent load balancing across all available/alive servers.
    """

    def __init__(
        self,
        servers: List[OllamaServerConfig],
        cooldown_seconds: int = 600,
        max_tasks_per_server: int = 1,
    ):
        if not servers:
            raise ValueError("OllamaPool requires at least one server configuration.")

        self.cooldown_seconds = cooldown_seconds
        self.max_tasks_per_server = max(1, max_tasks_per_server)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self.nodes: List[OllamaServerNode] = [
            OllamaServerNode(
                url=s.url.rstrip("/"),
                priority=s.priority,
                name=s.name or f"node-{idx + 1}",
                capabilities=list(getattr(s, "capability", None) or getattr(s, "capabilities", None) or DEFAULT_CAPABILITIES),
            )
            for idx, s in enumerate(servers)
        ]

    @classmethod
    def from_urls(
        cls,
        urls: List[str],
        cooldown_seconds: int = 600,
        max_tasks_per_server: int = 1,
        capabilities: Optional[List[str]] = None,
    ) -> "OllamaPool":
        """Instantiate pool from an ordered list of URLs where list index reflects priority."""
        configs = [
            OllamaServerConfig(
                url=u,
                priority=idx + 1,
                name=f"server-{idx + 1}",
                capability=capabilities or list(DEFAULT_CAPABILITIES),
            )
            for idx, u in enumerate(urls)
        ]
        return cls(
            servers=configs,
            cooldown_seconds=cooldown_seconds,
            max_tasks_per_server=max_tasks_per_server,
        )

    @property
    def alive_nodes(self) -> List[OllamaServerNode]:
        """Return list of currently healthy/eligible server nodes."""
        with self._lock:
            return [n for n in self.nodes if n.is_eligible(self.cooldown_seconds)]

    def nodes_for_capability(self, capability: Optional[str] = None) -> List[OllamaServerNode]:
        """Return nodes that support the requested capability."""
        with self._lock:
            if not capability:
                return list(self.nodes)
            return [n for n in self.nodes if n.has_capability(capability)]

    def alive_nodes_for_capability(self, capability: Optional[str] = None) -> List[OllamaServerNode]:
        """Return eligible/healthy nodes that support the requested capability."""
        with self._lock:
            candidates = self.nodes_for_capability(capability)
            return [n for n in candidates if n.is_eligible(self.cooldown_seconds)]

    def for_capability(self, capability: str) -> "OllamaPool":
        """Return a sub-pool containing only nodes that support the specified capability."""
        matching_servers = [
            OllamaServerConfig(
                url=n.url,
                priority=n.priority,
                name=n.name,
                capability=list(n.capabilities),
            )
            for n in self.nodes
            if n.has_capability(capability)
        ]
        if not matching_servers:
            logger.warning(
                f"No servers in pool support capability '{capability}'; falling back to all pool servers."
            )
            matching_servers = [
                OllamaServerConfig(
                    url=n.url,
                    priority=n.priority,
                    name=n.name,
                    capability=list(n.capabilities),
                )
                for n in self.nodes
            ]
        return OllamaPool(
            servers=matching_servers,
            cooldown_seconds=self.cooldown_seconds,
            max_tasks_per_server=self.max_tasks_per_server,
        )

    def get_ordered_servers(self, capability: Optional[str] = None) -> List[OllamaServerNode]:
        """
        Return servers ordered by priority for execution.
        If capability is specified, filters only nodes supporting that capability.
        Eligible servers (healthy or cooldown expired) are prioritized.
        If all servers are in cooldown, falls back to the server that failed longest ago.
        """
        with self._lock:
            candidate_nodes = self.nodes_for_capability(capability)
            if not candidate_nodes:
                candidate_nodes = self.nodes

            eligible = [n for n in candidate_nodes if n.is_eligible(self.cooldown_seconds)]
            if eligible:
                return sorted(eligible, key=lambda n: n.priority)

            # Fallback: all servers cooling down; try server whose failure was longest ago
            logger.warning(
                f"All {len(candidate_nodes)} Ollama server(s) for capability '{capability or 'all'}' are in cooldown; probing oldest failed server."
            )
            return sorted(candidate_nodes, key=lambda n: n.failed_at or 0.0)

    def get_candidate_servers(self, capability: Optional[str] = None) -> List[OllamaServerNode]:
        """
        Return servers ordered for execution:
        1. Eligible (healthy) servers ordered primarily by least active tasks (load balancing),
           and secondarily by priority ascending.
        2. Cooling down servers as fallback ordered by oldest failure.
        """
        with self._lock:
            candidate_nodes = self.nodes_for_capability(capability)
            if not candidate_nodes:
                candidate_nodes = self.nodes

            eligible = [n for n in candidate_nodes if n.is_eligible(self.cooldown_seconds)]
            if eligible:
                sorted_eligible = sorted(
                    eligible,
                    key=lambda n: (n.active_tasks, n.priority),
                )
                cooling = [n for n in candidate_nodes if not n.is_eligible(self.cooldown_seconds)]
                sorted_cooling = sorted(cooling, key=lambda n: n.failed_at or 0.0)
                return sorted_eligible + sorted_cooling
            else:
                return sorted(candidate_nodes, key=lambda n: n.failed_at or 0.0)

    @property
    def primary_server(self) -> OllamaServerNode:
        """Return the highest priority currently eligible server."""
        ordered = self.get_ordered_servers()
        return ordered[0]

    def primary_server_for_capability(self, capability: Optional[str] = None) -> OllamaServerNode:
        """Return the highest priority currently eligible server for a capability."""
        ordered = self.get_ordered_servers(capability)
        return ordered[0]

    def reset_server(self, url: str, model: Optional[str] = None) -> bool:
        """
        Unload model runner on an Ollama server to terminate hung generation processes and free VRAM.
        Uses Ollama's official keep_alive: 0 endpoint to cleanly reset stuck model runners.
        """
        try:
            body_dict: Dict[str, Any] = {"keep_alive": 0}
            if model:
                body_dict["model"] = model
            data = json.dumps(body_dict).encode("utf-8")
            req = urllib.request.Request(
                f"{url.rstrip('/')}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                logger.info(f"Sent keep_alive: 0 reset to unstick runner on {url}.")
                return True
        except Exception as e:
            if shutil.which("curl"):
                try:
                    payload = json.dumps({"model": model, "keep_alive": 0}) if model else '{"keep_alive": 0}'
                    cmd = [
                        "curl",
                        "-s",
                        "--max-time",
                        "2",
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        payload,
                        f"{url.rstrip('/')}/api/generate",
                    ]
                    subprocess.run(cmd, capture_output=True, text=True, check=False)
                    return True
                except Exception:
                    pass
            logger.debug(f"Reset request to {url} completed with: {e}")
            return False

    def execute_with_failover(
        self,
        operation: Callable[[str], Any],
        on_failover: Optional[Callable[[OllamaServerNode, Exception, OllamaServerNode], None]] = None,
        retries: int = 1,
        backoff_base: float = 1.0,
        max_task_duration: float = 75.0,
        model_name: Optional[str] = None,
        quarantine_server: bool = True,
        exclude_urls: Optional[Set[str]] = None,
        capability: Optional[str] = None,
    ) -> Any:
        """
        Execute an operation passing base_url.
        Enforces sequential submission/execution per Ollama server (at most max_tasks_per_server per node),
        routes across idle servers by priority, retries on transient errors, and automatically fails over.
        Optionally filters candidate servers by required capability ('llm', 'embedding', 'verification').
        """
        pool_nodes = self.nodes_for_capability(capability)
        if not pool_nodes:
            logger.warning(
                f"No servers in pool support capability '{capability}'; falling back to all pool servers."
            )
            pool_nodes = self.nodes

        last_exception: Optional[Exception] = None
        errors: List[str] = []
        tried_urls: set = set(exclude_urls) if exclude_urls else set()
        eligible_all = [n.url for n in pool_nodes if n.is_eligible(self.cooldown_seconds)]
        if eligible_all and all(u in tried_urls for u in eligible_all):
            tried_urls = set()

        while True:
            selected_node: Optional[OllamaServerNode] = None

            with self._condition:
                while True:
                    now = time.time()
                    # Watchdog: detect and unstick any node whose task exceeded max_task_duration
                    for n in self.nodes:
                        if n.active_tasks > 0 and n.last_task_start is not None:
                            elapsed = now - n.last_task_start
                            if elapsed > max_task_duration:
                                logger.warning(
                                    f"Watchdog: Ollama server '{n.url}' active task has run for {elapsed:.1f}s (> {max_task_duration:.1f}s). "
                                    f"Forcing reset of active task counter and sending runner cleanup."
                                )
                                n.active_tasks = 0
                                n.last_task_start = None
                                if quarantine_server:
                                    n.mark_failure(f"Watchdog: task exceeded {max_task_duration:.1f}s")
                                else:
                                    n.release_without_cooldown(f"Watchdog: task exceeded {max_task_duration:.1f}s")
                                self.reset_server(n.url, model=model_name)
                                self._condition.notify_all()

                    # 1. Eligible servers not yet tried in this invocation (supporting required capability)
                    eligible = [
                        n for n in pool_nodes
                        if n.url not in tried_urls and n.is_eligible(self.cooldown_seconds)
                    ]

                    if eligible:
                        # Find servers with available capacity (< max_tasks_per_server)
                        available = [
                            n for n in eligible
                            if n.active_tasks < self.max_tasks_per_server
                        ]
                        if available:
                            # Prefer lowest active tasks (idle first), then highest priority (lowest number)
                            available.sort(key=lambda n: (n.active_tasks, n.priority))
                            selected_node = available[0]
                            selected_node.active_tasks += 1
                            selected_node.last_task_start = time.time()
                            break
                        else:
                            # All eligible servers are currently busy processing another task sequentially.
                            # Wait until a server completes its task before submitting!
                            self._condition.wait(timeout=1.0)
                            continue
                    else:
                        # No eligible servers remaining among untried.
                        # Check cooling down servers as fallback
                        cooling = [
                            n for n in pool_nodes
                            if n.url not in tried_urls and not n.is_eligible(self.cooldown_seconds)
                        ]
                        if cooling:
                            cooling.sort(key=lambda n: n.failed_at or 0.0)
                            available_cooling = [
                                n for n in cooling
                                if n.active_tasks < self.max_tasks_per_server
                            ]
                            if available_cooling:
                                selected_node = available_cooling[0]
                                selected_node.active_tasks += 1
                                selected_node.last_task_start = time.time()
                                logger.warning(
                                    f"All eligible Ollama servers cooling down; probing '{selected_node.url}'."
                                )
                                break
                            else:
                                self._condition.wait(timeout=1.0)
                                continue
                        else:
                            # All servers in the pool have been attempted!
                            break

            if selected_node is None:
                break

            tried_urls.add(selected_node.url)

            try:
                max_attempts = max(1, retries)
                attempt = 0
                while attempt < max_attempts:
                    try:
                        _thread_local.last_used_server = selected_node.label
                        _thread_local.last_used_server_url = selected_node.url
                        result = operation(selected_node.url)
                        with self._condition:
                            selected_node.mark_success()
                        return result
                    except Exception as e:
                        last_exception = e
                        conn_err = is_connection_error(e)

                        # If connection error (refused/offline), ensure at least 3 retry attempts on this server
                        if conn_err and max_attempts < 3:
                            max_attempts = 3

                        # If attempt failed or timed out, attempt to unstick runner only if server is reachable
                        if not conn_err:
                            self.reset_server(selected_node.url, model=model_name)

                        if attempt < max_attempts - 1:
                            sleep_time = min(3.0, backoff_base * (1.5 ** attempt))
                            err_tag = "connection refused / offline" if conn_err else "failed"
                            logger.warning(
                                f"Ollama server '{selected_node.url}' attempt {attempt + 1}/{max_attempts} {err_tag} ({e}). "
                                f"Retrying in {sleep_time:.1f}s..."
                            )
                            time.sleep(sleep_time)
                            attempt += 1
                        else:
                            errors.append(f"{selected_node.url} ({max_attempts} attempt(s)): {e}")
                            with self._condition:
                                selected_node.failure_count += 1
                                selected_node.last_error = str(e)

                                # Connection refused after retries, OR quarantine_server requested,
                                # OR 3 consecutive failures on this server -> MUST move to cooldown period!
                                should_cooldown = (
                                    quarantine_server
                                    or conn_err
                                    or selected_node.failure_count >= 3
                                )
                                if should_cooldown:
                                    selected_node.failed_at = time.time()
                                    if conn_err:
                                        reason_str = f"connection refused after {attempt + 1} retries"
                                    elif selected_node.failure_count >= 3:
                                        reason_str = f"{selected_node.failure_count} consecutive failures"
                                    else:
                                        reason_str = "server failed"
                                    logger.warning(
                                        f"Ollama server '{selected_node.url}' ({selected_node.label}, Priority {selected_node.priority}) failed: {e} "
                                        f"({reason_str}). Marked cooling down for {self.cooldown_seconds}s."
                                    )
                                else:
                                    selected_node.release_without_cooldown(str(e))
                                    logger.warning(
                                        f"Ollama server '{selected_node.url}' attempt failed ({e}). "
                                        f"Runner reset; server released immediately without cooldown for other tasks."
                                    )
                                    if not quarantine_server and not conn_err:
                                        raise RuntimeError(
                                            f"Ollama server '{selected_node.url}' attempt failed ({e}). "
                                            f"Released without cooldown for queue rotation."
                                        ) from e

                            if on_failover:
                                with self._condition:
                                    next_candidates = [
                                        n for n in pool_nodes
                                        if n.url not in tried_urls and n.is_eligible(self.cooldown_seconds)
                                    ]
                                    if next_candidates:
                                        next_node = sorted(next_candidates, key=lambda n: n.priority)[0]
                                        on_failover(selected_node, e, next_node)
                            break
            finally:
                with self._condition:
                    selected_node.active_tasks = max(0, selected_node.active_tasks - 1)
                    selected_node.last_task_start = None
                    self._condition.notify_all()

        err_summary = "; ".join(errors)
        cap_str = f" for capability '{capability}'" if capability else ""
        raise RuntimeError(
            f"All {len(pool_nodes)} Ollama server(s){cap_str} failed. Attempts: [{err_summary}]"
        ) from last_exception

    @classmethod
    def get_last_used_server(cls) -> Optional[str]:
        """Return the label of the last Ollama server used by the calling thread."""
        return getattr(_thread_local, "last_used_server", None)

    @classmethod
    def get_last_used_server_url(cls) -> Optional[str]:
        """Return the base URL of the last Ollama server used by the calling thread."""
        return getattr(_thread_local, "last_used_server_url", None)

    def get_status(self) -> List[Dict[str, Any]]:
        """Return status snapshot of all pool nodes."""
        with self._lock:
            status_list = []
            for n in sorted(self.nodes, key=lambda x: x.priority):
                eligible = n.is_eligible(self.cooldown_seconds)
                rem = n.cooldown_remaining(self.cooldown_seconds)
                status_list.append(
                    {
                        "url": n.url,
                        "priority": n.priority,
                        "name": n.label,
                        "capabilities": list(n.capabilities),
                        "status": "active" if eligible else "cooling_down",
                        "active_tasks": n.active_tasks,
                        "cooldown_remaining_sec": round(rem, 1),
                        "failure_count": n.failure_count,
                        "last_error": n.last_error,
                    }
                )
            return status_list


class FailoverOllamaEmbeddings(Embeddings):
    """
    LangChain Embeddings wrapper with multi-server failover support across OllamaPool.
    """

    def __init__(
        self,
        pool: OllamaPool,
        model: str = "nomic-embed-text",
    ):
        self.pool = pool
        self.model = model
        self._lock = threading.Lock()
        self.total_embedded_texts: int = 0
        self.total_embedded_chars: int = 0
        self.total_embedding_seconds: float = 0.0

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        model: Optional[str] = None,
    ) -> "FailoverOllamaEmbeddings":
        """Instantiate FailoverOllamaEmbeddings using dedicated embedding servers or fallback pool."""
        servers = getattr(settings, "resolved_embedding_servers", None) or getattr(settings, "resolved_ollama_servers", [])
        cooldown = getattr(settings, "failover_cooldown_seconds", 600)
        pool = OllamaPool(
            servers=servers,
            cooldown_seconds=cooldown,
            max_tasks_per_server=1,
        )
        return cls(
            pool=pool,
            model=model or getattr(settings, "embedding_model", "nomic-embed-text"),
        )

    def reset_stats(self) -> None:
        """Reset embedding throughput counters."""
        with self._lock:
            self.total_embedded_texts = 0
            self.total_embedded_chars = 0
            self.total_embedding_seconds = 0.0

    @property
    def embedding_stats(self) -> Dict[str, Any]:
        """Return embedding throughput statistics."""
        with self._lock:
            secs = self.total_embedding_seconds
            texts = self.total_embedded_texts
            chars = self.total_embedded_chars
            speed_texts = (texts / secs) if secs > 0 else 0.0
            speed_chars = (chars / secs) if secs > 0 else 0.0
            return {
                "total_texts": texts,
                "total_chars": chars,
                "total_seconds": secs,
                "speed_texts_per_sec": speed_texts,
                "speed_chars_per_sec": speed_chars,
            }

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed a list of documents with multi-server failover, batching, and curl fallback."""
        if not texts:
            return []

        def _embed_batch(batch_texts: List[str]) -> List[List[float]]:
            def _embed(url: str):
                try:
                    client = OllamaEmbeddings(base_url=url, model=self.model)
                    return client.embed_documents(batch_texts)
                except Exception as exc:
                    if shutil.which("curl"):
                        data = {"model": self.model, "input": batch_texts}
                        cmd = [
                            "curl",
                            "-s",
                            "--max-time",
                            "120",
                            "-X",
                            "POST",
                            "-H",
                            "Content-Type: application/json",
                            "-d",
                            json.dumps(data),
                            f"{url.rstrip('/')}/api/embed",
                        ]
                        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                        if res.returncode == 0 and res.stdout.strip():
                            resp_data = json.loads(res.stdout)
                            embs = resp_data.get("embeddings", [])
                            if embs and len(embs) == len(batch_texts):
                                return embs
                    raise exc

            t0 = time.perf_counter()
            result = self.pool.execute_with_failover(_embed, capability="embedding")
            elapsed = time.perf_counter() - t0
            with self._lock:
                self.total_embedded_texts += len(batch_texts)
                self.total_embedded_chars += sum(len(t) for t in batch_texts)
                self.total_embedding_seconds += elapsed
            return result

        if len(texts) <= batch_size:
            return _embed_batch(texts)

        all_embs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            all_embs.extend(_embed_batch(texts[i : i + batch_size]))
        return all_embs

    def embed_query(self, text: str) -> List[float]:
        """Embed query text with multi-server failover and curl fallback."""
        def _embed(url: str):
            try:
                client = OllamaEmbeddings(base_url=url, model=self.model)
                return client.embed_query(text)
            except Exception as exc:
                if shutil.which("curl"):
                    data = {"model": self.model, "input": [text]}
                    cmd = [
                        "curl",
                        "-s",
                        "--max-time",
                        "60",
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        json.dumps(data),
                        f"{url.rstrip('/')}/api/embed",
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if res.returncode == 0 and res.stdout.strip():
                        resp_data = json.loads(res.stdout)
                        embs = resp_data.get("embeddings", [[]])
                        if embs and embs[0]:
                            return embs[0]
                raise exc

        t0 = time.perf_counter()
        result = self.pool.execute_with_failover(_embed, capability="embedding")
        elapsed = time.perf_counter() - t0
        with self._lock:
            self.total_embedded_texts += 1
            self.total_embedded_chars += len(text)
            self.total_embedding_seconds += elapsed
        return result
