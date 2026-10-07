"""
Multi-server smart proxy and failover client for Ollama with priority routing and automatic recovery.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from bookeeper.config import OllamaServerConfig

logger = logging.getLogger(__name__)


@dataclass
class OllamaServerNode:
    """Tracks dynamic health, priority, active tasks, and cooldown for an individual Ollama endpoint."""

    url: str
    priority: int = 1
    name: Optional[str] = None
    failed_at: Optional[float] = None
    failure_count: int = 0
    last_error: Optional[str] = None
    active_tasks: int = 0

    @property
    def label(self) -> str:
        return self.name or self.url

    @property
    def consecutive_failures(self) -> int:
        return self.failure_count

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


class OllamaPool:
    """
    Manages N Ollama servers with priority-based routing, automated failover,
    and concurrent load balancing across all available/alive servers.
    """

    def __init__(
        self,
        servers: List[OllamaServerConfig],
        cooldown_seconds: int = 600,
    ):
        if not servers:
            raise ValueError("OllamaPool requires at least one server configuration.")

        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.RLock()
        self.nodes: List[OllamaServerNode] = [
            OllamaServerNode(
                url=s.url.rstrip("/"),
                priority=s.priority,
                name=s.name or f"node-{idx + 1}",
            )
            for idx, s in enumerate(servers)
        ]

    @classmethod
    def from_urls(
        cls,
        urls: List[str],
        cooldown_seconds: int = 600,
    ) -> "OllamaPool":
        """Instantiate pool from an ordered list of URLs where list index reflects priority."""
        configs = [
            OllamaServerConfig(url=u, priority=idx + 1, name=f"server-{idx + 1}")
            for idx, u in enumerate(urls)
        ]
        return cls(servers=configs, cooldown_seconds=cooldown_seconds)

    @property
    def alive_nodes(self) -> List[OllamaServerNode]:
        """Return list of currently healthy/eligible server nodes."""
        with self._lock:
            return [n for n in self.nodes if n.is_eligible(self.cooldown_seconds)]

    def get_ordered_servers(self) -> List[OllamaServerNode]:
        """
        Return servers ordered by priority for execution.
        Eligible servers (healthy or cooldown expired) are prioritized.
        If all servers are in cooldown, falls back to the server that failed longest ago.
        """
        with self._lock:
            eligible = [n for n in self.nodes if n.is_eligible(self.cooldown_seconds)]
            if eligible:
                return sorted(eligible, key=lambda n: n.priority)

            # Fallback: all servers cooling down; try server whose failure was longest ago
            logger.warning(
                f"All {len(self.nodes)} Ollama server(s) are in cooldown; probing oldest failed server."
            )
            return sorted(self.nodes, key=lambda n: n.failed_at or 0.0)

    def get_candidate_servers(self) -> List[OllamaServerNode]:
        """
        Return servers ordered for execution:
        1. Eligible (healthy) servers ordered primarily by least active tasks (load balancing),
           and secondarily by priority ascending.
        2. Cooling down servers as fallback ordered by oldest failure.
        """
        with self._lock:
            eligible = [n for n in self.nodes if n.is_eligible(self.cooldown_seconds)]
            if eligible:
                sorted_eligible = sorted(
                    eligible,
                    key=lambda n: (n.active_tasks, n.priority),
                )
                cooling = [n for n in self.nodes if not n.is_eligible(self.cooldown_seconds)]
                sorted_cooling = sorted(cooling, key=lambda n: n.failed_at or 0.0)
                return sorted_eligible + sorted_cooling
            else:
                return sorted(self.nodes, key=lambda n: n.failed_at or 0.0)

    @property
    def primary_server(self) -> OllamaServerNode:
        """Return the highest priority currently eligible server."""
        ordered = self.get_ordered_servers()
        return ordered[0]

    def execute_with_failover(
        self,
        operation: Callable[[str], Any],
        on_failover: Optional[Callable[[OllamaServerNode, Exception, OllamaServerNode], None]] = None,
    ) -> Any:
        """
        Execute an operation passing base_url.
        Balances concurrent tasks across least-loaded alive servers,
        and automatically retries on the next candidate if any server fails.
        """
        candidates = self.get_candidate_servers()
        last_exception: Optional[Exception] = None
        errors = []

        for idx, node in enumerate(candidates):
            with self._lock:
                node.active_tasks += 1
            try:
                result = operation(node.url)
                with self._lock:
                    node.mark_success()
                return result
            except Exception as e:
                last_exception = e
                errors.append(f"{node.url}: {e}")
                with self._lock:
                    node.mark_failure(str(e))
                logger.warning(
                    f"Ollama server '{node.url}' ({node.label}, Priority {node.priority}) failed: {e}. "
                    f"Marked cooling down for {self.cooldown_seconds}s."
                )

                if idx + 1 < len(candidates):
                    next_node = candidates[idx + 1]
                    logger.warning(
                        f"Failing over to server '{next_node.url}' ({next_node.label}, Priority {next_node.priority})..."
                    )
                    if on_failover:
                        on_failover(node, e, next_node)
            finally:
                with self._lock:
                    node.active_tasks = max(0, node.active_tasks - 1)

        err_summary = "; ".join(errors)
        raise RuntimeError(
            f"All {len(self.nodes)} Ollama server(s) failed. Attempts: [{err_summary}]"
        ) from last_exception

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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with multi-server failover."""
        def _embed(url: str):
            client = OllamaEmbeddings(base_url=url, model=self.model)
            return client.embed_documents(texts)

        return self.pool.execute_with_failover(_embed)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text with multi-server failover."""
        def _embed(url: str):
            client = OllamaEmbeddings(base_url=url, model=self.model)
            return client.embed_query(text)

        return self.pool.execute_with_failover(_embed)
