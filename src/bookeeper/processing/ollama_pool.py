"""
Multi-server smart proxy and failover client for Ollama with priority routing and automatic recovery.
"""

import json
import logging
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from bookeeper.config import DEFAULT_CAPABILITIES, OllamaServerConfig

logger = logging.getLogger(__name__)


def send_wake_on_lan(
    mac_address: str,
    broadcast_ip: str = "255.255.255.255",
    port: int = 9,
    secret: Optional[str] = None,
) -> bool:
    """
    Send Wake-on-LAN magic packet over UDP broadcast to power on a remote Ollama server.
    Magic packet consists of 6 bytes of 0xFF followed by 16 repetitions of the target MAC address (102 bytes total),
    plus an optional SecureOn password/secret appended at the end (4 or 6 bytes).
    """
    clean_mac = re.sub(r"[^0-9a-fA-F]", "", mac_address or "")
    if len(clean_mac) != 12:
        raise ValueError(f"Invalid MAC address for Wake-on-LAN: '{mac_address}'")

    mac_bytes = bytes.fromhex(clean_mac)
    packet = bytearray(b"\xff" * 6 + mac_bytes * 16)

    if secret:
        # Check IPv4 dotted-quad format (4 bytes)
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", secret.strip()):
            packet.extend(socket.inet_aton(secret.strip()))
        else:
            clean_secret = re.sub(r"[^0-9a-fA-F]", "", secret)
            if len(clean_secret) == 12:  # 6-byte hex password (standard SecureOn)
                packet.extend(bytes.fromhex(clean_secret))
            elif len(clean_secret) == 8:  # 4-byte hex password
                packet.extend(bytes.fromhex(clean_secret))
            else:
                raw = secret.encode("utf-8")
                if len(raw) <= 6:
                    packet.extend(raw.ljust(6, b"\x00"))
                else:
                    raise ValueError(
                        f"Invalid WOL secret '{secret}': expected 6 hex bytes (12 hex chars), "
                        f"4 hex bytes (8 hex chars), or up to 6 ASCII characters."
                    )

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(bytes(packet), (broadcast_ip, port))
    logger.info(f"Broadcasted Wake-on-LAN magic packet for MAC {mac_address} to {broadcast_ip}:{port}")
    return True


@dataclass
class OllamaServerNode:
    """Tracks dynamic health, priority, capabilities, active tasks, and cooldown for an individual Ollama endpoint."""

    url: str
    priority: int = 1
    name: Optional[str] = None
    capabilities: List[str] = field(default_factory=lambda: list(DEFAULT_CAPABILITIES))
    mac: Optional[str] = None
    wol_broadcast: str = "255.255.255.255"
    wol_port: int = 9
    wol_secret: Optional[str] = None
    failed_at: Optional[float] = None
    failure_count: int = 0
    last_error: Optional[str] = None
    active_tasks: int = 0
    last_task_start: Optional[float] = None

    # Workload and effort statistics
    total_tasks: int = 0
    success_tasks: int = 0
    failed_tasks: int = 0
    success_chars: int = 0
    total_duration: float = 0.0
    success_duration: float = 0.0
    stats_by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def wake(self) -> bool:
        """Trigger Wake-on-LAN if node has a MAC address configured."""
        if not self.mac:
            return False
        return send_wake_on_lan(
            self.mac,
            broadcast_ip=self.wol_broadcast,
            port=self.wol_port,
            secret=self.wol_secret,
        )

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

    def record_task_attempt(
        self,
        duration: float,
        success: bool,
        chars: int = 0,
        task_type: str = "llm",
    ) -> None:
        """Record the outcome, execution time, and characters for a task on this server."""
        self.total_tasks += 1
        self.total_duration += max(0.0, duration)
        tt = (task_type or "llm").strip().lower()

        if tt not in self.stats_by_type:
            self.stats_by_type[tt] = {
                "total_tasks": 0,
                "success_tasks": 0,
                "failed_tasks": 0,
                "success_chars": 0,
                "total_duration": 0.0,
                "success_duration": 0.0,
            }

        sub = self.stats_by_type[tt]
        sub["total_tasks"] += 1
        sub["total_duration"] += max(0.0, duration)

        if success:
            self.success_tasks += 1
            self.success_duration += max(0.0, duration)
            self.success_chars += max(0, chars)
            sub["success_tasks"] += 1
            sub["success_duration"] += max(0.0, duration)
            sub["success_chars"] += max(0, chars)
        else:
            self.failed_tasks += 1
            sub["failed_tasks"] += 1

    @property
    def avg_duration(self) -> float:
        """Average duration across all task attempts on this server."""
        return (self.total_duration / self.total_tasks) if self.total_tasks > 0 else 0.0

    @property
    def avg_success_duration(self) -> float:
        """Average duration across successful tasks on this server."""
        return (self.success_duration / self.success_tasks) if self.success_tasks > 0 else 0.0

    def reset_statistics(self) -> None:
        """Reset workload statistics counters."""
        self.total_tasks = 0
        self.success_tasks = 0
        self.failed_tasks = 0
        self.success_chars = 0
        self.total_duration = 0.0
        self.success_duration = 0.0
        self.stats_by_type.clear()


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
        "errno 65",          # macOS No route to host
        "remotedisconnected",
        "remote end closed connection",
        "connection reset by peer",
        "connection timed out",
        "connecttimeout",
        "connecterror",
        "could not resolve host",
        "nodename nor servname provided",
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
                mac=getattr(s, "mac", None),
                wol_broadcast=getattr(s, "wol_broadcast", "255.255.255.255"),
                wol_port=getattr(s, "wol_port", 9),
                wol_secret=getattr(s, "wol_secret", None),
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
        input_chars: int = 0,
        task_type: Optional[str] = None,
    ) -> Any:
        """
        Execute an operation passing base_url.
        Enforces sequential submission/execution per Ollama server (at most max_tasks_per_server per node),
        routes across idle servers by priority, retries on transient errors, and automatically fails over.
        Optionally filters candidate servers by required capability ('llm', 'embedding', 'verification').
        Tracks execution efforts, task counts, durations, and valuable characters processed per node.
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
                                n.record_task_attempt(
                                    duration=elapsed,
                                    success=False,
                                    chars=0,
                                    task_type=task_type or capability or "llm",
                                )
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
                    t_attempt_start = time.perf_counter()
                    try:
                        _thread_local.last_used_server = selected_node.label
                        _thread_local.last_used_server_url = selected_node.url
                        result = operation(selected_node.url)
                        dur = time.perf_counter() - t_attempt_start
                        with self._condition:
                            selected_node.mark_success()
                            selected_node.record_task_attempt(
                                duration=dur,
                                success=True,
                                chars=input_chars,
                                task_type=task_type or capability or "llm",
                            )
                        return result
                    except Exception as e:
                        dur = time.perf_counter() - t_attempt_start
                        with self._condition:
                            selected_node.record_task_attempt(
                                duration=dur,
                                success=False,
                                chars=0,
                                task_type=task_type or capability or "llm",
                            )
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
                            mac_hint = ""
                            if sys.platform == "darwin" and any(p in str(e).lower() for p in ["errno 65", "no route to host"]):
                                mac_hint = " [macOS Tip: Check System Settings ➔ Privacy & Security ➔ Local Network to ensure your Terminal/iTerm2 app is allowed]"
                            logger.warning(
                                f"Ollama server '{selected_node.url}' attempt {attempt + 1}/{max_attempts} {err_tag} ({e}){mac_hint}. "
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

    def get_server_statistics(self) -> List[Dict[str, Any]]:
        """Return snapshot list of workload, task counts, and effort statistics for all pool nodes."""
        with self._lock:
            stats_list = []
            for n in sorted(self.nodes, key=lambda x: x.priority):
                stats_list.append(
                    {
                        "name": n.label,
                        "url": n.url,
                        "priority": n.priority,
                        "capabilities": list(n.capabilities),
                        "total_tasks": n.total_tasks,
                        "success_tasks": n.success_tasks,
                        "failed_tasks": n.failed_tasks,
                        "success_chars": n.success_chars,
                        "total_duration": n.total_duration,
                        "success_duration": n.success_duration,
                        "avg_duration": n.avg_duration,
                        "avg_success_duration": n.avg_success_duration,
                        "stats_by_type": {k: dict(v) for k, v in n.stats_by_type.items()},
                    }
                )
            return stats_list

    def reset_server_statistics(self) -> None:
        """Reset workload and effort statistics across all nodes."""
        with self._lock:
            for n in self.nodes:
                n.reset_statistics()


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

            batch_chars = sum(len(t) for t in batch_texts)
            t0 = time.perf_counter()
            result = self.pool.execute_with_failover(
                _embed,
                capability="embedding",
                input_chars=batch_chars,
                task_type="embedding",
            )
            elapsed = time.perf_counter() - t0
            with self._lock:
                self.total_embedded_texts += len(batch_texts)
                self.total_embedded_chars += batch_chars
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

        query_chars = len(text)
        t0 = time.perf_counter()
        result = self.pool.execute_with_failover(
            _embed,
            capability="embedding",
            input_chars=query_chars,
            task_type="embedding",
        )
        elapsed = time.perf_counter() - t0
        with self._lock:
            self.total_embedded_texts += 1
            self.total_embedded_chars += query_chars
            self.total_embedding_seconds += elapsed
        return result


def format_duration_friendly(seconds: float) -> str:
    """Format duration in seconds into a human-readable string (e.g. 12.3s, 5m 20s, 1h 14m 02s)."""
    if seconds <= 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.2f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h = int(m // 60)
    rem_m = int(m % 60)
    return f"{h}h {rem_m:02d}m {s:02d}s"


def format_chars_friendly(chars: int) -> str:
    """Format character count into a human-readable string (e.g. 485,120 (473.8 KB))."""
    if chars <= 0:
        return "0 chars"
    if chars < 1024:
        return f"{chars:,} chars"
    elif chars < 1024 * 1024:
        kb = chars / 1024.0
        return f"{chars:,} ({kb:.1f} KB)"
    else:
        mb = chars / (1024.0 * 1024.0)
        return f"{chars:,} ({mb:.2f} MB)"


def aggregate_server_statistics(pools: Sequence[OllamaPool]) -> List[Dict[str, Any]]:
    """
    Consolidate workload and effort metrics across multiple OllamaPool instances
    (e.g. LLM pool, embedding pool, verification pool), grouped by server URL.
    """
    aggregated: Dict[str, Dict[str, Any]] = {}
    for pool in pools:
        if not pool:
            continue
        for node_stat in pool.get_server_statistics():
            url = node_stat["url"]
            if url not in aggregated:
                aggregated[url] = {
                    "name": node_stat["name"],
                    "url": url,
                    "priority": node_stat["priority"],
                    "capabilities": set(node_stat.get("capabilities", [])),
                    "total_tasks": 0,
                    "success_tasks": 0,
                    "failed_tasks": 0,
                    "success_chars": 0,
                    "total_duration": 0.0,
                    "success_duration": 0.0,
                    "stats_by_type": {},
                }
            entry = aggregated[url]
            entry["capabilities"].update(node_stat.get("capabilities", []))
            entry["priority"] = min(entry["priority"], node_stat.get("priority", 1))
            entry["total_tasks"] += node_stat.get("total_tasks", 0)
            entry["success_tasks"] += node_stat.get("success_tasks", 0)
            entry["failed_tasks"] += node_stat.get("failed_tasks", 0)
            entry["success_chars"] += node_stat.get("success_chars", 0)
            entry["total_duration"] += node_stat.get("total_duration", 0.0)
            entry["success_duration"] += node_stat.get("success_duration", 0.0)

            for t_type, t_st in node_stat.get("stats_by_type", {}).items():
                if t_type not in entry["stats_by_type"]:
                    entry["stats_by_type"][t_type] = {
                        "total_tasks": 0,
                        "success_tasks": 0,
                        "failed_tasks": 0,
                        "success_chars": 0,
                        "total_duration": 0.0,
                    }
                sub = entry["stats_by_type"][t_type]
                sub["total_tasks"] += t_st.get("total_tasks", 0)
                sub["success_tasks"] += t_st.get("success_tasks", 0)
                sub["failed_tasks"] += t_st.get("failed_tasks", 0)
                sub["success_chars"] += t_st.get("success_chars", 0)
                sub["total_duration"] += t_st.get("total_duration", 0.0)

    result = []
    for url, entry in aggregated.items():
        tot = entry["total_tasks"]
        dur = entry["total_duration"]
        entry["capabilities"] = sorted(entry["capabilities"])
        entry["avg_duration"] = (dur / tot) if tot > 0 else 0.0
        result.append(entry)

    return sorted(result, key=lambda x: x["priority"])


def format_server_stats_table(
    stats: List[Dict[str, Any]],
    title: str = "Ollama Server Workload & Effort Statistics",
) -> Any:
    """
    Build a Rich Table presenting per-server job and effort statistics:
    - Valuable/accepted/successfully processed characters
    - Success tasks vs total tasks (with completion rate)
    - Average time and total time per server
    - Task type breakdown (LLM, Embedding, Verification)
    """
    from rich.table import Table

    table = Table(title=title, border_style="cyan")
    table.add_column("Server Node", style="bold", min_width=20)
    table.add_column("Capabilities", style="dim", justify="center")
    table.add_column("Success Tasks", justify="right", style="green")
    table.add_column("Total Tasks", justify="right")
    table.add_column("Valuable Chars", justify="right", style="cyan")
    table.add_column("Avg Time", justify="right", style="bold green")
    table.add_column("Total Time", justify="right", style="bold yellow")
    table.add_column("Task Breakdown", style="dim", min_width=22)

    total_success = 0
    total_tasks = 0
    total_chars = 0
    total_time = 0.0

    for s in stats:
        suc = s["success_tasks"]
        tot = s["total_tasks"]
        chars = s["success_chars"]
        dur = s["total_duration"]
        avg_dur = s.get("avg_duration", 0.0)

        total_success += suc
        total_tasks += tot
        total_chars += chars
        total_time += dur

        server_label = f"{s['name']}\n[dim]{s['url']}[/dim]"
        caps_str = ", ".join(s.get("capabilities", [])) or "all"

        if tot > 0:
            rate = (suc / tot * 100.0) if tot > 0 else 0.0
            rate_color = "green" if rate == 100.0 else ("yellow" if rate >= 80.0 else "red")
            tot_str = f"{tot} [{rate_color}]({rate:.1f}%)[/{rate_color}]"
            suc_str = f"[bold green]{suc:,}[/bold green]"
            chars_str = f"[bold cyan]{format_chars_friendly(chars)}[/bold cyan]"
            avg_str = f"{avg_dur:.2f}s"
            time_str = format_duration_friendly(dur)
        else:
            tot_str = "[dim]0 (--)[/dim]"
            suc_str = "[dim]0[/dim]"
            chars_str = "[dim]0 chars[/dim]"
            avg_str = "[dim]--[/dim]"
            time_str = "[dim]0.0s[/dim]"

        # Task breakdown string
        breakdown_parts = []
        for tt, sub in s.get("stats_by_type", {}).items():
            sub_tot = sub.get("total_tasks", 0)
            sub_suc = sub.get("success_tasks", 0)
            sub_chars = sub.get("success_chars", 0)
            if sub_tot > 0:
                breakdown_parts.append(
                    f"{tt.upper()}: {sub_suc}/{sub_tot} ({format_chars_friendly(sub_chars)})"
                )
        breakdown_str = "\n".join(breakdown_parts) if breakdown_parts else ("[dim]Idle[/dim]" if tot == 0 else "[dim]LLM[/dim]")

        table.add_row(
            server_label,
            caps_str,
            suc_str,
            tot_str,
            chars_str,
            avg_str,
            time_str,
            breakdown_str,
        )

    # Summary row
    table.add_section()
    overall_rate = (total_success / total_tasks * 100.0) if total_tasks > 0 else 0.0
    overall_avg = (total_time / total_tasks) if total_tasks > 0 else 0.0
    rate_color = "green" if overall_rate == 100.0 else ("yellow" if overall_rate >= 80.0 else "red")
    overall_tot_str = f"[bold]{total_tasks:,}[/bold] [{rate_color}]({overall_rate:.1f}%)[/{rate_color}]" if total_tasks > 0 else "0"

    table.add_row(
        f"[bold]Total Pool Effort ({len(stats)} nodes)[/bold]",
        "-",
        f"[bold green]{total_success:,}[/bold green]",
        overall_tot_str,
        f"[bold cyan]{format_chars_friendly(total_chars)}[/bold cyan]",
        f"[bold green]{overall_avg:.2f}s[/bold green]" if total_tasks > 0 else "--",
        f"[bold yellow]{format_duration_friendly(total_time)}[/bold yellow]",
        f"[bold]{total_success} success / {total_tasks} total[/bold]",
    )

    return table
