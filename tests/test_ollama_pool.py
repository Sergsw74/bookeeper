"""
Comprehensive unit tests for multi-server Ollama pool, priority failover, and cooldown recovery.
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from bookeeper.config import OllamaServerConfig, Settings
from bookeeper.processing.extractor import BookMetadata, Concept, KnowledgeExtractor, SectionExtraction
from bookeeper.processing.ollama_pool import FailoverOllamaEmbeddings, OllamaPool, OllamaServerNode


def test_ollama_server_config_and_settings_resolution():
    """Verify single-server fallback and multi-server config parsing."""
    # Default settings fallback to ollama_base_url
    s1 = Settings(ollama_base_url="http://node-alpha:11434")
    assert len(s1.resolved_ollama_servers) == 1
    assert s1.resolved_ollama_servers[0].url == "http://node-alpha:11434"
    assert s1.resolved_ollama_servers[0].priority == 1
    assert s1.failover_cooldown_seconds == 600

    # Multi-server explicit list
    s2 = Settings(
        ollama_servers=[
            {"url": "http://node-backup:11434", "priority": 2, "name": "backup"},
            {"url": "http://node-primary:11434", "priority": 1, "name": "primary"},
        ],
        failover_cooldown_seconds=300,
    )
    assert len(s2.resolved_ollama_servers) == 2
    assert s2.resolved_ollama_servers[0].url == "http://node-primary:11434"
    assert s2.resolved_ollama_servers[0].priority == 1
    assert s2.resolved_ollama_servers[1].url == "http://node-backup:11434"
    assert s2.resolved_ollama_servers[1].priority == 2


def test_ollama_pool_priority_ordering():
    """Verify that pool always orders servers by priority ascending."""
    servers = [
        OllamaServerConfig(url="http://srv-c:11434", priority=3, name="C"),
        OllamaServerConfig(url="http://srv-a:11434", priority=1, name="A"),
        OllamaServerConfig(url="http://srv-b:11434", priority=2, name="B"),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    assert pool.primary_server.url == "http://srv-a:11434"
    ordered = pool.get_ordered_servers()
    assert [s.url for s in ordered] == [
        "http://srv-a:11434",
        "http://srv-b:11434",
        "http://srv-c:11434",
    ]


def test_ollama_pool_failover_execution_success_on_primary():
    """When primary server succeeds, secondary servers are never invoked."""
    servers = [
        OllamaServerConfig(url="http://node1:11434", priority=1),
        OllamaServerConfig(url="http://node2:11434", priority=2),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    called_urls = []

    def mock_fn(url: str):
        called_urls.append(url)
        return f"result-from-{url}"

    res = pool.execute_with_failover(mock_fn)
    assert res == "result-from-http://node1:11434"
    assert called_urls == ["http://node1:11434"]
    assert pool.nodes[0].consecutive_failures == 0
    assert pool.nodes[0].failed_at is None


def test_ollama_pool_failover_to_secondary():
    """When primary server fails, request immediately fails over to secondary server."""
    servers = [
        OllamaServerConfig(url="http://node1:11434", priority=1, name="node1"),
        OllamaServerConfig(url="http://node2:11434", priority=2, name="node2"),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    called_urls = []

    def mock_fn(url: str):
        called_urls.append(url)
        if "node1" in url:
            raise ConnectionError("node1 timeout or unreachable")
        return "success-node2"

    res = pool.execute_with_failover(mock_fn)
    assert res == "success-node2"
    assert called_urls == ["http://node1:11434", "http://node2:11434"]

    # node1 should now be recorded as failed and cooling down
    node1 = pool.nodes[0]
    assert node1.consecutive_failures == 1
    assert node1.failed_at is not None
    assert not node1.is_eligible(600)

    # Next call should route directly to node2 without hitting node1
    called_urls.clear()
    res2 = pool.execute_with_failover(mock_fn)
    assert res2 == "success-node2"
    assert called_urls == ["http://node2:11434"]


def test_ollama_pool_cooldown_expiration():
    """After cooldown period (e.g. 600s / 10m), previously failed server is retried."""
    servers = [
        OllamaServerConfig(url="http://node1:11434", priority=1),
        OllamaServerConfig(url="http://node2:11434", priority=2),
    ]
    # Set a 600s cooldown
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    node1 = pool.nodes[0]
    # Simulate node1 failed 300 seconds ago (5 min ago -> still in cooldown)
    current_time = 10000.0
    with patch("time.time", return_value=current_time):
        node1.mark_failure("Mock timeout")

    assert not node1.is_eligible(cooldown_seconds=600, now=current_time + 300)

    # Simulate node1 after 601 seconds (10 min + 1 sec -> cooldown expired)
    assert node1.is_eligible(cooldown_seconds=600, now=current_time + 601)

    # Calling get_ordered_servers after cooldown expiry should put node1 back at the head
    with patch("time.time", return_value=current_time + 601):
        ordered = pool.get_ordered_servers()
        assert ordered[0].url == "http://node1:11434"


def test_ollama_pool_all_servers_fail():
    """If all servers fail, raise RuntimeError with detailed per-server failure breakdown."""
    servers = [
        OllamaServerConfig(url="http://node1:11434", priority=1),
        OllamaServerConfig(url="http://node2:11434", priority=2),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    def fail_all(url: str):
        raise ConnectionRefusedError(f"Connection refused at {url}")

    with pytest.raises(RuntimeError) as exc_info:
        pool.execute_with_failover(fail_all)

    err_str = str(exc_info.value)
    assert "All 2 Ollama server" in err_str
    assert "http://node1:11434" in err_str
    assert "http://node2:11434" in err_str


def test_ollama_pool_all_cooling_down_fallback():
    """When all servers are in cooldown, pool falls back to the oldest failed server rather than freezing."""
    servers = [
        OllamaServerConfig(url="http://node1:11434", priority=1),
        OllamaServerConfig(url="http://node2:11434", priority=2),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    t0 = 1000.0
    with patch("time.time", return_value=t0):
        pool.nodes[0].mark_failure("node1 error")
    with patch("time.time", return_value=t0 + 10):
        pool.nodes[1].mark_failure("node2 error")

    # At t0 + 20, both are within 600s cooldown
    with patch("time.time", return_value=t0 + 20):
        ordered = pool.get_ordered_servers()
        assert len(ordered) == 2
        # node1 failed earlier (t0 vs t0+10), so it should be prioritized for retry
        assert ordered[0].url == "http://node1:11434"


def test_failover_ollama_embeddings():
    """Verify FailoverOllamaEmbeddings executes embeddings through the failover pool."""
    servers = [
        OllamaServerConfig(url="http://node1:11434", priority=1),
        OllamaServerConfig(url="http://node2:11434", priority=2),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)
    embeddings = FailoverOllamaEmbeddings(pool=pool, model="nomic-embed-text")

    # Mock OllamaEmbeddings inside FailoverOllamaEmbeddings
    with patch("bookeeper.processing.ollama_pool.OllamaEmbeddings") as mock_emb_cls:
        instance1 = MagicMock()
        instance1.embed_documents.side_effect = Exception("node1 embedding failure")
        instance1.embed_query.side_effect = Exception("node1 query failure")

        instance2 = MagicMock()
        instance2.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        instance2.embed_query.return_value = [0.1, 0.2, 0.3]

        def get_instance(base_url, model):
            if "node1" in base_url:
                return instance1
            return instance2

        mock_emb_cls.side_effect = get_instance

        # Test embed_documents with failover
        docs_res = embeddings.embed_documents(["hello world"])
        assert docs_res == [[0.1, 0.2, 0.3]]
        assert pool.nodes[0].consecutive_failures == 1

        # Test embed_query routes to eligible node2
        query_res = embeddings.embed_query("hello world")
        assert query_res == [0.1, 0.2, 0.3]


def test_knowledge_extractor_with_failover_pool():
    """Verify KnowledgeExtractor executes clean_metadata with failover across servers."""
    settings = Settings(
        ollama_servers=[
            {"url": "http://node-faulty:11434", "priority": 1, "name": "faulty"},
            {"url": "http://node-healthy:11434", "priority": 2, "name": "healthy"},
        ],
        failover_cooldown_seconds=600,
    )
    extractor = KnowledgeExtractor.from_settings(settings)
    assert len(extractor.pool.nodes) == 2

    # Mock _execute_structured_invoke or ChatOllama
    with patch("bookeeper.processing.extractor.ChatOllama") as mock_chat_cls:
        instance_faulty = MagicMock()
        instance_faulty.with_structured_output.return_value.invoke.side_effect = ConnectionError("Node faulty down")

        expected_meta = BookMetadata(
            title="Clean Architecture",
            author="Robert C. Martin",
            summary="A comprehensive guide to software architecture and craftsmanship.",
        )
        instance_healthy = MagicMock()
        instance_healthy.with_structured_output.return_value.invoke.return_value = expected_meta

        def get_chat_instance(base_url, model, temperature, **kwargs):
            if "faulty" in base_url:
                return instance_faulty
            return instance_healthy

        mock_chat_cls.side_effect = get_chat_instance

        cleaned = extractor.clean_metadata(
            raw_title="clean architecture",
            raw_authors=["martin"],
            raw_comments=None,
        )

        assert cleaned.title == "Clean Architecture"
        assert cleaned.author == "Robert C. Martin"
        # Node faulty was marked failed
        assert extractor.pool.nodes[0].consecutive_failures == 1
        assert extractor.pool.nodes[0].failed_at is not None


def test_pool_concurrency_calculation():
    """Verify concurrency calculation: low = num_servers, default = 3x servers, cap = 10."""
    s = Settings()
    # 1 server -> 3 active tasks
    assert s.calculate_pool_concurrency(num_servers=1) == 3
    # 2 servers -> 6 active tasks
    assert s.calculate_pool_concurrency(num_servers=2) == 6
    # 3 servers -> 9 active tasks
    assert s.calculate_pool_concurrency(num_servers=3) == 9
    # 4 servers -> capped at 10
    assert s.calculate_pool_concurrency(num_servers=4) == 10

    # Custom override
    s_custom = Settings(max_active_tasks=5)
    assert s_custom.calculate_pool_concurrency(num_servers=2) == 5
    assert s_custom.calculate_pool_concurrency(num_servers=4) == 5


def test_concurrent_load_balancing_across_servers():
    """Verify tasks running in parallel are distributed across all available/alive servers."""
    from concurrent.futures import ThreadPoolExecutor

    servers = [
        OllamaServerConfig(url="http://node-a:11434", priority=1, name="node-a"),
        OllamaServerConfig(url="http://node-b:11434", priority=2, name="node-b"),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    started_events = {
        "http://node-a:11434": False,
        "http://node-b:11434": False,
    }
    server_calls = []

    import threading
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def slow_task(url: str):
        with lock:
            server_calls.append(url)
            started_events[url] = True
        # Wait until both tasks are in flight concurrently
        barrier.wait(timeout=2.0)
        return f"done-{url}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(pool.execute_with_failover, slow_task)
        f2 = executor.submit(pool.execute_with_failover, slow_task)
        res1 = f1.result(timeout=3.0)
        res2 = f2.result(timeout=3.0)

    # Both servers should have been utilized concurrently!
    assert "done-http://node-a:11434" in (res1, res2)
    assert "done-http://node-b:11434" in (res1, res2)
    assert set(server_calls) == {"http://node-a:11434", "http://node-b:11434"}

    # After completion, active_tasks on both nodes must return to 0
    assert pool.nodes[0].active_tasks == 0
    assert pool.nodes[1].active_tasks == 0


def test_active_tasks_tracking_with_error_and_failover():
    """Verify active_tasks resets to 0 even when an exception occurs or failover triggers."""
    servers = [
        OllamaServerConfig(url="http://node-fail:11434", priority=1, name="fail"),
        OllamaServerConfig(url="http://node-ok:11434", priority=2, name="ok"),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    def op(url: str):
        if "fail" in url:
            raise ConnectionError("node down")
        return "success"

    res = pool.execute_with_failover(op)
    assert res == "success"
    # Both nodes must have 0 active_tasks
    assert pool.nodes[0].active_tasks == 0
    assert pool.nodes[1].active_tasks == 0
    assert pool.nodes[0].consecutive_failures == 1


def test_sequential_execution_prevents_mutual_execution_on_same_server():
    """Verify that multiple active pipeline tasks targeting a single server execute sequentially one by one, never mutually executing."""
    from concurrent.futures import ThreadPoolExecutor

    servers = [
        OllamaServerConfig(url="http://node-single:11434", priority=1, name="single"),
    ]
    # Pool enforces max_tasks_per_server = 1 (sequential per server)
    pool = OllamaPool(servers=servers, cooldown_seconds=600, max_tasks_per_server=1)

    execution_intervals = []
    active_concurrent_count = 0
    max_observed_concurrent = 0
    lock = threading.Lock()

    def task_worker(task_id: int):
        nonlocal active_concurrent_count, max_observed_concurrent

        def _op(url: str):
            nonlocal active_concurrent_count, max_observed_concurrent
            with lock:
                active_concurrent_count += 1
                if active_concurrent_count > max_observed_concurrent:
                    max_observed_concurrent = active_concurrent_count
            start_t = time.time()
            time.sleep(0.05)  # Simulate LLM inference
            end_t = time.time()
            with lock:
                active_concurrent_count -= 1
                execution_intervals.append((task_id, start_t, end_t))
            return f"res-{task_id}"

        return pool.execute_with_failover(_op)

    # Launch 3 pipeline tasks simultaneously (fast invocation prefetch)
    with ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(task_worker, 1)
        f2 = executor.submit(task_worker, 2)
        f3 = executor.submit(task_worker, 3)
        res1 = f1.result(timeout=3.0)
        res2 = f2.result(timeout=3.0)
        res3 = f3.result(timeout=3.0)

    assert res1 == "res-1"
    assert res2 == "res-2"
    assert res3 == "res-3"

    # Crucial assertion: exactly 1 task executed on the server at any moment! (no mutual execution)
    assert max_observed_concurrent == 1

    # Verify all intervals are disjoint (strictly sequential execution)
    sorted_intervals = sorted(execution_intervals, key=lambda x: x[1])
    for i in range(len(sorted_intervals) - 1):
        prev_end = sorted_intervals[i][2]
        curr_start = sorted_intervals[i + 1][1]
        assert curr_start >= prev_end, f"Tasks {sorted_intervals[i][0]} and {sorted_intervals[i+1][0]} overlapped!"


def test_embedding_base_url_routing_separate_from_llm_pool():
    """Verify that embedding_base_url routes embeddings to local machine without touching remote LLM server."""
    from bookeeper.processing.deduplicator import ConceptDeduplicator
    from bookeeper.rag.lightrag_engine import LightRAGEngine

    settings = Settings(
        ollama_base_url="http://remote-gpu:11434",
        embedding_base_url="http://localhost:11434",
        embedding_model="nomic-embed-text",
        llm_model="llama3.1:8b",
    )

    # 1. Extractor LLM pool must target remote-gpu
    extractor = KnowledgeExtractor.from_settings(settings)
    assert len(extractor.pool.nodes) == 1
    assert extractor.pool.nodes[0].url == "http://remote-gpu:11434"

    # 2. FailoverOllamaEmbeddings must target localhost
    embeddings = FailoverOllamaEmbeddings.from_settings(settings)
    assert len(embeddings.pool.nodes) == 1
    assert embeddings.pool.nodes[0].url == "http://localhost:11434"
    assert embeddings.model == "nomic-embed-text"

    # 3. ConceptDeduplicator must use local embeddings
    deduplicator = ConceptDeduplicator.from_settings(settings)
    assert deduplicator.embeddings.pool.nodes[0].url == "http://localhost:11434"

    # 4. LightRAGEngine must use remote pool for LLM and local pool for embeddings
    engine = LightRAGEngine.from_settings(settings, working_dir="/tmp/test_lightrag")
    assert engine.pool.nodes[0].url == "http://remote-gpu:11434"
    assert engine.embedding_pool.nodes[0].url == "http://localhost:11434"

    # 5. Fallback behavior: if embedding_base_url is None, both default to ollama_base_url
    fallback_settings = Settings(ollama_base_url="http://remote-gpu:11434")
    assert fallback_settings.uses_llm_pool_for_embeddings is True
    assert fallback_settings.resolved_embedding_servers[0].url == "http://remote-gpu:11434"
    fallback_emb = FailoverOllamaEmbeddings.from_settings(fallback_settings)
    assert fallback_emb.pool.nodes[0].url == "http://remote-gpu:11434"

    # 6. 'pool' keyword behavior: explicitly sharing multi-server LLM pool
    pool_settings = Settings(
        ollama_servers=[
            OllamaServerConfig(url="http://node1:11434", priority=1),
            OllamaServerConfig(url="http://node2:11434", priority=2),
        ],
        embedding_base_url="pool",
    )
    assert pool_settings.uses_llm_pool_for_embeddings is True
    assert len(pool_settings.resolved_embedding_servers) == 2
    assert pool_settings.resolved_embedding_servers[0].url == "http://node1:11434"
    assert pool_settings.resolved_embedding_servers[1].url == "http://node2:11434"

    pool_emb = FailoverOllamaEmbeddings.from_settings(pool_settings)
    assert len(pool_emb.pool.nodes) == 2
    assert pool_emb.pool.nodes[0].url == "http://node1:11434"

    # Case insensitivity and whitespace handling
    assert Settings(embedding_base_url="POOL").uses_llm_pool_for_embeddings is True
    assert Settings(embedding_base_url=" pool ").uses_llm_pool_for_embeddings is True

    # CLI override --embedding-url pool
    from bookeeper.cli import _get_effective_settings
    cli_cfg = _get_effective_settings(None, embedding_url="pool")
    assert cli_cfg.uses_llm_pool_for_embeddings is True
    assert cli_cfg.embedding_base_url == "pool"
    assert cli_cfg.resolved_embedding_servers[0].url == cli_cfg.ollama_base_url


def test_failover_ollama_embeddings_speed_tracking():
    """Verify that FailoverOllamaEmbeddings measures texts, characters, duration, and speeds."""
    pool = OllamaPool(servers=[OllamaServerConfig(url="http://node1:11434", priority=1)])
    embeddings = FailoverOllamaEmbeddings(pool=pool, model="nomic-embed-text")

    # Initial stats should be zero
    init_stats = embeddings.embedding_stats
    assert init_stats["total_texts"] == 0
    assert init_stats["total_chars"] == 0
    assert init_stats["speed_texts_per_sec"] == 0.0

    mock_vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    with patch("bookeeper.processing.ollama_pool.OllamaEmbeddings") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = mock_vectors
        mock_cls.return_value = mock_instance

        res = embeddings.embed_documents(["alpha", "beta", "gamma"])
        assert res == mock_vectors

    stats = embeddings.embedding_stats
    assert stats["total_texts"] == 3
    assert stats["total_chars"] == len("alpha") + len("beta") + len("gamma")
    assert stats["total_seconds"] > 0
    assert stats["speed_texts_per_sec"] > 0
    assert stats["speed_chars_per_sec"] > 0

    # Reset stats
    embeddings.reset_stats()
    reset_stats = embeddings.embedding_stats
    assert reset_stats["total_texts"] == 0
    assert reset_stats["speed_texts_per_sec"] == 0.0


def test_ollama_pool_retries_transient_failures_on_same_node():
    """Verify that execute_with_failover retries transient failures on the same server before succeeding."""
    servers = [OllamaServerConfig(url="http://node1:11434", priority=1)]
    pool = OllamaPool(servers=servers)

    attempts = 0

    def flaky_operation(url: str):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError(f"Temporary timeout on {url}")
        return "flaky-success"

    res = pool.execute_with_failover(flaky_operation, retries=3, backoff_base=0.01)
    assert res == "flaky-success"
    assert attempts == 3
    # Server should be marked success, not failed
    assert pool.nodes[0].consecutive_failures == 0


def test_ollama_pool_retries_all_attempts_before_failover():
    """Verify that all retries are exhausted on primary node before failing over to secondary."""
    servers = [
        OllamaServerConfig(url="http://primary:11434", priority=1),
        OllamaServerConfig(url="http://secondary:11434", priority=2),
    ]
    pool = OllamaPool(servers=servers)

    called_history = []

    def failing_primary(url: str):
        called_history.append(url)
        if "primary" in url:
            raise TimeoutError("Stuck / timed out after 30s")
        return "secondary-ok"

    res = pool.execute_with_failover(failing_primary, retries=3, backoff_base=0.01)
    assert res == "secondary-ok"
    # Primary attempted 3 times, then secondary succeeded on 1st attempt
    assert called_history == [
        "http://primary:11434",
        "http://primary:11434",
        "http://primary:11434",
        "http://secondary:11434",
    ]
    assert pool.nodes[0].consecutive_failures == 1
    assert pool.nodes[1].consecutive_failures == 0


def test_knowledge_extractor_settings_timeout_and_retries():
    """Verify that KnowledgeExtractor initializes with configured request_timeout and max_retries."""
    settings = Settings(
        ollama_base_url="http://gpu-host:11434",
        request_timeout=30,
        max_retries=3,
    )
    extractor = KnowledgeExtractor.from_settings(settings)
    assert extractor.request_timeout == 30
    assert extractor.max_retries == 3


def test_default_retry_for_attempt_is_one():
    """Verify that retry for an attempt defaults to 1 across Settings and KnowledgeExtractor."""
    default_settings = Settings()
    assert default_settings.max_retries == 1
    assert default_settings.max_chunk_attempts == 6

    extractor = KnowledgeExtractor.from_settings(default_settings)
    assert extractor.max_retries == 1


def test_configurable_retry_for_attempt_and_aliases():
    """Verify that retry for an attempt can be configured via max_retries or aliases."""
    s1 = Settings(max_retries=2)
    assert s1.max_retries == 2

    s2 = Settings(retries_per_attempt=4)
    assert s2.max_retries == 4

    s3 = Settings(attempt_retries=5)
    assert s3.max_retries == 5

    from bookeeper.cli import _get_effective_settings
    eff = _get_effective_settings(max_retries=3, max_chunk_attempts=10)
    assert eff.max_retries == 3
    assert eff.max_chunk_attempts == 10



def test_ollama_pool_reset_server_sends_keep_alive_zero():
    """Verify reset_server sends keep_alive: 0 to Ollama generate endpoint to unstick runner."""
    servers = [OllamaServerConfig(url="http://node1:11434", priority=1)]
    pool = OllamaPool(servers=servers)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        ok = pool.reset_server("http://node1:11434", model="llama3.1:8b")
        assert ok is True
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://node1:11434/api/generate"
        body = json.loads(req.data.decode("utf-8"))
        assert body == {"keep_alive": 0, "model": "llama3.1:8b"}


def test_ollama_pool_watchdog_unfreezes_stuck_server():
    """Verify that watchdog detects stale active task (> max_task_duration) and resets active_tasks counter."""
    servers = [OllamaServerConfig(url="http://node1:11434", priority=1)]
    pool = OllamaPool(servers=servers)

    node = pool.nodes[0]
    # Simulate a leaked task that began 100 seconds ago
    node.active_tasks = 1
    node.last_task_start = time.time() - 100.0

    with patch.object(pool, "reset_server") as mock_reset:
        # A new task arrives; watchdog should detect stale node and unstick it
        def quick_op(url: str):
            return "recovered"

        res = pool.execute_with_failover(quick_op, max_task_duration=60.0)
        assert res == "recovered"
        assert node.active_tasks == 0
        assert mock_reset.call_count >= 1


def test_ollama_pool_finally_cleanup_on_base_exception():
    """Verify that active_tasks is decremented even if an unhandled BaseException occurs."""
    servers = [OllamaServerConfig(url="http://node1:11434", priority=1)]
    pool = OllamaPool(servers=servers)
    node = pool.nodes[0]

    def aborting_op(url: str):
        raise KeyboardInterrupt("Simulated Ctrl+C")

    with pytest.raises(KeyboardInterrupt):
        pool.execute_with_failover(aborting_op)

    # active_tasks must be cleanly reset to 0 by finally block
    assert node.active_tasks == 0
    assert node.last_task_start is None


def test_is_connection_error_detection():
    """Verify is_connection_error properly distinguishes connection refused/offline from other errors."""
    import urllib.error
    from bookeeper.processing.ollama_pool import is_connection_error

    assert is_connection_error(ConnectionRefusedError(61, "Connection refused")) is True
    assert is_connection_error(urllib.error.URLError("[Errno 61] Connection refused")) is True
    assert is_connection_error(urllib.error.URLError("[Errno 111] Connection refused")) is True
    assert is_connection_error(urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))) is True
    assert is_connection_error(RuntimeError("Failed to connect to host: 192.168.50.20 port 11434")) is True
    assert is_connection_error(TimeoutError("Operation timed out after 30s")) is False
    assert is_connection_error(ValueError("Invalid JSON returned by LLM")) is False


def test_connection_refused_retries_and_quarantines_server():
    """Verify that connection refused retries 3 times and moves to cooldown period even with quarantine_server=False."""
    import urllib.error
    servers = [
        OllamaServerConfig(url="http://node1:11434", priority=1),
        OllamaServerConfig(url="http://node2:11434", priority=2),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)
    node1 = pool.nodes[0]
    node2 = pool.nodes[1]

    node1_attempts = 0

    def op(url: str):
        nonlocal node1_attempts
        if "node1" in url:
            node1_attempts += 1
            raise urllib.error.URLError("[Errno 61] Connection refused")
        return "success_from_node2"

    # execute_with_failover with quarantine_server=False
    # Should retry node1 3 times on connection refused, then move node1 to cooldown, then fail over to node2
    res = pool.execute_with_failover(op, retries=1, backoff_base=0.01, quarantine_server=False)
    assert res == "success_from_node2"
    assert node1_attempts == 3

    # node1 must now be in cooldown!
    assert node1.failed_at is not None
    assert node1.is_eligible(600) is False

    # Second invocation should directly pick node2 because node1 is cooling down
    node2_picked = False

    def op2(url: str):
        nonlocal node2_picked
        if "node2" in url:
            node2_picked = True
        return "fast_result"

    res2 = pool.execute_with_failover(op2, quarantine_server=False)
    assert res2 == "fast_result"
    assert node2_picked is True
    # node1 must NOT have been called again!
    assert node1_attempts == 3


def test_consecutive_failures_triggers_cooldown():
    """Verify that 3 consecutive task failures on a server triggers cooldown even with quarantine_server=False."""
    servers = [OllamaServerConfig(url="http://node1:11434", priority=1)]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)
    node1 = pool.nodes[0]

    # Call 1: fails
    with pytest.raises(RuntimeError):
        pool.execute_with_failover(lambda url: (_ for _ in ()).throw(ValueError("task fail 1")), retries=1, quarantine_server=False)
    assert node1.failure_count == 1
    assert node1.failed_at is None  # not cooling down yet

    # Call 2: fails
    with pytest.raises(RuntimeError):
        pool.execute_with_failover(lambda url: (_ for _ in ()).throw(ValueError("task fail 2")), retries=1, quarantine_server=False)
    assert node1.failure_count == 2
    assert node1.failed_at is None  # still eligible

    # Call 3: fails -> 3 consecutive failures triggers cooldown!
    with pytest.raises(RuntimeError):
        pool.execute_with_failover(lambda url: (_ for _ in ()).throw(ValueError("task fail 3")), retries=1, quarantine_server=False)
    assert node1.failure_count == 3
    assert node1.failed_at is not None  # cooling down now!
    assert node1.is_eligible(600) is False


def test_non_quarantine_timeout_releases_without_cooldown_and_allows_immediate_load():
    """Verify that when quarantine_server=False, a timeout releases server without cooldown and allows immediate subsequent load."""
    servers = [
        OllamaServerConfig(url="http://node1:11434", priority=1, name="node1"),
        OllamaServerConfig(url="http://node2:11434", priority=2, name="node2"),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)
    node1 = pool.nodes[0]
    node2 = pool.nodes[1]

    called_urls = []

    def timeout_op(url: str):
        called_urls.append(url)
        raise TimeoutError(f"Ollama request to {url}/api/chat timed out after 90s")

    # Invocation with quarantine_server=False must raise immediately rather than failover across pool
    with pytest.raises(RuntimeError) as exc_info:
        pool.execute_with_failover(timeout_op, retries=1, quarantine_server=False)

    assert "http://node1:11434" in str(exc_info.value)
    # node2 was NOT called in this invocation! The task was immediately returned to queue
    assert called_urls == ["http://node1:11434"]

    # node1 must NOT be in cooldown: failed_at is None and is_eligible is True
    assert node1.failed_at is None
    assert node1.is_eligible(600) is True
    assert node1.active_tasks == 0

    # Immediate subsequent task can immediately load node1!
    second_called = []

    def quick_op(url: str):
        second_called.append(url)
        return "success"

    res = pool.execute_with_failover(quick_op, quarantine_server=False)
    assert res == "success"
    assert second_called == ["http://node1:11434"]
    assert pool.get_last_used_server() == "node1"
    assert pool.get_last_used_server_url() == "http://node1:11434"


def test_execute_with_failover_respects_exclude_urls():
    """Verify that exclude_urls forces selection of another server for poison pill rotation."""
    servers = [
        OllamaServerConfig(url="http://node1:11434", priority=1, name="node1"),
        OllamaServerConfig(url="http://node2:11434", priority=2, name="node2"),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    called = []

    def op(url: str):
        called.append(url)
        return "ok"

    # With exclude_urls set to node1, it must route to node2
    res = pool.execute_with_failover(op, exclude_urls={"http://node1:11434"})
    assert res == "ok"
    assert called == ["http://node2:11434"]


def test_server_node_capabilities_parsing_and_defaults():
    """Verify default capabilities, alias handling, and normalization."""
    # Omitted capability defaults to all 3 capabilities
    cfg_default = OllamaServerConfig(url="http://node:11434", priority=1)
    assert cfg_default.capability == ["llm", "embedding", "verification"]
    assert cfg_default.supports_llm is True
    assert cfg_default.supports_embedding is True
    assert cfg_default.supports_verification is True

    # Plural alias 'capabilities' and case-insensitive normalization
    cfg_custom = OllamaServerConfig(
        url="http://node:11434",
        priority=1,
        capabilities=["LLM", "Embeddings", "Verify"],
    )
    assert cfg_custom.capability == ["llm", "embedding", "verification"]
    assert cfg_custom.has_capability("llm") is True
    assert cfg_custom.has_capability("embedding") is True
    assert cfg_custom.has_capability("verification") is True

    # Single capability
    cfg_emb = OllamaServerConfig(
        url="http://embed-only:11434",
        priority=10,
        capability=["embedding"],
    )
    assert cfg_emb.supports_embedding is True
    assert cfg_emb.supports_llm is False
    assert cfg_emb.supports_verification is False
    assert cfg_emb.capability_str == "embedding"


def test_settings_resolution_with_capabilities():
    """Verify Settings resolves LLM, embedding, and verification servers per node capabilities."""
    raw_servers = [
        {
            "url": "http://192.168.50.118:11434",
            "priority": 1,
            "name": "mypc-gpu-node",
            "capability": ["llm", "embedding", "verification"],
        },
        {
            "url": "http://192.168.50.15:11434",
            "priority": 2,
            "name": "primary-gpu-node",
            "capability": ["llm", "embedding"],
        },
        {
            "url": "http://localhost:11434",
            "priority": 4,
            "name": "mac-node",
            "capability": ["llm", "embedding"],
        },
        {
            "url": "http://192.168.50.20:11434",
            "priority": 10,
            "name": "secondary-cpu-node",
            "capability": ["embedding"],
        },
    ]

    settings = Settings(ollama_servers=raw_servers)

    # 1. All servers sorted by priority
    all_servers = settings.resolved_ollama_servers
    assert len(all_servers) == 4
    assert [s.name for s in all_servers] == [
        "mypc-gpu-node",
        "primary-gpu-node",
        "mac-node",
        "secondary-cpu-node",
    ]

    # 2. LLM servers must exclude secondary-cpu-node
    llm_servers = settings.resolved_llm_servers
    assert len(llm_servers) == 3
    assert [s.name for s in llm_servers] == [
        "mypc-gpu-node",
        "primary-gpu-node",
        "mac-node",
    ]

    # 3. Embedding servers include all 4 nodes (since all support embedding)
    embed_servers = settings.resolved_embedding_servers
    assert len(embed_servers) == 4

    # 4. Verification servers should only include mypc-gpu-node
    verify_servers = settings.resolved_verification_servers
    assert len(verify_servers) == 1
    assert verify_servers[0].name == "mypc-gpu-node"


def test_pool_execute_routes_strictly_by_capability():
    """Verify execute_with_failover respects capability parameter and never routes to ineligible nodes."""
    servers = [
        OllamaServerConfig(
            url="http://node-gpu:11434",
            priority=1,
            name="gpu-node",
            capability=["llm"],
        ),
        OllamaServerConfig(
            url="http://node-cpu:11434",
            priority=2,
            name="cpu-node",
            capability=["embedding"],
        ),
        OllamaServerConfig(
            url="http://node-audit:11434",
            priority=3,
            name="audit-node",
            capability=["verification"],
        ),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    # Calling with capability='llm' must only target node-gpu
    called = []
    pool.execute_with_failover(lambda url: called.append(url), capability="llm")
    assert called == ["http://node-gpu:11434"]

    # Calling with capability='embedding' must only target node-cpu
    called = []
    pool.execute_with_failover(lambda url: called.append(url), capability="embedding")
    assert called == ["http://node-cpu:11434"]

    # Calling with capability='verification' must only target node-audit
    called = []
    pool.execute_with_failover(lambda url: called.append(url), capability="verification")
    assert called == ["http://node-audit:11434"]


def test_pool_for_capability_subpool():
    """Verify for_capability produces an isolated pool containing only capable nodes."""
    servers = [
        OllamaServerConfig(
            url="http://node1:11434",
            priority=1,
            name="node1",
            capability=["llm", "embedding"],
        ),
        OllamaServerConfig(
            url="http://node2:11434",
            priority=2,
            name="node2",
            capability=["embedding"],
        ),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    llm_pool = pool.for_capability("llm")
    assert len(llm_pool.nodes) == 1
    assert llm_pool.nodes[0].url == "http://node1:11434"

    embed_pool = pool.for_capability("embedding")
    assert len(embed_pool.nodes) == 2


def test_ollama_server_node_record_task_attempt():
    """Verify task attempt metrics: characters, task counts, duration, and averages."""
    node = OllamaServerNode(url="http://node-gpu:11434", priority=1, name="gpu-node")
    assert node.total_tasks == 0
    assert node.success_tasks == 0
    assert node.failed_tasks == 0
    assert node.success_chars == 0
    assert node.total_duration == 0.0

    # 1. Record successful LLM task
    node.record_task_attempt(duration=2.5, success=True, chars=1200, task_type="llm")
    assert node.total_tasks == 1
    assert node.success_tasks == 1
    assert node.failed_tasks == 0
    assert node.success_chars == 1200
    assert pytest.approx(node.total_duration, 0.01) == 2.5
    assert pytest.approx(node.avg_duration, 0.01) == 2.5

    # 2. Record failed task (chars should NOT be added to success_chars)
    node.record_task_attempt(duration=1.0, success=False, chars=1500, task_type="llm")
    assert node.total_tasks == 2
    assert node.success_tasks == 1
    assert node.failed_tasks == 1
    assert node.success_chars == 1200  # Still 1200, failed attempt chars discarded!
    assert pytest.approx(node.total_duration, 0.01) == 3.5
    assert pytest.approx(node.avg_duration, 0.01) == 1.75
    assert pytest.approx(node.avg_success_duration, 0.01) == 2.5

    # 3. Record successful embedding task
    node.record_task_attempt(duration=0.5, success=True, chars=800, task_type="embedding")
    assert node.total_tasks == 3
    assert node.success_tasks == 2
    assert node.success_chars == 2000
    assert "embedding" in node.stats_by_type
    assert node.stats_by_type["embedding"]["success_tasks"] == 1
    assert node.stats_by_type["embedding"]["success_chars"] == 800

    # 4. Reset stats
    node.reset_statistics()
    assert node.total_tasks == 0
    assert node.success_chars == 0


def test_ollama_pool_execute_with_failover_metrics():
    """Verify execute_with_failover automatically records task counts and valuable characters."""
    servers = [
        OllamaServerConfig(url="http://node-fail:11434", priority=1, name="fail-node"),
        OllamaServerConfig(url="http://node-ok:11434", priority=2, name="ok-node"),
    ]
    pool = OllamaPool(servers=servers, cooldown_seconds=600)

    # Invocation where fail-node fails and ok-node succeeds
    def mock_operation(url: str):
        if "node-fail" in url:
            raise ConnectionRefusedError("Offline")
        return "success_data"

    res = pool.execute_with_failover(
        mock_operation,
        retries=1,
        input_chars=3500,
        task_type="llm",
    )
    assert res == "success_data"

    fail_node = next(n for n in pool.nodes if n.url == "http://node-fail:11434")
    ok_node = next(n for n in pool.nodes if n.url == "http://node-ok:11434")

    # Fail node has 1 attempted task, 0 success, 0 accepted chars
    assert fail_node.total_tasks >= 1
    assert fail_node.success_tasks == 0
    assert fail_node.success_chars == 0

    # Ok node has 1 attempted task, 1 success, 3500 accepted chars
    assert ok_node.total_tasks == 1
    assert ok_node.success_tasks == 1
    assert ok_node.success_chars == 3500


def test_aggregate_server_statistics_and_table_formatting():
    """Verify multi-pool stats aggregation and Rich table formatting."""
    from bookeeper.processing.ollama_pool import (
        aggregate_server_statistics,
        format_server_stats_table,
    )

    # Pool 1 (LLM pool)
    p1 = OllamaPool.from_urls(["http://node-gpu:11434"])
    p1.nodes[0].record_task_attempt(duration=5.0, success=True, chars=10000, task_type="llm")

    # Pool 2 (Embedding pool, sharing same node-gpu)
    p2 = OllamaPool.from_urls(["http://node-gpu:11434", "http://node-cpu:11434"])
    p2.nodes[0].record_task_attempt(duration=1.0, success=True, chars=4000, task_type="embedding")
    p2.nodes[1].record_task_attempt(duration=2.0, success=True, chars=2500, task_type="embedding")

    aggregated = aggregate_server_statistics([p1, p2])
    assert len(aggregated) == 2

    gpu_stat = next(s for s in aggregated if s["url"] == "http://node-gpu:11434")
    assert gpu_stat["total_tasks"] == 2
    assert gpu_stat["success_tasks"] == 2
    assert gpu_stat["success_chars"] == 14000  # 10000 + 4000
    assert pytest.approx(gpu_stat["total_duration"], 0.01) == 6.0

    cpu_stat = next(s for s in aggregated if s["url"] == "http://node-cpu:11434")
    assert cpu_stat["total_tasks"] == 1
    assert cpu_stat["success_tasks"] == 1
    assert cpu_stat["success_chars"] == 2500

    # Generate Rich table
    table = format_server_stats_table(aggregated, title="Test Server Statistics")
    assert table is not None
    assert len(table.rows) == 3  # 2 servers + 1 summary row
    assert table.title == "Test Server Statistics"








