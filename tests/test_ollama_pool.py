"""
Comprehensive unit tests for multi-server Ollama pool, priority failover, and cooldown recovery.
"""

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

        def get_chat_instance(base_url, model, temperature):
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
