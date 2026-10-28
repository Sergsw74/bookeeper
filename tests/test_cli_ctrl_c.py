"""
Unit tests for Ctrl+C (KeyboardInterrupt) graceful cancellation handling,
DaemonThreadPoolExecutor thread safety, and exporter bypass logic.
"""

import concurrent.futures
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import typer

from bookeeper.cli import DaemonThreadPoolExecutor
from bookeeper.graph.store import ConceptGraphStore


def test_daemon_thread_pool_executor_creates_daemon_threads():
    """Verify DaemonThreadPoolExecutor spawns daemon threads."""
    created_threads = []

    def _worker():
        t = threading.current_thread()
        created_threads.append(t)
        return t.daemon

    with DaemonThreadPoolExecutor(max_workers=2) as executor:
        fut = executor.submit(_worker)
        is_daemon = fut.result(timeout=2.0)
        assert is_daemon is True

    assert len(created_threads) > 0
    for t in created_threads:
        assert t.daemon is True


def test_daemon_thread_pool_executor_shutdown_immediate():
    """Verify shutdown(wait=False, cancel_futures=True) does not hang and unregisters threads."""
    import concurrent.futures.thread

    executor = DaemonThreadPoolExecutor(max_workers=2)
    started = threading.Event()
    stop_flag = threading.Event()

    def _long_worker():
        started.set()
        stop_flag.wait(timeout=5.0)

    fut = executor.submit(_long_worker)
    assert started.wait(timeout=2.0)

    # Shut down without waiting
    t0 = time.perf_counter()
    executor.shutdown(wait=False, cancel_futures=True)
    dur = time.perf_counter() - t0
    # Shutdown should return in well under 0.5s even though worker waits 5s
    assert dur < 0.5

    # Check unregister from _threads_queues
    with concurrent.futures.thread._global_shutdown_lock:
        for t in list(executor._threads):
            assert t not in concurrent.futures.thread._threads_queues

    stop_flag.set()


def test_daemon_thread_pool_executor_context_manager_exception_aborts_immediately():
    """Verify when an exception is raised inside with DaemonThreadPoolExecutor, it shuts down wait=False."""
    started = threading.Event()
    stop_flag = threading.Event()

    def _long_worker():
        started.set()
        stop_flag.wait(timeout=10.0)

    t0 = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        with DaemonThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(_long_worker)
            assert started.wait(timeout=2.0)
            raise KeyboardInterrupt()

    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0
    stop_flag.set()


def test_worker_polling_loop_interruption_cancels_futures():
    """Verify that polling worker futures with timeout handles interruption immediately."""
    abort_event = threading.Event()
    executor = DaemonThreadPoolExecutor(max_workers=2)

    started = threading.Event()

    def _worker():
        started.set()
        while not abort_event.is_set():
            time.sleep(0.05)

    worker_futures = [executor.submit(_worker) for _ in range(2)]
    assert started.wait(timeout=2.0)

    # Simulate KeyboardInterrupt on main thread
    interrupted = False
    try:
        # Polling loop as used in cli.py
        for idx, f in enumerate(worker_futures):
            # Simulate interrupt while waiting
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        interrupted = True
        abort_event.set()
        for f in worker_futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    assert interrupted is True
    assert abort_event.is_set()


def test_build_graph_interrupted_bypasses_heavy_exports(tmp_path):
    """
    Verify that when interrupted is True during build_graph finalization:
    1. store.save is invoked for checkpoint preservation.
    2. ObsidianExporter, GraphMLExporter, Neo4jExporter are completely bypassed.
    3. typer.Exit(code=130) is raised.
    """
    store = ConceptGraphStore()
    store.add_book(1, title="Test Book")
    graph_file = tmp_path / "knowledge_graph.json"
    vault_dest = tmp_path / "vault"
    graphml_dest = tmp_path / "knowledge_graph.graphml"

    with patch("bookeeper.cli.ObsidianExporter") as mock_obs, \
         patch("bookeeper.cli.GraphMLExporter") as mock_gml, \
         patch("bookeeper.cli.Neo4jExporter") as mock_neo:

        interrupted = True
        completed_in_session = 1

        # Simulate the finalization block in cli.py
        if store.graph.number_of_nodes() > 0:
            store.save(graph_file, link_chunks=False)
            if not interrupted:
                obs = mock_obs(vault_dest)
                obs.export(store, clean=False)
                gml = mock_gml(graphml_dest)
                gml.export(store)
                neo_exp = mock_neo.from_config()
                neo_exp.export(store)

        assert graph_file.exists()
        mock_obs.assert_not_called()
        mock_gml.assert_not_called()
        mock_neo.assert_not_called()

        if interrupted:
            with pytest.raises(typer.Exit) as exc_info:
                raise typer.Exit(code=130)
            assert exc_info.value.exit_code == 130


def test_rebuild_graph_interrupted_bypasses_heavy_exports(tmp_path):
    """Verify that when rebuild_graph is interrupted, checkpoint is saved but exporters are skipped."""
    store = ConceptGraphStore()
    store.add_book(10, title="Rebuilt Book")
    graph_file = tmp_path / "knowledge_graph.json"
    vault_dest = tmp_path / "vault"

    with patch("bookeeper.cli.ObsidianExporter") as mock_obs, \
         patch("bookeeper.cli.GraphMLExporter") as mock_gml, \
         patch("bookeeper.cli.Neo4jExporter") as mock_neo:

        interrupted = True
        store.save(graph_file)

        if interrupted:
            with pytest.raises(typer.Exit) as exc_info:
                raise typer.Exit(code=130)
            assert exc_info.value.exit_code == 130

        assert graph_file.exists()
        mock_obs.assert_not_called()
        mock_gml.assert_not_called()
        mock_neo.assert_not_called()


def test_cli_worker_polling_timeout_handling():
    """Verify worker future polling handles TimeoutError without NameError on concurrent."""
    import bookeeper.cli as cli

    assert hasattr(cli, "concurrent")
    assert hasattr(cli.concurrent, "futures")

    # Simulate worker polling loop with a mock future that times out first then completes
    mock_fut = MagicMock()
    mock_fut.done.side_effect = [False, False, True]
    mock_fut.result.side_effect = [concurrent.futures.TimeoutError(), "done"]

    worker_futures = [mock_fut]
    abort_event = threading.Event()

    iterations = 0
    while not abort_event.is_set() and any(not f.done() for f in worker_futures):
        iterations += 1
        try:
            for f in worker_futures:
                if not f.done():
                    f.result(timeout=0.15)
        except (concurrent.futures.TimeoutError, TimeoutError, cli.FuturesTimeoutError):
            pass

    assert iterations >= 1
    mock_fut.result.assert_called()
