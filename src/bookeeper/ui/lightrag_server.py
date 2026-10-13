"""
LightRAG Embedded Server Subsystem for Bookeeper.

Provides programmatic management of the official LightRAG FastAPI server
and WebUI/Workspace inside Bookeeper processes or CLI subcommands.
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn

from bookeeper.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LightRAGServerManager:
    """
    Manages the lifecycle of an embedded LightRAG FastAPI server instance,
    configuring Ollama LLM and embedding backends based on Bookeeper settings.
    """

    def __init__(
        self,
        config: Settings,
        working_dir: Optional[Path] = None,
        host: str = "127.0.0.1",
        port: int = 9621,
    ):
        self.config = config
        self.working_dir = (working_dir or config.resolved_lightrag_dir).expanduser().resolve()
        self.host = host
        self.port = port
        self.server: Optional[uvicorn.Server] = None
        self.thread: Optional[threading.Thread] = None
        self._is_running = False

    def _configure_environment(self) -> None:
        """Inject environment variables required by LightRAG FastAPI server."""
        self.working_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WORKING_DIR"] = str(self.working_dir)
        os.environ["HOST"] = self.host
        os.environ["PORT"] = str(self.port)

        # Configure LLM binding from Bookeeper settings
        os.environ["LLM_BINDING"] = "ollama"
        os.environ["LLM_MODEL"] = self.config.llm_model
        os.environ["LLM_BINDING_HOST"] = str(self.config.ollama_base_url)

        # Configure Embedding binding from Bookeeper settings
        os.environ["EMBEDDING_BINDING"] = "ollama"
        os.environ["EMBEDDING_MODEL"] = self.config.embedding_model
        # If pool is used, fallback to first server or base url
        emb_host = str(self.config.ollama_base_url)
        if self.config.ollama_servers:
            for s in self.config.ollama_servers:
                if "embedding" in (s.capability or []):
                    emb_host = str(s.url)
                    break
        os.environ["EMBEDDING_BINDING_HOST"] = emb_host
        os.environ["EMBEDDING_DIM"] = str(self.config.embedding_dim)

        # LightRAG WebUI and workspace defaults
        os.environ["DEFAULT_UI"] = "webui"
        os.environ["ENABLE_API_DOCS"] = "true"

    def start(self, wait_ready: bool = True, timeout: float = 15.0) -> bool:
        """
        Start the LightRAG server in a background daemon thread.
        Returns True once the server accepts connections, or False on timeout/failure.
        """
        if self._is_running:
            return True

        self._configure_environment()

        try:
            saved_argv = list(sys.argv)
            try:
                sys.argv = [
                    "lightrag_server",
                    "--working-dir",
                    str(self.working_dir),
                    "--host",
                    self.host,
                    "--port",
                    str(self.port),
                ]
                from lightrag.api.config import initialize_config, parse_args
                parsed = parse_args()
                global_args = initialize_config(args=parsed, force=True)

                from lightrag.api.lightrag_server import create_app
                app = create_app(global_args)
            finally:
                sys.argv = saved_argv

            uv_config = uvicorn.Config(
                app=app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
            self.server = uvicorn.Server(uv_config)

            self.thread = threading.Thread(target=self.server.run, daemon=True, name="LightRAG-Server-Thread")
            self.thread.start()
            self._is_running = True

            if wait_ready:
                t0 = time.time()
                while time.time() - t0 < timeout:
                    if self.is_alive():
                        logger.info(f"LightRAG server is active on http://{self.host}:{self.port}/")
                        return True
                    time.sleep(0.3)
                logger.warning(f"LightRAG server started but timed out waiting for readiness after {timeout}s.")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to start embedded LightRAG server: {e}", exc_info=True)
            self._is_running = False
            return False

    def stop(self) -> None:
        """Gracefully shut down the embedded LightRAG server."""
        if self.server is not None:
            self.server.should_exit = True
        self._is_running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.server = None
        self.thread = None

    def is_alive(self) -> bool:
        """Check if the LightRAG server is responding on its HTTP port."""
        import socket

        try:
            with socket.create_connection((self.host, self.port), timeout=1.0):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    @property
    def webui_url(self) -> str:
        return f"http://{self.host}:{self.port}/webui/"

    @property
    def workspace_url(self) -> str:
        return f"http://{self.host}:{self.port}/workspace/"

    @property
    def api_docs_url(self) -> str:
        return f"http://{self.host}:{self.port}/docs"
