"""
Bookeeper Web Server and REST API for Idea-Based Search Engine.
Provides HTTP API endpoints, LightRAG-compatible Query API (/query),
and serves the frontend web UI.
"""

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from bookeeper.config import Settings, get_settings
from bookeeper.ui.backend import SearchEngineBackend

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class BookeeperUIRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for Bookeeper UI and REST API."""

    backend: SearchEngineBackend = None  # Class-level reference set by server
    lightrag_mgr: Optional[Any] = None  # Optional stub for backward compatibility

    def _send_json(self, data: dict, status: int = HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, content: str, status: int = HTTPStatus.OK):
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, content_type: str = "text/plain"):
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File Not Found")
            return
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        # 1. API: Health Check & System Status
        if path in ("/health", "/api/health"):
            self._send_json({
                "status": "ok",
                "service": "bookkeeper-search",
                "neo4j_uri": self.backend.neo4j_cfg.uri,
                "neo4j_db": self.backend.neo4j_cfg.database,
            })
            return

        # 2. API: Node Details (Obsidian Note View)
        if path == "/api/node":
            qs = parse_qs(parsed.query)
            node_id = qs.get("id", [""])[0]
            if not node_id:
                self._send_json({"error": "Missing 'id' query parameter."}, HTTPStatus.BAD_REQUEST)
                return
            details = self.backend.get_node_details(node_id)
            self._send_json(details)
            return

        # 2b. API: Node Expand (Undirected Neighborhood Expansion)
        if path == "/api/node/expand":
            qs = parse_qs(parsed.query)
            node_id = qs.get("id", [""])[0] or qs.get("node_id", [""])[0]
            if not node_id:
                self._send_json({"error": "Missing 'id' query parameter."}, HTTPStatus.BAD_REQUEST)
                return
            type_filter = qs.get("type_filter", ["all"])[0]
            limit = int(qs.get("limit", [50])[0])
            expansion = self.backend.expand_node_neighborhood(node_id=node_id, type_filter=type_filter, limit=limit)
            self._send_json(expansion)
            return

        # 2c. API: Book File Streaming & Direct Download
        if path in ("/api/book/file", "/api/book/download"):
            qs = parse_qs(parsed.query)
            filename = qs.get("filename", [""])[0] or qs.get("file", [""])[0] or qs.get("path", [""])[0]
            title = qs.get("title", [""])[0]
            book_id = qs.get("id", [""])[0] or qs.get("book_id", [""])[0]
            breadcrumb = qs.get("breadcrumb", [""])[0]

            resolved = self.backend.resolve_book_file(
                title=title,
                book_id=book_id,
                breadcrumb=breadcrumb,
                filename=filename,
            )

            if not resolved or not resolved.get("found"):
                self._send_json({"error": "Book file not found or not available at download path."}, HTTPStatus.NOT_FOUND)
                return

            abs_path = resolved.get("absolute_path")
            if not abs_path or not Path(abs_path).is_file():
                self._send_json({"error": "Target book file does not exist on disk."}, HTTPStatus.NOT_FOUND)
                return

            target_file = Path(abs_path)
            ext = target_file.suffix.lower()
            content_type = {
                ".pdf": "application/pdf",
                ".epub": "application/epub+zip",
                ".mobi": "application/x-mobipocket-ebook",
                ".azw3": "application/vnd.amazon.ebook",
                ".djvu": "image/vnd.djvu",
                ".txt": "text/plain; charset=utf-8",
                ".opf": "application/oebps-package+xml",
            }.get(ext, "application/octet-stream")

            disposition = "attachment" if path == "/api/book/download" else "inline"
            safe_filename = target_file.name.replace('"', '\\"')

            try:
                data = target_file.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'{disposition}; filename="{safe_filename}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                logger.error(f"Error serving book file {target_file}: {e}")
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error streaming book: {e}")
            return

        # 2d. API: Book Metadata / Resolution Info
        if path == "/api/book/info":
            qs = parse_qs(parsed.query)
            filename = qs.get("filename", [""])[0] or qs.get("file", [""])[0]
            title = qs.get("title", [""])[0]
            book_id = qs.get("id", [""])[0] or qs.get("book_id", [""])[0]
            breadcrumb = qs.get("breadcrumb", [""])[0]

            resolved = self.backend.resolve_book_file(
                title=title,
                book_id=book_id,
                breadcrumb=breadcrumb,
                filename=filename,
            )
            if resolved and resolved.get("found"):
                safe_info = {k: v for k, v in resolved.items() if k != "absolute_path"}
                self._send_json(safe_info)
            else:
                self._send_json({"found": False, "error": "Book file not found."}, HTTPStatus.NOT_FOUND)
            return

        # 3. Serve Frontend HTML / Static files
        if path in ("/", "/index.html"):
            index_html = STATIC_DIR / "index.html"
            if index_html.is_file():
                self._send_html(index_html.read_text(encoding="utf-8"))
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "index.html not found.")
            return

        # Static assets (css, js)
        clean_name = path.lstrip("/")
        target_file = (STATIC_DIR / clean_name).resolve()
        if target_file.is_file() and target_file.is_relative_to(STATIC_DIR):
            ct = "text/plain"
            if target_file.suffix == ".css":
                ct = "text/css"
            elif target_file.suffix == ".js":
                ct = "application/javascript"
            elif target_file.suffix in (".html", ".htm"):
                ct = "text/html"
            elif target_file.suffix == ".json":
                ct = "application/json"
            self._send_file(target_file, ct)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self._send_json({"error": "Empty request body."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            body_str = self.rfile.read(content_length).decode("utf-8")
            body = json.loads(body_str)
        except Exception as e:
            self._send_json({"error": f"Invalid JSON payload: {e}"}, HTTPStatus.BAD_REQUEST)
            return

        # Node Expand API (/api/node/expand)
        if path == "/api/node/expand":
            node_id = body.get("id") or body.get("node_id") or ""
            if not node_id:
                self._send_json({"error": "Missing 'id' parameter in expansion request."}, HTTPStatus.BAD_REQUEST)
                return
            type_filter = body.get("type_filter", "all")
            limit = int(body.get("limit", 50))
            try:
                expansion = self.backend.expand_node_neighborhood(node_id=node_id, type_filter=type_filter, limit=limit)
                self._send_json(expansion)
            except Exception as e:
                logger.error(f"Node expansion API error: {e}", exc_info=True)
                self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        query = body.get("query", "").strip()
        if not query:
            self._send_json({"error": "Please provide a 'query' string."}, HTTPStatus.BAD_REQUEST)
            return

        mode = body.get("mode", "ideas")
        denoise_level = body.get("denoise_level", "standard")
        top_k = int(body.get("top_k", 40))
        chunk_top_k = int(body.get("chunk_top_k", 20))
        graph_limit = int(body.get("graph_limit", 60))

        # 1. LightRAG-Compatible Query API (/query and /api/query)
        if path in ("/query", "/api/query"):
            include_references = body.get("include_references", True)
            include_chunk_content = body.get("include_chunk_content", True)
            try:
                res = self.backend.query_api(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    graph_limit=graph_limit,
                    denoise_level=denoise_level,
                    include_references=include_references,
                    include_chunk_content=include_chunk_content,
                )
                self._send_json(res)
            except Exception as e:
                logger.error(f"Query API error: {e}", exc_info=True)
                self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        # 2. Web UI Search API (/api/search)
        if path == "/api/search":
            similarity_threshold = float(body.get("similarity_threshold", 0.70))
            top_k_seeds = int(body.get("top_k_seeds", 6))
            try:
                result = self.backend.query_ideas(
                    query=query,
                    mode=mode,
                    denoise_level=denoise_level,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    graph_limit=graph_limit,
                    similarity_threshold=similarity_threshold,
                    top_k_seeds=top_k_seeds,
                )
                self._send_json(result)
            except Exception as e:
                logger.error(f"Search API execution error: {e}", exc_info=True)
                self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, format: str, *args):
        logger.debug("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    config_path: Optional[str] = None,
    backend: Optional[SearchEngineBackend] = None,
    lightrag_mgr: Optional[Any] = None,
    auto_port: bool = True,
) -> ThreadingHTTPServer:
    """
    Run the Bookeeper UI and REST API server.
    If the requested port is in use and auto_port is True, automatically increments
    to find an available port.
    """
    if backend is None:
        cfg = get_settings(config_path)
        backend = SearchEngineBackend(cfg)

    BookeeperUIRequestHandler.backend = backend
    BookeeperUIRequestHandler.lightrag_mgr = lightrag_mgr

    class ReusableThreadingServer(ThreadingHTTPServer):
        allow_reuse_address = True

    current_port = port
    max_attempts = 20 if auto_port else 1
    last_err = None

    for attempt in range(max_attempts):
        try:
            server = ReusableThreadingServer((host, current_port), BookeeperUIRequestHandler)
            logger.info(f"Bookeeper Search UI running at http://{host}:{current_port}/")
            return server
        except OSError as e:
            last_err = e
            if not auto_port or attempt == max_attempts - 1:
                raise
            current_port += 1

    raise last_err or OSError(f"Could not bind to {host}:{port}")
