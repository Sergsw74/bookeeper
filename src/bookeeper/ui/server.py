"""
Bookeeper Web Server and REST API for Idea-Based Search Engine.
Provides HTTP API endpoints, LightRAG-compatible Query API (/query),
and serves the frontend web UI.
"""

import json
import logging
import re
import sys
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from bookeeper.config import Settings, get_settings
from bookeeper.ui.backend import SearchEngineBackend

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def parse_multipart_to_file(
    stream: Any,
    boundary: str,
    dest_file_obj: Any,
    max_content_length: Optional[int] = None,
) -> tuple[Optional[str], dict[str, str], int]:
    """
    Stream a multipart/form-data payload directly from stream to disk.
    Avoids loading large uploaded archive files into RAM.
    Returns (filename, fields_dict, bytes_written).
    """
    raw_boundary = boundary.encode("ascii") if isinstance(boundary, str) else boundary
    delimiter = b"\r\n--" + raw_boundary
    delim_len = len(delimiter)

    remaining = [max_content_length] if max_content_length is not None else [float("inf")]

    def read_exact_or_less(n: int) -> bytes:
        if remaining[0] <= 0:
            return b""
        to_read = min(n, int(remaining[0]))
        chunk = stream.read(to_read)
        remaining[0] -= len(chunk)
        return chunk

    buf = bytearray()

    def read_line() -> bytes:
        nonlocal buf
        while b"\n" not in buf:
            chunk = read_exact_or_less(8192)
            if not chunk:
                break
            buf.extend(chunk)
        if not buf:
            return b""
        idx = buf.find(b"\n")
        if idx != -1:
            line = bytes(buf[:idx + 1])
            del buf[:idx + 1]
            return line
        line = bytes(buf)
        buf.clear()
        return line

    fields: dict[str, str] = {}
    filename: Optional[str] = None
    file_bytes_written = 0

    first_line = read_line().strip()
    if not first_line.startswith(b"--" + raw_boundary):
        raise ValueError(f"Invalid multipart start: {first_line}")
    if first_line.endswith(b"--"):
        return filename, fields, file_bytes_written

    while True:
        headers = []
        while True:
            hline = read_line()
            if not hline or hline in (b"\r\n", b"\n"):
                break
            headers.append(hline.decode("latin1", errors="replace"))
        if not headers:
            break
        header_text = "".join(headers)
        fname_m = re.search(r'filename="([^"]+)"', header_text)
        name_m = re.search(r'name="([^"]+)"', header_text)
        name = name_m.group(1) if name_m else None

        if fname_m:
            filename = fname_m.group(1)
            # Stream binary file content directly to dest_file_obj in 64KB chunks
            while True:
                idx = buf.find(delimiter)
                if idx != -1:
                    dest_file_obj.write(buf[:idx])
                    file_bytes_written += idx
                    del buf[:idx + len(delimiter)]
                    post_line = read_line()
                    if post_line.strip().startswith(b"--"):
                        return filename, fields, file_bytes_written
                    break
                else:
                    safe = len(buf) - delim_len
                    if safe > 0:
                        dest_file_obj.write(buf[:safe])
                        file_bytes_written += safe
                        del buf[:safe]
                    chunk = read_exact_or_less(65536)
                    if not chunk:
                        dest_file_obj.write(buf)
                        file_bytes_written += len(buf)
                        buf.clear()
                        return filename, fields, file_bytes_written
                    buf.extend(chunk)
        else:
            # Regular text field: read until next delimiter
            val_parts = []
            while True:
                idx = buf.find(delimiter)
                if idx != -1:
                    val_parts.append(buf[:idx])
                    del buf[:idx + len(delimiter)]
                    post_line = read_line()
                    val = b"".join(val_parts).rstrip(b"\r\n").decode("utf-8", errors="replace")
                    if name:
                        fields[name] = val
                    if post_line.strip().startswith(b"--"):
                        return filename, fields, file_bytes_written
                    break
                else:
                    safe = len(buf) - delim_len
                    if safe > 0:
                        val_parts.append(bytes(buf[:safe]))
                        del buf[:safe]
                    chunk = read_exact_or_less(8192)
                    if not chunk:
                        val_parts.append(bytes(buf))
                        buf.clear()
                        val = b"".join(val_parts).rstrip(b"\r\n").decode("utf-8", errors="replace")
                        if name:
                            fields[name] = val
                        return filename, fields, file_bytes_written
                    buf.extend(chunk)

    return filename, fields, file_bytes_written


class BookeeperUIRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for Bookeeper UI and REST API."""

    backend: SearchEngineBackend = None  # Class-level reference set by server
    lightrag_mgr: Optional[Any] = None  # Optional stub for backward compatibility

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug(f"Client disconnected during request from {self.client_address}")

    def send_error(self, code, message=None, explain=None):
        try:
            super().send_error(code, message=message, explain=explain)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Client disconnected while sending error response")

    def _send_json(self, data: dict, status: int = HTTPStatus.OK):
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Client disconnected while sending JSON response")

    def _send_html(self, content: str, status: int = HTTPStatus.OK):
        try:
            body = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Client disconnected while sending HTML response")

    def _send_file(self, file_path: Path, content_type: str = "text/plain"):
        try:
            if not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File Not Found")
                return
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Client disconnected while sending file response")

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
            from bookeeper import __version__, __release_date__
            consistency = self.backend.validate_database_consistency()
            self._send_json({
                "status": "ok",
                "service": "bookkeeper-search",
                "version": __version__,
                "release_date": __release_date__,
                "neo4j_uri": self.backend.neo4j_cfg.uri,
                "neo4j_db": self.backend.neo4j_cfg.database,
                "consistency": consistency,
            })
            return

        # 1a. API: Version Information
        if path == "/api/version":
            from bookeeper import __version__, __release_date__
            self._send_json({
                "version": __version__,
                "release_date": __release_date__,
                "service": "bookkeeper-search",
            })
            return

        # 1b. API: Consistency & Database Sync Validation
        if path in ("/api/sync/status", "/api/consistency"):
            qs = parse_qs(parsed.query)
            force_refresh = qs.get("refresh", ["false"])[0].lower() in ("true", "1")
            consistency = self.backend.validate_database_consistency(force_refresh=force_refresh)
            self._send_json(consistency)
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

        # 2e. API: Indexed Books List
        if path in ("/api/books", "/api/books/indexed"):
            qs = parse_qs(parsed.query)
            limit_str = qs.get("limit", [""])[0]
            limit = int(limit_str) if limit_str.isdigit() else None
            books = self.backend.get_indexed_books(limit=limit)
            self._send_json({"books": books, "count": len(books)})
            return

        # 2e-2. API: Ideas List & Search (Exact / Embedded + Month/Year Range Filter)
        if path in ("/api/ideas", "/api/ideas/search"):
            qs = parse_qs(parsed.query)
            q_str = qs.get("q", [""])[0] or qs.get("query", [""])[0]
            search_type = qs.get("search_type", ["exact"])[0] or qs.get("mode", ["exact"])[0]
            month_from = qs.get("month_from", [""])[0] or qs.get("from", [""])[0] or None
            month_to = qs.get("month_to", [""])[0] or qs.get("to", [""])[0] or None
            category = qs.get("category", [""])[0] or None
            sort_by = qs.get("sort_by", ["month_desc"])[0] or "month_desc"
            limit_str = qs.get("limit", ["500"])[0]
            limit = int(limit_str) if limit_str.isdigit() else 500
            sim_str = qs.get("similarity_threshold", ["0.45"])[0]
            try:
                sim_thresh = float(sim_str)
            except ValueError:
                sim_thresh = 0.45

            res = self.backend.get_ideas_list_and_search(
                query=q_str,
                search_type=search_type,
                month_from=month_from,
                month_to=month_to,
                category=category,
                sort_by=sort_by,
                limit=limit,
                similarity_threshold=sim_thresh,
            )
            self._send_json(res)
            return

        # 2f. API: Backup Streaming & Direct Download
        if path == "/api/backup/download":
            qs = parse_qs(parsed.query)
            filename = qs.get("filename", [""])[0]
            if not filename or ".." in filename or "/" in filename or "\\" in filename:
                self._send_json({"error": "Invalid or missing backup filename."}, HTTPStatus.BAD_REQUEST)
                return

            out_dir = Path(getattr(self.backend.config, "resolved_output_dir", Path("output"))).expanduser().resolve()
            backup_file = out_dir / "backups" / filename
            if not backup_file.is_file():
                self._send_json({"error": f"Backup file '{filename}' not found."}, HTTPStatus.NOT_FOUND)
                return

            ext = backup_file.suffix.lower()
            content_type = "application/zip" if ext == ".zip" else "application/x-7z-compressed"
            safe_filename = backup_file.name.replace('"', '\\"')

            try:
                data = backup_file.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'attachment; filename="{safe_filename}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                logger.error(f"Error streaming backup archive {backup_file}: {e}")
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error streaming backup: {e}")
            return

        # 2g. API: Restore Status Check
        if path == "/api/backup/restore/status":
            status = self.backend.get_restore_status()
            self._send_json(status)
            return

        # 2h. API: Incremental Backup Diff Preview
        if path == "/api/backup/incremental/preview":
            qs = parse_qs(parsed.query)
            baseline_type = qs.get("baseline_type", ["neo4j"])[0]
            baseline_path = qs.get("baseline_path", [None])[0]
            try:
                preview = self.backend.preview_incremental_diff(
                    baseline_type=baseline_type,
                    baseline_path=baseline_path,
                )
                self._send_json(preview)
            except Exception as e:
                logger.error(f"Failed to preview incremental diff: {e}", exc_info=True)
                self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
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

        content_type = self.headers.get("Content-Type", "")

        # Backup Restore API (/api/backup/restore) - Handles both multipart/form-data file upload and raw binary
        if path in ("/api/backup/restore", "/api/restore"):
            if self.backend.is_restoring:
                self._send_json({"error": "A restore operation is already in progress."}, HTTPStatus.CONFLICT)
                return

            orig_filename = "uploaded_backup.zip"
            clean_mode = False
            clean_neo4j = False
            incremental_mode = False
            temp_file = None

            try:
                # Stream uploaded archive directly to temporary file on disk
                with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tf:
                    temp_file = Path(tf.name)

                    if "multipart/form-data" in content_type:
                        boundary_match = re.search(r'boundary=([^;]+)', content_type)
                        if not boundary_match:
                            self._send_json({"error": "Missing multipart boundary."}, HTTPStatus.BAD_REQUEST)
                            return
                        boundary = boundary_match.group(1).strip().strip('"')
                        fname, fields, _ = parse_multipart_to_file(
                            self.rfile, boundary, tf, max_content_length=content_length
                        )
                        if fname:
                            orig_filename = fname
                        clean_mode = fields.get("clean", "").strip().lower() in ("true", "1", "yes")
                        clean_neo4j = fields.get("clean_neo4j", "").strip().lower() in ("true", "1", "yes")
                        incremental_mode = fields.get(
                            "incremental", fields.get("is_incremental", "")
                        ).strip().lower() in ("true", "1", "yes")
                    else:
                        # Raw binary payload
                        remaining = content_length
                        while remaining > 0:
                            chunk_size = min(remaining, 65536)
                            chunk = self.rfile.read(chunk_size)
                            if not chunk:
                                break
                            tf.write(chunk)
                            remaining -= len(chunk)
                        qs = parse_qs(parsed.query)
                        orig_filename = qs.get("filename", [orig_filename])[0]
                        clean_mode = qs.get("clean", ["false"])[0].lower() in ("true", "1", "yes")
                        clean_neo4j = qs.get("clean_neo4j", ["false"])[0].lower() in ("true", "1", "yes")
                        incremental_mode = qs.get("incremental", ["false"])[0].lower() in ("true", "1", "yes")

                if not temp_file.is_file() or temp_file.stat().st_size == 0:
                    self._send_json({"error": "No backup file uploaded in request payload."}, HTTPStatus.BAD_REQUEST)
                    return

                # Ensure temp file has proper archive extension
                ext = Path(orig_filename).suffix.lower() or ".zip"
                if temp_file.suffix != ext:
                    proper_temp_file = temp_file.with_suffix(ext)
                    temp_file.replace(proper_temp_file)
                    temp_file = proper_temp_file

                res = self.backend.restore_from_backup(
                    archive_file=temp_file,
                    clean=clean_mode,
                    export_neo4j=True,
                    clean_neo4j=clean_neo4j,
                    incremental=incremental_mode,
                )
                self._send_json(res)
            except Exception as e:
                logger.error(f"Backup restore failed: {e}", exc_info=True)
                self._send_json({"error": f"Backup restore failed: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            finally:
                if temp_file and temp_file.is_file():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass
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

        # Incremental Backup Diff Preview API (/api/backup/incremental/preview)
        if path == "/api/backup/incremental/preview":
            baseline_type = body.get("baseline_type", "neo4j")
            baseline_path = body.get("baseline_path")
            try:
                preview = self.backend.preview_incremental_diff(
                    baseline_type=baseline_type,
                    baseline_path=baseline_path,
                )
                self._send_json(preview)
            except Exception as e:
                logger.error(f"Failed to preview incremental diff: {e}", exc_info=True)
                self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        # Incremental Backup Prepare API (/api/backup/incremental/prepare)
        if path == "/api/backup/incremental/prepare":
            baseline_type = body.get("baseline_type", "neo4j")
            baseline_path = body.get("baseline_path")
            fmt = body.get("format", "zip")
            include_books = body.get("include_books", True)
            try:
                res = self.backend.prepare_incremental_backup(
                    baseline_type=baseline_type,
                    baseline_path=baseline_path,
                    format_type=fmt,
                    include_books=include_books,
                )
                self._send_json(res)
            except Exception as e:
                logger.error(f"Failed to prepare incremental backup: {e}", exc_info=True)
                self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        # Backup Prepare API (/api/backup/prepare)
        if path in ("/api/backup/prepare", "/api/backup/create"):
            fmt = body.get("format", "zip")
            b_type = body.get("type", "full")
            indexed_only = body.get("indexed_only", True)
            if b_type == "incremental":
                baseline_type = body.get("baseline_type", "neo4j")
                baseline_path = body.get("baseline_path")
                include_books = body.get("include_books", True)
                try:
                    res = self.backend.prepare_incremental_backup(
                        baseline_type=baseline_type,
                        baseline_path=baseline_path,
                        format_type=fmt,
                        include_books=include_books,
                    )
                    self._send_json(res)
                except Exception as e:
                    logger.error(f"Failed to prepare incremental backup: {e}", exc_info=True)
                    self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            try:
                res = self.backend.prepare_full_backup(
                    format_type=fmt,
                    backup_type=b_type,
                    indexed_only=indexed_only,
                )
                self._send_json(res)
            except Exception as e:
                logger.error(f"Failed to prepare backup: {e}", exc_info=True)
                self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        # Ideas List & Search API (/api/ideas and /api/ideas/search)
        if path in ("/api/ideas", "/api/ideas/search"):
            q_str = str(body.get("query") or body.get("q") or "").strip()
            search_type = str(body.get("search_type") or body.get("mode") or "exact").strip()
            month_from = body.get("month_from") or body.get("from") or None
            month_to = body.get("month_to") or body.get("to") or None
            category = body.get("category") or None
            sort_by = str(body.get("sort_by") or "month_desc").strip()
            limit = int(body.get("limit") or 500)
            try:
                sim_thresh = float(body.get("similarity_threshold", 0.45))
            except (TypeError, ValueError):
                sim_thresh = 0.45

            try:
                res = self.backend.get_ideas_list_and_search(
                    query=q_str,
                    search_type=search_type,
                    month_from=month_from,
                    month_to=month_to,
                    category=category,
                    sort_by=sort_by,
                    limit=limit,
                    similarity_threshold=sim_thresh,
                )
                self._send_json(res)
            except Exception as e:
                logger.error(f"Ideas search API error: {e}", exc_info=True)
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

        def handle_error(self, request, client_address):
            exc_type, _, _ = sys.exc_info()
            if exc_type in (BrokenPipeError, ConnectionResetError):
                logger.debug(f"Client disconnected ({exc_type.__name__}) from {client_address}")
                return
            super().handle_error(request, client_address)

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
