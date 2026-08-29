"""Serve the Redis AI Workbench on localhost."""

from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import signal
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from redis_ai_portfolio.config import PortfolioSettings
from redis_ai_portfolio.redis import create_redis_client
from redis_ai_portfolio.workbench import RedisAIWorkbench

MAX_REQUEST_BYTES = 16 * 1024
STATIC_ROOT = Path(__file__).resolve().parent / "static"


class WorkbenchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], engine: RedisAIWorkbench) -> None:
        super().__init__(address, WorkbenchHandler)
        self.engine = engine


class WorkbenchHandler(BaseHTTPRequestHandler):
    """Small JSON/SSE API plus static-file delivery with restrictive headers."""

    server: WorkbenchHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format_string: str, *args: Any) -> None:
        # Never echo request bodies or query strings into terminal history.
        safe_path = urlsplit(self.path).path
        sys.stderr.write(f"workbench {self.command} {safe_path} {args[1]}\n")

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'",
        )

    def _json(self, payload: Mapping[str, Any], *, status: int = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self._security_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def _read_json(self) -> Mapping[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError("Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length", "")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("A valid Content-Length is required") from exc
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"Request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must contain valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/ready":
            status = self.server.engine.status()
            self._json(status, status=HTTPStatus.OK if status["ready"] else HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if path == "/api/status":
            self._json(self.server.engine.status())
            return
        if path == "/api/redis":
            try:
                self._json(self.server.engine.redis_inspector())
            except Exception:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "Redis inspector is unavailable")
            return
        if path.startswith("/api/runs/"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "events":
                self._stream_events(parts[2])
                return
            if len(parts) == 3:
                try:
                    self._json(self.server.engine.events.snapshot(parts[2]))
                except KeyError:
                    self._error(HTTPStatus.NOT_FOUND, "Run not found")
                return
        if path.startswith("/api/"):
            self._error(HTTPStatus.NOT_FOUND, "API route not found")
            return
        self._static(path)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path != "/api/runs":
            self._error(HTTPStatus.NOT_FOUND, "API route not found")
            return
        try:
            payload = self._read_json()
            demo = payload.get("demo", "")
            if not isinstance(demo, str):
                raise ValueError("demo must be text")
            run_id = self.server.engine.start(demo, payload)
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._json({"run_id": run_id, "status": "running"}, status=HTTPStatus.ACCEPTED)

    def do_DELETE(self) -> None:
        path = urlsplit(self.path).path
        if path != "/api/workbench":
            self._error(HTTPStatus.NOT_FOUND, "API route not found")
            return
        try:
            payload = self._read_json()
            if payload.get("confirm") != "reset":
                raise ValueError("Reset confirmation is required")
            result = self.server.engine.reset()
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except Exception:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "Workbench reset failed")
            return
        self._json(result)

    def _stream_events(self, run_id: str) -> None:
        try:
            self.server.engine.events.snapshot(run_id)
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "Run not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._security_headers()
        self.end_headers()
        sequence = 0
        try:
            while True:
                events, terminal = self.server.engine.events.wait_for_events(run_id, sequence)
                for event in events:
                    sequence = int(event["sequence"])
                    self._write_sse("flight", event, event_id=str(sequence))
                if terminal:
                    snapshot = self.server.engine.events.snapshot(run_id)
                    self._write_sse("complete", snapshot)
                    break
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _write_sse(
        self,
        event_name: str,
        payload: Mapping[str, Any],
        *,
        event_id: str | None = None,
    ) -> None:
        if event_id is not None:
            self.wfile.write(f"id: {event_id}\n".encode("utf-8"))
        self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        self.wfile.write(f"data: {encoded}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (STATIC_ROOT / relative).resolve()
        if not target.is_relative_to(STATIC_ROOT) or not target.is_file():
            self._error(HTTPStatus.NOT_FOUND, "Asset not found")
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; localhost is the safe default")
    parser.add_argument("--port", type=int, default=8123, help="Workbench port")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    root = Path(__file__).resolve().parents[1]
    try:
        settings = PortfolioSettings.from_env(root / ".env")
    except ValueError as exc:
        print(f"Workbench configuration error: {exc}", file=sys.stderr)
        return 2
    redis_client = create_redis_client(settings.redis_url)
    try:
        engine = RedisAIWorkbench(settings, redis_client)
    except ValueError as exc:
        redis_client.close()
        print(f"Workbench configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        server = WorkbenchHTTPServer((args.host, args.port), engine)
    except OSError as exc:
        engine.close()
        redis_client.close()
        if exc.errno == errno.EADDRINUSE:
            print(
                f"Port {args.port} is already in use. Stop the existing listener or run "
                f"`make workbench WORKBENCH_PORT={args.port + 1}`.",
                file=sys.stderr,
            )
            return 2
        raise

    def stop_server(*_: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    print(f"Redis AI Workbench: http://{args.host}:{args.port}")
    print(f"Model backend: {engine.backend.display_name}")
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        engine.close()
        redis_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
