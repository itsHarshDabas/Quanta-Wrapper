"""Static server for the Quanta web UI (dedicated local port, no dependencies).

The UI talks to the real API from the browser; the only thing served besides
static files is ``/config.json`` (the API base URL). The API key is typed into
the page and kept in that browser's localStorage, never written by this server.
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

UI_DIR = Path(__file__).resolve().parent / "ui"
DEFAULT_UI_PORT = 8788


class Handoff:
    """One-time, short-lived nonce that lets a freshly opened browser tab fetch the API key.

    The key itself is never placed in a URL or process argument; only the nonce is, and it
    is useless after one exchange or after ``ttl`` seconds.
    """

    def __init__(self, key: str, ttl: float = 60.0):
        self._key, self._ttl = key, ttl
        self._nonce: str | None = None
        self._expires = 0.0
        self._lock = threading.Lock()

    def issue(self) -> str:
        with self._lock:
            self._nonce = secrets.token_urlsafe(24)
            self._expires = time.monotonic() + self._ttl
            return self._nonce

    def redeem(self, nonce: str) -> str | None:
        with self._lock:
            ok = self._nonce is not None and time.monotonic() < self._expires and hmac.compare_digest(self._nonce, nonce)
            self._nonce = None  # single use, even on a wrong guess
            return self._key if ok else None


class _Handler(SimpleHTTPRequestHandler):
    api_base = "http://127.0.0.1:8787"
    handoff: Handoff | None = None

    def do_GET(self):
        parts = urlsplit(self.path)
        path = parts.path
        if path == "/handoff":
            nonce = dict(pair.split("=", 1) for pair in parts.query.split("&") if "=" in pair).get("n", "")
            local = self.client_address[0] in ("127.0.0.1", "::1")
            key = self.handoff.redeem(nonce) if self.handoff and local else None
            body = json.dumps({"key": key} if key else {"error": "invalid_or_expired"}).encode()
            self.send_response(200 if key else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/config.json":
            body = json.dumps({"api": self.api_base}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def end_headers(self):
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(self.api_base))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         f"default-src 'self'; connect-src 'self' {origin}; style-src 'self' 'unsafe-inline'; "
                         "img-src 'self' data:; frame-ancestors 'none'")
        super().end_headers()

    def log_message(self, *args):  # keep the console quiet
        pass


def make_ui_server(api_base: str, host: str = "127.0.0.1", port: int = DEFAULT_UI_PORT,
                   handoff: Handoff | None = None) -> ThreadingHTTPServer:
    handler = type("QuantaUIHandler", (_Handler,), {"api_base": api_base.rstrip("/"), "handoff": handoff})
    return ThreadingHTTPServer((host, port), partial(handler, directory=str(UI_DIR)))


def start_ui_thread(api_base: str, host: str = "127.0.0.1", port: int = DEFAULT_UI_PORT,
                    handoff: Handoff | None = None) -> ThreadingHTTPServer:
    server = make_ui_server(api_base, host, port, handoff)
    threading.Thread(target=server.serve_forever, name="quanta-ui", daemon=True).start()
    return server
