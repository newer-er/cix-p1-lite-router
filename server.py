#!/usr/bin/env python3
"""Serve the demo and forward requests to the Lite Router on the board."""

import argparse
import http.client
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from replay import replay_chat


ROOT = Path(__file__).resolve().parent
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/hero.js": ("hero.js", "text/javascript; charset=utf-8"),
    "/anyint-mark.png": ("anyint-mark.png", "image/png"),
}
PROMPT_LIMIT = 12_000
REQUEST_LIMIT = 64 * 1024
RESPONSE_LIMIT = 4 * 1024 * 1024
ROUTER_HTTP = build_opener(ProxyHandler({}))
DEFAULT_CLASSIFIER = "m47b"


@dataclass(frozen=True)
class Classifier:
    name: str
    parameters: str
    version_prefix: str
    router_url: str
    head_sources: dict[str, str] | None = None


class DemoServer(ThreadingHTTPServer):
    def __init__(self, port: int, router_url: str, intent_router_url: str | None = None, *, replay: bool = False):
        self.replay = replay
        self.classifiers = {DEFAULT_CLASSIFIER: Classifier("M47b", "0.8B", "m47b_", router_url)}
        if intent_router_url or replay:
            self.classifiers["intent06b"] = Classifier(
                "Qwen3", "0.6B", "intent06b_", intent_router_url or "",
                {"domain": "0.8B", "intent_primary": "0.6B", "requires_tools": "0.8B"},
            )
        super().__init__(("127.0.0.1", port), DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def send_body(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status: HTTPStatus, data: dict) -> None:
        self.send_body(status, "application/json; charset=utf-8", json.dumps(data, ensure_ascii=True).encode("utf-8"))

    def check_origin(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        port = self.server.server_port
        origin = self.headers.get("Origin")
        if host in (f"127.0.0.1:{port}", f"localhost:{port}") and origin in (None, f"http://{host}"):
            return True
        self.send_json(HTTPStatus.FORBIDDEN, {"error": "local_origin_required"})
        return False

    def get_classifier(self, classifier_id: object) -> Classifier | None:
        profile = self.server.classifiers.get(classifier_id) if isinstance(classifier_id, str) else None
        if profile is None:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "classifier_unknown"})
        return profile

    def forward(self, classifier: Classifier, path: str, timeout: int, payload: dict | None = None) -> None:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            classifier.router_url + path,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with ROUTER_HTTP.open(request, timeout=timeout) as upstream:
                response = upstream.read(RESPONSE_LIMIT + 1)
            if len(response) > RESPONSE_LIMIT:
                raise ValueError("response exceeded limit")
            data = json.loads(response)
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
        except HTTPError as error:
            error.close()
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": "router_rejected", "router_status": error.code})
        except TimeoutError:
            self.send_json(HTTPStatus.GATEWAY_TIMEOUT, {"error": "router_timeout"})
        except (URLError, OSError, http.client.HTTPException):
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": "router_unreachable"})
        except (ValueError, RecursionError):
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": "router_invalid_response"})
        else:
            self.send_json(HTTPStatus.OK, data)

    def do_GET(self) -> None:
        if not self.check_origin():
            return
        if self.path == "/api/classifiers":
            self.send_json(HTTPStatus.OK, {
                "default": DEFAULT_CLASSIFIER,
                "models": [
                    {"id": key, "name": value.name, "parameters": value.parameters,
                     "version_prefix": value.version_prefix, "head_sources": value.head_sources}
                    for key, value in self.server.classifiers.items()
                ],
            })
            return
        path, _, query = self.path.partition("?")
        if path == "/api/health":
            ids = parse_qs(query, keep_blank_values=True).get("classifier", [DEFAULT_CLASSIFIER])
            profile = self.get_classifier(ids[0] if len(ids) == 1 else None)
            if profile:
                if self.server.replay:
                    self.send_json(HTTPStatus.OK, {"mode": "replay"})
                else:
                    self.forward(profile, "/health", timeout=4)
            return
        file = FILES.get(self.path)
        if file is None:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        name, kind = file
        self.send_body(HTTPStatus.OK, kind, (ROOT / name).read_bytes())

    def do_POST(self) -> None:
        if not self.check_origin():
            return
        if self.path != "/api/chat":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "json_required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= REQUEST_LIMIT:
            status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE if length > REQUEST_LIMIT else HTTPStatus.BAD_REQUEST
            self.send_json(status, {"error": "request_too_large" if length > REQUEST_LIMIT else "invalid_request"})
            return
        try:
            data = json.loads(self.rfile.read(length))
        except (ValueError, RecursionError):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        prompt = data.get("prompt") if isinstance(data, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "prompt_required"})
            return
        if len(prompt) > PROMPT_LIMIT:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "prompt_too_long"})
            return

        classifier_id = data.get("classifier", DEFAULT_CLASSIFIER)
        profile = self.get_classifier(classifier_id)
        if not profile:
            return
        if self.server.replay:
            result = replay_chat(prompt.strip(), classifier_id)
            if result:
                self.send_json(HTTPStatus.OK, result)
            else:
                self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "replay_example_required"})
            return
        self.forward(profile, "/openai/v1/chat/completions", timeout=95, payload={
            "model": "auto",
            "messages": [{"role": "user", "content": prompt.strip()}],
            "stream": False,
            "max_tokens": 640,
        })


def router_origin(address: str) -> str:
    try:
        origin = urlsplit(address)
        origin_port = origin.port
    except ValueError:
        raise argparse.ArgumentTypeError("需要有效的 HTTP 地址") from None
    if (origin.scheme not in ("http", "https") or not origin.hostname or origin.username is not None
            or origin.password is not None or origin.path not in ("", "/") or origin.query or origin.fragment
            or origin_port == 0):
        raise argparse.ArgumentTypeError("需要 HTTP 服务的根地址，例如 http://127.0.0.1:18810")
    return address.rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行此芯 P1 智能路由演示页")
    parser.add_argument("--port", type=int, default=18812, help="本机页面端口，默认 18812")
    parser.add_argument("--router-url", type=router_origin, default="http://127.0.0.1:18810", help="M47b 0.8B 的板端 Router 地址")
    parser.add_argument("--intent-router-url", type=router_origin, help="0.6B 意图模式的板端 Router 地址")
    parser.add_argument("--replay", action="store_true", help="离线回放公开示例")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port 必须在 0 至 65535 之间")
    if args.intent_router_url == args.router_url:
        parser.error("两个分类器需要各自的 Router 地址")

    with DemoServer(args.port, args.router_url, args.intent_router_url, replay=args.replay) as server:
        print(f"Local:  http://127.0.0.1:{server.server_port}", flush=True)
        if server.replay:
            print("离线回放", flush=True)
        else:
            for profile in server.classifiers.values():
                print(f"{profile.name} {profile.parameters}: {profile.router_url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
