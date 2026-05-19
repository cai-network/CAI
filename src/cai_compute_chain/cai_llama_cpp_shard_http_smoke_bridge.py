# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from .cai_llama_cpp_shard_smoke_runner import handle_smoke_runner_request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local HTTP smoke bridge for the CAI LLM shard ABI.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9257)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(
        (str(args.host or "127.0.0.1"), int(args.port or 9257)),
        _BridgeHandler,
    )
    print(
        json.dumps(
            {
                "status": "listening",
                "host": server.server_address[0],
                "port": server.server_address[1],
                "endpoint": (
                    f"http://{server.server_address[0]}:"
                    f"{server.server_address[1]}/cai-shard"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def handle_http_smoke_bridge_request_body(
    raw_body: bytes,
) -> tuple[int, dict[str, Any]]:
    try:
        request = json.loads(bytes(raw_body or b"").decode("utf-8") or "{}")
        if not isinstance(request, dict):
            raise ValueError("Shard adapter request must be an object.")
        return 200, handle_smoke_runner_request(request)
    except Exception as exc:
        return 400, {"status": "error", "error": str(exc)}


class _BridgeHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        status_code, payload = handle_http_smoke_bridge_request_body(raw_body)
        self._send_json(status_code, payload)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    raise SystemExit(main())
