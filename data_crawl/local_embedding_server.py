#!/usr/bin/env python3
"""Serve the local BGE-M3 embedding demo and inference endpoint."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from local_embeddings import MODEL_NAME, choose_device, embed_texts, load_model


HOST = "127.0.0.1"
PORT = 8765
MAX_TEXT_LENGTH = 25_000
PAGE = Path(__file__).with_name("embedding_demo.html")


class EmbeddingHandler(BaseHTTPRequestHandler):
    model = None
    device = "cpu"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            body = json.dumps({"ok": True, "model": MODEL_NAME, "device": self.device}).encode()
            self._send(200, body, "application/json; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/api/embed":
            self._send(404, b'{"error":"Not found"}', "application/json; charset=utf-8")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_TEXT_LENGTH * 4:
                raise ValueError(f"Request body must be 1 to {MAX_TEXT_LENGTH * 4} bytes.")
            request = json.loads(self.rfile.read(length))
            text = request.get("text", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Enter some text before generating an embedding.")
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"Text must be {MAX_TEXT_LENGTH:,} characters or fewer.")

            embedding = embed_texts(self.model, [text])[0]
            response = {
                "model": MODEL_NAME,
                "device": self.device,
                "dimensions": int(embedding.shape[0]),
                "embedding": embedding.tolist(),
            }
            self._send(200, json.dumps(response).encode(), "application/json; charset=utf-8")
        except (ValueError, json.JSONDecodeError) as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self._send(400, body, "application/json; charset=utf-8")
        except Exception as exc:
            body = json.dumps({"error": f"Embedding failed: {exc}"}).encode()
            self._send(500, body, "application/json; charset=utf-8")

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    device = choose_device("auto")
    print(f"Loading {MODEL_NAME} locally on {device}; this may take a moment...")
    EmbeddingHandler.model = load_model(device)
    EmbeddingHandler.device = device
    server = ThreadingHTTPServer((HOST, PORT), EmbeddingHandler)
    print(f"Ready: http://{HOST}:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping local embedding server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
