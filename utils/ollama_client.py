"""
Thin Ollama HTTP client (embeddings + chat).

Why hand-rolled instead of the ``ollama`` pip package?
    * Zero extra dependencies — uses only the stdlib ``urllib``.
    * One place owns the base URL, timeout, and error handling.
    * Reused by THREE consumers (embedder, generator, judge) — DRY.

Ollama exposes a local REST API (default http://localhost:11434):
    POST /api/embeddings  {model, prompt}     -> {embedding: [...]}
    POST /api/chat        {model, messages}   -> {message: {content}}

``is_up()`` lets callers cheaply decide whether the local model server is
available before committing to it (used by the "auto" backends).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from utils.logger import get_logger

log = get_logger(__name__)


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def is_up(self, timeout_s: float = 3.0) -> bool:
        """Return True if the Ollama server answers /api/tags quickly."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def embed(self, model: str, text: str) -> list[float]:
        """Return the embedding vector for a single string."""
        out = self._post("/api/embeddings", {"model": model, "prompt": text})
        return out["embedding"]

    def chat(self, model: str, system: str, user: str,
             temperature: float = 0.3, max_tokens: int = 400) -> str:
        """Single-turn chat completion; returns the assistant message text."""
        out = self._post("/api/chat", {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        })
        return out["message"]["content"].strip()
