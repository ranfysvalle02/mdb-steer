"""Client over Ollama's native API.

We use the native API rather than the OpenAI-compatible one because it reports server-side
timings. That separates model *load* time (an artefact of memory pressure) from real compute
(prompt processing + token generation), which is what we price.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

_NS_PER_MS = 1_000_000
_RETRIES = 4  # a killed runner (e.g. memory pressure) is restarted by Ollama on the next request


@dataclass(frozen=True)
class Completion:
    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float  # client wall-clock, includes load and network
    compute_ms: float  # server-side prompt eval + generation only
    load_ms: float


class Ollama:
    def __init__(self, base_url: str, timeout_s: float = 900) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> Completion:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"

        start = time.perf_counter()
        data = self._post("/api/chat", body)
        latency_ms = (time.perf_counter() - start) * 1000

        return Completion(
            model=model,
            text=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            compute_ms=(data.get("prompt_eval_duration", 0) + data.get("eval_duration", 0)) / _NS_PER_MS,
            load_ms=data.get("load_duration", 0) / _NS_PER_MS,
        )

    def embed(self, model: str, text: str) -> list[float]:
        return self._post("/api/embed", {"model": model, "input": text, "keep_alive": -1})["embeddings"][0]

    def _post(self, path: str, body: dict) -> dict:
        """POST with exponential backoff on server errors and dropped connections."""
        for attempt in range(_RETRIES + 1):
            try:
                res = self._http.post(path, json=body)
                if res.status_code < 500:
                    res.raise_for_status()
                    return res.json()
                error: Exception = httpx.HTTPStatusError(f"{res.status_code} from {path}", request=res.request, response=res)
            except httpx.TransportError as exc:
                error = exc
            if attempt == _RETRIES:
                raise error
            time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")
