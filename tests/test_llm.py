import httpx
import pytest

from mdb_steer import llm as llm_mod
from mdb_steer.llm import Ollama


def client_with(responses, monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    calls = iter(responses)

    def handler(request):
        r = next(calls)
        if isinstance(r, Exception):
            raise r
        return r

    ollama = Ollama("http://ollama")
    ollama._http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return ollama


CHAT = {"message": {"content": "hi"}, "prompt_eval_count": 3, "eval_count": 5,
        "prompt_eval_duration": 2_000_000, "eval_duration": 8_000_000, "load_duration": 50_000_000}


def test_chat_prices_compute_not_load(monkeypatch) -> None:
    c = client_with([httpx.Response(200, json=CHAT)], monkeypatch).chat("m", "q")
    assert (c.text, c.prompt_tokens, c.completion_tokens) == ("hi", 3, 5)
    assert c.compute_ms == 10.0 and c.load_ms == 50.0


def test_retries_server_errors_and_dropped_connections(monkeypatch) -> None:
    responses = [httpx.Response(500), httpx.ConnectError("reset"), httpx.Response(200, json=CHAT)]
    assert client_with(responses, monkeypatch).chat("m", "q").text == "hi"


def test_client_errors_are_not_retried(monkeypatch) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        client_with([httpx.Response(404)], monkeypatch).chat("m", "q")


def test_gives_up_after_retries(monkeypatch) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        client_with([httpx.Response(500)] * (llm_mod._RETRIES + 1), monkeypatch).chat("m", "q")
