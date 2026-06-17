from __future__ import annotations

import json

from olivaw.config import LocalProviderConfig
from olivaw.models import CompletionRequest
from olivaw.providers.ollama import OllamaProvider


class FakeHTTPResponse:
    status = 200

    def __init__(self, data: dict[str, object]):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.data).encode("utf-8")


def test_ollama_models_parse_api_tags_response(monkeypatch):
    def fake_urlopen(url, timeout):
        assert url == "http://127.0.0.1:11434/api/tags"
        assert timeout == 1.0
        return FakeHTTPResponse(
            {
                "models": [
                    {"name": "llama3.2:3b", "model": "llama3.2:3b"},
                    {"name": "llama3.1:8b", "model": "llama3.1:8b"},
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = OllamaProvider(
        LocalProviderConfig(base_url="http://127.0.0.1:11434", model="llama3.2:3b")
    )

    assert provider.models() == ("llama3.2:3b", "llama3.1:8b")


def test_ollama_complete_posts_model_prompt_and_reads_response(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["method"] = request.get_method()
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"response": "generated review"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = OllamaProvider(
        LocalProviderConfig(base_url="http://127.0.0.1:11434", model="llama3.2:3b"),
        complete_timeout=45.0,
    )
    response = provider.complete(
        CompletionRequest(prompt="hello", system_prompt="system")
    )

    assert captured == {
        "url": "http://127.0.0.1:11434/api/generate",
        "timeout": 45.0,
        "method": "POST",
        "payload": {
            "model": "llama3.2:3b",
            "prompt": "hello",
            "system": "system",
            "stream": False,
        },
    }
    assert response.text == "generated review"
    assert response.provider == "ollama"
    assert response.model == "llama3.2:3b"
