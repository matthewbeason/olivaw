from __future__ import annotations

import json
import urllib.error

import pytest

from olivaw.capabilities.chat import ChatCapability
from olivaw.config import OlivawConfig
from olivaw.models import CompletionRequest


class CustomProviderError(Exception):
    pass


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("No provider is available."),
        urllib.error.URLError("connection refused"),
        urllib.error.HTTPError(
            url="http://localhost:11434/api/generate",
            code=500,
            msg="server error",
            hdrs=None,
            fp=None,
        ),
        TimeoutError("request timed out"),
        json.JSONDecodeError("bad json", "{", 1),
        KeyError("choices"),
        CustomProviderError("provider-specific failure"),
    ],
)
def test_chat_returns_actionable_message_for_provider_failures(monkeypatch, failure):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise failure

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    result = ChatCapability().run("hello", config=OlivawConfig())

    assert result.startswith("Chat provider unavailable:")
    assert type(failure).__name__ in result
    assert "olivaw health" in result
    assert "Ollama" in result

