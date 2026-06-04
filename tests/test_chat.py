from __future__ import annotations

import json
import urllib.error

import pytest

from olivaw.capabilities.chat import ChatCapability
from olivaw.config import OlivawConfig
from olivaw.models import CompletionRequest, CompletionResponse


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


def test_chat_sends_identity_context_to_provider(monkeypatch):
    captured: dict[str, CompletionRequest] = {}

    class CapturingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            captured["request"] = request
            return CompletionResponse(
                text="grounded response",
                provider="test",
                model="test-model",
            )

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", CapturingRouter)

    result = ChatCapability().run("hello", config=OlivawConfig())

    request = captured["request"]
    assert result == "grounded response"
    assert request.prompt == "hello"
    assert request.system_prompt is not None
    assert "You are Olivaw." in request.system_prompt
    assert "Current implemented capabilities:" in request.system_prompt
    assert "Not implemented yet:" in request.system_prompt
    assert "calendar integration" in request.system_prompt
    assert "Do not claim unavailable capabilities." in request.system_prompt


def test_capability_question_returns_grounded_answer_without_provider(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("capability question should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    result = ChatCapability().run("What can you currently do?", config=OlivawConfig())
    implemented, not_implemented = result.split("Not implemented yet:")

    assert "deterministic briefing generation" in implemented
    assert "provider health reporting" in implemented
    assert "calendar integration" not in implemented
    assert "weather lookup" not in implemented
    assert "calendar integration" in not_implemented
    assert "weather lookup" in not_implemented
    assert "persistent memory" in not_implemented
