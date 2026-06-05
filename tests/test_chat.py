from __future__ import annotations

import json
import urllib.error

import pytest

from olivaw.assistant.attribution import (
    CAPABILITY_UNAVAILABLE,
    MODEL_REASONED,
    SOURCE_BACKED,
)
from olivaw.capabilities.chat import ChatCapability
from olivaw.config import OlivawConfig
from olivaw.config import FileSourceConfig
from olivaw.models import CompletionRequest, CompletionResponse


class CustomProviderError(Exception):
    pass


WEATHER_PROMPT = "Hi could you tell me what the weather is in Phoenix az"
FORBIDDEN_WEATHER_CLAIMS = (
    "enable_openai_weather",
    "provide weather via cloud openai provider support",
    "openai can retrieve live weather",
)


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


def test_provider_failure_is_attributed_as_unavailable(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise RuntimeError("No provider is available.")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    result = ChatCapability().run_with_attribution("hello", config=OlivawConfig())

    assert result.attribution == CAPABILITY_UNAVAILABLE
    assert result.capability == "model provider"
    assert result.text.startswith("Chat provider unavailable:")


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
    assert "Distinguish source-backed facts from model reasoning." in request.system_prompt


def test_reasoning_request_calls_provider_and_is_model_reasoned(monkeypatch):
    captured: dict[str, CompletionRequest] = {}

    class CapturingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            captured["request"] = request
            return CompletionResponse(
                text="local-first means local systems are preferred",
                provider="test",
                model="test-model",
            )

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", CapturingRouter)

    result = ChatCapability().run_with_attribution(
        "Explain local-first architecture.", config=OlivawConfig()
    )

    assert captured["request"].prompt == "Explain local-first architecture."
    assert result.attribution == MODEL_REASONED
    assert result.capability == "chat"
    assert "local systems are preferred" in result.text


def test_weather_request_is_capability_unavailable_without_provider(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("unavailable capability should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    result = ChatCapability().run_with_attribution(
        "What's the weather in Phoenix?", config=OlivawConfig()
    )

    assert result.attribution == CAPABILITY_UNAVAILABLE
    assert result.capability == "weather source"
    assert "do not currently have a weather source configured" in result.text
    assert "WeatherSource" in result.text


def test_exact_weather_request_is_guarded_without_provider(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("weather request should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    result = ChatCapability().run_with_attribution(
        WEATHER_PROMPT,
        config=OlivawConfig(),
    )

    assert result.attribution == CAPABILITY_UNAVAILABLE
    assert result.capability == "weather source"
    assert "do not currently have a weather source configured" in result.text
    assert "WeatherSource" in result.text
    normalized = result.text.lower()
    for forbidden in FORBIDDEN_WEATHER_CLAIMS:
        assert forbidden not in normalized


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


def test_capability_question_is_source_backed_without_provider(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("capability question should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    result = ChatCapability().run_with_attribution(
        "What can you currently do?", config=OlivawConfig()
    )

    assert result.attribution == SOURCE_BACKED
    assert result.sources == ("capability-registry",)


def test_source_question_is_source_backed_without_provider(monkeypatch, tmp_path):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("source question should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)
    config = OlivawConfig(files=FileSourceConfig(directory=tmp_path))

    result = ChatCapability().run_with_attribution(
        "What sources are registered?", config=config
    )

    assert result.attribution == SOURCE_BACKED
    assert result.sources == ("manual", "files", "prime_observer", "core_signal")
    assert "Manual example source" in result.text
    assert "Local files" in result.text
    assert "Prime Observer" in result.text
    assert "Core Signal" in result.text
    assert "WeatherSource" in result.text
