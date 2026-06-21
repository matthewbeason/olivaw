from __future__ import annotations

import json
import urllib.error

import pytest

from olivaw.assistant.attribution import (
    DERIVED,
    MODEL_KNOWLEDGE,
    MODEL_UNAVAILABLE,
    SOURCE_BACKED,
    UNKNOWN_OPERATIONAL_STATE,
    UNAVAILABLE_SOURCE_BACKED,
)
from olivaw.capabilities.chat import ChatCapability
from olivaw.config import OlivawConfig
from olivaw.config import (
    CoreSignalSourceConfig,
    FileSourceConfig,
    PrimeObserverSourceConfig,
    WeatherSourceConfig,
)
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

    assert result.attribution == MODEL_UNAVAILABLE
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


def test_reasoning_request_calls_provider_and_is_model_knowledge(monkeypatch):
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
    assert result.attribution == MODEL_KNOWLEDGE
    assert result.capability == "chat"
    assert "local systems are preferred" in result.text
    assert result.provenance_label == "Knowledge mode"
    assert result.provenance_detail == "Model knowledge"


def test_general_knowledge_request_stays_model_knowledge(monkeypatch):
    class CapturingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            return CompletionResponse(
                text="You have power over your mind, not outside events.",
                provider="test",
                model="test-model",
            )

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", CapturingRouter)

    result = ChatCapability().run_with_attribution(
        "Give me a Marcus Aurelius quote.", config=OlivawConfig()
    )

    assert result.attribution == MODEL_KNOWLEDGE
    assert result.provenance_label == "Knowledge mode"
    assert result.provenance_detail == "Model knowledge"


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

    assert result.attribution == UNAVAILABLE_SOURCE_BACKED
    assert result.capability == "weather conditions"
    assert "The source exists but is currently unavailable." in result.text
    assert "I would need a source that provides weather data." in result.text
    assert result.provenance_label == "Knowledge mode"
    assert result.provenance_detail == "Unavailable source-backed state"


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

    assert result.attribution == UNAVAILABLE_SOURCE_BACKED
    assert result.capability == "weather conditions"
    assert "The source exists but is currently unavailable." in result.text
    normalized = result.text.lower()
    for forbidden in FORBIDDEN_WEATHER_CLAIMS:
        assert forbidden not in normalized


def test_weather_request_uses_weather_source_when_available(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("weather request should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)
    monkeypatch.setattr(
        "olivaw.sources.weather.OpenMeteoProvider.fetch_forecast",
        lambda self, *, latitude, longitude, units: {
            "current": {
                "time": "2026-06-17T08:00",
                "temperature_2m": 72,
                "weather_code": 0,
                "wind_speed_10m": 6,
            },
            "current_units": {
                "temperature_2m": "°F",
                "wind_speed_10m": "mph",
            },
            "daily": {
                "time": ["2026-06-17"],
                "temperature_2m_max": [86],
                "temperature_2m_min": [68],
                "precipitation_probability_max": [10],
                "weather_code": [0],
            },
            "daily_units": {
                "temperature_2m_max": "°F",
                "temperature_2m_min": "°F",
            },
        },
    )

    result = ChatCapability().run_with_attribution(
        "What's the weather today?",
        config=OlivawConfig(
            weather=WeatherSourceConfig(
                enabled=True,
                latitude=33.4484,
                longitude=-112.074,
                location_name="Phoenix",
            )
        ),
    )

    assert result.attribution == SOURCE_BACKED
    assert result.sources == ("weather",)
    assert result.capability == "weather lookup"
    assert result.provenance_label == "Source"
    assert result.provenance_detail == "Weather"
    assert result.text.startswith("Weather:")
    assert "Currently 72°F and clear." in result.text


def test_network_question_uses_source_grounded_derived_response(monkeypatch, tmp_path):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("network question should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    prime_dir.mkdir()
    core_dir.mkdir()
    (prime_dir / "network_attribution.json").write_text(
        """
{
  "generated_at": "2026-06-17T14:23:00+00:00",
  "current_attribution": {
    "label": "Likely upstream (ISP / path)",
    "status": "likely_upstream",
    "confidence": "medium",
    "evidence": ["WAN degraded while LAN remained healthy."]
  }
}
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Morning Brief",
  "date": "2026-06-17",
  "status": "Watch",
  "summary": "One interpreted slowdown event is present.",
  "recommended_action": "No action unless people noticed issues."
}
""",
        encoding="utf-8",
    )

    result = ChatCapability().run_with_attribution(
        "How was the network overnight?",
        config=OlivawConfig(
            prime_observer=PrimeObserverSourceConfig(directory=prime_dir),
            core_signal=CoreSignalSourceConfig(directory=core_dir),
        ),
    )

    assert result.attribution == DERIVED
    assert result.sources == ("prime_observer", "core_signal")
    assert result.provenance_label == "Derived from"
    assert result.provenance_detail == "Prime Observer + Core Signal"
    assert "One interpreted slowdown event is present." in result.text
    assert "Prime Observer reports:" in result.text
    assert "WAN degraded while LAN remained healthy." in result.text


def test_network_question_returns_unavailable_when_sources_cannot_answer(monkeypatch, tmp_path):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("network question should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    result = ChatCapability().run_with_attribution(
        "How was the network overnight?",
        config=OlivawConfig(
            prime_observer=PrimeObserverSourceConfig(directory=tmp_path / "missing-prime"),
            core_signal=CoreSignalSourceConfig(directory=tmp_path / "missing-core"),
        ),
    )

    assert result.attribution == UNAVAILABLE_SOURCE_BACKED
    assert "The source exists but is currently unavailable." in result.text
    assert (
        "I would need a source that provides Prime Observer evidence or Core Signal interpretation."
        in result.text
    )
    assert result.provenance_label == "Knowledge mode"
    assert result.provenance_detail == "Unavailable source-backed state"


def test_disk_usage_question_declines_without_invented_estimate(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("disk usage question should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    result = ChatCapability().run_with_attribution(
        "What is disk usage?",
        config=OlivawConfig(),
    )

    assert result.attribution == UNKNOWN_OPERATIONAL_STATE
    assert result.provenance_label == "Knowledge mode"
    assert result.provenance_detail == "Unknown operational state"
    assert "I don't currently have a source that can answer that." in result.text
    assert "I would need a source that provides disk utilization." in result.text
    normalized = result.text.lower()
    assert "estimate" not in normalized
    assert "gb" not in normalized
    assert "monitoring" not in normalized


def test_memory_usage_question_declines_without_invented_telemetry(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("memory usage question should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    result = ChatCapability().run_with_attribution(
        "How much memory am I using?",
        config=OlivawConfig(),
    )

    assert result.attribution == UNKNOWN_OPERATIONAL_STATE
    assert "I would need a source that provides memory utilization." in result.text
    normalized = result.text.lower()
    assert "estimate" not in normalized
    assert "monitoring" not in normalized
    assert "i've been monitoring" not in normalized


def test_monitoring_question_declines_without_claiming_monitoring(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request: CompletionRequest):
            raise AssertionError("monitoring question should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    result = ChatCapability().run_with_attribution(
        "Have you been monitoring the provider?",
        config=OlivawConfig(),
    )

    assert result.attribution == UNKNOWN_OPERATIONAL_STATE
    assert "I would need a source that provides monitoring telemetry." in result.text
    assert "I've been monitoring" not in result.text


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
    assert "weather lookup" in implemented
    assert "calendar integration" in not_implemented
    assert "weather lookup" not in not_implemented
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
    assert result.sources == (
        "manual",
        "files",
        "prime_observer",
        "core_signal",
        "weather",
    )
    assert "Manual example source" in result.text
    assert "Local files" in result.text
    assert "Prime Observer" in result.text
    assert "Core Signal" in result.text
    assert "CalendarSource" in result.text
