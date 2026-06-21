from __future__ import annotations

import json

from olivaw.cli import (
    BenchmarkResult,
    benchmark_chat_models,
    installed_ollama_models,
    prompt_size_report,
    summarize_benchmark_results,
)
from olivaw.config import OlivawConfig


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


def test_installed_ollama_models_parse_runtime_metadata(monkeypatch):
    def fake_urlopen(url, timeout):
        assert url == "http://localhost:11434/api/tags"
        assert timeout == 2.0
        return FakeHTTPResponse(
            {
                "models": [
                    {
                        "model": "llama3.2:3b",
                        "size": 2019393189,
                        "details": {
                            "parameter_size": "3.2B",
                            "quantization_level": "Q4_K_M",
                            "context_length": 131072,
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    models = installed_ollama_models(OlivawConfig())

    assert len(models) == 1
    assert models[0].name == "llama3.2:3b"
    assert models[0].size_bytes == 2019393189
    assert models[0].parameter_size == "3.2B"
    assert models[0].quantization_level == "Q4_K_M"
    assert models[0].context_length == 131072


def test_benchmark_chat_models_uses_transient_model_override(monkeypatch):
    seen_models: list[str] = []

    class FakeChatCapability:
        def run_with_attribution(self, prompt, config):
            from olivaw.assistant.attribution import AttributedResponse, MODEL_KNOWLEDGE

            seen_models.append(config.local.model)
            return AttributedResponse(
                text=f"{config.local.model}: {prompt}",
                attribution=MODEL_KNOWLEDGE,
                capability="chat",
                provenance_label="Knowledge mode",
                provenance_detail="Model knowledge",
                metrics={
                    "model_invoked": True,
                    "total_request_duration_ms": 100,
                    "ollama_total_duration_ms": 90,
                },
            )

    monkeypatch.setattr("olivaw.cli.ChatCapability", FakeChatCapability)

    results = benchmark_chat_models(
        config=OlivawConfig(),
        models=("llama3.2:3b", "llama3.1:8b"),
        attempts=1,
        prompts=(("knowledge", "Who was Marcus Aurelius?"),),
    )

    assert seen_models == ["llama3.2:3b", "llama3.1:8b"]
    assert [result.model for result in results] == ["llama3.2:3b", "llama3.1:8b"]
    assert all(result.provenance_detail == "Model knowledge" for result in results)


def test_summarize_benchmark_results_does_not_depend_on_timing_thresholds():
    results = (
        BenchmarkResult(
            model="llama3.2:3b",
            category="weather",
            prompt="What's the weather today?",
            attempt=1,
            text="Weather: clear.",
            attribution="source_backed",
            sources=("weather",),
            capability="weather lookup",
            provenance_label="Source",
            provenance_detail="Weather",
            metrics={
                "model_invoked": False,
                "total_request_duration_ms": 10,
            },
        ),
        BenchmarkResult(
            model="llama3.2:3b",
            category="weather",
            prompt="What's the weather today?",
            attempt=2,
            text="Weather: clear.",
            attribution="source_backed",
            sources=("weather",),
            capability="weather lookup",
            provenance_label="Source",
            provenance_detail="Weather",
            metrics={
                "model_invoked": False,
                "total_request_duration_ms": 20,
            },
        ),
    )

    summary = summarize_benchmark_results(results)

    assert summary == [
        {
            "model": "llama3.2:3b",
            "category": "weather",
            "attempts": 2,
            "avg_total_request_duration_ms": 15,
            "avg_ollama_total_duration_ms": None,
            "model_invoked": False,
            "attribution": "source_backed",
            "provenance": "Weather",
        }
    ]


def test_prompt_size_report_exposes_estimated_prompt_sizes():
    report = prompt_size_report()

    assert report["chat_system_prompt"]["characters"] > 0
    assert report["chat_system_prompt"]["approx_tokens"] > 0
    assert report["health_review_system_prompt"]["characters"] > 0
    assert report["health_review_user_prompt"]["characters"] == 0
