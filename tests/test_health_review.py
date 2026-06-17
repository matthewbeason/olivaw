from __future__ import annotations

import pytest

from olivaw.briefing.health_review import (
    HEALTH_REVIEW_SYSTEM_PROMPT,
    UNAVAILABLE_TEXT,
    build_health_review_digest,
    build_health_review_prompt,
    generate_health_review,
)
from olivaw.config import OlivawConfig
from olivaw.models import CompletionRequest, CompletionResponse, ProviderStatus


class FakeProvider:
    def __init__(
        self,
        text: str,
        *,
        available: bool = True,
        model: str = "fake-model",
        models: tuple[str, ...] | None = None,
    ):
        self.text = text
        self.available = available
        self.model = model
        self._models = models
        self.request: CompletionRequest | None = None

    def health(self):
        state = "available" if self.available else "unavailable"
        return ProviderStatus(
            name="fake-local",
            kind="local",
            state=state,
            message="fake health",
            model=self.model,
        )

    def models(self):
        return self._models if self._models is not None else (self.model,)

    def complete(self, request: CompletionRequest):
        self.request = request
        return CompletionResponse(
            text=self.text,
            provider="fake-local",
            model=self.model,
        )


def test_health_review_successful_generation_uses_bounded_prompt():
    provider = FakeProvider(
        "Prime Observer observed repeated WAN degradation while LAN stayed stable. "
        "Core Signal assessed the event as mixed evidence with medium confidence. "
        "The primary uncertainty is distinguishing ISP congestion from transient routing instability."
    )

    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=provider,
    )

    assert result.available is True
    assert result.status == "available"
    assert result.text.count(".") == 3
    assert result.provider == "fake-local"
    assert result.model == "fake-model"
    assert result.latency_ms is not None
    assert provider.request is not None
    assert provider.request.system_prompt == HEALTH_REVIEW_SYSTEM_PROMPT
    assert "raw telemetry" not in provider.request.prompt.lower()
    assert "bucket-level" not in provider.request.prompt.lower()
    assert "Do not create new events" in provider.request.prompt
    assert "mixed evidence" in provider.request.prompt


def test_health_review_generation_disabled(monkeypatch):
    monkeypatch.setenv("OLIVAW_HEALTH_REVIEW_ENABLED", "false")
    provider = FakeProvider("This should not be called.")

    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=provider,
    )

    assert result.available is False
    assert result.status == "disabled"
    assert result.text.startswith(f"{UNAVAILABLE_TEXT}:")
    assert "disabled" in result.reason
    assert provider.request is None


@pytest.mark.parametrize("failure", [RuntimeError("boom")])
def test_health_review_completion_failure_returns_unavailable(failure):
    class FailingProvider(FakeProvider):
        def complete(self, request: CompletionRequest):
            raise failure

    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FailingProvider(""),
    )

    assert result.available is False
    assert result.status == "generation_failed"
    assert result.text.startswith(f"{UNAVAILABLE_TEXT}:")
    assert "local model generation failed" in result.reason
    assert "boom" in result.reason


def test_health_review_timeout_returns_precise_reason():
    class TimingOutProvider(FakeProvider):
        def complete(self, request: CompletionRequest):
            raise TimeoutError("slow")

    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=TimingOutProvider(""),
    )

    assert result.available is False
    assert result.status == "timeout"
    assert "timed out" in result.reason
    assert result.text.startswith("Health review unavailable: local model timed out")


def test_health_review_provider_unavailable_returns_precise_reason():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider("", available=False),
    )

    assert result.available is False
    assert result.status == "provider_unavailable"
    assert result.reason == "fake health"
    assert result.provider == "fake-local"


def test_health_review_missing_model_returns_precise_reason():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider("", model="missing-model", models=("other-model",)),
    )

    assert result.available is False
    assert result.status == "model_unavailable"
    assert "missing-model is not installed" in result.reason
    assert "other-model" in result.reason


def test_health_review_empty_model_response_returns_precise_reason():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(" \n "),
    )

    assert result.available is False
    assert result.status == "generation_failed"
    assert "empty response" in result.reason


def test_health_review_empty_source_data_returns_unavailable():
    result = generate_health_review(
        {
            "current_status_label": "Healthy now",
            "status_explanation": "Sources do not report a condition needing attention.",
            "what_we_know": [],
            "worth_knowing": [],
            "network_status": [],
            "dns_activity": [],
            "prime_investigations": [],
            "core_signal_events": [],
            "core_signal_explanation": {},
            "uncertainty_items": [],
        },
        config=OlivawConfig(),
        provider=FakeProvider("Should not run."),
    )

    assert result.available is False
    assert result.status == "no_source_data"
    assert result.text == "Health review unavailable: no source data available."
    assert result.reason == "no source data available."


def test_health_review_hallucination_guardrails_reject_new_decisions():
    provider = FakeProvider(
        "I believe Cox was responsible. You should reboot the router. "
        "The confidence should be high."
    )

    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=provider,
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"
    assert result.text.startswith(f"{UNAVAILABLE_TEXT}:")
    assert result.reason == "generated response was rejected by guardrails."
    assert result.guardrail_rejected is True


def test_health_review_prompt_construction_uses_structured_fields_only():
    digest = build_health_review_digest(_dashboard())
    prompt = build_health_review_prompt(digest)

    assert "current_system_status:" in prompt
    assert "prime_observer:" in prompt
    assert "core_signal:" in prompt
    assert "current_attribution:" in prompt
    assert "attribution_assessment:" in prompt
    assert "evidence_strength:" in prompt
    assert "raw telemetry rows" not in prompt.lower()
    assert "raw investigation samples" not in prompt.lower()
    assert "bucket-level evidence" not in prompt.lower()


def _dashboard() -> dict[str, object]:
    return {
        "current_status_label": "Watch now",
        "status_explanation": "There is a condition worth monitoring.",
        "executive_summary": "Sustained slowdown was detected.",
        "network_status": [
            "Current LAN/WAN state: Mixed evidence",
            "Current network state: mixed local and upstream signals",
        ],
        "what_we_know": [
            "WAN degradation affected both internet and resolver probes.",
            "LAN remained below local degradation thresholds.",
        ],
        "worth_knowing": ["People may have noticed slower network performance."],
        "dns_activity": ["Top resolved domain: example.test"],
        "prime_investigations": [{"title": "WAN samples"}],
        "core_signal_events": [
            {
                "summary": "1 sustained slowdown period was found.",
                "confidence": "medium",
                "confidence_reason": "WAN degraded while LAN remained healthy.",
                "uncertainties": [
                    "Unable to distinguish ISP congestion from transient routing instability."
                ],
            }
        ],
        "core_signal_explanation": {
            "summary": "Sustained slowdown was detected.",
            "why": "Sustained slowdown can affect operator-visible performance.",
            "confidence": "medium",
            "confidence_reason": "WAN degraded while LAN remained healthy.",
            "uncertainties": [
                "Unable to distinguish ISP congestion from transient routing instability."
            ],
            "attribution_assessment": {
                "value": "mixed evidence",
                "confidence": "medium",
                "reason": "WAN degraded while LAN remained healthy.",
            },
            "evidence_strength": {
                "value": "moderate",
                "reason": "Multiple sustained WAN periods were observed.",
            },
            "recommendation_trace": [
                {
                    "stage": "Recommendation",
                    "detail": "Check provider status if symptoms matched.",
                }
            ],
        },
        "attribution_assessment": {
            "value": "mixed evidence",
            "confidence": "medium",
            "reason": "WAN degraded while LAN remained healthy.",
        },
        "evidence_strength": {
            "value": "moderate",
            "reason": "Multiple sustained WAN periods were observed.",
        },
        "uncertainty_items": [
            "Unable to distinguish ISP congestion from transient routing instability."
        ],
    }
