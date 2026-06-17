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
        "There was a sustained slowdown earlier, but the current state is only a watch item. "
        "The pattern points to WAN degradation while the local network stayed stable. "
        "It is still not clear whether this was provider congestion or transient routing instability. "
        "You probably only need to care if people noticed slower performance at the time."
    )

    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=provider,
    )

    assert result.available is True
    assert result.status == "available"
    _assert_operator_voice(result.text)
    assert result.provider == "fake-local"
    assert result.model == "fake-model"
    assert result.latency_ms is not None
    assert provider.request is not None
    assert provider.request.system_prompt == HEALTH_REVIEW_SYSTEM_PROMPT
    assert "raw telemetry" not in provider.request.prompt.lower()
    assert "bucket-level" not in provider.request.prompt.lower()
    assert "Do not create new events" in provider.request.prompt
    assert "mixed evidence" in provider.request.prompt
    assert "prime_observer:" not in provider.request.prompt
    assert "core_signal:" not in provider.request.prompt


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


def test_health_review_accepts_healthy_current_state_with_historical_incident():
    result = generate_health_review(
        _dashboard(
            current_status_label="Healthy now",
            status_explanation="Sources do not report a condition needing attention.",
            core_signal_explanation={
                "summary": "A sustained slowdown was observed earlier. User impact was possible.",
                "why": "The slowdown appears resolved in the current state.",
                "uncertainties": [
                    "Unable to distinguish provider congestion from transient routing instability."
                ],
            },
        ),
        config=OlivawConfig(),
        provider=FakeProvider(
            "Things look healthy right now. "
            "There was a sustained slowdown earlier, but it appears to have cleared. "
            "It is still not clear whether provider congestion or transient routing instability caused it. "
            "You probably do not need to care unless users noticed symptoms during that window."
        ),
    )

    assert result.available is True
    _assert_operator_voice(result.text)
    assert "healthy right now" in result.text
    assert "appears to have cleared" in result.text


def test_health_review_accepts_active_incident_without_inventing_attribution():
    result = generate_health_review(
        _dashboard(
            current_status_label="Watch now",
            status_explanation="There is an active condition worth monitoring.",
            worth_knowing=["People may currently notice slower network performance."],
        ),
        config=OlivawConfig(),
        provider=FakeProvider(
            "There is an active slowdown worth watching right now. "
            "The recent pattern shows WAN degradation while the local network stayed stable. "
            "The available evidence does not cleanly identify the cause. "
            "This matters because people may currently notice slower network performance."
        ),
    )

    assert result.available is True
    _assert_operator_voice(result.text)
    assert "active slowdown" in result.text
    assert "cause" in result.text
    assert "Cox" not in result.text


def test_health_review_accepts_uncertainty_heavy_incident():
    result = generate_health_review(
        _dashboard(
            uncertainty_items=[
                "Unable to distinguish local Wi-Fi impairment from upstream congestion.",
                "Resolver history is sparse for the affected window.",
                "The current sample does not establish whether symptoms were user-visible.",
            ],
            core_signal_explanation={
                "summary": "A network slowdown was detected.",
                "why": "Operator-visible impact is possible.",
                "confidence": "low",
                "confidence_reason": "Several key observations are sparse.",
                "uncertainties": [
                    "Unable to distinguish local Wi-Fi impairment from upstream congestion.",
                    "Resolver history is sparse for the affected window.",
                ],
                "attribution_assessment": {
                    "value": "mixed evidence",
                    "confidence": "low",
                    "reason": "Local and upstream signals are both incomplete.",
                },
                "evidence_strength": {
                    "value": "limited",
                    "reason": "Several key observations are sparse.",
                },
            },
        ),
        config=OlivawConfig(),
        provider=FakeProvider(
            "A slowdown was detected, but the picture is still incomplete. "
            "Recent signals are mixed between local and upstream explanations. "
            "It is not yet clear which path mattered most, and resolver history is sparse. "
            "This is worth awareness, but the available evidence is limited."
        ),
    )

    assert result.available is True
    _assert_operator_voice(result.text)
    assert "not yet clear" in result.text
    assert "evidence strength" not in result.text.lower()
    assert "low confidence" not in result.text.lower()


def test_health_review_accepts_minimal_data():
    result = generate_health_review(
        {
            "current_status_label": "Unknown",
            "status_explanation": "",
            "what_we_know": ["No source-backed facts or observations are available."],
            "worth_knowing": [],
            "network_status": [],
            "dns_activity": [],
            "prime_investigations": [],
            "core_signal_events": [
                {
                    "summary": "Only minimal interpreted status is available.",
                    "uncertainties": ["The current operating state is not fully described."],
                }
            ],
            "core_signal_explanation": {},
            "uncertainty_items": ["The current operating state is not fully described."],
        },
        config=OlivawConfig(),
        provider=FakeProvider(
            "There is not enough detail to give a strong health read right now. "
            "Only minimal status information is available. "
            "The current operating state is not fully described. "
            "For the operator, this is mainly a visibility gap rather than a confirmed incident."
        ),
    )

    assert result.available is True
    _assert_operator_voice(result.text)
    assert "not enough detail" in result.text


def test_health_review_allows_supplied_recommendation_passthrough():
    result = generate_health_review(
        _dashboard(
            core_signal_explanation={
                "summary": "The current state is healthy after an earlier slowdown.",
                "why": "The issue appears resolved.",
                "uncertainties": ["User-visible impact is unknown."],
                "recommendation_trace": [
                    {
                        "stage": "Recommendation",
                        "detail": "No action is recommended unless symptoms matched the slowdown window.",
                    }
                ],
            },
        ),
        config=OlivawConfig(),
        provider=FakeProvider(
            "Things look healthy right now. "
            "There was an earlier slowdown, but it appears resolved. "
            "User-visible impact is still unknown. "
            "You probably do not need to take action unless symptoms matched that slowdown window."
        ),
    )

    assert result.available is True
    _assert_operator_voice(result.text)
    assert "do not need to take action unless symptoms matched" in result.text


def test_health_review_rejects_invented_recommendation_without_source_support():
    result = generate_health_review(
        _dashboard(core_signal_explanation={"summary": "A slowdown was detected."}),
        config=OlivawConfig(),
        provider=FakeProvider(
            "There is a slowdown right now. "
            "You should restart the router. "
            "The cause is probably local Wi-Fi."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_advice_when_only_placeholder_recommendation_exists():
    result = generate_health_review(
        _dashboard(
            core_signal_explanation={
                "summary": "A slowdown was detected.",
                "recommendation_trace": [
                    {
                        "stage": "Recommendation",
                        "detail": "No specific recommendation is available.",
                    }
                ],
            }
        ),
        config=OlivawConfig(),
        provider=FakeProvider(
            "Things look healthy right now. "
            "An earlier slowdown appears to have cleared. "
            "An investigation should be performed for matching symptoms."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_invented_attribution():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "There is a slowdown right now. "
            "Cox is responsible for the issue. "
            "You probably need to wait for the provider."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_source_bookkeeping_language_and_counts():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "Prime Observer reports one investigation artifact. "
            "Core Signal reports medium confidence. "
            "Investigation counts show one event."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_generic_source_reporting_language():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "Things look healthy right now. "
            "There is no condition needing attention reported by sources. "
            "Earlier slowdown periods appear to have cleared."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_report_preamble():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "I'd like to bring you up to speed on the current health review. "
            "Things look healthy right now. "
            "There was an earlier slowdown."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_overstated_operator_urgency():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "You need to care about this now because the current state is healthy. "
            "This matters because it is essential to investigate if symptoms matched."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_investigation_and_impact_overstatement():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "Things look healthy right now. "
            "There were two slowdown periods earlier, but they appear to have cleared. "
            "You probably should investigate if users noticed symptoms because sustained slowdowns can have a significant impact on user experience."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_awkward_operator_relevance_phrasing():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "Healthy now. "
            "The operator needs to care because slowdown periods were detected. "
            "This matter matters because it could have resulted in user impact."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_slowdown_contradiction_and_invented_impact():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "Healthy now. "
            "No sustained slowdown has been reported recently. "
            "This matters because sustained slowdown could lead to decreased productivity."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_field_label_narration():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "The current state is labeled as Healthy now. "
            "Sources do not report a condition needing attention."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_rejects_prompt_field_names():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "Current_state is Healthy now. "
            "Observed facts show a prior slowdown."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_enforces_four_sentence_maximum():
    result = generate_health_review(
        _dashboard(),
        config=OlivawConfig(),
        provider=FakeProvider(
            "Things look healthy right now. "
            "There was an earlier slowdown. "
            "It appears resolved. "
            "The cause remains uncertain. "
            "You probably only need to care if users noticed symptoms."
        ),
    )

    assert result.available is False
    assert result.status == "guardrail_rejected"


def test_health_review_prompt_construction_uses_structured_fields_only():
    digest = build_health_review_digest(_dashboard())
    prompt = build_health_review_prompt(digest)

    assert "current_state:" in prompt
    assert "observed_facts:" in prompt
    assert "interpreted_findings:" in prompt
    assert "current_attribution:" in prompt
    assert "attribution_assessment:" in prompt
    assert "evidence_strength:" in prompt
    assert "investigation_counts:" not in prompt
    assert "dns_summary:" not in prompt
    assert "Top resolved domain" not in prompt
    assert "raw telemetry rows" not in prompt.lower()
    assert "raw investigation samples" not in prompt.lower()
    assert "bucket-level evidence" not in prompt.lower()


def _dashboard(**overrides: object) -> dict[str, object]:
    dashboard: dict[str, object] = {
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
    dashboard.update(overrides)
    return dashboard


def _assert_operator_voice(text: str) -> None:
    lowered = text.lower()
    assert 1 <= text.count(".") <= 4
    forbidden = (
        "prime observer",
        "core signal",
        "investigation count",
        "evidence strength",
        "medium confidence",
        "low confidence",
        "high confidence",
        "attribution assessment",
    )
    for phrase in forbidden:
        assert phrase not in lowered
