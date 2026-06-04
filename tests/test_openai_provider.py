from __future__ import annotations

from types import SimpleNamespace

import pytest

from olivaw.config import CloudProviderConfig
from olivaw.config import OlivawConfig, PolicyConfig
from olivaw.models import CompletionRequest
from olivaw.models import ProviderStatus
from olivaw.providers.openai import OpenAIProvider
from olivaw.providers.router import RouterProvider


class FakeResponses:
    def __init__(self, output_text="OpenAI response", failure: Exception | None = None):
        self.output_text = output_text
        self.failure = failure
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, responses: FakeResponses):
        self.responses = responses


class UnavailableLocalProvider:
    name = "ollama"

    def health(self):
        return ProviderStatus(
            name="ollama",
            kind="local",
            state="unavailable",
            message="local unavailable",
        )

    def complete(self, request):
        raise AssertionError("local provider should not be called")


def test_openai_health_disabled_by_default():
    status = OpenAIProvider(CloudProviderConfig()).health()

    assert status.state == "disabled"
    assert "disabled" in status.message


def test_openai_health_enabled_without_key_is_unavailable():
    status = OpenAIProvider(CloudProviderConfig(enabled=True)).health()

    assert status.state == "unavailable"
    assert "OPENAI_API_KEY" in status.detail


def test_openai_health_enabled_with_key_is_configured_without_model_call():
    responses = FakeResponses()

    def factory(**kwargs):
        return FakeClient(responses)

    provider = OpenAIProvider(
        CloudProviderConfig(enabled=True, api_key="secret"),
        client_factory=factory,
    )

    status = provider.health()

    assert status.state == "available"
    assert "does not make a model call" in status.detail
    assert responses.calls == []


def test_openai_complete_uses_responses_api_with_instructions_and_input():
    responses = FakeResponses(output_text="grounded answer")
    captured_factory_kwargs = {}

    def factory(**kwargs):
        captured_factory_kwargs.update(kwargs)
        return FakeClient(responses)

    provider = OpenAIProvider(
        CloudProviderConfig(enabled=True, model="gpt-4.1-mini", api_key="secret"),
        timeout=3.0,
        client_factory=factory,
    )

    response = provider.complete(
        CompletionRequest(
            prompt="hello",
            system_prompt="identity grounding",
        )
    )

    assert response.text == "grounded answer"
    assert response.provider == "openai"
    assert response.model == "gpt-4.1-mini"
    assert captured_factory_kwargs == {"api_key": "secret", "timeout": 3.0}
    assert responses.calls == [
        {
            "model": "gpt-4.1-mini",
            "instructions": "identity grounding",
            "input": "hello",
        }
    ]


def test_openai_complete_missing_key_fails_before_client_creation():
    def factory(**kwargs):
        raise AssertionError("client should not be created without an API key")

    provider = OpenAIProvider(
        CloudProviderConfig(enabled=True),
        client_factory=factory,
    )

    with pytest.raises(RuntimeError, match="API key is not configured"):
        provider.complete(CompletionRequest(prompt="hello"))


def test_openai_complete_wraps_sdk_errors_without_exposing_key():
    def factory(**kwargs):
        return FakeClient(FakeResponses(failure=TimeoutError("request timed out")))

    provider = OpenAIProvider(
        CloudProviderConfig(enabled=True, api_key="secret-api-key"),
        client_factory=factory,
    )

    with pytest.raises(RuntimeError) as exc:
        provider.complete(CompletionRequest(prompt="hello"))

    message = str(exc.value)
    assert "OpenAI Responses API request failed" in message
    assert "TimeoutError" in message
    assert "secret-api-key" not in message


def test_openai_complete_requires_output_text():
    def factory(**kwargs):
        return FakeClient(FakeResponses(output_text=""))

    provider = OpenAIProvider(
        CloudProviderConfig(enabled=True, api_key="secret"),
        client_factory=factory,
    )

    with pytest.raises(RuntimeError, match="output_text"):
        provider.complete(CompletionRequest(prompt="hello"))


def test_router_can_use_openai_provider_when_local_unavailable_and_fallback_enabled():
    responses = FakeResponses(output_text="cloud answer")

    def factory(**kwargs):
        return FakeClient(responses)

    router = RouterProvider(
        OlivawConfig(
            cloud=CloudProviderConfig(enabled=True, api_key="secret"),
            policy=PolicyConfig(cloud_fallback="enabled"),
        ),
        local_provider=UnavailableLocalProvider(),
        cloud_provider=OpenAIProvider(
            CloudProviderConfig(enabled=True, api_key="secret"),
            client_factory=factory,
        ),
    )

    response = router.complete(
        CompletionRequest(prompt="hello", system_prompt="identity grounding")
    )

    assert response.provider == "openai"
    assert response.text == "cloud answer"
    assert responses.calls[0]["input"] == "hello"
    assert responses.calls[0]["instructions"] == "identity grounding"
