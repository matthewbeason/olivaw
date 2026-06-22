from __future__ import annotations

from dataclasses import dataclass

import pytest

from olivaw.config import CloudProviderConfig, OlivawConfig, PolicyConfig
from olivaw.models import CompletionRequest, CompletionResponse, ProviderStatus
from olivaw.providers.router import RouterProvider


@dataclass
class StubProvider:
    status: ProviderStatus
    failure: Exception | None = None
    text: str | None = None

    @property
    def name(self) -> str:
        return self.status.name

    def health(self) -> ProviderStatus:
        return self.status

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        if self.failure is not None:
            raise self.failure
        return CompletionResponse(
            text=self.text or f"{self.status.name}: {request.prompt}",
            provider=self.status.name,
            model=self.status.model,
            provider_kind=self.status.kind,
        )


def status(name: str, kind: str, state: str) -> ProviderStatus:
    return ProviderStatus(name=name, kind=kind, state=state, message="test")


def test_router_prefers_local_provider():
    router = RouterProvider(
        OlivawConfig(),
        local_provider=StubProvider(status("local", "local", "available")),
        cloud_provider=StubProvider(status("cloud", "cloud", "available")),
    )

    report = router.health()
    response = router.complete(CompletionRequest(prompt="hello"))

    assert report.selected_provider == "local"
    assert response.provider == "local"


def test_router_requires_explicit_cloud_fallback():
    router = RouterProvider(
        OlivawConfig(),
        local_provider=StubProvider(status("local", "local", "unavailable")),
        cloud_provider=StubProvider(status("cloud", "cloud", "available")),
    )

    report = router.health()

    assert report.selected_provider is None
    with pytest.raises(RuntimeError):
        router.complete(CompletionRequest(prompt="hello"))


def test_router_does_not_use_enabled_cloud_when_fallback_disabled():
    router = RouterProvider(
        OlivawConfig(
            cloud=CloudProviderConfig(enabled=True, api_key="x"),
            policy=PolicyConfig(cloud_fallback="disabled"),
        ),
        local_provider=StubProvider(status("local", "local", "unavailable")),
        cloud_provider=StubProvider(status("cloud", "cloud", "available")),
    )

    report = router.health()

    assert report.selected_provider is None
    assert "Cloud fallback is disabled by policy." in report.notes
    with pytest.raises(RuntimeError):
        router.complete(CompletionRequest(prompt="hello"))


def test_router_uses_cloud_when_enabled_and_allowed():
    config = OlivawConfig(
        cloud=CloudProviderConfig(enabled=True, api_key="x"),
        policy=PolicyConfig(cloud_fallback="enabled"),
    )
    router = RouterProvider(
        config,
        local_provider=StubProvider(status("local", "local", "unavailable")),
        cloud_provider=StubProvider(status("cloud", "cloud", "available")),
    )

    assert router.health().selected_provider == "cloud"
    assert router.complete(CompletionRequest(prompt="hello")).provider == "cloud"


def test_router_manual_only_requires_request_intent_for_cloud():
    config = OlivawConfig(
        cloud=CloudProviderConfig(enabled=True, api_key="x"),
        policy=PolicyConfig(cloud_fallback="manual-only"),
    )
    router = RouterProvider(
        config,
        local_provider=StubProvider(status("local", "local", "unavailable")),
        cloud_provider=StubProvider(status("cloud", "cloud", "available")),
    )

    assert router.health().selected_provider is None
    assert "requires explicit user intent" in " ".join(router.health().notes)
    with pytest.raises(RuntimeError):
        router.complete(CompletionRequest(prompt="hello"))

    response = router.complete(
        CompletionRequest(
            prompt="hello",
            cloud_fallback_allowed=True,
            cloud_fallback_reason="use_best_model",
        )
    )

    assert response.provider == "cloud"
    assert response.provider_kind == "cloud"
    assert response.fallback_reason == "use_best_model"
    assert response.local_model_call_count == 0
    assert response.cloud_model_call_count == 1


def test_router_local_success_does_not_use_cloud_when_request_is_cloud_eligible():
    config = OlivawConfig(
        cloud=CloudProviderConfig(enabled=True, api_key="x"),
        policy=PolicyConfig(cloud_fallback="manual-only"),
    )
    router = RouterProvider(
        config,
        local_provider=StubProvider(status("local", "local", "available")),
        cloud_provider=StubProvider(status("cloud", "cloud", "available")),
    )

    response = router.complete(
        CompletionRequest(
            prompt="hello",
            cloud_fallback_allowed=True,
            cloud_fallback_reason="think_harder_requested",
        )
    )

    assert response.provider == "local"
    assert response.local_model_call_count == 1
    assert response.cloud_model_call_count == 0


def test_router_local_timeout_can_fall_back_to_cloud_when_manual_intent_allows():
    config = OlivawConfig(
        cloud=CloudProviderConfig(enabled=True, api_key="x"),
        policy=PolicyConfig(cloud_fallback="manual-only"),
    )
    router = RouterProvider(
        config,
        local_provider=StubProvider(
            status("local", "local", "available"),
            failure=TimeoutError("request timed out"),
        ),
        cloud_provider=StubProvider(status("cloud", "cloud", "available")),
    )

    response = router.complete(
        CompletionRequest(
            prompt="hello",
            cloud_fallback_allowed=True,
            cloud_fallback_reason="think_harder_requested",
        )
    )

    assert response.provider == "cloud"
    assert response.fallback_reason == "think_harder_requested"
    assert response.local_model_call_count == 1
    assert response.cloud_model_call_count == 1


def test_router_local_unusable_output_can_fall_back_to_cloud_automatically():
    config = OlivawConfig(
        cloud=CloudProviderConfig(enabled=True, api_key="x"),
        policy=PolicyConfig(cloud_fallback="automatic"),
    )
    router = RouterProvider(
        config,
        local_provider=StubProvider(
            status("local", "local", "available"),
            text="I don't know",
        ),
        cloud_provider=StubProvider(status("cloud", "cloud", "available")),
    )

    response = router.complete(CompletionRequest(prompt="hello"))

    assert response.provider == "cloud"
    assert response.fallback_reason == "local_provider_unusable_output"
