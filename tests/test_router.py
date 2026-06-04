from __future__ import annotations

from dataclasses import dataclass

import pytest

from olivaw.config import CloudProviderConfig, OlivawConfig, PolicyConfig
from olivaw.models import CompletionRequest, CompletionResponse, ProviderStatus
from olivaw.providers.router import RouterProvider


@dataclass
class StubProvider:
    status: ProviderStatus

    @property
    def name(self) -> str:
        return self.status.name

    def health(self) -> ProviderStatus:
        return self.status

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            text=f"{self.status.name}: {request.prompt}",
            provider=self.status.name,
            model=self.status.model,
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

