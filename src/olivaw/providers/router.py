from __future__ import annotations

from dataclasses import dataclass

from olivaw.config import OlivawConfig
from olivaw.models import CompletionRequest, CompletionResponse, HealthReport
from olivaw.providers.base import Provider
from olivaw.providers.ollama import OllamaProvider
from olivaw.providers.openai import OpenAIProvider


@dataclass
class RouterProvider:
    config: OlivawConfig
    local_provider: Provider | None = None
    cloud_provider: Provider | None = None

    name: str = "router"

    def __post_init__(self) -> None:
        if self.local_provider is None:
            self.local_provider = OllamaProvider(self.config.local)
        if self.cloud_provider is None:
            self.cloud_provider = OpenAIProvider(self.config.cloud)

    def health(self) -> HealthReport:
        local = self.local_provider.health()
        cloud = self.cloud_provider.health()
        selected_provider = None
        notes: list[str] = []

        if local.available:
            selected_provider = local.name
        elif self.config.policy.cloud_fallback_enabled and cloud.available:
            selected_provider = cloud.name
            notes.append("Local provider unavailable; explicit cloud fallback is enabled.")
        elif not self.config.policy.cloud_fallback_enabled:
            notes.append("Cloud fallback is disabled by policy.")
        else:
            notes.append("No configured provider is currently available.")

        return HealthReport(
            local=local,
            cloud=cloud,
            selected_provider=selected_provider,
            cloud_fallback=self.config.policy.cloud_fallback,
            notes=notes,
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        report = self.health()
        if report.local.available:
            return self.local_provider.complete(request)
        if (
            self.config.policy.cloud_fallback_enabled
            and report.cloud.available
            and self.cloud_provider is not None
        ):
            return self.cloud_provider.complete(request)
        raise RuntimeError("No provider is available. Run `olivaw health` for details.")

