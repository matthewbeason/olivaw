from __future__ import annotations

from dataclasses import dataclass, replace

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

        mode = self.config.policy.cloud_fallback_mode

        if local.available:
            selected_provider = local.name
        elif mode == "automatic" and cloud.available:
            selected_provider = cloud.name
            notes.append("Local provider unavailable; automatic cloud fallback is enabled.")
        elif mode == "manual-only" and cloud.available:
            notes.append(
                "Local provider unavailable; cloud fallback requires explicit user intent."
            )
        elif mode == "disabled":
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
            try:
                response = self.local_provider.complete(request)
            except Exception as exc:
                fallback_reason = _fallback_reason(
                    request,
                    default=f"local_provider_failure:{type(exc).__name__}",
                )
                if self._cloud_fallback_allowed(request) and report.cloud.available:
                    return self._complete_with_cloud(
                        request,
                        fallback_reason=fallback_reason,
                        local_model_call_count=1,
                    )
                raise
            if _is_unusable_output(response.text):
                fallback_reason = _fallback_reason(
                    request,
                    default="local_provider_unusable_output",
                )
                if self._cloud_fallback_allowed(request) and report.cloud.available:
                    return self._complete_with_cloud(
                        request,
                        fallback_reason=fallback_reason,
                        local_model_call_count=1,
                    )
            return replace(
                response,
                provider_kind=response.provider_kind or "local",
                local_model_call_count=response.local_model_call_count or 1,
            )
        fallback_reason = _fallback_reason(request, default="local_provider_unavailable")
        if self._cloud_fallback_allowed(request) and report.cloud.available:
            return self._complete_with_cloud(
                request,
                fallback_reason=fallback_reason,
                local_model_call_count=0,
            )
        raise RuntimeError("No provider is available. Run `olivaw health` for details.")

    def _cloud_fallback_allowed(self, request: CompletionRequest) -> bool:
        mode = self.config.policy.cloud_fallback_mode
        if mode == "automatic":
            return True
        return mode == "manual-only" and request.cloud_fallback_allowed

    def _complete_with_cloud(
        self,
        request: CompletionRequest,
        *,
        fallback_reason: str,
        local_model_call_count: int,
    ) -> CompletionResponse:
        if self.cloud_provider is None:
            raise RuntimeError("Cloud provider is not configured.")
        response = self.cloud_provider.complete(request)
        return replace(
            response,
            provider_kind=response.provider_kind or "cloud",
            fallback_reason=fallback_reason,
            local_model_call_count=local_model_call_count,
            cloud_model_call_count=response.cloud_model_call_count or 1,
        )


def _fallback_reason(request: CompletionRequest, *, default: str) -> str:
    return request.cloud_fallback_reason or default


def _is_unusable_output(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    return not normalized or normalized.lower() in {
        "i don't know",
        "i dont know",
        "i do not know",
        "unknown",
        "unavailable",
    }
