from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from olivaw.config import CloudProviderConfig
from olivaw.models import CompletionRequest, CompletionResponse, ProviderStatus

ClientFactory = Callable[..., Any]


@dataclass
class OpenAIProvider:
    config: CloudProviderConfig
    timeout: float = 10.0
    client_factory: ClientFactory | None = None

    name: str = "openai"

    def health(self) -> ProviderStatus:
        if not self.config.enabled:
            return ProviderStatus(
                name=self.name,
                kind="cloud",
                state="disabled",
                message="OpenAI cloud provider is disabled.",
                detail="Set OLIVAW_CLOUD_ENABLED=true to opt in.",
                model=self.config.model,
            )
        if not self.config.api_key_present:
            return ProviderStatus(
                name=self.name,
                kind="cloud",
                state="unavailable",
                message="OpenAI cloud provider is enabled but no API key is configured.",
                detail="Set OPENAI_API_KEY or OLIVAW_OPENAI_API_KEY.",
                model=self.config.model,
            )
        return ProviderStatus(
            name=self.name,
            kind="cloud",
            state="available",
            message="OpenAI cloud provider is configured.",
            detail=(
                "An OpenAI API key is present and cloud provider is explicitly enabled. "
                "Health check does not make a model call."
            ),
            model=self.config.model,
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        if not self.config.enabled:
            raise RuntimeError("OpenAI provider is disabled.")
        if not self.config.api_key:
            raise RuntimeError("OpenAI API key is not configured.")

        client = self._create_client()
        try:
            response = client.responses.create(
                model=self.config.model,
                instructions=request.system_prompt,
                input=request.prompt,
            )
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI Responses API request failed: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            text = getattr(response, "output_text", None)
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI response text extraction failed: {type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("OpenAI response did not include output_text.")

        return CompletionResponse(
            text=text,
            provider=self.name,
            model=self.config.model,
            provider_kind="cloud",
            cloud_model_call_count=1,
        )

    def _create_client(self):
        factory = self.client_factory or _default_client_factory
        return factory(api_key=self.config.api_key, timeout=self.timeout)


def _default_client_factory(**kwargs):
    from openai import OpenAI

    return OpenAI(**kwargs)
