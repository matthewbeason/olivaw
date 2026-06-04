from __future__ import annotations

from typing import Protocol

from olivaw.models import CompletionRequest, CompletionResponse, ProviderStatus


class Provider(Protocol):
    name: str

    def health(self) -> ProviderStatus:
        """Return provider availability without raising for expected failures."""

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a response or raise a provider-specific runtime error."""

