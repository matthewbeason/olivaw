from __future__ import annotations

from dataclasses import dataclass

from olivaw.config import OlivawConfig, load_config
from olivaw.models import CompletionRequest
from olivaw.providers.router import RouterProvider

HEALTH_HINT = (
    "Run `olivaw health` to inspect provider status. "
    "If using Ollama, verify it is installed and running at the configured endpoint."
)


@dataclass
class ChatCapability:
    name: str = "chat"
    description: str = "Minimal provider-routed chat placeholder."

    def run(self, prompt: str, config: OlivawConfig | None = None) -> str:
        resolved_config = config or load_config()
        router = RouterProvider(resolved_config)
        try:
            response = router.complete(CompletionRequest(prompt=prompt))
        except Exception as exc:
            return _format_chat_failure(exc)
        return response.text


def _format_chat_failure(exc: Exception) -> str:
    detail = str(exc).strip() or "provider failed without additional detail"
    return (
        "Chat provider unavailable: "
        f"{type(exc).__name__}: {detail}. {HEALTH_HINT}"
    )
