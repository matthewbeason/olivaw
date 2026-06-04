from __future__ import annotations

from dataclasses import dataclass

from olivaw.config import OlivawConfig, load_config
from olivaw.models import CompletionRequest
from olivaw.providers.router import RouterProvider


@dataclass
class ChatCapability:
    name: str = "chat"
    description: str = "Minimal provider-routed chat placeholder."

    def run(self, prompt: str, config: OlivawConfig | None = None) -> str:
        resolved_config = config or load_config()
        router = RouterProvider(resolved_config)
        try:
            response = router.complete(CompletionRequest(prompt=prompt))
        except RuntimeError as exc:
            return f"Chat provider unavailable: {exc}"
        return response.text

