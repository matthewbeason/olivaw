from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from olivaw.config import CloudProviderConfig
from olivaw.models import CompletionRequest, CompletionResponse, ProviderStatus


@dataclass
class OpenAIProvider:
    config: CloudProviderConfig
    timeout: float = 10.0

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
            detail="Health check only verifies local configuration in v0.",
            model=self.config.model,
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        if not self.config.enabled:
            raise RuntimeError("OpenAI provider is disabled.")
        if not self.config.api_key:
            raise RuntimeError("OpenAI API key is not configured.")

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {"model": self.config.model, "messages": messages}
        http_request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return CompletionResponse(text=text, provider=self.name, model=self.config.model)

