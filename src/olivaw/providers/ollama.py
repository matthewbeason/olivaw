from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from olivaw.config import LocalProviderConfig
from olivaw.models import CompletionRequest, CompletionResponse, ProviderStatus


@dataclass
class OllamaProvider:
    config: LocalProviderConfig
    timeout: float = 1.0
    complete_timeout: float = 30.0

    name: str = "ollama"

    def models(self) -> tuple[str, ...]:
        url = f"{self.config.base_url.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = data.get("models", [])
        if not isinstance(models, list):
            return ()
        names: list[str] = []
        for model in models:
            if not isinstance(model, dict):
                continue
            for key in ("model", "name"):
                value = str(model.get(key) or "").strip()
                if value and value not in names:
                    names.append(value)
        return tuple(names)

    def health(self) -> ProviderStatus:
        url = f"{self.config.base_url.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                if 200 <= response.status < 300:
                    return ProviderStatus(
                        name=self.name,
                        kind="local",
                        state="available",
                        message="Ollama is reachable.",
                        detail=f"Connected to {self.config.base_url}",
                        model=self.config.model,
                    )
        except (OSError, urllib.error.URLError) as exc:
            return ProviderStatus(
                name=self.name,
                kind="local",
                state="unavailable",
                message="Unable to connect to Ollama.",
                detail=(
                    f"Expected endpoint: {self.config.base_url}. "
                    "Install Ollama and run: ollama serve. "
                    f"Reason: {exc}"
                ),
                model=self.config.model,
            )

        return ProviderStatus(
            name=self.name,
            kind="local",
            state="unknown",
            message="Ollama returned an unexpected health response.",
            detail=f"Checked {url}",
            model=self.config.model,
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self.config.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": request.prompt,
            "system": request.system_prompt,
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(http_request, timeout=self.complete_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        request_duration_ms = int((time.perf_counter() - started) * 1000)
        return CompletionResponse(
            text=str(data.get("response", "")),
            provider=self.name,
            model=self.config.model,
            request_duration_ms=request_duration_ms,
            ollama_total_duration_ms=_duration_ms(data.get("total_duration")),
            ollama_load_duration_ms=_duration_ms(data.get("load_duration")),
            ollama_prompt_eval_duration_ms=_duration_ms(
                data.get("prompt_eval_duration")
            ),
            ollama_eval_duration_ms=_duration_ms(data.get("eval_duration")),
            prompt_eval_count=_optional_int(data.get("prompt_eval_count")),
            eval_count=_optional_int(data.get("eval_count")),
        )


def _duration_ms(value: object) -> int | None:
    parsed = _optional_int(value)
    if parsed is None:
        return None
    return int(parsed / 1_000_000)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
