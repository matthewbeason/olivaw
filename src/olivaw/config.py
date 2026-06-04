from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _bool_from_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


@dataclass(frozen=True)
class LocalProviderConfig:
    type: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"


@dataclass(frozen=True)
class CloudProviderConfig:
    type: str = "openai"
    enabled: bool = False
    model: str = "gpt-4.1-mini"
    api_key: str | None = field(default=None, repr=False)

    @property
    def api_key_present(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class PolicyConfig:
    cloud_fallback: str = "disabled"

    @property
    def cloud_fallback_enabled(self) -> bool:
        return self.cloud_fallback.strip().lower() in {"enabled", "true", "on"}


@dataclass(frozen=True)
class OlivawConfig:
    local: LocalProviderConfig = field(default_factory=LocalProviderConfig)
    cloud: CloudProviderConfig = field(default_factory=CloudProviderConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    config_path: Path | None = None


def load_config(path: str | Path | None = None) -> OlivawConfig:
    config_path = _resolve_config_path(path)
    data = _read_toml(config_path) if config_path else {}

    providers = data.get("providers", {})
    local_data = providers.get("local", {})
    cloud_data = providers.get("cloud", {})
    policy_data = data.get("policy", {})

    local = LocalProviderConfig(
        type=str(local_data.get("type", "ollama")),
        base_url=str(
            os.getenv(
                "OLIVAW_LOCAL_BASE_URL",
                local_data.get("base_url", "http://localhost:11434"),
            )
        ),
        model=str(
            os.getenv("OLIVAW_LOCAL_MODEL", local_data.get("model", "llama3.1:8b"))
        ),
    )

    cloud_enabled_default = bool(cloud_data.get("enabled", False))
    cloud = CloudProviderConfig(
        type=str(cloud_data.get("type", "openai")),
        enabled=_bool_from_env(os.getenv("OLIVAW_CLOUD_ENABLED"), cloud_enabled_default),
        model=str(
            os.getenv("OLIVAW_CLOUD_MODEL", cloud_data.get("model", "gpt-4.1-mini"))
        ),
        api_key=os.getenv("OLIVAW_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )

    policy = PolicyConfig(
        cloud_fallback=str(
            os.getenv(
                "OLIVAW_CLOUD_FALLBACK",
                policy_data.get("cloud_fallback", "disabled"),
            )
        )
    )

    return OlivawConfig(
        local=local,
        cloud=cloud,
        policy=policy,
        config_path=config_path,
    )


def public_config(config: OlivawConfig) -> dict[str, object]:
    return {
        "local": {
            "type": config.local.type,
            "base_url": config.local.base_url,
            "model": config.local.model,
        },
        "cloud": {
            "type": config.cloud.type,
            "enabled": config.cloud.enabled,
            "model": config.cloud.model,
            "api_key_present": config.cloud.api_key_present,
        },
        "policy": {"cloud_fallback": config.policy.cloud_fallback},
        "config_path": str(config.config_path) if config.config_path else None,
    }


def _resolve_config_path(path: str | Path | None) -> Path | None:
    configured = path or os.getenv("OLIVAW_CONFIG")
    if configured:
        candidate = Path(configured).expanduser()
        return candidate if candidate.exists() else None

    default = Path("olivaw.toml")
    return default if default.exists() else None


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data

