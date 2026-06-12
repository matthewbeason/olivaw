from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(RuntimeError):
    """Base error for configuration failures."""


class ConfigPathError(ConfigError):
    """Raised when an explicitly configured path cannot be loaded."""


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
class FileSourceConfig:
    directory: Path = field(default_factory=lambda: default_user_data_path())
    max_bytes: int = 1_048_576


@dataclass(frozen=True)
class PrimeObserverSourceConfig:
    directory: Path = field(default_factory=lambda: default_prime_observer_path())
    enabled: bool = True
    base_url: str | None = None


@dataclass(frozen=True)
class CoreSignalSourceConfig:
    directory: Path = field(default_factory=lambda: default_core_signal_path())
    enabled: bool = True


@dataclass(frozen=True)
class OlivawConfig:
    local: LocalProviderConfig = field(default_factory=LocalProviderConfig)
    cloud: CloudProviderConfig = field(default_factory=CloudProviderConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    files: FileSourceConfig = field(default_factory=FileSourceConfig)
    prime_observer: PrimeObserverSourceConfig = field(
        default_factory=PrimeObserverSourceConfig
    )
    core_signal: CoreSignalSourceConfig = field(default_factory=CoreSignalSourceConfig)
    config_path: Path | None = None
    config_file_exists: bool = False


def load_config(path: str | Path | None = None) -> OlivawConfig:
    config_path = _resolve_config_path(path)
    config_file_exists = bool(config_path and config_path.exists())
    data = _read_toml(config_path) if config_file_exists and config_path else {}

    providers = data.get("providers", {})
    local_data = providers.get("local", {})
    cloud_data = providers.get("cloud", {})
    policy_data = data.get("policy", {})
    secrets_data = data.get("secrets", {})
    sources_data = data.get("sources", {})
    files_data = sources_data.get("files", {})
    prime_observer_data = sources_data.get("prime_observer", {})
    core_signal_data = sources_data.get("core_signal", {})

    local = LocalProviderConfig(
        type=str(local_data.get("type", "ollama")),
        base_url=str(
            _env_value(
                "OLIVAW_LOCAL_BASE_URL",
                local_data.get("base_url", "http://localhost:11434"),
            )
        ),
        model=str(
            _env_value("OLIVAW_LOCAL_MODEL", local_data.get("model", "llama3.1:8b"))
        ),
    )

    cloud_enabled_default = bool(cloud_data.get("enabled", False))
    cloud = CloudProviderConfig(
        type=str(cloud_data.get("type", "openai")),
        enabled=_bool_from_env(
            _env_value("OLIVAW_CLOUD_ENABLED"), cloud_enabled_default
        ),
        model=str(
            _env_value("OLIVAW_CLOUD_MODEL", cloud_data.get("model", "gpt-4.1-mini"))
        ),
        api_key=_first_present(
            _env_value("OLIVAW_OPENAI_API_KEY"),
            _env_value("OPENAI_API_KEY"),
            secrets_data.get("openai_api_key"),
        ),
    )

    policy = PolicyConfig(
        cloud_fallback=str(
            _env_value(
                "OLIVAW_CLOUD_FALLBACK",
                policy_data.get("cloud_fallback", "disabled"),
            )
        )
    )

    files = FileSourceConfig(
        directory=Path(
            _env_value(
                "OLIVAW_FILES_DIR",
                files_data.get("directory", default_user_data_path()),
            )
        ).expanduser(),
        max_bytes=int(
            _env_value(
                "OLIVAW_FILES_MAX_BYTES", files_data.get("max_bytes", 1_048_576)
            )
        ),
    )

    prime_observer = PrimeObserverSourceConfig(
        directory=Path(
            _env_value(
                "OLIVAW_PRIME_OBSERVER_DIR",
                prime_observer_data.get("directory", default_prime_observer_path()),
            )
        ).expanduser(),
        enabled=_bool_from_env(
            _env_value("OLIVAW_PRIME_OBSERVER_ENABLED"),
            bool(prime_observer_data.get("enabled", True)),
        ),
        base_url=_optional_str(
            _env_value(
                "OLIVAW_PRIME_OBSERVER_BASE_URL",
                prime_observer_data.get("base_url"),
            )
        ),
    )

    core_signal = CoreSignalSourceConfig(
        directory=Path(
            _env_value(
                "OLIVAW_CORE_SIGNAL_DIR",
                core_signal_data.get("directory", default_core_signal_path()),
            )
        ).expanduser(),
        enabled=_bool_from_env(
            _env_value("OLIVAW_CORE_SIGNAL_ENABLED"),
            bool(core_signal_data.get("enabled", True)),
        ),
    )

    return OlivawConfig(
        local=local,
        cloud=cloud,
        policy=policy,
        files=files,
        prime_observer=prime_observer,
        core_signal=core_signal,
        config_path=config_path,
        config_file_exists=config_file_exists,
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
        "sources": {
            "files": {
                "directory": str(config.files.directory),
                "max_bytes": config.files.max_bytes,
            },
            "prime_observer": {
                "directory": str(config.prime_observer.directory),
                "enabled": config.prime_observer.enabled,
                "base_url": config.prime_observer.base_url,
            },
            "core_signal": {
                "directory": str(config.core_signal.directory),
                "enabled": config.core_signal.enabled,
            },
        },
        "config_path": str(config.config_path) if config.config_path else None,
        "config_file_exists": config.config_file_exists,
    }


def format_config_report(config: OlivawConfig) -> str:
    path = str(config.config_path) if config.config_path else "not resolved"
    exists = "yes" if config.config_file_exists else "no"
    key_present = "yes" if config.cloud.api_key_present else "no"
    cloud_enabled = "yes" if config.cloud.enabled else "no"
    return "\n".join(
        [
            "Olivaw Configuration",
            "",
            f"Config file path: {path}",
            f"Config file exists: {exists}",
            "",
            "Local Provider:",
            f"- Type: {config.local.type}",
            f"- Base URL: {config.local.base_url}",
            f"- Model: {config.local.model}",
            "",
            "Cloud Provider:",
            f"- Type: {config.cloud.type}",
            f"- Enabled: {cloud_enabled}",
            f"- Model: {config.cloud.model}",
            f"- API key configured: {key_present}",
            "",
            "Policy:",
            f"- Cloud fallback: {config.policy.cloud_fallback}",
            "",
            "Sources:",
            f"- Files directory: {config.files.directory}",
            f"- Files max bytes: {config.files.max_bytes}",
            f"- Prime Observer enabled: {'yes' if config.prime_observer.enabled else 'no'}",
            f"- Prime Observer directory: {config.prime_observer.directory}",
            f"- Prime Observer base URL: {config.prime_observer.base_url or 'not configured'}",
            f"- Core Signal enabled: {'yes' if config.core_signal.enabled else 'no'}",
            f"- Core Signal directory: {config.core_signal.directory}",
        ]
    )


def _resolve_config_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return _require_config_path(Path(path).expanduser(), "explicit config path")

    configured = os.getenv("OLIVAW_CONFIG")
    if configured:
        return _require_config_path(
            Path(configured).expanduser(), "OLIVAW_CONFIG"
        )

    user_config = default_user_config_path()
    if user_config.exists():
        return user_config

    local_config = Path("olivaw.toml")
    if local_config.exists():
        return local_config

    return user_config


def _require_config_path(path: Path, source: str) -> Path:
    if path.exists():
        return path
    raise ConfigPathError(
        f"Olivaw config file from {source} does not exist: {path}"
    )


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data


def default_user_config_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "Olivaw" / "config.toml"


def default_user_data_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "Olivaw" / "data"


def default_prime_observer_path() -> Path:
    return Path.home() / "prime-observer" / "viz"


def default_core_signal_path() -> Path:
    return Path.home() / "core-signal" / "reports"


def _first_present(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return None


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _env_value(name: str, default: object | None = None) -> object | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value
