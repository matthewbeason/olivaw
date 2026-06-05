from __future__ import annotations

import pytest

from olivaw.config import (
    ConfigPathError,
    default_user_config_path,
    format_config_report,
    load_config,
    public_config,
)


def clear_config_env(monkeypatch):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_LOCAL_BASE_URL",
        "OLIVAW_LOCAL_MODEL",
        "OLIVAW_CLOUD_ENABLED",
        "OLIVAW_CLOUD_MODEL",
        "OLIVAW_CLOUD_FALLBACK",
        "OLIVAW_FILES_DIR",
        "OLIVAW_FILES_MAX_BYTES",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_PRIME_OBSERVER_ENABLED",
        "OPENAI_API_KEY",
        "OLIVAW_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_local_first_when_implicit_config_is_missing(
    monkeypatch, tmp_path
):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.local.type == "ollama"
    assert config.local.base_url == "http://localhost:11434"
    assert config.cloud.enabled is False
    assert config.policy.cloud_fallback == "disabled"
    assert config.config_path == default_user_config_path()
    assert config.config_file_exists is False


def test_user_config_loads_persistent_settings_and_secrets(monkeypatch, tmp_path):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = default_user_config_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[providers.local]
base_url = "http://localhost:11435"
model = "mistral"

[providers.cloud]
enabled = true
model = "gpt-4.1"

[policy]
cloud_fallback = "enabled"

[sources.prime_observer]
directory = "~/prime-observer/viz"
enabled = true

[secrets]
openai_api_key = "config-secret"
""",
        encoding="utf-8",
    )

    config = load_config()

    assert config.config_path == config_path
    assert config.config_file_exists is True
    assert config.local.base_url == "http://localhost:11435"
    assert config.local.model == "mistral"
    assert config.cloud.enabled is True
    assert config.cloud.model == "gpt-4.1"
    assert config.cloud.api_key == "config-secret"
    assert config.policy.cloud_fallback == "enabled"
    assert config.prime_observer.directory == tmp_path / "prime-observer" / "viz"
    assert config.prime_observer.enabled is True


def test_environment_overrides_user_config_values_and_secrets(monkeypatch, tmp_path):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = default_user_config_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[providers.local]
base_url = "http://localhost:11435"
model = "mistral"

[providers.cloud]
enabled = false
model = "gpt-4.1"

[policy]
cloud_fallback = "disabled"

[secrets]
openai_api_key = "config-secret"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLIVAW_LOCAL_MODEL", "llama3.2")
    monkeypatch.setenv("OLIVAW_CLOUD_ENABLED", "true")
    monkeypatch.setenv("OLIVAW_CLOUD_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OLIVAW_CLOUD_FALLBACK", "enabled")
    monkeypatch.setenv("OLIVAW_OPENAI_API_KEY", "env-secret")

    config = load_config()
    public = public_config(config)

    assert config.local.base_url == "http://localhost:11435"
    assert config.local.model == "llama3.2"
    assert config.cloud.enabled is True
    assert config.cloud.model == "gpt-4.1-mini"
    assert config.cloud.api_key == "env-secret"
    assert config.policy.cloud_fallback == "enabled"
    assert public["cloud"]["api_key_present"] is True
    assert "env-secret" not in str(public)
    assert "config-secret" not in str(public)


def test_blank_environment_values_do_not_mask_user_config(monkeypatch, tmp_path):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = default_user_config_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[providers.local]
base_url = "http://localhost:11435"
model = "mistral"

[providers.cloud]
enabled = true
model = "gpt-4.1"

[policy]
cloud_fallback = "enabled"

[sources.files]
directory = "~/Library/Application Support/Olivaw/data"
max_bytes = 2048

[sources.prime_observer]
directory = "~/prime-observer/custom"
enabled = true

[secrets]
openai_api_key = "config-secret"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLIVAW_LOCAL_BASE_URL", "")
    monkeypatch.setenv("OLIVAW_LOCAL_MODEL", "")
    monkeypatch.setenv("OLIVAW_CLOUD_ENABLED", "")
    monkeypatch.setenv("OLIVAW_CLOUD_MODEL", "")
    monkeypatch.setenv("OLIVAW_CLOUD_FALLBACK", "")
    monkeypatch.setenv("OLIVAW_FILES_DIR", "")
    monkeypatch.setenv("OLIVAW_FILES_MAX_BYTES", "")
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", "")
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_ENABLED", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OLIVAW_OPENAI_API_KEY", "")

    config = load_config()

    assert config.local.base_url == "http://localhost:11435"
    assert config.local.model == "mistral"
    assert config.cloud.enabled is True
    assert config.cloud.model == "gpt-4.1"
    assert config.policy.cloud_fallback == "enabled"
    assert config.files.directory == (
        tmp_path / "Library" / "Application Support" / "Olivaw" / "data"
    )
    assert config.files.max_bytes == 2048
    assert config.prime_observer.directory == tmp_path / "prime-observer" / "custom"
    assert config.prime_observer.enabled is True
    assert config.cloud.api_key == "config-secret"


def test_explicit_false_environment_overrides_user_config(monkeypatch, tmp_path):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = default_user_config_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[providers.cloud]
enabled = true

[policy]
cloud_fallback = "enabled"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLIVAW_CLOUD_ENABLED", "false")
    monkeypatch.setenv("OLIVAW_CLOUD_FALLBACK", "disabled")

    config = load_config()

    assert config.cloud.enabled is False
    assert config.policy.cloud_fallback == "disabled"


def test_prime_observer_environment_overrides_user_config(monkeypatch, tmp_path):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = default_user_config_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[sources.prime_observer]
directory = "~/prime-observer/viz"
enabled = true
""",
        encoding="utf-8",
    )
    override_dir = tmp_path / "custom-prime-observer"
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(override_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_ENABLED", "false")

    config = load_config()

    assert config.prime_observer.directory == override_dir
    assert config.prime_observer.enabled is False


def test_openai_api_key_environment_is_supported(monkeypatch, tmp_path):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OLIVAW_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    config = load_config()

    assert config.cloud.api_key == "openai-secret"


def test_public_config_and_report_redact_key(monkeypatch, tmp_path):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = default_user_config_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[providers.cloud]
enabled = true

[secrets]
openai_api_key = "config-secret"
""",
        encoding="utf-8",
    )

    config = load_config()
    public = public_config(config)
    report = format_config_report(config)

    assert public["config_path"] == str(config_path)
    assert public["config_file_exists"] is True
    assert public["cloud"]["api_key_present"] is True
    assert "config-secret" not in str(public)
    assert "API key configured: yes" in report
    assert "config-secret" not in report


def test_explicit_missing_config_path_raises_clear_error(tmp_path):
    missing = tmp_path / "missing.toml"

    with pytest.raises(ConfigPathError, match="explicit config path") as exc:
        load_config(path=missing)

    assert str(missing) in str(exc.value)


def test_missing_olivaw_config_env_raises_clear_error(monkeypatch, tmp_path):
    missing = tmp_path / "missing.toml"
    monkeypatch.setenv("OLIVAW_CONFIG", str(missing))

    with pytest.raises(ConfigPathError, match="OLIVAW_CONFIG") as exc:
        load_config()

    assert str(missing) in str(exc.value)
