from __future__ import annotations

import pytest

from olivaw.config import ConfigPathError, load_config, public_config


def test_defaults_are_local_first_when_implicit_config_is_missing(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("OLIVAW_CONFIG", raising=False)
    monkeypatch.delenv("OLIVAW_CLOUD_ENABLED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLIVAW_OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.local.type == "ollama"
    assert config.local.base_url == "http://localhost:11434"
    assert config.cloud.enabled is False
    assert config.policy.cloud_fallback == "disabled"


def test_environment_overrides_sensitive_values(monkeypatch):
    monkeypatch.setenv("OLIVAW_CLOUD_ENABLED", "true")
    monkeypatch.setenv("OLIVAW_OPENAI_API_KEY", "secret")

    config = load_config()
    public = public_config(config)

    assert config.cloud.enabled is True
    assert config.cloud.api_key == "secret"
    assert public["cloud"]["api_key_present"] is True
    assert "secret" not in str(public)


def test_openai_api_key_environment_is_supported(monkeypatch):
    monkeypatch.delenv("OLIVAW_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    config = load_config()

    assert config.cloud.api_key == "openai-secret"


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
