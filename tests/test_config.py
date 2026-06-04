from __future__ import annotations

from pathlib import Path

from olivaw.config import load_config, public_config


def test_defaults_are_local_first(monkeypatch):
    monkeypatch.delenv("OLIVAW_CONFIG", raising=False)
    monkeypatch.delenv("OLIVAW_CLOUD_ENABLED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLIVAW_OPENAI_API_KEY", raising=False)

    config = load_config(path=Path("does-not-exist.toml"))

    assert config.local.type == "ollama"
    assert config.local.base_url == "http://localhost:11434"
    assert config.cloud.enabled is False
    assert config.policy.cloud_fallback == "disabled"


def test_environment_overrides_sensitive_values(monkeypatch):
    monkeypatch.setenv("OLIVAW_CLOUD_ENABLED", "true")
    monkeypatch.setenv("OLIVAW_OPENAI_API_KEY", "secret")

    config = load_config(path=Path("does-not-exist.toml"))
    public = public_config(config)

    assert config.cloud.enabled is True
    assert config.cloud.api_key == "secret"
    assert public["cloud"]["api_key_present"] is True
    assert "secret" not in str(public)

