from __future__ import annotations

from olivaw.assistant.prompts import build_chat_system_prompt


def test_prompt_builder_includes_current_capabilities():
    prompt = build_chat_system_prompt()

    assert "You are Olivaw." in prompt
    assert "Current implemented capabilities:" in prompt
    assert "deterministic briefing generation from structured input" in prompt
    assert "provider health reporting" in prompt
    assert "read-only configuration display" in prompt
    assert "source inspection" in prompt
    assert "file inspection" in prompt
    assert "source-backed briefing generation" in prompt
    assert "PrimeObserverSource" in prompt


def test_prompt_builder_warns_against_claiming_unimplemented_capabilities():
    prompt = build_chat_system_prompt()

    assert "Not implemented yet:" in prompt
    assert "calendar integration" in prompt
    assert "weather lookup" in prompt
    assert "CoreSignalSource" in prompt
    assert "Source aggregation" in prompt
    assert "Do not claim unavailable capabilities." in prompt
    assert 'Say "not implemented yet" when asked about missing features.' in prompt
    assert "Avoid speculation about your own implementation." in prompt
