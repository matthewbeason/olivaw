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
    assert "CoreSignalSource" in prompt


def test_prompt_builder_warns_against_claiming_unimplemented_capabilities():
    prompt = build_chat_system_prompt()

    assert "Not implemented yet:" in prompt
    assert "calendar integration" in prompt
    assert "weather lookup" in prompt
    assert "Source aggregation" in prompt
    assert "Do not claim unavailable capabilities." in prompt
    assert (
        "Default to 2 to 4 short sentences unless the operator asks for detail."
        in prompt
    )
    assert 'Say "not implemented yet" when asked about missing features.' in prompt
    assert (
        "For general knowledge, answer from model knowledge without naming registered sources."
        in prompt
    )
    assert (
        "For general knowledge, do not claim database, record, source, or knowledge-base access."
        in prompt
    )
    assert (
        "Do not claim Prime Observer, Core Signal, Weather, or any registered source supplied a generic answer unless Olivaw actually used that source."
        in prompt
    )
    assert "For current operational state, do not guess if no registered source supplied it." in prompt
    assert "Avoid speculation about your own implementation." in prompt
