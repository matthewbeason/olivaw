from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from olivaw.models import HealthReport, ProviderStatus
from olivaw.web import (
    _briefing_dashboard,
    _dashboard_status,
    _human_generated_time,
    _normalize_briefing_dashboard,
    app,
)


client = TestClient(app)
WEATHER_PROMPT = "Hi could you tell me what the weather is in Phoenix az"


@pytest.fixture(autouse=True)
def mock_health_checks(monkeypatch):
    def fake_health(config=None):
        return HealthReport(
            local=ProviderStatus(
                name="ollama",
                kind="local",
                state="unavailable",
                message="Mocked local provider status.",
                detail="Mocked test health check; no local network probe.",
                model="llama3.1:8b",
            ),
            cloud=ProviderStatus(
                name="openai",
                kind="cloud",
                state="disabled",
                message="Mocked cloud provider status.",
                model="gpt-4.1-mini",
            ),
            selected_provider=None,
            cloud_fallback="disabled",
            notes=["Mocked web health check."],
        )

    monkeypatch.setattr("olivaw.web.run_health_checks", fake_health)


def test_home_route_renders():
    response = client.get("/")

    assert response.status_code == 200
    assert "Assistant Home" in response.text
    assert "Example Briefing" in response.text
    assert "Briefing renders without repo fixtures" in response.text


def test_home_route_renders_from_non_repo_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "Assistant Home" in response.text
    assert "Briefing renders without repo fixtures" in response.text


def test_health_route_renders():
    response = client.get("/health")

    assert response.status_code == 200
    assert "Local Provider" in response.text
    assert "Cloud Provider" in response.text
    assert "Mocked local provider status." in response.text


def test_capabilities_route_renders_identity_grounding():
    response = client.get("/capabilities")

    assert response.status_code == 200
    assert "Implemented" in response.text
    assert "Not Implemented Yet" in response.text
    assert "deterministic briefing generation from structured input" in response.text
    assert "calendar integration" in response.text
    assert "weather lookup" in response.text


def test_sources_route_renders_registered_sources():
    # Route should render both default sources without requiring a real data dir.
    response = client.get("/sources")

    assert response.status_code == 200
    assert "Sources" in response.text
    assert "Manual example source" in response.text
    assert "Local files" in response.text
    assert "Prime Observer" in response.text
    assert "Core Signal" in response.text
    assert "manual" in response.text
    assert "files" in response.text
    assert "prime_observer" in response.text
    assert "core_signal" in response.text
    assert "Example item" in response.text
    assert "Demonstrates source plumbing." in response.text


def test_sources_route_renders_investigations_events_and_metadata(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "id": "inv-20260608",
    "title": "June 8 WAN samples",
    "status": "available",
    "path": "viz/investigation.json"
  }
]
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Attention",
  "summary": "The network had 1 sustained slowdown period.",
  "events": [
    {
      "summary": "1 sustained slowdown period was found.",
      "confidence": "0.82",
      "confidence_reason": "Matched sustained slowdown threshold.",
      "supporting_facts": [
        {"summary": "WAN p95 exceeded threshold.", "source": "prime_observer"}
      ],
      "recommendation_trace": {
        "recommendation": "Check provider status if symptoms matched.",
        "interpretation": "Core Signal classified the event."
      },
      "interpretation_source": "core_signal",
      "attribution_source": "prime_observer_incident"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/sources")

    assert response.status_code == 200
    assert "Source path:" in response.text
    assert "Investigation index" in response.text
    assert "Investigation index loaded: 1 investigations." in response.text
    assert "Investigation index path" in response.text
    assert "Catalog entries" in response.text
    assert "Investigation entries: 1" in response.text
    assert "June 8 WAN samples" in response.text
    assert "Interpreted events" in response.text
    assert "Core Signal events loaded: 1." in response.text
    assert "Event objects found" in response.text
    assert "Interpreted events rendered" in response.text
    assert "1 sustained slowdown period was found." in response.text
    assert "Confidence" in response.text
    assert "0.82" in response.text
    assert "Why" in response.text
    assert "Matched sustained slowdown threshold." in response.text
    assert "Supporting facts" in response.text


def test_sources_route_renders_precise_empty_diagnostics(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    (prime_dir / "investigation_index.json").write_text("[]\n", encoding="utf-8")
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Healthy",
  "summary": "Stable."
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))

    response = client.get("/sources")

    assert response.status_code == 200
    assert "Investigation index loaded but contains no catalog entries." in response.text
    assert "Core Signal reports loaded, but no event objects were emitted." in (
        response.text
    )
    assert "Event objects found" in response.text


def test_briefing_route_replaces_ambiguous_empty_states(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "No Core Signal report file found at configured path." in response.text
    assert "Investigation index file was not found at configured path." in response.text
    assert "No interpreted Core Signal events are available." not in response.text
    assert "No Prime Observer investigation index is available." not in response.text


def test_briefing_route_renders_source_backed_briefing(monkeypatch, tmp_path):
    for name in ("OLIVAW_CONFIG", "OLIVAW_FILES_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    data_path = tmp_path / "Library" / "Application Support" / "Olivaw" / "data"
    (data_path / "status").mkdir(parents=True)
    (data_path / "status" / "system.txt").write_text(
        "System status\nAll local.\n",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert "Today&apos;s Assessment" in response.text
    assert "Intelligence briefing" in response.text
    assert "Generated" in response.text
    assert re.search(r"Generated (just now|\d+ minutes? ago|today at)", response.text)
    assert "Refresh briefing" in response.text
    assert "Current assessment" in response.text
    assert "Healthy" in response.text
    assert "Sources do not report a condition needing attention." in response.text
    assert "What Matters" in response.text
    assert "Priority signals" in response.text
    assert "Recommended Action" in response.text
    assert "Why We Believe This" in response.text
    assert "Source attribution and current facts" in response.text
    assert "Core Signal events" in response.text
    assert "Category:" not in response.text
    assert "Show raw briefing" in response.text
    assert "manual, files" in response.text
    assert "Example item from manual source" in response.text
    assert "File found: status/system.txt" in response.text


def test_briefing_route_reflects_changed_source_data_between_requests(
    monkeypatch,
    tmp_path,
):
    for name in ("OLIVAW_CONFIG", "OLIVAW_FILES_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    data_path = tmp_path / "Library" / "Application Support" / "Olivaw" / "data"
    data_path.mkdir(parents=True)
    source_file = data_path / "status.txt"
    source_file.write_text("Version one\n", encoding="utf-8")

    first = client.get("/briefing")

    source_file.write_text("Version two\n", encoding="utf-8")
    second = client.get("/briefing")

    assert first.status_code == 200
    assert second.status_code == 200
    assert "Version one" in first.text
    assert "Version two" in second.text
    assert "Version one" not in second.text
    assert "This briefing is source-backed using: manual, files." in second.text


def test_briefing_dashboard_promotes_dns_domains_and_collapses_metrics():
    dashboard = _briefing_dashboard(
        """# Source Briefing

## Prime Observer
- DNS summary: available from Prime Observer.
  - Total queries: 1000
  - Blocked queries: 20
  - Encrypted queries: 800
  - Block rate: 2.0%
  - Raw block rate: 0.02
  - Top queried domain: www.example.test (count 300, share 0.3)
  - Top blocked domain: ads.example.test (count 10, share 0.5)
  - Top resolved domain: api.example.test (count 250, share 0.25)

## Core Signal
- Core Signal Morning Brief - 2026-06-05 (2026-06-05) [Healthy]: Stable.
  - Recommended action: No action.

## Attribution
This briefing is source-backed using: prime_observer, core_signal.
""",
        "2026-06-06T00:14:59+00:00",
        ("prime_observer", "core_signal"),
    )

    assert dashboard["dns_activity"] == [
        "Top blocked domain: ads.example.test",
        "Top resolved domain: api.example.test",
        "Top queried domain: www.example.test",
    ]
    assert "Total queries: 1000" in dashboard["dns_details"]
    assert "Block rate: 2.0%" in dashboard["dns_details"]
    assert "Top queried domain: www.example.test (count 300, share 0.3)" in (
        dashboard["dns_details"]
    )
    assert dashboard["sources"] == ("prime_observer", "core_signal")


def test_briefing_dashboard_extracts_core_signal_events_and_safe_references():
    dashboard = _briefing_dashboard(
        """# Source Briefing

## Core Signal
- Core Signal Morning Brief - 2026-06-08 (2026-06-08) [Attention]: Slowdown.
  - Event: 1 sustained slowdown period was found.
    - Event ID: core-signal-sustained_slowdown-abc123
    - Event kind: sustained_slowdown
    - Severity/status: Attention / attention
    - Affected window: 2026-06-08T11:11:30+00:00 to 2026-06-08T11:12:09+00:00
    - Confidence: High
    - Issue location: Likely upstream/ISP issue
    - Recommended action: Check provider status if symptoms matched.
    - Attribution source: Prime Observer incident attribution
    - View investigation: viz/investigate.html?start=1&end=2

## Attribution
This briefing is source-backed using: core_signal.
""",
        "2026-06-08T18:00:00+00:00",
        ("core_signal",),
    )

    event = dashboard["core_signal_events"][0]
    assert event["summary"] == "1 sustained slowdown period was found."
    assert event["severity_status"] == "Attention / attention"
    assert event["affected_window"] == (
        "2026-06-08T11:11:30+00:00 to 2026-06-08T11:12:09+00:00"
    )
    assert event["confidence"] == "High"
    assert event["recommended_action"] == "Check provider status if symptoms matched."
    assert event["investigation_reference"] == "viz/investigate.html?start=1&end=2"
    assert "investigation_href" not in event


def test_briefing_dashboard_extracts_core_signal_explanation_metadata():
    dashboard = _briefing_dashboard(
        """# Source Briefing

## Core Signal
- Core Signal Morning Brief - 2026-06-08 (2026-06-08) [Attention]: Slowdown.
  - Event: 1 sustained slowdown period was found.
    - Interpretation: Core Signal
    - Presentation: Olivaw
    - Confidence: High
    - Why it matters: Sustained slowdown was detected.
    - Confidence rationale: Matched a sustained slowdown policy threshold.
    - Supporting facts: 1
      - Fact: WAN p95 exceeded the sustained threshold.
        - Source: prime_observer
        - Reference: http://127.0.0.1:8000/investigate.html?start=1&end=2
    - Recommended action: Check provider status if symptoms matched.
    - Recommendation trace:
      - Recommendation: Check provider status if symptoms matched.
      - Supporting facts: WAN p95 exceeded the sustained threshold.
      - Interpretation: Core Signal classified this as sustained slowdown.
    - Evidence: Prime Observer incident attribution
    - Related events:
      - Related event: core-signal-dns-watch-def456 - same_window

## Attribution
This briefing is source-backed using: core_signal.
""",
        "2026-06-08T18:00:00+00:00",
        ("core_signal",),
    )

    event = dashboard["core_signal_events"][0]
    assert event["confidence"] == "High"
    assert event["confidence_reason"] == "Matched a sustained slowdown policy threshold."
    assert event["supporting_fact_count"] == "1"
    assert event["supporting_facts"] == [
        {
            "summary": "WAN p95 exceeded the sustained threshold.",
            "source": "prime_observer",
            "reference": "http://127.0.0.1:8000/investigate.html?start=1&end=2",
        }
    ]
    assert event["recommendation_trace"] == [
        {
            "stage": "Recommendation",
            "detail": "Check provider status if symptoms matched.",
        },
        {
            "stage": "Supporting facts",
            "detail": "WAN p95 exceeded the sustained threshold.",
        },
        {
            "stage": "Interpretation",
            "detail": "Core Signal classified this as sustained slowdown.",
        },
    ]
    assert event["evidence"] == "Prime Observer incident attribution"
    assert event["interpretation"] == "Core Signal"
    assert event["presentation"] == "Olivaw"
    assert event["related_events"] == [
        "core-signal-dns-watch-def456 - same_window"
    ]


def test_briefing_route_renders_compact_wave_metadata(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "title": "June 8 WAN samples",
    "created_at": "2026-06-08T18:00:00+00:00",
    "event_count": 3,
    "status": "available",
    "path": "viz/investigation.json"
  }
]
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-08",
  "status": "Attention",
  "summary": "The network had 1 sustained slowdown period.",
  "events": [
    {
      "summary": "1 sustained slowdown period was found.",
      "confidence": "0.82",
      "confidence_reason": "Matched sustained slowdown threshold.",
      "supporting_facts": [
        {"summary": "WAN p95 exceeded threshold.", "source": "prime_observer"}
      ],
      "recommendation_trace": {
        "recommendation": "Check provider status if symptoms matched.",
        "interpretation": "Core Signal classified the event."
      },
      "related_events": [
        {"id": "core-signal-dns-watch-def456", "relationship": "same_window"}
      ],
      "interpretation_source": "core_signal",
      "attribution_source": "prime_observer_incident"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Evidence" in response.text
    assert "Prime Observer" in response.text
    assert "Interpretation" in response.text
    assert "Core Signal" in response.text
    assert "Presentation" in response.text
    assert "Olivaw" in response.text
    assert "Confidence" in response.text
    assert "0.82" in response.text
    assert "Why" in response.text
    assert "Matched sustained slowdown threshold." in response.text
    assert "Supporting facts" in response.text
    assert "WAN p95 exceeded threshold." in response.text
    assert "Recommendation trace" in response.text
    assert "Related events" in response.text
    assert "Show supporting detail" in response.text
    assert "June 8 WAN samples" in response.text


def test_briefing_route_renders_executive_summary_from_source_data(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "network_attribution.json").write_text(
        """
{
  "generated_at": "2026-06-08T18:00:00+00:00",
  "current_attribution": {
    "label": "No active network issue detected",
    "status": "no_issue_detected",
    "confidence": "high"
  }
}
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-08",
  "status": "Healthy",
  "summary": "Current LAN and WAN state appears stable.",
  "recommended_action": "No action."
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Today&apos;s Assessment" in response.text
    assert "Current LAN and WAN state appears stable." in response.text
    assert "Prime Observer reports: Current LAN/WAN state: No active network issue detected." in response.text
    assert "What Matters" in response.text
    assert "Recommended Action" in response.text
    assert "No action." in response.text


def test_briefing_route_does_not_invent_recommendation_or_confidence(
    monkeypatch,
    tmp_path,
):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-08",
  "status": "Healthy",
  "summary": "Stable."
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "No specific recommendation is available from Core Signal." in response.text
    assert "Confidence" not in response.text
    assert "Restart" not in response.text
    assert "Call provider" not in response.text


def test_briefing_route_handles_sparse_live_source_shapes(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "title": "Sparse investigation",
    "status": "",
    "path": ""
  }
]
""",
        encoding="utf-8",
    )
    (prime_dir / "investigation.json").write_text(
        """
{
  "generated_at": "2026-06-08T18:00:00+00:00",
  "navigation": {
    "first_event": {},
    "previous_event": null,
    "next_event": {},
    "last_event": null
  },
  "event_neighborhoods": [
    {
      "event": {},
      "nearby_events": null
    }
  ]
}
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "date": "2026-06-08",
  "status": "Watch",
  "summary": "Sparse report loaded.",
  "confidence": null,
  "confidence_reason": null,
  "supporting_facts": null,
  "recommendation_trace": null,
  "related_events": null,
  "events": [
    {
      "summary": "Sparse event loaded.",
      "confidence": null,
      "confidence_reason": null,
      "supporting_facts": null,
      "recommendation_trace": null,
      "related_events": null,
      "prime_observer_reference": {"path": ""}
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Today&apos;s Assessment" in response.text
    assert "Sparse report loaded." in response.text
    assert "Sparse event loaded." in response.text
    assert "Internal Server Error" not in response.text


def test_briefing_dashboard_normalizes_optional_collections():
    dashboard = _normalize_briefing_dashboard(
        {
            "core_signal_explanation": {
                "supporting_facts": None,
                "recommendation_trace": None,
            },
            "core_signal_events": [
                {
                    "summary": "Older event shape.",
                    "supporting_facts": None,
                    "recommendation_trace": None,
                    "related_events": None,
                }
            ],
            "prime_nearby_events": [{"anchor": "event-1", "events": None}],
            "investigation_references": [
                {"label": "Empty target", "target": "", "href": ""},
                {"label": "Investigation", "target": "viz/investigation.json"},
            ],
            "what_matters": None,
            "worth_knowing": None,
            "network_status": None,
            "dns_activity": None,
            "dns_details": None,
            "prime_investigations": None,
            "prime_investigation_navigation": None,
            "core_signal_findings": None,
            "source_details": None,
        }
    )

    assert dashboard["core_signal_explanation"]["supporting_facts"] == []
    assert dashboard["core_signal_explanation"]["recommendation_trace"] == []
    assert dashboard["core_signal_events"][0]["supporting_facts"] == []
    assert dashboard["core_signal_events"][0]["recommendation_trace"] == []
    assert dashboard["core_signal_events"][0]["related_events"] == []
    assert dashboard["prime_nearby_events"][0]["events"] == []
    assert dashboard["investigation_references"] == [
        {
            "label": "Investigation",
            "target": "viz/investigation.json",
            "href": "",
        }
    ]
    assert dashboard["what_matters"] == []


def test_briefing_route_preserves_attribution_boundaries(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_FILES_DIR",
        "OLIVAW_PRIME_OBSERVER_DIR",
        "OLIVAW_CORE_SIGNAL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    prime_dir = tmp_path / "prime"
    core_dir = tmp_path / "core"
    files_dir = tmp_path / "files"
    prime_dir.mkdir()
    core_dir.mkdir()
    files_dir.mkdir()
    monkeypatch.setenv("OLIVAW_FILES_DIR", str(files_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "title": "WAN fact packet",
    "status": "available",
    "path": "viz/investigation.json"
  }
]
""",
        encoding="utf-8",
    )
    (core_dir / "latest.json").write_text(
        """
{
  "title": "Core Signal Summary",
  "status": "Attention",
  "summary": "Sustained slowdown was interpreted.",
  "events": [
    {
      "summary": "Slowdown event.",
      "confidence": "medium",
      "confidence_reason": "Core Signal matched the policy.",
      "recommended_action": "Monitor the next report.",
      "interpretation_source": "core_signal",
      "attribution_source": "prime_observer_incident"
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Evidence" in response.text
    assert "Prime Observer incident attribution" in response.text
    assert "Interpretation" in response.text
    assert "Core Signal" in response.text
    assert "Presentation" in response.text
    assert "Olivaw" in response.text


def test_briefing_dashboard_links_absolute_prime_observer_investigation_url():
    dashboard = _briefing_dashboard(
        """# Source Briefing

## Core Signal
- Core Signal Morning Brief - 2026-06-08 (2026-06-08) [Attention]: Slowdown.
  - Event: 1 sustained slowdown period was found.
    - View investigation: http://127.0.0.1:8000/investigate.html?start=1&end=2
""",
        "2026-06-08T18:00:00+00:00",
        ("core_signal",),
    )

    event = dashboard["core_signal_events"][0]
    assert event["investigation_href"] == (
        "http://127.0.0.1:8000/investigate.html?start=1&end=2"
    )


def test_briefing_dashboard_extracts_prime_observer_investigations():
    dashboard = _briefing_dashboard(
        """# Source Briefing

## Prime Observer
- Current-state observations only; interpretation belongs to Core Signal.
- Investigation index data: from Prime Observer.
  - Investigation: June 8 WAN samples
    - Created at: 2026-06-08T18:00:00+00:00
    - Event count: 2
    - Status: available
    - Path: http://127.0.0.1:8000/investigation.json
- Investigation metadata: from Prime Observer.
  - Navigation metadata: from Prime Observer.
    - First event: First sample (id event-1, target #event-1)
    - Previous event: Previous sample (id event-0)
    - Next event: Next sample (id event-2, target #event-2)
    - Last event: Last sample (id event-3)
  - Nearby-event facts: from Prime Observer.
    - Events in the same investigation window for Next sample:
      - First sample (id event-1)
      - Later sample (id event-3)
- Latest sample timestamp: 2026-06-08T19:00:00+00:00

## Attribution
This briefing is source-backed using: prime_observer.
""",
        "2026-06-08T18:00:00+00:00",
        ("prime_observer",),
    )

    assert dashboard["prime_investigations"] == [
        {
            "title": "June 8 WAN samples",
            "created_at": "2026-06-08T18:00:00+00:00",
            "event_count": "2",
            "status": "available",
            "path": "http://127.0.0.1:8000/investigation.json",
            "href": "http://127.0.0.1:8000/investigation.json",
        }
    ]
    assert dashboard["prime_investigation_navigation"] == [
        {"label": "First event", "target": "First sample (id event-1, target #event-1)"},
        {"label": "Previous event", "target": "Previous sample (id event-0)"},
        {"label": "Next event", "target": "Next sample (id event-2, target #event-2)"},
        {"label": "Last event", "target": "Last sample (id event-3)"},
    ]
    assert dashboard["prime_nearby_events"] == [
        {
            "anchor": "Next sample",
            "events": ["First sample (id event-1)", "Later sample (id event-3)"],
        }
    ]


def test_briefing_route_renders_core_signal_event_without_broken_local_link(
    monkeypatch,
    tmp_path,
):
    for name in ("OLIVAW_CONFIG", "OLIVAW_FILES_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    core_dir = tmp_path / "core"
    prime_dir = tmp_path / "prime"
    core_dir.mkdir()
    prime_dir.mkdir()
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_DIR", str(core_dir))
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    (core_dir / "latest.md").write_text(
        """# Core Signal Morning Brief - 2026-06-08

Status: Attention

The network had 1 sustained slowdown period(s).

Why This Status:
Sustained slowdown was detected.

Issue Location: Likely upstream/ISP issue

Recommended Action: Check provider status if symptoms matched.

Technical Evidence:
- Window: 2026-06-08T11:11:30+00:00 to 2026-06-08T11:12:09+00:00
- Prime Observer investigation: viz/investigate.html?start=1&end=2
- Attribution source: Prime Observer incident attribution
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Interpreted events" in response.text
    assert "The network had 1 sustained slowdown period(s)." in response.text
    assert "Severity/status" in response.text
    assert "Attention / attention" in response.text
    assert "Affected window" in response.text
    assert "Confidence" not in response.text
    assert "View investigation: viz/investigate.html?start=1&amp;end=2" in response.text
    assert 'href="viz/investigate.html?start=1&amp;end=2"' not in response.text


def test_briefing_route_renders_prime_observer_investigation_metadata(
    monkeypatch,
    tmp_path,
):
    for name in ("OLIVAW_CONFIG", "OLIVAW_FILES_DIR", "OLIVAW_CORE_SIGNAL_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OLIVAW_CORE_SIGNAL_ENABLED", "false")
    prime_dir = tmp_path / "prime"
    prime_dir.mkdir()
    monkeypatch.setenv("OLIVAW_PRIME_OBSERVER_DIR", str(prime_dir))
    (prime_dir / "investigation_index.json").write_text(
        """
[
  {
    "id": "inv-20260608",
    "title": "June 8 WAN samples",
    "created_at": "2026-06-08T18:00:00+00:00",
    "event_count": 2,
    "status": "available",
    "path": "viz/investigation.json"
  }
]
""",
        encoding="utf-8",
    )
    (prime_dir / "investigation.json").write_text(
        """
{
  "generated_at": "2026-06-08T18:00:00+00:00",
  "navigation": {
    "first_event": {"id": "event-1", "label": "First sample", "anchor": "#event-1"},
    "last_event": {"id": "event-2", "label": "Last sample", "anchor": "#event-2"}
  },
  "event_neighborhoods": [
    {
      "event": {"id": "event-2", "label": "Last sample"},
      "nearby_events": [
        {"id": "event-1", "label": "First sample"}
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    response = client.get("/briefing")

    assert response.status_code == 200
    assert "Prime Observer investigations" in response.text
    assert "June 8 WAN samples" in response.text
    assert "Prime Observer investigation index" in response.text
    assert "Investigation navigation" in response.text
    assert "Prime Observer navigation metadata" in response.text
    assert "Nearby events" in response.text
    assert "Events in the same investigation window for Last sample" in response.text
    assert "Prime Observer factual discovery" in response.text
    assert "Core Signal" in response.text
    forbidden = ("correlated", "caused by", "likely related", "root cause")
    assert not any(term in response.text.lower() for term in forbidden)


def test_human_generated_time_formats_relative_and_today():
    generated = datetime(2026, 6, 6, 0, 14, 59, tzinfo=timezone.utc)

    assert _human_generated_time(generated, now=generated + timedelta(minutes=2)) == (
        "Generated 2 minutes ago"
    )

    later = generated + timedelta(hours=2)
    assert "Generated today at" in _human_generated_time(generated, now=later)


def test_dashboard_status_maps_no_action_watch_to_healthy():
    core_lines = [
        "- Core Signal Morning Brief - 2026-06-05 (2026-06-05) [Watch]: "
        "Performance was unusually slow compared with the normal pattern.",
        "- Why/status reasoning: Performance was noticeably different from "
        "historical norms, but it was not actionable because no sustained "
        "instability or user-impacting issue was detected.",
        "- Recommended action: No action unless people noticed slow calls.",
        "- DNS filtering looked normal: 2.4% of queries were blocked.",
    ]
    prime_lines = [
        "- Network attribution generated at 2026-06-05T23:18:43+00:00.",
        "- Current LAN/WAN state: No network issue detected",
        "- Current status: no_issue_detected",
    ]

    status = _dashboard_status(core_lines, prime_lines)

    assert status["label"] == "Healthy"
    assert status["tone"] == "healthy"
    assert "No action is recommended" in status["explanation"]


def test_dashboard_status_keeps_watch_when_monitoring_is_warranted():
    core_lines = [
        "- Core Signal Morning Brief - 2026-06-05 (2026-06-05) [Watch]: "
        "Recurring latency should be watched.",
        "- Why/status reasoning: Confidence: medium; worth monitoring.",
        "- Recommended action: Monitor the next weekly report.",
    ]

    status = _dashboard_status(core_lines, [])

    assert status["label"] == "Watch"
    assert status["tone"] == "watch"
    assert "worth monitoring" in status["explanation"]


def test_dashboard_status_does_not_recalculate_from_confidence_metadata():
    core_lines = [
        "- Core Signal Morning Brief - 2026-06-05 (2026-06-05) [Healthy]: Stable.",
        "- Confidence: Low",
        "- Confidence rationale: Sparse data.",
        "- Recommended action: No action.",
    ]

    status = _dashboard_status(core_lines, [])

    assert status["label"] == "Healthy"
    assert status["tone"] == "healthy"


def test_dashboard_status_marks_action_needed_for_actionable_recommendation():
    core_lines = [
        "- Core Signal Morning Brief - 2026-06-05 (2026-06-05) [Attention]: "
        "Sustained slowdown detected.",
        "- Why/status reasoning: A sustained slowdown was detected.",
        "- Recommended action: Restart the router and recheck the connection.",
    ]

    status = _dashboard_status(core_lines, [])

    assert status["label"] == "Action Needed"
    assert status["tone"] == "action"
    assert "needs attention" in status["explanation"]


def test_chat_post_renders_chat_response(monkeypatch):
    class FakeResponse:
        text = "mocked OpenAI-capable chat response"

    def fake_run_with_attribution(self, prompt, config=None):
        assert prompt == "hello"
        return FakeResponse()

    monkeypatch.setattr(
        "olivaw.web.ChatCapability.run_with_attribution",
        fake_run_with_attribution,
    )

    response = client.post("/chat", data={"prompt": "hello"})

    assert response.status_code == 200
    assert "mocked OpenAI-capable chat response" in response.text


def test_chat_post_handles_unavailable_capability_without_provider(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request):
            raise AssertionError("weather request should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    response = client.post("/chat", data={"prompt": "What's the weather in Phoenix?"})

    assert response.status_code == 200
    assert "do not currently have a weather source configured" in response.text
    assert "WeatherSource" in response.text


def test_chat_post_exact_weather_request_matches_cli_guardrails(monkeypatch):
    class FailingRouter:
        def __init__(self, config):
            self.config = config

        def complete(self, request):
            raise AssertionError("weather request should not call provider")

    monkeypatch.setattr("olivaw.capabilities.chat.RouterProvider", FailingRouter)

    response = client.post("/chat", data={"prompt": WEATHER_PROMPT})

    assert response.status_code == 200
    assert "do not currently have a weather source configured" in response.text
    assert "WeatherSource" in response.text
    assert "enable_openai_weather" not in response.text
    assert "provide weather via cloud OpenAI provider support" not in response.text
    assert "OpenAI can retrieve live weather" not in response.text


def test_settings_does_not_expose_secret(monkeypatch):
    monkeypatch.setenv("OLIVAW_OPENAI_API_KEY", "very-secret")

    response = client.get("/settings")

    assert response.status_code == 200
    assert "API key present" in response.text
    assert "very-secret" not in response.text


def test_config_route_renders_redacted_user_config(monkeypatch, tmp_path):
    for name in (
        "OLIVAW_CONFIG",
        "OLIVAW_CLOUD_ENABLED",
        "OPENAI_API_KEY",
        "OLIVAW_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = (
        tmp_path / "Library" / "Application Support" / "Olivaw" / "config.toml"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
[providers.cloud]
enabled = true
model = "gpt-4.1"

[secrets]
openai_api_key = "config-secret"
""",
        encoding="utf-8",
    )

    response = client.get("/config")

    assert response.status_code == 200
    assert "Configuration" in response.text
    assert str(config_path) in response.text
    assert "API key present" in response.text
    assert "yes" in response.text
    assert "config-secret" not in response.text
