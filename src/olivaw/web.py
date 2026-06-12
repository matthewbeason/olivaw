from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from olivaw.briefing import compose_briefing, compose_source_briefing
from olivaw.briefing.schemas import DailyContext, Priority, ProjectState, Signal
from olivaw.capabilities.chat import ChatCapability
from olivaw.capabilities.sources import SourceInspectionCapability
from olivaw.config import load_config, public_config
from olivaw.health import run_health_checks
from olivaw.assistant.identity import get_identity

TEMPLATE_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
templates.env.auto_reload = False
app = FastAPI(title="Olivaw", version="0.6.1")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    config = load_config()
    health = run_health_checks(config)
    briefing = _example_briefing()
    return templates.TemplateResponse(
        request,
        "home.html",
        {"health": health, "briefing": briefing, "config": public_config(config)},
    )


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(
        request,
        "chat.html",
        {"response": None, "prompt": ""},
    )


@app.post("/chat", response_class=HTMLResponse)
def chat_submit(request: Request, prompt: str = Form(...)):
    response = ChatCapability().run_with_attribution(prompt).text
    return templates.TemplateResponse(
        request,
        "chat.html",
        {"response": response, "prompt": prompt},
    )


@app.get("/briefing", response_class=HTMLResponse)
def briefing_page(request: Request):
    config = load_config()
    briefing = compose_source_briefing(config=config)
    generated_dt = datetime.now(timezone.utc)
    generated_at = generated_dt.isoformat(timespec="seconds")
    dashboard = _briefing_dashboard(
        briefing.text,
        generated_at,
        briefing.sources,
        prime_observer_directory=config.prime_observer.directory,
        prime_observer_base_url=config.prime_observer.base_url,
    )
    dashboard["generated_display"] = _human_generated_time(generated_dt)
    response = templates.TemplateResponse(
        request,
        "briefing.html",
        {
            "briefing": briefing,
            "dashboard": dashboard,
            "generated_at": generated_at,
            "config": public_config(config),
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def _briefing_dashboard(
    text: str,
    generated_at: str,
    sources: tuple[str, ...],
    *,
    prime_observer_directory: Path | None = None,
    prime_observer_base_url: str | None = None,
) -> dict[str, object]:
    sections = _markdown_sections(text)
    core_lines = sections.get("Core Signal", [])
    prime_lines = sections.get("Prime Observer", [])
    highlights = sections.get("Highlights", [])
    source_lines = [
        *sections.get("Sources", []),
        *sections.get("Source Notes", []),
        *sections.get("Attribution", []),
    ]

    status = _dashboard_status(core_lines, prime_lines)
    worth = _important_lines(
        core_lines,
        include=("worth", "slower", "normal", "noticed", "detected", "performance"),
        exclude=("recommended action", "why/status", "dns interpretation"),
        fallback=highlights,
        limit=3,
    )
    recommended = _first_matching(core_lines, "Recommended action") or (
        "No specific recommendation is available from Core Signal."
    )
    network = _important_lines(
        prime_lines,
        include=("lan/wan", "current status", "confidence", "latest sample", "p95"),
        exclude=("dns",),
        limit=5,
    )
    dns = _important_lines(
        prime_lines,
        include=("top blocked domain", "top resolved domain", "top queried domain"),
        exclude=("available from prime observer",),
        limit=3,
    )
    dns = _ordered_dns_facts(dns)
    dns_details = _dns_detail_lines(prime_lines)
    investigations = _prime_investigation_catalog(prime_lines)
    prime_investigation_empty_message = _diagnostic_message(
        prime_lines,
        "Investigation index",
        fallback="Investigation index file was not found at configured path.",
    )
    investigation_navigation = _prime_investigation_navigation(prime_lines)
    nearby_events = _prime_nearby_events(prime_lines)
    events = _core_signal_events(core_lines)
    core_event_empty_message = _diagnostic_message(
        core_lines,
        "Core Signal events",
        fallback="No Core Signal report file found at configured path.",
    )
    core = _important_lines(
        core_lines,
        include=(
            "why/status",
            "performance",
            "concentration",
            "weekday",
            "window",
            "dns interpretation",
        ),
        exclude=("recommended action", "event id", "event kind", "view investigation"),
        limit=5,
    )
    explanation = _core_signal_explanation(core_lines, events)
    investigations_summary = _investigation_references(
        events,
        investigations,
        investigation_navigation,
        nearby_events,
        prime_observer_directory=prime_observer_directory,
        prime_observer_base_url=prime_observer_base_url,
    )
    what_matters = _what_matters(
        network=network,
        core=core,
        dns=dns,
        worth=worth,
        recommended=recommended,
    )
    historical_findings = _historical_findings(events)
    uncertainty_items = _uncertainty_items(explanation, events)

    return _normalize_briefing_dashboard(
        {
            "status": status,
            "status_label": status["label"],
            "current_status_label": _current_status_label(status["label"]),
            "status_tone": status["tone"],
            "status_explanation": status["explanation"],
            "executive_summary": _executive_summary(
                core_lines,
                prime_lines,
                fallback=status["explanation"],
            ),
            "generated_at": generated_at,
            "generated_display": generated_at,
            "sources": sources,
            "what_matters": what_matters,
            "historical_findings": historical_findings,
            "uncertainty_items": uncertainty_items,
            "worth_knowing": worth,
            "recommended_action": recommended,
            "recommended_action_text": _recommended_action_text(
                recommended,
                status=status,
                events=events,
                investigation_references=investigations_summary,
            ),
            "network_status": network,
            "dns_activity": dns,
            "dns_details": dns_details,
            "prime_investigations": investigations,
            "prime_investigation_empty_message": prime_investigation_empty_message,
            "prime_investigation_navigation": investigation_navigation,
            "prime_nearby_events": nearby_events,
            "core_signal_findings": core,
            "core_signal_events": events,
            "core_signal_event_empty_message": core_event_empty_message,
            "core_signal_explanation": explanation,
            "investigation_references": investigations_summary,
            "investigation_actions": _investigation_actions(investigations_summary),
            "source_details": source_lines,
        }
    )


def _normalize_briefing_dashboard(dashboard: dict[str, object]) -> dict[str, object]:
    list_keys = (
        "what_matters",
        "worth_knowing",
        "historical_findings",
        "uncertainty_items",
        "network_status",
        "dns_activity",
        "dns_details",
        "prime_investigations",
        "prime_investigation_navigation",
        "prime_nearby_events",
        "core_signal_findings",
        "core_signal_events",
        "investigation_references",
        "source_details",
    )
    for key in list_keys:
        dashboard[key] = _list_value(dashboard.get(key))

    explanation = dashboard.get("core_signal_explanation")
    if not isinstance(explanation, dict):
        explanation = {}
    explanation["supporting_facts"] = _dict_list(explanation.get("supporting_facts"))
    explanation["recommendation_trace"] = _dict_list(
        explanation.get("recommendation_trace")
    )
    dashboard["core_signal_explanation"] = explanation

    events = []
    for event in _dict_list(dashboard.get("core_signal_events")):
        event["supporting_facts"] = _dict_list(event.get("supporting_facts"))
        event["recommendation_trace"] = _dict_list(event.get("recommendation_trace"))
        event["related_events"] = _list_value(event.get("related_events"))
        events.append(event)
    dashboard["core_signal_events"] = events

    nearby_groups = []
    for group in _dict_list(dashboard.get("prime_nearby_events")):
        group["events"] = _list_value(group.get("events"))
        nearby_groups.append(group)
    dashboard["prime_nearby_events"] = nearby_groups

    references = []
    for reference in _dict_list(dashboard.get("investigation_references")):
        target = str(reference.get("target") or "").strip()
        if not target:
            continue
        references.append(
            {
                "label": str(reference.get("label") or "Reference"),
                "target": target,
                "href": str(reference.get("href") or ""),
                "kind": str(reference.get("kind") or ""),
            }
        )
    dashboard["investigation_references"] = references
    actions = dashboard.get("investigation_actions")
    if not isinstance(actions, dict):
        actions = {}
    dashboard["investigation_actions"] = {
        "primary": _dict_list(actions.get("primary")),
        "event_navigation": _dict_list(actions.get("event_navigation")),
        "supporting": _dict_list(actions.get("supporting")),
        "technical": _dict_list(actions.get("technical")),
    }

    return dashboard


def _list_value(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _markdown_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        if line.startswith("## "):
            current = line.removeprefix("## ").strip()
            sections.setdefault(current, [])
            continue
        if current and line.strip():
            sections[current].append(line.strip())
    return sections


def _dashboard_status(
    core_lines: list[str],
    prime_lines: list[str],
) -> dict[str, str]:
    current_core = _current_core_lines(core_lines)
    current_text = " ".join(current_core).lower()
    prime_text = " ".join(prime_lines).lower()
    raw_status = _raw_status(current_core).lower()
    recommended_action = _first_matching(current_core, "Recommended action") or ""

    if _needs_action(raw_status, recommended_action, current_text):
        return {
            "label": "Action Needed",
            "tone": "action",
            "explanation": (
                "Core Signal is reporting an actionable condition or a "
                "recommendation that needs attention."
            ),
        }

    if _is_no_action(recommended_action) and _current_state_is_clear(
        current_text,
        prime_text,
    ):
        return {
            "label": "Healthy",
            "tone": "healthy",
            "explanation": (
                "No action is recommended, Prime Observer reports no current "
                "network issue, and DNS activity is normal."
            ),
        }

    if raw_status == "healthy":
        return {
            "label": "Healthy",
            "tone": "healthy",
            "explanation": "Sources do not report a condition needing attention.",
        }

    if raw_status == "watch" or _monitoring_is_warranted(
        recommended_action,
        current_text,
    ):
        return {
            "label": "Watch",
            "tone": "watch",
            "explanation": (
                "There is a meaningful condition worth monitoring, but no "
                "immediate action is required."
            ),
        }

    return {
        "label": "Healthy",
        "tone": "healthy",
        "explanation": "Sources do not report a condition needing attention.",
    }


def _executive_summary(
    core_lines: list[str],
    prime_lines: list[str],
    *,
    fallback: str,
) -> str:
    core_summary = _core_report_summary(core_lines)
    current_state = _first_clean_matching(
        prime_lines,
        (
            "Current LAN/WAN state:",
            "Current status:",
            "Latest sample timestamp:",
            "Status:",
        ),
    )
    if core_summary and current_state:
        return f"{core_summary} Prime Observer reports: {current_state}."
    if core_summary:
        return core_summary
    if current_state:
        return f"Prime Observer reports: {current_state}."
    return fallback


def _core_report_summary(lines: list[str]) -> str:
    for line in lines:
        cleaned = _clean_briefing_line(line)
        if not cleaned.startswith("Core Signal "):
            continue
        if cleaned.startswith(
            (
                "Core Signal events:",
                "Core Signal reports ",
            )
        ):
            continue
        _, separator, summary = cleaned.partition(":")
        if separator and summary.strip():
            return summary.strip()
        return cleaned
    return ""


def _first_clean_matching(lines: list[str], prefixes: tuple[str, ...]) -> str:
    lowered_prefixes = tuple(prefix.lower() for prefix in prefixes)
    for line in lines:
        cleaned = _clean_briefing_line(line)
        if cleaned.lower().startswith(lowered_prefixes):
            return cleaned
    return ""


def _what_matters(
    *,
    network: list[str],
    core: list[str],
    dns: list[str],
    worth: list[str],
    recommended: str,
) -> list[str]:
    candidates = [
        *network[:2],
        *core[:2],
        *dns[:1],
        *worth[:2],
    ]
    if recommended and not _is_no_action(recommended):
        candidates.append(recommended)
    deduped: list[str] = []
    for item in candidates:
        cleaned = _clean_briefing_line(item)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped[:5]


def _current_core_lines(lines: list[str]) -> list[str]:
    current: list[str] = []
    for line in lines:
        if line.startswith("- Core Signal ") and current:
            break
        if line.startswith("- Core Signal ") or current:
            current.append(line)
    return current or lines


def _raw_status(lines: list[str]) -> str:
    for line in lines:
        if "[" in line and "]" in line:
            return line.split("[", 1)[1].split("]", 1)[0].strip()
    return ""


def _needs_action(raw_status: str, recommended_action: str, text: str) -> bool:
    if raw_status in {"attention", "action needed"}:
        return True
    action = _action_text(recommended_action)
    if _is_no_action(recommended_action):
        return False
    if any(term in action for term in ("restart", "call", "fix", "disable")):
        return True
    return "sustained slowdown" in text and "0 sustained" not in text


def _current_state_is_clear(current_text: str, prime_text: str) -> bool:
    no_issue = (
        "no network issue detected" in prime_text
        or "current status: no_issue_detected" in prime_text
        or "current network state: no active issue detected" in prime_text
    )
    not_actionable = (
        "not actionable" in current_text
        or "no sustained instability" in current_text
        or "no user-impacting issue" in current_text
    )
    dns_normal = (
        "dns filtering looked normal" in current_text
        or "dns activity is normal" in current_text
        or "dns is normal" in current_text
    )
    return no_issue and not_actionable and dns_normal


def _monitoring_is_warranted(recommended_action: str, text: str) -> bool:
    action = _action_text(recommended_action)
    if any(term in action for term in ("monitor", "watch", "review", "investigate")):
        return True
    return any(
        term in text
        for term in (
            "worth monitoring",
            "review recommended",
        )
    )


def _is_no_action(recommended_action: str) -> bool:
    action = _action_text(recommended_action)
    return action.startswith("no action") or "no action unless" in action


def _action_text(recommended_action: str) -> str:
    return recommended_action.lower().removeprefix("recommended action:").strip()


def _display_action_text(recommended_action: str) -> str:
    return recommended_action.removeprefix("Recommended action:").strip()


def _recommended_action_text(
    recommended_action: str,
    *,
    status: dict[str, str],
    events: list[dict[str, object]],
    investigation_references: list[dict[str, str]],
) -> str:
    action = _action_text(recommended_action)
    display_action = _display_action_text(recommended_action)
    if not display_action:
        return recommended_action
    if (
        status.get("label") == "Healthy"
        and _mentions_symptoms_or_affected_window(action)
        and (events or investigation_references)
    ):
        return (
            "No immediate network change is recommended. If people noticed "
            "symptoms during the affected window, compare reports with the "
            "investigation."
        )
    return display_action


def _mentions_symptoms_or_affected_window(action: str) -> bool:
    return any(
        term in action
        for term in (
            "noticed",
            "symptom",
            "affected",
            "matched",
            "during",
        )
    )


def _current_status_label(label: str) -> str:
    if label == "Healthy":
        return "Healthy now"
    if label == "Watch":
        return "Watch now"
    return label


def _historical_findings(events: list[dict[str, object]]) -> list[str]:
    findings: list[str] = []
    for event in events:
        summary = str(event.get("summary") or "").strip()
        if not summary:
            continue
        affected_window = str(event.get("affected_window") or "").strip()
        if affected_window:
            summary = f"{summary} Affected window: {affected_window}."
        findings.append(summary)
    return findings[:3]


def _uncertainty_items(
    explanation: dict[str, object],
    events: list[dict[str, object]],
) -> list[str]:
    candidates: list[str] = []
    candidates.extend(
        _uncertainty_candidate_texts(
            explanation,
            fields=("summary", "why", "confidence_reason"),
        )
    )
    candidates.extend(_supporting_fact_texts(explanation.get("supporting_facts")))
    candidates.extend(_trace_texts(explanation.get("recommendation_trace")))
    for event in events:
        candidates.extend(
            _uncertainty_candidate_texts(
                event,
                fields=("summary", "why", "confidence_reason"),
            )
        )
        candidates.extend(_supporting_fact_texts(event.get("supporting_facts")))
        candidates.extend(_trace_texts(event.get("recommendation_trace")))
        candidates.extend(str(item) for item in _list_value(event.get("related_events")))

    uncertain: list[str] = []
    for candidate in candidates:
        text = candidate.strip()
        if text and _looks_uncertain(text) and text not in uncertain:
            uncertain.append(text)
    return uncertain[:3]


def _uncertainty_candidate_texts(
    item: dict[str, object],
    *,
    fields: tuple[str, ...],
) -> list[str]:
    return [str(item.get(field) or "") for field in fields]


def _supporting_fact_texts(value: object) -> list[str]:
    texts: list[str] = []
    for fact in _dict_list(value):
        for key in ("summary", "source", "reference"):
            text = str(fact.get(key) or "").strip()
            if text:
                texts.append(text)
    return texts


def _trace_texts(value: object) -> list[str]:
    texts: list[str] = []
    for step in _dict_list(value):
        detail = str(step.get("detail") or "").strip()
        if detail:
            texts.append(detail)
    return texts


def _looks_uncertain(text: str) -> bool:
    lowered = text.lower()
    uncertainty_terms = (
        "unclear",
        "does not clearly distinguish",
        "cannot distinguish",
        "not enough evidence",
        "available evidence",
        "local wi-fi",
        "local wifi",
        "router",
        "upstream isp",
        "upstream/path",
        "upstream path",
    )
    return any(term in lowered for term in uncertainty_terms)


def _important_lines(
    lines: list[str],
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...] = (),
    fallback: list[str] | None = None,
    limit: int,
) -> list[str]:
    selected = [
        _clean_briefing_line(line)
        for line in lines
        if not _is_report_heading(line)
        and _matches(line, include=include, exclude=exclude)
    ]
    if not selected and fallback:
        selected = [_clean_briefing_line(line) for line in fallback]

    deduped: list[str] = []
    for line in selected:
        if line and line not in deduped:
            deduped.append(line)
    return deduped[:limit]


def _ordered_dns_facts(lines: list[str]) -> list[str]:
    ordered: list[str] = []
    for label in (
        "Top blocked domain",
        "Top resolved domain",
        "Top queried domain",
    ):
        for line in lines:
            if line.lower().startswith(label.lower()):
                ordered.append(_strip_parenthetical_metric(line))
                break
    return ordered


def _dns_detail_lines(lines: list[str]) -> list[str]:
    detail_terms = (
        "queries:",
        "query rate:",
        "block rate:",
        "encrypted",
        "raw ",
        "count ",
        "share ",
    )
    details = [
        _clean_briefing_line(line)
        for line in lines
        if any(term in line.lower() for term in detail_terms)
        and not line.lower().startswith("  - dns summary:")
    ]
    deduped: list[str] = []
    for line in details:
        if line and line not in deduped:
            deduped.append(line)
    return deduped


def _core_signal_events(lines: list[str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in lines:
        cleaned = _clean_briefing_line(line)
        if cleaned.startswith("Core Signal "):
            current = None
            continue
        label, separator, value = cleaned.partition(":")
        if not separator:
            continue
        key = label.strip().lower()
        value = value.strip()
        if key == "event":
            current = {"summary": value}
            events.append(current)
            continue
        if current is None:
            continue
        if key == "recommendation trace":
            current["_trace_open"] = "1"
            continue
        if current.get("_trace_open") and key in {
            "recommendation",
            "supporting facts",
            "interpretation",
        }:
            trace = current.setdefault("recommendation_trace", [])
            if isinstance(trace, list):
                trace.append({"stage": label.strip(), "detail": value})
            continue
        if key == "event id":
            current["event_id"] = value
        elif key == "event kind":
            current["kind"] = value
        elif key == "severity/status":
            current["severity_status"] = value
        elif key == "affected window":
            current["affected_window"] = value
        elif key == "confidence":
            current["confidence"] = value
        elif key == "confidence rationale":
            current["confidence_reason"] = value
        elif key == "why it matters":
            current["why"] = value
        elif key == "supporting facts":
            current["supporting_fact_count"] = value
        elif key == "fact":
            facts = current.setdefault("supporting_facts", [])
            if isinstance(facts, list):
                facts.append({"summary": value})
        elif key in {"source", "reference"}:
            facts = current.get("supporting_facts")
            if isinstance(facts, list) and facts and isinstance(facts[-1], dict):
                facts[-1][key] = value
        elif key == "recommended action":
            current["recommended_action"] = value
        elif key == "issue location":
            current["issue_location"] = value
        elif key == "evidence":
            current["evidence"] = value
        elif key in {"interpretation", "presentation", "interpretation source"}:
            current[key.replace(" ", "_")] = value
        elif key == "view investigation":
            current["investigation_reference"] = value
            if _is_external_url(value):
                current["investigation_href"] = value
        elif key == "evidence window":
            current["evidence_window"] = value
        elif key == "related event":
            related_events = current.setdefault("related_events", [])
            if isinstance(related_events, list):
                related_events.append(value)
    return events[:5]


def _core_signal_explanation(
    lines: list[str],
    events: list[dict[str, object]],
) -> dict[str, object]:
    explanation: dict[str, object] = {}
    for line in lines:
        cleaned = _clean_briefing_line(line)
        if cleaned.startswith("Event:"):
            break
        label, separator, value = cleaned.partition(":")
        if not separator:
            continue
        key = label.strip().lower()
        value = value.strip()
        if key == "confidence":
            explanation["confidence"] = value
        elif key == "confidence rationale":
            explanation["confidence_reason"] = value
        elif key == "why/status reasoning":
            explanation["why"] = value
        elif key == "supporting facts":
            explanation["supporting_fact_count"] = value
        elif key == "fact":
            facts = explanation.setdefault("supporting_facts", [])
            if isinstance(facts, list):
                facts.append({"summary": value})
        elif key in {"source", "reference"}:
            facts = explanation.get("supporting_facts")
            if isinstance(facts, list) and facts and isinstance(facts[-1], dict):
                facts[-1][key] = value
        elif key == "recommendation trace":
            explanation["_trace_open"] = "1"
        elif explanation.get("_trace_open") and key in {
            "recommendation",
            "supporting facts",
            "interpretation",
        }:
            trace = explanation.setdefault("recommendation_trace", [])
            if isinstance(trace, list):
                trace.append({"stage": label.strip(), "detail": value})

    if not any(
        explanation.get(key)
        for key in (
            "confidence",
            "confidence_reason",
            "why",
            "supporting_fact_count",
            "supporting_facts",
            "recommendation_trace",
        )
    ):
        for event in events:
            if any(
                event.get(key)
                for key in (
                    "confidence",
                    "confidence_reason",
                    "why",
                    "supporting_fact_count",
                    "supporting_facts",
                    "recommendation_trace",
                )
            ):
                explanation = {
                    key: event[key]
                    for key in (
                        "summary",
                        "confidence",
                        "confidence_reason",
                        "why",
                        "supporting_fact_count",
                        "supporting_facts",
                        "recommendation_trace",
                    )
                    if key in event
                }
                break

    explanation.pop("_trace_open", None)
    return explanation


def _investigation_references(
    events: list[dict[str, object]],
    investigations: list[dict[str, str]],
    navigation: list[dict[str, str]],
    nearby_events: list[dict[str, object]],
    *,
    prime_observer_directory: Path | None = None,
    prime_observer_base_url: str | None = None,
) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for event in events:
        investigation = str(event.get("investigation_reference") or "").strip()
        if investigation:
            href = str(event.get("investigation_href") or "").strip()
            if not href:
                href = _prime_observer_reference_href(
                    investigation,
                    prime_observer_directory=prime_observer_directory,
                    prime_observer_base_url=prime_observer_base_url,
                )
            references.append(
                {
                    "label": "Prime Observer investigation",
                    "target": investigation,
                    "href": href,
                    "kind": "primary_investigation",
                }
            )
        evidence_window = str(event.get("evidence_window") or "").strip()
        if evidence_window:
            references.append(
                {
                    "label": "Affected telemetry window",
                    "target": evidence_window,
                    "href": "",
                    "kind": "affected_window",
                }
            )
        for fact in event.get("supporting_facts", []):
            if not isinstance(fact, dict):
                continue
            reference = str(fact.get("reference") or "").strip()
            if reference:
                references.append(
                    {
                        "label": "Supporting fact reference",
                        "target": reference,
                        "href": reference if _is_external_url(reference) else "",
                        "kind": "supporting_fact",
                    }
                )
        for related_event in event.get("related_events", []):
            references.append(
                {
                    "label": "Related event",
                    "target": str(related_event),
                    "href": "",
                    "kind": "related_event",
                }
            )
    for investigation in investigations:
        target = str(investigation.get("path") or investigation.get("title") or "")
        if target:
            references.append(
                {
                    "label": str(investigation.get("title") or "Investigation"),
                    "target": target,
                    "href": str(investigation.get("href") or ""),
                    "kind": "investigation_index",
                }
            )
    for item in _preferred_navigation_items(navigation):
        references.append(
            {
                "label": item["label"],
                "target": item["target"],
                "href": "",
                "kind": "event_navigation",
            }
        )
    for group in nearby_events[:1]:
        events_list = group.get("events")
        if isinstance(events_list, list) and events_list:
            references.append(
                {
                    "label": "Nearby events",
                    "target": f"{len(events_list)} Prime Observer event reference(s)",
                    "href": "",
                    "kind": "nearby_events",
                }
            )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for reference in references:
        key = (reference["label"], reference["target"])
        if reference["target"] and key not in seen:
            seen.add(key)
            deduped.append(reference)
    return deduped[:12]


def _preferred_navigation_items(
    navigation: list[dict[str, str]],
) -> list[dict[str, str]]:
    selected = []
    for wanted in ("First event", "Last event"):
        for item in navigation:
            if item.get("label") == wanted:
                selected.append(item)
                break
    if selected:
        return selected
    return navigation[:2]


def _investigation_actions(
    references: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    actions: dict[str, list[dict[str, str]]] = {
        "primary": [],
        "event_navigation": [],
        "supporting": [],
        "technical": [],
    }
    for reference in references:
        kind = reference.get("kind", "")
        target = reference.get("target", "")
        href = reference.get("href", "")
        if kind == "primary_investigation":
            unavailable_reason = ""
            if not href:
                unavailable_reason = (
                    "Start the Prime Observer local server to open investigation "
                    "evidence."
                )
            actions["primary"].append(
                {
                    "label": "Open investigation evidence",
                    "detail": "Prime Observer investigation",
                    "href": href,
                    "target": target,
                    "attribution": "Prime Observer",
                    "unavailable_reason": unavailable_reason,
                }
            )
        elif kind == "affected_window":
            actions["primary"].append(
                {
                    "label": "Review affected telemetry window",
                    "detail": "Core Signal affected window",
                    "href": "",
                    "target": target,
                    "attribution": "Core Signal evidence window",
                }
            )
        elif kind == "event_navigation":
            actions["event_navigation"].append(
                {
                    "label": _navigation_action_label(reference.get("label", "")),
                    "detail": target,
                    "href": "",
                    "target": target,
                    "attribution": "Prime Observer",
                }
            )
        elif kind == "nearby_events":
            actions["event_navigation"].append(
                {
                    "label": "View nearby events",
                    "detail": target,
                    "href": "",
                    "target": target,
                    "attribution": "Prime Observer",
                }
            )
        elif kind == "supporting_fact":
            actions["supporting"].append(
                {
                    "label": "Review supporting fact",
                    "detail": "Core Signal supporting fact reference",
                    "href": href,
                    "target": target,
                    "attribution": "Core Signal",
                }
            )
        actions["technical"].append(
            {
                "label": reference.get("label", "Reference"),
                "target": target,
                "href": href,
                "attribution": _reference_attribution(kind),
            }
        )
    return actions


def _navigation_action_label(label: str) -> str:
    lowered = label.lower()
    if "first" in lowered:
        return "Inspect first event"
    if "last" in lowered:
        return "Inspect last event"
    if "previous" in lowered:
        return "Inspect previous event"
    if "next" in lowered:
        return "Inspect next event"
    return "Inspect event"


def _reference_attribution(kind: str) -> str:
    if kind in {
        "primary_investigation",
        "investigation_index",
        "event_navigation",
        "nearby_events",
    }:
        return "Prime Observer"
    if kind in {"affected_window", "supporting_fact", "related_event"}:
        return "Core Signal"
    return "Olivaw"


def _prime_observer_reference_href(
    reference: str,
    *,
    prime_observer_directory: Path | None,
    prime_observer_base_url: str | None = None,
) -> str:
    if _is_external_url(reference):
        return reference
    if not prime_observer_base_url:
        return ""
    parsed = urlparse(reference)
    path_text = parsed.path
    if not path_text or not path_text.endswith(".html"):
        return ""
    base = prime_observer_base_url.rstrip("/")
    if not _is_external_url(base):
        return ""
    relative = path_text.lstrip("/")
    if prime_observer_directory and relative.startswith(
        f"{prime_observer_directory.name}/"
    ):
        relative = relative.removeprefix(f"{prime_observer_directory.name}/")
    href = f"{base}/{relative}"
    if parsed.query:
        href = f"{href}?{parsed.query}"
    if parsed.fragment:
        href = f"{href}#{parsed.fragment}"
    return href


def _prime_investigation_catalog(lines: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in lines:
        cleaned = _clean_briefing_line(line)
        label, separator, value = cleaned.partition(":")
        if not separator:
            continue
        key = label.strip().lower()
        value = value.strip()
        if key == "investigation":
            current = {"title": value}
            entries.append(current)
            continue
        if current is None:
            continue
        if key == "created at":
            current["created_at"] = value
        elif key == "event count":
            current["event_count"] = value
        elif key == "status":
            current["status"] = value
        elif key == "path":
            current["path"] = value
            if _is_external_url(value):
                current["href"] = value
    return entries[:5]


def _prime_investigation_navigation(lines: list[str]) -> list[dict[str, str]]:
    navigation = []
    labels = {
        "first event": "First event",
        "previous event": "Previous event",
        "next event": "Next event",
        "last event": "Last event",
    }
    for line in lines:
        cleaned = _clean_briefing_line(line)
        label, separator, value = cleaned.partition(":")
        if not separator:
            continue
        key = label.strip().lower()
        if key in labels:
            navigation.append({"label": labels[key], "target": value.strip()})
    return navigation


def _prime_nearby_events(lines: list[str]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    in_nearby = False
    for line in lines:
        cleaned = _clean_briefing_line(line)
        lowered = cleaned.lower()
        if lowered.startswith("nearby-event facts:"):
            in_nearby = True
            continue
        if _ends_nearby_event_block(lowered):
            in_nearby = False
            current = None
            continue
        if not in_nearby:
            continue
        if lowered.startswith("events in the same investigation window for "):
            anchor = cleaned.removeprefix(
                "Events in the same investigation window for "
            ).removesuffix(":")
            current = {"anchor": anchor, "events": []}
            groups.append(current)
            continue
        if lowered == "nearby events:":
            current = {"anchor": "", "events": []}
            groups.append(current)
            continue
        if current is not None and cleaned:
            events = current.setdefault("events", [])
            if isinstance(events, list):
                events.append(cleaned)
    return [
        group
        for group in groups[:3]
        if isinstance(group.get("events"), list) and group.get("events")
    ]


def _diagnostic_message(
    lines: list[str],
    label: str,
    *,
    fallback: str,
) -> str:
    prefix = f"{label}:"
    for line in lines:
        cleaned = _clean_briefing_line(line)
        if cleaned.startswith(prefix):
            return cleaned.removeprefix(prefix).strip() or fallback
    return fallback


def _ends_nearby_event_block(line: str) -> bool:
    return line.startswith(
        (
            "investigation metadata:",
            "navigation metadata:",
            "investigation index data:",
            "network attribution generated",
            "dns summary:",
            "latest sample timestamp:",
            "current-state observations only",
        )
    )


def _is_external_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _strip_parenthetical_metric(line: str) -> str:
    return line.split(" (", 1)[0]


def _human_generated_time(
    generated_at: datetime,
    *,
    now: datetime | None = None,
) -> str:
    local_generated = generated_at.astimezone()
    local_now = (now or datetime.now(timezone.utc)).astimezone()
    delta_seconds = max(0, int((local_now - local_generated).total_seconds()))

    if delta_seconds < 60:
        return "Generated just now"
    if delta_seconds < 3600:
        minutes = max(1, delta_seconds // 60)
        unit = "minute" if minutes == 1 else "minutes"
        return f"Generated {minutes} {unit} ago"

    time_text = local_generated.strftime("%I:%M %p").lstrip("0")
    if local_generated.date() == local_now.date():
        return f"Generated today at {time_text}"
    date_text = f"{local_generated.strftime('%b')} {local_generated.day}"
    return f"Generated {date_text} at {time_text}"


def _first_matching(lines: list[str], label: str) -> str | None:
    lowered = label.lower()
    for line in lines:
        if lowered in line.lower():
            return _clean_briefing_line(line)
    return None


def _is_report_heading(line: str) -> bool:
    return line.startswith("- Core Signal ") or line.startswith("- Prime Observer ")


def _matches(line: str, *, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    lowered = line.lower()
    return any(term in lowered for term in include) and not any(
        term in lowered for term in exclude
    )


def _clean_briefing_line(line: str) -> str:
    cleaned = line.strip()
    while cleaned.startswith("-"):
        cleaned = cleaned[1:].strip()
    return cleaned


@app.get("/health", response_class=HTMLResponse)
def health_page(request: Request):
    report = run_health_checks()
    return templates.TemplateResponse(request, "health.html", {"health": report})


@app.get("/capabilities", response_class=HTMLResponse)
def capabilities_page(request: Request):
    return templates.TemplateResponse(
        request,
        "capabilities.html",
        {"identity": get_identity()},
    )


@app.get("/sources", response_class=HTMLResponse)
def sources_page(request: Request):
    report = SourceInspectionCapability().run(config=load_config())
    return templates.TemplateResponse(request, "sources.html", {"report": report})


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return templates.TemplateResponse(
        request,
        "settings.html",
        {"config": public_config(load_config())},
    )


@app.get("/config", response_class=HTMLResponse)
def config_page(request: Request):
    return templates.TemplateResponse(
        request,
        "config.html",
        {"config": public_config(load_config())},
    )


def _example_briefing() -> str:
    return compose_briefing(
        DailyContext(
            date="2026-06-04",
            focus="Stabilize Olivaw v0.1 as a local-first assistant foundation.",
            summary=(
                "Keep the framework small, deterministic, and ready for future "
                "assistant capabilities."
            ),
            priorities=[
                Priority(
                    title="Prefer local providers",
                    why="Cloud escalation should remain explicit.",
                    status="active",
                ),
                Priority(
                    title="Keep health checks actionable",
                    why="Missing services should guide setup instead of crashing.",
                    status="active",
                ),
            ],
            signals=[
                Signal(
                    source="built-in sample",
                    title="Briefing renders without repo fixtures",
                    detail="The web home page can run from an installed package.",
                )
            ],
            projects=[
                ProjectState(
                    name="Olivaw",
                    state="v0.1 cleanup",
                    next_step="Run tests and validate CLI behavior.",
                )
            ],
            reminders=[
                "Keep Prime Observer and Core Signal read-only as sources.",
                "Do not implement memory or background automation yet.",
            ],
        )
    )
