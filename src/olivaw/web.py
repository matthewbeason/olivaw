from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
app = FastAPI(title="Olivaw", version="0.1.0")


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
    dashboard = _briefing_dashboard(briefing.text, generated_at, briefing.sources)
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
        exclude=("recommended action",),
        limit=5,
    )

    return {
        "status": status,
        "status_label": status["label"],
        "status_tone": status["tone"],
        "status_explanation": status["explanation"],
        "generated_at": generated_at,
        "generated_display": generated_at,
        "sources": sources,
        "worth_knowing": worth,
        "recommended_action": recommended,
        "network_status": network,
        "dns_activity": dns,
        "dns_details": dns_details,
        "core_signal_findings": core,
        "source_details": source_lines,
    }


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
            "confidence: medium",
            "confidence: low",
        )
    )


def _is_no_action(recommended_action: str) -> bool:
    action = _action_text(recommended_action)
    return action.startswith("no action") or "no action unless" in action


def _action_text(recommended_action: str) -> str:
    return recommended_action.lower().removeprefix("recommended action:").strip()


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
