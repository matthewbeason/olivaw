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
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dashboard = _briefing_dashboard(briefing.text, generated_at, briefing.sources)
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

    status = _overall_status(core_lines)
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
        [*prime_lines, *core_lines],
        include=("dns", "query", "domain", "block", "encrypted"),
        exclude=("available from prime observer",),
        limit=6,
    )
    core = _important_lines(
        core_lines,
        include=("why/status", "performance", "concentration", "weekday", "window"),
        exclude=("dns interpretation", "recommended action"),
        limit=5,
    )

    return {
        "status": status,
        "generated_at": generated_at,
        "sources": sources,
        "worth_knowing": worth,
        "recommended_action": recommended,
        "network_status": network,
        "dns_activity": dns,
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


def _overall_status(lines: list[str]) -> str:
    for line in lines:
        if "[" in line and "]" in line:
            return line.split("[", 1)[1].split("]", 1)[0].strip()
    return "Source-backed"


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
        if _matches(line, include=include, exclude=exclude)
    ]
    if not selected and fallback:
        selected = [_clean_briefing_line(line) for line in fallback]

    deduped: list[str] = []
    for line in selected:
        if line and line not in deduped:
            deduped.append(line)
    return deduped[:limit]


def _first_matching(lines: list[str], label: str) -> str | None:
    lowered = label.lower()
    for line in lines:
        if lowered in line.lower():
            return _clean_briefing_line(line)
    return None


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
