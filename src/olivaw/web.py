from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from olivaw.briefing import compose_briefing_from_file
from olivaw.capabilities.chat import ChatCapability
from olivaw.config import load_config, public_config
from olivaw.health import run_health_checks

TEMPLATE_DIR = Path(__file__).parent / "templates"
EXAMPLE_BRIEFING = Path("examples/daily_context.json")

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
    response = ChatCapability().run(prompt)
    return templates.TemplateResponse(
        request,
        "chat.html",
        {"response": response, "prompt": prompt},
    )


@app.get("/health", response_class=HTMLResponse)
def health_page(request: Request):
    report = run_health_checks()
    return templates.TemplateResponse(request, "health.html", {"health": report})


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return templates.TemplateResponse(
        request,
        "settings.html",
        {"config": public_config(load_config())},
    )


def _example_briefing() -> str:
    if EXAMPLE_BRIEFING.exists():
        return compose_briefing_from_file(EXAMPLE_BRIEFING)
    return "Example briefing unavailable. Run from the repository root to load fixtures."

