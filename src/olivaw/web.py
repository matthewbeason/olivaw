from __future__ import annotations

import base64
import hmac
import hashlib
import json
import os
import secrets
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

from fastapi import FastAPI, Form, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from olivaw.actions import (
    ActionExecutionContext,
    ActionHistory,
    IntentResolver,
    ActionRequest,
    create_builtin_action_registry,
    execute_action,
)
from olivaw.briefing import compose_briefing, compose_source_briefing
from olivaw.briefing.health_review import generate_health_review
from olivaw.briefing.schemas import DailyContext, Priority, ProjectState, Signal
from olivaw.capabilities.chat import ChatCapability
from olivaw.capabilities.sources import SourceInspectionCapability
from olivaw.config import OlivawConfig, load_config, public_config
from olivaw.health import run_health_checks
from olivaw.models import HealthReport
from olivaw.assistant.identity import get_identity
from olivaw.sources.registry import create_default_registry

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_AUTO_RELOAD_ENV = "OLIVAW_TEMPLATE_AUTO_RELOAD"


def _template_auto_reload_enabled() -> bool:
    value = os.environ.get(TEMPLATE_AUTO_RELOAD_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _configure_template_environment() -> bool:
    auto_reload = _template_auto_reload_enabled()
    templates.env.auto_reload = auto_reload
    if auto_reload and templates.env.cache is not None:
        templates.env.cache.clear()
    return auto_reload


def _template_response(
    request: Request,
    template_name: str,
    context: dict[str, object],
) -> HTMLResponse:
    auto_reload = _configure_template_environment()
    response = templates.TemplateResponse(request, template_name, context)
    if auto_reload:
        response.headers["Cache-Control"] = "no-store"
    return response

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
_configure_template_environment()
app = FastAPI(title="Olivaw", version="0.7.0")
HEALTH_REVIEW_CACHE_TTL = timedelta(minutes=15)
ASSISTANT_SESSION_TTL = timedelta(hours=12)
ASSISTANT_SESSION_COOKIE = "olivaw_session"


@dataclass(frozen=True)
class HealthReviewCacheEntry:
    text: str
    status: str
    reason: str = ""
    provider: str = ""
    model: str = ""
    latency_ms: int | None = None
    guardrail_rejected: bool = False
    generated_at: datetime | None = None
    expires_at: datetime | None = None
    source_fingerprint: str = ""
    cache_state: str = "missing"
    generated_display: str = ""

    @property
    def available(self) -> bool:
        return self.status == "available"


_HEALTH_REVIEW_CACHE: HealthReviewCacheEntry | None = None
_ACTION_REGISTRY = create_builtin_action_registry()
_ACTION_HISTORY = ActionHistory()
_INTENT_RESOLVER = IntentResolver(_ACTION_REGISTRY)
_ASSISTANT_SESSION_SIGNING_KEY = secrets.token_bytes(32)
_ASSISTANT_SESSIONS: dict[str, AssistantSessionState] = {}


@dataclass
class AssistantConversationTurn:
    prompt: str = ""
    response: str | None = None
    suggested_action: dict[str, object] | None = None
    action_result: object | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AssistantSessionState:
    history: ActionHistory = field(default_factory=ActionHistory)
    turns: list[AssistantConversationTurn] = field(default_factory=list)
    dismissed_capsule_keys: set[str] = field(default_factory=set)
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    session_id, session_state, is_new = _assistant_session(request)
    response = _template_response(
        request,
        "home.html",
        {
            **_assistant_template_context(
            request=request,
            session_state=session_state,
            prompt=request.query_params.get("prompt", ""),
            show_action_result=_show_action_result(request),
            )
        },
    )
    _set_assistant_session_cookie(response, session_id, set_cookie=is_new)
    return response


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    session_id, session_state, is_new = _assistant_session(request)
    response = _template_response(
        request,
        "chat.html",
        {
            **_assistant_template_context(
            request=request,
            session_state=session_state,
            prompt=request.query_params.get("prompt", ""),
            show_action_result=_show_action_result(request),
            )
        },
    )
    _set_assistant_session_cookie(response, session_id, set_cookie=is_new)
    return response


@app.post("/chat", response_class=HTMLResponse)
def chat_submit(request: Request, prompt: str = Form(...)):
    session_id, session_state, is_new = _assistant_session(request)
    suggestion = _resolve_action_suggestion(prompt)
    if suggestion is not None:
        suggestion_request = ActionRequest(action_id=str(suggestion["action_id"]))
        session_state.history.record_suggestion(suggestion_request)
        _ACTION_HISTORY.record_suggestion(suggestion_request)
        _store_assistant_interaction(
            session_state,
            prompt=prompt,
            response="I can do that.",
            suggested_action=suggestion,
        )
        response = _template_response(
            request,
            "chat.html",
            {
                **_assistant_template_context(
                request=request,
                session_state=session_state,
                prompt=prompt,
                response="I can do that.",
                suggested_action=suggestion,
                )
            },
        )
        _set_assistant_session_cookie(response, session_id, set_cookie=is_new)
        return response

    message = ChatCapability().run_with_attribution(prompt).text
    _store_assistant_interaction(
        session_state,
        prompt=prompt,
        response=message,
    )
    response = _template_response(
        request,
        "chat.html",
        {
            **_assistant_template_context(
            request=request,
            session_state=session_state,
            prompt=prompt,
            response=message,
            )
        },
    )
    _set_assistant_session_cookie(response, session_id, set_cookie=is_new)
    return response


@app.get("/chat/actions/approve")
def approve_chat_action_fallback(request: Request):
    redirect_to = _redirect_with_params(
        "/chat",
        prompt=request.query_params.get("prompt", ""),
    )
    return RedirectResponse(redirect_to, status_code=303)


@app.post("/chat/actions/approve")
def approve_chat_action(
    request: FastAPIRequest,
    action_id: str = Form(...),
    prompt: str = Form(""),
):
    session_id, session_state, is_new = _assistant_session(request)
    result = _execute_web_action(action_id, history=session_state.history)
    _store_assistant_interaction(
        session_state,
        prompt="",
        response="Action executed.",
        action_result=result,
    )
    redirect_to = _safe_redirect(request.headers.get("referer") or "/chat")
    response = RedirectResponse(
        _redirect_with_params(
            redirect_to,
            prompt=prompt,
            action_result="1",
            action_response="1",
        ),
        status_code=303,
    )
    _set_assistant_session_cookie(response, session_id, set_cookie=is_new)
    return response


@app.get("/briefing", response_class=HTMLResponse)
def briefing_page(request: Request):
    config = load_config()
    briefing, dashboard, generated_at = _source_backed_dashboard(config)
    overview = _overview_context(dashboard, run_health_checks(config))
    response = _template_response(
        request,
        "briefing.html",
        {
            "briefing": briefing,
            "dashboard": dashboard,
            "overview": overview,
            "generated_at": generated_at,
            "config": public_config(config),
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/health-review/refresh")
def refresh_health_review(request: FastAPIRequest):
    session_id, session_state, is_new = _assistant_session(request)
    result = _execute_web_action("refresh_health_review", history=session_state.history)
    _store_assistant_interaction(
        session_state,
        prompt="",
        response="Action executed.",
        action_result=result,
    )
    redirect_to = _safe_redirect(request.headers.get("referer") or "/briefing")
    response = RedirectResponse(
        _redirect_with_params(
            redirect_to,
            action_result="1",
            action_response="1",
        ),
        status_code=303,
    )
    _set_assistant_session_cookie(response, session_id, set_cookie=is_new)
    return response


@app.get("/health-review/refresh")
def refresh_health_review_fallback():
    return RedirectResponse("/briefing", status_code=303)


@app.post("/actions/execute")
def execute_action_request(
    request: FastAPIRequest,
    action_id: str = Form(...),
):
    session_id, session_state, is_new = _assistant_session(request)
    result = _execute_web_action(action_id, history=session_state.history)
    _store_assistant_interaction(
        session_state,
        prompt="",
        response="Action executed.",
        action_result=result,
    )
    redirect_to = _safe_redirect(request.headers.get("referer") or "/")
    response = RedirectResponse(
        _redirect_with_params(
            redirect_to,
            action_result="1",
            action_response="1",
        ),
        status_code=303,
    )
    _set_assistant_session_cookie(response, session_id, set_cookie=is_new)
    return response


@app.get("/actions/execute")
def execute_action_request_fallback():
    return RedirectResponse("/", status_code=303)


@app.post("/assistant/cards/dismiss")
def dismiss_assistant_card(
    request: FastAPIRequest,
    card_key: str = Form(...),
):
    session_id, session_state, is_new = _assistant_session(request)
    _dismiss_assistant_card(session_state, card_key)
    redirect_to = _safe_redirect(request.headers.get("referer") or "/")
    response = RedirectResponse(redirect_to, status_code=303)
    _set_assistant_session_cookie(response, session_id, set_cookie=is_new)
    return response


@app.get("/assistant/cards/dismiss")
def dismiss_assistant_card_fallback(request: Request):
    redirect_to = _safe_redirect(request.headers.get("referer") or "/")
    return RedirectResponse(redirect_to, status_code=303)


def _execute_web_action(action_id: str, *, history: ActionHistory | None = None):
    config = load_config()
    request = ActionRequest(action_id=action_id)
    active_history = history or _ACTION_HISTORY
    active_history.record_approval(request)
    _ACTION_HISTORY.record_approval(request)
    result = execute_action(
        _ACTION_REGISTRY,
        request,
        _action_execution_context(config),
        history=active_history,
    )
    if active_history is not _ACTION_HISTORY:
        _ACTION_HISTORY.record(request, result)
    return result


def _safe_redirect(redirect_to: str) -> str:
    parsed = urlparse(redirect_to)
    if parsed.netloc:
        redirect_to = parsed.path or "/"
        if parsed.query:
            redirect_to = f"{redirect_to}?{parsed.query}"
    return redirect_to


def _redirect_with_params(redirect_to: str, **params: str) -> str:
    split = urlsplit(redirect_to)
    query = dict(parse_qsl(split.query, keep_blank_values=True))
    for key, value in params.items():
        if value:
            query[key] = value
    encoded_query = urlencode(query)
    return urlunsplit(
        (
            split.scheme,
            split.netloc,
            split.path or "/",
            encoded_query,
            split.fragment,
        )
    )


def _assistant_session(
    request: Request,
) -> tuple[str, AssistantSessionState, bool]:
    now = datetime.now(timezone.utc)
    _prune_assistant_sessions(now)
    token = request.cookies.get(ASSISTANT_SESSION_COOKIE, "")
    session_id = _verified_session_id(token)
    if session_id and session_id in _ASSISTANT_SESSIONS:
        state = _ASSISTANT_SESSIONS[session_id]
        state.last_seen_at = now
        return session_id, state, False

    session_id = secrets.token_urlsafe(18)
    state = AssistantSessionState(last_seen_at=now)
    _ASSISTANT_SESSIONS[session_id] = state
    return session_id, state, True


def _prune_assistant_sessions(now: datetime) -> None:
    expires_before = now - ASSISTANT_SESSION_TTL
    for session_id, state in list(_ASSISTANT_SESSIONS.items()):
        if state.last_seen_at < expires_before:
            del _ASSISTANT_SESSIONS[session_id]


def _signed_session_id(session_id: str) -> str:
    signature = hmac.new(
        _ASSISTANT_SESSION_SIGNING_KEY,
        session_id.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{session_id}.{encoded_signature}"


def _verified_session_id(token: str) -> str | None:
    session_id, _, signature = token.partition(".")
    if not session_id or not signature:
        return None
    expected = _signed_session_id(session_id).partition(".")[2]
    if hmac.compare_digest(signature, expected):
        return session_id
    return None


def _set_assistant_session_cookie(
    response: HTMLResponse | RedirectResponse,
    session_id: str,
    *,
    set_cookie: bool,
) -> None:
    if not set_cookie:
        return
    response.set_cookie(
        ASSISTANT_SESSION_COOKIE,
        _signed_session_id(session_id),
        httponly=True,
        samesite="lax",
        path="/",
    )


def _store_assistant_interaction(
    session_state: AssistantSessionState,
    *,
    prompt: str,
    response: str | None = None,
    suggested_action: dict[str, object] | None = None,
    action_result: object | None = None,
) -> None:
    session_state.last_seen_at = datetime.now(timezone.utc)
    session_state.turns.append(
        AssistantConversationTurn(
            prompt=prompt,
            response=response,
            suggested_action=(
                dict(suggested_action) if suggested_action is not None else None
            ),
            action_result=action_result,
            created_at=session_state.last_seen_at,
        )
    )
    session_state.turns = session_state.turns[-8:]


def _dismiss_assistant_card(
    session_state: AssistantSessionState,
    card_key: str,
) -> None:
    normalized = card_key.strip().lower()
    if not normalized:
        return
    session_state.last_seen_at = datetime.now(timezone.utc)
    session_state.dismissed_capsule_keys.add(normalized)


def _action_execution_context(config: OlivawConfig) -> ActionExecutionContext:
    def refresh_review() -> dict[str, object]:
        _, dashboard, _ = _source_backed_dashboard(config, refresh_health_review=True)
        health_review = dashboard.get("health_review")
        return {
            "status": getattr(health_review, "status", "unknown"),
            "provider": getattr(health_review, "provider", ""),
            "model": getattr(health_review, "model", ""),
            "cache_state": getattr(health_review, "cache_state", ""),
        }

    def refresh_sources() -> dict[str, object]:
        _, dashboard, generated_at = _source_backed_dashboard(config)
        aggregate = dashboard.get("source_aggregate")
        sources = (
            _dict_list(aggregate.get("sources"))
            if isinstance(aggregate, dict)
            else []
        )
        return _source_status_summary(sources) | {"generated_at": generated_at}

    def source_diagnostics() -> dict[str, object]:
        registry = create_default_registry(config)
        aggregate = registry.aggregate().as_dict()
        sources = _dict_list(aggregate.get("sources"))
        diagnostics = _dict_list(aggregate.get("diagnostics"))
        return _source_status_summary(sources) | {"diagnostics": diagnostics}

    def evidence_package() -> dict[str, object]:
        _, dashboard, _ = _source_backed_dashboard(config)
        actions = dashboard.get("investigation_actions")
        primary = (
            _dict_list(actions.get("primary"))
            if isinstance(actions, dict)
            else []
        )
        for action in primary:
            if action.get("label") == "Open Evidence Package":
                return action
        return {
            "label": "Open Evidence Package",
            "href": "",
            "unavailable_reason": "No existing evidence package link is available.",
        }

    return ActionExecutionContext(
        config=config,
        refresh_health_review=refresh_review,
        refresh_sources=refresh_sources,
        source_diagnostics=source_diagnostics,
        evidence_package=evidence_package,
    )


def _source_status_summary(sources: list[dict[str, object]]) -> dict[str, object]:
    statuses: dict[str, int] = {}
    for source in sources:
        status = str(source.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "source_count": len(sources),
        "ok_count": statuses.get("ok", 0),
        "statuses": statuses,
        "sources": [
            {
                "source_id": str(source.get("source_id") or ""),
                "source_name": str(source.get("source_name") or ""),
                "status": str(source.get("status") or "unknown"),
                "message": str(source.get("message") or ""),
                "freshness": str(source.get("freshness") or ""),
            }
            for source in sources
        ],
    }


def _action_view(
    config: OlivawConfig,
    dashboard: dict[str, object],
    *,
    history: ActionHistory,
    show_result: bool,
) -> dict[str, object]:
    available_actions = _available_actions(config, dashboard)
    history_view = _action_history_view(history)
    return {
        "available": available_actions,
        "history": history_view,
        "last_result": history_view["last_result"] if show_result else None,
        "last_action": history_view["last_action"] if show_result else None,
        "last_run_display": _action_last_run_display(history_view.get("last_run")),
    }


def _prime_observer_action_disabled(action: dict[str, object]) -> dict[str, object]:
    return {
        **action,
        "helper": "Configure a Prime Observer HTTP URL before opening it here.",
    }


def _has_evidence_href(dashboard: dict[str, object]) -> bool:
    actions = dashboard.get("investigation_actions")
    primary = _dict_list(actions.get("primary")) if isinstance(actions, dict) else []
    return any(
        action.get("label") == "Open Evidence Package" and action.get("href")
        for action in primary
    )


def _show_action_result(request: Request) -> bool:
    return request.query_params.get("action_result") == "1"


def _action_last_run_display(value: object) -> str:
    if not isinstance(value, datetime):
        return ""
    return _human_generated_time(value)


def _assistant_template_context(
    *,
    request: Request,
    session_state: AssistantSessionState,
    prompt: str,
    response: str | None = None,
    suggested_action: dict[str, object] | None = None,
    action_result: object | None = None,
    show_action_result: bool = False,
) -> dict[str, object]:
    config = load_config()
    _, dashboard, generated_at = _source_backed_dashboard(config)
    health = run_health_checks(config)
    action_view = _action_view(
        config,
        dashboard,
        history=session_state.history,
        show_result=show_action_result,
    )
    displayed_result = action_result if action_result is not None else action_view.get("last_result")
    utility = _assistant_utility_rail(
        dashboard=dashboard,
        health=health,
        actions=action_view["available"],
        generated_at=generated_at,
    )
    return {
        "greeting": "Good morning Matthew.",
        "intro": "How can I help you today?",
        "orb": _assistant_orb_state(
            dashboard=dashboard,
            action_result=displayed_result if show_action_result else None,
        ),
        "prompt": prompt,
        "timeline": _assistant_timeline(
            session_state=session_state,
            dashboard=dashboard,
        ),
        "actions": action_view,
        "prompt_suggestions": _assistant_prompt_suggestions(),
        "utility": utility,
        "dashboard": dashboard,
        "health": health,
        "generated_at": generated_at,
        "config": public_config(config),
        "is_home": request.url.path == "/",
    }


def _assistant_timeline(
    *,
    session_state: AssistantSessionState,
    dashboard: dict[str, object],
) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []
    for turn in session_state.turns:
        if turn.prompt:
            timeline.append(
                {
                    "role": "user",
                    "speaker": "You",
                    "body": turn.prompt,
                    "capsules": [],
                }
            )

        capsules = _assistant_context_capsules(
            session_state=session_state,
            prompt=turn.prompt,
            response=turn.response,
            dashboard=dashboard,
            suggested_action=turn.suggested_action,
            action_result=turn.action_result,
        )
        if turn.response or capsules:
            timeline.append(
                {
                    "role": "assistant",
                    "speaker": "Olivaw",
                    "body": turn.response or "",
                    "capsules": capsules,
                }
            )
    return timeline


def _assistant_prompt_suggestions() -> list[str]:
    return [
        "How was the network overnight?",
        "Anything important happened?",
        "What's the weather today?",
        "What changed recently?",
        "What should I know today?",
    ]


def _assistant_utility_rail(
    *,
    dashboard: dict[str, object],
    health: HealthReport,
    actions: list[dict[str, object]],
    generated_at: str,
) -> dict[str, object]:
    return {
        "links": [
            {"href": "/sources", "label": "Sources"},
            {"href": "/settings", "label": "Settings"},
            {"href": "/sources", "label": "Diagnostics"},
            {"href": "/health", "label": "Health"},
            {"href": "/config", "label": "Configuration"},
        ],
        "available_count": 5,
        "actions": actions,
        "generated_at": generated_at,
        "health": health,
        "dashboard": dashboard,
    }


def _assistant_context_capsules(
    *,
    session_state: AssistantSessionState,
    prompt: str,
    response: str | None,
    dashboard: dict[str, object],
    suggested_action: dict[str, object] | None,
    action_result: object | None,
) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    dismissed_keys = {
        str(key).strip().lower() for key in session_state.dismissed_capsule_keys
    }
    lower_prompt = prompt.lower()
    lower_response = (response or "").lower()

    if suggested_action is not None:
        cards.append(_action_suggestion_context_card(suggested_action, prompt))

    if action_result is not None:
        cards.append(_action_result_context_card(action_result))
        diagnostics = _diagnostics_context_card_from_result(action_result)
        if diagnostics is not None:
            cards.append(diagnostics)

    if _assistant_mentions_weather(lower_prompt, lower_response):
        weather = _weather_context_card(dashboard)
        if weather is not None:
            cards.append(weather)

    if _assistant_mentions_network(lower_prompt, lower_response):
        cards.append(_network_context_card(dashboard))
        evidence = _evidence_context_card(dashboard)
        if evidence is not None:
            cards.append(evidence)
    elif _assistant_mentions_evidence(lower_prompt, lower_response):
        evidence = _evidence_context_card(dashboard)
        if evidence is not None:
            cards.append(evidence)

    if _assistant_mentions_diagnostics(lower_prompt, lower_response):
        cards.append(_diagnostics_context_card(dashboard))

    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for card in cards:
        dismiss_key = str(card.get("dismiss_key") or "").strip().lower()
        if dismiss_key and dismiss_key in dismissed_keys:
            continue
        key = (str(card.get("kind") or ""), str(card.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
    return deduped[:3]


def _assistant_mentions_weather(prompt: str, response: str) -> bool:
    return _contains_any(prompt, ("weather", "forecast", "temperature", "temp", "rain")) or _contains_any(
        response,
        ("weather", "forecast", "temperature", "rain chance"),
    )


def _assistant_mentions_network(prompt: str, response: str) -> bool:
    return _contains_any(
        prompt,
        ("network", "slowdown", "overnight", "changed recently", "what changed"),
    ) or _contains_any(
        response,
        ("network", "slowdown", "incident", "healthy overall"),
    )


def _assistant_mentions_evidence(prompt: str, response: str) -> bool:
    return _contains_any(
        prompt,
        (
            "evidence",
            "package",
            "investigation",
            "telemetry",
            "why does core signal",
            "why core signal",
            "why do you think",
        ),
    ) or _contains_any(
        response,
        ("evidence package", "telemetry", "investigation", "core signal thinks"),
    )


def _assistant_mentions_diagnostics(prompt: str, response: str) -> bool:
    return _contains_any(prompt, ("diagnostics", "sources", "freshness")) or _contains_any(
        response,
        ("diagnostics", "sources refreshed", "source diagnostics"),
    )


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _context_card(
    kind: str,
    title: str,
    *,
    label: str,
    tone: str,
    summary: str = "",
    helper: str = "",
    meta_rows: list[dict[str, str]] | None = None,
    details: list[str] | None = None,
    links: list[dict[str, str]] | None = None,
    approval_action_id: str = "",
    prompt: str = "",
    button_label: str = "",
    cta_href: str = "",
    cta_label: str = "",
    source_note: str = "",
    dismissible: bool = True,
    auto_dismiss_seconds: int | None = None,
) -> dict[str, object]:
    return {
        "capsule_id": f"{kind}-{secrets.token_hex(4)}",
        "dismiss_key": kind,
        "kind": kind,
        "label": label,
        "title": title,
        "tone": tone,
        "summary": summary,
        "helper": helper,
        "meta_rows": meta_rows or [],
        "details": details or [],
        "links": links or [],
        "approval_action_id": approval_action_id,
        "prompt": prompt,
        "button_label": button_label,
        "cta_href": cta_href,
        "cta_label": cta_label,
        "source_note": source_note,
        "dismissible": dismissible,
        "auto_dismiss_seconds": auto_dismiss_seconds,
    }


def _action_suggestion_context_card(
    action: dict[str, object],
    prompt: str,
) -> dict[str, object]:
    label = str(action.get("label") or "Suggested action").strip()
    return _context_card(
        "action",
        label,
        label="Approval Required",
        tone="muted",
        summary="Olivaw can take this step, but only after you explicitly approve it.",
        helper=str(action.get("helper") or ""),
        approval_action_id=str(action.get("action_id") or ""),
        prompt=prompt,
        button_label=_conversational_action_label(str(action.get("action_id") or "")),
    )


def _action_result_context_card(result: object) -> dict[str, object]:
    metadata = getattr(result, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    meta_rows = []
    for label, key in (
        ("Status", "status"),
        ("Provider", "provider"),
        ("Model", "model"),
        ("Cache", "cache_state"),
    ):
        value = str(metadata.get(key) or "").strip()
        if value:
            meta_rows.append({"label": label, "value": value})
    cta_href = str(metadata.get("href") or "").strip()
    cta_label = str(metadata.get("label") or "Open result").strip()
    title = str(getattr(result, "message", "")).strip() or "Action completed."
    return _context_card(
        "action-result",
        title,
        label="Result",
        tone="healthy" if getattr(result, "success", False) else "action",
        summary=_action_result_summary(result),
        meta_rows=meta_rows,
        cta_href=cta_href,
        cta_label=cta_label,
        auto_dismiss_seconds=45 if getattr(result, "success", False) else None,
    )


def _conversational_action_label(action_id: str) -> str:
    labels = {
        "refresh_health_review": "Generate Health Review",
        "refresh_sources": "Refresh Sources",
        "open_evidence_package": "Open Evidence Package",
        "open_prime_observer": "Open Prime Observer",
        "source_diagnostics": "Show Diagnostics",
    }
    return labels.get(action_id, "Continue")


def _assistant_orb_state(
    *,
    dashboard: dict[str, object],
    action_result: object | None = None,
) -> dict[str, str | bool]:
    if action_result is not None:
        return {
            "state": "working",
            "label": "Working",
            "description": "An action just ran using existing source-backed context.",
            "animate": True,
        }

    source_aggregate = dashboard.get("source_aggregate")
    sources = (
        _dict_list(source_aggregate.get("sources"))
        if isinstance(source_aggregate, dict)
        else []
    )
    source_status = {
        str(source.get("source_id") or ""): str(source.get("status") or "")
        for source in sources
    }
    if _assistant_context_is_degraded(source_status):
        return {
            "state": "degraded",
            "label": "Degraded Context",
            "description": "Some key context is currently unavailable.",
            "animate": False,
        }

    events = _dict_list(dashboard.get("core_signal_events"))
    if _assistant_has_attention_condition(dashboard, events):
        return {
            "state": "attention",
            "label": "Attention",
            "description": "Current sources indicate an active condition worth attention.",
            "animate": False,
        }
    if _assistant_has_notable_context(dashboard, events):
        return {
            "state": "notable",
            "label": "Notable",
            "description": "There is something new worth asking about.",
            "animate": False,
        }
    return {
        "state": "calm",
        "label": "Calm",
        "description": "Everything appears normal from the current source-backed context.",
        "animate": False,
    }


def _assistant_context_is_degraded(source_status: dict[str, str]) -> bool:
    key_sources = ("prime_observer", "core_signal")
    degraded = {"error", "unavailable"}
    return any(source_status.get(source_id) in degraded for source_id in key_sources)


def _assistant_has_attention_condition(
    dashboard: dict[str, object],
    events: list[dict[str, object]],
) -> bool:
    if str(dashboard.get("status_label") or "").strip().lower() == "action needed":
        return True
    for event in events:
        status = str(event.get("status") or "").strip().lower()
        severity = str(event.get("severity") or "").strip().lower()
        if status in {"attention", "action", "active"} or severity in {
            "attention",
            "action",
            "critical",
        }:
            return True
    return False


def _assistant_has_notable_context(
    dashboard: dict[str, object],
    events: list[dict[str, object]],
) -> bool:
    if events:
        return True
    return str(dashboard.get("status_label") or "").strip().lower() == "watch"


def _weather_context_card(dashboard: dict[str, object]) -> dict[str, object] | None:
    weather = dashboard.get("weather_context")
    if not isinstance(weather, dict) or not weather.get("summary"):
        return None
    headline, details = _weather_capsule_copy(str(weather.get("summary") or ""))
    return _context_card(
        "weather",
        headline,
        label="Weather",
        tone="weather",
        summary=details,
        meta_rows=[
            {
                "label": "Updated",
                "value": str(
                    weather.get("freshness") or weather.get("observed_at") or "Available"
                ),
            }
        ],
        source_note="Weather details appear only when the current request calls for them.",
        auto_dismiss_seconds=45,
    )


def _network_context_card(dashboard: dict[str, object]) -> dict[str, object]:
    signal = _network_signal(dashboard)
    fields = _dict_list(signal.get("fields"))
    return _context_card(
        "network",
        str(signal.get("headline") or "Network context"),
        label="Network Context",
        tone=str(signal.get("tone") or "healthy"),
        summary=str(signal.get("summary") or ""),
        meta_rows=[
            {"label": item["label"], "value": item["value"]}
            for item in fields[:4]
        ],
        source_note="Network context is grounded in Prime Observer facts and Core Signal interpretation.",
    )


def _evidence_context_card(dashboard: dict[str, object]) -> dict[str, object] | None:
    actions = dashboard.get("investigation_actions")
    if not isinstance(actions, dict):
        return None
    links = [
        {
            "label": str(item.get("label") or "Reference"),
            "href": str(item.get("href") or ""),
            "detail": str(item.get("detail") or item.get("target") or ""),
            "unavailable_reason": str(item.get("unavailable_reason") or ""),
        }
        for item in (
            _dict_list(actions.get("primary"))[:1]
            + _dict_list(actions.get("event_navigation"))[:2]
            + _dict_list(actions.get("supporting"))[:2]
        )
    ]
    if not links:
        return None
    return _context_card(
        "evidence",
        "Evidence Package",
        label="Evidence",
        tone="watch",
        summary="Source-backed evidence and related navigation are available in this conversation.",
        details=[str(item) for item in _list_value(dashboard.get("what_we_know"))[:3]],
        links=links,
        source_note=(
            "Evidence links come from Prime Observer, interpretation comes from Core Signal, "
            "and presentation is by Olivaw."
        ),
    )


def _diagnostics_context_card(dashboard: dict[str, object]) -> dict[str, object]:
    source_aggregate = dashboard.get("source_aggregate")
    sources = (
        _dict_list(source_aggregate.get("sources"))
        if isinstance(source_aggregate, dict)
        else []
    )
    return _context_card(
        "diagnostics",
        "Source Diagnostics",
        label="Diagnostics",
        tone=_source_panel_tone(sources),
        summary=_source_panel_summary(sources),
        meta_rows=[
            {"label": "Sources", "value": str(len(sources))},
            {
                "label": "Health Review",
                "value": str(getattr(dashboard.get("health_review"), "status", "unknown")),
            },
        ],
        details=[
            (
                f"{source.get('source_name') or source.get('source_id')} "
                f"({source.get('source_id')}): {source.get('status')}"
            )
            for source in sources[:4]
        ],
    )


def _diagnostics_context_card_from_result(result: object) -> dict[str, object] | None:
    metadata = getattr(result, "metadata", {})
    if not isinstance(metadata, dict):
        return None
    sources = _dict_list(metadata.get("sources"))
    if not sources:
        return None
    summary = str(getattr(result, "message", "")).strip() or "Source diagnostics are available."
    return _context_card(
        "diagnostics",
        "Source Diagnostics",
        label="Diagnostics",
        tone="healthy" if getattr(result, "success", False) else "action",
        summary=summary,
        details=[
            (
                f"{source.get('source_name') or source.get('source_id')} "
                f"({source.get('source_id')}): {source.get('status')}"
                f"{'; ' + str(source.get('freshness')) if source.get('freshness') else ''}"
            )
            for source in sources
        ],
    )


def _action_result_summary(result: object) -> str:
    metadata = getattr(result, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if "source_count" in metadata and "ok_count" in metadata:
        return (
            f"{metadata.get('ok_count')}/{metadata.get('source_count')} sources are available."
        )
    return ""


def _weather_capsule_copy(summary: str) -> tuple[str, str]:
    parts = [part.strip() for part in summary.split(";", 1)]
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    return "Weather", summary


def _resolve_action_suggestion(prompt: str) -> dict[str, object] | None:
    match = _INTENT_RESOLVER.resolve(prompt)
    if match is None:
        return None
    config = load_config()
    _, dashboard, _ = _source_backed_dashboard(config)
    for action in _available_actions(config, dashboard):
        if action["action_id"] == match.action_id:
            return action
    return None


def _available_actions(
    config: OlivawConfig,
    dashboard: dict[str, object],
) -> list[dict[str, object]]:
    available_actions = [
        definition.as_dict() | {"disabled": False, "helper": ""}
        for definition in _ACTION_REGISTRY.list_actions()
    ]
    if not config.prime_observer.base_url:
        available_actions = [
            _prime_observer_action_disabled(action)
            if action["action_id"] == "open_prime_observer"
            else action
            for action in available_actions
        ]
    if not _has_evidence_href(dashboard):
        available_actions = [
            {
                **action,
                "helper": "No HTTP-backed evidence link is available yet.",
            }
            if action["action_id"] == "open_evidence_package"
            else action
            for action in available_actions
        ]
    return available_actions


def _action_history_view(history: ActionHistory) -> dict[str, object]:
    payload = history.as_dict()
    return payload | {
        "last_run_display": _action_last_run_display(payload.get("last_run")),
    }


def _source_backed_dashboard(
    config: OlivawConfig,
    *,
    refresh_health_review: bool = False,
) -> tuple[object, dict[str, object], str]:
    registry = create_default_registry(config)
    source_aggregate = registry.aggregate()
    briefing = compose_source_briefing(registry=registry)
    generated_dt = datetime.now(timezone.utc)
    generated_at = generated_dt.isoformat(timespec="seconds")
    dashboard = _briefing_dashboard(
        briefing.text,
        generated_at,
        briefing.sources,
        prime_observer_directory=config.prime_observer.directory,
        prime_observer_base_url=config.prime_observer.base_url,
    )
    dashboard["source_aggregate"] = source_aggregate.as_dict()
    dashboard["weather_context"] = _weather_context(source_aggregate.as_dict())
    fingerprint = _health_review_source_fingerprint(dashboard)
    dashboard["health_review"] = _cached_health_review(
        dashboard,
        config=config,
        source_fingerprint=fingerprint,
        refresh=refresh_health_review,
    )
    dashboard["generated_display"] = _human_generated_time(generated_dt)
    return briefing, dashboard, generated_at


def _cached_health_review(
    dashboard: dict[str, object],
    *,
    config: OlivawConfig,
    source_fingerprint: str,
    refresh: bool = False,
    now: datetime | None = None,
) -> HealthReviewCacheEntry:
    global _HEALTH_REVIEW_CACHE
    current_time = now or datetime.now(timezone.utc)
    cached = _HEALTH_REVIEW_CACHE
    if not refresh and cached is not None:
        cache_state = "cached"
        if cached.expires_at is not None and cached.expires_at <= current_time:
            cache_state = "stale"
        return replace(
            cached,
            cache_state=cache_state,
            generated_display=_health_review_generated_display(cached, current_time),
        )

    if not refresh:
        return HealthReviewCacheEntry(
            text="Health review not generated yet.",
            status="not_generated",
            reason="Use refresh to generate a cached Health Review.",
            source_fingerprint=source_fingerprint,
            cache_state="missing",
        )

    result = generate_health_review(dashboard, config=config)
    _HEALTH_REVIEW_CACHE = HealthReviewCacheEntry(
        text=result.text,
        status=result.status,
        reason=result.reason,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
        guardrail_rejected=result.guardrail_rejected,
        generated_at=current_time,
        expires_at=current_time + HEALTH_REVIEW_CACHE_TTL,
        source_fingerprint=source_fingerprint,
        cache_state="cached",
        generated_display=_health_review_generated_display(
            HealthReviewCacheEntry(
                text=result.text,
                status=result.status,
                generated_at=current_time,
            ),
            current_time,
        ),
    )
    return _HEALTH_REVIEW_CACHE


def _health_review_generated_display(
    entry: HealthReviewCacheEntry,
    now: datetime,
) -> str:
    if entry.generated_at is None:
        return ""
    return _human_generated_time(entry.generated_at, now=now)


def _health_review_source_fingerprint(dashboard: dict[str, object]) -> str:
    payload = {
        "source_aggregate": dashboard.get("source_aggregate"),
        "status": dashboard.get("status"),
        "network_status": dashboard.get("network_status"),
        "core_signal_events": dashboard.get("core_signal_events"),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _overview_context(dashboard: dict[str, object], health: HealthReport) -> dict[str, object]:
    source_aggregate = dashboard.get("source_aggregate")
    if not isinstance(source_aggregate, dict):
        source_aggregate = {}
    sources = _dict_list(source_aggregate.get("sources"))
    health_review = dashboard.get("health_review")
    if not hasattr(health_review, "status"):
        health_review = None

    source_freshness = _source_freshness_items(sources, health_review, health)
    return {
        "context_cards": _overview_context_cards(dashboard, sources, health),
        "recent_activity": _recent_activity_items(dashboard, sources, health_review),
        "source_freshness": source_freshness,
        "network_signal": _network_signal(dashboard),
        "ask_prompts": [
            "How was the network overnight?",
            "Anything important today?",
            "Show me the evidence package.",
            "What changed recently?",
        ],
    }


def _overview_context_cards(
    dashboard: dict[str, object],
    sources: list[dict[str, object]],
    health: HealthReport,
) -> list[dict[str, str]]:
    network = _list_value(dashboard.get("network_status"))
    core_events = _dict_list(dashboard.get("core_signal_events"))
    cards = [
        {
            "title": "Network",
            "tone": str(dashboard.get("status_tone") or "healthy"),
            "summary": str(network[0]) if network else str(dashboard.get("status_explanation") or ""),
            "meta": f"{len(core_events)} interpreted event{'s' if len(core_events) != 1 else ''}",
        },
        {
            "title": "System / Sources",
            "tone": _source_panel_tone(sources),
            "summary": _source_panel_summary(sources),
            "meta": f"{len(sources)} registered sources",
        },
        {
            "title": "Calendar",
            "tone": "muted",
            "summary": "Calendar source is not configured in this wave.",
            "meta": "placeholder",
        },
    ]
    weather = dashboard.get("weather_context")
    if isinstance(weather, dict) and weather.get("summary"):
        cards.insert(
            1,
            {
                "title": "Weather",
                "tone": "weather",
                "summary": str(weather.get("summary")),
                "meta": str(weather.get("freshness") or weather.get("observed_at") or "available"),
            },
        )
    cards.append(
        {
            "title": "Local Model",
            "tone": str(health.local.state or "muted"),
            "summary": health.local.message,
            "meta": health.local.model or health.local.name,
        }
    )
    return cards


def _source_freshness_items(
    sources: list[dict[str, object]],
    health_review: object | None,
    health: HealthReport,
) -> list[dict[str, str]]:
    items = []
    for source in sources:
        label = str(source.get("source_name") or source.get("source_id") or "Source")
        status = str(source.get("status") or "unknown")
        freshness = str(source.get("freshness") or source.get("message") or "")
        items.append(
            {
                "label": label,
                "status": status,
                "detail": freshness or "no freshness timestamp",
            }
        )
    if health_review is not None:
        detail = health_review.reason or health_review.model or health_review.provider or ""
        items.append(
            {
                "label": "Health Review",
                "status": health_review.status,
                "detail": detail or ("available" if health_review.available else "unavailable"),
            }
        )
    items.append(
        {
            "label": "Ollama",
            "status": health.local.state,
            "detail": health.local.message,
        }
    )
    return items


def _recent_activity_items(
    dashboard: dict[str, object],
    sources: list[dict[str, object]],
    health_review: object | None,
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if health_review is not None:
        if health_review.available:
            detail = health_review.model or health_review.provider or "available"
            items.append({"label": "Health review generated", "detail": detail})
        elif health_review.reason or health_review.status:
            detail = health_review.reason or health_review.status
            items.append({"label": "Health review unavailable", "detail": detail})

    weather = dashboard.get("weather_context")
    if isinstance(weather, dict) and weather.get("summary"):
        detail = str(weather.get("observed_at") or weather.get("freshness") or weather["summary"])
        items.append({"label": "Weather updated", "detail": detail})

    network = _list_value(dashboard.get("network_status"))
    if network:
        items.append({"label": "Network status available", "detail": str(network[0])})

    for source_id, label in (
        ("prime_observer", "Prime Observer artifact updated"),
        ("core_signal", "Core Signal event present"),
    ):
        source = _first_source(sources, source_id)
        if not source:
            continue
        detail = str(
            source.get("observed_at")
            or source.get("generated_at")
            or source.get("freshness")
            or source.get("message")
            or source.get("status")
        )
        if detail:
            items.append({"label": label, "detail": detail})

    return _dedupe_activity(items)[:5]


def _network_signal(dashboard: dict[str, object]) -> dict[str, object]:
    network = [str(item) for item in _list_value(dashboard.get("network_status"))]
    events = _dict_list(dashboard.get("core_signal_events"))
    thought = _metadata_mapping(dashboard.get("what_we_think"))
    latest_sample = _first_line_value(network, ("Latest sample timestamp",))
    current_state = _first_line_value(
        network,
        (
            "Current LAN/WAN state",
            "Current network state",
            "Current status",
            "Status",
        ),
    )
    attribution = (
        thought.get("value")
        or _metadata_mapping(dashboard.get("attribution_assessment")).get("value")
        or _first_event_value(events, "issue_location")
    )
    confidence = thought.get("confidence") or _first_event_value(events, "confidence")
    affected_window = _first_event_value(events, "affected_window")
    field_candidates = [
        {
            "label": "Status",
            "value": current_state or str(dashboard.get("current_status_label") or ""),
        },
        {"label": "Attribution", "value": attribution},
        {"label": "Confidence", "value": confidence},
        {"label": "Latest sample", "value": latest_sample},
        {"label": "Last incident", "value": affected_window},
        {
            "label": "Interpreted events",
            "value": f"{len(events)} interpreted event{'s' if len(events) != 1 else ''}",
        },
    ]
    return {
        "headline": attribution or current_state or "Network context",
        "summary": current_state
        or str(dashboard.get("status_explanation") or "No active condition reported."),
        "fields": [
            {"label": item["label"], "value": str(item["value"])}
            for item in field_candidates
            if str(item["value"] or "").strip()
        ],
        "event_count": len(events),
        "tone": str(dashboard.get("status_tone") or "healthy"),
    }


def _first_line_value(lines: list[str], labels: tuple[str, ...]) -> str:
    lowered_labels = tuple(label.lower() for label in labels)
    for line in lines:
        label, separator, value = line.partition(":")
        if separator and label.strip().lower() in lowered_labels:
            return value.strip()
    return ""


def _first_event_value(events: list[dict[str, object]], key: str) -> str:
    for event in events:
        value = str(event.get(key) or "").strip()
        if value:
            return value
    return ""


def _source_panel_tone(sources: list[dict[str, object]]) -> str:
    statuses = {str(source.get("status") or "") for source in sources}
    if "error" in statuses:
        return "action"
    if statuses.intersection({"unavailable", "disabled"}):
        return "watch"
    return "healthy"


def _source_panel_summary(sources: list[dict[str, object]]) -> str:
    if not sources:
        return "No registered source statuses are available."
    ok_count = sum(1 for source in sources if source.get("status") == "ok")
    unavailable = [source for source in sources if source.get("status") != "ok"]
    if unavailable:
        label = str(unavailable[0].get("source_name") or unavailable[0].get("source_id"))
        status = str(unavailable[0].get("status") or "unavailable")
        return f"{ok_count} sources fresh; {label} is {status}."
    return f"All {ok_count} normalized sources are fresh."


def _first_source(
    sources: list[dict[str, object]],
    source_id: str,
) -> dict[str, object]:
    for source in sources:
        if source.get("source_id") == source_id:
            return source
    return {}


def _dedupe_activity(items: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item["label"], item["detail"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


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
    attribution_assessment = _first_metadata_mapping(
        explanation,
        events,
        "attribution_assessment",
    )
    evidence_strength = _first_metadata_mapping(explanation, events, "evidence_strength")

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
            "what_we_know": _what_we_know(
                network=network,
                core=core,
                explanation=explanation,
            ),
            "what_we_think": _what_we_think(
                explanation=explanation,
                attribution_assessment=attribution_assessment,
            ),
            "attribution_assessment": attribution_assessment,
            "evidence_strength": evidence_strength,
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


def _weather_context(source_aggregate: dict[str, object]) -> dict[str, str]:
    for source in _dict_list(source_aggregate.get("sources")):
        if source.get("source_id") != "weather" or source.get("status") != "ok":
            continue
        for item in _dict_list(source.get("summary_items")):
            summary = str(item.get("summary") or "").strip()
            if not summary:
                continue
            return {
                "summary": summary,
                "freshness": str(source.get("freshness") or ""),
                "observed_at": str(source.get("observed_at") or ""),
            }
    return {}


def _normalize_briefing_dashboard(dashboard: dict[str, object]) -> dict[str, object]:
    list_keys = (
        "what_matters",
        "what_we_know",
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
    explanation["uncertainties"] = [
        str(item) for item in _list_value(explanation.get("uncertainties"))
    ]
    explanation["attribution_assessment"] = _metadata_mapping(
        explanation.get("attribution_assessment")
    )
    explanation["evidence_strength"] = _metadata_mapping(
        explanation.get("evidence_strength")
    )
    dashboard["core_signal_explanation"] = explanation

    events = []
    for event in _dict_list(dashboard.get("core_signal_events")):
        event["supporting_facts"] = _dict_list(event.get("supporting_facts"))
        event["recommendation_trace"] = _dict_list(event.get("recommendation_trace"))
        event["uncertainties"] = [
            str(item) for item in _list_value(event.get("uncertainties"))
        ]
        event["attribution_assessment"] = _metadata_mapping(
            event.get("attribution_assessment")
        )
        event["evidence_strength"] = _metadata_mapping(event.get("evidence_strength"))
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
    dashboard["what_we_think"] = _metadata_mapping(dashboard.get("what_we_think"))
    dashboard["attribution_assessment"] = _metadata_mapping(
        dashboard.get("attribution_assessment")
    )
    dashboard["evidence_strength"] = _metadata_mapping(
        dashboard.get("evidence_strength")
    )

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


def _metadata_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("value", "confidence", "reason"):
        item = str(value.get(key) or "").strip()
        if item:
            result[key] = item
    return result


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


def _what_we_know(
    *,
    network: list[str],
    core: list[str],
    explanation: dict[str, object],
) -> list[str]:
    candidates = [
        *network[:3],
        *core[:2],
    ]
    for fact in _dict_list(explanation.get("supporting_facts")):
        summary = str(fact.get("summary") or "").strip()
        if summary:
            candidates.append(summary)

    deduped: list[str] = []
    for item in candidates:
        cleaned = _clean_briefing_line(item)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped[:5]


def _what_we_think(
    *,
    explanation: dict[str, object],
    attribution_assessment: dict[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("value", "confidence", "reason"):
        value = str(attribution_assessment.get(key) or "").strip()
        if value:
            result[key] = value
    confidence = str(explanation.get("confidence") or "").strip()
    if confidence and "confidence" not in result:
        result["confidence"] = confidence
    reason = str(
        explanation.get("confidence_reason") or explanation.get("why") or ""
    ).strip()
    if reason and "reason" not in result:
        result["reason"] = reason
    return result


def _first_metadata_mapping(
    explanation: dict[str, object],
    events: list[dict[str, object]],
    key: str,
) -> dict[str, str]:
    mapping = _metadata_mapping(explanation.get(key))
    if mapping:
        return mapping
    for event in events:
        mapping = _metadata_mapping(event.get(key))
        if mapping:
            return mapping
    return {}


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
            "evidence package."
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
        issue_location = str(event.get("issue_location") or "").strip()
        if issue_location:
            summary = f"{summary} Evidence suggests {issue_location}."
        confidence = str(event.get("confidence") or "").strip()
        if confidence:
            summary = f"{summary} Confidence: {confidence}."
        findings.append(summary)
    return findings[:3]


def _uncertainty_items(
    explanation: dict[str, object],
    events: list[dict[str, object]],
) -> list[str]:
    candidates = [str(item) for item in _list_value(explanation.get("uncertainties"))]
    for event in events:
        candidates.extend(str(item) for item in _list_value(event.get("uncertainties")))

    uncertain: list[str] = []
    for candidate in candidates:
        text = candidate.strip()
        if text and text not in uncertain:
            uncertain.append(text)
    return uncertain[:5]


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
        elif key == "uncertainty":
            uncertainties = current.setdefault("uncertainties", [])
            if isinstance(uncertainties, list) and value:
                uncertainties.append(value)
        elif key == "attribution assessment":
            _set_metadata_value(current, "attribution_assessment", "value", value)
        elif key == "attribution confidence":
            _set_metadata_value(
                current,
                "attribution_assessment",
                "confidence",
                value,
            )
        elif key == "attribution reason":
            _set_metadata_value(current, "attribution_assessment", "reason", value)
        elif key == "evidence strength":
            _set_metadata_value(current, "evidence_strength", "value", value)
        elif key == "evidence strength reason":
            _set_metadata_value(current, "evidence_strength", "reason", value)
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
        elif key == "uncertainty":
            uncertainties = explanation.setdefault("uncertainties", [])
            if isinstance(uncertainties, list) and value:
                uncertainties.append(value)
        elif key == "attribution assessment":
            _set_metadata_value(explanation, "attribution_assessment", "value", value)
        elif key == "attribution confidence":
            _set_metadata_value(
                explanation,
                "attribution_assessment",
                "confidence",
                value,
            )
        elif key == "attribution reason":
            _set_metadata_value(explanation, "attribution_assessment", "reason", value)
        elif key == "evidence strength":
            _set_metadata_value(explanation, "evidence_strength", "value", value)
        elif key == "evidence strength reason":
            _set_metadata_value(explanation, "evidence_strength", "reason", value)

    if not any(
        explanation.get(key)
        for key in (
            "confidence",
            "confidence_reason",
            "why",
            "supporting_fact_count",
            "supporting_facts",
            "recommendation_trace",
            "uncertainties",
            "attribution_assessment",
            "evidence_strength",
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
                        "uncertainties",
                        "attribution_assessment",
                        "evidence_strength",
                    )
                    if key in event
                }
                break

    explanation.pop("_trace_open", None)
    return explanation


def _set_metadata_value(
    item: dict[str, object],
    mapping_key: str,
    field: str,
    value: str,
) -> None:
    if not value:
        return
    mapping = item.setdefault(mapping_key, {})
    if isinstance(mapping, dict):
        mapping[field] = value


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
                    "Start the Prime Observer local server to open telemetry "
                    "evidence."
                )
            actions["primary"].append(
                {
                    "label": "Open Evidence Package",
                    "detail": "Prime Observer telemetry evidence",
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
    return _template_response(request, "health.html", {"health": report})


@app.get("/capabilities", response_class=HTMLResponse)
def capabilities_page(request: Request):
    return _template_response(
        request,
        "capabilities.html",
        {"identity": get_identity()},
    )


@app.get("/sources", response_class=HTMLResponse)
def sources_page(request: Request):
    report = SourceInspectionCapability().run(config=load_config())
    return _template_response(request, "sources.html", {"report": report})


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return _template_response(
        request,
        "settings.html",
        {"config": public_config(load_config())},
    )


@app.get("/config", response_class=HTMLResponse)
def config_page(request: Request):
    return _template_response(
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
