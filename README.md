# Olivaw

Olivaw is a local-first personal assistant framework. The first v0 capability is
a deterministic personal intelligence briefing, but the repository is structured
for broader assistant behavior over time: conversation, memory, recurring tasks,
notifications, tool usage, project awareness, local reasoning, and optional cloud
escalation.

v0 is intentionally small. It establishes clean seams for capabilities,
providers, sources, health checks, renderers, and future services without
implementing always-on autonomy yet.

## Principles

- Local first: local providers are preferred by default.
- Explicit cloud use: cloud fallback is disabled unless configured.
- Graceful degradation: missing Ollama, missing API keys, and offline operation
  produce useful health messages instead of crashes.
- Testable core: tests pass without Ollama, OpenAI credentials, or internet.
- Extensible assistant: briefings are one capability, not the whole system.

## Identity And Capability Grounding

Olivaw answers as Olivaw, a local-first personal assistant framework named
after R. Daneel Olivaw from Isaac Asimov's fiction. Chat prompts include
structured identity and capability context so a local model does not overstate
what the framework can currently do.

Today, Olivaw should describe only these implemented capabilities:

- Deterministic briefing generation from structured input
- Provider health reporting
- Local Ollama provider access
- Cloud OpenAI provider support when explicitly enabled
- Provider routing
- CLI interface
- Lightweight web interface
- Read-only configuration display

These roadmap capabilities are not implemented yet:

- Persistent memory
- Calendar integration
- Email integration
- Notifications/reminders
- Weather lookup
- Local business lookup
- Prime Observer integration
- Core Signal integration
- Autonomous background tasks
- Tool execution
- Desktop automation

Capability grounding exists to keep local-model responses honest. If asked
about unavailable features, Olivaw should say they are not implemented yet and
distinguish roadmap direction from current functionality.

Example:

```bash
olivaw chat "What can you currently do?"
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
olivaw --help
```

Run health checks:

```bash
olivaw health
```

Generate the example briefing:

```bash
olivaw brief --input examples/daily_context.json
```

Start the local web app:

```bash
olivaw web
```

Then open `http://127.0.0.1:8765`.

## Configuration

Olivaw loads defaults, then optional TOML configuration, then environment
variables. Sensitive values should only be supplied through environment
variables.

Example `olivaw.toml`:

```toml
[providers.local]
type = "ollama"
base_url = "http://localhost:11434"
model = "llama3.1:8b"

[providers.cloud]
type = "openai"
enabled = false
model = "gpt-4.1-mini"

[policy]
cloud_fallback = "disabled"
```

Environment overrides:

- `OLIVAW_CONFIG`
- `OLIVAW_LOCAL_BASE_URL`
- `OLIVAW_LOCAL_MODEL`
- `OLIVAW_CLOUD_ENABLED`
- `OLIVAW_CLOUD_MODEL`
- `OLIVAW_CLOUD_FALLBACK`
- `OPENAI_API_KEY` or `OLIVAW_OPENAI_API_KEY`

Equivalent `.env` values:

```bash
OPENAI_API_KEY=
OLIVAW_CLOUD_ENABLED=false
OLIVAW_CLOUD_MODEL=gpt-4.1-mini
OLIVAW_CLOUD_FALLBACK=disabled
```

## CLI

```bash
olivaw health
olivaw brief --input examples/daily_context.json
olivaw chat
olivaw web
```

`olivaw chat` is a placeholder that routes through the configured provider when
available. It remains conservative in v0 and reports provider health when a
model is unavailable.

## Web UI

The web UI uses FastAPI and Jinja templates. Routes:

- `/` shows assistant status, selected provider, and an example briefing.
- `/chat` provides a minimal placeholder chat surface.
- `/capabilities` shows implemented capabilities, roadmap capabilities, and
  operating principles.
- `/health` shows local/cloud provider and configuration status.
- `/settings` shows read-only configuration without exposing secrets.

By default, `olivaw web` listens on `127.0.0.1:8765`:

```bash
olivaw web --host 127.0.0.1 --port 8765
```

Olivaw uses port `8765` instead of `8000` so it does not collide with Prime
Observer or other local development services that commonly use port `8000`.

## macOS LaunchAgent

Olivaw can run the web UI continuously in the background on macOS through
launchd. This keeps the local web interface available without leaving a terminal
window open. It is an always-on web UI only; v0 does not add memory, background
task scheduling, notifications, or autonomous work.

The LaunchAgent template is:

```text
deploy/com.beason.olivaw.plist
```

It runs:

```bash
/Users/mbeason/olivaw/.venv/bin/olivaw web --host 127.0.0.1 --port 8765
```

Install and start the LaunchAgent:

```bash
scripts/install_launch_agent.sh
```

Check status:

```bash
scripts/status_launch_agent.sh
```

Uninstall and stop the LaunchAgent:

```bash
scripts/uninstall_launch_agent.sh
```

Logs are written to:

```text
/Users/mbeason/Library/Logs/olivaw.log
/Users/mbeason/Library/Logs/olivaw-error.log
```

This launchd support is the first deployment step toward future always-on
assistant behavior: a persistent local web surface that can later host recurring
tasks, memory, notifications, and integrations when those capabilities are
implemented.

## Architecture

```text
src/olivaw/
  assistant/      Assistant orchestration and capability registry
  briefing/       Deterministic briefing schemas, composer, and renderer
  capabilities/   Assistant capabilities such as briefing, chat, and health
  providers/      Provider protocol, Ollama, OpenAI, and router
  services/       Future service extension points
  cli.py          CLI entrypoint
  config.py       Defaults, TOML, and environment overrides
  health.py       Health report composition
  models.py       Shared dataclasses
  web.py          FastAPI application
```

## Provider Behavior

Ollama is the default local provider at `http://localhost:11434`. If Ollama is
not installed or not running, health checks explain the expected endpoint and
suggest running:

```bash
ollama serve
```

OpenAI support is present as an optional cloud provider using the official
OpenAI Python SDK and the Responses API. It is disabled by default and only
considered when `OLIVAW_CLOUD_ENABLED=true` or equivalent TOML configuration is
set. The API key is read from `OPENAI_API_KEY` or `OLIVAW_OPENAI_API_KEY` and is
never shown in public configuration output.

Cloud fallback remains disabled unless explicitly enabled with:

```bash
OLIVAW_CLOUD_FALLBACK=enabled
```

Local routing remains preferred. If Ollama is available, `olivaw chat` uses the
local provider even when OpenAI is configured. If Ollama is unavailable, OpenAI
can be used only when the cloud provider is enabled, an API key is present, and
cloud fallback is explicitly enabled.

## Briefing Capability

The briefing capability reads structured JSON and renders deterministic
Markdown. It does not require an LLM. This makes the first capability reliable
in offline tests while leaving room for future model-assisted composition.

## Roadmap

Future versions may add:

- Persistent memory and context stores
- Background scheduling and daily briefing generation
- Notifications and reminders
- Project monitoring
- Prime Observer integration
- Core Signal integration
- Email, calendar, and contact integrations
- Tool execution and multi-step task planning
- Always-on assistant service support
- LaunchAgent or system service installation
- Desktop automation and computer-use style interactions

These are documented as direction only. v0 does not implement autonomous task
execution, memory, governance systems, Prime Observer, or Core Signal
integration.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

The tests mock external providers and are expected to pass without network
access, Ollama, or OpenAI credentials.
