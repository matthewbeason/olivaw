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

Then open `http://127.0.0.1:8000`.

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
- `/health` shows local/cloud provider and configuration status.
- `/settings` shows read-only configuration without exposing secrets.

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

OpenAI support is present as an optional cloud provider. It is disabled by
default and only considered when `OLIVAW_CLOUD_ENABLED=true` or equivalent TOML
configuration is set. Cloud fallback remains disabled unless explicitly enabled
with `OLIVAW_CLOUD_FALLBACK=enabled`.

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
