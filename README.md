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
- Source inspection
- File inspection
- Source-aware response attribution
- Source-backed briefing generation
- PrimeObserverSource
- CoreSignalSource

These roadmap capabilities are not implemented yet:

- Persistent memory
- Calendar integration
- Email integration
- Notifications/reminders
- Weather lookup
- Local business lookup
- WeatherSource
- CalendarSource
- EmailSource
- Source aggregation
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

For LAN access from another device, bind to all local interfaces:

```bash
olivaw web --host 0.0.0.0 --port 8765
```

Then open `http://home:8765` from another device on the same network, assuming
`home` resolves to this Mac. LAN mode exposes Olivaw to devices on your local
network, so keep `127.0.0.1` for localhost-only use.

## Configuration

Olivaw loads configuration in this order:

1. Environment variables
2. User config at `~/Library/Application Support/Olivaw/config.toml`
3. Optional local `./olivaw.toml` for development
4. Built-in defaults

Environment variables always win. This lets terminal sessions override settings
temporarily while the LaunchAgent and terminal both share the same persistent
user config file by default.

Create the user config file from the checked-in template:

```bash
olivaw init-config
```

Example `~/Library/Application Support/Olivaw/config.toml`:

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

[sources.files]
directory = "~/Library/Application Support/Olivaw/data"
max_bytes = 1048576

[sources.prime_observer]
directory = "~/prime-observer/viz"
enabled = true

[sources.core_signal]
directory = "~/core-signal/reports"
enabled = true

[secrets]
openai_api_key = ""
```

Use [config.example.toml](/Users/mbeason/olivaw/config.example.toml) as a
starter, but do not commit a real key. Secrets belong only in the user config
file or environment variables.

Check the active redacted configuration:

```bash
olivaw config
```

Environment overrides:

- `OLIVAW_CONFIG`
- `OLIVAW_LOCAL_BASE_URL`
- `OLIVAW_LOCAL_MODEL`
- `OLIVAW_CLOUD_ENABLED`
- `OLIVAW_CLOUD_MODEL`
- `OLIVAW_CLOUD_FALLBACK`
- `OLIVAW_PRIME_OBSERVER_DIR`
- `OLIVAW_PRIME_OBSERVER_ENABLED`
- `OLIVAW_CORE_SIGNAL_DIR`
- `OLIVAW_CORE_SIGNAL_ENABLED`
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
olivaw brief-sources
olivaw chat
olivaw sources
olivaw config
olivaw init-config
olivaw init-data
olivaw web
```

`olivaw chat` is a placeholder that routes through the configured provider when
available. It remains conservative in v0 and reports provider health when a
model is unavailable.

## Sources Vs Models

Olivaw separates source-backed information from model reasoning. This is the
foundation for trustworthy assistant responses as more local sources are added.

Responses can carry one of three internal attribution states:

- Source-backed: the answer came from registered source or capability metadata,
  such as the source registry or identity/capability registry.
- Model-reasoned: the answer is explanation, summarization, synthesis, or other
  reasoning generated by the selected provider.
- Capability unavailable: the request needs a source or tool Olivaw does not
  currently implement.

This attribution exists so Olivaw does not imply access to facts, tools, or
integrations that are not present. For example, a weather request returns a
friendly message that no weather source is configured instead of asking a model
to guess. Questions like "Explain local-first architecture" still route to the
configured model provider as model-reasoned responses.

Model providers are not tools. OpenAI or Ollama can produce text, but they do
not give Olivaw live weather, web search, calendar, email, news, sports, or
stock-price access. Until a source such as WeatherSource exists, both CLI chat
and web chat should answer weather requests with a capability-unavailable
message.

The current source-backed chat paths are intentionally small: capability
questions and source availability questions can be answered from Olivaw's own
metadata. Source-backed answers can cite registered sources such as
PrimeObserverSource and CoreSignalSource. Future source-backed answers can cite
sources such as WeatherSource, CalendarSource, and EmailSource once those
sources exist. Those sources are roadmap items only today.

## Web UI

The web UI uses FastAPI and Jinja templates. Routes:

- `/` shows assistant status, selected provider, and an example briefing.
- `/chat` provides a minimal placeholder chat surface.
- `/briefing` shows a deterministic source-backed briefing.
- `/sources` shows registered sources, source status, and example source data.
- `/capabilities` shows implemented capabilities, roadmap capabilities, and
  operating principles.
- `/health` shows local/cloud provider and configuration status.
- `/config` shows redacted configuration source and provider settings.
- `/settings` shows read-only configuration without exposing secrets.

By default, `olivaw web` listens on `127.0.0.1:8765`:

```bash
olivaw web --host 127.0.0.1 --port 8765
```

For LAN access, bind to `0.0.0.0`:

```bash
olivaw web --host 0.0.0.0 --port 8765
```

Then open `http://home:8765` from another device on the same LAN, assuming
`home` resolves to this Mac. LAN mode exposes Olivaw to your local network.

Olivaw uses port `8765` instead of `8000` so it does not collide with Prime
Observer or other local development services that commonly use port `8000`.

## Sources

Olivaw has a lightweight Sources framework so integrations can expose structured
information before the project adds memory. A source has an id, display name,
health status, and `fetch()` method.

v0 includes four local sources:

```bash
olivaw init-data
olivaw sources
```

The manual source returns deterministic fixture data:

```json
{
  "source": "manual",
  "status": "ok",
  "items": [
    {
      "title": "Example item",
      "summary": "Demonstrates source plumbing."
    }
  ]
}
```

The file source reads structured local files from:

```text
~/Library/Application Support/Olivaw/data/
```

`olivaw init-data` creates:

```text
data/
  notes/
  reports/
  status/
```

FileSource is read-only. It does not index, embed, store memory, or use a vector
database. It inspects `.txt`, `.md`, and `.json` files, ignores hidden files,
ignores unsupported/binary file types, and skips files larger than the configured
limit, which defaults to 1 MB.

PrimeObserverSource reads Prime Observer outputs without modifying Prime
Observer. Prime Observer remains the system of record for current and
near-real-time observed state; Olivaw consumes and presents those facts without
adding interpretation.

The default directory is:

```text
~/prime-observer/viz/
```

Expected files include:

```text
viz/
  latest.csv
  network_attribution.json
  nextdns_summary.json
```

Structured JSON is preferred when available. `network_attribution.json` is used
for current LAN/WAN state and attribution status. `latest.csv` is used for the
latest sample timestamp and raw p95/loss facts. `nextdns_summary.json` is used
for DNS summary availability and raw DNS counts, plus top blocked/resolved
domains when Prime Observer exports them. Redacted DNS entities are labeled as
redacted; Olivaw does not treat `entity_1`-style placeholders as meaningful
briefing facts. Markdown and text reports are read as short previews. Missing
directories, empty directories, disabled configuration, and malformed files
degrade through source health/status instead of crashing.

CoreSignalSource reads Core Signal outputs without modifying Core Signal. Core
Signal remains the authoritative interpretation layer; Olivaw consumes and
presents interpretations Core Signal has already produced.

The default directory is:

```text
~/core-signal/reports/
```

Expected files include:

```text
reports/
  latest.md
  morning-brief-YYYY-MM-DD.md
  patterns/latest.md
  latest.json
```

Structured JSON is preferred when available. Markdown morning briefs are used
for status, status reasoning, recommended action, and "Worth knowing" findings,
including DNS interpretation when Core Signal provides it. Pattern reports are
read as concise interpretation summaries and noteworthy pattern titles. Missing
directories, empty directories, disabled configuration, and malformed files
degrade through source health/status instead of crashing.

Prime Observer and Core Signal stay separate:

- Prime Observer answers: "What happened?"
- Core Signal answers: "What does it mean?"
- Olivaw answers: "What should I know?"

Source-backed briefings preserve that separation. The Prime Observer section is
current-state focused: latest sample timestamp, current LAN/WAN state, current
network attribution/status, DNS summary availability, and raw DNS facts. The
Core Signal section owns status reasoning, recommendations, worth-knowing
interpretation, trends, patterns, and DNS interpretation.

WeatherSource, CalendarSource, EmailSource, and source aggregation are roadmap
items only. They are not integrated yet.

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

The checked-in plist uses localhost-only mode, binding to `127.0.0.1`. Install
and start the LaunchAgent in localhost-only mode:

```bash
scripts/install_launch_agent.sh
```

Install and start it in LAN mode:

```bash
scripts/install_launch_agent.sh --lan
```

This writes an installed plist that runs:

```bash
/Users/mbeason/olivaw/.venv/bin/olivaw web --host 0.0.0.0 --port 8765
```

In LAN mode, the local URL remains `http://127.0.0.1:8765`, and another device
on the same network can use `http://home:8765` when `home` resolves to this Mac.
LAN mode exposes Olivaw to devices on your local network.

Custom host and port are also supported:

```bash
scripts/install_launch_agent.sh --host 0.0.0.0 --port 8765
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
  sources/        Structured source interface, registry, manual/file/operational sources
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

Olivaw has two deterministic briefing paths.

The fixture briefing path reads structured JSON and renders Markdown:

```bash
olivaw brief --input examples/daily_context.json
```

This path is useful for fixtures, tests, and known daily context payloads.

The source-backed briefing path fetches registered sources and renders a
grounded Markdown briefing:

```bash
olivaw brief-sources
```

It currently uses ManualSource, FileSource, PrimeObserverSource, and
CoreSignalSource. The output includes source status, source-backed highlights,
file previews, a current-state Prime Observer section, an interpretation-focused
Core Signal section, source notes, and attribution such as:

```text
This briefing is source-backed using: manual, files, prime_observer, core_signal.
```

Source-backed briefings do not require Ollama, OpenAI, embeddings, memory, or a
vector database. They are deterministic because they only transform current
source payloads into Markdown. Future versions may add optional model-enhanced
synthesis on top of source-backed facts, but the source attribution contract
should remain visible internally.

## Roadmap

Future versions may add:

- Persistent memory and context stores
- Background scheduling and daily briefing generation
- Notifications and reminders
- Project monitoring
- Email, calendar, and contact integrations
- Tool execution and multi-step task planning
- Always-on assistant service support
- LaunchAgent or system service installation
- Desktop automation and computer-use style interactions

These are documented as direction only. v0 does not implement autonomous task
execution, memory, governance-system modification, or any modification of Prime
Observer or Core Signal.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

The tests mock external providers and are expected to pass without network
access, Ollama, or OpenAI credentials.
