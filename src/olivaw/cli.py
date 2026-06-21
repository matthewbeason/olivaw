from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from olivaw.assistant.prompts import build_chat_system_prompt
from olivaw.briefing import compose_briefing_from_file, compose_source_briefing
from olivaw.briefing.health_review import (
    HEALTH_REVIEW_SYSTEM_PROMPT,
    build_health_review_digest,
    build_health_review_prompt,
    format_health_review_diagnostic,
    generate_health_review,
)
from olivaw.capabilities.chat import ChatCapability
from olivaw.config import ConfigError, OlivawConfig, format_config_report, load_config
from olivaw.health import format_health_report, run_health_checks

DEFAULT_BENCHMARK_PROMPTS: tuple[tuple[str, str], ...] = (
    ("network", "How was the network overnight?"),
    ("weather", "What's the weather today?"),
    ("model_knowledge", "Who was Marcus Aurelius?"),
    ("operational", "What is disk usage?"),
    ("action", "Refresh the health review."),
)


@dataclass(frozen=True)
class BenchmarkModel:
    name: str
    size_bytes: int | None = None
    parameter_size: str = ""
    quantization_level: str = ""
    context_length: int | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    category: str
    prompt: str
    attempt: int
    text: str
    attribution: str
    sources: tuple[str, ...]
    capability: str
    provenance_label: str
    provenance_detail: str
    metrics: dict[str, object]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="olivaw",
        description="Local-first personal assistant framework.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Report provider and configuration health.")
    health_review = subparsers.add_parser(
        "health-review",
        help="Run a Health Review generation diagnostic.",
    )
    health_review.add_argument(
        "--model",
        help="Temporarily override the local model for this diagnostic run.",
    )
    health_review.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of diagnostic attempts to run.",
    )

    brief = subparsers.add_parser("brief", help="Generate a deterministic briefing.")
    brief.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a structured daily context JSON file.",
    )

    brief_sources = subparsers.add_parser(
        "brief-sources",
        help="Generate a deterministic source-backed briefing.",
    )
    brief_sources.add_argument(
        "--format",
        choices=("markdown",),
        default="markdown",
        help="Output format. Only markdown is currently supported.",
    )

    chat = subparsers.add_parser("chat", help="Run placeholder provider-routed chat.")
    chat.add_argument("prompt", nargs="?", default="Hello from Olivaw.")

    benchmark = subparsers.add_parser(
        "benchmark-local",
        help="Run local chat latency diagnostics against installed Ollama models.",
    )
    benchmark.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model to benchmark. May be passed more than once.",
    )
    benchmark.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Sequential attempts per prompt/model.",
    )

    subparsers.add_parser("sources", help="Inspect registered knowledge sources.")
    subparsers.add_parser("init-config", help="Create the user config file if missing.")
    subparsers.add_parser("init-data", help="Create the user data directory if missing.")

    web = subparsers.add_parser("web", help="Start the local web application.")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    subparsers.add_parser(
        "restart-web",
        help="Restart the macOS LaunchAgent that serves the local web application.",
    )

    subparsers.add_parser("config", help="Print non-secret effective configuration.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "health":
            print(format_health_report(run_health_checks()))
            return 0

        if args.command == "health-review":
            config = load_config()
            if args.model:
                config = replace(
                    config,
                    local=replace(config.local, model=args.model),
                )
            attempts = max(1, args.attempts)
            briefing = compose_source_briefing(config=config)
            from olivaw.web import _briefing_dashboard

            dashboard = _briefing_dashboard(
                briefing.text,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                briefing.sources,
                prime_observer_directory=config.prime_observer.directory,
                prime_observer_base_url=config.prime_observer.base_url,
            )
            for attempt in range(1, attempts + 1):
                if attempts > 1:
                    print(f"Attempt: {attempt}/{attempts}")
                print(
                    format_health_review_diagnostic(
                        generate_health_review(dashboard, config=config)
                    )
                )
                if attempt != attempts:
                    print()
            return 0

        if args.command == "brief":
            print(compose_briefing_from_file(args.input), end="")
            return 0

        if args.command == "brief-sources":
            print(compose_source_briefing().text, end="")
            return 0

        if args.command == "chat":
            print(ChatCapability().run(args.prompt))
            return 0

        if args.command == "benchmark-local":
            config = load_config()
            if args.models:
                models = tuple(dict.fromkeys(args.models))
            else:
                installed = tuple(model.name for model in installed_ollama_models(config))
                preferred = (config.local.model, "llama3.2:3b", "llama3.1:8b")
                models = tuple(
                    dict.fromkeys(
                        model
                        for model in (*preferred, *installed)
                        if model in installed
                    )
                )
            payload = benchmark_payload(
                config=config,
                models=models,
                attempts=max(1, args.attempts),
            )
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "sources":
            from olivaw.capabilities.sources import (
                SourceInspectionCapability,
                format_sources_report,
            )

            print(format_sources_report(SourceInspectionCapability().run()))
            return 0

        if args.command == "init-config":
            from olivaw.bootstrap import init_config

            result = init_config()
            action = "Created" if result.created else "Configuration already exists"
            print(f"{action}: {result.path}")
            return 0

        if args.command == "init-data":
            from olivaw.bootstrap import init_data

            result = init_data()
            action = "Created" if result.created else "Data directory already exists"
            print(f"{action}: {result.path}")
            return 0

        if args.command == "web":
            import uvicorn

            uvicorn.run("olivaw.web:app", host=args.host, port=args.port, reload=False)
            return 0

        if args.command == "restart-web":
            if sys.platform != "darwin":
                print("restart-web is only supported on macOS.", file=sys.stderr)
                return 2
            plist_path = Path.home() / "Library/LaunchAgents/com.beason.olivaw.plist"
            if not plist_path.exists():
                print(
                    "LaunchAgent not installed. Run scripts/install_launch_agent.sh first.",
                    file=sys.stderr,
                )
                return 2
            label = f"gui/{os.getuid()}/com.beason.olivaw"
            try:
                subprocess.run(
                    ["launchctl", "kickstart", "-k", label],
                    check=True,
                )
            except FileNotFoundError:
                print("launchctl is not available on this system.", file=sys.stderr)
                return 2
            except subprocess.CalledProcessError as exc:
                print(f"Failed to restart {label}: {exc}", file=sys.stderr)
                return 2
            print(f"Restarted {label}.")
            return 0

        if args.command == "config":
            print(format_config_report(load_config()))
            return 0
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


def installed_ollama_models(config: OlivawConfig) -> tuple[BenchmarkModel, ...]:
    url = f"{config.local.base_url.rstrip('/')}/api/tags"
    with urllib.request.urlopen(url, timeout=2.0) as response:
        data = json.loads(response.read().decode("utf-8"))
    models = data.get("models", [])
    if not isinstance(models, list):
        return ()
    results: list[BenchmarkModel] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        details = item.get("details")
        details = details if isinstance(details, dict) else {}
        name = str(item.get("model") or item.get("name") or "").strip()
        if not name:
            continue
        results.append(
            BenchmarkModel(
                name=name,
                size_bytes=_optional_int(item.get("size")),
                parameter_size=str(details.get("parameter_size") or ""),
                quantization_level=str(details.get("quantization_level") or ""),
                context_length=_optional_int(details.get("context_length")),
            )
        )
    return tuple(results)


def benchmark_chat_models(
    *,
    config: OlivawConfig,
    models: tuple[str, ...],
    attempts: int = 1,
    prompts: tuple[tuple[str, str], ...] = DEFAULT_BENCHMARK_PROMPTS,
) -> tuple[BenchmarkResult, ...]:
    results: list[BenchmarkResult] = []
    for model in models:
        model_config = replace(config, local=replace(config.local, model=model))
        for category, prompt in prompts:
            for attempt in range(1, max(1, attempts) + 1):
                started = time.perf_counter()
                response = ChatCapability().run_with_attribution(
                    prompt,
                    config=model_config,
                )
                metrics = dict(response.metrics)
                metrics.setdefault(
                    "total_request_duration_ms",
                    int((time.perf_counter() - started) * 1000),
                )
                results.append(
                    BenchmarkResult(
                        model=model,
                        category=category,
                        prompt=prompt,
                        attempt=attempt,
                        text=response.text,
                        attribution=str(response.attribution.value),
                        sources=response.sources,
                        capability=response.capability or "",
                        provenance_label=response.provenance_label,
                        provenance_detail=response.provenance_detail,
                        metrics=metrics,
                    )
                )
    return tuple(results)


def prompt_size_report(dashboard: dict[str, object] | None = None) -> dict[str, object]:
    health_prompt = ""
    if dashboard is not None:
        health_prompt = build_health_review_prompt(build_health_review_digest(dashboard))
    return {
        "chat_system_prompt": _size_summary(build_chat_system_prompt()),
        "health_review_system_prompt": _size_summary(HEALTH_REVIEW_SYSTEM_PROMPT),
        "health_review_user_prompt": _size_summary(health_prompt),
    }


def summarize_benchmark_results(
    results: tuple[BenchmarkResult, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault((result.model, result.category), []).append(result)

    rows: list[dict[str, object]] = []
    for (model, category), group in sorted(grouped.items()):
        totals = [
            int(item.metrics.get("total_request_duration_ms") or 0) for item in group
        ]
        ollama_totals = [
            int(item.metrics.get("ollama_total_duration_ms") or 0)
            for item in group
            if item.metrics.get("ollama_total_duration_ms") is not None
        ]
        rows.append(
            {
                "model": model,
                "category": category,
                "attempts": len(group),
                "avg_total_request_duration_ms": _mean_int(totals),
                "avg_ollama_total_duration_ms": _mean_int(ollama_totals),
                "model_invoked": any(
                    bool(item.metrics.get("model_invoked")) for item in group
                ),
                "attribution": group[-1].attribution,
                "provenance": group[-1].provenance_detail,
            }
        )
    return rows


def benchmark_payload(
    *,
    config: OlivawConfig,
    models: tuple[str, ...],
    attempts: int,
    dashboard: dict[str, object] | None = None,
) -> dict[str, object]:
    installed = installed_ollama_models(config)
    results = benchmark_chat_models(config=config, models=models, attempts=attempts)
    return {
        "config": {
            "base_url": config.local.base_url,
            "production_model": config.local.model,
        },
        "installed_models": [asdict(model) for model in installed],
        "prompt_sizes": prompt_size_report(dashboard),
        "results": [asdict(result) for result in results],
        "summary": summarize_benchmark_results(results),
    }


def _size_summary(text: str) -> dict[str, int]:
    return {
        "characters": len(text),
        "approx_tokens": int(len(text) / 4) if text else 0,
        "lines": len(text.splitlines()) if text else 0,
    }


def _mean_int(values: list[int]) -> int | None:
    if not values:
        return None
    return int(statistics.fmean(values))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
