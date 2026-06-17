from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from olivaw.briefing import compose_briefing_from_file, compose_source_briefing
from olivaw.briefing.health_review import (
    format_health_review_diagnostic,
    generate_health_review,
)
from olivaw.config import ConfigError, format_config_report, load_config
from olivaw.health import format_health_report, run_health_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="olivaw",
        description="Local-first personal assistant framework.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Report provider and configuration health.")
    subparsers.add_parser(
        "health-review",
        help="Run a Health Review generation diagnostic.",
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

    subparsers.add_parser("sources", help="Inspect registered knowledge sources.")
    subparsers.add_parser("init-config", help="Create the user config file if missing.")
    subparsers.add_parser("init-data", help="Create the user data directory if missing.")

    web = subparsers.add_parser("web", help="Start the local web application.")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)

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
            briefing = compose_source_briefing(config=config)
            from olivaw.web import _briefing_dashboard

            dashboard = _briefing_dashboard(
                briefing.text,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                briefing.sources,
                prime_observer_directory=config.prime_observer.directory,
                prime_observer_base_url=config.prime_observer.base_url,
            )
            print(
                format_health_review_diagnostic(
                    generate_health_review(dashboard, config=config)
                )
            )
            return 0

        if args.command == "brief":
            print(compose_briefing_from_file(args.input), end="")
            return 0

        if args.command == "brief-sources":
            print(compose_source_briefing().text, end="")
            return 0

        if args.command == "chat":
            from olivaw.capabilities.chat import ChatCapability

            print(ChatCapability().run(args.prompt))
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

        if args.command == "config":
            print(format_config_report(load_config()))
            return 0
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
