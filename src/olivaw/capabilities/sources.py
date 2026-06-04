from __future__ import annotations

from dataclasses import dataclass

from olivaw.config import OlivawConfig
from olivaw.sources.registry import SourceRegistry, inspect_sources


@dataclass
class SourceInspectionCapability:
    name: str = "source inspection"
    description: str = "Inspect registered structured knowledge sources."

    def run(
        self,
        registry: SourceRegistry | None = None,
        config: OlivawConfig | None = None,
    ) -> dict[str, object]:
        return inspect_sources(registry=registry, config=config)


def format_sources_report(report: dict[str, object]) -> str:
    lines = ["Olivaw Sources", ""]

    sources = report.get("sources", [])
    lines.append("Registered Sources:")
    if sources:
        for source in sources:
            lines.append(
                "- {display_name} ({source_id}): {status} - {message}".format(
                    **source
                )
            )
    else:
        lines.append("- No sources registered.")

    lines.extend(["", "Sample Data:"])
    data = report.get("data", [])
    if data:
        for payload in data:
            lines.append(
                "- {source}: {status}".format(
                    source=payload.get("source", "unknown"),
                    status=payload.get("status", "unknown"),
                )
            )
            if "count" in payload:
                lines.append(f"  Count: {payload['count']}")
            for item in payload.get("items", []):
                summary = item.get("summary") or item.get("preview", "")
                title = item.get("title") or item.get("path") or "Untitled"
                lines.append(
                    "  - {title}: {summary}".format(
                        title=title,
                        summary=summary,
                    )
                )
    else:
        lines.append("- No source data available.")

    return "\n".join(lines)
