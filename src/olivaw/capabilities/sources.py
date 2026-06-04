from __future__ import annotations

from dataclasses import dataclass

from olivaw.sources.registry import SourceRegistry, inspect_sources


@dataclass
class SourceInspectionCapability:
    name: str = "source inspection"
    description: str = "Inspect registered structured knowledge sources."

    def run(self, registry: SourceRegistry | None = None) -> dict[str, object]:
        return inspect_sources(registry)


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
            for item in payload.get("items", []):
                lines.append(
                    "  - {title}: {summary}".format(
                        title=item.get("title", "Untitled"),
                        summary=item.get("summary", ""),
                    )
                )
    else:
        lines.append("- No source data available.")

    return "\n".join(lines)

