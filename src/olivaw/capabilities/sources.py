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
            root = payload.get("root")
            if root:
                lines.append(f"  Root: {root}")
            if "count" in payload:
                lines.append(f"  Count: {payload['count']}")
            diagnostics = payload.get("diagnostics")
            if isinstance(diagnostics, dict):
                for note in _diagnostic_lines(diagnostics):
                    lines.append(f"  {note}")
            errors = payload.get("errors")
            if isinstance(errors, list):
                for error in errors:
                    lines.append(f"  Load error: {error}")
            for item in payload.get("items", []):
                summary = item.get("summary") or item.get("preview", "")
                title = item.get("title") or item.get("path") or "Untitled"
                lines.append(
                    "  - {title}: {summary}".format(
                        title=title,
                        summary=summary,
                    )
                )
                lines.extend(_item_detail_lines(item, indent="    "))
    else:
        lines.append("- No source data available.")

    return "\n".join(lines)


def _diagnostic_lines(diagnostics: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for key, label in (
        ("selection", "Selection"),
        ("investigation_index", "Investigation index"),
        ("interpreted_events", "Interpreted events"),
    ):
        value = diagnostics.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return lines


def _item_detail_lines(item: object, *, indent: str) -> list[str]:
    if not isinstance(item, dict):
        return []
    report_type = item.get("report_type")
    lines: list[str] = []
    if report_type == "investigation_index":
        catalog = item.get("investigation_catalog")
        if isinstance(catalog, list) and catalog:
            lines.append(f"{indent}Investigation entries: {len(catalog)}")
            for entry in catalog[:5]:
                if not isinstance(entry, dict):
                    continue
                title = entry.get("title") or entry.get("id") or "Investigation"
                path = entry.get("path") or "no path"
                lines.append(f"{indent}- Investigation: {title} ({path})")
        else:
            lines.append(f"{indent}Investigation entries: 0")
    events = item.get("events")
    if isinstance(events, list):
        lines.append(f"{indent}Interpreted events: {len(events)}")
        for event in events[:5]:
            if not isinstance(event, dict):
                continue
            summary = event.get("summary") or event.get("id") or "Core Signal event"
            lines.append(f"{indent}- Event: {summary}")
            confidence = event.get("confidence")
            if confidence:
                lines.append(f"{indent}  Confidence: {confidence}")
            supporting_facts = event.get("supporting_facts")
            if isinstance(supporting_facts, list):
                lines.append(f"{indent}  Supporting facts: {len(supporting_facts)}")
            if event.get("confidence_reason"):
                lines.append(f"{indent}  Why: {event['confidence_reason']}")
    return lines
