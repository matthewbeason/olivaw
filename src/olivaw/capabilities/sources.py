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

    aggregate = report.get("aggregate")
    normalized_sources = []
    if isinstance(aggregate, dict):
        raw_sources = aggregate.get("sources", [])
        if isinstance(raw_sources, list):
            normalized_sources = [
                source for source in raw_sources if isinstance(source, dict)
            ]
    if normalized_sources:
        lines.extend(["", "Normalized Sources:"])
        for source in normalized_sources:
            lines.append(
                "- {source_name} ({source_id}, {source_type}): {status}".format(
                    source_name=source.get("source_name", "Unknown source"),
                    source_id=source.get("source_id", "unknown"),
                    source_type=source.get("source_type", "unknown"),
                    status=source.get("status", "unknown"),
                )
            )
            if source.get("freshness"):
                lines.append(f"  Freshness: {source['freshness']}")
            if source.get("raw_available") is not None:
                raw = "yes" if source.get("raw_available") else "no"
                lines.append(f"  Raw available: {raw}")
            diagnostics = source.get("diagnostics")
            if isinstance(diagnostics, dict):
                for note in _diagnostic_lines(diagnostics):
                    lines.append(f"  {note}")

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
        ("configured_path", "Configured path"),
        ("base_url", "Prime Observer base URL"),
        ("investigate_http_url", "Prime Observer investigate URL"),
        ("investigate_http_status", "Prime Observer investigate HTTP"),
        ("investigation_links_enabled", "Investigation links enabled"),
        ("link_configuration_guidance", "Link configuration guidance"),
        ("selection", "Selection"),
        ("investigation_index", "Investigation index"),
        ("investigation_index_path", "Investigation index path"),
        ("investigation_index_status", "Investigation index status"),
        ("investigation", "Investigation export"),
        ("investigation_path", "Investigation export path"),
        ("investigation_status", "Investigation export status"),
        ("catalog_entry_count", "Catalog entries"),
        ("investigation_event_count", "Investigation events"),
        ("latest_investigation_timestamp", "Latest investigation timestamp"),
        ("investigation_index_modified", "Investigation index modified"),
        ("investigation_modified", "Investigation export modified"),
        ("investigation_index_generated_at", "Investigation index generated"),
        ("investigation_generated_at", "Investigation export generated"),
        ("interpreted_events", "Interpreted events"),
        ("event_objects_found", "Event objects found"),
        ("interpreted_events_rendered", "Interpreted events rendered"),
        ("latest_event_timestamp", "Latest event timestamp"),
        ("enabled", "Weather enabled"),
        ("configured", "Weather configured"),
        ("provider", "Weather provider"),
        ("provider_status", "Weather provider status"),
        ("location_name", "Weather location"),
        ("units", "Weather units"),
        ("last_fetch_status", "Weather last fetch"),
        ("forecast_date", "Weather forecast date"),
    ):
        value = diagnostics.get(key)
        if value not in (None, "", []):
            lines.append(f"{label}: {value}")
    generated = diagnostics.get("generated_timestamps")
    if isinstance(generated, list) and generated:
        lines.append("Generated timestamps: " + ", ".join(str(item) for item in generated))
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
            attribution = event.get("attribution_assessment")
            if isinstance(attribution, dict) and attribution.get("value"):
                lines.append(
                    f"{indent}  Attribution assessment: {attribution['value']}"
                )
            strength = event.get("evidence_strength")
            if isinstance(strength, dict) and strength.get("value"):
                lines.append(f"{indent}  Evidence strength: {strength['value']}")
            uncertainties = event.get("uncertainties")
            if isinstance(uncertainties, list):
                lines.append(f"{indent}  Uncertainties: {len(uncertainties)}")
    return lines
