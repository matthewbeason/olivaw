from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olivaw.sources.base import SourceHealth, SourcePayload

JSON_PRIORITY = ("latest.json", "summary.json", "status.json", "briefing.json")
SUPPORTED_EXTENSIONS = {".json", ".md", ".txt"}
PREVIEW_CHARS = 700


@dataclass(frozen=True)
class CoreSignalSource:
    directory: Path
    enabled: bool = True
    source_id: str = "core_signal"
    display_name: str = "Core Signal"

    def health(self) -> SourceHealth:
        if not self.enabled:
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="unavailable",
                message="CoreSignalSource is disabled in configuration.",
            )
        if not self.directory.exists():
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="unavailable",
                message=f"Directory does not exist: {self.directory}",
            )
        if not self.directory.is_dir():
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="error",
                message=f"Configured path is not a directory: {self.directory}",
            )

        reports = self._discover_reports()
        if not reports:
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="unavailable",
                message=f"No Core Signal report files found in {self.directory}",
            )
        return SourceHealth(
            source_id=self.source_id,
            display_name=self.display_name,
            status="ok",
            message=f"Found {len(reports)} Core Signal report file(s).",
        )

    def fetch(self) -> SourcePayload:
        health = self.health()
        if health.status != "ok":
            return self._payload(status=health.status, items=[])

        items: list[dict[str, object]] = []
        errors: list[str] = []
        reports = self._discover_reports()
        for path in reports:
            try:
                item = self._item_from_report(path)
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            if item:
                items.append(item)

        status = "ok" if items else "error"
        payload = self._payload(status=status, items=items)
        if errors:
            payload["errors"] = errors
        payload["diagnostics"] = self._diagnostics(reports, items, errors)
        return payload

    def _payload(self, status: str, items: list[dict[str, object]]) -> SourcePayload:
        return {
            "source": self.source_id,
            "status": status,
            "root": str(self.directory),
            "count": len(items),
            "items": items,
            "diagnostics": self._diagnostics([], items, []),
        }

    def _diagnostics(
        self,
        reports: list[Path],
        items: list[dict[str, object]],
        errors: list[str],
    ) -> dict[str, object]:
        selected = [str(path.relative_to(self.directory)) for path in reports]
        event_count = sum(
            len(events)
            for item in items
            for events in (item.get("events"),)
            if isinstance(events, list)
        )
        if event_count:
            event_status = f"{event_count} interpreted event(s) loaded"
        elif errors:
            event_status = "no interpreted events loaded; one or more reports failed"
        elif items:
            event_status = "no interpreted events found in selected reports"
        else:
            event_status = "no selected Core Signal reports produced items"
        return {
            "selection": (
                "Selected latest JSON files, recent JSON files, and latest "
                "markdown per category: "
                + ", ".join(selected)
                if selected
                else f"No supported Core Signal files found in {self.directory}"
            ),
            "interpreted_events": event_status,
        }

    def _discover_reports(self) -> list[Path]:
        files = [
            path
            for path in self.directory.rglob("*")
            if path.is_file()
            and not _is_hidden(path, self.directory)
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and path.name != ".gitkeep"
        ]
        by_name = {path.name: path for path in files}
        prioritized = [by_name[name] for name in JSON_PRIORITY if name in by_name]
        remaining_json = sorted(
            (
                path
                for path in files
                if path.suffix.lower() == ".json" and path not in prioritized
            ),
            key=_mtime,
            reverse=True,
        )
        markdown = sorted(
            (path for path in files if path.suffix.lower() in {".md", ".txt"}),
            key=lambda path: (_markdown_priority(path), _mtime(path)),
            reverse=True,
        )
        selected_markdown = _latest_by_category(markdown)
        return [*prioritized, *remaining_json[:3], *selected_markdown]

    def _item_from_report(self, path: Path) -> dict[str, object]:
        if path.suffix.lower() == ".json":
            return self._json_item(path)
        return self._markdown_item(path)

    def _json_item(self, path: Path) -> dict[str, object]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON report must be an object")

        title = str(data.get("title") or path.stem.replace("_", " ").title())
        status = str(data.get("status") or data.get("state") or "unknown")
        summary = str(
            data.get("summary")
            or data.get("interpretation")
            or data.get("description")
            or status
            or "Core Signal report is available."
        )
        status_reason = _first_present(
            data.get("status_reason"),
            data.get("why_status"),
            data.get("reason"),
            data.get("reasoning"),
        )
        recommended_action = _first_present(
            data.get("recommended_action"),
            data.get("recommendation"),
            data.get("action"),
        )
        findings = _coerce_findings(
            data.get("noteworthy_findings")
            or data.get("findings")
            or data.get("worth_knowing")
        )
        events = _coerce_events(data.get("events"))
        dns_findings = _coerce_findings(
            data.get("dns_findings")
            or data.get("dns_interpretation")
            or data.get("dns_summary")
            or _dict(data.get("dns")).get("findings")
            or _dict(data.get("dns")).get("interpretation")
        )
        return _base_item(
            path=path,
            title=title,
            summary=summary,
            report_date=str(data.get("date") or data.get("generated_at") or _modified(path)),
            status=status,
            status_reason=status_reason,
            recommended_action=recommended_action,
            findings=findings,
            events=events,
            dns_status=_first_present(
                data.get("dns_status"),
                _dict(data.get("dns")).get("status"),
            ),
            dns_meaning=_first_present(
                data.get("dns_meaning"),
                data.get("dns_interpretation"),
                _dict(data.get("dns")).get("meaning"),
                _dict(data.get("dns")).get("interpretation"),
            ),
            dns_recommended_action=_first_present(
                data.get("dns_recommended_action"),
                _dict(data.get("dns")).get("recommended_action"),
            ),
            dns_findings=dns_findings,
            confidence=_first_present(data.get("confidence")),
            confidence_reason=_first_present(data.get("confidence_reason")),
            supporting_facts=_coerce_supporting_facts(data.get("supporting_facts")),
            recommendation_trace=_coerce_recommendation_trace(
                data.get("recommendation_trace")
            ),
            interpretation_source=_first_present(data.get("interpretation_source")),
            related_events=_coerce_related_events(data.get("related_events")),
            preview=summary,
            report_type="json",
        )

    def _markdown_item(self, path: Path) -> dict[str, object]:
        text = path.read_text(encoding="utf-8", errors="replace")
        title = _title(text, path)
        sections = _sections(text)
        status = _line_value(text, "Status") or "unknown"
        summary = _summary(text, sections)
        status_reason = _section_text(sections, "Why This Status")
        recommended_action = _line_value(text, "Recommended Action") or _section_text(
            sections, "Recommendation"
        )
        findings = _worth_knowing(sections) or _pattern_titles(text)
        events = _markdown_events(
            path=path,
            title=title,
            summary=summary,
            status=status,
            status_reason=status_reason,
            recommended_action=recommended_action,
            sections=sections,
            text=text,
        )
        dns_findings = _dns_findings(text, sections)
        report_type = (
            "pattern_report"
            if "pattern" in path.name.lower() or "Pattern Report" in title
            else "morning_brief"
        )
        return _base_item(
            path=path,
            title=title,
            summary=summary,
            report_date=_date_from_text(title, path) or _modified(path),
            status=status,
            status_reason=status_reason,
            recommended_action=recommended_action,
            findings=findings,
            events=events,
            dns_findings=dns_findings,
            preview=_preview(text),
            report_type=report_type,
        )


def _base_item(
    *,
    path: Path,
    title: str,
    summary: str,
    report_date: str,
    status: str,
    status_reason: str | None = None,
    recommended_action: str | None = None,
    findings: list[str] | None = None,
    preview: str | None = None,
    report_type: str,
    **extra: object,
) -> dict[str, object]:
    item = {
        "title": title,
        "summary": summary,
        "path": path.name,
        "report_date": report_date,
        "status": status,
        "status_reason": status_reason or "",
        "recommended_action": recommended_action or "",
        "findings": findings or [],
        "preview": preview or summary,
        "report_type": report_type,
    }
    item.update(extra)
    return item


def _title(text: str, path: Path) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def _sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("-"):
            current = stripped.rstrip(":")
            sections.setdefault(current, [])
            continue
        if stripped.startswith("## "):
            current = stripped.removeprefix("## ").strip()
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(stripped)
    return sections


def _summary(text: str, sections: dict[str, list[str]]) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("Status:"):
            continue
        if stripped.endswith(":"):
            continue
        return stripped
    executive = _section_text(sections, "Executive Summary")
    return executive or "Core Signal report is available."


def _worth_knowing(sections: dict[str, list[str]]) -> list[str]:
    lines = sections.get("Worth knowing", [])
    findings = []
    for line in lines:
        if line.startswith("- "):
            findings.append(line.removeprefix("- ").strip())
    return findings[:5]


def _pattern_titles(text: str) -> list[str]:
    titles = []
    for line in text.splitlines():
        if line.startswith("### "):
            titles.append(line.removeprefix("### ").strip())
    return titles[:5]


def _dns_findings(text: str, sections: dict[str, list[str]]) -> list[str]:
    findings: list[str] = []
    for section_name in ("Worth knowing", "Concentration Signals"):
        for line in sections.get(section_name, []):
            cleaned = line.removeprefix("- ").removeprefix("### ").strip()
            if cleaned and _is_dns_related(cleaned):
                findings.append(cleaned)

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### ") and _is_dns_related(stripped):
            findings.append(stripped.removeprefix("### ").strip())

    deduped: list[str] = []
    for finding in findings:
        if finding not in deduped:
            deduped.append(finding)
    return deduped[:5]


def _is_dns_related(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered
        for term in (
            "dns",
            "domain",
            "quer",
            "resolved",
            "block",
            "encrypt",
            "concentration",
        )
    )


def _line_value(text: str, label: str) -> str | None:
    prefix = f"{label}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line.removeprefix(prefix).strip()
            return value or None
    return None


def _section_text(sections: dict[str, list[str]], name: str) -> str | None:
    lines = [
        line
        for line in sections.get(name, [])
        if line
        and not line.startswith("- ")
        and not _looks_like_inline_label(line)
    ]
    if not lines:
        return None
    return " ".join(lines).strip()


def _looks_like_inline_label(line: str) -> bool:
    label, separator, _value = line.partition(":")
    return bool(separator and label and len(label.split()) <= 4)


def _coerce_findings(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item][:5]
    if value:
        return [str(value)]
    return []


def _coerce_events(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    events: list[dict[str, object]] = []
    for raw in value:
        event = _event_from_mapping(_dict(raw))
        if event:
            events.append(event)
    return events[:5]


def _coerce_supporting_facts(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    facts: list[dict[str, str]] = []
    for raw in value:
        if isinstance(raw, dict):
            summary = _first_present(raw.get("summary"), raw.get("title"))
            source = _first_present(raw.get("source"), raw.get("attribution"))
            reference = _reference_label(raw.get("reference"))
        else:
            summary = str(raw) if raw else None
            source = None
            reference = None
        fact = {
            "summary": summary or "",
            "source": source or "",
            "reference": reference or "",
        }
        if any(fact.values()):
            facts.append(fact)
    return facts[:5]


def _coerce_recommendation_trace(value: object) -> list[dict[str, str]]:
    if isinstance(value, dict):
        trace = []
        for key, label in (
            ("recommendation", "Recommendation"),
            ("supporting_facts", "Supporting facts"),
            ("interpretation", "Interpretation"),
        ):
            text = _trace_value(value.get(key))
            if text:
                trace.append({"stage": label, "detail": text})
        if trace:
            return trace
    if isinstance(value, list):
        trace = []
        for raw in value:
            if isinstance(raw, dict):
                stage = _first_present(raw.get("stage"), raw.get("label"), raw.get("type"))
                detail = _first_present(
                    raw.get("detail"),
                    raw.get("summary"),
                    raw.get("value"),
                    raw.get("text"),
                )
            else:
                stage = None
                detail = str(raw) if raw else None
            if detail:
                trace.append({"stage": stage or "Trace", "detail": detail})
        return trace[:6]
    text = _trace_value(value)
    if text:
        return [{"stage": "Trace", "detail": text}]
    return []


def _coerce_related_events(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    related_events: list[dict[str, str]] = []
    for raw in value:
        if isinstance(raw, dict):
            event_id = _first_present(raw.get("id"), raw.get("event_id"))
            relationship = _first_present(
                raw.get("relationship"),
                raw.get("relationship_type"),
                raw.get("type"),
            )
            summary = _first_present(raw.get("summary"), raw.get("title"))
            reference = _reference_label(raw.get("reference"))
        else:
            event_id = str(raw) if raw else None
            relationship = None
            summary = None
            reference = None
        event = {
            "id": event_id or "",
            "relationship": relationship or "",
            "summary": summary or "",
            "reference": reference or "",
        }
        if any(event.values()):
            related_events.append(event)
    return related_events[:5]


def _event_from_mapping(data: dict[str, Any]) -> dict[str, object]:
    if not data:
        return {}
    window_start = _first_present(
        data.get("window_start"),
        data.get("affected_window_start"),
        _dict(data.get("affected_window")).get("start"),
    )
    window_end = _first_present(
        data.get("window_end"),
        data.get("affected_window_end"),
        _dict(data.get("affected_window")).get("end"),
    )
    reference = _dict(
        data.get("prime_observer_reference")
        or data.get("investigation_reference")
        or data.get("investigation")
    )
    investigation = _first_present(
        data.get("prime_observer_investigation"),
        data.get("investigation_url"),
        data.get("investigation_reference"),
        reference.get("url"),
        reference.get("path"),
    )
    evidence_window = _dict(data.get("evidence_window"))
    event: dict[str, object] = {
        "id": str(data.get("id") or data.get("event_id") or ""),
        "kind": str(
            data.get("kind") or data.get("type") or data.get("event_type") or ""
        ),
        "status": str(data.get("status") or ""),
        "severity": str(data.get("severity") or ""),
        "confidence": str(data.get("confidence") or ""),
        "window_start": window_start or "",
        "window_end": window_end or "",
        "summary": str(data.get("summary") or data.get("title") or ""),
        "why": str(
            data.get("why") or data.get("status_reason") or data.get("reason") or ""
        ),
        "recommended_action": str(
            data.get("recommended_action") or data.get("recommendation") or ""
        ),
        "confidence_reason": str(data.get("confidence_reason") or ""),
        "supporting_facts": _coerce_supporting_facts(data.get("supporting_facts")),
        "recommendation_trace": _coerce_recommendation_trace(
            data.get("recommendation_trace")
        ),
        "interpretation_source": str(data.get("interpretation_source") or ""),
        "related_events": _coerce_related_events(data.get("related_events")),
        "issue_location": str(data.get("issue_location") or data.get("location") or ""),
        "attribution_source": str(data.get("attribution_source") or ""),
        "prime_observer_investigation": investigation or "",
        "prime_observer_reference": reference,
        "evidence_window": evidence_window,
    }
    return {key: value for key, value in event.items() if value not in ("", {}, None)}


def _markdown_events(
    *,
    path: Path,
    title: str,
    summary: str,
    status: str,
    status_reason: str | None,
    recommended_action: str | None,
    sections: dict[str, list[str]],
    text: str,
) -> list[dict[str, object]]:
    technical = sections.get("Technical Evidence", [])
    investigation = _bullet_value(technical, "Prime Observer investigation")
    issue_location = _line_value(text, "Issue Location")
    attribution_source = _bullet_value(technical, "Attribution source")
    window = _bullet_value(technical, "Window")
    if not any((investigation, issue_location, attribution_source, window)):
        return []

    event: dict[str, object] = {
        "id": _line_value(text, "Event ID")
        or _line_value(text, "Event Id")
        or f"{path.stem}",
        "kind": _line_value(text, "Event Kind")
        or _line_value(text, "Event Type")
        or "",
        "status": status,
        "severity": _severity_from_status(status),
        "summary": summary,
        "why": status_reason or "",
        "recommended_action": recommended_action or "",
        "issue_location": issue_location or "",
        "attribution_source": attribution_source or "",
        "prime_observer_investigation": investigation or "",
        "evidence_window": {"label": window} if window else {},
    }
    if window:
        start, end = _split_window(window)
        if start:
            event["window_start"] = start
        if end:
            event["window_end"] = end
    return [{key: value for key, value in event.items() if value not in ("", {}, None)}]


def _bullet_value(lines: list[str], label: str) -> str | None:
    prefix = f"- {label}:"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            value = stripped.removeprefix(prefix).strip()
            return value or None
    return None


def _split_window(value: str) -> tuple[str | None, str | None]:
    start, separator, end = value.partition(" to ")
    if not separator:
        return value.strip() or None, None
    return start.strip() or None, end.strip() or None


def _severity_from_status(status: str) -> str:
    key = status.strip().lower()
    if key == "attention":
        return "attention"
    if key == "watch":
        return "watch"
    if key == "healthy":
        return "none"
    return key or "unknown"


def _first_present(*values: object) -> str | None:
    for value in values:
        if value:
            return str(value)
    return None


def _trace_value(value: object) -> str | None:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                part = _first_present(
                    item.get("summary"),
                    item.get("title"),
                    item.get("id"),
                    item.get("reference"),
                )
            else:
                part = str(item) if item else None
            if part:
                parts.append(part)
        return "; ".join(parts) or None
    if isinstance(value, dict):
        return _first_present(
            value.get("summary"),
            value.get("title"),
            value.get("id"),
            value.get("reference"),
        )
    if value:
        return str(value)
    return None


def _reference_label(value: object) -> str | None:
    if isinstance(value, dict):
        return _first_present(
            value.get("url"),
            value.get("path"),
            value.get("id"),
            value.get("label"),
        )
    if value:
        return str(value)
    return None


def _date_from_text(text: str, path: Path) -> str | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}", f"{text} {path.name}")
    return match.group(0) if match else None


def _preview(text: str) -> str:
    return " ".join(text.split())[:PREVIEW_CHARS].strip()


def _modified(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _markdown_priority(path: Path) -> int:
    name = path.name.lower()
    if name == "latest.md":
        return 4
    if "morning-brief" in name or "brief" in name:
        return 3
    if "pattern" in name:
        return 2
    if "status" in name:
        return 1
    return 0


def _latest_by_category(paths: list[Path]) -> list[Path]:
    selected: dict[str, Path] = {}
    for path in paths:
        category = _report_category(path)
        if category not in selected:
            selected[category] = path
    return [
        selected[category]
        for category in ("briefing", "pattern", "status", "report")
        if category in selected
    ]


def _report_category(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    stem = path.stem.lower()
    if "patterns" in parts or "pattern" in name:
        return "pattern"
    if "status" in name or stem == "status":
        return "status"
    if "brief" in name or name == "latest.md":
        return "briefing"
    return "report"


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)
