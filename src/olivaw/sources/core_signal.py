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
        for path in self._discover_reports():
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
        return payload

    def _payload(self, status: str, items: list[dict[str, object]]) -> SourcePayload:
        return {
            "source": self.source_id,
            "status": status,
            "root": str(self.directory),
            "count": len(items),
            "items": items,
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
        return _base_item(
            path=path,
            title=title,
            summary=summary,
            report_date=str(data.get("date") or data.get("generated_at") or _modified(path)),
            status=status,
            recommended_action=recommended_action,
            findings=findings,
            preview=summary,
            report_type="json",
        )

    def _markdown_item(self, path: Path) -> dict[str, object]:
        text = path.read_text(encoding="utf-8", errors="replace")
        title = _title(text, path)
        sections = _sections(text)
        status = _line_value(text, "Status") or "unknown"
        summary = _summary(text, sections)
        recommended_action = _line_value(text, "Recommended Action") or _section_text(
            sections, "Recommendation"
        )
        findings = _worth_knowing(sections) or _pattern_titles(text)
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
            recommended_action=recommended_action,
            findings=findings,
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
    recommended_action: str | None = None,
    findings: list[str] | None = None,
    preview: str | None = None,
    report_type: str,
) -> dict[str, object]:
    return {
        "title": title,
        "summary": summary,
        "path": path.name,
        "report_date": report_date,
        "status": status,
        "recommended_action": recommended_action or "",
        "findings": findings or [],
        "preview": preview or summary,
        "report_type": report_type,
    }


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


def _line_value(text: str, label: str) -> str | None:
    prefix = f"{label}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line.removeprefix(prefix).strip()
            return value or None
    return None


def _section_text(sections: dict[str, list[str]], name: str) -> str | None:
    lines = [line for line in sections.get(name, []) if line and not line.startswith("- ")]
    if not lines:
        return None
    return " ".join(lines).strip()


def _coerce_findings(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item][:5]
    if value:
        return [str(value)]
    return []


def _first_present(*values: object) -> str | None:
    for value in values:
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
