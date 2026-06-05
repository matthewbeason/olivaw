from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olivaw.sources.base import SourceHealth, SourcePayload

JSON_PRIORITY = ("network_attribution.json", "nextdns_summary.json", "latest.json")
SUPPORTED_EXTENSIONS = {".json", ".md", ".txt", ".csv"}
PREVIEW_CHARS = 600


@dataclass(frozen=True)
class PrimeObserverSource:
    directory: Path
    enabled: bool = True
    source_id: str = "prime_observer"
    display_name: str = "Prime Observer"

    def health(self) -> SourceHealth:
        if not self.enabled:
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="unavailable",
                message="PrimeObserverSource is disabled in configuration.",
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
                message=f"No Prime Observer report files found in {self.directory}",
            )
        return SourceHealth(
            source_id=self.source_id,
            display_name=self.display_name,
            status="ok",
            message=f"Found {len(reports)} Prime Observer report file(s).",
        )

    def fetch(self) -> SourcePayload:
        health = self.health()
        if health.status != "ok":
            return self._payload(status=health.status, items=[])

        reports = self._discover_reports()
        items: list[dict[str, object]] = []
        errors: list[str] = []
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
            for path in self.directory.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        prioritized: list[Path] = []
        by_name = {path.name: path for path in files}
        for name in JSON_PRIORITY:
            if name in by_name:
                prioritized.append(by_name[name])

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
            key=_mtime,
            reverse=True,
        )
        latest_csv = [path for path in files if path.name == "latest.csv"]
        other_csv = sorted(
            (
                path
                for path in files
                if path.suffix.lower() == ".csv" and path.name != "latest.csv"
            ),
            key=_mtime,
            reverse=True,
        )
        csv_files = [*latest_csv, *other_csv]
        return [*prioritized, *remaining_json, *markdown, *csv_files]

    def _item_from_report(self, path: Path) -> dict[str, object]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._json_item(path)
        if suffix == ".csv":
            return self._csv_item(path)
        return self._text_item(path)

    def _json_item(self, path: Path) -> dict[str, object]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON report must be an object")

        if path.name == "network_attribution.json":
            return _network_attribution_item(path, data)
        if path.name == "nextdns_summary.json":
            return _nextdns_item(path, data)
        return _generic_json_item(path, data)

    def _csv_item(self, path: Path) -> dict[str, object]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return _base_item(
                path=path,
                title="Prime Observer CSV report",
                summary="CSV report is empty.",
                report_type="csv",
            )

        latest = rows[-1]
        ts = latest.get("ts") or _modified(path)
        phase = latest.get("phase_label") or "unknown phase"
        host = latest.get("host") or "unknown host"
        p95 = latest.get("p95_ms") or "unknown"
        loss = latest.get("loss_pct") or "unknown"
        return _base_item(
            path=path,
            title="Prime Observer latest samples",
            summary=(
                f"Latest sample {ts}: {phase} to {host}, "
                f"p95 {p95} ms, loss {loss}%."
            ),
            report_date=ts,
            status="ok",
            findings=[f"{len(rows)} sample row(s) in {path.name}."],
            report_type="csv",
        )

    def _text_item(self, path: Path) -> dict[str, object]:
        preview = _preview(path)
        return _base_item(
            path=path,
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            summary=preview or "Text report is empty.",
            report_date=_modified(path),
            status="ok",
            preview=preview,
            report_type=path.suffix.lower().lstrip("."),
        )


def _network_attribution_item(path: Path, data: dict[str, Any]) -> dict[str, object]:
    current = _dict(data.get("current_attribution"))
    window = _dict(data.get("window_attribution"))
    evidence = _dict(data.get("attribution_evidence"))
    summary = str(
        evidence.get("summary")
        or _first_text(current.get("evidence"))
        or data.get("attribution_label")
        or "Prime Observer network attribution is available."
    )
    status = str(
        current.get("status")
        or data.get("attribution_status")
        or "unknown"
    )
    confidence = str(
        current.get("confidence")
        or data.get("attribution_confidence")
        or "unknown"
    )
    findings = [
        f"Current: {current.get('label') or data.get('attribution_label') or status}",
        f"Confidence: {confidence}",
    ]
    if window:
        findings.append(f"Window: {window.get('label') or window.get('status')}")
    return _base_item(
        path=path,
        title="Prime Observer network attribution",
        summary=summary,
        report_date=str(data.get("generated_at") or _modified(path)),
        status=status,
        findings=findings,
        report_type="network_attribution",
    )


def _nextdns_item(path: Path, data: dict[str, Any]) -> dict[str, object]:
    summary = _dict(data.get("summary"))
    status = str(data.get("status") or "unknown")
    block_rate = summary.get("block_rate_pct")
    encrypted_rate = summary.get("encrypted_rate_pct")
    top_entities = summary.get("top_entities")
    findings = []
    if block_rate is not None:
        findings.append(f"Block rate: {block_rate}%")
    if encrypted_rate is not None:
        findings.append(f"Encrypted query rate: {encrypted_rate}%")
    if isinstance(top_entities, list) and top_entities:
        first = _dict(top_entities[0])
        label = first.get("label") or "top entity"
        share = first.get("share_of_total")
        findings.append(f"Top redacted entity: {label} ({share} share)")
    return _base_item(
        path=path,
        title="Prime Observer NextDNS summary",
        summary="NextDNS summary is available from Prime Observer.",
        report_date=str(data.get("generated_at") or _modified(path)),
        status=status,
        findings=findings,
        report_type="nextdns_summary",
    )


def _generic_json_item(path: Path, data: dict[str, Any]) -> dict[str, object]:
    title = str(data.get("title") or path.stem.replace("_", " ").title())
    summary = str(
        data.get("summary")
        or data.get("description")
        or data.get("status")
        or "Structured Prime Observer report is available."
    )
    return _base_item(
        path=path,
        title=title,
        summary=summary,
        report_date=str(data.get("date") or data.get("generated_at") or _modified(path)),
        status=str(data.get("status") or "ok"),
        report_type="json",
    )


def _base_item(
    *,
    path: Path,
    title: str,
    summary: str,
    report_date: str | None = None,
    status: str | None = None,
    findings: list[str] | None = None,
    preview: str | None = None,
    report_type: str,
) -> dict[str, object]:
    return {
        "title": title,
        "summary": summary,
        "path": path.name,
        "report_date": report_date or _modified(path),
        "status": status or "unknown",
        "findings": findings or [],
        "preview": preview or summary,
        "report_type": report_type,
    }


def _preview(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return " ".join(text.split())[:PREVIEW_CHARS].strip()


def _modified(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_text(value: object) -> str | None:
    if isinstance(value, list):
        for item in value:
            if item:
                return str(item)
    if value:
        return str(value)
    return None
