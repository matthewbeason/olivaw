from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from olivaw.sources.base import SourceHealth, SourcePayload

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}
DEFAULT_MAX_BYTES = 1_048_576
PREVIEW_LINES = 4


@dataclass(frozen=True)
class FileSource:
    root: Path
    max_bytes: int = DEFAULT_MAX_BYTES
    source_id: str = "files"
    display_name: str = "Local files"

    def health(self) -> SourceHealth:
        if not self.root.exists():
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="unavailable",
                message=f"Directory does not exist: {self.root}",
            )
        if not self.root.is_dir():
            return SourceHealth(
                source_id=self.source_id,
                display_name=self.display_name,
                status="error",
                message=f"Configured path is not a directory: {self.root}",
            )
        return SourceHealth(
            source_id=self.source_id,
            display_name=self.display_name,
            status="ok",
            message=f"Scanning {self.root}",
        )

    def fetch(self) -> SourcePayload:
        health = self.health()
        if health.status != "ok":
            return {
                "source": self.source_id,
                "status": health.status,
                "root": str(self.root),
                "count": 0,
                "items": [],
            }

        items = [self._metadata(path) for path in self._iter_supported_files()]
        return {
            "source": self.source_id,
            "status": "ok",
            "root": str(self.root),
            "count": len(items),
            "items": items,
        }

    def _iter_supported_files(self) -> list[Path]:
        paths: list[Path] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if _is_hidden(path, self.root):
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if path.stat().st_size > self.max_bytes:
                continue
            paths.append(path)
        return sorted(paths, key=lambda item: item.relative_to(self.root).as_posix())

    def _metadata(self, path: Path) -> dict[str, object]:
        stat = path.stat()
        return {
            "path": path.relative_to(self.root).as_posix(),
            "title": path.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "preview": _preview(path),
        }


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _preview(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[:PREVIEW_LINES]).strip()
