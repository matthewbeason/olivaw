from __future__ import annotations

from dataclasses import dataclass

from olivaw.sources.base import SourceHealth, SourcePayload


@dataclass(frozen=True)
class ManualSource:
    source_id: str = "manual"
    display_name: str = "Manual example source"

    def health(self) -> SourceHealth:
        return SourceHealth(
            source_id=self.source_id,
            display_name=self.display_name,
            status="ok",
            message="Manual source is available.",
        )

    def fetch(self) -> SourcePayload:
        return {
            "source": self.source_id,
            "status": "ok",
            "items": [
                {
                    "title": "Example item",
                    "summary": "Demonstrates source plumbing.",
                }
            ],
        }

