from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from olivaw.briefing import compose_briefing_from_file


@dataclass
class BriefingCapability:
    name: str = "briefing"
    description: str = "Render a deterministic personal intelligence briefing."

    def run(self, input_path: str | Path) -> str:
        return compose_briefing_from_file(input_path)

