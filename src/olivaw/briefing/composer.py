from __future__ import annotations

import json
from pathlib import Path

from olivaw.briefing.renderer import render_markdown
from olivaw.briefing.schemas import DailyContext


def load_daily_context(path: str | Path) -> DailyContext:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Briefing input must be a JSON object.")
    return DailyContext.from_mapping(data)


def compose_briefing(context: DailyContext) -> str:
    return render_markdown(context)


def compose_briefing_from_file(path: str | Path) -> str:
    return compose_briefing(load_daily_context(path))

