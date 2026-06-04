from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Priority:
    title: str
    why: str
    status: str


@dataclass(frozen=True)
class Signal:
    source: str
    title: str
    detail: str


@dataclass(frozen=True)
class ProjectState:
    name: str
    state: str
    next_step: str


@dataclass(frozen=True)
class DailyContext:
    date: str
    focus: str
    summary: str
    priorities: list[Priority] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    projects: list[ProjectState] = field(default_factory=list)
    reminders: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DailyContext":
        return cls(
            date=str(data["date"]),
            focus=str(data["focus"]),
            summary=str(data.get("summary", "")),
            priorities=[
                Priority(
                    title=str(item["title"]),
                    why=str(item.get("why", "")),
                    status=str(item.get("status", "unknown")),
                )
                for item in data.get("priorities", [])
            ],
            signals=[
                Signal(
                    source=str(item.get("source", "unknown")),
                    title=str(item["title"]),
                    detail=str(item.get("detail", "")),
                )
                for item in data.get("signals", [])
            ],
            projects=[
                ProjectState(
                    name=str(item["name"]),
                    state=str(item.get("state", "unknown")),
                    next_step=str(item.get("next_step", "")),
                )
                for item in data.get("projects", [])
            ],
            reminders=[str(item) for item in data.get("reminders", [])],
        )

