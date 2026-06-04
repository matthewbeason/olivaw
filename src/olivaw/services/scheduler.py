from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerService:
    enabled: bool = False

    def status(self) -> str:
        if self.enabled:
            return "Scheduler extension point is enabled, but v0 has no tasks."
        return "Scheduler extension point is present; background tasks are not implemented in v0."

