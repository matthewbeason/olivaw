from __future__ import annotations

from dataclasses import dataclass

from olivaw.config import OlivawConfig
from olivaw.health import run_health_checks
from olivaw.models import HealthReport


@dataclass
class HealthCapability:
    name: str = "health"
    description: str = "Run local and cloud provider health checks."

    def run(self, config: OlivawConfig | None = None) -> HealthReport:
        return run_health_checks(config)

