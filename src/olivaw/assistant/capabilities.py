from __future__ import annotations

from typing import Protocol


class Capability(Protocol):
    name: str
    description: str

    def run(self, **kwargs):
        """Execute the capability."""

