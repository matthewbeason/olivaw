from __future__ import annotations

from dataclasses import dataclass, field

from olivaw.assistant.capabilities import Capability


@dataclass
class Assistant:
    name: str = "Olivaw"
    capabilities: dict[str, Capability] = field(default_factory=dict)

    def register(self, capability: Capability) -> None:
        self.capabilities[capability.name] = capability

    def run(self, capability_name: str, **kwargs):
        capability = self.capabilities.get(capability_name)
        if capability is None:
            available = ", ".join(sorted(self.capabilities)) or "none"
            raise KeyError(
                f"Unknown capability: {capability_name}. Available: {available}"
            )
        return capability.run(**kwargs)


def create_default_assistant() -> Assistant:
    from olivaw.capabilities.briefing import BriefingCapability
    from olivaw.capabilities.chat import ChatCapability
    from olivaw.capabilities.health import HealthCapability

    assistant = Assistant()
    assistant.register(BriefingCapability())
    assistant.register(ChatCapability())
    assistant.register(HealthCapability())
    return assistant

