from __future__ import annotations

from olivaw.briefing.schemas import DailyContext


def render_markdown(context: DailyContext) -> str:
    lines = [
        "# Daily Briefing",
        "",
        f"Date: {context.date}",
        "",
        "## Focus",
        context.focus,
        "",
    ]

    if context.summary:
        lines.extend(["## Summary", context.summary, ""])

    lines.append("## Priorities")
    if context.priorities:
        for index, priority in enumerate(context.priorities, start=1):
            lines.append(
                f"{index}. {priority.title} [{priority.status}] - {priority.why}"
            )
    else:
        lines.append("- No priorities provided.")
    lines.append("")

    lines.append("## Signals")
    if context.signals:
        for signal in context.signals:
            lines.append(f"- {signal.title} ({signal.source}): {signal.detail}")
    else:
        lines.append("- No signals provided.")
    lines.append("")

    lines.append("## Projects")
    if context.projects:
        for project in context.projects:
            lines.append(
                f"- {project.name}: {project.state}. Next: {project.next_step}"
            )
    else:
        lines.append("- No project state provided.")
    lines.append("")

    lines.append("## Reminders")
    if context.reminders:
        lines.extend(f"- {reminder}" for reminder in context.reminders)
    else:
        lines.append("- No reminders provided.")

    return "\n".join(lines).strip() + "\n"

