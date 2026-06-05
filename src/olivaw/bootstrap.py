from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from olivaw.config import default_user_config_path, default_user_data_path

CONFIG_TEMPLATE = """[providers.local]
type = "ollama"
base_url = "http://localhost:11434"
model = "llama3.1:8b"

[providers.cloud]
type = "openai"
enabled = false
model = "gpt-4.1-mini"

[policy]
cloud_fallback = "disabled"

[sources.files]
directory = "~/Library/Application Support/Olivaw/data"
max_bytes = 1048576

[sources.prime_observer]
directory = "~/prime-observer/viz"
enabled = true

[secrets]
# Replace with a real key in your user config file only:
# ~/Library/Application Support/Olivaw/config.toml
openai_api_key = ""
"""

EXAMPLE_FILES = {
    "notes/welcome.md": "# Welcome\n\nThis note demonstrates Olivaw's FileSource.\n",
    "reports/example.json": '{\n  "title": "Example report",\n  "summary": "Structured JSON can be inspected by FileSource."\n}\n',
    "status/system.txt": "System status\n\nOlivaw FileSource example data is available.\n",
}


@dataclass(frozen=True)
class InitResult:
    path: Path
    created: bool


def init_config(path: Path | None = None, template_path: Path | None = None) -> InitResult:
    destination = path or default_user_config_path()
    if destination.exists():
        return InitResult(path=destination, created=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_config_template_text(template_path), encoding="utf-8")
    return InitResult(path=destination, created=True)


def init_data(root: Path | None = None) -> InitResult:
    destination = root or default_user_data_path()
    destination.mkdir(parents=True, exist_ok=True)
    for directory in ("notes", "reports", "status"):
        (destination / directory).mkdir(parents=True, exist_ok=True)

    created_any = False
    for relative_path, content in EXAMPLE_FILES.items():
        file_path = destination / relative_path
        if file_path.exists():
            continue
        file_path.write_text(content, encoding="utf-8")
        created_any = True

    return InitResult(path=destination, created=created_any)


def _config_template_text(template_path: Path | None = None) -> str:
    path = template_path or Path(__file__).parents[2] / "config.example.toml"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return CONFIG_TEMPLATE
