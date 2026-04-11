"""Prompt loading and rendering."""

from __future__ import annotations

from pathlib import Path


class PromptRenderer:
    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir

    def load(self, name: str) -> str:
        path = self._prompts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    def render(self, name: str, **replacements: str) -> str:
        rendered = self.load(name)
        for key, value in replacements.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", value)
        return rendered
