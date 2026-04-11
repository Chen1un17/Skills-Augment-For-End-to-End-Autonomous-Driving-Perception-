"""Skill manifest markdown rendering."""

from __future__ import annotations

from ad_cornercase.schemas.skill import SkillManifest


def build_skill_markdown(manifest: SkillManifest, reflection_summary: str) -> str:
    lines = [
        f"# {manifest.name}",
        "",
        f"- Skill ID: `{manifest.skill_id}`",
        f"- Source Case: `{manifest.source_case_id}`",
        f"- Focus Region: `{manifest.focus_region}`",
        f"- Fallback Label: `{manifest.fallback_label}`",
        "",
        "## Trigger Tags",
        ", ".join(manifest.trigger_tags) or "none",
        "",
        "## Dynamic Question Tree",
    ]
    lines.extend(f"- {item}" for item in manifest.dynamic_question_tree)
    lines.extend(
        [
            "",
            "## Output Constraints",
        ]
    )
    lines.extend(f"- {item}" for item in manifest.output_constraints)
    lines.extend(
        [
            "",
            "## Reflection Summary",
            reflection_summary,
            "",
        ]
    )
    return "\n".join(lines)
