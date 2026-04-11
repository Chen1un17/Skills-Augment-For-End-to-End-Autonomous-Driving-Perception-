"""File-backed skill repository."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from ad_cornercase.schemas.skill import SkillActionResult, SkillBundle, SkillManifest, SkillStatus, SkillUpdate


class SkillIndexEntry(BaseModel):
    skill_id: str
    embedding: list[float]
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class SkillRepository:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self._root / "index.json"

    def _load_index(self) -> dict[str, SkillIndexEntry]:
        if not self.index_path.exists():
            return {}
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        return {item["skill_id"]: SkillIndexEntry.model_validate(item) for item in data}

    def _save_index(self, entries: dict[str, SkillIndexEntry]) -> None:
        payload = [entry.model_dump(mode="json") for entry in entries.values()]
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def replace_index(self, embeddings_by_skill_id: dict[str, list[float]]) -> None:
        entries = {
            skill_id: SkillIndexEntry(skill_id=skill_id, embedding=embedding)
            for skill_id, embedding in embeddings_by_skill_id.items()
        }
        self._save_index(entries)

    def save_bundle(self, bundle: SkillBundle, embedding: list[float]) -> Path:
        skill_dir = self._root / bundle.manifest.skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "manifest.json").write_text(
            json.dumps(bundle.manifest.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        (skill_dir / "SKILL.md").write_text(bundle.skill_markdown, encoding="utf-8")
        entries = self._load_index()
        entries[bundle.manifest.skill_id] = SkillIndexEntry(skill_id=bundle.manifest.skill_id, embedding=embedding)
        self._save_index(entries)
        return skill_dir

    def list_manifests(self) -> list[SkillManifest]:
        manifests: list[SkillManifest] = []
        for manifest_path in sorted(self._root.glob("*/manifest.json")):
            manifests.append(SkillManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")))
        return manifests

    def list_index_entries(self) -> list[SkillIndexEntry]:
        return list(self._load_index().values())

    def get_bundle(self, skill_id: str) -> SkillBundle:
        skill_dir = self._root / skill_id
        manifest_path = skill_dir / "manifest.json"
        markdown_path = skill_dir / "SKILL.md"
        if not manifest_path.exists() or not markdown_path.exists():
            raise FileNotFoundError(f"Skill bundle not found: {skill_id}")
        manifest = SkillManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")
        return SkillBundle(manifest=manifest, skill_markdown=markdown)

    def update_skill(self, skill_id: str, bundle: SkillBundle, embedding: list[float]) -> SkillActionResult:
        """Update an existing skill, preserving history."""
        skill_dir = self._root / skill_id
        if not skill_dir.exists():
            return SkillActionResult(
                action="update_existing",
                skill_id=None,
                success=False,
                message=f"Skill not found: {skill_id}",
            )
        # Save history
        history_dir = skill_dir / "history"
        history_dir.mkdir(exist_ok=True)
        manifest_path = skill_dir / "manifest.json"
        existing_manifest = SkillManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        history_file = history_dir / f"{existing_manifest.version}.json"
        history_file.write_text(json.dumps(existing_manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
        # Update manifest
        bundle.manifest.version = self._bump_version(existing_manifest.version)
        bundle.manifest.parent_skill_id = skill_id
        manifest_path.write_text(json.dumps(bundle.manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
        (skill_dir / "SKILL.md").write_text(bundle.skill_markdown, encoding="utf-8")
        # Update index
        entries = self._load_index()
        entries[skill_id] = SkillIndexEntry(skill_id=skill_id, embedding=embedding)
        self._save_index(entries)
        return SkillActionResult(
            action="update_existing",
            skill_id=skill_id,
            success=True,
            message=f"Updated skill {skill_id} to version {bundle.manifest.version}",
        )

    def archive_skill(self, skill_id: str) -> SkillActionResult:
        """Archive a skill (soft delete)."""
        skill_dir = self._root / skill_id
        if not skill_dir.exists():
            return SkillActionResult(
                action="skip",
                skill_id=None,
                success=False,
                message=f"Skill not found: {skill_id}",
            )
        manifest_path = skill_dir / "manifest.json"
        manifest = SkillManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        manifest.status = SkillStatus.ARCHIVED
        manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
        return SkillActionResult(
            action="skip",
            skill_id=skill_id,
            success=True,
            message=f"Archived skill {skill_id}",
        )

    def merge_skills(
        self, target_id: str, source_ids: list[str], merged_bundle: SkillBundle, embedding: list[float]
    ) -> SkillActionResult:
        """Merge multiple skills into a target skill."""
        # Archive source skills
        for source_id in source_ids:
            if source_id != target_id:
                self.archive_skill(source_id)
        # Save merged skill
        skill_dir = self._root / target_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        merged_bundle.manifest.skill_id = target_id
        merged_bundle.manifest.status = SkillStatus.ACTIVE
        merged_bundle.manifest.family_id = target_id
        (skill_dir / "manifest.json").write_text(
            json.dumps(merged_bundle.manifest.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        (skill_dir / "SKILL.md").write_text(merged_bundle.skill_markdown, encoding="utf-8")
        # Update index
        entries = self._load_index()
        entries[target_id] = SkillIndexEntry(skill_id=target_id, embedding=embedding)
        self._save_index(entries)
        return SkillActionResult(
            action="merge_with",
            skill_id=target_id,
            success=True,
            message=f"Merged {len(source_ids)} skills into {target_id}",
        )

    def get_active_skills(self) -> list[SkillManifest]:
        """Get all active (non-archived, non-merged) skills."""
        manifests = self.list_manifests()
        return [m for m in manifests if m.status == SkillStatus.ACTIVE]

    def get_skills_by_family(self, family_id: str) -> list[SkillManifest]:
        """Get all skills in the same family."""
        manifests = self.list_manifests()
        return [m for m in manifests if m.family_id == family_id]

    def increment_usage(self, skill_id: str) -> None:
        """Increment usage count for a skill."""
        skill_dir = self._root / skill_id
        manifest_path = skill_dir / "manifest.json"
        if not manifest_path.exists():
            return
        manifest = SkillManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        manifest.usage_count += 1
        manifest.last_used_at = datetime.now(tz=timezone.utc)
        manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")

    @staticmethod
    def _bump_version(version: str) -> str:
        """Bump patch version: 0.1.0 -> 0.1.1"""
        parts = version.split(".")
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)
        return ".".join(parts)
