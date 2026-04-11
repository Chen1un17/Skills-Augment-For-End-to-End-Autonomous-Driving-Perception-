"""MCP-facing cloud service."""

from __future__ import annotations

from ad_cornercase.cloud.reflector import CloudReflector
from ad_cornercase.providers.base import EmbeddingProvider
from ad_cornercase.schemas.reflection import ReflectionRequest, ReflectionResult
from ad_cornercase.schemas.skill import SkillBundle, SkillMatchRequest, SkillMatchResult
from ad_cornercase.skill_store.repository import SkillRepository
from ad_cornercase.skill_store.matcher import SkillMatcher


class CloudReflectionService:
    def __init__(
        self,
        *,
        repository: SkillRepository,
        matcher: SkillMatcher,
        reflector: CloudReflector,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._repository = repository
        self._matcher = matcher
        self._reflector = reflector
        self._embedding_provider = embedding_provider
        self._skill_index_synced = False

    async def _sync_skill_index(self) -> None:
        if self._skill_index_synced:
            return
        manifests = self._repository.list_manifests()
        if not manifests:
            self._skill_index_synced = True
            return
        embeddings = await self._embedding_provider.embed([manifest.trigger_embedding_text for manifest in manifests])
        self._repository.replace_index(
            {
                manifest.skill_id: embedding
                for manifest, embedding in zip(manifests, embeddings, strict=False)
            }
        )
        self._skill_index_synced = True

    async def match_skills(self, payload: dict) -> SkillMatchResult:
        await self._sync_skill_index()
        request = SkillMatchRequest.model_validate(payload)
        return await self._matcher.match(request)

    async def reflect_anomaly(self, payload: dict) -> ReflectionResult:
        await self._sync_skill_index()
        request = ReflectionRequest.model_validate(payload)
        result = await self._reflector.reflect(request)
        if result.should_persist_skill and result.new_skill and result.skill_markdown:
            embedding = (await self._embedding_provider.embed([result.new_skill.trigger_embedding_text]))[0]
            self._repository.save_bundle(
                SkillBundle(manifest=result.new_skill, skill_markdown=result.skill_markdown),
                embedding=embedding,
            )
            self._skill_index_synced = True
        return result

    def read_skill(self, skill_id: str) -> SkillBundle:
        return self._repository.get_bundle(skill_id)
