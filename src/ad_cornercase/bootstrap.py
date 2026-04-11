"""Factory helpers for runtime objects."""

from __future__ import annotations

from ad_cornercase.cloud.reflector import CloudReflector
from ad_cornercase.cloud.skill_compiler import SkillCompiler
from ad_cornercase.config import ProjectSettings, RuntimeSettings, get_project_settings, get_runtime_settings
from ad_cornercase.edge.agent import EdgeAgent
from ad_cornercase.mcp.tools import CloudReflectionService
from ad_cornercase.prompts.renderer import PromptRenderer
from ad_cornercase.providers.base import EmbeddingProvider, StructuredVisionProvider
from ad_cornercase.providers.embeddings import HashEmbeddingProvider, OpenAIEmbeddingProvider
from ad_cornercase.providers.judge import JudgeProvider
from ad_cornercase.providers.openai_responses import OpenAIResponsesVisionProvider
from ad_cornercase.skill_store.manager import SkillManager
from ad_cornercase.skill_store.matcher import SkillMatcher
from ad_cornercase.skill_store.repository import SkillRepository


def load_settings() -> tuple[RuntimeSettings, ProjectSettings]:
    return get_runtime_settings(), get_project_settings()


def build_prompt_renderer(runtime_settings: RuntimeSettings) -> PromptRenderer:
    return PromptRenderer(runtime_settings.prompts_dir)


def build_structured_provider(runtime_settings: RuntimeSettings) -> StructuredVisionProvider:
    if not runtime_settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the cloud-backed prototype.")
    return OpenAIResponsesVisionProvider(
        api_key=runtime_settings.openai_api_key,
        base_url=runtime_settings.openai_base_url,
        timeout=runtime_settings.request_timeout_seconds,
        max_retries=runtime_settings.max_retries,
    )


def build_embedding_provider(runtime_settings: RuntimeSettings) -> EmbeddingProvider:
    if not runtime_settings.openai_api_key:
        return HashEmbeddingProvider()
    return OpenAIEmbeddingProvider(
        api_key=runtime_settings.openai_api_key,
        base_url=runtime_settings.openai_base_url,
        model=runtime_settings.embedding_model,
        timeout=runtime_settings.request_timeout_seconds,
    )


def build_edge_agent(
    runtime_settings: RuntimeSettings,
    project_settings: ProjectSettings,
    *,
    model_override: str | None = None,
) -> EdgeAgent:
    effective_settings = runtime_settings
    if model_override is not None and model_override != runtime_settings.edge_model:
        effective_settings = runtime_settings.model_copy(update={"edge_model": model_override})
    return EdgeAgent(
        provider=build_structured_provider(effective_settings),
        prompt_renderer=build_prompt_renderer(effective_settings),
        runtime_settings=effective_settings,
        project_settings=project_settings,
    )


def build_cloud_service(runtime_settings: RuntimeSettings, project_settings: ProjectSettings) -> CloudReflectionService:
    prompt_renderer = build_prompt_renderer(runtime_settings)
    repository = SkillRepository(runtime_settings.skill_store_dir)
    embedding_provider = build_embedding_provider(runtime_settings)
    structured_provider = build_structured_provider(runtime_settings)
    matcher = SkillMatcher(
        repository=repository,
        embedding_provider=embedding_provider,
        threshold=runtime_settings.skill_match_threshold,
        max_matches=runtime_settings.max_skills_per_call,
    )
    # Create SkillManager for dynamic skill lifecycle
    skill_manager = SkillManager(
        repository=repository,
        embedding_provider=embedding_provider,
        llm_provider=structured_provider,
        prompt_renderer=prompt_renderer,
        model_name=runtime_settings.cloud_model,
    )
    reflector = CloudReflector(
        provider=structured_provider,
        prompt_renderer=prompt_renderer,
        runtime_settings=runtime_settings,
        skill_compiler=SkillCompiler(project_settings),
        skill_manager=skill_manager,
    )
    return CloudReflectionService(
        repository=repository,
        matcher=matcher,
        reflector=reflector,
        embedding_provider=embedding_provider,
    )


def build_judge_provider(runtime_settings: RuntimeSettings) -> JudgeProvider:
    return JudgeProvider(build_structured_provider(runtime_settings), runtime_settings.judge_model)
