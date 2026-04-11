"""Configuration loading."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SILICONFLOW_API_KEY = None
DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EDGE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_CLOUD_MODEL = "Pro/moonshotai/Kimi-K2.5"
DEFAULT_JUDGE_MODEL = "Pro/moonshotai/Kimi-K2.5"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0


class ServerSettings(BaseModel):
    mount_path: str = "/mcp"
    stateless_http: bool = True
    json_response: bool = True


class ProjectSettings(BaseModel):
    project_name: str = "ad-cornercase"
    default_split: str = "Mini"
    default_task: str = "region_perception"
    fallback_label: str = "Critical_Unknown_Obstacle"
    default_focus_region: str = "lower_center"
    max_cases_per_run: int = 50
    judge_score_threshold: float = 70.0
    entropy_trigger_floor: float = 1.0
    server: ServerSettings = Field(default_factory=ServerSettings)
    skill_defaults: dict[str, list[str]] = Field(default_factory=dict)


class RuntimeSettings(BaseModel):
    openai_api_key: str | None = DEFAULT_SILICONFLOW_API_KEY
    openai_base_url: str = DEFAULT_SILICONFLOW_BASE_URL
    edge_model: str = DEFAULT_EDGE_MODEL
    edge_max_completion_tokens: int = 1024
    cloud_model: str = DEFAULT_CLOUD_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    coda_lm_root: Path = Path("./data/coda_lm")
    dtpqa_root: Path = Path("./data/dtpqa")
    artifacts_dir: Path = Path("./data/artifacts")
    skill_store_dir: Path = Path("./data/skills")
    prompts_dir: Path = Path("./configs/prompts")
    settings_path: Path = Path("./configs/settings.yaml")
    uncertainty_entropy_threshold: float = 1.0
    skill_match_threshold: float = 0.82
    max_skills_per_call: int = 3
    mcp_server_host: str = "127.0.0.1"
    mcp_server_port: int = 8000
    mcp_server_url: HttpUrl = "http://127.0.0.1:8000/mcp"
    log_level: str = "INFO"
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_retries: int = 3
    enable_dtpqa_people_reflection_trigger: bool = True

    def project_root(self) -> Path:
        return self.settings_path.resolve().parent.parent


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in settings file: {path}")
    return data


@lru_cache(maxsize=1)
def get_runtime_settings() -> RuntimeSettings:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or DEFAULT_SILICONFLOW_API_KEY
    return RuntimeSettings(
        openai_api_key=api_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL),
        edge_model=os.getenv("EDGE_MODEL", DEFAULT_EDGE_MODEL),
        edge_max_completion_tokens=int(os.getenv("EDGE_MAX_COMPLETION_TOKENS", "1024")),
        cloud_model=os.getenv("CLOUD_MODEL", DEFAULT_CLOUD_MODEL),
        judge_model=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        coda_lm_root=_env_path("CODA_LM_ROOT", "./data/coda_lm"),
        dtpqa_root=_env_path("DTPQA_ROOT", "./data/dtpqa"),
        artifacts_dir=_env_path("ARTIFACTS_DIR", "./data/artifacts"),
        skill_store_dir=_env_path("SKILL_STORE_DIR", "./data/skills"),
        prompts_dir=_env_path("PROMPTS_DIR", "./configs/prompts"),
        settings_path=_env_path("SETTINGS_PATH", "./configs/settings.yaml"),
        uncertainty_entropy_threshold=float(os.getenv("UNCERTAINTY_ENTROPY_THRESHOLD", "1.0")),
        skill_match_threshold=float(os.getenv("SKILL_MATCH_THRESHOLD", "0.55")),
        max_skills_per_call=int(os.getenv("MAX_SKILLS_PER_CALL", "3")),
        mcp_server_host=os.getenv("MCP_SERVER_HOST", "127.0.0.1"),
        mcp_server_port=int(os.getenv("MCP_SERVER_PORT", "8000")),
        mcp_server_url=os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS))),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        enable_dtpqa_people_reflection_trigger=os.getenv("ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER", "1").lower()
        not in {"0", "false", "no"},
    )


@lru_cache(maxsize=1)
def get_project_settings() -> ProjectSettings:
    runtime = get_runtime_settings()
    return ProjectSettings.model_validate(_load_yaml(runtime.settings_path))
