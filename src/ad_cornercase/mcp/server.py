"""MCP server construction."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from ad_cornercase.config import ProjectSettings, RuntimeSettings
from ad_cornercase.mcp.tools import CloudReflectionService


def create_mcp_server(
    *,
    service: CloudReflectionService,
    runtime_settings: RuntimeSettings,
    project_settings: ProjectSettings,
) -> FastMCP:
    server = FastMCP(
        project_settings.project_name,
        host=runtime_settings.mcp_server_host,
        port=runtime_settings.mcp_server_port,
        streamable_http_path=project_settings.server.mount_path,
        json_response=project_settings.server.json_response,
        stateless_http=project_settings.server.stateless_http,
        log_level=runtime_settings.log_level.upper(),
    )

    @server.tool(structured_output=True)
    async def match_skills(payload: dict[str, Any]) -> dict[str, Any]:
        result = await service.match_skills(payload)
        return result.model_dump(mode="json")

    @server.tool(structured_output=True)
    async def reflect_anomaly(payload: dict[str, Any]) -> dict[str, Any]:
        result = await service.reflect_anomaly(payload)
        return result.model_dump(mode="json")

    @server.resource("skill://{skill_id}")
    async def skill_resource(skill_id: str) -> str:
        bundle = service.read_skill(skill_id)
        return json.dumps(
            {
                "manifest": bundle.manifest.model_dump(mode="json"),
                "skill_markdown": bundle.skill_markdown,
            },
            ensure_ascii=False,
        )

    return server
