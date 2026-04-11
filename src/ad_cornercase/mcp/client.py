"""MCP client wrapper."""

from __future__ import annotations

import datetime
import json
import logging
from contextlib import AbstractAsyncContextManager
from typing import Any

from mcp import ClientSession, McpError
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import TextContent
from pydantic import AnyUrl, TypeAdapter

from ad_cornercase.schemas.reflection import ReflectionRequest, ReflectionResult
from ad_cornercase.schemas.skill import SkillMatchRequest, SkillMatchResult


class MCPGatewayClient(AbstractAsyncContextManager["MCPGatewayClient"]):
    def __init__(self, url: str, *, timeout_seconds: float = 30.0, httpx_client_factory=create_mcp_http_client) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._httpx_client_factory = httpx_client_factory or create_mcp_http_client
        self._http_client = None
        self._streams_cm = None
        self._session_cm = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "MCPGatewayClient":
        # Use a very large httpx timeout (3600s) to avoid httpcore.ReadTimeout
        # from crossing anyio task group boundaries and causing
        # "cancel scope in different task" errors.
        # anyio.fail_after in session.call_tool will handle actual timeouts.
        self._http_client = self._httpx_client_factory(
            timeout=3600.0,
        )
        self._streams_cm = streamable_http_client(
            self._url,
            http_client=self._http_client,
        )
        read_stream, write_stream, _ = await self._streams_cm.__aenter__()
        self._session_cm = ClientSession(read_stream, write_stream)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        # Catch and ignore errors during cleanup, especially those caused by timeouts.
        # The "cancel scope in different task" error occurs when the server times out
        # and the async generator cleanup conflicts with the anyio task group.
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(exc_type, exc, exc_tb)
            except BaseExceptionGroup as e:
                if not any("cancel scope" in str(x).lower() for x in e.exceptions):
                    raise
        if self._streams_cm is not None:
            try:
                await self._streams_cm.__aexit__(exc_type, exc, exc_tb)
            except BaseExceptionGroup as e:
                if not any("cancel scope" in str(x).lower() for x in e.exceptions):
                    raise
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
        self._session = None
        self._session_cm = None
        self._streams_cm = None
        self._http_client = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCPGatewayClient must be used as an async context manager.")
        return self._session

    @staticmethod
    def _extract_structured_payload(result: Any) -> dict[str, Any]:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        if hasattr(result, "model_dump"):
            dumped = result.model_dump()
            if dumped.get("structuredContent") is not None:
                return dumped["structuredContent"]
        text_chunks = []
        for item in getattr(result, "content", []):
            if isinstance(item, TextContent):
                text_chunks.append(item.text)
            elif hasattr(item, "text"):
                text_chunks.append(item.text)
            elif isinstance(item, dict) and item.get("text") is not None:
                text_chunks.append(item["text"])
        if not text_chunks:
            raise ValueError("No structured MCP payload returned.")
        raw_text = "\n".join(text_chunks)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid MCP text payload: {raw_text!r}") from exc

    async def match_skills(self, request: SkillMatchRequest) -> SkillMatchResult:
        session = self._require_session()
        result = await session.call_tool("match_skills", arguments={"payload": request.model_dump(mode="json")})
        return SkillMatchResult.model_validate(self._extract_structured_payload(result))

    async def reflect_anomaly(self, request: ReflectionRequest) -> ReflectionResult:
        session = self._require_session()
        # reflect_anomaly can take >60s because it involves LLM reflection + skill creation
        try:
            result = await session.call_tool(
                "reflect_anomaly",
                arguments={"payload": request.model_dump(mode="json")},
                read_timeout_seconds=datetime.timedelta(seconds=300),
            )
            return ReflectionResult.model_validate(self._extract_structured_payload(result))
        except (ValueError, McpError) as e:
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                baseline_label = ""
                if getattr(request.baseline_result, "qa_report", None):
                    baseline_label = request.baseline_result.qa_report[0].answer
                elif getattr(request.baseline_result, "top_k_candidates", None):
                    baseline_label = request.baseline_result.top_k_candidates[0].label
                # Timeout - return a failed result instead of crashing
                logging.getLogger(__name__).warning(
                    f"reflect_anomaly timed out for case {getattr(request.anomaly_case, 'case_id', 'unknown')}, returning fallback result"
                )
                return ReflectionResult(
                    corrected_label=baseline_label,
                    corrected_triplets=getattr(request.baseline_result, "triplets", []),
                    reflection_summary=f"Cloud reflection timed out: {error_msg}",
                    should_persist_skill=False,
                )
            raise

    async def read_skill(self, skill_id: str) -> dict[str, Any]:
        session = self._require_session()
        uri = TypeAdapter(AnyUrl).validate_python(f"skill://{skill_id}")
        result = await session.read_resource(uri)
        if not result.contents:
            raise ValueError(f"Skill resource is empty: {skill_id}")
        content = result.contents[0]
        return json.loads(content.text)
