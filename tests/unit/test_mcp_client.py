from mcp.shared._httpx_utils import create_mcp_http_client
import pytest

from ad_cornercase.mcp.client import MCPGatewayClient
from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.reflection import ReflectionRequest
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult


def test_mcp_client_falls_back_to_default_httpx_factory_when_none_is_passed() -> None:
    client = MCPGatewayClient("http://127.0.0.1:8000/mcp", httpx_client_factory=None)

    assert client._httpx_client_factory is create_mcp_http_client


class _TimeoutSession:
    async def call_tool(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        raise ValueError("reflect_anomaly timed out")


def _request_with_baseline(baseline_result: EdgePerceptionResult) -> ReflectionRequest:
    return ReflectionRequest(
        anomaly_case=AnomalyCase(
            case_id="case-timeout",
            frame_id="frame-1",
            image_path="scene.png",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            metadata={"benchmark": "dtpqa", "question_type": "category_1"},
        ),
        baseline_result=baseline_result,
        applied_skill_ids=[],
    )


@pytest.mark.asyncio
async def test_mcp_client_timeout_falls_back_to_baseline_qa_answer() -> None:
    client = MCPGatewayClient("http://127.0.0.1:8000/mcp", httpx_client_factory=None)
    client._session = _TimeoutSession()
    request = _request_with_baseline(
        EdgePerceptionResult.model_validate(
            {
                "qa_report": [{"question": "Are there any people in the image?", "answer": "No"}],
                "top_k_candidates": [{"label": "Clear_Roadway", "probability": 0.9}],
                "recommended_action": "monitor",
            }
        )
    )

    result = await client.reflect_anomaly(request)

    assert result.corrected_label == "No"
    assert result.should_persist_skill is False
    assert "timed out" in result.reflection_summary


@pytest.mark.asyncio
async def test_mcp_client_timeout_falls_back_to_top_candidate_when_qa_report_is_missing() -> None:
    client = MCPGatewayClient("http://127.0.0.1:8000/mcp", httpx_client_factory=None)
    client._session = _TimeoutSession()
    request = _request_with_baseline(
        EdgePerceptionResult.model_validate(
            {
                "qa_report": [],
                "top_k_candidates": [{"label": "No", "probability": 0.9}],
                "recommended_action": "monitor",
            }
        )
    )

    result = await client.reflect_anomaly(request)

    assert result.corrected_label == "No"
    assert result.should_persist_skill is False
