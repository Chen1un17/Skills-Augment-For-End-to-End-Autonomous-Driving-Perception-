"""Replay orchestration."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

from ad_cornercase.config import ProjectSettings, RuntimeSettings
from ad_cornercase.edge.agent import EdgeAgent
from ad_cornercase.mcp.client import MCPGatewayClient
from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.evaluation import CasePredictionRecord
from ad_cornercase.schemas.reflection import ReflectionRequest, ReflectionResult
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult, QAItem, SceneGraphTriplet
from ad_cornercase.schemas.skill import SkillManifest, SkillMatch, SkillMatchRequest

logger = logging.getLogger(__name__)

_PERSON_LIKE_TOKENS = ("pedestrian", "person", "people", "worker")
ExecutionMode = Literal["edge_only", "cloud_only", "hybrid"]


def new_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")


def _reflection_result_to_edge_result(
    *,
    corrected_label: str,
    corrected_triplets: list[SceneGraphTriplet],
    previous: EdgePerceptionResult,
    fallback_label: str,
) -> EdgePerceptionResult:
    return EdgePerceptionResult(
        triplets=corrected_triplets,
        qa_report=[QAItem(question="What is the anomaly?", answer=corrected_label)],
        top_k_candidates=previous.top_k_candidates,
        entropy=previous.entropy,
        recommended_action=previous.recommended_action,
        latency_ms=previous.latency_ms,
        vision_tokens=previous.vision_tokens,
        applied_skill_ids=previous.applied_skill_ids,
        used_fallback_label=corrected_label == fallback_label,
    )


class ReplayOrchestrator:
    def __init__(
        self,
        *,
        edge_agent: EdgeAgent,
        runtime_settings: RuntimeSettings,
        project_settings: ProjectSettings,
        cloud_perception_agent: EdgeAgent | None = None,
    ) -> None:
        self._edge_agent = edge_agent
        self._runtime_settings = runtime_settings
        self._project_settings = project_settings
        self._cloud_perception_agent = cloud_perception_agent

    def _is_dtpqa_people_question(self, case: AnomalyCase) -> bool:
        benchmark = str(case.metadata.get("benchmark") or "").strip().lower()
        if benchmark != "dtpqa":
            return False
        question_type = str(case.metadata.get("question_type") or "").strip().lower()
        if question_type == "category_1":
            return True
        question = case.question.strip().lower()
        return any(token in question for token in _PERSON_LIKE_TOKENS)

    def _answers_no(self, result: EdgePerceptionResult) -> bool:
        if result.qa_report:
            answer = result.qa_report[0].answer
        elif result.top_k_candidates:
            answer = result.top_k_candidates[0].label
        else:
            answer = ""
        return answer.strip().lower().startswith("no")

    def _has_person_like_secondary_candidate(self, result: EdgePerceptionResult) -> bool:
        return any(
            any(token in candidate.label.strip().lower() for token in _PERSON_LIKE_TOKENS)
            for candidate in result.top_k_candidates[1:]
        )

    def _has_person_evidence(self, result: EdgePerceptionResult) -> bool:
        if any(result.general_perception.vulnerable_road_users):
            return True
        for region in result.regional_perception:
            region_text = " ".join(
                part
                for part in (
                    getattr(region, "category_name", None),
                    getattr(region, "description", None),
                    getattr(region, "explanation", None),
                )
                if part
            ).lower()
            if any(token in region_text for token in _PERSON_LIKE_TOKENS):
                return True
        for triplet in result.triplets:
            triplet_text = " ".join((triplet.subject, triplet.relation, triplet.object)).lower()
            if any(token in triplet_text for token in _PERSON_LIKE_TOKENS):
                return True
        if any(any(token in candidate.label.strip().lower() for token in _PERSON_LIKE_TOKENS) for candidate in result.top_k_candidates):
            return True
        return False

    def _should_reflect(self, case: AnomalyCase, result: EdgePerceptionResult) -> bool:
        if result.entropy >= self._runtime_settings.uncertainty_entropy_threshold or result.used_fallback_label:
            return True
        if not self._runtime_settings.enable_dtpqa_people_reflection_trigger:
            return False
        if not self._is_dtpqa_people_question(case) or not self._answers_no(result):
            return False
        question_type = str(case.metadata.get("question_type") or "").strip().lower()
        if question_type == "category_1":
            return self._has_person_like_secondary_candidate(result) or self._has_person_evidence(result)
        return self._has_person_like_secondary_candidate(result) or self._has_person_evidence(result)

    def _build_trigger_text(self, case: AnomalyCase, baseline_result: EdgePerceptionResult) -> str:
        labels = ",".join(candidate.label for candidate in baseline_result.top_k_candidates)
        weather = ",".join(case.weather_tags)
        return f"{case.sensor_context};{weather};{labels};{case.question}"

    def _synthetic_skill_match(self, manifest: SkillManifest) -> SkillMatch:
        return SkillMatch(
            skill_id=manifest.skill_id,
            score=1.0,
            prompt_patch=(
                f"Skill `{manifest.skill_id}`\n"
                f"Focus region: {manifest.focus_region}\n"
                + "\n".join(f"- {item}" for item in manifest.dynamic_question_tree)
            ),
            manifest=manifest,
        )

    def _load_existing_case_ids(self, predictions_path: Path) -> set[str]:
        if not predictions_path.exists():
            return set()
        existing_case_ids: set[str] = set()
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                case_id = payload.get("case_id")
                if isinstance(case_id, str) and case_id:
                    existing_case_ids.add(case_id)
        return existing_case_ids

    async def run(
        self,
        *,
        cases: list[AnomalyCase],
        server_url: str | None,
        run_id: str | None = None,
        append: bool = False,
        execution_mode: ExecutionMode = "hybrid",
    ) -> Path:
        return await self.run_with_client_factory(
            cases=cases,
            server_url=server_url,
            run_id=run_id,
            append=append,
            execution_mode=execution_mode,
        )

    async def run_with_client_factory(
        self,
        *,
        cases: list[AnomalyCase],
        server_url: str | None,
        run_id: str | None = None,
        append: bool = False,
        httpx_client_factory=None,
        execution_mode: ExecutionMode = "hybrid",
    ) -> Path:
        run_id = run_id or new_run_id()
        run_dir = self._runtime_settings.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = run_dir / "predictions.jsonl"
        if predictions_path.exists() and not append:
            raise FileExistsError(
                f"Predictions file already exists for run {run_id}: {predictions_path}. "
                "Use append=True to resume or pick a different run_id."
            )
        existing_case_ids = self._load_existing_case_ids(predictions_path) if append else set()
        mode = "a" if append else "w"
        with predictions_path.open(mode, encoding="utf-8") as handle:
            if execution_mode == "hybrid":
                if not server_url:
                    raise ValueError("server_url is required for hybrid execution mode")
                async with MCPGatewayClient(
                    server_url,
                    timeout_seconds=self._runtime_settings.request_timeout_seconds,
                    httpx_client_factory=httpx_client_factory,
                ) as client:
                    for case in cases:
                        await self._process_case(
                            case=case,
                            handle=handle,
                            existing_case_ids=existing_case_ids,
                            execution_mode=execution_mode,
                            client=client,
                        )
            else:
                for case in cases:
                    await self._process_case(
                        case=case,
                        handle=handle,
                        existing_case_ids=existing_case_ids,
                        execution_mode=execution_mode,
                        client=None,
                    )
        return run_dir

    async def _process_case(
        self,
        *,
        case: AnomalyCase,
        handle,
        existing_case_ids: set[str],
        execution_mode: ExecutionMode,
        client: MCPGatewayClient | None,
    ) -> None:
        if case.case_id in existing_case_ids:
            logger.info("Skipping case %s: already present in predictions", case.case_id)
            return

        if execution_mode == "edge_only":
            record = await self._run_single_model_case(case, execution_mode, self._edge_agent)
        elif execution_mode == "cloud_only":
            if self._cloud_perception_agent is None:
                raise RuntimeError("cloud_perception_agent is required for cloud_only execution mode")
            record = await self._run_single_model_case(case, execution_mode, self._cloud_perception_agent)
        else:
            if client is None:
                raise RuntimeError("Hybrid execution requires an MCP client")
            record = await self._run_hybrid_case(case, client)

        handle.write(record.model_dump_json() + "\n")
        handle.flush()
        existing_case_ids.add(case.case_id)
        logger.info("Finished case %s in mode %s", case.case_id, execution_mode)

    async def _run_single_model_case(
        self,
        case: AnomalyCase,
        execution_mode: ExecutionMode,
        agent: EdgeAgent,
    ) -> CasePredictionRecord:
        start = perf_counter()
        result = await agent.perceive(case)
        pipeline_latency_ms = (perf_counter() - start) * 1000
        metadata = {
            **case.metadata,
            "execution_mode": execution_mode,
            "pipeline_latency_ms": pipeline_latency_ms,
            "stage_latency_ms": {
                execution_mode: pipeline_latency_ms,
            },
            "ground_truth_triplets": [
                triplet.model_dump(mode="json") for triplet in case.ground_truth_triplets
            ],
        }
        return CasePredictionRecord(
            case_id=case.case_id,
            question=case.question,
            ground_truth_answer=case.ground_truth_answer,
            baseline_result=result.model_copy(deep=True),
            final_result=result.model_copy(deep=True),
            matched_skill_ids=[],
            reflection_result=None,
            metadata=metadata,
        )

    async def _run_hybrid_case(
        self,
        case: AnomalyCase,
        client: MCPGatewayClient,
    ) -> CasePredictionRecord:
        stage_latency_ms = {
            "baseline_edge": 0.0,
            "skill_rerun": 0.0,
            "cloud_reflection": 0.0,
            "final_edge_rerun": 0.0,
        }
        pipeline_start = perf_counter()
        hybrid_strategy = "edge_only_passthrough"

        logger.info("Running case %s: baseline perception", case.case_id)
        start = perf_counter()
        baseline_result = await self._edge_agent.perceive(case)
        stage_latency_ms["baseline_edge"] = (perf_counter() - start) * 1000

        match_request = SkillMatchRequest(
            case_id=case.case_id,
            sensor_context=case.sensor_context,
            weather_tags=case.weather_tags,
            top_k_labels=[candidate.label for candidate in baseline_result.top_k_candidates],
            entropy=baseline_result.entropy,
            trigger_text=self._build_trigger_text(case, baseline_result),
        )
        logger.info("Running case %s: skill matching", case.case_id)
        match_result = await client.match_skills(match_request)
        current_result = baseline_result
        if match_result.matches:
            logger.info(
                "Running case %s: re-perception with %d matched skills",
                case.case_id,
                len(match_result.matches),
            )
            start = perf_counter()
            current_result = await self._edge_agent.perceive(case, match_result.matches)
            stage_latency_ms["skill_rerun"] = (perf_counter() - start) * 1000
            hybrid_strategy = "skill_augmented_edge"

        reflection_result = None
        final_result = current_result
        should_force_reflect = bool(case.metadata.get("force_reflection")) and not match_result.matches
        if should_force_reflect or self._should_reflect(case, current_result):
            if self._cloud_perception_agent is not None and self._is_dtpqa_people_question(case):
                logger.info("Running case %s: direct cloud re-perception", case.case_id)
                start = perf_counter()
                final_result = await self._cloud_perception_agent.perceive(case)
                stage_latency_ms["cloud_reflection"] = (perf_counter() - start) * 1000
                cloud_answer = ""
                if final_result.qa_report:
                    cloud_answer = final_result.qa_report[0].answer
                elif final_result.top_k_candidates:
                    cloud_answer = final_result.top_k_candidates[0].label
                reflection_result = ReflectionResult(
                    corrected_label=cloud_answer,
                    corrected_triplets=final_result.triplets,
                    reflection_summary="Direct cloud re-perception used for DTPQA category_1.",
                    should_persist_skill=False,
                )
                hybrid_strategy = "direct_cloud_perception"
            else:
                logger.info("Running case %s: cloud reflection", case.case_id)
                start = perf_counter()
                reflection_result = await client.reflect_anomaly(
                    ReflectionRequest(
                        anomaly_case=case,
                        baseline_result=current_result,
                        applied_skill_ids=current_result.applied_skill_ids,
                    )
                )
                stage_latency_ms["cloud_reflection"] = (perf_counter() - start) * 1000
                hybrid_strategy = "mcp_reflection"
                if reflection_result.new_skill:
                    logger.info(
                        "Running case %s: re-perception with reflected skill %s",
                        case.case_id,
                        reflection_result.new_skill.skill_id,
                    )
                    start = perf_counter()
                    final_result = await self._edge_agent.perceive(
                        case,
                        [self._synthetic_skill_match(reflection_result.new_skill)],
                    )
                    stage_latency_ms["final_edge_rerun"] = (perf_counter() - start) * 1000
                    hybrid_strategy = "mcp_reflection_with_skill_rerun"
                else:
                    final_result = _reflection_result_to_edge_result(
                        corrected_label=reflection_result.corrected_label,
                        corrected_triplets=reflection_result.corrected_triplets,
                        previous=current_result,
                        fallback_label=self._project_settings.fallback_label,
                    )

        pipeline_latency_ms = (perf_counter() - pipeline_start) * 1000
        metadata = {
            **case.metadata,
            "execution_mode": "hybrid",
            "hybrid_strategy": hybrid_strategy,
            "pipeline_latency_ms": pipeline_latency_ms,
            "stage_latency_ms": stage_latency_ms,
            "ground_truth_triplets": [
                triplet.model_dump(mode="json") for triplet in case.ground_truth_triplets
            ],
        }
        return CasePredictionRecord(
            case_id=case.case_id,
            question=case.question,
            ground_truth_answer=case.ground_truth_answer,
            baseline_result=baseline_result,
            final_result=final_result,
            matched_skill_ids=[match.skill_id for match in match_result.matches],
            reflection_result=reflection_result,
            metadata=metadata,
        )
