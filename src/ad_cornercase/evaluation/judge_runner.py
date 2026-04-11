"""Judging utilities."""

from __future__ import annotations

import json

from ad_cornercase.prompts.renderer import PromptRenderer
from ad_cornercase.schemas.evaluation import CasePredictionRecord


class JudgeRunner:
    def __init__(self, *, judge_provider, prompt_renderer: PromptRenderer) -> None:
        self._judge_provider = judge_provider
        self._prompt_renderer = prompt_renderer

    async def score_record(self, record: CasePredictionRecord) -> float:
        instructions = self._prompt_renderer.load("judge_alignment.md")
        prompt = json.dumps(
            {
                "question": record.question,
                "reference": record.ground_truth_answer,
                "prediction": record.final_result.qa_report[0].answer if record.final_result.qa_report else "",
                "triplets": [triplet.model_dump(mode="json") for triplet in record.final_result.triplets],
                "recommended_action": record.final_result.recommended_action,
            },
            ensure_ascii=False,
        )
        result = await self._judge_provider.judge(instructions=instructions, prompt=prompt, metadata={"case_id": record.case_id})
        return result.score
