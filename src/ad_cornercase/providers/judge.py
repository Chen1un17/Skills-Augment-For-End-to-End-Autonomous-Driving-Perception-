"""Judge provider wrappers."""

from __future__ import annotations

import json

from ad_cornercase.providers.base import StructuredVisionProvider
from ad_cornercase.schemas.evaluation import JudgementScore


class JudgeProvider:
    def __init__(self, provider: StructuredVisionProvider, model: str) -> None:
        self._provider = provider
        self._model = model

    async def judge(self, *, instructions: str, prompt: str, metadata: dict[str, str] | None = None) -> JudgementScore:
        response = await self._provider.generate_structured(
            model=self._model,
            instructions=instructions,
            prompt=prompt,
            response_model=JudgementScore,
            metadata=metadata,
            max_completion_tokens=256,
        )
        return response.parsed


class HeuristicJudgeProvider:
    """Simple lexical fallback for tests."""

    async def judge(self, *, instructions: str, prompt: str, metadata: dict[str, str] | None = None) -> JudgementScore:
        del instructions, metadata
        try:
            payload = json.loads(prompt)
        except json.JSONDecodeError:
            return JudgementScore(score=0.0, rationale="Invalid payload.", hallucination_risk="high")
        prediction = payload["prediction"].lower()
        reference = payload["reference"].lower()
        score = 100.0 if reference in prediction or prediction in reference else 40.0
        return JudgementScore(score=score, rationale="Heuristic lexical overlap.", hallucination_risk="low" if score >= 80 else "medium")
