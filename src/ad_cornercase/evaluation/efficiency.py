"""Efficiency helpers."""

from __future__ import annotations

from ad_cornercase.schemas.evaluation import CasePredictionRecord


def token_delta(record: CasePredictionRecord) -> int:
    return record.baseline_result.vision_tokens - record.final_result.vision_tokens


def latency_delta(record: CasePredictionRecord) -> float:
    return record.baseline_result.latency_ms - record.final_result.latency_ms
