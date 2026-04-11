"""Evaluation metrics."""

from __future__ import annotations

import re

from ad_cornercase.schemas.evaluation import CasePredictionRecord, EvaluationSummary
from ad_cornercase.schemas.scene_graph import SceneGraphTriplet


def triplet_recall(record: CasePredictionRecord) -> float:
    ground_truth = {
        SceneGraphTriplet.model_validate(triplet).model_dump_json()
        for triplet in record.metadata.get("ground_truth_triplets", [])
    }
    if not ground_truth:
        return 0.0
    predicted = {triplet.model_dump_json() for triplet in record.final_result.triplets}
    return len(ground_truth & predicted) / len(ground_truth)


def compute_skill_success_rate(records: list[CasePredictionRecord], judge_score_threshold: float) -> float:
    relevant = [record for record in records if record.matched_skill_ids or record.reflection_result]
    if not relevant:
        return 0.0
    successful = [record for record in relevant if (record.judge_score or 0.0) >= judge_score_threshold]
    return len(successful) / len(relevant)


def normalize_answer_text(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"^option\s+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def canonicalize_answer(value: str, options: list[str] | None = None) -> str:
    normalized = normalize_answer_text(value)
    if not options:
        return normalized

    labels = "abcdefghijklmnopqrstuvwxyz"
    option_map = {labels[index]: normalize_answer_text(option) for index, option in enumerate(options[: len(labels)])}
    if normalized in option_map:
        return option_map[normalized]
    for option in options:
        normalized_option = normalize_answer_text(option)
        if normalized == normalized_option:
            return normalized_option
        if normalized_option and normalized_option in normalized:
            return normalized_option
    return normalized


def exact_match(record: CasePredictionRecord) -> float:
    prediction = record.final_result.qa_report[0].answer if record.final_result.qa_report else ""
    reference = record.ground_truth_answer
    options = record.metadata.get("answer_options")
    normalized_prediction = canonicalize_answer(prediction, options if isinstance(options, list) else None)
    normalized_reference = canonicalize_answer(reference, options if isinstance(options, list) else None)
    if not normalized_prediction or not normalized_reference:
        return 0.0
    if normalized_reference in {"yes", "no"}:
        if normalized_prediction == normalized_reference:
            return 1.0
        if normalized_prediction.startswith(f"{normalized_reference} "):
            return 1.0
    return 1.0 if normalized_prediction == normalized_reference else 0.0


def group_average(records: list[CasePredictionRecord], key: str, value_fn) -> tuple[dict[str, float], dict[str, int]]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        raw_key = record.metadata.get(key)
        group_key = str(raw_key or "unknown")
        grouped.setdefault(group_key, []).append(float(value_fn(record)))
    averages = {group_key: sum(values) / len(values) for group_key, values in grouped.items()}
    counts = {group_key: len(values) for group_key, values in grouped.items()}
    return averages, counts


def summarize_dtpqa_records(
    run_id: str,
    records: list[CasePredictionRecord],
    judge_score_threshold: float,
) -> EvaluationSummary:
    summary = summarize_records(run_id, records, judge_score_threshold)
    exact_match_accuracy = sum(exact_match(record) for record in records) / len(records) if records else 0.0
    distance_bin_accuracy, distance_bin_counts = group_average(records, "distance_bin", exact_match)
    distance_group_accuracy, _ = group_average(records, "distance_group", exact_match)
    distance_group_judge_score_mean, _ = group_average(records, "distance_group", lambda record: record.judge_score or 0.0)
    summary.exact_match_accuracy = exact_match_accuracy
    summary.distance_bin_accuracy = distance_bin_accuracy
    summary.distance_bin_counts = distance_bin_counts
    summary.distance_group_accuracy = distance_group_accuracy
    summary.distance_group_judge_score_mean = distance_group_judge_score_mean
    return summary


def summarize_records(run_id: str, records: list[CasePredictionRecord], judge_score_threshold: float) -> EvaluationSummary:
    total_cases = len(records)
    judge_score_mean = sum(record.judge_score or 0.0 for record in records) / total_cases if total_cases else 0.0
    regional_triplet_recall = sum(triplet_recall(record) for record in records) / total_cases if total_cases else 0.0
    skill_success_rate = compute_skill_success_rate(records, judge_score_threshold)
    latency_delta_ms = (
        sum(record.baseline_result.latency_ms - record.final_result.latency_ms for record in records) / total_cases
        if total_cases
        else 0.0
    )
    vision_token_delta = (
        sum(record.baseline_result.vision_tokens - record.final_result.vision_tokens for record in records) / total_cases
        if total_cases
        else 0.0
    )
    return EvaluationSummary(
        run_id=run_id,
        total_cases=total_cases,
        judge_score_mean=judge_score_mean,
        regional_triplet_recall=regional_triplet_recall,
        skill_success_rate=skill_success_rate,
        latency_delta_ms=latency_delta_ms,
        vision_token_delta=vision_token_delta,
    )
