"""DTPQA dataset loader."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.common import BoundingBox


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [_stringify(item).strip() for item in value if _stringify(item).strip()]
    return [_stringify(value).strip()] if _stringify(value).strip() else []


def _parse_bbox(value: Any) -> BoundingBox | None:
    if isinstance(value, dict):
        return BoundingBox.model_validate(value)
    if isinstance(value, list) and len(value) == 4:
        return BoundingBox(x1=int(value[0]), y1=int(value[1]), x2=int(value[2]), y2=int(value[3]))
    return None


def _coerce_distance_meters(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        if match:
            return float(match.group(1))
    return None


def _derive_distance_group(distance_meters: float | None, distance_bin: str | None) -> str:
    if distance_meters is not None:
        if distance_meters <= 20:
            return "near"
        if distance_meters <= 30:
            return "mid"
        return "far"
    normalized = (distance_bin or "").strip().lower()
    if not normalized:
        return "unknown"
    numeric_matches = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", normalized)]
    if numeric_matches:
        upper = max(numeric_matches)
        if upper <= 20:
            return "near"
        if upper <= 30:
            return "mid"
        return "far"
    if "far" in normalized or "long" in normalized or "30" in normalized or "40" in normalized or "50" in normalized:
        return "far"
    if "mid" in normalized or "20" in normalized:
        return "mid"
    if "near" in normalized or "0-10" in normalized or "10-20" in normalized:
        return "near"
    return "unknown"


def _normalize_distance_bin(raw_bin: Any, raw_distance: Any) -> str:
    if raw_bin is not None:
        return _stringify(raw_bin).strip()
    distance_meters = _coerce_distance_meters(raw_distance)
    if distance_meters is None:
        return "unknown"
    if distance_meters <= 10:
        return "0-10m"
    if distance_meters <= 20:
        return "10-20m"
    if distance_meters <= 30:
        return "20-30m"
    return "30m+"


def _format_question(question: str, options: list[str]) -> str:
    if not options:
        return question
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option_lines = [f"{labels[index]}. {option}" for index, option in enumerate(options[: len(labels)])]
    return f"{question}\nOptions:\n" + "\n".join(option_lines)


def _resolve_answer(raw_answer: Any, options: list[str]) -> str:
    if isinstance(raw_answer, int) and 0 <= raw_answer < len(options):
        return options[raw_answer]
    if isinstance(raw_answer, str):
        stripped = raw_answer.strip()
        if stripped.isdigit():
            index = int(stripped)
            if 0 <= index < len(options):
                return options[index]
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if len(stripped) == 1 and stripped.upper() in labels:
            index = labels.index(stripped.upper())
            if index < len(options):
                return options[index]
        return stripped
    return _stringify(raw_answer).strip()


def _looks_like_sample(row: dict[str, Any]) -> bool:
    has_image = _first_present(row, "image", "image_path", "img_path", "img_name", "file_name") is not None
    has_question = _first_present(row, "question", "query", "prompt") is not None
    has_answer = _first_present(row, "answer", "gt_answer", "correct_answer", "label") is not None
    return has_image and (has_question or has_answer)


def _strip_container_keys(row: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in keys}


def _expand_rows(payload: Any, inherited: dict[str, Any] | None = None) -> Iterable[dict[str, Any]]:
    inherited = inherited or {}
    if isinstance(payload, list):
        for item in payload:
            yield from _expand_rows(item, inherited)
        return
    if not isinstance(payload, dict):
        return

    container_keys = {"data", "samples", "records", "items", "entries", "annotations"}
    for key in container_keys:
        nested = payload.get(key)
        if isinstance(nested, list):
            shared = {**inherited, **_strip_container_keys(payload, container_keys)}
            for item in nested:
                yield from _expand_rows(item, shared)
            return

    qa_keys = {"qa_pairs", "qas", "question_answers", "questions"}
    for key in qa_keys:
        nested = payload.get(key)
        if isinstance(nested, list) and nested and all(isinstance(item, dict) for item in nested):
            shared = {**inherited, **_strip_container_keys(payload, qa_keys)}
            for item in nested:
                yield {**shared, **item}
            return

    row = {**inherited, **payload}
    if _looks_like_sample(row):
        yield row
        return

    sample_keys = {
        "image",
        "image_path",
        "img_path",
        "img_name",
        "file_name",
        "question",
        "query",
        "prompt",
        "answer",
        "gt_answer",
        "correct_answer",
        "label",
    }
    if set(payload) & sample_keys:
        return

    for key, value in payload.items():
        if not isinstance(value, (list, dict)):
            continue
        inherited_fields = dict(inherited)
        lowered = key.lower()
        if lowered in {"synth", "real"}:
            inherited_fields.setdefault("subset", lowered)
            inherited_fields.setdefault("source_split", lowered)
        else:
            inherited_fields.setdefault("category", key)
            inherited_fields.setdefault("question_type", key)
        yield from _expand_rows(value, inherited_fields)


class DTPQADatasetLoader:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _candidate_annotation_paths(self, subset: str, annotation_glob: str | None) -> list[Path]:
        if annotation_glob:
            paths = sorted(self._root.glob(annotation_glob))
            return [path for path in paths if path.is_file()]

        subset_roots: list[Path] = []
        normalized_subset = subset.lower()
        if subset not in {"all", "*"}:
            candidates = [
                self._root / subset,
                self._root / normalized_subset,
                self._root / f"dtp_{normalized_subset}",
            ]
            subset_roots = [path for path in candidates if path.exists()]
        search_roots = [self._root]
        for path in subset_roots:
            if path not in search_roots:
                search_roots.append(path)

        patterns = (
            "*.jsonl",
            "*qa*.json",
            "*question*.json",
            "*anno*.json",
            "*label*.json",
            "*annotation*.json",
            "*benchmark*.json",
        )
        discovered: dict[Path, None] = {}
        for base in search_roots:
            for pattern in patterns:
                for path in base.rglob(pattern):
                    if path.is_file():
                        discovered[path] = None
        return sorted(discovered)

    def _resolve_image_path(self, annotation_path: Path, image_value: str) -> Path:
        raw = Path(image_value)
        candidates = [
            raw,
            annotation_path.parent / raw,
            annotation_path.parent.parent / raw,
            self._root / raw,
        ]
        for path in candidates:
            if path.exists():
                return path.resolve()
        return candidates[-1].resolve()

    def _iter_annotation_rows(self, annotation_path: Path) -> Iterable[dict[str, Any]]:
        if annotation_path.suffix == ".jsonl":
            with annotation_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield from _expand_rows(row)
            return

        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        yield from _expand_rows(payload)

    def load(
        self,
        *,
        subset: str = "all",
        question_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        annotation_glob: str | None = None,
    ) -> list[AnomalyCase]:
        if offset < 0:
            raise ValueError("offset must be >= 0")
        annotation_paths = self._candidate_annotation_paths(subset, annotation_glob)
        if not annotation_paths:
            raise FileNotFoundError(
                f"Unable to locate DTPQA annotations under {self._root} for subset={subset!r}"
            )

        cases: list[AnomalyCase] = []
        seen_after_filter = 0
        for annotation_path in annotation_paths:
            for index, row in enumerate(self._iter_annotation_rows(annotation_path)):
                normalized_question_type = _stringify(
                    _first_present(row, "question_type", "task", "category", "type")
                ).strip()
                if question_type and normalized_question_type and normalized_question_type != question_type:
                    continue

                raw_question = _stringify(_first_present(row, "question", "query", "prompt")).strip()
                options = _normalize_list(_first_present(row, "options", "choices", "candidate_answers", "answers"))
                if not raw_question:
                    continue
                image_value = _stringify(
                    _first_present(row, "image", "image_path", "img_path", "img_name", "file_name")
                ).strip()
                if not image_value:
                    continue

                raw_answer = _first_present(row, "answer", "gt_answer", "correct_answer", "label")
                ground_truth_answer = _resolve_answer(raw_answer, options)
                distance_meters = _coerce_distance_meters(
                    _first_present(row, "distance_meters", "distance", "object_distance", "range_meters")
                )
                distance_bin = _normalize_distance_bin(
                    _first_present(row, "distance_bin", "range_label", "distance_range"),
                    distance_meters,
                )
                subset_value = _stringify(_first_present(row, "subset", "source_split")).strip() or subset
                annotation_parts = {part.lower() for part in annotation_path.parts}
                if subset == "all" and subset_value in {"", "all"}:
                    if "dtp_real" in annotation_parts or "real" in annotation_parts:
                        subset_value = "real"
                    elif "dtp_synth" in annotation_parts or "synthetic" in annotation_parts or "synth" in annotation_parts:
                        subset_value = "synth"
                if subset not in {"all", "*"} and subset_value not in {"", subset}:
                    continue
                if seen_after_filter < offset:
                    seen_after_filter += 1
                    continue
                if limit is not None and len(cases) >= limit:
                    return cases

                question = _format_question(raw_question, options)
                case_id = _stringify(_first_present(row, "question_id", "sample_id", "id", "uid")).strip() or (
                    f"{annotation_path.stem}-{index}"
                )
                frame_id = _stringify(
                    _first_present(row, "frame_id", "image_id", "frame", "sample_id", "id")
                ).strip() or case_id
                weather_tags = _normalize_list(
                    _first_present(row, "weather_tags", "weather", "condition", "conditions", "time_of_day")
                )
                sensor_context = _stringify(
                    _first_present(row, "sensor_context", "camera", "view", "source")
                ).strip() or f"dtpqa_{subset_value}"

                metadata = {
                    "benchmark": "dtpqa",
                    "subset": subset_value,
                    "annotation_path": str(annotation_path),
                    "question_type": normalized_question_type or "unknown",
                    "distance_meters": distance_meters,
                    "distance_bin": distance_bin,
                    "distance_group": _derive_distance_group(distance_meters, distance_bin),
                    "answer_options": options,
                    "raw_question": raw_question,
                }
                for key in ("scene_id", "sequence_id", "video_id", "source_split"):
                    value = row.get(key)
                    if value is not None:
                        metadata[key] = value

                case = AnomalyCase(
                    case_id=case_id,
                    frame_id=frame_id,
                    image_path=self._resolve_image_path(annotation_path, image_value),
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    crop_bbox=_parse_bbox(_first_present(row, "crop_bbox", "bbox", "box", "bounding_box")),
                    weather_tags=weather_tags,
                    sensor_context=sensor_context,
                    metadata=metadata,
                )
                cases.append(case)
        if not cases:
            raise ValueError(f"No DTPQA samples were parsed from annotations under {self._root}")
        return cases
