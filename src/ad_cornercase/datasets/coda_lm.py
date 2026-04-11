"""CODA-LM dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.common import BoundingBox
from ad_cornercase.schemas.scene_graph import SceneGraphTriplet


def _parse_triplets(value: Any) -> list[SceneGraphTriplet]:
    if not value:
        return []
    triplets: list[SceneGraphTriplet] = []
    for item in value:
        if isinstance(item, dict):
            triplets.append(SceneGraphTriplet.model_validate(item))
    return triplets


def _parse_bbox(value: Any) -> BoundingBox | None:
    if isinstance(value, dict):
        return BoundingBox.model_validate(value)
    if isinstance(value, list) and len(value) == 4:
        return BoundingBox(x1=int(value[0]), y1=int(value[1]), x2=int(value[2]), y2=int(value[3]))
    return None


class CodaLMDatasetLoader:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _resolve_annotation_path(self, split: str, task: str) -> Path:
        candidates = [
            self._root / split / "vqa_anno" / f"{task}.jsonl",
            self._root / split / f"{task}.jsonl",
            self._root / f"{task}.jsonl",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Unable to locate CODA-LM annotation file for split={split}, task={task}")

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

    def load(self, *, split: str, task: str, limit: int | None = None) -> list[AnomalyCase]:
        annotation_path = self._resolve_annotation_path(split, task)
        cases: list[AnomalyCase] = []
        with annotation_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if limit is not None and len(cases) >= limit:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                question_id = str(row.get("question_id") or row.get("case_id") or f"{split}-{task}-{index}")
                weather_tags = row.get("weather_tags") or row.get("weather") or []
                if isinstance(weather_tags, str):
                    weather_tags = [item.strip() for item in weather_tags.split(",") if item.strip()]
                case = AnomalyCase(
                    case_id=question_id,
                    frame_id=str(row.get("frame_id") or question_id),
                    image_path=self._resolve_image_path(annotation_path, str(row.get("image") or row.get("image_path"))),
                    question=str(row["question"]),
                    ground_truth_answer=str(row.get("answer") or row.get("ground_truth_answer") or ""),
                    crop_bbox=_parse_bbox(row.get("crop_bbox")),
                    weather_tags=list(weather_tags),
                    sensor_context=str(row.get("sensor_context") or "front_camera"),
                    metadata={
                        **(row.get("metadata") or {}),
                        "benchmark": "coda_lm",
                        "split": split,
                        "task": task,
                        "annotation_path": str(annotation_path),
                    },
                    ground_truth_triplets=_parse_triplets(row.get("ground_truth_triplets")),
                )
                cases.append(case)
        return cases
