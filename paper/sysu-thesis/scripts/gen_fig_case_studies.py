#!/usr/bin/env python3
"""Create a simple success/failure case-study panel."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from plot_common import ROOT, save, set_style


ANNOTATIONS = ROOT / "data/Distance-Annotated Traffic Perception Question Ans/DTPQA/annotations.json"
DTPQA_ROOT = ROOT / "data/Distance-Annotated Traffic Perception Question Ans/DTPQA"


def _iter_rows(payload: object, inherited: dict[str, object] | None = None):
    inherited = inherited or {}
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_rows(item, inherited)
        return
    if not isinstance(payload, dict):
        return

    for key in ("data", "samples", "records", "items", "entries", "annotations"):
        nested = payload.get(key)
        if isinstance(nested, list):
            shared = {**inherited, **{k: v for k, v in payload.items() if k != key}}
            for item in nested:
                yield from _iter_rows(item, shared)
            return

    sample_keys = {"image_path", "image", "img_path", "img_name", "file_name", "question", "answer"}
    row = {**inherited, **payload}
    if set(row) & sample_keys and ("question" in row or "answer" in row):
        yield row
        return

    for key, value in payload.items():
        if not isinstance(value, (list, dict)):
            continue
        next_inherited = dict(inherited)
        lowered = key.lower()
        if lowered in {"synth", "real"}:
            next_inherited.setdefault("subset", lowered)
        else:
            next_inherited.setdefault("category", key)
        yield from _iter_rows(value, next_inherited)


def _load_case_image(case_id: str) -> Path:
    with ANNOTATIONS.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    prefix, _, raw_index = case_id.partition("-")
    if prefix != "annotations" or not raw_index.isdigit():
        raise ValueError(f"Unsupported case identifier: {case_id}")
    target_index = int(raw_index)

    for index, row in enumerate(_iter_rows(data)):
        if index != target_index:
            continue
        rel_path = row.get("image_path") or row.get("image") or row.get("img_path")
        if not rel_path:
            raise KeyError(f"Resolved row for {case_id} does not have an image path")
        rel_path = Path(str(rel_path))
        candidates = [
            DTPQA_ROOT / rel_path,
            DTPQA_ROOT / "Distance-Annotated Traffic Perception Question Ans/DTPQA" / rel_path,
            ROOT / "data" / rel_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    raise IndexError(f"Unable to resolve image path for {case_id}")


def _render_missing_image(ax: plt.Axes, title: str, note: str) -> None:
    ax.set_facecolor("#F7F7F5")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        0.58,
        "原始图像未随当前仓库提供",
        ha="center",
        va="center",
        fontsize=10,
        color="#264653",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.36,
        title,
        ha="center",
        va="center",
        fontsize=9.2,
        color="#1F2933",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.16,
        note,
        ha="center",
        va="center",
        fontsize=8,
        color="#4B5563",
        transform=ax.transAxes,
    )


def main() -> None:
    set_style()
    success_case_id = "annotations-15"
    failure_case_id = "annotations-9594"
    success_img = _load_case_image(success_case_id)
    failure_img = _load_case_image(failure_case_id)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for ax, path, title, note in [
        (
            axes[0],
            success_img,
            "成功案例：hybrid 修复 far false negative",
            f"Synth / Cat.1 / {success_case_id}\nedge: No  → hybrid: Yes",
        ),
        (
            axes[1],
            failure_img,
            "失败案例：real pilot 持续 far-collapse",
            f"Real / Cat.1 / {failure_case_id}\nprediction: No, ground truth: Yes",
        ),
    ]:
        if path.exists():
            ax.imshow(mpimg.imread(path))
            ax.set_title(title, fontsize=9.5)
            ax.axis("off")
        else:
            _render_missing_image(ax, title, note)
        ax.text(
            0.5,
            -0.08,
            note,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
        )

    save(fig, "fig_case_studies")


if __name__ == "__main__":
    main()
