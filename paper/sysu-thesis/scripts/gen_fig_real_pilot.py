#!/usr/bin/env python3
"""Plot the real-pilot collapse pattern."""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt

from plot_common import DATA_DIR, EDGE_COLOR, HYBRID_COLOR, save, set_style


def main() -> None:
    set_style()
    with (DATA_DIR / "evidence_ledger.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    real = payload["real_pilot_summary"]["distance_group_summary"]

    groups = ["far", "mid", "near", "unknown"]
    acc = [real[g]["exact_match_accuracy"] for g in groups]
    judge = [real[g]["judge_score_mean_judged_only"] / 100.0 for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    x = np.arange(len(groups))

    bars1 = axes[0].bar(x, acc, color=EDGE_COLOR)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Real pilot: exact match")

    bars2 = axes[1].bar(x, judge, color=HYBRID_COLOR)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Judge score / 100")
    axes[1].set_title("Real pilot: judge score")

    for bars, ax in [(bars1, axes[0]), (bars2, axes[1])]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444444",
            )

    save(fig, "fig_real_pilot_failure")


if __name__ == "__main__":
    main()
