#!/usr/bin/env python3
"""Plot main Synth50 system effectiveness results."""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt

from plot_common import DATA_DIR, EDGE_COLOR, CLOUD_COLOR, HYBRID_COLOR, save, set_style


def main() -> None:
    set_style()
    with (DATA_DIR / "evidence_ledger.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    modes = payload["synth50_summary"]["mode_summaries"]

    categories = ["Overall", "Far", "Mid", "Near", "Unknown"]
    edge = [
        modes["edge_only"]["exact_match_accuracy"],
        modes["edge_only"]["distance_summary"]["far"]["exact_match_accuracy"],
        modes["edge_only"]["distance_summary"]["mid"]["exact_match_accuracy"],
        modes["edge_only"]["distance_summary"]["near"]["exact_match_accuracy"],
        modes["edge_only"]["distance_summary"]["unknown"]["exact_match_accuracy"],
    ]
    cloud = [
        modes["cloud_only"]["exact_match_accuracy"],
        modes["cloud_only"]["distance_summary"]["far"]["exact_match_accuracy"],
        modes["cloud_only"]["distance_summary"]["mid"]["exact_match_accuracy"],
        modes["cloud_only"]["distance_summary"]["near"]["exact_match_accuracy"],
        modes["cloud_only"]["distance_summary"]["unknown"]["exact_match_accuracy"],
    ]
    hybrid = [
        modes["hybrid"]["exact_match_accuracy"],
        modes["hybrid"]["distance_summary"]["far"]["exact_match_accuracy"],
        modes["hybrid"]["distance_summary"]["mid"]["exact_match_accuracy"],
        modes["hybrid"]["distance_summary"]["near"]["exact_match_accuracy"],
        modes["hybrid"]["distance_summary"]["unknown"]["exact_match_accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(categories))
    width = 0.24
    bars = [
        ax.bar(x - width, edge, width, label="edge_only", color=EDGE_COLOR),
        ax.bar(x, cloud, width, label="cloud_only", color=CLOUD_COLOR),
        ax.bar(x + width, hybrid, width, label="hybrid", color=HYBRID_COLOR),
    ]

    for group in bars:
        for bar in group:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444444",
            )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(ncol=3, loc="upper center")
    ax.set_title("Synth50 子集上的三路系统效果对比")
    save(fig, "fig_main_results")


if __name__ == "__main__":
    main()
