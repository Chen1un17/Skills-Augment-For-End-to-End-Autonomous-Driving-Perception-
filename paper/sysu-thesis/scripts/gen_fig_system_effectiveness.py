#!/usr/bin/env python3
"""Plot system-level gains for the hybrid route."""

from __future__ import annotations

import json
import matplotlib.pyplot as plt

from plot_common import DATA_DIR, HYBRID_COLOR, save, set_style


def main() -> None:
    set_style()
    with (DATA_DIR / "evidence_ledger.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    thesis_metrics = payload["synth50_summary"]["thesis_metrics"]

    labels = [
        "Rescue rate",
        "Harm rate",
        "Cloud-call reduction",
        "Latency reduction",
    ]
    values = [
        thesis_metrics["hybrid_rescue_rate_over_edge_errors"],
        thesis_metrics["hybrid_harm_rate_over_edge_correct"],
        thesis_metrics["hybrid_cloud_call_reduction_vs_cloud_only"],
        thesis_metrics["hybrid_latency_reduction_vs_cloud_only"],
    ]

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bars = ax.bar(labels, values, color=[HYBRID_COLOR, "#BF616A", "#5E81AC", "#A3BE8C"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ratio")
    ax.set_title("Hybrid 路由的系统级收益")

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

    save(fig, "fig_system_effectiveness")


if __name__ == "__main__":
    main()
