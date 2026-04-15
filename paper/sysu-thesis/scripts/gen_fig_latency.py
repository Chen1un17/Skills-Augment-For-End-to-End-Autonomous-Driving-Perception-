#!/usr/bin/env python3
"""Plot latency comparison for the three-way Synth50 evaluation."""

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

    categories = ["Mean", "P50", "P95"]
    scale = 1000.0
    edge = [
        modes["edge_only"]["mean_latency_ms"] / scale,
        modes["edge_only"]["p50_latency_ms"] / scale,
        modes["edge_only"]["p95_latency_ms"] / scale,
    ]
    cloud = [
        modes["cloud_only"]["mean_latency_ms"] / scale,
        modes["cloud_only"]["p50_latency_ms"] / scale,
        modes["cloud_only"]["p95_latency_ms"] / scale,
    ]
    hybrid = [
        modes["hybrid"]["mean_latency_ms"] / scale,
        modes["hybrid"]["p50_latency_ms"] / scale,
        modes["hybrid"]["p95_latency_ms"] / scale,
    ]

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(categories))
    width = 0.24
    for offset, values, label, color in [
        (-width, edge, "edge_only", EDGE_COLOR),
        (0, cloud, "cloud_only", CLOUD_COLOR),
        (width, hybrid, "hybrid", HYBRID_COLOR),
    ]:
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444444",
            )

    ax.set_ylabel("Latency (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("Synth50 子集上的系统时延对比")
    save(fig, "fig_latency")


if __name__ == "__main__":
    main()
