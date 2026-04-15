#!/usr/bin/env python3
"""Shared plotting helpers for thesis figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PAPER_ROOT / "image" / "thesis"
DATA_DIR = PAPER_ROOT / "data"

COLORS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#0072B2", "#56B4E9"]
OUR_COLOR = "#E76F51"
BASELINE_COLOR = "#B0BEC5"
EDGE_COLOR = "#264653"
CLOUD_COLOR = "#2A9D8F"
HYBRID_COLOR = "#E76F51"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "STSong",
                "Songti SC",
                "SimSun",
                "Noto Serif CJK SC",
                "Source Han Serif SC",
                "Times New Roman",
                "DejaVu Serif",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(IMAGE_DIR / f"{stem}.pdf")
    fig.savefig(IMAGE_DIR / f"{stem}.png", dpi=300)
