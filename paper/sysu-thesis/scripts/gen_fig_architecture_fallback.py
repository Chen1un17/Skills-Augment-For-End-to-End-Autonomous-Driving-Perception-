#!/usr/bin/env python3
"""Fallback architecture diagram when Gemini image generation is unavailable."""

from __future__ import annotations

from matplotlib import patches
import matplotlib.pyplot as plt

from plot_common import save, set_style


def draw_box(ax, xy, width, height, title, body, facecolor):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=facecolor,
        edgecolor="#D0D7DE",
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.72, title, ha="center", va="center", fontsize=11, weight="bold")
    ax.text(xy[0] + width / 2, xy[1] + height * 0.34, body, ha="center", va="center", fontsize=8.5)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Background bands.
    ax.add_patch(patches.Rectangle((0.3, 3.6), 9.4, 1.8, facecolor="#E8EDF2", edgecolor="none"))
    ax.add_patch(patches.Rectangle((0.3, 1.2), 9.4, 1.8, facecolor="#F5F0E8", edgecolor="none"))

    draw_box(ax, (0.7, 3.95), 2.2, 1.1, "Edge Agent", "小模型视觉感知\n不确定性估计", "#FFFFFF")
    draw_box(ax, (3.5, 3.95), 2.2, 1.1, "Router", "熵/回退标签\n触发 selective cloud", "#FFFFFF")
    draw_box(ax, (6.3, 3.95), 2.6, 1.1, "Cloud Reflector", "direct cloud re-perception\n生成纠正结果", "#FFFFFF")

    draw_box(ax, (1.2, 1.55), 2.5, 1.1, "Skill Store", "manifest.json\nSKILL.md\nembedding index", "#FFFFFF")
    draw_box(ax, (4.2, 1.55), 2.0, 1.1, "Skill Matcher", "相似度检索\n重排 / 去重", "#FFFFFF")
    draw_box(ax, (6.8, 1.55), 2.1, 1.1, "Skill Compiler", "结构化 skill 编译\n沉淀可复用知识", "#FFFFFF")

    arrow_style = dict(arrowstyle="->", color="#6B7280", lw=1.6)
    ax.annotate("", xy=(3.5, 4.5), xytext=(2.9, 4.5), arrowprops=arrow_style)
    ax.annotate("", xy=(6.3, 4.5), xytext=(5.7, 4.5), arrowprops=arrow_style)
    ax.annotate("", xy=(4.2, 2.1), xytext=(3.7, 2.1), arrowprops=arrow_style)
    ax.annotate("", xy=(6.8, 2.1), xytext=(6.2, 2.1), arrowprops=arrow_style)
    ax.annotate("", xy=(2.0, 3.95), xytext=(2.0, 2.65), arrowprops=arrow_style)
    ax.annotate("", xy=(7.8, 2.65), xytext=(7.8, 3.95), arrowprops=arrow_style)
    ax.text(5.0, 5.55, "本文提出的 training-free edge-cloud-skill 闭环架构", ha="center", fontsize=12, weight="bold")
    ax.text(5.0, 0.55, "说明：本图为按 Academic Plotting 工作流绘制的 fallback 向量图；Gemini 图像生成当前环境不可用。", ha="center", fontsize=7.5, color="#555555")

    save(fig, "fig_architecture")


if __name__ == "__main__":
    main()
