#!/usr/bin/env python3
"""Generate academic plots for experiment results."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_predictions(run_id: str):
    """Load predictions from a run."""
    predictions_path = Path(f"data/artifacts/{run_id}/predictions.jsonl")
    if not predictions_path.exists():
        return []

    results = []
    with open(predictions_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def analyze_run(run_id: str):
    """Analyze a single run."""
    results = load_predictions(run_id)

    if not results:
        return None

    total = len(results)
    correct = 0
    distance_data = {"near": [], "mid": [], "far": [], "unknown": []}

    for r in results:
        gt = r.get("ground_truth_answer", "").lower().strip()
        qa_report = r.get("final_result", {}).get("qa_report", [])

        pred = ""
        if qa_report:
            pred_answer = qa_report[0].get("answer", "").lower().strip()
            if "yes" in pred_answer:
                pred = "yes"
            elif "no" in pred_answer:
                pred = "no"

        is_correct = (gt == pred) or (gt == "yes" and "yes" in pred) or (gt == "no" and "no" in pred)
        if is_correct:
            correct += 1

        dist_group = r.get("metadata", {}).get("distance_group", "unknown")
        latency = r.get("final_result", {}).get("latency_ms", 0)
        distance_data[dist_group].append({
            "correct": is_correct,
            "latency": latency,
            "entropy": r.get("final_result", {}).get("entropy", 0),
        })

    return {
        "run_id": run_id,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "distance_data": distance_data,
    }

def generate_accuracy_plot(runs: list, output_path: str):
    """Generate accuracy comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    run_ids = [r["run_id"][:20] + "..." if len(r["run_id"]) > 20 else r["run_id"] for r in runs]
    accuracies = [r["accuracy"] * 100 for r in runs]

    colors = ["#2ecc71" if acc == 100 else "#e74c3c" for acc in accuracies]

    ax.barh(run_ids, accuracies, color=colors, alpha=0.8)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_ylabel("Run ID", fontsize=12)
    ax.set_title("Experiment Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)

    for i, (acc, total) in enumerate(zip(accuracies, [r["total"] for r in runs])):
        ax.text(acc + 1, i, f"{acc:.0f}% (n={total})", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved accuracy plot: {output_path}")

def generate_distance_plot(runs: list, output_path: str):
    """Generate distance-stratified accuracy plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    distances = ["near", "mid", "far", "unknown"]
    x = np.arange(len(distances))
    width = 0.25

    for i, run in enumerate(runs[:3]):  # Max 3 runs
        accs = []
        for dist in distances:
            stats = run["distance_data"].get(dist, [])
            if stats:
                correct = sum(1 for s in stats if s["correct"])
                accs.append(correct / len(stats) * 100)
            else:
                accs.append(0)

        offset = width * (i - 1)
        ax.bar(x + offset, accs, width, label=run["run_id"][:15], alpha=0.8)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Distance Group", fontsize=12)
    ax.set_title("Distance-Stratified Accuracy", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(distances)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved distance plot: {output_path}")

def generate_latency_plot(runs: list, output_path: str):
    """Generate latency distribution plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    distances = ["near", "mid", "far"]
    colors = {"near": "#3498db", "mid": "#f39c12", "far": "#e74c3c"}

    all_latencies = []
    labels = []
    colors_list = []

    for dist in distances:
        for run in runs:
            stats = run["distance_data"].get(dist, [])
            for s in stats:
                all_latencies.append(s["latency"] / 1000)  # Convert to seconds
                labels.append(dist)
                colors_list.append(colors[dist])

    if all_latencies:
        ax.scatter(range(len(all_latencies)), all_latencies, c=colors_list, alpha=0.6, s=100)
        ax.set_xlabel("Sample Index", fontsize=12)
        ax.set_ylabel("Latency (seconds)", fontsize=12)
        ax.set_title("Per-Sample Latency Distribution", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[d], label=d.capitalize()) for d in distances]
        ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved latency plot: {output_path}")

def main():
    """Generate all plots."""
    run_ids = [
        "test_synth_1775013498",
        "test_synth_1775013857",
        "test_single_1775014758",
    ]

    runs = []
    for run_id in run_ids:
        data = analyze_run(run_id)
        if data:
            runs.append(data)

    if not runs:
        print("No valid runs found")
        return

    output_dir = Path("experiments/dtpqa-integration/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_accuracy_plot(runs, output_dir / "accuracy_comparison.png")
    generate_distance_plot(runs, output_dir / "distance_accuracy.png")
    generate_latency_plot(runs, output_dir / "latency_distribution.png")

    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
