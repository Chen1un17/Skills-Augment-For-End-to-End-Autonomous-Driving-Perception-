#!/usr/bin/env python3
"""Analyze experiment results without judge evaluation."""

import json
import sys
from pathlib import Path

def analyze_run(run_id: str):
    """Analyze a run's predictions."""
    run_dir = Path(f"data/artifacts/{run_id}")
    predictions_path = run_dir / "predictions.jsonl"

    if not predictions_path.exists():
        print(f"Predictions not found: {predictions_path}")
        return

    results = []
    with open(predictions_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Calculate metrics
    total = len(results)
    baseline_correct = 0
    final_correct = 0
    reflection_count = 0
    invalid_final_answers = 0
    distance_stats = {"near": [], "mid": [], "far": [], "unknown": []}

    def normalize_answer(raw: str) -> str:
        answer = raw.lower().strip()
        if answer.startswith("yes") or "pedestrian_crossing" in answer or "crossing_pedestrian" in answer:
            return "yes"
        if answer.startswith("no") or "clear_road" in answer or "pedestrian_roadside" in answer:
            return "no"
        return ""

    for r in results:
        # Check exact match
        gt = r.get("ground_truth_answer", "").lower().strip()

        def extract_answer(result: dict) -> str:
            qa_report = result.get("qa_report", [])
            if qa_report:
                return normalize_answer(qa_report[0].get("answer", ""))
            top_k = result.get("top_k_candidates", [])
            if top_k:
                return normalize_answer(top_k[0].get("label", ""))
            return ""

        baseline_pred = extract_answer(r.get("baseline_result", {}))
        final_pred = extract_answer(r.get("final_result", {}))
        baseline_is_correct = gt == baseline_pred
        final_is_correct = gt == final_pred
        baseline_correct += int(baseline_is_correct)
        final_correct += int(final_is_correct)
        reflection_count += int(r.get("reflection_result") is not None)
        invalid_final_answers += int(final_pred == "")

        # Distance stats
        dist_group = r.get("metadata", {}).get("distance_group", "unknown")
        distance_stats[dist_group].append({
            "baseline_correct": baseline_is_correct,
            "final_correct": final_is_correct,
            "latency": r.get("metadata", {}).get("pipeline_latency_ms", r.get("final_result", {}).get("latency_ms", 0)),
        })

    # Print report
    print(f"\n{'='*60}")
    print(f"Experiment Report: {run_id}")
    print(f"{'='*60}")
    print(f"Total Cases: {total}")
    if total > 0:
        print(f"Baseline Accuracy: {baseline_correct/total:.3f} ({baseline_correct}/{total})")
        print(f"Final Accuracy: {final_correct/total:.3f} ({final_correct}/{total})")
        print(f"Delta Accuracy: {(final_correct - baseline_correct)/total:+.3f}")
        print(f"Reflection Rate: {reflection_count/total:.3f} ({reflection_count}/{total})")
        print(f"Invalid Final Answers: {invalid_final_answers}")
    else:
        print("N/A")

    print(f"\nDistance-Stratified:")
    for dist, stats in distance_stats.items():
        if stats:
            baseline_dist_correct = sum(1 for s in stats if s["baseline_correct"])
            final_dist_correct = sum(1 for s in stats if s["final_correct"])
            avg_latency = sum(s["latency"] for s in stats) / len(stats)
            print(
                f"  {dist.capitalize()}: baseline {baseline_dist_correct}/{len(stats)} ({baseline_dist_correct/len(stats):.3f}), "
                f"final {final_dist_correct}/{len(stats)} ({final_dist_correct/len(stats):.3f}), "
                f"avg pipeline latency: {avg_latency/1000:.1f}s"
            )

    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "test_synth_1775013498"
    analyze_run(run_id)
