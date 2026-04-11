# DTPQA Integration Analysis

## Pipeline Progress
- The loader in [src/ad_cornercase/datasets/dtpqa.py](/Users/didi/Code/Autonomous_Driving/src/ad_cornercase/datasets/dtpqa.py) now expands nested QA/annotation containers, normalizes question/answer text, and preserves distance metadata so every `AnomalyCase` carries `distance_meters`, `distance_bin`, and `distance_group` for downstream metrics.
- The DTPQA replay CLI at [src/ad_cornercase/cli/replay_dtpqa.py](/Users/didi/Code/Autonomous_Driving/src/ad_cornercase/cli/replay_dtpqa.py) reuses the standard `ReplayOrchestrator`, supports `run_id`/`append`, and therefore allows sample-by-sample resumable runs against the slow remote edge judge.
- Incremental judge persistence, Yes/No exact-match fixes, and explicit DTPQA prompting are documented in [research-log.md](/Users/didi/Code/Autonomous_Driving/research-log.md) and reflected in the current run metadata tracked in [research-state.yaml](/Users/didi/Code/Autonomous_Driving/research-state.yaml).

## Synth Baseline vs. Real Pilot Evidence
- The synth/category_1 sweep (`run-dtpqa-synth-cat1-kimi512`) evaluated 7 curated distances. Metrics show `exact_match_accuracy=1.0`, `judge_score_mean=94.57`, and mean latency `135,025 ms` with `far=96,084`, `mid=89,173`, `near=164,901`, and `unknown=169,132` group latencies, as recorded in [data/artifacts/run-dtpqa-synth-cat1-kimi512/metrics.json](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-synth-cat1-kimi512/metrics.json).
- The synth run triggered no skills, so the edge loop acts like a closed benchmark; the clean skill store ensured no extraneous reflections surfaced in the predictions archived under [data/artifacts/run-dtpqa-synth-cat1-kimi512/predictions.jsonl](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-synth-cat1-kimi512/predictions.jsonl).
- The expanded real pilot now contains 9 fully scored real/category_1 cases and no longer resembles the synth baseline. The current metrics in [data/artifacts/run-dtpqa-real-cat1-kimi512-pilot/metrics.json](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-real-cat1-kimi512-pilot/metrics.json) are `exact_match_accuracy=0.4444` and `judge_score_mean=45.0`, with distance-group accuracy `near=0.5`, `mid=0.5`, `far=0.0`, and `unknown=1.0`.
- The far-range collapse is especially sharp. Cases `annotations-9376`, `annotations-9377`, and `annotations-9378` are all real positive `far` samples, all were answered as `No`, and their judge scores are `5`, `5`, and `0`. This is the strongest current evidence that the synth baseline hides a real-data distance sensitivity.
- The real slice also reveals a strong answer asymmetry: the two sampled `unknown/No` cases (`annotations-11749`, `annotations-11768`) both remain correct with `judge_score=100`, while five of seven sampled real positive cases are false negatives.
- The dataset under [data/dtpqa/annotations.json](/Users/didi/Code/Autonomous_Driving/data/dtpqa/annotations.json) includes 9,368 synth and 9,781 real cases, with 2,581 labeled `category_1`, so there remains a large pool to stress-test the observed latency signal.

## Operational Blockers
- Long-tail replay failures still exist for some offsets such as `annotations-9370`, but they are no longer the only issue. Clean real replay now exposes fast wrong answers as a larger benchmark problem than sheer throughput.
- The MCP server must run with an empty skill store for DTPQA benchmarking. A restart against the default historical skill store immediately contaminated replay with matched CODA skills; the clean follow-up server uses `/tmp/dtpqa_skills_clean_20260331`.
- Judge scoring remains incrementally resumable but not perfectly stable. The final 9-case report required a resumed evaluation after a transient `APIConnectionError`, even though the persisted partial scores prevented progress loss.

## Root-Cause Update (2026-03-31)
- The original reflection gate in [replay.py](/Users/didi/Code/Autonomous_Driving/src/ad_cornercase/edge/replay.py) only fired on `entropy >= threshold` or `used_fallback_label`. That design is mismatched to the actual DTPQA real failure mode, because most false negatives are low-entropy `No` answers rather than uncertain fallbacks.
- The historical cloud reflection prompt in [reflector.py](/Users/didi/Code/Autonomous_Driving/src/ad_cornercase/cloud/reflector.py) leaked `ground_truth_answer`. This is now removed.
- The old clean MCP server on port `8002` was started without `REQUEST_TIMEOUT_SECONDS=300`, so the replay client could wait longer while the server-side cloud reflection still timed out first.
- Reflection itself was benchmark-misaligned even when it succeeded: it returned symbolic labels instead of DTPQA-compatible `Yes/No` answers, and it attempted to compile and persist new skills during benchmark evaluation.
- A paired comparison between the historical runs and the partial
  [run-dtpqa-real-cat1-kimi512-intervention-quick](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-real-cat1-kimi512-intervention-quick)
  showed cross-day model drift (`far` improved while `unknown` regressed) even
  with zero reflection activations. This means cross-day pre/post comparisons
  are not reliable evidence for the intervention.

## Current Intervention Status
- The code now supports a runtime-controlled DTPQA people-reflection heuristic
  through `ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER`.
- DTPQA reflection no longer persists skills, so benchmark cases do not teach
  later benchmark cases.
- The new clean MCP server on `http://127.0.0.1:8003/mcp` runs with
  `REQUEST_TIMEOUT_SECONDS=300`.
- The first same-day control attempt exposed a residual contamination path: the
  prior “clean” store `/tmp/dtpqa_skills_clean_20260331` already contained a
  DTPQA-derived skill, so `run-dtpqa-real-cat1-kimi512-sameday-off-quick` was
  invalidated by skill matches. The formal same-day baseline therefore restarts
  from `run-dtpqa-real-cat1-kimi512-sameday-off-quick-v2` on a fresh empty
  store `/tmp/dtpqa_skills_clean_20260331T173713`.
- Direct MCP probes on historical records now show the intended benchmark
  behavior:
  1. `annotations-9594` (historical false negative) reflects from baseline `No`
     to corrected label `Yes`, as recorded in
     [reflection_probe_9594.json](/Users/didi/Code/Autonomous_Driving/experiments/dtpqa-integration/results/reflection_probe_9594.json).
  2. `annotations-11749` (historical true negative) remains `No`, as recorded
     in [reflection_probe_11749.json](/Users/didi/Code/Autonomous_Driving/experiments/dtpqa-integration/results/reflection_probe_11749.json).

## Next Experiments
- Run a same-day controlled quick subset with `ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER=0` and `=1` on the `8003` server before making any new claims.
- Resume larger clean real/category_1 replay only after the same-day quick subset confirms that broad reflection improves positives without degrading negatives.
- Judge-score only after replay stability is recovered; for now, exact-match paired replay is the main trustworthy metric.
