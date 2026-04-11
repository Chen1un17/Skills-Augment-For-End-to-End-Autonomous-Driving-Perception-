# Research Log

## 2026-03-30

- Bootstrapped autoresearch state for the repository.
- Read `README.md`, `Guide.md`, `docs/architecture.md`, `docs/benchmark_setup.md`,
  and implementation files for replay and evaluation.
- Confirmed the main unfinished implementation gap is `DTPQA`; `DriveLM` also
  remains scaffolded, but the immediate executable research target is DTPQA.
- Confirmed the current environment cannot execute the project yet because the
  repository `.venv` points to a missing `/tmp/uv-python/.../python3.11`
  interpreter and system Python is only `3.9.6`.
- Locked the initial protocol:
  1. Reuse the existing `AnomalyCase -> ReplayOrchestrator -> Evaluation` path.
  2. Add DTPQA normalization and distance-aware metrics.
  3. Preserve benchmark metadata in artifacts for downstream analysis.
  4. Only after code/tests are in place, rebuild the runtime and download the
     real dataset for large-scale validation.
- Implemented DTPQA dataset loading, replay CLI, evaluation CLI, and
  distance-aware metrics with fixture-backed tests.
- Extracted the official DTPQA release to `data/dtpqa` and verified that the
  release contains 19,149 examples (`synth=9,368`, `real=9,781`).
- Found and fixed several execution-path bugs while moving from fixtures to the
  real dataset:
  1. DTPQA root annotations can live at the dataset root, not only under subset directories.
  2. `MCPGatewayClient(None)` did not fall back to the default HTTPX factory.
  3. Fallback label detection was too aggressive when the fallback only appeared
     in secondary candidates.
  4. Legacy CODA skill-store contents polluted DTPQA runs; clean DTPQA replay
     requires an empty skill store.
- Added resumable replay support (`offset`, `run_id`, `append`) so failed DTPQA
  runs can continue without rerunning earlier samples.
- Added configurable `EDGE_MAX_COMPLETION_TOKENS` to support latency-focused
  experiments.
- Fixed DTPQA exact-match evaluation for benchmark-style `Yes/No + explanation`
  answers.
- Fixed SiliconFlow request overrides so `Qwen3-VL` models do not receive the
  unsupported `enable_thinking` parameter.
- Added DTPQA-specific prompt guidance so the edge agent answers the benchmark
  question directly instead of only naming a hazard label.

## 2026-03-31

- Completed a clean `synth/category_1` 7-sample distance sweep in
  [run-dtpqa-synth-cat1-kimi512](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-synth-cat1-kimi512)
  using:
  1. `EDGE_MODEL=Pro/moonshotai/Kimi-K2.5`
  2. `EDGE_MAX_COMPLETION_TOKENS=512`
  3. `SKILL_STORE_DIR=/tmp/dtpqa_skills_empty`
- Quick metrics from the completed replay:
  1. `7/7` exact match accuracy on distances `50m, 40m, 30m, 20m, 10m, 5m, unknown`.
  2. Mean latency `135,025 ms`.
  3. Distance-group mean latency: `far=96,084 ms`, `mid=89,173 ms`,
     `near=164,901 ms`, `unknown=169,132 ms`.
- Compared edge configurations on the hard 10m sample:
  1. `Qwen/Qwen3.5-9B` timed out even with `EDGE_MAX_COMPLETION_TOKENS=512`.
  2. `Pro/moonshotai/Kimi-K2.5` completed reliably, though slowly.
  3. `Qwen/Qwen3-VL-8B-Instruct` became API-compatible after the override fix,
     but still showed unstable latency and was not chosen as the baseline model.
- Added incremental persistence to DTPQA judge evaluation so partial judge
  progress survives transient network failures. Judge scoring is still in
  progress / retry territory because remote API connections remain unstable.
- Completed judge scoring for
  [run-dtpqa-synth-cat1-kimi512](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-synth-cat1-kimi512):
  1. `judge_score_mean = 94.57`
  2. distance-group judge means: `far=95`, `mid=95`, `near=95`, `unknown=92`
  3. no skill/reflection path was activated (`skill_success_rate = 0.0`)
- Started a `real/category_1` pilot run in
  [run-dtpqa-real-cat1-kimi512-pilot](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-real-cat1-kimi512-pilot).
  The first real sample (`annotations-9368`, `10m`, near) completed with
  `exact_match = 1.0` and latency `92,990 ms`, but the second sample entered a
  long-tail stall and the run was interrupted deliberately so it can be resumed
  later via `append`.
- Resumed the real/category_1 pilot sample-by-sample and completed a 2-sample
  evaluated subset:
  1. [run-dtpqa-real-cat1-kimi512-pilot](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-real-cat1-kimi512-pilot)
  2. `exact_match_accuracy = 1.0`
  3. `judge_score_mean = 95.0`
  4. `mean_latency_ms = 110,297`
  5. completed distances: `10m (near)`, `30m (mid)`
- Attempted the third real/category_1 case (`annotations-9370`) with
  single-sample append mode and an extended timeout, but it still entered a
  long-tail stall and was interrupted deliberately so the pilot can continue
  later without losing completed cases.
- Added
  [run_real_cat1_offsets.sh](/Users/didi/Code/Autonomous_Driving/experiments/dtpqa-integration/code/run_real_cat1_offsets.sh)
  so real/category_1 offsets can be replayed in resumable batches without
  retyping the full environment every time. Fixed the script so `SIGINT`
  terminates the batch instead of rolling into later offsets.
- Found an important benchmark-integrity issue while restarting the MCP server:
  the server must also use an empty skill store. A non-empty server-side skill
  store immediately injected historical CODA skills into DTPQA replay and
  invalidated the benchmark setting, so the server was restarted with
  `/tmp/dtpqa_skills_clean_20260331`.
- Expanded the real/category_1 pilot with clean single-sample append runs:
  1. `annotations-9376` (`far`, `50m`, GT `Yes`) -> predicted `No`,
     `judge_score=5`, `latency_ms=7548`
  2. `annotations-9377` (`far`, `50m`, GT `Yes`) -> predicted `No`,
     `judge_score=5`, `latency_ms=7862`
  3. `annotations-9378` (`far`, `40m`, GT `Yes`) -> predicted `No`,
     `judge_score=0`, `latency_ms=9225`
  4. `annotations-9381` (`mid`, `30m`, GT `Yes`) -> predicted `No`,
     `judge_score=5`, `latency_ms=22980`
  5. `annotations-9375` (`near`, `20m`, GT `Yes`) -> predicted `No`,
     `judge_score=0`, `latency_ms=9114`
  6. `annotations-11749` (`unknown`, GT `No`) -> predicted `No`,
     `judge_score=100`, `latency_ms=19491`
  7. `annotations-11768` (`unknown`, GT `No`) -> predicted `No`,
     `judge_score=100`, `latency_ms=21654`
- Completed formal judge scoring for the expanded
  [run-dtpqa-real-cat1-kimi512-pilot](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-real-cat1-kimi512-pilot):
  1. `total_cases = 9`
  2. `exact_match_accuracy = 0.4444`
  3. `judge_score_mean = 45.0`
  4. distance-group accuracy: `near=0.5`, `mid=0.5`, `far=0.0`, `unknown=1.0`
  5. distance-group judge means: `near=47.5`, `mid=50.0`, `far=3.33`,
     `unknown=100.0`
  6. mean latency fell to about `35,385 ms`, largely because many of the wrong
     positive cases were answered quickly as `No`
- The initial 2-sample real pilot was therefore misleadingly optimistic. The
  current clean 9-sample slice is the first direct real-data evidence that the
  synth/category_1 baseline does not transfer cleanly to real DTPQA.
- Diagnosed the main real-benchmark control failure more precisely:
  1. `ReplayOrchestrator` only triggered reflection on entropy or fallback,
     which missed the dominant low-entropy `Yes -> No` false negatives.
  2. `CloudReflector` leaked `ground_truth_answer` into the reflection prompt.
  3. The clean MCP server on port `8002` was started without
     `REQUEST_TIMEOUT_SECONDS=300`, so server-side reflection could still time
     out even when the replay client waited longer.
  4. Reflection originally returned symbolic labels (for example
     `Distant_Unknown_Obstacle_Potentially_VRU`) that do not satisfy DTPQA
     yes/no evaluation.
  5. Reflection also attempted to compile and persist new skills during DTPQA
     benchmark replay, which would contaminate later benchmark cases.
- Implemented and tested the following code changes:
  1. DTPQA/category_1 `No` answers now trigger reflection through an explicit
     runtime-controlled heuristic instead of relying only on entropy.
  2. `ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER` was added so same-day `off/on`
     ablations can isolate code effects from remote model drift.
  3. `ground_truth_answer` was removed from the reflection prompt.
  4. DTPQA reflection now disables skill persistence and adds benchmark-aware
     `Yes/No` formatting guidance.
  5. New tests cover trigger routing, leakage removal, disabled persistence,
     and DTPQA reflection answer formatting.
- Built paired-comparison tooling in
  [compare_real_cat1_runs.py](/Users/didi/Code/Autonomous_Driving/experiments/dtpqa-integration/code/compare_real_cat1_runs.py)
  and used it to compare the historical baseline with the partial
  `run-dtpqa-real-cat1-kimi512-intervention-quick` replay. That comparison
  showed same-case drift across days (`far` improved while `unknown`
  regressed), confirming that cross-day pre/post claims are not scientifically
  reliable.
- Verified the new DTPQA-aware reflection behavior directly through MCP probes
  on historical records rather than waiting for a full replay:
  1. Historical false negative `annotations-9594` now reflects from baseline
     `No` to corrected label `Yes` without skill persistence.
  2. Historical true negative `annotations-11749` remains `No` under the same
     reflection path.
- Found another benchmark-integrity failure while starting the same-day control
  run:
  1. The supposedly clean skill store `/tmp/dtpqa_skills_clean_20260331`
     already contained a persisted DTPQA-derived skill from earlier debugging.
  2. As a result, the first `sameday-off` control run was contaminated by
     `matched_skill_ids=["distant-roadside-pedestrians-annotations-9594-6cdc8682"]`
     and is not valid as a benchmark baseline.
  3. The contaminated run was stopped and retained only as diagnostic evidence.
  4. A new empty skill store `/tmp/dtpqa_skills_clean_20260331T173713` was
     created, the clean MCP server on port `8003` was restarted against that
     store, and the formal same-day baseline was restarted as
     `run-dtpqa-real-cat1-kimi512-sameday-off-quick-v2`.
- 2026-04-10 inner-loop: audited the failed synth hybrid rerun
  `dtpqa_synth200_20260409_213139_hybrid_rerun` and found that the apparent
  `77.3%` final accuracy regression was not scientifically clean. The run wrote
  only 97 records because `run_large_scale_200.sh` piped `ad-replay-dtpqa`
  through `tail` without `pipefail`, and the live replay client still defaulted
  to `http://127.0.0.1:8000/mcp` while the script health-checked `8003`. The
  `/tmp/..._skills` store stayed empty while matched skill IDs came from
  historical `data/skills`, confirming server-side contamination.
- 2026-04-10 inner-loop: implemented experiment-integrity fixes:
  1. `run_large_scale_200.sh` now uses `set -euo pipefail`, passes
     `--server-url`, starts an isolated MCP server on an explicit port/URL, and
     logs failed offsets instead of silently skipping them.
  2. MCP health checks now treat `406 Not Acceptable` as a live MCP endpoint
     rather than a dead server.
  3. DTPQA reflection labels are normalized back to exact `Yes/No`, and
     malformed reflection outputs no longer persist skills.
  4. DTPQA/category_1 reflection triggering is now more conservative: `No`
     answers reflect only when there is person evidence rather than on every
     category_1 negative.
- 2026-04-10 inner-loop: finished the first clean synth/category_1
  `edge_only` sweep on 50 offsets as
  `dtpqa_synth50_20260410_edge_only_v1`. Exact-match accuracy is `0.90`
  overall, with `far=0.7333`, `mid=1.0`, `near=0.9524`, `unknown=1.0`. All
  observed errors in this slice are far positives (`offsets 7, 14, 15`).
- 2026-04-10 inner-loop: ran targeted clean single-offset experiments to
  diagnose reflection regressions on the three far false negatives.
  Results:
  1. Old clean hybrid reflection remained wrong on `7`, `14`, and `15`.
  2. `cloud_only` was correct on `7` and `15`, but still wrong on `14`.
  3. Therefore the old reflection path was wasting cloud-model capability on at
     least two of the three targeted errors.
- 2026-04-10 inner-loop: replaced DTPQA/category_1 MCP text reflection with
  direct cloud re-perception inside hybrid mode, while preserving the same
  trigger. The new targeted runs
  `dtpqa_target_offset7_hybrid_v2` and `dtpqa_target_offset15_hybrid_v2`
  recovered both far false negatives (`0 -> 1`), while
  `dtpqa_target_offset14_hybrid_v2` remained wrong. This supports H5: the old
  regression was partly architectural anchoring, but a residual subset still
  reflects true cloud-model visual-semantic failure.
- 2026-04-10 outer-loop: synthesized the new failure taxonomy.
  Direction: DEEPEN. Immediate next step is to finish the clean 50-sample
  `cloud_only` and `hybrid_direct_cloud` runs on shared offsets, quantify how
  much of the previous hybrid regression came from contamination versus true
  model error, and then decide whether DTPQA benchmark runs should permanently
  use direct cloud re-perception as the paper's hybrid intervention.
- 2026-04-10 outer-loop: extracted published DTPQA baselines from the source
  papers. The dataset paper (`Descriptor: Distance-Annotated Traffic
  Perception Question Answering`) points to the evaluation paper
  `Evaluating Small Vision-Language Models on Distance-Dependent Traffic
  Perception` for model baselines. From Table 4 and Table 5 of that paper:
  1. Best published small-VLM full-benchmark average is `59.4`
     (`Ovis2-2B` and `InternVL2.5-2B-MPO`).
  2. Best published small-VLM `DTP-Synth / Cat.1` score is `71.5`
     (`Ovis2-2B`).
  3. Negative-sample specificity on `Cat.1-Synth` is essentially saturated for
     all published small models, so the main discriminative signal remains
     positive recall / far-distance perception.
- 2026-04-10 outer-loop: built a thesis-oriented comparison report at
  `experiments/dtpqa-integration/results/thesis_synth50_three_way_20260410.md`
  and corresponding JSON. The report now includes:
  1. core three-way metrics,
  2. selective-routing metrics (`rescue_rate`, `harm_rate`,
     `gain_capture_vs_cloud`, cloud-call reduction),
  3. skill / routing metrics (`skill_match_rate`, `reflection_precision`),
  4. a direct table that aligns our current `Cat.1-Synth` runs with published
     DTPQA baselines.
