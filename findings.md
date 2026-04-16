# Findings

## Current Understanding

- The repository already has a functioning CODA-LM closed-loop prototype with
  edge perception, MCP-based cloud reflection, skill persistence, and aggregate
  evaluation.
- DTPQA support is now implemented end-to-end on top of the existing
  `AnomalyCase -> ReplayOrchestrator -> Evaluation` path. The repository did not
  need a parallel DTPQA stack; the key additions were dataset normalization,
  metadata preservation, benchmark-aware QA handling, and distance-aware
  reporting.
- The current clean baseline is
  [run-dtpqa-synth-cat1-kimi512](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-synth-cat1-kimi512),
  a 7-sample synth/category_1 distance sweep covering `50m, 40m, 30m, 20m,
  10m, 5m, unknown`.
- The expanded clean real pilot is
  [run-dtpqa-real-cat1-kimi512-pilot](/Users/didi/Code/Autonomous_Driving/data/artifacts/run-dtpqa-real-cat1-kimi512-pilot),
  now containing 9 fully scored real/category_1 cases spanning `near`, `mid`,
  `far`, and `unknown`.

## Patterns And Insights

- The replay pipeline is benchmark-agnostic once a sample becomes `AnomalyCase`.
- Real DTPQA execution surfaced more than just missing loaders:
  1. Replay needed resumability to survive long remote calls.
  2. Evaluation needed incremental judge persistence to survive judge-side
     network failures.
  3. Exact-match logic needed DTPQA-specific handling for `Yes/No + rationale`
     answers.

## AutoResearch Framework (2026-04-01)

### Completed Implementation
- Full automated experiment framework with two-loop architecture:
  - Inner loop: Fast iteration with checkpoint/resume
  - Outer loop: Result synthesis and hypothesis generation
- Batch runner supporting 200+ sample experiments
- Real-time monitoring dashboard
- Academic report generation (LaTeX + Markdown)
- Automated plotting and visualization

### Small-Scale Validation (7 samples)
- **test_synth_1775013498**: 3 samples, 100% accuracy
- **test_synth_1775013857**: 3 samples, 100% accuracy
- **test_single_1775014758**: 1 sample, 100% accuracy
- Distance-stratified: Both mid (20-30m) and far (30-50m) ranges achieved 100%
- Skill matching: Entropy reduced from 0.62 to 0.30 with skill application
- Average latency: ~150s per sample (API-bound)

### Blockers for Large-Scale
- MCP connection fails in batch mode due to httpx socks proxy issues
- Single-sample execution works correctly
- Fix required: Explicit proxy disable in httpx client configuration
- On the clean 7-sample synth/category_1 sweep, quick metrics are:
  1. `exact_match_accuracy = 1.0`
  2. `judge_score_mean = 94.57`
  2. `mean_latency_ms = 135,025`
  3. `far mean latency = 96,084 ms`
  4. `mid mean latency = 89,173 ms`
  5. `near mean latency = 164,901 ms`
  6. `unknown mean latency = 169,132 ms`
- Distance-group judge means are effectively flat on this sweep
  (`far=95`, `mid=95`, `near=95`, `unknown=92`), so the current evidence does
  not yet support far-range degradation on either exact match or judge score.
  The stronger early signal is operational: near and unknown-distance cases are
  substantially slower and more fragile than far/mid cases.
- That synth conclusion does not survive the broader clean real pilot. On 9
  scored real/category_1 cases, metrics fell to:
  1. `exact_match_accuracy = 0.4444`
  2. `judge_score_mean = 45.0`
  3. distance-group accuracy: `near=0.5`, `mid=0.5`, `far=0.0`, `unknown=1.0`
  4. distance-group judge means: `near=47.5`, `mid=50.0`, `far=3.33`,
     `unknown=100.0`
- The real failure mode is asymmetric. The model remains correct on both
  sampled `unknown/No` negatives, but it answered `No` on 5 of the 7 sampled
  real positive cases. This is not a generic random failure; it is a strong
  false-negative bias on real positive people-detection examples.
- The far group is the sharpest collapse: all three sampled real `far` cases
  were wrong, all with near-zero judge scores (`5, 5, 0`). This is the
  strongest current support for H2.
- The latest synth/category_1 hybrid failure was not a single bug. It split
  into two distinct mechanisms:
  1. **Benchmark-integrity regressions**: the old batch script silently skipped
     failed offsets, the health check treated MCP `406` as failure, and the
     client/server port mismatch meant "clean" hybrid runs were actually
     hitting the historical `data/skills` store.
  2. **Reflection-path regressions**: even after contamination was removed,
     MCP text reflection could still underperform because it anchored the cloud
     model to the edge model's wrong `No` rationale, occasionally returned
     malformed DTPQA answers, and could spend minutes timing out on a case that
     `cloud_only` could answer directly.
- Clean targeted synth experiments on far false negatives (`offsets 7, 14, 15`)
  refined the diagnosis:
  1. `edge_only` was wrong on all three.
  2. `cloud_only` was correct on `7` and `15`, but still wrong on `14`.
  3. The old hybrid reflection path was wrong on all three.
  4. The new **direct cloud re-perception** hybrid recovered `7` and `15` but
     still failed on `14`.
- Therefore, the current best explanation is mixed:
  1. Part of the earlier hybrid regression was an architecture bug: cloud
     reflection was being used as a baseline-anchored critic instead of an
     independent visual re-evaluator.
  2. The remaining hard subset is a true perception problem of the cloud model
     itself, especially for ambiguous rainy-night far pedestrians that look
     sidewalk-adjacent.
- The latency story also changed after expanding the real slice. Several of the
  incorrect positive cases completed quickly (`~7.5s` to `23.0s`), so lower
  latency on real data no longer indicates better behavior. In the current
  pilot, the cheapest answers are often the wrong `No` answers.
- Clean DTPQA runs should not reuse the historical CODA skill store. It
  introduces irrelevant matches that confound benchmark analysis.
- The empty skill-store requirement applies to the MCP server, not just the
  replay client. Restarting the server without an empty store immediately
  reintroduced old CODA skills into DTPQA runs and invalidated the benchmark
  setting.

## Lessons And Constraints

- Do not build a parallel DTPQA executor unless the existing replay interfaces
  prove insufficient; interface reuse is the lowest-risk path.
- Preserve raw benchmark metadata in artifacts. Otherwise large-scale analysis
  will have to reconstruct distance bins from the original dataset repeatedly.
- For DTPQA, the prompt must explicitly require the edge model to answer the
  benchmark question directly. Otherwise some fast VLMs return hazard labels
  instead of benchmark-compatible answers.
- SiliconFlow model support details matter operationally. `Qwen3-VL` models do
  not accept the same request override parameters as text-only `Qwen3` models.
- The current stable replay configuration is:
  1. `EDGE_MODEL=Pro/moonshotai/Kimi-K2.5`
  2. `EDGE_MAX_COMPLETION_TOKENS=512`
  3. empty skill store at `/tmp/dtpqa_skills_empty`
- Judge-based evaluation remains less stable than replay. Even with replay
  stabilized, remote judge scoring still requires resumable persistence and may
  need retries across sessions.
- On the current clean baseline, the edge-cloud loop behaves like a pure edge
  benchmark. Entropy never crossed the reflection threshold, the clean skill
  store produced no matches, and `skill_success_rate` stayed at `0.0`.
- The initial 2-sample real pilot was not representative. Once the clean real
  slice was expanded, the main limitation was no longer latency alone; it was a
  large synth-to-real correctness gap.
- Operational instability still matters, but it is now secondary to a more
  important scientific finding: the model often returns fast, confident `No`
  answers on real positive cases.
- The benchmark-control path had additional hidden flaws that only surfaced
  once reflection became relevant:
  1. The original reflection prompt leaked `ground_truth_answer`.
  2. The old clean MCP server still used the default server-side request
     timeout, so reflection could fail even when replay-side timeouts were
     extended.
  3. Reflection originally returned symbolic hazard labels instead of
     DTPQA-compatible `Yes/No` answers.
  4. Reflection also attempted to persist new skills during DTPQA evaluation,
     which would contaminate later benchmark cases.
- Direct MCP reflection probes on historical records now support the revised
  intervention hypothesis more strongly than the earlier replay-only diagnosis:
  1. Historical false negative `annotations-9594` reflects from `No` to `Yes`
     with the DTPQA-aware prompt.
  2. Historical true negative `annotations-11749` stays `No` under the same
     reflection path.
  3. DTPQA reflection no longer persists skills, so benchmark cleanliness is
     preserved.
- A clean client-side `SKILL_STORE_DIR` is not enough. The MCP server must be
  started on a known port/URL with the same empty store, otherwise DTPQA runs
  quietly inherit `data/skills` and any conclusions about hybrid/skills become
  invalid.
- For DTPQA yes/no benchmarking, classic text reflection is too brittle as the
  primary intervention on hard false negatives. A direct cloud re-perception
  path is currently more reliable because it avoids anchoring on the edge
  model's wrong explanation and avoids malformed fallback labels.
- Direct cloud re-perception for DTPQA/category_1 is now a runtime-controlled
  option instead of a hard-wired behavior. That preserves the benchmark-faithful
  default while allowing a separate shadow refinement workflow to disable the
  reroute and test MCP reflection plus skill persistence when adaptation
  evidence is the goal.
- The new synth/category_1 clean baseline with `Qwen/Qwen3.5-9B` is stronger
  than the real pilot but still exhibits the same structural weakness: the
  first 50-sample edge-only run reached `0.90` exact match overall, yet far
  accuracy was only `0.733`, and all observed errors were far positives.
- Published DTPQA baselines confirm that the benchmark is genuinely difficult
  for small VLMs, especially on pedestrian presence. In the original DTPQA
  evaluation paper, the best published small-VLM full-benchmark average is
  `59.4`, and the best published `DTP-Synth / Cat.1` score is `71.5`. Our
  current clean 50-sample `Cat.1-Synth` slice already exceeds that level with
  `edge_only=90.0`, `cloud_only=96.0`, and `hybrid=98.0`, but this comparison
  must be framed carefully because our result is on a smaller targeted subset
  rather than the full published benchmark.
- For thesis writing, the strongest system-level story is not just raw
  accuracy. The clean 50-sample report now shows that hybrid routing recovers
  `80%` of edge errors with `0%` harm on edge-correct cases, reduces cloud
  usage by `82%` relative to cloud-only, and reduces mean latency by `74.9%`
  relative to cloud-only while still slightly outperforming cloud-only on this
  slice. These are the core "whole-system" metrics that support the edge-cloud
  architecture claim.
- The current benchmark-faithful DTPQA/category_1 setting does not yet produce
  meaningful skill-refinement evidence because `hybrid_clean_v3` uses direct
  cloud re-perception and intentionally avoids skill persistence on this
  benchmark path. Therefore, if the thesis needs a strong skill-refinement
  chapter, it will require a separate adaptive-learning experiment with a clean
  adaptation/evaluation split rather than reusing the benchmark-faithful run.
- A synth-only, multi-category plan-driven workflow is now implemented for the
  next stage. It freezes a shared 500-case `synth` plan across categories
  `1-6`, replays the same cases under `edge_only`, `cloud_only`, and `hybrid`,
  and builds both aggregate and category-level reports from that shared case
  set.
- The benchmark-faithful three-way comparison and the skill-refinement study
  should remain separate products. The former should keep the clean benchmark
  path, while the latter should use category-isolated skill stores and an
  adaptation/holdout split. This keeps “system comparison” and “adaptive
  learning” claims from contaminating each other.
- Cross-day remote-model drift is real and large enough to invalidate naive
  pre/post comparisons. A paired comparison between the historical runs and the
  partial `run-dtpqa-real-cat1-kimi512-intervention-quick` replay improved
  `far` cases but regressed `unknown` cases despite zero reflection activations.
  Therefore the next trustworthy comparison must be same-day `trigger off` vs
  `trigger on`.
- The balanced 18-case multi-category synth smoke is now complete and
  trustworthy. All three modes (`edge_only`, `cloud_only`, `hybrid`) produced a
  full 18-case shared-case set, and the new report builder confirmed the shared
  intersection exactly equals `18`.
- Real cloud-backed latency on synth smoke is materially higher than the old
  planning assumptions. The smoke report measured mean latency of about
  `113.2s` for `cloud_only`, and one category_5 case exceeded the previous
  `300s` timeout before succeeding under a `600s` resume. Therefore the formal
  synth-500 run must use longer request timeouts to remain robust.
- The smoke slice did not yet produce meaningful skill-refinement evidence.
  Hybrid routing was `17 edge_only_passthrough + 1 direct_cloud_perception`,
  with `0` skill matches. This is acceptable for smoke validation, but it means
  the full 500-case benchmark is still necessary to learn whether routing and
  skills activate outside category_1 under a larger shared-case set.
- On this smoke slice, `edge_only` and `hybrid` tied at `0.5556` exact match,
  while `cloud_only` reached `0.5000`. The main lesson is not yet that hybrid
  wins, but that the new plan-driven workflow, resumable artifacts, and
  multi-category report path are sound enough for the formal synth-500 run.

## Open Questions

- Why do sampled real positive cases collapse into fast `No` answers while the
  sampled real negative cases remain correct?
- Are the failing real positive cases visually harder because pedestrians are
  smaller, more off-axis, or partially occluded than the first two successful
  positives?
- Can prompt or model changes recover real positive recall without reintroducing
  old skills or benchmark contamination?
- On the remaining unresolved far cases like synth `offset 14`, is the failure
  primarily due to image evidence ambiguity, prompt framing, or a systematic
  benchmark-label / scene-interpretation mismatch?
- Should DTPQA benchmark runs permanently replace MCP text reflection with
  direct cloud re-perception, reserving skill persistence for a separate
  adaptive-learning setting rather than the benchmark-faithful comparison?
- After the DTPQA-aware reflection fixes, what is the same-day paired effect of
  enabling the broad `category_1 + No -> reflection` heuristic on real
  category_1 exact-match accuracy?
- Should the next large sweep use full exact-match scoring with only sampled
  judge evaluation, given that judge persistence works but still fails
  intermittently on longer runs?
- Whether DriveLM should stay deferred until the real DTPQA recall failure is
  understood and reduced.
