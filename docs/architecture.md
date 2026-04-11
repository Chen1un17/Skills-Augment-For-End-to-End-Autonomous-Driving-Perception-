# Architecture

## Runtime shape

The prototype uses two logical agents:

- `EdgeAgent`: replays CODA-LM samples, produces scene graph triplets, estimates uncertainty, and decides whether to invoke the cloud path.
- `CloudReflectionService`: exposed through MCP, performs skill lookup, reflection, and skill persistence.

The cloud side merges three conceptual components into one process for v1:

- MCP gateway,
- cloud reflection VLM,
- dynamic cognitive skill library.

This keeps deployment simple while preserving stable interfaces for future decomposition.

## Closed loop

1. Load a CODA-LM sample.
2. Run edge perception without skills and compute entropy.
3. Query MCP for matching skills.
4. If skills are found, run a second edge pass with prompt patches.
5. If uncertainty stays high, call cloud reflection through MCP.
6. Persist the compiled skill.
7. Re-run edge perception with the new skill.
8. Save artifacts for replay and evaluation.

## Output artifacts

- `predictions.jsonl`: per-case baseline, final prediction, skill usage, and latency fields.
- `metrics.json`: aggregate metrics.
- `report.md`: readable summary.
- `skills/<skill_id>/`: `manifest.json` plus `SKILL.md`.
