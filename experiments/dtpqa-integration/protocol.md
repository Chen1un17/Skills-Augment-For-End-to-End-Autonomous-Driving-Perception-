# DTPQA Integration Protocol

## Question

Can the current AD corner-case prototype be extended to support DTPQA without
forking the replay pipeline, while preserving benchmark metadata needed for
distance-aware evaluation?

## Why This Matters

- The repository documents DTPQA as a target benchmark but currently ships only
  a placeholder loader.
- DTPQA is the clearest path to large-scale validation beyond the existing
  CODA-LM demo-scale artifacts.

## Locked Predictions

1. A tolerant DTPQA loader can normalize real/synthetic annotation variants into
   `AnomalyCase`.
2. The existing replay pipeline will run on DTPQA once case metadata is
   preserved in artifacts.
3. Distance-stratified metrics will expose meaningful performance differences
   that are invisible in the current aggregate-only summary.

## Success Criteria

- DTPQA loader implemented with fixture-backed tests.
- CLI commands exist to replay and evaluate DTPQA runs.
- Evaluation output includes distance-aware metrics in addition to the existing
  aggregate metrics.
- Documentation explains how to place/download the dataset and run the pipeline.

## Known Constraints

- No usable Python 3.11 runtime is available yet in the environment.
- Real dataset download requires network access and possibly user approval.
