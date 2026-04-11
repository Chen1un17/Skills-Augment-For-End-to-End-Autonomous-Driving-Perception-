# AD Corner Case Prototype

This repository implements a training-free autonomous driving perception prototype for corner cases in degraded visual environments. The current deliverables are offline replay and evaluation pipelines for CODA-LM and DTPQA, with:

- an edge perception agent,
- an MCP server exposing cloud reflection and skill retrieval,
- a file-backed skill store,
- a CODA-LM oriented evaluation flow,
- and a DTPQA distance-aware evaluation flow built on the same closed loop.

## Why this shape

The design follows the architecture described in [Guide.md](/Users/didi/Code/Autonomous_Driving/Guide.md), while keeping the first implementation compact:

- all models are accessed through cloud APIs,
- the edge and cloud agents are separated logically,
- the MCP transport uses Streamable HTTP,
- the skill store is local and file-backed,
- CODA-LM remains the primary corner-case benchmark,
- and DTPQA extends validation into distance-aware traffic perception.

## Quickstart

1. Create a virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and fill in the API settings.
3. Place a CODA-LM style split under `data/coda_lm` or point `CODA_LM_ROOT` to the dataset root.
4. Place a DTPQA release under `data/dtpqa` or point `DTPQA_ROOT` to the dataset root when running distance-aware validation.
5. Start the MCP server:

```bash
uv run ad-mcp-server
```

6. Replay a small CODA-LM subset:

```bash
uv run ad-replay-coda --split Mini --task region_perception --limit 10
```

7. Evaluate a CODA-LM run:

```bash
uv run ad-eval-coda --run-id <run_id>
```

8. Replay a DTPQA subset:

```bash
uv run ad-replay-dtpqa --subset real --limit 50
```

9. Evaluate a DTPQA run:

```bash
uv run ad-eval-dtpqa --run-id <run_id>
```

## Dataset layout

CODA-LM supports two layouts:

- Original CODA-LM `.../<Split>/vqa_anno/*.jsonl` format.
- A simplified local fixture format used by the tests.

DTPQA annotations are discovered recursively and normalized from flexible
JSON/JSONL layouts into the shared replay schema. When a release contains many
annotation files, use `--annotation-glob` with `ad-replay-dtpqa` to narrow the
selection.

## Interfaces

- OpenAI-compatible Responses API is used for text-plus-image reasoning.
- MCP uses the official Python SDK with Streamable HTTP transport.
- Skills are retrieved through an MCP tool and exposed as `skill://{skill_id}` resources.

See [docs/architecture.md](/Users/didi/Code/Autonomous_Driving/docs/architecture.md), [docs/protocol_mapping.md](/Users/didi/Code/Autonomous_Driving/docs/protocol_mapping.md), and [docs/benchmark_setup.md](/Users/didi/Code/Autonomous_Driving/docs/benchmark_setup.md) for the implementation contract.
