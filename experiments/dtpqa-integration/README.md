# DTPQA Real Dataset: Large-Scale Automated Research

This directory contains the automated experiment framework for large-scale evaluation on the DTPQA **real** dataset (only), following autoresearch principles.

## Quick Start

### 1. Start MCP Server

```bash
uv run ad-mcp-server
```

### 2. Run Automated Research Pipeline

```bash
# Full automated pipeline (baseline → reflection → analysis → report)
python experiments/dtpqa-integration/code/launch_full_research.py --auto

# Or with sample limit for testing
python experiments/dtpqa-integration/code/launch_full_research.py --auto --limit 100
```

### 3. Monitor Progress

```bash
# Watch all recent experiments
python experiments/dtpqa-integration/code/monitor_dashboard.py --watch

# Or monitor specific run
python experiments/dtpqa-integration/code/monitor_dashboard.py --run-id <run_id>
```

## Experiment Modes

### Baseline (No Reflection)

```bash
python experiments/dtpqa-integration/code/launch_full_research.py --baseline [--limit N]
```

Tests edge-only inference without cloud reflection.

### With Reflection

```bash
python experiments/dtpqa-integration/code/launch_full_research.py --reflection [--limit N]
```

Tests edge-cloud hierarchical system with reflection enabled.

### Full Pipeline

```bash
python experiments/dtpqa-integration/code/launch_full_research.py --auto --target-accuracy 0.7
```

Runs complete autoresearch pipeline:
1. **Phase 1**: Baseline experiment
2. **Phase 2**: Reflection experiment
3. **Phase 3**: Outer loop analysis (synthesize findings)
4. **Phase 4**: Iterative optimization (if needed)
5. **Phase 5**: Generate academic report

## Directory Structure

```
dtpqa-integration/
├── code/
│   ├── launch_full_research.py      # Main research launcher
│   ├── run_large_scale_experiments.py  # Experiment runner
│   ├── run_experiment.sh            # Shell wrapper
│   ├── monitor_dashboard.py         # Real-time monitoring
│   └── run_real_cat1_offsets.sh     # Legacy offset runner
├── configs/                         # Experiment configurations
├── results/                         # Experiment results & status
├── report/                          # Generated reports
├── protocols/                       # Locked experiment protocols
└── README.md                        # This file
```

## Autoresearch Architecture

### Inner Loop (Fast Iteration)

- Run constrained experiments with clear metrics
- Measure: exact_match_accuracy, judge_score, latency
- Distance-stratified analysis (near/mid/far/unknown)
- Checkpoint/resume for long-running experiments

### Outer Loop (Synthesis)

- Review results and find patterns
- Compare baseline vs intervention
- Decide direction: DEEPEN / BROADEN / PIVOT / CONCLUDE
- Generate new hypotheses
- Update findings.md

## Key Features

### Real-time Monitoring

```bash
# Text-based dashboard
python experiments/dtpqa-integration/code/monitor_dashboard.py --watch

# Generate progress report
python experiments/dtpqa-integration/code/monitor_dashboard.py --report <run_id>
```

### Iterative Optimization

The optimizer automatically:
1. Analyzes failure patterns
2. Generates hypotheses
3. Proposes improved configurations
4. Runs next iteration

### Academic Report Generation

Generates:
- Markdown reports with tables
- LaTeX tables for papers
- Comparison plots (matplotlib)
- JSON metrics for further analysis

## Configuration

Edit `src/ad_cornercase/experiments/config.py` for presets:

```python
EXPERIMENT_PRESETS = {
    "dtpqa_real_baseline": ExperimentConfig(...),
    "dtpqa_real_reflection": ExperimentConfig(...),
}
```

## Environment Variables

```bash
export EDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export EDGE_MAX_COMPLETION_TOKENS=512
export MCP_SERVER_PORT=8003
export SKILL_STORE_DIR=/tmp/dtpqa_skills_empty
export DTPQA_ROOT=./data/dtpqa
```

## Results Location

- Raw artifacts: `./data/artifacts/<run_id>/`
- Experiment configs: `./experiments/dtpqa-integration/configs/`
- Generated reports: `./experiments/dtpqa-integration/report/`
- Analysis results: `./experiments/dtpqa-integration/results/`

## Example Workflow

```bash
# 1. Quick test with 50 samples
python experiments/dtpqa-integration/code/launch_full_research.py --auto --limit 50

# 2. Check results
ls -la experiments/dtpqa-integration/report/

# 3. View summary
cat experiments/dtpqa-integration/report/summary.md

# 4. Full run (all real dataset)
python experiments/dtpqa-integration/code/launch_full_research.py --auto
```

## Dataset Constraint

**IMPORTANT**: This framework is configured to use **ONLY** the DTPQA real dataset (`subset="real"`), as specified. The synthetic dataset is excluded from all experiments.

## Troubleshooting

### MCP Server Not Running

```bash
# Start in background
uv run ad-mcp-server &
```

### Skill Store Not Empty

```bash
# Clean skill store
rm -rf /tmp/dtpqa_skills_empty/*
```

### Resume Interrupted Experiment

All experiments support automatic resume. Just re-run the same command.

## Citation

When using this framework, cite the autoresearch methodology:

```bibtex
@misc{autoresearch2025,
  title={Automated Research Framework for Edge-Cloud VLM Evaluation},
  author={AD Corner Case Research Team},
  year={2025}
}
```
