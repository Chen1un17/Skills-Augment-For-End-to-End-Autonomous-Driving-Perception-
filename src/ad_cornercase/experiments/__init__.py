"""Automated experiment orchestration framework."""

from __future__ import annotations

from .runner import ExperimentRunner
from .config import ExperimentConfig, ModelConfig, DatasetConfig, get_preset
from .monitor import ExperimentMonitor
from .report import ReportGenerator
from .batch_runner import LargeScaleBatchRunner, BatchExperiment
from .iterative_optimizer import IterativeOptimizer, AutomatedResearchLoop

__all__ = [
    "ExperimentRunner",
    "ExperimentConfig",
    "ModelConfig",
    "DatasetConfig",
    "get_preset",
    "ExperimentMonitor",
    "ReportGenerator",
    "LargeScaleBatchRunner",
    "BatchExperiment",
    "IterativeOptimizer",
    "AutomatedResearchLoop",
]