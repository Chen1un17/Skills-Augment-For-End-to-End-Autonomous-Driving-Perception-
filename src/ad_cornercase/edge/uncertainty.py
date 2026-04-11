"""Uncertainty estimation helpers."""

from __future__ import annotations

import math

from ad_cornercase.schemas.common import CandidateLabel


def normalized_entropy(candidates: list[CandidateLabel]) -> float:
    if not candidates:
        return 0.0
    probabilities = [max(candidate.probability, 1e-9) for candidate in candidates]
    total = sum(probabilities) or 1.0
    normalized = [probability / total for probability in probabilities]
    entropy = -sum(probability * math.log(probability) for probability in normalized)
    max_entropy = math.log(len(normalized)) if len(normalized) > 1 else 1.0
    return entropy / max_entropy if max_entropy else entropy
