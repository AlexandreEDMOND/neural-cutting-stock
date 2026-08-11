"""Small supervised scorer for pricing states and pattern candidates."""

import math
from collections.abc import Sequence

import numpy as np

from .features import FEATURE_SCHEMA_VERSION, pricing_features
from .interfaces import PatternCandidate, PatternScore, PricingState

MODEL_SCHEMA_VERSION = "linear-scorer-v1"


class LinearColumnScoringModel:
    """Deterministic linear regression over the versioned pricing features.

    The model only assigns scores. It does not select columns or certify pricing convergence.
    """

    def __init__(self, weights: Sequence[float], bias: float) -> None:
        self._weights = _finite_vector(weights, "weights")
        if not math.isfinite(bias):
            raise ValueError("bias must be finite")
        self._bias = float(bias)

    @classmethod
    def fit(
        cls, feature_rows: Sequence[Sequence[float]], targets: Sequence[float]
    ) -> "LinearColumnScoringModel":
        """Fit the scorer by ordinary least squares on supervised examples."""

        rows = tuple(_finite_vector(row, "feature row") for row in feature_rows)
        if not rows:
            raise ValueError("feature_rows must not be empty")
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError("feature_rows must have a consistent non-zero width")
        values = _finite_vector(targets, "targets")
        if len(values) != len(rows):
            raise ValueError("targets must have one value per feature row")

        design = np.column_stack((np.asarray(rows, dtype=float), np.ones(len(rows))))
        coefficients, _, _, _ = np.linalg.lstsq(design, np.asarray(values), rcond=None)
        return cls(coefficients[:-1], float(coefficients[-1]))

    @property
    def feature_width(self) -> int:
        return len(self._weights)

    def score(
        self, state: PricingState, candidates: Sequence[PatternCandidate]
    ) -> tuple[PatternScore, ...]:
        """Score candidates in input order using the state and candidate features."""

        scores = []
        weights = np.asarray(self._weights)
        for candidate in candidates:
            features = pricing_features(state, candidate)
            if len(features) != self.feature_width:
                raise ValueError(
                    f"feature width differs from model: expected {self.feature_width}, "
                    f"got {len(features)} ({FEATURE_SCHEMA_VERSION})"
                )
            value = float(np.dot(weights, features) + self._bias)
            scores.append(PatternScore(candidate.pattern, value))
        return tuple(scores)


def _finite_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in result
    ):
        raise ValueError(f"{name} must contain finite numbers")
    return tuple(float(value) for value in result)
