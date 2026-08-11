"""Deterministic training and persistence for the smallest learned scorer."""

import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_cutting_stock.benchmarks import DATASET_SCHEMA_VERSION, load_phase3_dataset

from .features import FEATURE_SCHEMA_VERSION, pricing_features
from .interfaces import PatternCandidate, PricingState
from .model import MODEL_SCHEMA_VERSION, LinearColumnScoringModel

TRAINING_ARTIFACT_SCHEMA_VERSION = "linear-training-artifact-v1"


def train_artifact(manifest_path: str | Path, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    """Train from the validated training partition and return a JSON-ready artifact."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")

    dataset = load_phase3_dataset(manifest_path)
    examples = tuple(example for example in dataset.examples if example.partition.value == "train")
    if not examples:
        raise ValueError("training partition contains no candidate examples")

    rows = []
    targets = []
    for example in examples:
        state = PricingState(
            instance_id=example.instance_id,
            iteration_index=example.iteration_index,
            stock_length=example.stock_length,
            kerf=example.kerf,
            piece_lengths=example.piece_lengths,
            demands=example.demands,
            dual_values=example.dual_values,
            current_patterns=example.current_patterns,
            rmp_objective_value=example.rmp_objective_value,
        )
        candidate = PatternCandidate(example.candidate_pattern, example.candidate_reduced_cost)
        rows.append(pricing_features(state, candidate))
        targets.append(float(example.selected))

    model = LinearColumnScoringModel.fit(rows, targets)
    return {
        "schema_version": TRAINING_ARTIFACT_SCHEMA_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "seed": seed,
        "config": config,
        "metadata": {
            "manifest": str(Path(manifest_path)),
            "partition": "train",
            "example_count": len(examples),
            "trajectory_ids": sorted({example.trajectory_id for example in examples}),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "interpreter": sys.executable,
        },
        "model": {
            "feature_width": model.feature_width,
            "weights": list(model.weights),
            "bias": model.bias,
        },
    }


def write_training_artifact(
    manifest_path: str | Path, output_path: str | Path, seed: int, config: dict[str, Any]
) -> dict[str, Any]:
    """Train and persist an artifact with stable JSON formatting."""

    artifact = train_artifact(manifest_path, seed, config)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def load_training_artifact(path: str | Path) -> LinearColumnScoringModel:
    """Load a compatible training artifact and reconstruct its scoring model.

    Artifacts are intentionally rejected when any interface version differs. This prevents a
    model trained with a different feature or dataset contract from being used silently.
    """

    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid training artifact: {path}") from error
    if not isinstance(artifact, dict):
        raise ValueError("training artifact must be a JSON object")

    expected_versions = {
        "schema_version": TRAINING_ARTIFACT_SCHEMA_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
    }
    for field, expected in expected_versions.items():
        actual = artifact.get(field)
        if actual != expected:
            raise ValueError(f"unsupported {field}: expected {expected!r}, got {actual!r}")

    model_data = artifact.get("model")
    if not isinstance(model_data, Mapping):
        raise ValueError("training artifact model must be a JSON object")
    weights = model_data.get("weights")
    bias = model_data.get("bias")
    feature_width = model_data.get("feature_width")
    if (
        isinstance(feature_width, bool)
        or not isinstance(feature_width, int)
        or feature_width <= 0
    ):
        raise ValueError("model feature_width must be a positive integer")
    if not isinstance(weights, list) or len(weights) != feature_width:
        raise ValueError("model feature_width must match the weights")
    try:
        return LinearColumnScoringModel(weights, bias)
    except (TypeError, ValueError) as error:
        raise ValueError("training artifact contains an invalid model") from error
