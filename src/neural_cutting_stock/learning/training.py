"""Deterministic training and persistence for the smallest learned scorer."""

import json
import platform
import sys
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
