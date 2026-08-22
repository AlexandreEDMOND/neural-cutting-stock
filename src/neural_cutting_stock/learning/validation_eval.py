"""Offline validation-partition evaluation of the trained quality policy.

Phase 9 measures quality only: the trained deep policy is applied, offline
and without any wall-clock criterion, to every instance of one frozen
phase-8 quality partition. Each instance is refined exactly like during
integration — a real classical column-generation start, agent proposals,
systematic independent review — through :func:`attempt_quality_refinement`,
so every attempt is either a verified published solution or a preserved
failure that stays visible in the report and never silently drops out of an
aggregate.

The reported quantity is the mean number of stock bars saved against the
classical restricted-integer baseline, aggregated over the whole partition,
by ``family_label`` and by piece-type count (the size dimension declared by
the retained phase-8 families). No duration enters the report: quality is
the metric under evaluation here.
"""

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_cutting_stock.benchmarks.quality_partitions import (
    SEED_PARTITIONS,
    materialize_partition_instances,
)
from neural_cutting_stock.benchmarks.quality_partitions import (
    validate_quality_partition_manifest as _validate_manifest,
)
from neural_cutting_stock.problem import AnyCuttingStockInstance
from neural_cutting_stock.solver import MaximalPatternLimits

from .neural_qc import NeuralQCBudget
from .publication import (
    OUTCOME_FAILURE,
    OUTCOME_SOLUTION,
    PUBLICATION_STATUS_EQUAL,
    PUBLICATION_STATUS_IMPROVED,
    NeuralQCAttemptRecord,
    attempt_quality_refinement,
)
from .quality_agent import QualityAgent
from .reproducibility import TrainingCurves, load_checkpoint, restore_module_state
from .rl_policy import QualityPolicyNetwork, QualityRLPolicy, RLQualityAgent

NEURAL_QC_VALIDATION_EVAL_SCHEMA_VERSION = "neural-qc-validation-eval-v1"

_CHECKPOINT_WEIGHT_KEY = "body.0.weight"


def quality_agent_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    pattern_limits: MaximalPatternLimits | None = None,
) -> RLQualityAgent:
    """Rebuild the greedy inference agent from one versioned checkpoint.

    The feature width is recovered from the persisted weight shapes and the
    hidden width from the recorded training configuration; the validated
    checkpoint reader rejects foreign or truncated artefacts before any
    weight reaches the network.
    """

    payload = load_checkpoint(checkpoint_path)
    config = payload["config"]
    hidden_width = config.get("hidden_width")
    if isinstance(hidden_width, bool) or not isinstance(hidden_width, int) or hidden_width < 1:
        raise ValueError("checkpoint config must declare a positive integer hidden_width")
    weights = payload["model_state_dict"].get(_CHECKPOINT_WEIGHT_KEY)
    if weights is None or len(weights.shape) != 2:
        raise ValueError(
            f"checkpoint model_state_dict lacks usable {_CHECKPOINT_WEIGHT_KEY} weights"
        )
    feature_width = int(weights.shape[1])
    module = QualityPolicyNetwork(feature_width, hidden_width)
    restore_module_state(module, payload)
    policy = QualityRLPolicy(
        module=module,
        feature_width=feature_width,
        seed=payload["seed"],
        config=dict(config),
        curves=TrainingCurves.from_payload(payload["curves"]),
        episodes=(),
        environment=dict(payload["environment"]),
    )
    return RLQualityAgent(policy, pattern_limits=pattern_limits)


def checkpoint_sha256(checkpoint_path: str | Path) -> str:
    """Return the SHA-256 of a checkpoint file for provenance records."""

    return hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()


def evaluate_quality_agent_on_partition(
    manifest: Mapping[str, Any],
    partition: str,
    agent: QualityAgent,
    *,
    budget: NeuralQCBudget,
    verification_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Run one refinement attempt per instance of a frozen partition.

    Every attempt flows through the publication guardrails, so the returned
    report carries verified solutions and preserved failures side by side,
    plus mean bars saved overall, by family and by size. Failures never feed
    a mean; they stay counted with their machine-readable reasons instead.
    """

    _validate_manifest(manifest)
    if not isinstance(partition, str) or partition not in SEED_PARTITIONS:
        raise ValueError(f"unknown partition: {partition!r}")
    instances = materialize_partition_instances(manifest, partition)
    metadata = {
        assignment["instance_id"]: assignment
        for assignment in manifest["assignments"]
        if assignment["partition"] == partition
    }

    entries = [
        _entry(
            instance_id,
            instances[instance_id],
            metadata[instance_id],
            agent,
            budget=budget,
            verification_tolerance=verification_tolerance,
        )
        for instance_id in sorted(instances)
    ]
    return {
        "schema_version": NEURAL_QC_VALIDATION_EVAL_SCHEMA_VERSION,
        "plan_id": manifest["plan_id"],
        "partition": partition,
        "budget": {"max_steps": budget.max_steps, "stall_patience": budget.stall_patience},
        "verification_tolerance": verification_tolerance,
        "counts": {
            "instance_count": len(entries),
            "published_solution_count": sum(
                entry["outcome"] == OUTCOME_SOLUTION for entry in entries
            ),
            "preserved_failure_count": sum(
                entry["outcome"] == OUTCOME_FAILURE for entry in entries
            ),
        },
        "overall": _group(None, entries),
        "by_family": sorted(
            (
                _group(label, [entry for entry in entries if entry["family_label"] == label])
                for label in {entry["family_label"] for entry in entries}
            ),
            key=lambda group: group["key"],
        ),
        "by_size": sorted(
            (
                _group(
                    types,
                    [entry for entry in entries if entry["number_of_piece_types"] == types],
                )
                for types in {entry["number_of_piece_types"] for entry in entries}
            ),
            key=lambda group: group["key"],
        ),
        "instances": entries,
    }


def _entry(
    instance_id: str,
    instance: AnyCuttingStockInstance,
    assignment: Mapping[str, Any],
    agent: QualityAgent,
    *,
    budget: NeuralQCBudget,
    verification_tolerance: float,
) -> dict[str, Any]:
    record: NeuralQCAttemptRecord = attempt_quality_refinement(
        instance, instance_id, agent, budget=budget, verification_tolerance=verification_tolerance
    )
    entry: dict[str, Any] = {
        "instance_id": instance_id,
        "family_label": assignment["family_label"],
        "seed": assignment["seed"],
        "number_of_piece_types": instance.number_of_types,
        "total_demand": sum(instance.demands),
        "outcome": record.outcome,
        "initial_bars": None,
        "final_bars": None,
        "bars_saved": None,
        "status": None,
        "failure_reason": record.failure_reason,
        "failure_message": record.failure_message,
    }
    solution = record.solution
    if solution is not None:
        entry.update(
            initial_bars=solution.initial_bars,
            final_bars=solution.final_bars,
            bars_saved=solution.bars_saved,
            status=solution.status,
        )
    return entry


def _group(key: Any, entries: list[dict[str, Any]]) -> dict[str, Any]:
    solutions = [entry for entry in entries if entry["outcome"] == OUTCOME_SOLUTION]
    total_bars_saved = sum(entry["bars_saved"] for entry in solutions)
    reasons: dict[str, int] = defaultdict(int)
    for entry in entries:
        reason = entry["failure_reason"]
        if reason is not None:
            reasons[reason] += 1
    mean = total_bars_saved / len(solutions) if solutions else None
    return {
        "key": key,
        "instance_count": len(entries),
        "published_solution_count": len(solutions),
        "preserved_failure_count": len(entries) - len(solutions),
        "failure_reasons": dict(sorted(reasons.items())),
        "improved_count": sum(
            entry["status"] == PUBLICATION_STATUS_IMPROVED for entry in solutions
        ),
        "equal_count": sum(entry["status"] == PUBLICATION_STATUS_EQUAL for entry in solutions),
        "total_bars_saved": total_bars_saved,
        "mean_bars_saved": mean,
    }


__all__ = [
    "NEURAL_QC_VALIDATION_EVAL_SCHEMA_VERSION",
    "checkpoint_sha256",
    "evaluate_quality_agent_on_partition",
    "quality_agent_from_checkpoint",
]
