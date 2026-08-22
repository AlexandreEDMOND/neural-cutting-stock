"""Mandatory equal-budget ablations of the Phase 9 quality policy.

The claim that the trained deep policy contributes anything of its own is
only meaningful against baselines that share everything except the learned
component. This module provides those ablations and their partition-level
comparison:

- :class:`GreedyQualityAgent` decodes zero sampled usage counts through the
  shared deterministic completion used by the learned decoder: a bulk
  whole-pattern pass over the enumerated maximal-pattern basis in
  enumeration order, then single-bar sweeps until full coverage. No model,
  no randomness, no learning.
- :class:`RandomSearchQualityAgent` samples integer usage counts uniformly
  over ``[0, cap]`` for every enumerated candidate — exactly the capped
  action space the deep policy explores during training — and decodes them
  with the same shared completion. Each instance owns one RNG stream seeded
  from ``(seed, instance_id)``, so results never depend on which other
  instances run before or after.

Both agents flow through the systematic independent review like any other
agent output. :func:`evaluate_quality_ablations_on_partition` evaluates any
set of named agents — the learned checkpoint alongside its ablations — on
one frozen quality partition under the exact same declared budget, and
:func:`summarize_ablation_deltas` isolates what the learned reference saves
beyond each ablation with every non-paired instance preserved instead of
silently dropped. Quality stays the only metric; no duration enters these
reports.
"""

import hashlib
import random
from collections.abc import Mapping
from typing import Any

from .imitation import enumerated_candidates
from .neural_qc import NeuralQCBudget
from .quality_agent import QualityAgent, QualityAgentInput, QualityAgentProposal
from .rl_policy import _candidate_caps, _decode_proposal
from .validation_eval import evaluate_quality_agent_on_partition

QUALITY_ABLATION_EVAL_SCHEMA_VERSION = "neural-qc-ablation-eval-v1"
GREEDY_ABLATION_IDENTIFIER = "greedy-basis-completion-v1"
RANDOM_SEARCH_ABLATION_IDENTIFIER = "random-search-uniform-counts-v1"

EXCLUSION_NO_PUBLISHED_SOLUTION = "no_published_solution_for_this_pair"


class GreedyQualityAgent:
    """Deterministic greedy ablation: the shared decoder without any counts.

    Every proposal is the plan obtained by decoding all-zero usage counts
    with the deterministic bulk-plus-sweep completion over the maximal-pattern
    basis declared by the observation. It carries no guarantee of its own:
    the independent verifier may reject it, and only a verified strict bar
    reduction can ever move an incumbent.
    """

    def __init__(self, pattern_limits=None) -> None:
        self._pattern_limits = pattern_limits

    def propose(self, observation: QualityAgentInput) -> QualityAgentProposal:
        """Return the greedily completed plan for one interface observation."""

        candidates = enumerated_candidates(observation, self._pattern_limits)
        if not candidates:
            raise ValueError("candidate enumeration produced no maximal pattern")
        return _decode_proposal(candidates, [0] * len(candidates), observation.demands)


def _stream_seed(seed: int, instance_id: str) -> int:
    label = f"{RANDOM_SEARCH_ABLATION_IDENTIFIER}:{seed}:{instance_id}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


class RandomSearchQualityAgent:
    """Seeded random-search ablation over the learned action space.

    At every step each enumerated candidate receives an integer usage count
    drawn uniformly from ``[0, cap]``, where ``cap`` is the demand bound of
    the candidate — the same bounds that cap the deep policy's Poisson rates.
    The sampled counts are decoded by the shared deterministic completion, so
    the only difference from the learned pipeline is where the counts come
    from. Streams are per instance and derived deterministically from
    ``(seed, instance_id)``, making every evaluation reproducible and order
    independent.
    """

    def __init__(self, seed: int, pattern_limits=None) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self._seed = seed
        self._pattern_limits = pattern_limits
        self._streams: dict[str, random.Random] = {}

    def propose(self, observation: QualityAgentInput) -> QualityAgentProposal:
        """Return the randomly sampled plan for one interface observation."""

        candidates = enumerated_candidates(observation, self._pattern_limits)
        if not candidates:
            raise ValueError("candidate enumeration produced no maximal pattern")
        rng = self._streams.get(observation.instance_id)
        if rng is None:
            rng = random.Random(_stream_seed(self._seed, observation.instance_id))
            self._streams[observation.instance_id] = rng
        caps = _candidate_caps(observation, candidates)
        counts = [rng.randint(0, cap) for cap in caps]
        return _decode_proposal(candidates, counts, observation.demands)


def evaluate_quality_ablations_on_partition(
    manifest: Mapping[str, Any],
    partition: str,
    agents: Mapping[str, QualityAgent],
    *,
    budget: NeuralQCBudget,
    verification_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Evaluate named agents on one frozen partition under one equal budget.

    Every agent runs through the unchanged publication-guarded evaluation of
    :func:`evaluate_quality_agent_on_partition`, so verified solutions and
    preserved failures stay side by side. The returned payload keeps each
    agent's complete sub-report plus a per-instance matrix of outcomes and
    verified bars saved, ready for paired delta analysis. All agents share
    the exact same manifest, partition and :class:`NeuralQCBudget`.
    """

    if not isinstance(agents, Mapping) or not agents:
        raise ValueError("agents must be a non-empty mapping of named quality agents")
    names = sorted(agents)
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("every agent needs a non-empty string name")
        if not callable(getattr(agents[name], "propose", None)):
            raise ValueError(f"agent {name!r} must expose a propose(observation) method")

    evaluations = {
        name: evaluate_quality_agent_on_partition(
            manifest,
            partition,
            agents[name],
            budget=budget,
            verification_tolerance=verification_tolerance,
        )
        for name in names
    }
    first = evaluations[names[0]]
    return {
        "schema_version": QUALITY_ABLATION_EVAL_SCHEMA_VERSION,
        "plan_id": first["plan_id"],
        "partition": first["partition"],
        "budget": dict(first["budget"]),
        "verification_tolerance": verification_tolerance,
        "agent_names": names,
        "evaluations": evaluations,
        "per_instance": [
            {
                "instance_id": entry["instance_id"],
                "family_label": entry["family_label"],
                "number_of_piece_types": entry["number_of_piece_types"],
                "outcomes": {
                    name: evaluations[name]["instances"][index]["outcome"] for name in names
                },
                "bars_saved": {
                    name: evaluations[name]["instances"][index]["bars_saved"] for name in names
                },
            }
            for index, entry in enumerate(first["instances"])
        ],
    }


def summarize_ablation_deltas(
    report: Mapping[str, Any],
    *,
    reference_agent: str,
) -> dict[str, Any]:
    """Pair every other agent's verified savings against the reference agent.

    Deltas are computed instance by instance as ``other - reference`` bars
    saved, so positive values mean the ablation beat the reference. A pair
    only contributes when both sides published a solution; every excluded
    instance stays listed with its reason, never silently dropped.
    """

    if (
        not isinstance(report, Mapping)
        or report.get("schema_version") != QUALITY_ABLATION_EVAL_SCHEMA_VERSION
    ):
        raise ValueError("report must be a phase-9 ablation evaluation payload")
    names = report["agent_names"]
    if reference_agent not in names:
        raise ValueError(f"unknown reference agent: {reference_agent!r}")

    comparisons: dict[str, Any] = {}
    for name in names:
        if name == reference_agent:
            continue
        deltas: list[int] = []
        excluded: dict[str, str] = {}
        for cell in report["per_instance"]:
            reference_saved = cell["bars_saved"][reference_agent]
            candidate_saved = cell["bars_saved"][name]
            if reference_saved is None or candidate_saved is None:
                excluded[cell["instance_id"]] = EXCLUSION_NO_PUBLISHED_SOLUTION
            else:
                deltas.append(candidate_saved - reference_saved)
        total = sum(deltas)
        comparisons[name] = {
            "paired_instance_count": len(deltas),
            "excluded_instances": dict(sorted(excluded.items())),
            "delta_total_bars_saved": total,
            "delta_mean_bars_saved": total / len(deltas) if deltas else None,
            "instances_where_reference_saves_more": sum(delta < 0 for delta in deltas),
            "equal_instances": sum(delta == 0 for delta in deltas),
            "instances_where_candidate_saves_more": sum(delta > 0 for delta in deltas),
        }
    return {
        "schema_version": QUALITY_ABLATION_EVAL_SCHEMA_VERSION,
        "reference_agent": reference_agent,
        "comparisons": comparisons,
    }


__all__ = [
    "EXCLUSION_NO_PUBLISHED_SOLUTION",
    "GREEDY_ABLATION_IDENTIFIER",
    "QUALITY_ABLATION_EVAL_SCHEMA_VERSION",
    "RANDOM_SEARCH_ABLATION_IDENTIFIER",
    "GreedyQualityAgent",
    "RandomSearchQualityAgent",
    "evaluate_quality_ablations_on_partition",
    "summarize_ablation_deltas",
]
