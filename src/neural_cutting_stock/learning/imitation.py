"""Imitation-learning baseline of the exact choice on small instances.

Before any deep RL policy, this module validates the whole
quality-agent pipeline with the smallest possible learned component:

1. ``collect_exact_choice_demonstrations`` pairs, for small bounded
   instances, the versioned ``quality-agent-interface-v1`` observation built
   from the classical column-generation outcome with the expert proposal
   equal to the certified exact optimum computed by
   :class:`CompleteIntegerMaster` over the enumerated maximal patterns;
2. ``train_imitation_policy`` clones that exact choice by regressing, for
   every enumerated candidate column, its usage count in the expert plan,
   with a tiny tanh network trained under the seeded, checkpointed and
   curve-persisted discipline of Phase 9;
3. :class:`ImitationQualityAgent` turns the trained network into a
   :class:`QualityAgent` whose proposals flow through the systematic
   independent review like any other proposal.

The baseline is an interface validation, not a performance claim: it is
trained and evaluated on the same small instances, it never certifies
optimality, and an infeasible or non-improving decoded plan is simply
rejected by the verifier exactly as any agent proposal would be.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from neural_cutting_stock.problem import AnyCuttingStockInstance, CuttingStockInstance
from neural_cutting_stock.solver import (
    ColumnGeneration,
    CompleteIntegerMaster,
    MaximalPatternLimits,
    iter_maximal_patterns,
    verify_plan,
)

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without the extra
    torch = None
    nn = None

from .features import _summary
from .quality_agent import QualityAgentInput, QualityAgentProposal
from .reproducibility import TrainingCurvePoint, TrainingCurves, set_reproducible_seed

IMITATION_BASELINE_SCHEMA_VERSION = "quality-imitation-baseline-v1"
DEFAULT_EPOCHS = 3000
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_HIDDEN_WIDTH = 32

_IMITATION_TORCH_HINT = (
    "PyTorch is required for the imitation baseline; install the versioned "
    "'learning' extra (for example: uv sync --extra dev --extra learning)"
)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(_IMITATION_TORCH_HINT)


@dataclass(frozen=True, slots=True)
class ExactChoiceDemonstration:
    """One expert demonstration pairing an observation with the exact choice.

    ``observation`` is the frozen pool-and-solution state produced by the
    classical loop; ``expert_proposal`` is the certified exact optimal plan.
    ``baseline_bars`` and ``expert_bars`` are verified bar counts, so
    ``bars_saved`` measures where the classical restricted master actually
    loses bars on this instance.
    """

    instance_id: str
    observation: QualityAgentInput
    expert_proposal: QualityAgentProposal
    baseline_bars: int
    expert_bars: int

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        for name in ("baseline_bars", "expert_bars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def bars_saved(self) -> int:
        """Return the verified bar margin between classical and exact plans."""

        return self.baseline_bars - self.expert_bars


def collect_exact_choice_demonstrations(
    instances: Mapping[str, AnyCuttingStockInstance],
    *,
    pattern_limits: MaximalPatternLimits | None = None,
) -> tuple[ExactChoiceDemonstration, ...]:
    """Build one demonstration per instance from real classical and exact runs.

    Instances are processed in sorted identifier order so the returned tuple
    does not depend on mapping insertion order. Every stage is required to
    succeed honestly — a non-converged classical run, a failed exact master
    or a plan that does not verify raises instead of being silently skipped.
    """

    demonstrations = []
    for instance_id in sorted(instances):
        instance = instances[instance_id]
        cg_result = ColumnGeneration(instance, instance_id=instance_id).solve()
        if cg_result.status != "converged":
            raise ValueError(
                f"classical column generation did not converge on {instance_id}: "
                f"status {cg_result.status} ({cg_result.termination_reason})"
            )
        integer_master = cg_result.integer_master_result
        if (
            integer_master is None
            or integer_master.objective_value is None
            or cg_result.verification is None
            or not cg_result.verification.feasible
        ):
            raise ValueError(f"classical run on {instance_id} has no verified integer solution")
        baseline_bars = int(round(integer_master.objective_value))
        if baseline_bars != cg_result.verification.number_of_stock_bars:
            raise ValueError(f"classical solution on {instance_id} does not verify")
        observation = QualityAgentInput(
            instance_id=instance_id,
            stock_length=instance.stock_length,
            kerf=instance.kerf,
            piece_lengths=instance.piece_lengths,
            demands=instance.demands,
            column_pool=cg_result.patterns,
            solution_patterns=cg_result.patterns,
            solution_column_values=integer_master.column_values,
        )

        candidates = enumerated_candidates(observation, pattern_limits)
        exact_result = CompleteIntegerMaster(instance, pattern_limits).solve()
        if exact_result.status != 0 or exact_result.objective_value is None:
            raise ValueError(
                f"exact reference failed on {instance_id}: status "
                f"{exact_result.status} ({exact_result.message})"
            )
        values = exact_result.column_values
        if len(values) != len(candidates):
            raise ValueError(
                f"exact reference on {instance_id} does not match the candidate enumeration"
            )
        usage = dict(zip(candidates, values, strict=True))
        expert_proposal = QualityAgentProposal(
            tuple(pattern for pattern, count in usage.items() if count > 0),
            tuple(count for count in usage.values() if count > 0),
        )
        expert_verification = verify_plan(
            instance, expert_proposal.patterns, expert_proposal.column_values
        )
        expert_bars = int(round(exact_result.objective_value))
        if (
            not expert_verification.feasible
            or expert_verification.number_of_stock_bars != expert_bars
        ):
            raise ValueError(f"exact plan on {instance_id} does not verify")

        demonstrations.append(
            ExactChoiceDemonstration(
                instance_id=instance_id,
                observation=observation,
                expert_proposal=expert_proposal,
                baseline_bars=baseline_bars,
                expert_bars=expert_bars,
            )
        )
    return tuple(demonstrations)


def enumerated_candidates(
    observation: QualityAgentInput,
    pattern_limits: MaximalPatternLimits | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate the maximal-pattern action basis declared by an observation.

    The observation mirrors the largest declared format of its instance, so
    the basis is rebuilt through the single-format view; an unnormalized
    observation would silently reorder types and is refused explicitly.
    """

    view = CuttingStockInstance(
        observation.stock_length, observation.kerf,
        observation.piece_lengths, observation.demands,
    )
    if view.piece_lengths != observation.piece_lengths or view.demands != observation.demands:
        raise ValueError("observation must be normalized before enumerating candidates")
    return tuple(iter_maximal_patterns(view, pattern_limits))


def imitation_candidate_features(
    observation: QualityAgentInput, pattern: tuple[int, ...]
) -> tuple[float, ...]:
    """Build the fixed-width feature vector for one candidate column."""

    return imitation_candidate_features_batch(observation, (pattern,))[0]


def imitation_candidate_features_batch(
    observation: QualityAgentInput, patterns: Sequence[tuple[int, ...]]
) -> tuple[tuple[float, ...], ...]:
    """Build features for several candidate columns of one observation.

    Type-indexed quantities are reduced to symmetric statistics exactly like
    the pricing features of Phase 4, so jointly permuting the piece types,
    the demands and every pattern leaves each vector unchanged and the width
    never depends on the number of types.
    """

    number_of_types = len(observation.piece_lengths)
    total_demand = sum(observation.demands)
    incumbent_usage = [0] * number_of_types
    for pattern_in_solution, value in zip(
        observation.solution_patterns, observation.solution_column_values, strict=True
    ):
        for index, count in enumerate(pattern_in_solution):
            incumbent_usage[index] += value * count

    common = [
        observation.kerf / observation.stock_length,
        float(number_of_types),
        float(total_demand),
        *_summary([length / observation.stock_length for length in observation.piece_lengths]),
        *_summary([demand / total_demand for demand in observation.demands]),
        *_summary(
            [
                usage / demand
                for usage, demand in zip(incumbent_usage, observation.demands, strict=True)
            ]
        ),
    ]
    rows = []
    for pattern in patterns:
        if len(pattern) != number_of_types:
            raise ValueError("candidate pattern must follow observation piece_lengths order")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in pattern
        ):
            raise ValueError("candidate pattern must contain non-negative integers")
        used_ratio = (
            sum(
                (length + observation.kerf) * count
                for length, count in zip(observation.piece_lengths, pattern, strict=True)
            )
            / observation.stock_length
        )
        rows.append(
            (
                *common,
                *_summary(
                    [
                        count / demand
                        for count, demand in zip(pattern, observation.demands, strict=True)
                    ]
                ),
                used_ratio,
                1.0 - used_ratio,
                sum(pattern) / total_demand,
                float(pattern in observation.column_pool),
                float(pattern in observation.solution_patterns),
            )
        )
    return tuple(rows)


class ImitationPolicyNetwork(nn.Module):
    """Tiny tanh regressor predicting the expert usage of one candidate."""

    def __init__(self, feature_width: int, hidden_width: int = DEFAULT_HIDDEN_WIDTH) -> None:
        super().__init__()
        for name, value in (("feature_width", feature_width), ("hidden_width", hidden_width)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.feature_width = feature_width
        self.hidden_width = hidden_width
        self.body = nn.Sequential(
            nn.Linear(feature_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, features: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        """Return one predicted usage score per candidate row."""

        return self.body(features).squeeze(-1)


@dataclass(frozen=True, slots=True)
class ImitationPolicy:
    """A trained imitation policy together with its training provenance."""

    module: ImitationPolicyNetwork
    feature_width: int
    seed: int
    config: Mapping[str, Any]
    curves: TrainingCurves
    schema_version: str = IMITATION_BASELINE_SCHEMA_VERSION


def train_imitation_policy(
    demonstrations: Sequence[ExactChoiceDemonstration],
    *,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    hidden_width: int = DEFAULT_HIDDEN_WIDTH,
    pattern_limits: MaximalPatternLimits | None = None,
) -> ImitationPolicy:
    """Clone the exact choice of every demonstration by supervised regression.

    The dataset holds one row per enumerated candidate column of every
    demonstration, labelled with its usage count in the expert plan. Training
    is full batch MSE under the Phase 9 reproducibility discipline: seeding,
    validated configuration and complete loss curves are part of the artifact.
    """

    _require_torch()
    if not isinstance(demonstrations, Sequence) or not all(
        isinstance(item, ExactChoiceDemonstration) for item in demonstrations
    ):
        raise ValueError("demonstrations must be a sequence of ExactChoiceDemonstration")
    if not demonstrations:
        raise ValueError("imitation training requires at least one demonstration")
    for name, value in (("epochs", epochs), ("hidden_width", hidden_width)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")

    rows: list[tuple[float, ...]] = []
    targets: list[float] = []
    for demonstration in demonstrations:
        candidates = enumerated_candidates(demonstration.observation, pattern_limits)
        expert_usage = dict(
            zip(
                demonstration.expert_proposal.patterns,
                demonstration.expert_proposal.column_values,
                strict=True,
            )
        )
        unknown = set(expert_usage) - set(candidates)
        if unknown:
            raise ValueError(
                f"expert proposal of {demonstration.instance_id} leaves the enumerated basis"
            )
        rows.extend(imitation_candidate_features_batch(demonstration.observation, candidates))
        targets.extend(float(expert_usage.get(candidate, 0)) for candidate in candidates)

    set_reproducible_seed(seed)
    features_tensor = torch.tensor(rows, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    module = ImitationPolicyNetwork(features_tensor.shape[1], hidden_width)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)

    points = []
    for step in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(module(features_tensor), targets_tensor)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise ValueError(f"imitation training diverged at epoch {step}")
        points.append(TrainingCurvePoint(step, {"loss": loss_value}))

    config: dict[str, Any] = {
        "activation": "tanh",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_width": hidden_width,
    }
    return ImitationPolicy(
        module=module,
        feature_width=features_tensor.shape[1],
        seed=seed,
        config=config,
        curves=TrainingCurves.from_points(points),
    )


class ImitationQualityAgent:
    """Quality agent proposing the plan decoded from an imitation policy.

    Every call enumerates the maximal-pattern basis declared by the received
    observation, scores each candidate with the trained network and decodes
    rounded non-negative usage counts into a plain proposal. The proposal
    carries no guarantee: the independent verifier may reject it, and only a
    verified strict bar reduction can ever move an incumbent.
    """

    def __init__(
        self,
        policy: ImitationPolicy,
        pattern_limits: MaximalPatternLimits | None = None,
    ) -> None:
        _require_torch()
        if not isinstance(policy, ImitationPolicy):
            raise ValueError("policy must be an ImitationPolicy")
        self._policy = policy
        self._pattern_limits = pattern_limits

    def propose(self, observation: QualityAgentInput) -> QualityAgentProposal:
        """Return the decoded plan for one observation of the interface."""

        _require_torch()
        candidates = enumerated_candidates(observation, self._pattern_limits)
        if not candidates:
            raise ValueError("candidate enumeration produced no maximal pattern")
        rows = imitation_candidate_features_batch(observation, candidates)
        features_tensor = torch.tensor(rows, dtype=torch.float32)
        with torch.no_grad():
            predictions = self._policy.module(features_tensor).tolist()
        capacity_bound = sum(observation.demands)
        selected_patterns = []
        selected_values = []
        for pattern, prediction in zip(candidates, predictions, strict=True):
            if not math.isfinite(prediction):
                raise ValueError("imitation policy produced a non-finite prediction")
            count = min(capacity_bound, max(0, math.floor(prediction + 0.5)))
            if count > 0:
                selected_patterns.append(pattern)
                selected_values.append(count)
        return QualityAgentProposal(tuple(selected_patterns), tuple(selected_values))


__all__ = [
    "DEFAULT_EPOCHS",
    "DEFAULT_HIDDEN_WIDTH",
    "DEFAULT_LEARNING_RATE",
    "IMITATION_BASELINE_SCHEMA_VERSION",
    "ExactChoiceDemonstration",
    "ImitationPolicy",
    "ImitationPolicyNetwork",
    "ImitationQualityAgent",
    "collect_exact_choice_demonstrations",
    "enumerated_candidates",
    "imitation_candidate_features",
    "imitation_candidate_features_batch",
    "train_imitation_policy",
]
