"""Deep policy-gradient training of the quality agent on verified refinements.

The trainer runs :class:`QualityRefinementEnv` episodes on real classical
starting points and optimizes a deep stochastic policy with REINFORCE, as
documented and justified in ``docs/phase-9-rl-algorithm.md``:

- one step's action factorizes into independent Poisson usage counts over the
  enumerated maximal-pattern basis declared by the observation; sampled
  counts are capped by the demand bounds before they are executed;
- the decoded plan is deterministically completed from the same basis so
  coverage always verifies: every executed proposal stays in the graded,
  verified reward regime instead of the flat invalid-plan penalty;
- each epoch collects one episode per training instance in sorted identifier
  order, computes reward-to-go advantages against a per-instance running-mean
  baseline, standardizes them across the epoch and takes a single Adam step.

The policy never certifies anything: proposals flow through the systematic
independent review like any other agent output, acceptance still requires a
verified strict bar reduction, and no claim of global optimality is made.
Training is seeded through the Phase 9 reproducibility discipline; bit-exact
reproduction across machines or builds stays out of scope.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from neural_cutting_stock.problem import AnyCuttingStockInstance
from neural_cutting_stock.solver import ColumnGeneration, MaximalPatternLimits

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without the extra
    torch = None
    nn = None

from .imitation import enumerated_candidates, imitation_candidate_features_batch
from .quality_agent import QualityAgentInput, QualityAgentProposal
from .quality_env import QualityRefinementEnv
from .reproducibility import TrainingCurvePoint, TrainingCurves, set_reproducible_seed

QUALITY_RL_POLICY_SCHEMA_VERSION = "quality-rl-policy-v1"
TRAINING_JOURNAL_SCHEMA_VERSION = "phase-9-training-journal-v1"
ALGORITHM_IDENTIFIER = "reinforce-poisson-completion-v1"
DEFAULT_EPOCHS = 300
DEFAULT_LEARNING_RATE = 3e-3
DEFAULT_HIDDEN_WIDTH = 64
DEFAULT_MAX_STEPS = 4
DEFAULT_BASELINE_MOMENTUM = 0.8

_MIN_RATE = 1e-6
_ADVANTAGE_EPSILON = 1e-8

_TORCH_HINT = (
    "PyTorch is required for deep RL training; install the versioned 'learning' "
    "extra (for example: uv sync --extra dev --extra learning)"
)

ALGORITHM_DOCUMENTATION: Mapping[str, str] = {
    "identifier": ALGORITHM_IDENTIFIER,
    "name": "REINFORCE with running-mean baseline and standardized advantages",
    "reference": "docs/phase-9-rl-algorithm.md",
    "action_space": (
        "one usage count per enumerated maximal-pattern candidate, completed "
        "deterministically so every proposal covers the demands"
    ),
    "sampling": "independent Poisson rates emitted by the network, capped by demand bounds",
    "reward": (
        "verified signed bar reduction of each reviewed step; strict negative "
        "penalty for invalid plans"
    ),
    "update": (
        "per-epoch batch score-function ascent on reward-to-go returns with "
        "advantages standardized across the epoch"
    ),
    "baseline": "exponential running mean of episode returns, tracked per instance",
    "optimizer": "Adam",
}


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(_TORCH_HINT)


class QualityPolicyNetwork(nn.Module):
    """Tiny tanh network emitting one strictly positive Poisson rate per candidate."""

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
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        """Return one strictly positive Poisson rate per candidate row."""

        return nn.functional.softplus(self.body(features)).squeeze(-1) + _MIN_RATE


@dataclass(frozen=True, slots=True)
class RLEpisodeRecord:
    """Journal entry of one verified refinement episode.

    ``return_value`` is the summed reward of the episode, mixing verified bar
    reductions with any invalid-plan penalty; ``bars_saved`` counts only
    accepted improvements, so it always satisfies
    ``final_bars == initial_bars - bars_saved``.
    """

    episode_index: int
    instance_id: str
    steps_taken: int
    return_value: float
    bars_saved: int
    accepted_steps: int
    invalid_steps: int
    initial_bars: int
    final_bars: int

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        for name in ("episode_index", "steps_taken", "initial_bars", "final_bars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.steps_taken < 1:
            raise ValueError("steps_taken must be at least one")
        for name in ("bars_saved", "accepted_steps", "invalid_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.return_value, bool) or not isinstance(self.return_value, (int, float)):
            raise ValueError("return_value must be a real number")
        if not math.isfinite(float(self.return_value)):
            raise ValueError("return_value must be finite")
        if self.accepted_steps > self.steps_taken or self.invalid_steps > self.steps_taken:
            raise ValueError("step counters cannot exceed steps_taken")
        if self.final_bars != self.initial_bars - self.bars_saved:
            raise ValueError("final_bars must equal initial_bars minus bars_saved")


@dataclass(frozen=True, slots=True)
class QualityRLPolicy:
    """A trained deep policy together with its complete training provenance."""

    module: QualityPolicyNetwork
    feature_width: int
    seed: int
    config: Mapping[str, Any]
    curves: TrainingCurves
    episodes: tuple[RLEpisodeRecord, ...]
    environment: Mapping[str, Any]
    schema_version: str = QUALITY_RL_POLICY_SCHEMA_VERSION

    @property
    def totals(self) -> dict[str, int]:
        """Return the aggregate activity of every recorded episode."""

        return {
            "episode_count": len(self.episodes),
            "step_count": sum(record.steps_taken for record in self.episodes),
            "bars_saved_total": sum(record.bars_saved for record in self.episodes),
            "accepted_step_count": sum(record.accepted_steps for record in self.episodes),
            "invalid_step_count": sum(record.invalid_steps for record in self.episodes),
        }

    @property
    def trained_instance_ids(self) -> tuple[str, ...]:
        """Return the sorted identifiers of every instance seen during training."""

        return tuple(sorted({record.instance_id for record in self.episodes}))


def train_quality_rl_policy(
    instances: Mapping[str, AnyCuttingStockInstance],
    *,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    hidden_width: int = DEFAULT_HIDDEN_WIDTH,
    max_steps: int = DEFAULT_MAX_STEPS,
    baseline_momentum: float = DEFAULT_BASELINE_MOMENTUM,
    pattern_limits: MaximalPatternLimits | None = None,
) -> QualityRLPolicy:
    """Train the deep refinement policy on the given instances by REINFORCE.

    Every instance contributes one verified classical starting point and one
    episode per epoch, processed in sorted identifier order. The returned
    policy carries its full provenance: hyperparameters, seed, environment
    metadata, complete curves and one journal record per episode.
    """

    _require_torch()
    if not isinstance(instances, Mapping) or not instances:
        raise ValueError("training requires a non-empty mapping of instances")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    for name, value in (("epochs", epochs), ("hidden_width", hidden_width)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if (
        isinstance(baseline_momentum, bool)
        or not isinstance(baseline_momentum, (int, float))
        or not math.isfinite(baseline_momentum)
        or not 0 <= baseline_momentum < 1
    ):
        raise ValueError("baseline_momentum must lie within [0, 1)")

    ordered_ids = sorted(instances)
    starting_points: dict[str, tuple[AnyCuttingStockInstance, tuple, tuple]] = {}
    bases: dict[str, tuple[tuple[int, ...], ...]] = {}
    caps: dict[str, list[float]] = {}
    for instance_id in ordered_ids:
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
        starting_points[instance_id] = (
            instance,
            cg_result.patterns,
            integer_master.column_values,
        )
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
        basis = enumerated_candidates(observation, pattern_limits)
        if not basis:
            raise ValueError(f"{instance_id} enumerates an empty maximal-pattern basis")
        bases[instance_id] = basis
        caps[instance_id] = [
            min(
                demand
                for count, demand in zip(pattern, observation.demands, strict=True)
                if count > 0
            )
            for pattern in basis
        ]

    first_rows = imitation_candidate_features_batch(
        QualityAgentInput(
            instance_id=ordered_ids[0],
            stock_length=instances[ordered_ids[0]].stock_length,
            kerf=instances[ordered_ids[0]].kerf,
            piece_lengths=instances[ordered_ids[0]].piece_lengths,
            demands=instances[ordered_ids[0]].demands,
            column_pool=starting_points[ordered_ids[0]][1],
            solution_patterns=starting_points[ordered_ids[0]][1],
            solution_column_values=starting_points[ordered_ids[0]][2],
        ),
        bases[ordered_ids[0]],
    )
    feature_width = len(first_rows[0])

    metadata = set_reproducible_seed(seed)
    module = QualityPolicyNetwork(feature_width, hidden_width)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)

    baselines = {instance_id: 0.0 for instance_id in ordered_ids}
    points: list[TrainingCurvePoint] = []
    records: list[RLEpisodeRecord] = []
    for epoch in range(epochs):
        batch_log_probs: list[torch.Tensor] = []
        batch_advantages: list[float] = []
        epoch_returns: list[float] = []
        bars_saved = accepted_steps = invalid_steps = 0
        for instance_id in ordered_ids:
            instance, pool, values = starting_points[instance_id]
            env = QualityRefinementEnv(
                instance, instance_id, pool, pool, values, max_steps=max_steps
            )
            log_probs, rewards_to_go, episode_return, accepted, invalid = _rollout(
                module, env, bases[instance_id], caps[instance_id]
            )
            baseline = baselines[instance_id]
            batch_log_probs.extend(log_probs)
            batch_advantages.extend(value - baseline for value in rewards_to_go)
            baselines[instance_id] = (
                baseline_momentum * baseline + (1.0 - baseline_momentum) * episode_return
            )
            records.append(
                RLEpisodeRecord(
                    episode_index=len(records),
                    instance_id=instance_id,
                    steps_taken=env.steps_taken,
                    return_value=episode_return,
                    bars_saved=env.total_bars_saved,
                    accepted_steps=accepted,
                    invalid_steps=invalid,
                    initial_bars=env.initial_bars,
                    final_bars=env.current_bars,
                )
            )
            epoch_returns.append(episode_return)
            bars_saved += env.total_bars_saved
            accepted_steps += accepted
            invalid_steps += invalid
        advantages = torch.tensor(batch_advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + _ADVANTAGE_EPSILON)
        loss = -(torch.stack(batch_log_probs) * advantages).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise ValueError(f"policy gradient diverged at epoch {epoch}")
        points.append(
            TrainingCurvePoint(
                epoch,
                {
                    "policy_loss": loss_value,
                    "mean_episode_return": sum(epoch_returns) / len(epoch_returns),
                    "bars_saved_total": float(bars_saved),
                    "accepted_steps": float(accepted_steps),
                    "invalid_steps": float(invalid_steps),
                },
            )
        )

    config: dict[str, Any] = {
        "algorithm": ALGORITHM_IDENTIFIER,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_width": hidden_width,
        "max_steps": max_steps,
        "baseline_momentum": baseline_momentum,
    }
    return QualityRLPolicy(
        module=module,
        feature_width=feature_width,
        seed=seed,
        config=config,
        curves=TrainingCurves.from_points(points),
        episodes=tuple(records),
        environment=metadata,
    )


def training_journal_payload(
    policy: QualityRLPolicy, *, source: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble the versioned experiment journal of one training run.

    ``source`` carries the run's provenance beyond the policy itself: the
    frozen partition manifest and plan id, the trained partition, the trained
    instance identifiers and the persisted checkpoint reference.
    """

    if not isinstance(policy, QualityRLPolicy):
        raise ValueError("policy must be a QualityRLPolicy")
    for name in ("partition_manifest", "plan_id", "partition"):
        value = source.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"journal source requires a non-empty {name}")
    instance_ids = source.get("instance_ids")
    if (
        not isinstance(instance_ids, list)
        or not all(isinstance(item, str) and item.strip() for item in instance_ids)
        or set(instance_ids) != set(policy.trained_instance_ids)
    ):
        raise ValueError("journal source instance_ids must match the trained instances")
    checkpoint_path = source.get("checkpoint_path")
    checkpoint_sha256 = source.get("checkpoint_sha256")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError("journal source requires a non-empty checkpoint_path")
    if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256.strip():
        raise ValueError("journal source requires a non-empty checkpoint_sha256")
    return {
        "schema_version": TRAINING_JOURNAL_SCHEMA_VERSION,
        "algorithm": dict(ALGORITHM_DOCUMENTATION),
        "source": {
            "partition_manifest": source["partition_manifest"],
            "plan_id": source["plan_id"],
            "partition": source["partition"],
            "instance_ids": sorted(instance_ids),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "policy_schema_version": policy.schema_version,
        "seed": policy.seed,
        "config": dict(policy.config),
        "environment": dict(policy.environment),
        "totals": policy.totals,
        "curves": policy.curves.to_payload(),
        "episodes": [asdict(record) for record in policy.episodes],
    }


def _rollout(
    module: QualityPolicyNetwork,
    env: QualityRefinementEnv,
    basis: tuple[tuple[int, ...], ...],
    caps: Sequence[float],
) -> tuple[list["torch.Tensor"], list[float], float, int, int]:
    """Run one verified episode and return its score-function ingredients."""

    observation = env.reset()
    demands = observation.demands
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    accepted_steps = 0
    invalid_steps = 0
    for _ in range(env.max_steps):
        rows = imitation_candidate_features_batch(observation, basis)
        features = torch.tensor(rows, dtype=torch.float32)
        distribution = torch.distributions.Poisson(module(features))
        sampled = distribution.sample()
        counts = torch.minimum(sampled, torch.tensor(caps, dtype=torch.float32))
        log_probs.append(distribution.log_prob(counts).sum())
        proposal = _decode_proposal(basis, [int(value) for value in counts.tolist()], demands)
        step = env.step(proposal)
        rewards.append(step.reward)
        feasible = (
            step.review.baseline_verification.feasible
            and step.review.proposal_verification.feasible
        )
        if step.accepted:
            accepted_steps += 1
        elif not feasible:
            invalid_steps += 1
        observation = step.observation
    rewards_to_go: list[float] = []
    accumulated = 0.0
    for reward in reversed(rewards):
        accumulated += reward
        rewards_to_go.append(accumulated)
    rewards_to_go.reverse()
    return log_probs, rewards_to_go, accumulated, accepted_steps, invalid_steps


def _decode_proposal(
    basis: tuple[tuple[int, ...], ...],
    counts: Sequence[int],
    demands: Sequence[int],
) -> QualityAgentProposal:
    """Decode sampled counts plus a deterministic completion into one plan.

    The completion greedily restores coverage from the same maximal-pattern
    basis, so the executed proposal always verifies: the bulk pass adds whole
    multiplicities while they help, then single-bar sweeps exhaust any
    residual demand. Overproduction stays legal, only underproduction fails.
    """

    combined: dict[tuple[int, ...], int] = {}
    residual = list(demands)
    for pattern, count in zip(basis, counts, strict=True):
        if count <= 0:
            continue
        combined[pattern] = combined.get(pattern, 0) + count
        for index, pieces in enumerate(pattern):
            residual[index] -= count * pieces
    completion: dict[int, int] = {}
    for index, pattern in enumerate(basis):
        support = [i for i, pieces in enumerate(pattern) if pieces > 0]
        unmet = [residual[i] for i in support if residual[i] > 0]
        if not support or not unmet:
            continue
        bulk = min(residual[i] // pattern[i] for i in support if residual[i] > 0)
        if bulk > 0:
            completion[index] = bulk
            for i in support:
                residual[i] -= bulk * pattern[i]
    sweeps = 0
    while any(remaining > 0 for remaining in residual):
        sweeps += 1
        if sweeps > sum(demands):
            raise RuntimeError("coverage completion failed to converge")
        progressed = False
        for index, pattern in enumerate(basis):
            if any(pieces > 0 and residual[i] > 0 for i, pieces in enumerate(pattern)):
                completion[index] = completion.get(index, 0) + 1
                for i, pieces in enumerate(pattern):
                    residual[i] -= pieces
                progressed = True
                break
        if not progressed:
            raise RuntimeError("no basis pattern covers the remaining demand")
    for index, extra in completion.items():
        pattern = basis[index]
        combined[pattern] = combined.get(pattern, 0) + extra
    return QualityAgentProposal(tuple(combined), tuple(combined.values()))


class RLQualityAgent:
    """Deterministic inference wrapper turning a trained policy into an agent.

    Training explores by sampling Poisson counts; at refinement time the
    trained policy is applied greedily: every enumerated candidate receives
    its rounded emitted rate capped by the demand bounds, and the same
    deterministic completion as during training restores full coverage. The
    resulting proposal carries no guarantee of its own — it flows through
    the systematic independent review exactly like any other agent output,
    and only a verified strict bar reduction can move an incumbent.
    """

    def __init__(
        self,
        policy: QualityRLPolicy,
        pattern_limits: MaximalPatternLimits | None = None,
    ) -> None:
        _require_torch()
        if not isinstance(policy, QualityRLPolicy):
            raise ValueError("policy must be a QualityRLPolicy")
        self._policy = policy
        self._pattern_limits = pattern_limits

    def propose(self, observation: QualityAgentInput) -> QualityAgentProposal:
        """Return the greedy decoded plan for one observation of the interface."""

        _require_torch()
        candidates = enumerated_candidates(observation, self._pattern_limits)
        if not candidates:
            raise ValueError("candidate enumeration produced no maximal pattern")
        rows = imitation_candidate_features_batch(observation, candidates)
        features = torch.tensor(rows, dtype=torch.float32)
        with torch.no_grad():
            rates = self._policy.module(features).tolist()
        caps = [
            min(
                demand
                for count, demand in zip(pattern, observation.demands, strict=True)
                if count > 0
            )
            for pattern in candidates
        ]
        counts = []
        for rate, cap in zip(rates, caps, strict=True):
            if not math.isfinite(rate):
                raise ValueError("quality policy produced a non-finite rate")
            counts.append(int(min(cap, max(0.0, math.floor(rate + 0.5)))))
        return _decode_proposal(candidates, counts, observation.demands)


__all__ = [
    "ALGORITHM_DOCUMENTATION",
    "ALGORITHM_IDENTIFIER",
    "DEFAULT_BASELINE_MOMENTUM",
    "DEFAULT_EPOCHS",
    "DEFAULT_HIDDEN_WIDTH",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MAX_STEPS",
    "QUALITY_RL_POLICY_SCHEMA_VERSION",
    "TRAINING_JOURNAL_SCHEMA_VERSION",
    "QualityPolicyNetwork",
    "QualityRLPolicy",
    "RLQualityAgent",
    "RLEpisodeRecord",
    "train_quality_rl_policy",
    "training_journal_payload",
]
