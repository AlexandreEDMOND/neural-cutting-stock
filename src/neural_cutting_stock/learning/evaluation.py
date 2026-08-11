"""Out-of-sample ranking metrics for learned column scorers.

The evaluation unit is one pricing iteration.  A candidate is relevant when the
classical trajectory recorded it as selected.  The fixed metrics are Hit@1,
Hit@3, Hit@5, mean reciprocal rank (MRR), and nDCG@5.  Iterations without a
relevant candidate are reported but excluded from metric denominators.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from neural_cutting_stock.benchmarks import DatasetPartition, TrajectoryDataset

from .interfaces import PatternCandidate, PatternScore, PricingState
from .model import LinearColumnScoringModel

EVALUATION_SCHEMA_VERSION = "ranking-evaluation-v1"
RANKING_CUTOFFS = (1, 3, 5)


def evaluate_model(
    dataset: TrajectoryDataset,
    model: LinearColumnScoringModel,
    partition: DatasetPartition | str,
) -> dict[str, Any]:
    """Evaluate a fitted model on one partition without fitting or changing it."""

    selected_partition = DatasetPartition(partition)
    groups: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    for example in dataset.examples:
        if example.partition is selected_partition:
            groups[(example.trajectory_id, example.instance_id, example.iteration_index)].append(
                example
            )

    learned_ranks: list[tuple[int, ...]] = []
    exact_ranks: list[tuple[int, ...]] = []
    positive_count = 0
    groups_without_positive = 0
    for examples in (groups[key] for key in sorted(groups)):
        patterns = [example.candidate_pattern for example in examples]
        if len(set(patterns)) != len(patterns):
            raise ValueError("each evaluation group must contain distinct candidate patterns")
        state = _state_from_example(examples[0])
        candidates = tuple(
            PatternCandidate(example.candidate_pattern, example.candidate_reduced_cost)
            for example in examples
        )
        learned_scores = model.score(state, candidates)
        if len(learned_scores) != len(candidates):
            raise ValueError("model must return one score per candidate")
        selected = {example.candidate_pattern for example in examples if example.selected}
        positive_count += len(selected)
        if not selected:
            groups_without_positive += 1
            continue
        learned_ranks.append(_relevant_ranks(learned_scores, selected, reverse=True))
        exact_ranks.append(
            _relevant_ranks(
                tuple(
                    PatternScore(candidate.pattern, candidate.reduced_cost)
                    for candidate in candidates
                ),
                selected,
                reverse=False,
            )
        )

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "partition": selected_partition.value,
        "example_count": sum(len(group) for group in groups.values()),
        "ranking_group_count": len(groups),
        "evaluated_group_count": len(learned_ranks),
        "positive_example_count": positive_count,
        "groups_without_positive": groups_without_positive,
        "metrics": {
            "learned": _aggregate_metrics(learned_ranks),
            "exact_reduced_cost": _aggregate_metrics(exact_ranks),
        },
    }


def _state_from_example(example: Any) -> PricingState:
    return PricingState(
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


def _relevant_ranks(
    scores: Sequence[PatternScore], relevant: set[tuple[int, ...]], *, reverse: bool
) -> tuple[int, ...]:
    ordered = sorted(
        scores, key=lambda item: ((-item.score if reverse else item.score), item.pattern)
    )
    return tuple(index for index, item in enumerate(ordered, start=1) if item.pattern in relevant)


def _aggregate_metrics(ranks_by_group: Sequence[tuple[int, ...]]) -> dict[str, float]:
    if not ranks_by_group:
        return {
            **{f"hit_rate_at_{cutoff}": 0.0 for cutoff in RANKING_CUTOFFS},
            "mrr": 0.0,
            "ndcg_at_5": 0.0,
        }
    return {
        **{
            f"hit_rate_at_{cutoff}": sum(
                any(rank <= cutoff for rank in ranks) for ranks in ranks_by_group
            )
            / len(ranks_by_group)
            for cutoff in RANKING_CUTOFFS
        },
        "mrr": sum(1.0 / ranks[0] for ranks in ranks_by_group) / len(ranks_by_group),
        "ndcg_at_5": sum(_ndcg_at_5(ranks) for ranks in ranks_by_group) / len(ranks_by_group),
    }


def _ndcg_at_5(ranks: Sequence[int]) -> float:
    actual = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 5)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(ranks), 5)))
    return actual / ideal if ideal else 0.0
