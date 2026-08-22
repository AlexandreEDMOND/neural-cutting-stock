"""Structured demand profiles produce non-trivial integer gaps (Phase 8).

Every case materializes a deterministic structured-profile instance, solves it
with the classical column generation loop, and compares its restricted-master
integer objective (`optimal_over_generated_columns_only`) with an optimal,
independently verified MILP reference over the complete enumerated pattern
set. A positive difference is exactly the integer hole these profiles must
exhibit before any quality-margin campaign can rely on them.
"""

from decimal import Decimal

from neural_cutting_stock.benchmarks import (
    AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    TIGHT_RATIO_LENGTH_DISTRIBUTION,
    EnvironmentMetadata,
    ExactReferenceStatus,
    SyntheticInstanceGenerator,
    solve_milp_exact_reference,
    verify_milp_exact_reference,
)
from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import ColumnGeneration

ENVIRONMENT = EnvironmentMetadata("structured-profile-test", "test", "test", "test")
REDUCED_COST_TOLERANCE = 1e-9
INTEGRALITY_TOLERANCE = 1e-9
FEASIBILITY_TOLERANCE = 1e-9
CASES = tuple((types, seed) for types in (3, 4) for seed in range(1, 7))
MINIMUM_POSITIVE_GAP_COUNT = len(CASES) // 2


def _structured_generator(number_of_types: int, seed: int) -> SyntheticInstanceGenerator:
    return SyntheticInstanceGenerator(
        seed=seed,
        number_of_types=number_of_types,
        demand_range=(5, 30),
        length_distribution=TIGHT_RATIO_LENGTH_DISTRIBUTION,
        demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    )


def _natural_multiplicity(instance: CuttingStockInstance, length: float) -> int:
    return int(
        Decimal(str(instance.stock_length))
        // (Decimal(str(length)) + Decimal(str(instance.kerf)))
    )


def test_structured_profiles_produce_non_trivial_integer_gaps() -> None:
    gaps: list[int] = []
    for number_of_types, seed in CASES:
        generator = _structured_generator(number_of_types, seed)
        instance = generator.generate()

        assert len(set(instance.piece_lengths)) == number_of_types
        assert all(
            _natural_multiplicity(instance, length) == 2
            for length in instance.piece_lengths
        )
        for length, demand in zip(instance.piece_lengths, instance.demands, strict=True):
            assert 5 <= demand <= 30
            assert demand % _natural_multiplicity(instance, length) != 0

        result = ColumnGeneration(instance, REDUCED_COST_TOLERANCE).solve()
        assert result.status == "converged"
        assert result.integer_solution_guarantee == "optimal_over_generated_columns_only"
        assert result.verification is not None and result.verification.feasible
        baseline_bars = result.integer_master_result.objective_value

        outcome, reference = solve_milp_exact_reference(
            generator.instance_id,
            instance,
            environment=ENVIRONMENT,
            integrality_tolerance=INTEGRALITY_TOLERANCE,
            feasibility_tolerance=FEASIBILITY_TOLERANCE,
        )
        assert reference.status is ExactReferenceStatus.OPTIMAL, reference.error_message
        verification = verify_milp_exact_reference(
            generator.instance_id, instance, outcome, reference
        )
        assert verification.passed, verification.errors
        optimum_bars = reference.integer_optimum_bars
        assert result.rmp_result.objective_value <= optimum_bars + FEASIBILITY_TOLERANCE

        gaps.append(baseline_bars - optimum_bars)

    positive_count = sum(gap > 0 for gap in gaps)
    assert positive_count >= MINIMUM_POSITIVE_GAP_COUNT, gaps
