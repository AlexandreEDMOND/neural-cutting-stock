"""Scaled instance sizes keep maintainable exact references (Phase 8).

P8.04 pushes instance sizes — more piece types and higher demands — as far as
an exact MILP reference over enumerated maximal patterns stays maintainable
and independently verifiable. Every case is a real execution: a deterministic
scaled instance is generated, its maximal patterns are enumerated under
explicitly declared guards, the complete integer master is solved by MILP,
and the resulting reference is re-derived by the independent verifier. The
default enumeration guards still certify twelve to fourteen multiplicity-two
types; larger type counts remain certified through explicitly declared,
honestly recorded guards, and instances beyond any declared guard surface as
failed references carrying their diagnosis instead of truncated numbers.
"""

from decimal import Decimal

import pytest

from neural_cutting_stock.benchmarks import (
    AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    TIGHT_RATIO_LENGTH_DISTRIBUTION,
    EnvironmentMetadata,
    ExactReferenceStatus,
    SyntheticInstanceGenerator,
    solve_milp_exact_reference,
    verify_milp_exact_reference,
)
from neural_cutting_stock.solver import ColumnGeneration, MaximalPatternLimits

ENVIRONMENT = EnvironmentMetadata("scaled-size-test", "test", "test", "test")
REDUCED_COST_TOLERANCE = 1e-9
INTEGRALITY_TOLERANCE = 1e-9
FEASIBILITY_TOLERANCE = 1e-9

DEFAULT_LIMITS = MaximalPatternLimits()
DECLARED_SCALED_LIMITS = MaximalPatternLimits(
    max_search_space_size=10**16,
    max_patterns=1_000_000,
)


def _tight_generator(
    number_of_types: int,
    *,
    stock_length: float = 100.0,
    demand_range: tuple[int, int] = (20, 100),
    length_range: tuple[int, int] = (10, 90),
) -> SyntheticInstanceGenerator:
    return SyntheticInstanceGenerator(
        seed=17,
        stock_length=stock_length,
        number_of_types=number_of_types,
        demand_range=demand_range,
        piece_length_range=length_range,
        length_distribution=TIGHT_RATIO_LENGTH_DISTRIBUTION,
        demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    )


def _uniform_generator() -> SyntheticInstanceGenerator:
    return SyntheticInstanceGenerator(
        seed=17,
        number_of_types=14,
        demand_range=(20, 100),
        piece_length_range=(10, 90),
        demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    )


def _natural_multiplicity(stock_length: float, kerf: float, length: float) -> int:
    return int(Decimal(str(stock_length)) // (Decimal(str(length)) + Decimal(str(kerf))))


def _solve_reference(
    generator: SyntheticInstanceGenerator,
    limits: MaximalPatternLimits | None = None,
):
    instance = generator.generate()
    outcome, reference = solve_milp_exact_reference(
        generator.instance_id,
        instance,
        environment=ENVIRONMENT,
        integrality_tolerance=INTEGRALITY_TOLERANCE,
        feasibility_tolerance=FEASIBILITY_TOLERANCE,
        limits=limits,
    )
    assert reference.status is ExactReferenceStatus.OPTIMAL, reference.error_message
    verification = verify_milp_exact_reference(
        generator.instance_id,
        instance,
        outcome,
        reference,
        limits=limits,
    )
    assert verification.passed, verification.errors
    assert verification.lp_bound_bars <= reference.integer_optimum_bars + FEASIBILITY_TOLERANCE
    assert reference.certified_lower_bound_bars <= reference.integer_optimum_bars
    return reference


@pytest.mark.parametrize("number_of_types", [12, 14])
def test_default_guards_still_certify_scaled_tight_ratio_instances(
    number_of_types: int,
) -> None:
    generator = _tight_generator(number_of_types)
    instance = generator.generate()

    assert len(set(instance.piece_lengths)) == number_of_types
    assert all(
        _natural_multiplicity(instance.stock_length, instance.kerf, length) == 2
        for length in instance.piece_lengths
    )

    reference = _solve_reference(generator)

    assert reference.method_limits == (
        "maximal_patterns:max_search_space_size="
        f"{DEFAULT_LIMITS.max_search_space_size},max_patterns={DEFAULT_LIMITS.max_patterns}"
    )


def test_default_guards_refuse_beyond_the_multiplicity_two_grid_frontier() -> None:
    generator = _tight_generator(15)

    _, reference = solve_milp_exact_reference(
        generator.instance_id,
        generator.generate(),
        environment=ENVIRONMENT,
        integrality_tolerance=INTEGRALITY_TOLERANCE,
        feasibility_tolerance=FEASIBILITY_TOLERANCE,
    )

    assert reference.status is ExactReferenceStatus.FAILED
    assert reference.integer_optimum_bars is None
    assert reference.certified_lower_bound_bars is None
    assert "search space" in reference.error_message
    assert str(DEFAULT_LIMITS.max_search_space_size) in reference.error_message


@pytest.mark.parametrize(
    ("stock_length", "number_of_types", "length_range"),
    [(100.0, 17, (10, 90)), (300.0, 24, (10, 150)), (500.0, 32, (10, 250))],
)
def test_declared_guards_push_types_far_beyond_the_existing_corpus(
    stock_length: float, number_of_types: int, length_range: tuple[int, int]
) -> None:
    generator = _tight_generator(
        number_of_types, stock_length=stock_length, length_range=length_range
    )
    instance = generator.generate()

    assert len(set(instance.piece_lengths)) == number_of_types
    assert all(
        _natural_multiplicity(instance.stock_length, instance.kerf, length) == 2
        for length in instance.piece_lengths
    )

    reference = _solve_reference(generator, limits=DECLARED_SCALED_LIMITS)

    assert str(DECLARED_SCALED_LIMITS.max_search_space_size) in reference.method_limits


def test_high_demands_keep_certified_optimal_references() -> None:
    generator = _tight_generator(12, demand_range=(100, 400))
    instance = generator.generate()

    assert all(100 <= demand <= 400 for demand in instance.demands)

    _solve_reference(generator)


def test_uniform_family_scales_past_the_phase_six_type_frontier() -> None:
    generator = _uniform_generator()
    instance = generator.generate()

    assert len(set(instance.piece_lengths)) == 14

    _solve_reference(generator)


@pytest.mark.parametrize(
    "generator",
    [
        _tight_generator(32, stock_length=500.0, length_range=(10, 250)),
        _uniform_generator(),
    ],
    ids=["tight-ratio-32-types", "uniform-14-types"],
)
def test_classical_generation_converges_on_pushed_sizes(
    generator: SyntheticInstanceGenerator,
) -> None:
    result = ColumnGeneration(generator.generate(), REDUCED_COST_TOLERANCE).solve()

    assert result.status == "converged"
    assert result.integer_solution_guarantee == "optimal_over_generated_columns_only"
    assert result.verification is not None and result.verification.feasible
