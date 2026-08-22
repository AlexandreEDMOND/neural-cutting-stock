from decimal import Decimal

import pytest

from neural_cutting_stock.benchmarks import (
    AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    TIGHT_RATIO_LENGTH_DISTRIBUTION,
    SyntheticInstanceGenerator,
)


def test_synthetic_generator_is_reproducible_and_seeded() -> None:
    configuration = SyntheticInstanceGenerator(seed=17, number_of_types=4)

    first = configuration.generate()
    second = configuration.generate()
    other = SyntheticInstanceGenerator(seed=18, number_of_types=4).generate()

    assert first == second
    assert first != other
    assert len(first.piece_lengths) == 4
    assert all(
        first.capacity_used(pattern) <= first.stock_length
        for pattern in first.initial_patterns()
    )


def test_synthetic_generator_instance_id_is_stable_for_normalized_data() -> None:
    configuration = SyntheticInstanceGenerator(seed=17, number_of_types=4)

    assert configuration.instance_id == configuration.instance_id
    assert len(configuration.instance_id) == 64
    assert configuration.instance_id != SyntheticInstanceGenerator(
        seed=18, number_of_types=4
    ).instance_id


def test_synthetic_generator_keeps_length_and_demand_distribution_metadata() -> None:
    generator = SyntheticInstanceGenerator(
        seed=17,
        length_distribution="short_uniform_v1",
        demand_distribution="high_uniform_v1",
    )

    assert generator.length_distribution == "short_uniform_v1"
    assert generator.demand_distribution == "high_uniform_v1"


def test_unrecognized_labels_keep_the_historical_uniform_sampler() -> None:
    labelled = SyntheticInstanceGenerator(
        seed=17,
        number_of_types=4,
        length_distribution="short_uniform_v1",
        demand_distribution="high_uniform_v1",
    )
    default = SyntheticInstanceGenerator(seed=17, number_of_types=4)

    assert labelled.generate() == default.generate()
    assert labelled.family_id != default.family_id


def test_structured_profiles_are_reproducible_and_form_their_own_family() -> None:
    configuration = SyntheticInstanceGenerator(
        seed=17,
        number_of_types=4,
        demand_range=(5, 30),
        length_distribution=TIGHT_RATIO_LENGTH_DISTRIBUTION,
        demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    )
    twin = SyntheticInstanceGenerator(seed=17, number_of_types=4, demand_range=(5, 30))

    first = configuration.generate()
    second = configuration.generate()

    assert first == second
    assert configuration.instance_id == configuration.instance_id
    assert len(configuration.instance_id) == 64
    assert configuration.instance_id != twin.instance_id
    assert configuration.family_id != twin.family_id


def _natural_multiplicity(stock_length: float, kerf: float, length: float) -> int:
    return int(Decimal(str(stock_length)) // (Decimal(str(length)) + Decimal(str(kerf))))


@pytest.mark.parametrize("kerf,window", [(0.0, (34, 50)), (1.0, (33, 49)), (3.0, (31, 47))])
def test_tight_ratio_profile_bounds_natural_multiplicity_to_two(
    kerf: float, window: tuple[int, int]
) -> None:
    configuration = SyntheticInstanceGenerator(
        seed=17,
        number_of_types=4,
        kerf=kerf,
        length_distribution=TIGHT_RATIO_LENGTH_DISTRIBUTION,
    )

    instance = configuration.generate()

    assert all(window[0] <= length <= window[1] for length in instance.piece_lengths)
    assert all(
        _natural_multiplicity(instance.stock_length, instance.kerf, length) == 2
        for length in instance.piece_lengths
    )
    for index in range(instance.number_of_types):
        pattern = [0] * instance.number_of_types
        pattern[index] = 2
        assert instance.capacity_used(tuple(pattern)) <= instance.stock_length


@pytest.mark.parametrize("seed", range(1, 9))
def test_awkward_divisibility_profile_forces_non_divisible_demands(seed: int) -> None:
    configuration = SyntheticInstanceGenerator(
        seed=seed,
        number_of_types=4,
        demand_range=(5, 30),
        length_distribution=TIGHT_RATIO_LENGTH_DISTRIBUTION,
        demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    )

    instance = configuration.generate()

    assert len(instance.piece_lengths) == 4
    for length, demand in zip(instance.piece_lengths, instance.demands, strict=True):
        multiplicity = _natural_multiplicity(instance.stock_length, instance.kerf, length)
        assert multiplicity == 2
        assert 5 <= demand <= 30
        assert demand % multiplicity != 0


def test_mixed_multiplicities_keep_only_trivial_types_divisible() -> None:
    configuration = SyntheticInstanceGenerator(
        seed=3,
        stock_length=100.0,
        number_of_types=3,
        piece_length_range=(10, 90),
        demand_range=(2, 20),
        demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    )

    instance = configuration.generate()

    for length, demand in zip(instance.piece_lengths, instance.demands, strict=True):
        multiplicity = _natural_multiplicity(instance.stock_length, instance.kerf, length)
        if multiplicity >= 2:
            assert demand % multiplicity != 0


def test_synthetic_generator_exercises_strictly_positive_kerf_reproducibly() -> None:
    configuration = SyntheticInstanceGenerator(seed=17, number_of_types=4, kerf=2.0)
    twin = SyntheticInstanceGenerator(seed=17, number_of_types=4, kerf=0.0)

    first = configuration.generate()
    second = configuration.generate()
    without_kerf = twin.generate()

    assert first == second
    assert first.piece_lengths == without_kerf.piece_lengths
    assert first.demands == without_kerf.demands
    assert first.kerf == 2.0
    assert first != without_kerf
    assert configuration.instance_id == configuration.instance_id
    assert configuration.instance_id != twin.instance_id
    assert configuration.family_id != twin.family_id


def test_generated_instance_applies_conservative_per_piece_kerf_convention() -> None:
    configuration = SyntheticInstanceGenerator(
        seed=5,
        stock_length=100.0,
        kerf=1.0,
        number_of_types=1,
        piece_length_range=(50, 50),
        demand_range=(3, 3),
    )
    without_kerf = SyntheticInstanceGenerator(
        seed=5,
        stock_length=100.0,
        kerf=0.0,
        number_of_types=1,
        piece_length_range=(50, 50),
        demand_range=(3, 3),
    )

    instance = configuration.generate()
    reference = without_kerf.generate()

    assert instance.initial_patterns() == ((1,),)
    assert reference.initial_patterns() == ((2,),)
    assert instance.capacity_used((1,)) == 51.0
    assert instance.capacity_used((2,)) == 102.0
    assert all(
        instance.capacity_used(pattern) <= instance.stock_length
        for pattern in instance.initial_patterns()
    )
    assert reference.capacity_used((2,)) == 100.0


def test_synthetic_generator_accepts_exact_fit_with_exercised_kerf() -> None:
    configuration = SyntheticInstanceGenerator(
        seed=5,
        stock_length=100.0,
        kerf=10.0,
        number_of_types=1,
        piece_length_range=(90, 90),
        demand_range=(2, 2),
    )

    instance = configuration.generate()

    assert instance.piece_lengths == (90.0,)
    assert instance.initial_patterns() == ((1,),)
    assert instance.capacity_used((1,)) == 100.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": 1, "number_of_types": 0},
        {"seed": 1, "number_of_types": True},
        {"seed": 1, "stock_length": 0},
        {"seed": 1, "kerf": -1},
        {"seed": 1, "kerf": float("inf")},
        {"seed": 1, "kerf": 95.0},
        {"seed": 1, "stock_length": 5.0, "kerf": 5.0},
        {"seed": 1, "piece_length_range": (0, 10)},
        {"seed": 1, "demand_range": (3, 2)},
        {"seed": 1, "piece_length_range": (10, 101)},
        {"seed": 1, "piece_length_range": None},
    ],
)
def test_synthetic_generator_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SyntheticInstanceGenerator(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "seed": 1,
            "piece_length_range": (10, 25),
            "length_distribution": TIGHT_RATIO_LENGTH_DISTRIBUTION,
        },
        {
            "seed": 1,
            "number_of_types": 7,
            "piece_length_range": (45, 50),
            "length_distribution": TIGHT_RATIO_LENGTH_DISTRIBUTION,
        },
        {
            "seed": 1,
            "stock_length": 10.0,
            "kerf": 6.0,
            "number_of_types": 1,
            "piece_length_range": (4, 4),
            "length_distribution": TIGHT_RATIO_LENGTH_DISTRIBUTION,
        },
        {
            "seed": 1,
            "number_of_types": 1,
            "piece_length_range": (50, 50),
            "demand_range": (10, 10),
            "demand_distribution": AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
        },
    ],
)
def test_structured_profiles_reject_unsatisfiable_generation(
    kwargs: dict[str, object]
) -> None:
    generator = SyntheticInstanceGenerator(**kwargs)

    with pytest.raises(ValueError):
        generator.generate()
