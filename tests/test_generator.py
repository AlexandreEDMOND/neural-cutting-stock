import pytest

from neural_cutting_stock.benchmarks import SyntheticInstanceGenerator


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": 1, "number_of_types": 0},
        {"seed": 1, "number_of_types": True},
        {"seed": 1, "stock_length": 0},
        {"seed": 1, "kerf": -1},
        {"seed": 1, "kerf": float("inf")},
        {"seed": 1, "piece_length_range": (0, 10)},
        {"seed": 1, "demand_range": (3, 2)},
        {"seed": 1, "piece_length_range": (10, 101)},
        {"seed": 1, "piece_length_range": None},
    ],
)
def test_synthetic_generator_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SyntheticInstanceGenerator(**kwargs)
