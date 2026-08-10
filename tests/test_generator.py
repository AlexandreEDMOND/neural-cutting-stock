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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": 1, "number_of_types": 0},
        {"seed": 1, "stock_length": 0},
        {"seed": 1, "kerf": -1},
        {"seed": 1, "kerf": float("inf")},
        {"seed": 1, "piece_length_range": (0, 10)},
        {"seed": 1, "demand_range": (3, 2)},
        {"seed": 1, "piece_length_range": (10, 101)},
    ],
)
def test_synthetic_generator_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SyntheticInstanceGenerator(**kwargs)
