import pytest

from neural_cutting_stock.problem import CuttingStockInstance


def test_instance_normalizes_piece_types_and_applies_kerf() -> None:
    instance = CuttingStockInstance(
        stock_length=100,
        kerf=2,
        piece_lengths=[30, 10, 30],
        demands=[2, 4, 3],
    )

    assert instance.piece_lengths == (10.0, 30.0)
    assert instance.demands == (4, 5)
    assert instance.number_of_types == 2
    assert instance.capacity_used((1, 2)) == 76


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stock_length": 0, "kerf": 0, "piece_lengths": [1], "demands": [1]}, "stock_length"),
        ({"stock_length": 10, "kerf": -1, "piece_lengths": [1], "demands": [1]}, "kerf"),
        ({"stock_length": 10, "kerf": 0, "piece_lengths": [], "demands": []}, "piece_lengths"),
        ({"stock_length": 10, "kerf": 0, "piece_lengths": [1], "demands": [1, 2]}, "same length"),
        ({"stock_length": 10, "kerf": 0, "piece_lengths": [0], "demands": [1]}, "piece lengths"),
        ({"stock_length": 10, "kerf": 1, "piece_lengths": [10], "demands": [1]}, "fit"),
        (
            {"stock_length": 10, "kerf": 0, "piece_lengths": [1], "demands": [0]},
            "positive integers",
        ),
    ],
)
def test_instance_rejects_invalid_data(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CuttingStockInstance(**kwargs)


def test_capacity_used_rejects_invalid_pattern() -> None:
    instance = CuttingStockInstance(10, 0, [2], [1])

    with pytest.raises(ValueError, match="one count"):
        instance.capacity_used((1, 2))
    with pytest.raises(ValueError, match="non-negative integers"):
        instance.capacity_used((-1,))


def test_initial_patterns_are_demand_bounded_and_feasible_with_kerf() -> None:
    instance = CuttingStockInstance(
        stock_length=100,
        kerf=2,
        piece_lengths=[30, 10],
        demands=[10, 4],
    )

    assert instance.initial_patterns() == ((4, 0), (0, 3))
    assert all(
        instance.capacity_used(pattern) <= instance.stock_length
        for pattern in instance.initial_patterns()
    )
    assert all(any(pattern[index] for pattern in instance.initial_patterns()) for index in range(2))
