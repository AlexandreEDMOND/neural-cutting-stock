import json

import pytest

from neural_cutting_stock.problem import (
    MULTI_STOCK_FORMAT_SCHEMA_VERSION,
    CuttingStockInstance,
    MultiFormatCuttingStockInstance,
)


def test_multi_format_instance_normalizes_formats_and_pieces() -> None:
    instance = MultiFormatCuttingStockInstance(
        stock_lengths=[100, 50],
        kerf=0,
        piece_lengths=[70, 20, 20],
        demands=[1, 2, 3],
    )

    assert instance.stock_lengths == (50.0, 100.0)
    assert instance.largest_stock_length == 100.0
    assert instance.piece_lengths == (20.0, 70.0)
    assert instance.demands == (5, 1)
    assert instance.number_of_types == 2


def test_multi_format_instance_is_invariant_to_input_order() -> None:
    first = MultiFormatCuttingStockInstance([100.0, 50.0], 1.0, [70, 20], [1, 2])
    second = MultiFormatCuttingStockInstance([50, 100], 1.0, [20, 70], [2, 1])

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_multi_format_instance_accepts_three_declared_formats() -> None:
    instance = MultiFormatCuttingStockInstance(
        stock_lengths=[120.0, 40.0, 80.0],
        kerf=0.0,
        piece_lengths=[35.0],
        demands=[4],
    )

    assert instance.stock_lengths == (40.0, 80.0, 120.0)


@pytest.mark.parametrize("stock_lengths", [[100.0], [100.0, 50.0, 25.0, 10.0]])
def test_multi_format_instance_requires_two_or_three_formats(stock_lengths: list[float]) -> None:
    with pytest.raises(ValueError, match="between two and three"):
        MultiFormatCuttingStockInstance(stock_lengths, 0.0, [10.0], [1])


@pytest.mark.parametrize(
    "stock_lengths",
    [
        None,
        [100.0, 100.0],
        [100.0, 50.0, 100.0],
        [True, 50.0],
        [0.0, 50.0],
        [-100.0, 50.0],
        [float("inf"), 50.0],
        [float("nan"), 50.0],
    ],
)
def test_multi_format_instance_rejects_invalid_stock_lengths(stock_lengths: object) -> None:
    with pytest.raises(ValueError):
        MultiFormatCuttingStockInstance(stock_lengths, 0.0, [10.0], [1])  # type: ignore[arg-type]


def test_multi_format_instance_reuses_single_format_kerf_validation() -> None:
    with pytest.raises(ValueError, match="kerf must be non-negative"):
        MultiFormatCuttingStockInstance([100.0, 50.0], -1.0, [10.0], [1])
    with pytest.raises(ValueError, match="positive integers"):
        MultiFormatCuttingStockInstance([100.0, 50.0], 0.0, [10.0], [0])
    with pytest.raises(ValueError, match="same length"):
        MultiFormatCuttingStockInstance([100.0, 50.0], 0.0, [10.0], [1, 2])


def test_multi_format_instance_only_requires_a_fit_on_the_largest_format() -> None:
    instance = MultiFormatCuttingStockInstance(
        stock_lengths=[60.0, 100.0],
        kerf=10.0,
        piece_lengths=[90.0, 45.0],
        demands=[1, 1],
    )

    assert instance.piece_lengths == (45.0, 90.0)
    assert instance.capacity_used((0, 1)) == 100.0
    assert instance.fits_on((0, 1), 100.0)
    assert not instance.fits_on((0, 1), 60.0)

    with pytest.raises(ValueError, match="every piece must fit"):
        MultiFormatCuttingStockInstance([60.0, 100.0], 10.0, [95.0], [1])

    with pytest.raises(ValueError, match="declared stock lengths"):
        instance.fits_on((0, 1), 80.0)


def test_capacity_used_keeps_the_conservative_per_piece_kerf_rule() -> None:
    reference = CuttingStockInstance(100.0, 2.0, [30.0, 10.0], [10, 4])
    multi = MultiFormatCuttingStockInstance([100.0, 50.0], 2.0, [30.0, 10.0], [10, 4])

    pattern = (2, 1)
    assert multi.capacity_used(pattern) == 56.0
    assert multi.capacity_used(pattern) == reference.capacity_used(pattern)

    with pytest.raises(ValueError, match="one count"):
        multi.capacity_used((2,))
    with pytest.raises(ValueError, match="non-negative integers"):
        multi.capacity_used((-1, 0))


def test_decimal_capacity_does_not_lose_a_piece_to_binary_rounding() -> None:
    instance = MultiFormatCuttingStockInstance([0.5, 0.3], 0.0, [0.1], [3])

    assert instance.capacity_used((3,)) == 0.3


def test_to_dict_matches_the_versioned_schema() -> None:
    instance = MultiFormatCuttingStockInstance([100.0, 50.0], 1.0, [70, 20], [1, 2])

    payload = instance.to_dict()

    assert payload["schema_version"] == MULTI_STOCK_FORMAT_SCHEMA_VERSION
    assert MULTI_STOCK_FORMAT_SCHEMA_VERSION == "multi-stock-format-v1"
    assert payload["stock_lengths"] == [50.0, 100.0]
    assert payload["kerf"] == 1.0
    assert payload["piece_lengths"] == [20.0, 70.0]
    assert payload["demands"] == [2, 1]


def test_from_dict_round_trips_through_json() -> None:
    instance = MultiFormatCuttingStockInstance([100.0, 50.0], 1.0, [70, 20], [1, 2])

    encoded = json.dumps(instance.to_dict())
    restored = MultiFormatCuttingStockInstance.from_dict(json.loads(encoded))

    assert restored == instance
    assert restored.to_dict() == instance.to_dict()


def test_from_dict_rejects_invalid_payloads() -> None:
    valid = MultiFormatCuttingStockInstance([100.0, 50.0], 0.0, [10.0], [1]).to_dict()

    with pytest.raises(ValueError, match="JSON object"):
        MultiFormatCuttingStockInstance.from_dict([valid])
    wrong_version = dict(valid, schema_version="multi-stock-format-v2")
    with pytest.raises(ValueError, match="unsupported schema_version"):
        MultiFormatCuttingStockInstance.from_dict(wrong_version)
    missing_kerf = {key: value for key, value in valid.items() if key != "kerf"}
    with pytest.raises(ValueError, match="missing fields"):
        MultiFormatCuttingStockInstance.from_dict(missing_kerf)
    with pytest.raises(ValueError, match="unknown fields"):
        MultiFormatCuttingStockInstance.from_dict(dict(valid, extra_field=1))
