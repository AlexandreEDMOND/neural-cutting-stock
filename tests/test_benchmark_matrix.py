import pytest

from neural_cutting_stock.benchmarks import BenchmarkMatrix, DistributionSpec


def test_benchmark_matrix_builds_cartesian_product_of_independent_dimensions() -> None:
    matrix = BenchmarkMatrix(
        seeds=(1, 2),
        number_of_types=(2, 3),
        stock_lengths=(100.0,),
        kerfs=(0.0, 1.0),
        length_distributions=(DistributionSpec("short", (10, 30)),),
        demand_distributions=(
            DistributionSpec("low", (1, 3)),
            DistributionSpec("high", (8, 10)),
        ),
    )

    generators = matrix.generators()

    assert matrix.size == 16
    assert len(generators) == matrix.size
    assert [(item.seed, item.number_of_types, item.kerf) for item in generators[:4]] == [
        (1, 2, 0.0),
        (1, 2, 0.0),
        (1, 2, 1.0),
        (1, 2, 1.0),
    ]
    assert {(item.length_distribution, item.demand_distribution) for item in generators} == {
        ("short", "low"),
        ("short", "high"),
    }
    assert all(item.generate().stock_length == 100.0 for item in generators)


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": ()},
        {"number_of_types": (0,)},
        {"stock_lengths": (0.0,)},
        {"kerfs": (-1.0,)},
        {"length_distributions": ()},
        {"demand_distributions": ()},
    ],
)
def test_benchmark_matrix_rejects_empty_or_invalid_axes(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "seeds": (1,),
        "number_of_types": (2,),
        "stock_lengths": (100.0,),
        "kerfs": (0.0,),
        "length_distributions": (DistributionSpec("lengths", (10, 30)),),
        "demand_distributions": (DistributionSpec("demands", (1, 3)),),
    }
    values.update(changes)

    with pytest.raises(ValueError):
        BenchmarkMatrix(**values)
