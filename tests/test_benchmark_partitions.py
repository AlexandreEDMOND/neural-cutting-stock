import pytest

from neural_cutting_stock.benchmarks import (
    DatasetPartition,
    PartitionPlan,
    SyntheticInstanceGenerator,
)


def _plan(*generators: SyntheticInstanceGenerator) -> PartitionPlan:
    families = [generator.family_id for generator in generators]
    return PartitionPlan(
        train_seeds=(11,),
        validation_seeds=(12,),
        test_seeds=(13,),
        train_families=(families[0],),
        validation_families=(families[1],),
        test_families=(families[2],),
    )


def test_family_id_is_stable_across_seeds_and_changes_with_family_configuration() -> None:
    first = SyntheticInstanceGenerator(seed=11, number_of_types=2)
    second = SyntheticInstanceGenerator(seed=12, number_of_types=2)
    other = SyntheticInstanceGenerator(seed=11, number_of_types=3)

    assert first.family_id == second.family_id
    assert first.family_id != other.family_id


def test_partition_plan_assigns_by_seed_and_family_before_collection() -> None:
    generators = tuple(
        SyntheticInstanceGenerator(seed=seed, number_of_types=number_of_types)
        for seed, number_of_types in zip((11, 12, 13), (2, 3, 4), strict=True)
    )
    plan = _plan(*generators)

    assignments = plan.assignments(generators)

    assert [assignment.partition for assignment in assignments] == [
        DatasetPartition.TRAIN,
        DatasetPartition.VALIDATION,
        DatasetPartition.TEST,
    ]
    assert plan.to_dict()["schema_version"] == "trajectory-partitions-v1"


def test_partition_plan_rejects_cross_partition_seed_and_family() -> None:
    generators = tuple(
        SyntheticInstanceGenerator(seed=seed, number_of_types=number_of_types)
        for seed, number_of_types in zip((11, 12, 13), (2, 3, 4), strict=True)
    )
    plan = _plan(*generators)
    cross_partition = SyntheticInstanceGenerator(seed=11, number_of_types=3)

    with pytest.raises(ValueError, match="different partitions"):
        plan.assign(cross_partition)


def test_partition_plan_rejects_overlapping_axes() -> None:
    generator = SyntheticInstanceGenerator(seed=11)

    with pytest.raises(ValueError, match="disjoint"):
        PartitionPlan(
            train_seeds=(11,),
            validation_seeds=(11,),
            test_seeds=(12,),
            train_families=(generator.family_id,),
            validation_families=("validation",),
            test_families=("test",),
        )
