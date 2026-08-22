"""Non-regression of the classical baseline on instances with an exact reference.

Every instance carrying an optimal, independently verified `exact-reference-v1`
record in the persisted exact-gap report must keep the classical column
generation LP-optimal — converged under the exact pricing control at the same
LP value as the complete master — and unchanged in objective compared with the
persisted classical campaign runs.
"""

import json
from pathlib import Path

import pytest

from neural_cutting_stock.benchmarks import (
    EXACT_GAP_SCHEMA_VERSION,
    SolverMode,
    SyntheticInstanceGenerator,
)
from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import ColumnGeneration
from neural_cutting_stock.visualization.phase4 import load_phase4_runs

ROOT = Path(__file__).parents[1]
EXACT_GAP_PATH = ROOT / "results/exact-gap.json"
FINAL_MANIFEST_PATH = ROOT / "data/phase-6-final/manifest.json"
PHASE4_RUNS_PATH = ROOT / "results/phase-4-benchmark-runs.csv"
REDUCED_COST_TOLERANCE = 1e-9
LP_BOUND_TOLERANCE = 1e-6
OBJECTIVE_TOLERANCE = 1e-9


def _reference_cases() -> list[object]:
    """Materialize every instance holding an optimal, verified exact reference."""

    report = json.loads(EXACT_GAP_PATH.read_text(encoding="utf-8"))
    if report.get("schema_version") != EXACT_GAP_SCHEMA_VERSION:
        raise ValueError("unsupported exact-gap report")
    manifest = json.loads(FINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    materialized = {
        entry["instance_id"]: CuttingStockInstance(
            entry["stock_length"], entry["kerf"], entry["piece_lengths"], entry["demands"]
        )
        for entry in manifest["instances"]
    }
    phase4_generators = _phase4_generator_parameters()

    cases = []
    for index, entry in enumerate(report["instances"]):
        if entry["reference_status"] != "optimal" or entry["verification_passed"] is not True:
            continue
        instance_id = entry["instance_id"]
        source_tag = Path(entry["source"]).name
        if instance_id in materialized:
            instance = materialized[instance_id]
            origin = "phase6-manifest"
        elif instance_id in phase4_generators:
            seed, number_of_types = phase4_generators[instance_id]
            generator = SyntheticInstanceGenerator(seed=seed, number_of_types=number_of_types)
            if generator.instance_id != instance_id:
                raise ValueError(
                    f"{source_tag}: generator reconstruction does not reproduce {instance_id}"
                )
            instance = generator.generate()
            origin = "phase4-runs"
        else:
            raise ValueError(f"{source_tag}: no persisted data to materialize {instance_id}")
        _require_same_instance_data(instance_id, instance, entry)
        cases.append(
            pytest.param(
                instance,
                entry,
                id=f"{index:02d}-{origin}-{entry['size_class']}",
            )
        )
    if not cases:
        raise ValueError("the exact-gap report holds no optimal verified reference")
    return cases


def _phase4_generator_parameters() -> dict[str, tuple[int, int]]:
    """Map each phase-4 instance to its recorded (seed, type count) signature."""

    signatures: dict[str, tuple[int, int]] = {}
    for record in load_phase4_runs(PHASE4_RUNS_PATH):
        if record.solver_mode is not SolverMode.CLASSICAL:
            continue
        signature = (record.seed, record.number_of_piece_types)
        previous = signatures.setdefault(record.instance_id, signature)
        if previous != signature:
            raise ValueError(
                f"inconsistent generator parameters across baseline runs of {record.instance_id}"
            )
    return signatures


def _require_same_instance_data(
    instance_id: str, instance: CuttingStockInstance, entry: dict
) -> None:
    total_demand = sum(instance.demands)
    matches = (
        instance.stock_length == entry["stock_length"]
        and instance.kerf == entry["kerf"]
        and instance.number_of_types == entry["number_of_piece_types"]
        and total_demand == entry["total_demand"]
    )
    if not matches:
        raise ValueError(f"materialized data does not match the exact-gap entry: {instance_id}")


@pytest.mark.parametrize("instance,entry", _reference_cases())
def test_classical_cg_stays_lp_optimal_and_unchanged_against_exact_reference(
    instance: CuttingStockInstance, entry: dict
) -> None:
    result = ColumnGeneration(instance, REDUCED_COST_TOLERANCE).solve()

    assert result.status == "converged"
    assert result.termination_reason == "no_improving_column"
    assert result.rmp_result.objective_value == pytest.approx(
        float(entry["lp_bound_bars"]), abs=LP_BOUND_TOLERANCE
    )

    bars = result.integer_master_result.objective_value
    assert result.verification.feasible
    assert result.verification.number_of_stock_bars == bars
    recorded_objectives = [float(value) for value in entry["baseline_objective_bars"]]
    assert recorded_objectives, "the persisted baseline must contain at least one objective"
    for recorded in recorded_objectives:
        assert bars == pytest.approx(recorded, abs=OBJECTIVE_TOLERANCE)


def test_reference_corpus_covers_every_optimal_verified_entry() -> None:
    report = json.loads(EXACT_GAP_PATH.read_text(encoding="utf-8"))
    expected = [
        entry
        for entry in report["instances"]
        if entry["reference_status"] == "optimal" and entry["verification_passed"] is True
    ]

    assert expected, "the exact-gap report must contain references to regress against"
    assert len(_reference_cases()) == len(expected)
