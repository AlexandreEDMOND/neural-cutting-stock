import json
from pathlib import Path

import pytest
from test_benchmark_comparison import _record as base_record

from neural_cutting_stock.benchmarks import (
    PAIRED_TABLES_SCHEMA_VERSION,
    RunStatus,
    SolverMode,
    build_paired_tables,
)
from neural_cutting_stock.visualization.phase4 import load_phase4_runs
from neural_cutting_stock.visualization.phase6 import write_paired_tables_markdown

ROOT = Path(__file__).parents[1]


def _campaign_record(mode: SolverMode, instance: str, repetition: int, **changes: object):
    values: dict[str, object] = {
        "peak_memory_bytes": 1000 + repetition,
        "number_of_cg_iterations": 1 + repetition,
        "number_of_generated_columns": 2 + repetition,
        "number_of_columns_added": 2 + repetition,
        "final_column_count": 4 + repetition,
    }
    values.update(changes)
    return base_record(
        mode,
        f"{mode.value}-{instance}-{repetition}",
        instance_id=instance,
        repetition=repetition,
        config_id=f"config-{mode.value}",
        **values,
    )


def _both_modes(instance: str, repetition: int, **changes: object):
    return [
        _campaign_record(SolverMode.CLASSICAL, instance, repetition, **changes),
        _campaign_record(SolverMode.NEURAL, instance, repetition, **changes),
    ]


def test_paired_tables_cover_quality_runtime_memory_iterations_and_columns() -> None:
    records = [
        record
        for instance in ("i-1", "i-2")
        for repetition in (0, 1)
        for record in _both_modes(instance, repetition)
    ]

    report = build_paired_tables(
        [r for r in records if r.solver_mode is SolverMode.CLASSICAL],
        [r for r in records if r.solver_mode is SolverMode.NEURAL],
    )

    assert (
        report["schema_version"] == PAIRED_TABLES_SCHEMA_VERSION == "phase-6-paired-tables-v1"
    )
    assert report["run_count"] == 8 and report["pair_count"] == 4
    assert report["admissible_pair_count"] == 4
    pair = report["pairs"][0]
    for field in (
        "objective_difference_vs_classical",
        "quality_preserved",
        "speedup_vs_classical",
        "classical_peak_memory_bytes",
        "neural_peak_memory_bytes",
        "classical_cg_iterations",
        "neural_cg_iterations",
        "classical_generated_columns",
        "neural_final_columns",
    ):
        assert field in pair
    assert pair["quality_preserved"] is True and pair["admissible"] is True
    assert pair["objective_difference_vs_classical"] == 0.0
    assert pair["speedup_vs_classical"] == pytest.approx(2.0)
    assert len(report["instances"]) == 2
    item = report["instances"][0]
    assert item["repetition_count"] == 2 and item["admissible_repetition_count"] == 2
    assert item["objective_difference_vs_classical_median"] == 0.0
    assert item["speedup_vs_classical_median"] == pytest.approx(2.0)
    assert item["classical_peak_memory_bytes_median"] == 1000.5
    assert item["neural_peak_memory_bytes_median"] == 1000.5
    assert item["classical_cg_iterations_median"] == 1.5
    assert item["classical_added_columns_median"] == 2.5
    assert item["neural_final_columns_median"] == 4.5


def test_paired_tables_keep_failed_repetitions_visible_and_out_of_medians() -> None:
    timeout_changes = {
        "run_status": RunStatus.TIMEOUT,
        "termination_reason": "resource_limit",
        "error_message": "max runtime reached",
        "objective_value": None,
        "plan_feasible": None,
        "total_runtime_seconds": None,
        "peak_memory_bytes": None,
        "number_of_cg_iterations": None,
        "number_of_generated_columns": None,
        "number_of_columns_added": None,
        "final_column_count": None,
    }
    records = [
        *_both_modes("i-1", 0),
        _campaign_record(SolverMode.CLASSICAL, "i-1", 1),
        _campaign_record(SolverMode.NEURAL, "i-1", 1, **timeout_changes),
    ]

    report = build_paired_tables(
        [r for r in records if r.solver_mode is SolverMode.CLASSICAL],
        [r for r in records if r.solver_mode is SolverMode.NEURAL],
    )

    assert report["admissible_pair_count"] == 1
    item = report["instances"][0]
    assert item["repetition_count"] == 2 and item["admissible_repetition_count"] == 1
    assert item["status_counts"]["neural:timeout"] == 1
    assert item["classical_runtime_seconds_median"] == 2.0
    assert item["classical_peak_memory_bytes_median"] == 1000
    failed_pair = report["pairs"][1]
    assert failed_pair["admissible"] is False
    assert failed_pair["neural_run_status"] == "timeout"
    assert failed_pair["speedup_vs_classical"] is None


def test_paired_tables_flag_quality_violations_without_aggregating_them() -> None:
    records = [
        *_both_modes("i-1", 0),
        _campaign_record(SolverMode.CLASSICAL, "i-1", 1),
        _campaign_record(
            SolverMode.NEURAL,
            "i-1",
            1,
            objective_value=6.0,
            peak_memory_bytes=1001,
            number_of_cg_iterations=2,
            number_of_generated_columns=3,
            number_of_columns_added=3,
            final_column_count=5,
        ),
    ]

    report = build_paired_tables(
        [r for r in records if r.solver_mode is SolverMode.CLASSICAL],
        [r for r in records if r.solver_mode is SolverMode.NEURAL],
        quality_tolerance=0.0,
    )

    violating = report["pairs"][1]
    assert violating["objective_difference_vs_classical"] == 1.0
    assert violating["quality_preserved"] is False and violating["admissible"] is False
    item = report["instances"][0]
    assert item["quality_violation_pair_count"] == 1
    assert item["admissible_repetition_count"] == 1
    assert item["objective_difference_vs_classical_median"] == 0.0


def test_paired_tables_reject_invalid_tolerance() -> None:
    with pytest.raises(ValueError, match="quality_tolerance"):
        build_paired_tables(
            [_campaign_record(SolverMode.CLASSICAL, "i-1", 0)],
            [_campaign_record(SolverMode.NEURAL, "i-1", 0)],
            quality_tolerance=-1.0,
        )


def test_phase6_paired_tables_match_the_published_artifacts(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text(encoding="utf-8"))
    classical = load_phase4_runs(ROOT / "results/phase-6-classical-runs.csv")
    neural = load_phase4_runs(ROOT / "results/phase-6-neural-runs.csv")

    report = build_paired_tables(
        classical,
        neural,
        quality_tolerance=config["protocol"]["quality_tolerance_bars"],
    )

    assert report["run_count"] == 72 and report["pair_count"] == 36
    assert report["admissible_pair_count"] == 36
    assert len(report["instances"]) == 12
    assert all(item["admissible_repetition_count"] == 3 for item in report["instances"])
    assert all(item["quality_violation_pair_count"] == 0 for item in report["instances"])
    assert all(pair["objective_difference_vs_classical"] == 0.0 for pair in report["pairs"])
    assert all(
        pair["classical_peak_memory_bytes"] is not None
        and pair["neural_peak_memory_bytes"] is not None
        for pair in report["pairs"]
    )
    assert all(pair["classical_cg_iterations"] >= 1 for pair in report["pairs"])

    json_path = tmp_path / "phase-6-paired-tables.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert json_path.read_text(encoding="utf-8") == (
        ROOT / "results/phase-6-paired-tables.json"
    ).read_text(encoding="utf-8")

    markdown_path = tmp_path / "phase-6-paired-tables.md"
    write_paired_tables_markdown(
        report,
        markdown_path,
        "results/phase-6-classical-runs.csv",
        "results/phase-6-neural-runs.csv",
    )
    assert markdown_path.read_text(encoding="utf-8") == (
        ROOT / "results/phase-6-paired-tables.md"
    ).read_text(encoding="utf-8")
