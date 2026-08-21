import json
from pathlib import Path

import pytest
from test_benchmark_comparison import _record as base_record
from test_benchmark_schema import _trajectory_metadata

from neural_cutting_stock.benchmarks import (
    CORPUS_SCHEMA_VERSION,
    ColumnGenerationTrajectory,
    EnvironmentMetadata,
    RunStatus,
    SolverMode,
    TrajectoryIteration,
    TrajectoryStatus,
    evaluate_size_generalization,
    pair_campaign_records,
    training_size_frontier,
    trajectory_sha256,
    write_trajectory,
)
from neural_cutting_stock.visualization.phase4 import load_phase4_runs

ROOT = Path(__file__).parents[1]


def _campaign_record(mode: SolverMode, instance: str, repetition: int, **changes: object):
    return base_record(
        mode,
        f"{mode.value}-{instance}-{repetition}",
        instance_id=instance,
        repetition=repetition,
        config_id=f"config-{mode.value}",
        **changes,
    )


def _write_corpus(tmp_path: Path, entries: list[tuple[str, str, int]]) -> Path:
    manifest_entries = []
    for name, partition, type_count in entries:
        lengths = tuple(10.0 + 5.0 * index for index in range(type_count))
        metadata = _trajectory_metadata(
            trajectory_id=f"trajectory-{name}",
            piece_lengths=lengths,
            dual_type_order=lengths,
            demands=tuple(1 + index for index in range(type_count)),
        )
        iteration = TrajectoryIteration(
            1,
            "optimal",
            dual_values=tuple(1.0 + index for index in range(type_count)),
            candidate_patterns=((1,) * type_count,),
            candidate_reduced_costs=(-0.5,),
            selected_patterns=((1,) * type_count,),
        )
        trajectory = ColumnGenerationTrajectory(
            metadata, (iteration,), TrajectoryStatus.CONVERGED, "no_improving_column"
        )
        path = tmp_path / f"{name}.json"
        write_trajectory(path, trajectory)
        manifest_entries.append(
            {
                "trajectory_id": metadata.trajectory_id,
                "path": f"{name}.json",
                "partition": partition,
                "sha256": trajectory_sha256(trajectory),
            }
        )
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "trajectories": manifest_entries,
        "statistics": {},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_training_size_frontier_is_the_corpus_maximum_across_partitions(tmp_path: Path) -> None:
    manifest_path = _write_corpus(
        tmp_path,
        [("train", "train", 2), ("validation", "validation", 3), ("test", "test", 4)],
    )

    frontier = training_size_frontier(manifest_path)

    assert frontier == {
        "source_schema_version": CORPUS_SCHEMA_VERSION,
        "maximum_training_piece_types": 4,
        "piece_types_by_partition": {"test": 4, "train": 2, "validation": 3},
    }


def test_training_size_frontier_rejects_tampered_trajectory(tmp_path: Path) -> None:
    manifest_path = _write_corpus(tmp_path, [("train", "train", 2)])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["trajectories"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="trajectory hash differs"):
        training_size_frontier(manifest_path)


def test_campaign_pairing_matches_repetitions_across_config_ids() -> None:
    classical = (
        _campaign_record(SolverMode.CLASSICAL, "i-1", 1),
        _campaign_record(SolverMode.CLASSICAL, "i-1", 0),
    )
    neural = (
        _campaign_record(SolverMode.NEURAL, "i-1", 0),
        _campaign_record(SolverMode.NEURAL, "i-1", 1),
    )

    pairs = pair_campaign_records(classical, neural)

    assert [
        (classical.repetition, neural.repetition) for classical, neural in pairs
    ] == [(0, 0), (1, 1)]
    assert {classical.config_id for classical, _ in pairs} == {"config-classical"}
    assert {neural.config_id for _, neural in pairs} == {"config-neural"}


def test_campaign_pairing_rejects_missing_duplicate_or_mismatched_runs() -> None:
    with pytest.raises(ValueError, match="missing paired campaign run"):
        pair_campaign_records(
            (_campaign_record(SolverMode.CLASSICAL, "i-1", 0),),
            (_campaign_record(SolverMode.NEURAL, "i-2", 0),),
        )
    with pytest.raises(ValueError, match="duplicate classical run"):
        pair_campaign_records(
            (
                _campaign_record(SolverMode.CLASSICAL, "i-1", 0),
                _campaign_record(SolverMode.CLASSICAL, "i-1", 0),
            ),
            (_campaign_record(SolverMode.NEURAL, "i-1", 0),),
        )
    with pytest.raises(ValueError, match="same environment"):
        pair_campaign_records(
            (_campaign_record(SolverMode.CLASSICAL, "i-1", 0),),
            (
                _campaign_record(
                    SolverMode.NEURAL,
                    "i-1",
                    0,
                    environment=EnvironmentMetadata("other", "3.11", "deps", "machine"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="same instance data: stock_length"):
        pair_campaign_records(
            (_campaign_record(SolverMode.CLASSICAL, "i-1", 0),),
            (_campaign_record(SolverMode.NEURAL, "i-1", 0, stock_length=101.0),),
        )


def test_generalization_flags_only_sizes_above_the_frontier() -> None:
    frontier = {"maximum_training_piece_types": 4}
    records = [
        record
        for mode in (SolverMode.CLASSICAL, SolverMode.NEURAL)
        for record in (
            _campaign_record(mode, "within", 0, number_of_piece_types=4),
            _campaign_record(mode, "above", 0, number_of_piece_types=6),
        )
    ]

    report = evaluate_size_generalization(
        [r for r in records if r.solver_mode is SolverMode.CLASSICAL],
        [r for r in records if r.solver_mode is SolverMode.NEURAL],
        frontier,
    )

    flags = {item["instance_id"]: item["above_training_size"] for item in report["instances"]}
    assert flags == {"within": False, "above": True}
    assert report["coverage"]["instance_count_above_training"] == 1
    assert report["coverage"]["instance_count_within_training"] == 1


def test_generalization_keeps_failed_repetitions_visible_and_non_admissible() -> None:
    frontier = {"maximum_training_piece_types": 4}
    classical = [
        _campaign_record(SolverMode.CLASSICAL, "i-1", 0, number_of_piece_types=6),
        _campaign_record(SolverMode.CLASSICAL, "i-1", 1, number_of_piece_types=6),
    ]
    neural = [
        _campaign_record(SolverMode.NEURAL, "i-1", 0, number_of_piece_types=6),
        _campaign_record(
            SolverMode.NEURAL,
            "i-1",
            1,
            number_of_piece_types=6,
            run_status=RunStatus.TIMEOUT,
            termination_reason="resource_limit",
            error_message="max runtime reached",
            objective_value=None,
            plan_feasible=None,
            total_runtime_seconds=None,
        ),
    ]

    report = evaluate_size_generalization(classical, neural, frontier)

    item = report["instances"][0]
    assert item["status_counts"] == {
        "classical:optimal_lp_restricted_ip": 2,
        "neural:optimal_lp_restricted_ip": 1,
        "neural:timeout": 1,
    }
    assert item["objective_differences_vs_classical"] == [0.0, None]
    assert item["admissible_repetition_count"] == 1
    assert item["above_training_size"] is True
    assert report["coverage"]["admissible_pair_count_above_training"] == 1


def test_generalization_excludes_quality_violations_from_admissible_counts() -> None:
    frontier = {"maximum_training_piece_types": 4}
    classical = [_campaign_record(SolverMode.CLASSICAL, "i-1", 0, number_of_piece_types=8)]
    neural = [
        _campaign_record(
            SolverMode.NEURAL, "i-1", 0, number_of_piece_types=8, objective_value=6.0
        )
    ]

    report = evaluate_size_generalization(classical, neural, frontier)

    item = report["instances"][0]
    assert item["objective_differences_vs_classical"] == [1.0]
    assert item["admissible_repetition_count"] == 0
    assert report["coverage"]["objective_differences_vs_classical_above_training"] == [1.0]


def test_generalization_validates_tolerance_and_frontier() -> None:
    classical = [_campaign_record(SolverMode.CLASSICAL, "i-1", 0)]
    neural = [_campaign_record(SolverMode.NEURAL, "i-1", 0)]

    with pytest.raises(ValueError, match="quality_tolerance"):
        evaluate_size_generalization(
            classical, neural, {"maximum_training_piece_types": 4}, -1.0
        )
    with pytest.raises(ValueError, match="frontier"):
        evaluate_size_generalization(
            classical, neural, {"maximum_training_piece_types": None}
        )


def test_phase6_final_campaigns_cover_sizes_above_training() -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text(encoding="utf-8"))
    frontier = training_size_frontier(ROOT / config["partitions"]["manifest"])

    report = evaluate_size_generalization(
        load_phase4_runs(ROOT / "results/phase-6-classical-runs.csv"),
        load_phase4_runs(ROOT / "results/phase-6-neural-runs.csv"),
        frontier,
        quality_tolerance=config["protocol"]["quality_tolerance_bars"],
    )

    assert frontier["maximum_training_piece_types"] == 4
    above = [item for item in report["instances"] if item["above_training_size"]]
    within = [item for item in report["instances"] if not item["above_training_size"]]
    assert {item["number_of_piece_types"] for item in above} == {6, 8}
    assert {item["number_of_piece_types"] for item in within} == {2, 4}
    assert len(above) == 6 and len(within) == 6
    assert all(item["repetition_count"] == 3 for item in report["instances"])
    assert report["pair_count"] == 36
    assert all(
        set(item["status_counts"])
        == {
            "classical:optimal_lp_restricted_ip",
            "neural:optimal_lp_restricted_ip",
        }
        and sum(item["status_counts"].values()) == 6
        for item in report["instances"]
    )
    assert all(
        difference == 0.0
        for item in report["instances"]
        for difference in item["objective_differences_vs_classical"]
    )
