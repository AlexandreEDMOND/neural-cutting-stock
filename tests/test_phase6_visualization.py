import json
from pathlib import Path

import pytest
from test_benchmark_comparison import _record as base_record

from neural_cutting_stock.benchmarks import SolverMode, validate_final_manifest
from neural_cutting_stock.visualization.phase4 import load_phase4_runs
from neural_cutting_stock.visualization.phase6 import (
    phase6_runtime_comparison_data,
    write_phase6_runtime_comparison,
    write_phase6_speedup_by_size,
)

ROOT = Path(__file__).parents[1]


def _campaign_record(mode: SolverMode, instance: str, repetition: int, **changes: object):
    return base_record(
        mode,
        f"{mode.value}-{instance}-{repetition}",
        instance_id=instance,
        repetition=repetition,
        **changes,
    )


def _runtime_pair(instance: str, repetition: int, classical_seconds: float, neural_seconds: float):
    return [
        _campaign_record(
            SolverMode.CLASSICAL, instance, repetition, total_runtime_seconds=classical_seconds
        ),
        _campaign_record(
            SolverMode.NEURAL, instance, repetition, total_runtime_seconds=neural_seconds
        ),
    ]


def _split_modes(records):
    return (
        [r for r in records if r.solver_mode is SolverMode.CLASSICAL],
        [r for r in records if r.solver_mode is SolverMode.NEURAL],
    )


def test_phase6_runtime_medians_aggregate_only_admissible_pairs(tmp_path: Path) -> None:
    violating_neural = _campaign_record(
        SolverMode.NEURAL,
        "i-large",
        2,
        total_runtime_seconds=9.0,
        objective_value=6.0,
    )
    records = [
        *_runtime_pair("i-small", 0, 2.0, 1.0),
        *_runtime_pair("i-small", 1, 4.0, 3.0),
        *_runtime_pair("i-large", 0, 10.0, 5.0),
        *_runtime_pair("i-large", 1, 12.0, 7.0),
        _campaign_record(SolverMode.CLASSICAL, "i-large", 2, total_runtime_seconds=20.0),
        violating_neural,
    ]
    classical, neural = _split_modes(records)

    data = phase6_runtime_comparison_data(
        classical, neural, {"i-small": "SMALL", "i-large": "LARGE"}
    )

    assert data["report"]["pair_count"] == 5
    assert data["report"]["admissible_pair_count"] == 4
    small = data["size_data"]["SMALL"]
    large = data["size_data"]["LARGE"]
    assert small == {
        "pair_count": 2,
        "instance_count": 1,
        "classical_median_seconds": 3.0,
        "neural_median_seconds": 2.0,
        "speedup_median": pytest.approx((2.0 / 1.0 + 4.0 / 3.0) / 2),
    }
    assert large["pair_count"] == 2 and large["instance_count"] == 1
    assert large["classical_median_seconds"] == 11.0
    assert large["neural_median_seconds"] == 6.0
    assert large["speedup_median"] == pytest.approx((10.0 / 5.0 + 12.0 / 7.0) / 2)
    for empty in ("MEDIUM", "XL"):
        assert data["size_data"][empty]["pair_count"] == 0
        assert data["size_data"][empty]["classical_median_seconds"] is None
        assert data["size_data"][empty]["speedup_median"] is None

    write_phase6_runtime_comparison(data, tmp_path)
    write_phase6_speedup_by_size(data, tmp_path)
    assert (tmp_path / "runtime_comparison.png").stat().st_size > 0
    assert (tmp_path / "speedup_by_size.png").stat().st_size > 0


def test_phase6_runtime_data_rejects_instances_missing_from_the_final_manifest() -> None:
    classical, neural = _split_modes(_runtime_pair("i-unknown", 0, 2.0, 1.0))

    with pytest.raises(ValueError, match="missing from the final instance manifest"):
        phase6_runtime_comparison_data(classical, neural, {})


def test_phase6_runtime_figure_refuses_to_plot_without_admissible_pairs(tmp_path: Path) -> None:
    records = [
        _campaign_record(SolverMode.CLASSICAL, "i-1", 0),
        _campaign_record(SolverMode.NEURAL, "i-1", 0, objective_value=6.0),
    ]
    classical, neural = _split_modes(records)

    data = phase6_runtime_comparison_data(classical, neural, {"i-1": "SMALL"})

    assert data["report"]["admissible_pair_count"] == 0
    with pytest.raises(ValueError, match="no admissible pair"):
        write_phase6_runtime_comparison(data, tmp_path)


def test_phase6_speedup_figure_refuses_to_plot_without_admissible_pairs(tmp_path: Path) -> None:
    records = [
        _campaign_record(SolverMode.CLASSICAL, "i-1", 0),
        _campaign_record(SolverMode.NEURAL, "i-1", 0, objective_value=6.0),
    ]
    classical, neural = _split_modes(records)

    data = phase6_runtime_comparison_data(classical, neural, {"i-1": "SMALL"})

    assert data["report"]["admissible_pair_count"] == 0
    assert data["size_data"]["SMALL"]["speedup_median"] is None
    with pytest.raises(ValueError, match="no admissible pair"):
        write_phase6_speedup_by_size(data, tmp_path)


def test_phase6_runtime_figure_is_derived_from_published_final_results(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / config["files"]["final_instance_manifest"]).read_text(encoding="utf-8")
    )
    validate_final_manifest(
        manifest,
        json.loads((ROOT / config["partitions"]["manifest"]).read_text(encoding="utf-8")),
    )
    size_class_by_instance = {
        entry["instance_id"]: entry["target_size_class"] for entry in manifest["instances"]
    }

    data = phase6_runtime_comparison_data(
        load_phase4_runs(ROOT / "results/phase-6-classical-runs.csv"),
        load_phase4_runs(ROOT / "results/phase-6-neural-runs.csv"),
        size_class_by_instance,
        quality_tolerance=config["protocol"]["quality_tolerance_bars"],
    )

    assert data["report"]["run_count"] == 72
    assert data["report"]["pair_count"] == 36
    assert data["report"]["admissible_pair_count"] == 36
    assert sum(item["pair_count"] for item in data["size_data"].values()) == 36
    assert sorted(data["size_data"]) == ["LARGE", "MEDIUM", "SMALL", "XL"]
    assert len(size_class_by_instance) == 12
    for item in data["size_data"].values():
        assert item["pair_count"] == 9 and item["instance_count"] == 3
        assert item["classical_median_seconds"] > 0.0
        assert item["neural_median_seconds"] > 0.0
        assert item["speedup_median"] > 0.0

    write_phase6_runtime_comparison(data, tmp_path)
    assert (tmp_path / "runtime_comparison.png").stat().st_size > 0
    write_phase6_speedup_by_size(data, tmp_path)
    assert (tmp_path / "speedup_by_size.png").stat().st_size > 0
