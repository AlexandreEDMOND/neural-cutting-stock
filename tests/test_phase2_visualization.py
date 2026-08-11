import json

import pytest

from neural_cutting_stock.visualization.phase2 import load_phase2_profile, phase2_report_data


def test_phase2_profile_rejects_duplicate_run_ids(tmp_path) -> None:
    profile = {
        "profile_schema_version": "baseline-profile-v1",
        "size_class_schema_version": "size-class-v1",
        "runs": [
            {"run_id": "same", "solver_mode": "classical"},
            {"run_id": "same", "solver_mode": "classical"},
        ],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="unique runs"):
        load_phase2_profile(path)


def test_phase2_report_data_aggregates_measured_runtime() -> None:
    profile = {
        "runs": [
            {
                "run_status": "optimal_lp_restricted_ip",
                "size_class": "SMALL",
                "total_runtime_seconds": 0.01,
                "number_of_cg_iterations": 1,
            },
            {
                "run_status": "optimal_lp_restricted_ip",
                "size_class": "SMALL",
                "total_runtime_seconds": 0.03,
                "number_of_cg_iterations": 3,
            },
            {"run_status": "solver_error", "size_class": None},
        ]
    }

    data = phase2_report_data(profile)

    assert data["size_data"]["SMALL"] == {
        "count": 2,
        "runtime_min_seconds": 0.01,
        "runtime_median_seconds": 0.02,
        "runtime_max_seconds": 0.03,
        "iterations_median": 2.0,
    }
