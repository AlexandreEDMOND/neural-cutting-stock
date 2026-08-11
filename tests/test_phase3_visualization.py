from pathlib import Path

from neural_cutting_stock.visualization.phase3 import (
    load_phase3_corpus,
    phase3_report_data,
)


def test_phase3_corpus_report_data_matches_persisted_manifest() -> None:
    manifest, trajectories = load_phase3_corpus(Path("data/phase-3-corpus/manifest.json"))

    data = phase3_report_data(manifest, trajectories)

    assert data["by_partition"] == {
        "train": {"trajectory_count": 1, "iteration_count": 1, "piece_type_count": 2},
        "validation": {"trajectory_count": 1, "iteration_count": 1, "piece_type_count": 3},
        "test": {"trajectory_count": 1, "iteration_count": 1, "piece_type_count": 4},
    }
    assert data["status_counts"] == {"converged": 3}
    assert data["total_demands"] == [16, 16, 13]
