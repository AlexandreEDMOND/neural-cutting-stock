from pathlib import Path

from neural_cutting_stock.visualization.phase5 import (
    load_phase5_runs,
    phase5_report_data,
    write_phase5_figures,
    write_phase5_summary,
)


def test_phase5_publication_reports_measured_non_freeze(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "results" / "phase-4-benchmark-runs.csv"
    data = phase5_report_data(load_phase5_runs(source), "linear-scorer-v1-zero-weight")

    assert not data["decision"].frozen
    assert data["decision"].reason == "no_total_runtime_improvement"
    assert data["quality_pair_count"] == 4

    write_phase5_figures(data, tmp_path)
    write_phase5_summary(data, tmp_path / "phase-5-summary.md", str(source))

    assert (tmp_path / "phase5_runtime_comparison.png").stat().st_size > 0
    assert (tmp_path / "phase5_speedup_by_size.png").stat().st_size > 0
    assert "n'est pas gelé" in (tmp_path / "phase-5-summary.md").read_text(encoding="utf-8")
