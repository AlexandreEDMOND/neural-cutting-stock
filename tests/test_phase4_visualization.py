from pathlib import Path

from neural_cutting_stock.benchmarks import (
    EnvironmentMetadata,
    PairedBenchmarkConfig,
    PairedBenchmarkRunner,
    SyntheticInstanceGenerator,
)
from neural_cutting_stock.learning import LearnedColumnSelectionPolicy, LinearColumnScoringModel
from neural_cutting_stock.visualization.phase4 import (
    load_phase4_runs,
    phase4_report_data,
    write_phase4_figures,
)


def test_phase4_publication_recomputes_and_writes_figures(tmp_path: Path) -> None:
    config = PairedBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=12, number_of_types=2),),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
        policy=LearnedColumnSelectionPolicy(LinearColumnScoringModel((0.0,) * 32, 0.0), 1),
        model_id="model-v1",
    )
    raw_path = tmp_path / "runs.csv"
    PairedBenchmarkRunner(config).run(raw_path)

    data = phase4_report_data(load_phase4_runs(raw_path))
    write_phase4_figures(data, tmp_path)

    assert data["quality_pair_count"] == 1
    assert (tmp_path / "runtime_comparison.png").stat().st_size > 0
    assert (tmp_path / "speedup_by_size.png").stat().st_size > 0
