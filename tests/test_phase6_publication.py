import hashlib
import json
from pathlib import Path

import pytest
from test_benchmark_comparison import _record as base_record

from neural_cutting_stock.benchmarks import SolverMode, validate_final_manifest
from neural_cutting_stock.visualization.phase4 import load_phase4_runs
from neural_cutting_stock.visualization.phase6 import (
    phase6_runtime_comparison_data,
    write_phase6_summary,
)

ROOT = Path(__file__).parents[1]

CONFIG = {
    "schema_version": "phase-6-final-freeze-v1",
    "model": {
        "artifact": "models/linear-scorer-v1-zero-weight.json",
        "model_id": "linear-scorer-v1-zero-weight",
        "policy": "bounded-column-selection-v1",
    },
    "protocol": {
        "benchmark_schema_version": "benchmark-run-v1",
        "comparison": "paired_instance_id",
        "quality_tolerance_bars": 0.0,
        "repetitions": 3,
        "warmup_runs": 0,
        "reduced_cost_tolerance": 1e-9,
        "max_runtime_seconds": None,
        "max_cg_iterations": None,
        "model_loading": "preloaded_per_process",
        "execution_order": "classical_then_neural",
        "size_class_version": "size-class-v1",
    },
}

MANIFEST = {
    "schema_version": "phase-6-instance-manifest-v1",
    "manifest_id": "manifest-id-1234",
    "statistics": {
        "instance_count": 2,
        "target_size_class_counts": {"SMALL": 1, "LARGE": 1},
    },
}


def _pair(instance: str, repetition: int, classical_seconds: float, neural_seconds: float):
    return [
        base_record(
            SolverMode.CLASSICAL,
            f"classical-{instance}-{repetition}",
            instance_id=instance,
            repetition=repetition,
            total_runtime_seconds=classical_seconds,
        ),
        base_record(
            SolverMode.NEURAL,
            f"neural-{instance}-{repetition}",
            instance_id=instance,
            repetition=repetition,
            total_runtime_seconds=neural_seconds,
        ),
    ]


def _summary_data(records):
    classical = [r for r in records if r.solver_mode is SolverMode.CLASSICAL]
    neural = [r for r in records if r.solver_mode is SolverMode.NEURAL]
    return phase6_runtime_comparison_data(
        classical, neural, {"inst-small": "SMALL", "inst-large": "LARGE"}
    )


def _write_summary(data, tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.json"
    model_path = tmp_path / "model.json"
    manifest_path.write_bytes(b"final-manifest-bytes")
    model_path.write_bytes(b"model-artifact-bytes")
    output = tmp_path / "phase-6-summary.md"
    write_phase6_summary(
        data,
        output,
        classical_source="results/phase-6-classical-runs.csv",
        neural_source="results/phase-6-neural-runs.csv",
        config_path="configs/phase-6-final.json",
        config=CONFIG,
        manifest=MANIFEST,
        final_manifest_path=manifest_path,
        model_artifact_path=model_path,
    )
    return output


def test_summary_reports_measured_counts_medians_and_artifact_hashes(tmp_path: Path) -> None:
    data = _summary_data([*_pair("inst-small", 0, 2.0, 1.0), *_pair("inst-large", 0, 3.0, 4.5)])
    output = _write_summary(data, tmp_path)
    content = output.read_text(encoding="utf-8")

    assert "# Bilan final de la Phase 6" in content
    assert "- Exécutions : **4** ; paires : **2** ; paires admissibles : **2**." in content
    assert "- Violations de qualité : **0** paire(s) à la tolérance déclarée." in content
    assert "- Statuts terminaux : `optimal_lp_restricted_ip` : 4." in content
    assert "| SMALL | 1 | 1 | 2.000000 | 1.000000 | 2.000000 |" in content
    assert "| LARGE | 1 | 1 | 3.000000 | 4.500000 | 0.666667 |" in content
    assert (
        "de **0.666667** à **2.000000** ; Neural CG est plus lent sur 1 des 2 instances" in content
    )
    assert "`manifest_id` `manifest-id-1234`" in content
    assert (
        hashlib.sha256(b"final-manifest-bytes").hexdigest() in content
        and hashlib.sha256(b"model-artifact-bytes").hexdigest() in content
    )
    assert "uv run python scripts/report_phase6_summary.py" in content


def test_summary_refuses_to_publish_without_any_admissible_pair(tmp_path: Path) -> None:
    records = _pair("inst-small", 0, 2.0, 1.0)
    records[1] = base_record(
        SolverMode.NEURAL,
        "neural-inst-small-0",
        instance_id="inst-small",
        repetition=0,
        total_runtime_seconds=1.0,
        objective_value=6.0,
    )
    data = _summary_data(records)

    with pytest.raises(ValueError, match="no admissible pair"):
        _write_summary(data, tmp_path)


def test_published_summary_is_regenerated_from_published_sources(tmp_path: Path) -> None:
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
    output = tmp_path / "phase-6-summary.md"
    write_phase6_summary(
        data,
        output,
        classical_source="results/phase-6-classical-runs.csv",
        neural_source="results/phase-6-neural-runs.csv",
        config_path="configs/phase-6-final.json",
        config=config,
        manifest=manifest,
        final_manifest_path=config["files"]["final_instance_manifest"],
        model_artifact_path=config["model"]["artifact"],
    )

    published = (ROOT / "results/phase-6-summary.md").read_text(encoding="utf-8")
    assert output.read_text(encoding="utf-8") == published
    assert data["report"]["run_count"] == 72
    assert data["report"]["admissible_pair_count"] == 36

    campaign = json.loads(
        (ROOT / "results/phase-6-neural-campaign.json").read_text(encoding="utf-8")
    )
    assert (
        hashlib.sha256((ROOT / config["files"]["final_instance_manifest"]).read_bytes()).hexdigest()
        == campaign["final_manifest_sha256"]
    )
    assert (
        hashlib.sha256((ROOT / config["model"]["artifact"]).read_bytes()).hexdigest()
        == campaign["solver"]["model_artifact_sha256"]
    )
