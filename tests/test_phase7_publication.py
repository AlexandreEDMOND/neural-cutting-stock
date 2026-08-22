import json
from pathlib import Path

import pytest

from neural_cutting_stock.visualization.phase7 import write_phase7_summary

ROOT = Path(__file__).parents[1]


def _row(
    instance_id: str,
    *,
    size_class: str | None = "SMALL",
    family_id: str | None = None,
    types: int = 2,
    gaps: list[int] | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    """One flat row shaped like a persisted exact-gap-v1 instance entry."""

    available = gaps is not None
    return {
        "instance_id": instance_id,
        "source": "tests",
        "size_class": size_class,
        "family_id": family_id,
        "number_of_piece_types": types,
        "reference_method": "milp_on_enumerated_patterns",
        "reference_method_limits": "maximal_patterns:test-limits",
        "reference_status": "optimal" if available else "failed",
        "verification_passed": available,
        "integer_optimum_bars": 3 if available else None,
        "baseline_objective_bars_median": (sum(gaps) / len(gaps) + 3) if available else None,
        "gap_available": available,
        "gap_unavailable_reason": None if available else reason,
        "gap_bars_per_repetition": list(gaps) if available else None,
        "gap_bars_median": (sum(gaps) / len(gaps)) if available else None,
        "zero_gap": all(gap == 0 for gap in gaps) if available else None,
    }


def _report(rows: list[dict[str, object]], excluded: list[dict[str, str]]) -> dict[str, object]:
    return {
        "schema_version": "exact-gap-v1",
        "integrality_tolerance": 1e-9,
        "feasibility_tolerance": 1e-9,
        "cross_check_with_enumeration": False,
        "environment": {
            "code_commit": "0123456789abcdef" * 5,
            "python_version": "3.11",
            "dependency_versions": "numpy/scipy",
            "hardware_id": "test-machine",
        },
        "counts": {
            "instance_count": len(rows),
            "excluded_instance_count": len(excluded),
            "optimal_reference_count": sum(row["reference_status"] == "optimal" for row in rows),
            "lower_bound_only_reference_count": 0,
            "failed_reference_count": sum(row["reference_status"] == "failed" for row in rows),
            "verification_failure_count": 0,
            "gap_available_count": sum(bool(row["gap_available"]) for row in rows),
            "zero_gap_count": sum(bool(row["zero_gap"]) for row in rows),
            "positive_gap_count": sum(
                bool(row["gap_available"]) and bool(row["gap_bars_median"] > 0) for row in rows
            ),
        },
        "instances": rows,
        "excluded": excluded,
    }


def test_summary_reports_measured_margins_guarantees_and_commands(tmp_path: Path) -> None:
    report = _report(
        [
            _row("a-zero", size_class="SMALL", family_id="fam-1", gaps=[0]),
            _row("b-positive", size_class="LARGE", family_id="fam-2", types=6, gaps=[1]),
        ],
        excluded=[
            {"instance_id": "x-1", "source": "profile.json", "reason": "no baseline run"},
            {"instance_id": "x-2", "source": "profile.json", "reason": "no baseline run"},
        ],
    )
    output = tmp_path / "phase-7-summary.md"

    write_phase7_summary(report, output, source_path="results/exact-gap.json")

    text = output.read_text(encoding="utf-8")
    assert "# Bilan de la Phase 7" in text
    assert "[`exact-gap.json`](exact-gap.json) (schéma `exact-gap-v1`)" in text
    assert (
        "- Référence : méthode `milp_on_enumerated_patterns`,"
        " limites `maximal_patterns:test-limits`." in text
    )
    assert "- Tolérances : intégralité **1e-09**, faisabilité **1e-09**." in text
    assert "- Contrôle croisé d'énumération : **désactivé**." in text
    assert (
        "- Instances avec référence exacte : **2** ; instances exclues sans référence rattachable"
        " : **2**" in text
    )
    assert "- Écarts disponibles : **2**, dont **1** nuls et **1** positifs." in text
    assert "- Exclusion : « no baseline run » — 2 instance(s)." in text
    assert "| SMALL | 1 | 2×1 | 1 | 1 | 0 | 0 |" in text
    assert "| LARGE | 1 | 6×1 | 1 | 0 | 1 | 1 |" in text
    assert "| `b-positive…` | LARGE | `fam-2…` | 6 | 3 | 4 | 1 | 1 |" in text
    assert "`docs/phase-7-gap-levers.md`](../docs/phase-7-gap-levers.md)" in text
    assert "`optimal_over_generated_columns_only`" in text
    assert "uv run python scripts/report_phase7_summary.py" in text


def test_summary_states_explicitly_when_no_instance_loses_a_bar(tmp_path: Path) -> None:
    report = _report([_row("a-zero", gaps=[0])], excluded=[])
    output = tmp_path / "phase-7-summary.md"

    write_phase7_summary(report, output, source_path="results/exact-gap.json")

    text = output.read_text(encoding="utf-8")
    assert "Aucune instance mesurée ne perd de barre" in text


def test_summary_refuses_to_publish_without_any_available_gap(tmp_path: Path) -> None:
    report = _report(
        [_row("a-failed", gaps=None, reason="reference_verification_failed")],
        excluded=[],
    )

    with pytest.raises(ValueError, match="no gap is available"):
        write_phase7_summary(
            report, tmp_path / "phase-7-summary.md", source_path="results/exact-gap.json"
        )


def test_published_summary_is_regenerated_from_published_sources(tmp_path: Path) -> None:
    report = json.loads((ROOT / "results/exact-gap.json").read_text(encoding="utf-8"))
    output = tmp_path / "phase-7-summary.md"

    write_phase7_summary(report, output, source_path="results/exact-gap.json")

    published = (ROOT / "results/phase-7-summary.md").read_text(encoding="utf-8")
    assert output.read_text(encoding="utf-8") == published
    counts = report["counts"]
    assert counts["gap_available_count"] == 16
    assert counts["zero_gap_count"] == 15
    assert counts["positive_gap_count"] == 1
