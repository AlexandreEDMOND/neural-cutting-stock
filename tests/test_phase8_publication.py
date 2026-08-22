"""Publication of the interim Phase 8 bilan from persisted, validated sources.

The rendered choice must cite only the ``family-margins-v1`` measurement and
the frozen ``phase-8-quality-partitions-v1`` manifest, must refuse any drift
between them, and the committed document must regenerate identically.
"""

import json
from pathlib import Path

import pytest

from neural_cutting_stock.benchmarks import (
    FAMILY_MARGINS_SCHEMA_VERSION,
    build_quality_partition_plan,
    phase8_family_specs,
)
from neural_cutting_stock.visualization.phase8 import (
    write_phase8_summary,
    write_quality_benchmark_choice,
)

ROOT = Path(__file__).parents[1]
MARGINS_PATH = ROOT / "results/phase-8-family-margins.json"
MANIFEST_PATH = ROOT / "data/phase-8-partitions/manifest.json"
PUBLISHED_PATH = ROOT / "docs/phase-8-quality-benchmark.md"

FAMILY = "structured-tight-divisibility-t3-v1"
NOT_RETAINED = "structured-tight-divisibility-t4-v1"


def _spec_of(label: str):
    return {spec.family_label: spec for spec in phase8_family_specs()}[label]


def _fabricated_report() -> dict:
    """A minimal but faithful family-margins-v1 payload built on real cells."""

    retained = _spec_of(FAMILY)
    rejected = _spec_of(NOT_RETAINED)
    instances = []
    for generator in retained.generators:
        instances.append(
            {
                "family_label": FAMILY,
                "seed": generator.seed,
                "instance_id": generator.instance_id,
                "gap_available": True,
                "gap_bars": 1,
            }
        )
    for generator in rejected.generators:
        instances.append(
            {
                "family_label": NOT_RETAINED,
                "seed": generator.seed,
                "instance_id": generator.instance_id,
                "gap_available": True,
                "gap_bars": 0 if generator.seed != 1 else 2,
            }
        )
    return {
        "schema_version": FAMILY_MARGINS_SCHEMA_VERSION,
        "significant_positive_share": 0.5,
        "reference_method_limits": "maximal_patterns:test-limits",
        "reduced_cost_tolerance": 1e-9,
        "integrality_tolerance": 1e-9,
        "feasibility_tolerance": 1e-9,
        "cross_check_with_enumeration": False,
        "environment": {
            "code_commit": "0123456789abcdef" * 5,
            "python_version": "3.11",
            "dependency_versions": "numpy/scipy",
        },
        "unmeasured_families": [],
        "counts": {
            "family_count": 2,
            "instance_count": 12,
            "gap_available_count": 12,
            "positive_gap_count": 7,
            "retained_family_count": 1,
        },
        "families": [
            {
                "family_label": NOT_RETAINED,
                "configuration": rejected.configuration,
                "retained": False,
                "instance_count": 6,
                "gap_available_count": 6,
                "zero_gap_count": 5,
                "positive_gap_count": 1,
                "positive_share_of_instances": 1 / 6,
                "max_gap_bars": 2,
            },
            {
                "family_label": FAMILY,
                "configuration": retained.configuration,
                "retained": True,
                "instance_count": 6,
                "gap_available_count": 6,
                "zero_gap_count": 0,
                "positive_gap_count": 6,
                "positive_share_of_instances": 1.0,
                "max_gap_bars": 1,
            },
        ],
        "instances": instances,
    }


def _write(report: dict, manifest: dict, output: Path) -> str:
    write_quality_benchmark_choice(
        report,
        manifest,
        output,
        margins_link="../results/phase-8-family-margins.json",
        partitions_link="../data/phase-8-partitions/manifest.json",
    )
    return output.read_text(encoding="utf-8")


def test_choice_reports_margins_partitions_exclusions_and_guarantees(tmp_path: Path) -> None:
    report = _fabricated_report()
    manifest = build_quality_partition_plan(report)

    text = _write(report, manifest, tmp_path / "phase-8-quality-benchmark.md")

    assert "# Choix du benchmark qualité final (Phase 8)" in text
    assert (
        "[`../results/phase-8-family-margins.json`](../results/phase-8-family-margins.json)" in text
    )
    assert "(schéma `family-margins-v1`)" in text
    assert "(schéma `phase-8-quality-partitions-v1`)" in text
    assert (
        "- Référence : méthode `milp_on_enumerated_patterns`,"
        " limites `maximal_patterns:test-limits`." in text
    )
    assert "- Contrôle croisé d'énumération : **désactivé**." in text
    assert "**50 %** des instances" in text
    assert "| `structured-tight-divisibility-t3-v1` | 6 | 6 | 0 | 6 | 100 % | 1 | 6 | oui |" in text
    assert "| `structured-tight-divisibility-t4-v1` | 6 | 6 | 5 | 1 | 17 % | 2 | 2 | non |" in text
    assert f"`{manifest['plan_id'][:12]}…`" in text
    assert "| train | 1–3 | 3 | 1 |" in text
    assert "| validation | 4 | 1 | 1 |" in text
    assert "| test | 5, 6 | 2 | 1 |" in text
    assert "| `structured-tight-divisibility-t4-v1` | 17 % | 1/6 | 2 |" in text
    assert "`optimal_over_generated_columns_only`" in text
    assert (
        "de 1 à 1 barres par instance retenue, soit 6 barres au total sur les 6 instances du plan"
        in text
    )
    assert "uv run python scripts/report_phase8_quality_benchmark.py" in text


def test_choice_refuses_drift_between_measurement_and_frozen_partitions(tmp_path: Path) -> None:
    report = _fabricated_report()
    manifest = build_quality_partition_plan(report)

    with pytest.raises(ValueError, match="unsupported margin report"):
        _write({**report, "schema_version": "family-margins-v0"}, manifest, tmp_path / "out.md")
    with pytest.raises(ValueError, match="unsupported quality partition manifest"):
        _write(
            report,
            {**manifest, "schema_version": "quality-partitions-v0"},
            tmp_path / "out.md",
        )
    with pytest.raises(ValueError, match="must cite the persisted margin measurement"):
        _write(
            report,
            {**manifest, "source": {**manifest["source"], "significant_positive_share": 0.9}},
            tmp_path / "out.md",
        )

    nothing_retained = {
        **report,
        "families": [{**family, "retained": False} for family in report["families"]],
    }
    with pytest.raises(ValueError, match="retains no family"):
        _write(nothing_retained, manifest, tmp_path / "out.md")

    extra_retained = [
        {**family, "retained": True} if family["family_label"] == NOT_RETAINED else family
        for family in report["families"]
    ]
    with pytest.raises(ValueError, match="do not cover exactly the retained families"):
        _write({**report, "families": extra_retained}, manifest, tmp_path / "out.md")


def test_published_choice_is_regenerated_from_published_sources(tmp_path: Path) -> None:
    margins_report = json.loads(MARGINS_PATH.read_text(encoding="utf-8"))
    partitions_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "phase-8-quality-benchmark.md"

    write_quality_benchmark_choice(
        margins_report,
        partitions_manifest,
        output,
        margins_link="../results/phase-8-family-margins.json",
        partitions_link="../data/phase-8-partitions/manifest.json",
    )
    published = PUBLISHED_PATH.read_text(encoding="utf-8")

    assert output.read_text(encoding="utf-8") == published
    labels = sorted(family["family_label"] for family in partitions_manifest["families"])
    assert labels == [
        "scaled-tight-divisibility-t12-v1",
        "structured-tight-divisibility-t3-v1",
        "structured-tight-divisibility-t4-v1",
    ]
    counts = margins_report["counts"]
    assert counts["retained_family_count"] == 3
    assert counts["positive_gap_count"] == 22
    assert counts["instance_count"] == 36
    assert partitions_manifest["statistics"]["instance_count"] == 18


def test_summary_reports_coverage_margins_benchmark_and_commands(tmp_path: Path) -> None:
    report = _fabricated_report()
    manifest = build_quality_partition_plan(report)

    write_phase8_summary(
        report,
        manifest,
        tmp_path / "phase-8-summary.md",
        margins_link="phase-8-family-margins.json",
        partitions_link="../data/phase-8-partitions/manifest.json",
    )
    text = (tmp_path / "phase-8-summary.md").read_text(encoding="utf-8")

    assert "# Bilan de la Phase 8" in text
    assert (
        "[`phase-8-family-margins.json`](phase-8-family-margins.json) (schéma `family-margins-v1`)"
        in text
    )
    assert "(schéma `phase-8-quality-partitions-v1`)" in text
    tol_line = "- Tolérances : coût réduit **1e-09**, intégralité **1e-09**, faisabilité **1e-09**."
    assert tol_line in text
    assert "- Contrôle croisé d'énumération : **désactivé**." in text
    assert "**50 %** des instances" in text
    assert (
        "- Familles déclarées et mesurées : **2**, soit **12** instances avec baseline classique "
        "et référence exacte." in text
    )
    assert "Variantes couvertes : " in text
    assert "- Écarts disponibles : **12**, dont **7** positifs ;" in text
    assert "| `structured-tight-divisibility-t3-v1` | 6 | 6 | 0 | 6 | 100 % | 1 | 6 | oui |" in text
    assert "| `structured-tight-divisibility-t4-v1` | 6 | 6 | 5 | 1 | 17 % | 2 | 2 | non |" in text
    counts_line = (
        "1 famille(s) retenue(s) sur 2 ; part positive globale : 7 instances positives sur 12"
    )
    assert counts_line in text
    assert f"`{manifest['plan_id'][:12]}…`" in text
    assert "| train | 1–3 | 3 | 1 |" in text
    assert "| validation | 4 | 1 | 1 |" in text
    assert "| test | 5, 6 | 2 | 1 |" in text
    assert "s'étend de 1 à 1 barres par instance, soit 6 barres gagnables au total" in text
    assert "`optimal_over_generated_columns_only`" in text
    assert "uv run python scripts/report_phase8_summary.py" in text


def test_summary_refuses_drift_and_publication_without_available_gap(tmp_path: Path) -> None:
    report = _fabricated_report()
    manifest = build_quality_partition_plan(report)
    output = tmp_path / "phase-8-summary.md"

    with pytest.raises(ValueError, match="unsupported margin report"):
        write_phase8_summary(
            {**report, "schema_version": "family-margins-v0"},
            manifest,
            output,
            margins_link="m.json",
            partitions_link="../p.json",
        )
    with pytest.raises(ValueError, match="do not cover exactly the retained families"):
        extra_retained = [
            {**family, "retained": True} if family["family_label"] == NOT_RETAINED else family
            for family in report["families"]
        ]
        write_phase8_summary(
            {**report, "families": extra_retained},
            manifest,
            output,
            margins_link="m.json",
            partitions_link="../p.json",
        )
    without_gaps = {
        **report,
        "counts": {**report["counts"], "gap_available_count": 0},
        "instances": [
            {
                **entry,
                "gap_available": False,
                "gap_unavailable_reason": "reference_failed",
                "gap_bars": None,
            }
            for entry in report["instances"]
        ],
    }
    with pytest.raises(ValueError, match="no gap is available"):
        write_phase8_summary(
            without_gaps,
            manifest,
            output,
            margins_link="m.json",
            partitions_link="../p.json",
        )


def test_published_summary_is_regenerated_from_published_sources(tmp_path: Path) -> None:
    margins_report = json.loads(MARGINS_PATH.read_text(encoding="utf-8"))
    partitions_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "phase-8-summary.md"

    write_phase8_summary(
        margins_report,
        partitions_manifest,
        output,
        margins_link="phase-8-family-margins.json",
        partitions_link="../data/phase-8-partitions/manifest.json",
    )

    published = (ROOT / "results/phase-8-summary.md").read_text(encoding="utf-8")
    assert output.read_text(encoding="utf-8") == published
