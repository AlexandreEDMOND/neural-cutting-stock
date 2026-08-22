"""Family-margin measurement and retention rule for the new Phase 8 families.

The pure aggregation tests pin the documented retention rule: a family is
retained only when every sampled instance carries an available, verified gap
and at least ``SIGNIFICANT_POSITIVE_SHARE`` of its instances lose at least one
bar against their certified integer optimum. The end-to-end test runs real
executions — classical column generation and independently verified MILP
references — on the deterministic structured-profile cells.
"""

import pytest

from neural_cutting_stock.benchmarks import (
    AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    FAMILY_MARGINS_SCHEMA_VERSION,
    SIGNIFICANT_POSITIVE_SHARE,
    TIGHT_RATIO_LENGTH_DISTRIBUTION,
    EnvironmentMetadata,
    FamilyMarginSpec,
    SyntheticInstanceGenerator,
    measure_family_margins,
    phase8_family_specs,
    summarize_family_margin_entries,
)

ENVIRONMENT = EnvironmentMetadata("family-margins-test", "test", "test", "test")


def _entry(gap_bars: int | None, *, reason: str | None = None) -> dict[str, object]:
    return {
        "gap_available": reason is None,
        "gap_unavailable_reason": reason,
        "gap_bars": gap_bars,
        "zero_gap": gap_bars == 0 if gap_bars is not None else None,
        "configuration": {"kerf": 0.0},
    }


def test_significant_positive_share_is_declared_upfront() -> None:
    assert SIGNIFICANT_POSITIVE_SHARE == 0.5


def test_summary_counts_margins_and_keeps_unavailable_reasons_visible() -> None:
    summary = summarize_family_margin_entries(
        "structured",
        (_entry(1), _entry(2), _entry(0), _entry(None, reason="reference_not_failed")),
    )

    assert summary["family_label"] == "structured"
    assert summary["instance_count"] == 4
    assert summary["gap_available_count"] == 3
    assert summary["zero_gap_count"] == 1
    assert summary["positive_gap_count"] == 2
    assert summary["max_gap_bars"] == 2
    assert summary["all_gaps_available"] is False
    assert summary["gap_unavailable_reasons"] == {"reference_not_failed": 1}
    assert summary["positive_share_of_instances"] == 0.5


def test_retention_requires_every_gap_available_and_a_significant_share() -> None:
    complete = summarize_family_margin_entries(
        "complete", (_entry(1), _entry(2), _entry(0), _entry(3))
    )
    incomplete = summarize_family_margin_entries(
        "incomplete", (_entry(1), _entry(2), _entry(0), _entry(None, reason="reference_failed"))
    )
    thin = summarize_family_margin_entries("thin", (_entry(1), _entry(0), _entry(0)))

    assert complete["all_gaps_available"] is True
    assert complete["retained"] is True
    assert incomplete["retained"] is False
    assert incomplete["positive_share_of_instances"] == 0.5
    assert thin["positive_share_of_instances"] == pytest.approx(1 / 3)
    assert thin["retained"] is False


def test_retention_threshold_is_configurable_and_validated() -> None:
    strict = summarize_family_margin_entries(
        "strict", (_entry(1), _entry(2), _entry(0)), significant_positive_share=1.0
    )
    permissive = summarize_family_margin_entries(
        "permissive", (_entry(1), _entry(0), _entry(0)), significant_positive_share=0.0
    )

    assert strict["retained"] is False
    assert permissive["retained"] is True

    with pytest.raises(ValueError):
        summarize_family_margin_entries("x", (_entry(1),), significant_positive_share=1.5)


def test_phase8_specs_are_deterministic_homogeneous_and_disjoint() -> None:
    specs = phase8_family_specs()

    assert [spec.family_label for spec in specs] == [
        "kerf-exercised-uniform-t4-v1",
        "kerf-exercised-uniform-t6-v1",
        "structured-tight-divisibility-t3-v1",
        "structured-tight-divisibility-t4-v1",
        "scaled-tight-divisibility-t12-v1",
    ]
    instance_ids = [
        generator.instance_id for spec in specs for generator in spec.generators
    ]
    assert len(set(instance_ids)) == len(instance_ids)
    counts = {spec.family_label: len(spec.generators) for spec in specs}
    assert counts == {
        "kerf-exercised-uniform-t4-v1": 6,
        "kerf-exercised-uniform-t6-v1": 6,
        "structured-tight-divisibility-t3-v1": 6,
        "structured-tight-divisibility-t4-v1": 6,
        "scaled-tight-divisibility-t12-v1": 6,
    }
    scaled = {spec.family_label: spec for spec in specs}["scaled-tight-divisibility-t12-v1"]
    assert scaled.configuration == {
        "stock_length": 100.0,
        "kerf": 0.0,
        "number_of_types": 12,
        "piece_length_range": (10, 90),
        "demand_range": (20, 100),
        "length_distribution": TIGHT_RATIO_LENGTH_DISTRIBUTION,
        "demand_distribution": AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    }


def test_spec_rejects_cells_that_do_not_share_one_configuration() -> None:
    with pytest.raises(ValueError):
        FamilyMarginSpec(
            "mixed",
            (
                SyntheticInstanceGenerator(seed=1),
                SyntheticInstanceGenerator(seed=2, number_of_types=4),
            ),
        )
    with pytest.raises(ValueError):
        FamilyMarginSpec(
            "duplicated-seeds",
            (SyntheticInstanceGenerator(seed=1), SyntheticInstanceGenerator(seed=1)),
        )
    with pytest.raises(ValueError):
        FamilyMarginSpec("empty", ())
    with pytest.raises(ValueError):
        FamilyMarginSpec("   ", (SyntheticInstanceGenerator(seed=1),))


def test_measure_family_margins_runs_real_executions_and_applies_the_rule() -> None:
    spec = FamilyMarginSpec(
        "structured-tiny",
        tuple(
            SyntheticInstanceGenerator(
                seed=seed,
                number_of_types=3,
                demand_range=(5, 30),
                length_distribution=TIGHT_RATIO_LENGTH_DISTRIBUTION,
                demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
            )
            for seed in range(1, 4)
        ),
    )

    report = measure_family_margins(
        (spec,),
        environment=ENVIRONMENT,
        unmeasured_families=(
            {
                "family_label": "multi-stock-formats",
                "reason": "not measurable without multi-format solver support",
            },
        ),
    )

    assert report["schema_version"] == FAMILY_MARGINS_SCHEMA_VERSION
    assert report["counts"]["instance_count"] == 3
    assert report["counts"]["gap_available_count"] == 3
    assert report["unmeasured_families"] == [
        {
            "family_label": "multi-stock-formats",
            "reason": "not measurable without multi-format solver support",
        }
    ]

    family = report["families"][0]
    assert family["instance_count"] == 3
    assert family["all_gaps_available"] is True
    assert family["retained"] == (family["positive_share_of_instances"] >= 0.5)
    assert family["configuration"]["length_distribution"] == TIGHT_RATIO_LENGTH_DISTRIBUTION

    for entry in report["instances"]:
        assert entry["classical_status"] == "converged"
        assert entry["classical_plan_feasible"] is True
        assert entry["reference_status"] == "optimal"
        assert entry["verification_passed"] is True
        assert entry["integer_optimum_bars"] >= 1
        assert entry["classical_bars"] >= entry["integer_optimum_bars"]
        assert entry["lp_bound_bars"] <= entry["integer_optimum_bars"] + 1e-9
        expected_gap = entry["classical_bars"] - entry["integer_optimum_bars"]
        assert entry["gap_bars"] == expected_gap
        assert entry["zero_gap"] == (expected_gap == 0)
        assert entry["positive_gap"] == (expected_gap > 0)


def test_measure_family_margins_rejects_invalid_inputs() -> None:
    spec = FamilyMarginSpec("tiny", (SyntheticInstanceGenerator(seed=1),))

    with pytest.raises(ValueError):
        measure_family_margins((spec,), environment="not-an-environment")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        measure_family_margins((spec,), environment=ENVIRONMENT, integrality_tolerance=-1.0)
    with pytest.raises(ValueError):
        measure_family_margins(
            (spec,),
            environment=ENVIRONMENT,
            unmeasured_families=({"reason": "missing label"},),
        )


def test_markdown_publication_renders_only_measured_facts(tmp_path) -> None:
    from neural_cutting_stock.visualization.phase8 import write_family_margins_markdown

    report = {
        "schema_version": FAMILY_MARGINS_SCHEMA_VERSION,
        "significant_positive_share": SIGNIFICANT_POSITIVE_SHARE,
        "reduced_cost_tolerance": 1e-9,
        "integrality_tolerance": 1e-9,
        "feasibility_tolerance": 1e-9,
        "cross_check_with_enumeration": False,
        "reference_method_limits": "maximal_patterns:test-limits",
        "environment": {
            "code_commit": "0123456789abcdef" * 5,
            "python_version": "3.11",
            "dependency_versions": "numpy/scipy",
            "hardware_id": "test-machine",
        },
        "counts": {
            "instance_count": 2,
            "family_count": 1,
            "gap_available_count": 2,
            "positive_gap_count": 2,
            "retained_family_count": 1,
        },
        "families": [
            {
                "family_label": "structured-tiny",
                "configuration": {"kerf": 0.0},
                "instance_count": 2,
                "gap_available_count": 2,
                "zero_gap_count": 0,
                "positive_gap_count": 2,
                "max_gap_bars": 2,
                "all_gaps_available": True,
                "gap_unavailable_reasons": {},
                "positive_share_of_instances": 1.0,
                "retained": True,
            }
        ],
        "instances": [
            {
                "family_label": "structured-tiny",
                "instance_id": "a" * 64,
                "number_of_piece_types": 3,
                "classical_bars": 25,
                "integer_optimum_bars": 23,
                "gap_available": True,
                "gap_unavailable_reason": None,
                "gap_bars": 2,
            },
            {
                "family_label": "structured-tiny",
                "instance_id": "b" * 64,
                "number_of_piece_types": 3,
                "classical_bars": 24,
                "integer_optimum_bars": 23,
                "gap_available": True,
                "gap_unavailable_reason": None,
                "gap_bars": 1,
            },
        ],
        "unmeasured_families": [
            {"family_label": "multi-stock-formats", "reason": "no solver support yet"}
        ],
    }

    output = tmp_path / "phase-8-family-margins.md"
    write_family_margins_markdown(report, output, source_path="phase-8-family-margins.json")

    text = output.read_text(encoding="utf-8")
    assert "`structured-tiny` | 2 | 2 | 0 | 2 | 100 % | oui |" in text
    assert "`multi-stock-formats` — no solver support yet." in text
    assert "| `" + "a" * 12 + "…` | `structured-tiny` | 3 | 23 | 25 | 2 |" in text
    assert "| `" + "b" * 12 + "…` | `structured-tiny` | 3 | 23 | 24 | 1 |" in text
    assert "au moins **50 %** des" in text
