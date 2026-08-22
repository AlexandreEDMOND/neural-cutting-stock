import pytest

from neural_cutting_stock.benchmarks import (
    EXACT_GAP_BREAKDOWN_SCHEMA_VERSION,
    BenchmarkRunRecord,
    CorpusBaseline,
    EnvironmentMetadata,
    RunStatus,
    SolverMode,
    build_exact_gap_breakdown,
    build_exact_gap_report,
)
from neural_cutting_stock.benchmarks.stats import median
from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.visualization.phase7 import write_exact_gap_breakdown_markdown

_ENVIRONMENT = EnvironmentMetadata("commit-breakdown", "3.11", "numpy/scipy", "test-machine")


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
        "reference_status": "optimal" if available else "failed",
        "verification_passed": available,
        "integer_optimum_bars": 3 if available else None,
        "baseline_objective_bars_median": median(gaps) + 3 if available else None,
        "gap_available": available,
        "gap_unavailable_reason": None if available else reason,
        "gap_bars_per_repetition": list(gaps) if available else None,
        "gap_bars_median": median(gaps) if available else None,
        "zero_gap": all(gap == 0 for gap in gaps) if available else None,
    }


def _report(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "exact-gap-v1",
        "counts": {
            "instance_count": len(rows),
            "excluded_instance_count": 0,
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
        "excluded": [],
    }


def test_breakdown_groups_margins_by_size_class_and_family() -> None:
    breakdown = build_exact_gap_breakdown(
        _report(
            [
                _row("a-zero", size_class="SMALL", family_id="fam-1", gaps=[0]),
                _row("b-positive", size_class="SMALL", family_id="fam-1", gaps=[1]),
                _row("c-zero", size_class="MEDIUM", family_id="fam-1", types=4, gaps=[0]),
                _row("d-zero", size_class="XL", family_id="fam-2", types=8, gaps=[0]),
            ]
        )
    )

    assert breakdown["schema_version"] == EXACT_GAP_BREAKDOWN_SCHEMA_VERSION
    assert breakdown["totals"]["instance_count"] == 4
    assert breakdown["totals"]["positive_gap_count"] == 1
    sizes = {group["key"]: group for group in breakdown["by_size_class"]}
    assert [group["key"] for group in breakdown["by_size_class"]] == [
        "SMALL",
        "MEDIUM",
        "XL",
    ]
    assert sizes["SMALL"]["instance_count"] == 2
    assert sizes["SMALL"]["gap_available_count"] == 2
    assert sizes["SMALL"]["zero_gap_count"] == 1
    assert sizes["SMALL"]["positive_gap_count"] == 1
    assert sizes["SMALL"]["max_gap_bars_median"] == 1.0
    assert sizes["SMALL"]["piece_type_counts"] == {2: 2}
    families = {group["key"]: group for group in breakdown["by_family"]}
    assert families["fam-1"]["instance_count"] == 3
    assert families["fam-1"]["zero_gap_count"] == 2
    assert families["fam-1"]["piece_type_counts"] == {2: 2, 4: 1}
    assert families["fam-2"]["positive_gap_count"] == 0


def test_positive_margin_instances_are_listed_with_their_repetitions() -> None:
    breakdown = build_exact_gap_breakdown(
        _report([_row("b-positive", family_id="fam-1", gaps=[1, 2])])
    )

    group = breakdown["by_size_class"][0]
    assert group["positive_instances"] == [
        {
            "instance_id": "b-positive",
            "size_class": "SMALL",
            "family_id": "fam-1",
            "number_of_piece_types": 2,
            "integer_optimum_bars": 3,
            "baseline_objective_bars_median": 4.5,
            "gap_bars_median": 1.5,
            "gap_bars_per_repetition": [1, 2],
        }
    ]


def test_unavailable_gaps_stay_visible_without_contributing() -> None:
    breakdown = build_exact_gap_breakdown(
        _report(
            [
                _row(
                    "a-failed",
                    size_class="LARGE",
                    gaps=None,
                    reason="reference_verification_failed",
                ),
                _row("b-zero", size_class="LARGE", family_id="fam-9", gaps=[0]),
            ]
        )
    )

    failed_group = breakdown["by_size_class"][0]
    assert failed_group["key"] == "LARGE"
    assert failed_group["instance_count"] == 2
    assert failed_group["gap_available_count"] == 1
    assert failed_group["zero_gap_count"] == 1
    assert failed_group["positive_gap_count"] == 0
    assert failed_group["gap_unavailable_reasons"] == {"reference_verification_failed": 1}
    assert failed_group["max_gap_bars_median"] == 0.0


def test_breakdown_requires_validated_exact_gap_report() -> None:
    with pytest.raises(ValueError, match="exact-gap-v1"):
        build_exact_gap_breakdown({"schema_version": "other-v1", "instances": []})
    with pytest.raises(ValueError, match="instance sequence"):
        build_exact_gap_breakdown({"schema_version": "exact-gap-v1", "instances": "nope"})


def test_breakdown_is_deterministic() -> None:
    report = _report(
        [
            _row("a-zero", size_class="SMALL", family_id="fam-1", gaps=[0]),
            _row("b-positive", size_class="MEDIUM", family_id="fam-2", gaps=[1]),
        ]
    )

    assert build_exact_gap_breakdown(report) == build_exact_gap_breakdown(report)


def _baseline_instance() -> CuttingStockInstance:
    return CuttingStockInstance(10.0, 0.0, (3.0, 4.0), (4, 4))


def _record(run_id: str, bars: int | None) -> BenchmarkRunRecord:
    values: dict[str, object] = {
        "run_id": run_id,
        "instance_id": "breakdown-instance",
        "solver_mode": SolverMode.CLASSICAL,
        "solver_version": "classical-cg-v1",
        "seed": 11,
        "config_id": "config-breakdown",
        "repetition": 0,
        "environment": _ENVIRONMENT,
        "stock_length": 10.0,
        "kerf": 0.0,
        "number_of_piece_types": 2,
        "total_demand": 8,
        "requested_length": 28.0,
        "length_distribution": "uniform_integer_v1",
        "demand_distribution": "uniform_integer_v1",
        "run_status": RunStatus.OPTIMAL_LP_RESTRICTED_IP,
        "master_status": "optimal",
        "pricing_status": "optimal",
        "integer_master_status": "optimal",
        "termination_reason": "no_improving_column",
        "objective_value": float(bars) if bars is not None else None,
        "number_of_stock_bars": bars,
        "plan_feasible": bars is not None,
    }
    return BenchmarkRunRecord(**values)


def test_breakdown_aggregates_a_real_exact_gap_report() -> None:
    corpus = [
        CorpusBaseline(
            instance_id="breakdown-instance",
            instance=_baseline_instance(),
            source="tests",
            size_class="MEDIUM",
            family_id="fam-real",
            classical_records=(_record("run-0", 3),),
        )
    ]

    report = build_exact_gap_report(corpus, environment=_ENVIRONMENT)
    breakdown = build_exact_gap_breakdown(report)

    assert breakdown["totals"]["gap_available_count"] == 1
    assert breakdown["by_size_class"][0]["key"] == "MEDIUM"
    assert breakdown["by_size_class"][0]["zero_gap_count"] == 1
    assert breakdown["by_family"][0]["key"] == "fam-real"


def test_markdown_writer_publishes_margin_tables(tmp_path) -> None:
    breakdown = build_exact_gap_breakdown(
        _report(
            [
                _row("a-zero", size_class="SMALL", family_id="fam-1", gaps=[0]),
                _row("b-positive", size_class="MEDIUM", family_id="fam-2", gaps=[1, 2]),
                _row(
                    "c-failed",
                    size_class="XL",
                    family_id="fam-2",
                    types=8,
                    gaps=None,
                    reason="reference_not_failed",
                ),
            ]
        )
    )
    output = tmp_path / "exact-gap-breakdown.md"

    write_exact_gap_breakdown_markdown(breakdown, output, source_path=tmp_path / "exact-gap.json")

    text = output.read_text(encoding="utf-8")
    assert "| Classe | Instances | Types |" in text
    assert "| SMALL | 1 | 2×1 | 1 | 1 | 0 | 0 |" in text
    assert "| MEDIUM | 1 | 2×1 | 1 | 0 | 1 | 1.5 |" in text
    assert "| XL | 1 | 8×1 | 0 | 0 | 0 | n/a |" in text
    assert "`b-posi" in text
    assert "| 3 | 4.5 | 1.5 | 1;2 |" in text
    assert "- XL (by_size_class) : `reference_not_failed` — 1 instance(s)." in text
    assert "- fam-2 (by_family) : `reference_not_failed` — 1 instance(s)." in text
    assert "`optimal_over_generated_columns_only`" in text


def test_markdown_without_any_positive_margin_states_it_explicitly(tmp_path) -> None:
    breakdown = build_exact_gap_breakdown(_report([_row("a-zero", gaps=[0])]))
    output = tmp_path / "exact-gap-breakdown.md"

    write_exact_gap_breakdown_markdown(breakdown, output, source_path=tmp_path / "exact-gap.json")

    text = output.read_text(encoding="utf-8")
    assert "Aucune instance mesurée ne perd de barre" in text
