"""Classical-vs-reference margin measurement across the new Phase 8 families.

Every Phase 8 family is measured by real executions on deterministic
instances: one classical column generation run per instance and one freshly
computed, independently verified MILP exact reference. The gap is
``classical bars minus certified integer optimum``; instances whose baseline
did not converge or whose reference is not optimal keep their explicit
diagnosis with no gap instead of being filtered silently.

A family is retained only when every sampled instance carries an available,
verified gap and at least ``SIGNIFICANT_POSITIVE_SHARE`` of its instances
lose at least one bar against their certified integer optimum. Nothing
time-dependent enters the measurement, so two runs over the same inputs are
identical.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Any

from neural_cutting_stock.solver import ColumnGeneration, MaximalPatternLimits

from ._validation import require_text as _require_text
from .exact_reference import ExactReferenceStatus, solve_milp_exact_reference
from .exact_reference_verification import verify_milp_exact_reference
from .generator import (
    AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    TIGHT_RATIO_LENGTH_DISTRIBUTION,
    SyntheticInstanceGenerator,
)
from .schema import EnvironmentMetadata

FAMILY_MARGINS_SCHEMA_VERSION = "family-margins-v1"

SIGNIFICANT_POSITIVE_SHARE = 0.5

_GENERATOR_CONFIGURATION_FIELDS = (
    "stock_length",
    "kerf",
    "number_of_types",
    "piece_length_range",
    "demand_range",
    "length_distribution",
    "demand_distribution",
)


@dataclass(frozen=True, slots=True)
class FamilyMarginSpec:
    """One Phase 8 family: a label and its deterministic instance cells.

    Every cell must share the same generator configuration except the seed,
    so the family identity is carried entirely by the configuration and the
    cells differ only through their reproducible seeds.
    """

    family_label: str
    generators: tuple[SyntheticInstanceGenerator, ...]

    def __post_init__(self) -> None:
        _require_text("family_label", self.family_label)
        if not self.generators:
            raise ValueError("a family margin spec requires at least one generator")
        configuration = self._configuration(self.generators[0])
        for generator in self.generators:
            if not isinstance(generator, SyntheticInstanceGenerator):
                raise ValueError("family margin cells must be synthetic generators")
            if self._configuration(generator) != configuration:
                raise ValueError(
                    "family margin cells must share one configuration modulo the seed"
                )
        seeds = [generator.seed for generator in self.generators]
        if len(set(seeds)) != len(seeds):
            raise ValueError("family margin cells must use distinct seeds")
        instance_ids = [generator.instance_id for generator in self.generators]
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("family margin cells must produce distinct instances")

    @property
    def configuration(self) -> dict[str, Any]:
        """Return the shared generator configuration without any seed."""

        return self._configuration(self.generators[0])

    @staticmethod
    def _configuration(generator: SyntheticInstanceGenerator) -> dict[str, Any]:
        return {name: getattr(generator, name) for name in _GENERATOR_CONFIGURATION_FIELDS}


def phase8_family_specs() -> tuple[FamilyMarginSpec, ...]:
    """Return the canonical deterministic families introduced by Phase 8.

    The kerf-exercised families isolate the P8.01 lever (strictly positive
    kerf under otherwise uniform sampling), the structured families combine
    the P8.03 levers (tight multiplicity-two ratios and awkward divisibility),
    and the scaled family pushes the structured levers to twelve piece types
    with higher demands while staying inside the default enumeration guards.
    Every spec keeps exactly one generator configuration so its label matches
    a single deterministic ``family_id``. The multi-format variant declared in
    P8.02 stays outside this list until the solver and the exact reference
    accept it.
    """

    return (
        FamilyMarginSpec(
            "kerf-exercised-uniform-t4-v1",
            tuple(
                SyntheticInstanceGenerator(
                    seed=seed,
                    number_of_types=4,
                    demand_range=(5, 30),
                    kerf=2.0,
                )
                for seed in range(1, 7)
            ),
        ),
        FamilyMarginSpec(
            "kerf-exercised-uniform-t6-v1",
            tuple(
                SyntheticInstanceGenerator(
                    seed=seed,
                    number_of_types=6,
                    demand_range=(5, 30),
                    kerf=2.0,
                )
                for seed in range(1, 7)
            ),
        ),
        FamilyMarginSpec(
            "structured-tight-divisibility-t3-v1",
            tuple(
                SyntheticInstanceGenerator(
                    seed=seed,
                    number_of_types=3,
                    demand_range=(5, 30),
                    length_distribution=TIGHT_RATIO_LENGTH_DISTRIBUTION,
                    demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
                )
                for seed in range(1, 7)
            ),
        ),
        FamilyMarginSpec(
            "structured-tight-divisibility-t4-v1",
            tuple(
                SyntheticInstanceGenerator(
                    seed=seed,
                    number_of_types=4,
                    demand_range=(5, 30),
                    length_distribution=TIGHT_RATIO_LENGTH_DISTRIBUTION,
                    demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
                )
                for seed in range(1, 7)
            ),
        ),
        FamilyMarginSpec(
            "scaled-tight-divisibility-t12-v1",
            tuple(
                SyntheticInstanceGenerator(
                    seed=seed,
                    number_of_types=12,
                    demand_range=(20, 100),
                    length_distribution=TIGHT_RATIO_LENGTH_DISTRIBUTION,
                    demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
                )
                for seed in range(1, 7)
            ),
        ),
    )


def measure_family_margins(
    specs: tuple[FamilyMarginSpec, ...],
    *,
    environment: EnvironmentMetadata,
    reduced_cost_tolerance: float = 1e-9,
    integrality_tolerance: float = 1e-9,
    feasibility_tolerance: float = 1e-9,
    limits: MaximalPatternLimits | None = None,
    cross_check_with_enumeration: bool = False,
    unmeasured_families: tuple[dict[str, str], ...] = (),
) -> dict[str, Any]:
    """Measure every family and apply the documented retention rule.

    Each instance is solved once classically and once exactly; failures stay
    visible as unavailable gaps carrying their reason. The report is pure
    with respect to its inputs: no duration enters it.
    """

    if not isinstance(environment, EnvironmentMetadata):
        raise ValueError("environment must be EnvironmentMetadata")
    for name, value in (
        ("reduced_cost_tolerance", reduced_cost_tolerance),
        ("integrality_tolerance", integrality_tolerance),
        ("feasibility_tolerance", feasibility_tolerance),
    ):
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    labels = [spec.family_label for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("family labels must be unique")
    validated_unmeasured = tuple(_validated_unmeasured(item) for item in unmeasured_families)

    entries = [
        _entry(
            spec,
            generator,
            environment=environment,
            reduced_cost_tolerance=reduced_cost_tolerance,
            integrality_tolerance=integrality_tolerance,
            feasibility_tolerance=feasibility_tolerance,
            limits=limits,
            cross_check_with_enumeration=cross_check_with_enumeration,
        )
        for spec in sorted(specs, key=lambda spec: spec.family_label)
        for generator in sorted(spec.generators, key=lambda generator: generator.seed)
    ]
    summaries = [
        summarize_family_margin_entries(spec.family_label, _entries_of(entries, spec))
        for spec in sorted(specs, key=lambda spec: spec.family_label)
    ]
    retained_count = sum(family["retained"] for family in summaries)
    effective_limits = limits if limits is not None else MaximalPatternLimits()
    return {
        "schema_version": FAMILY_MARGINS_SCHEMA_VERSION,
        "significant_positive_share": SIGNIFICANT_POSITIVE_SHARE,
        "reduced_cost_tolerance": reduced_cost_tolerance,
        "integrality_tolerance": integrality_tolerance,
        "feasibility_tolerance": feasibility_tolerance,
        "cross_check_with_enumeration": cross_check_with_enumeration,
        "reference_method_limits": (
            f"maximal_patterns:max_search_space_size="
            f"{effective_limits.max_search_space_size},max_patterns="
            f"{effective_limits.max_patterns}"
        ),
        "environment": {
            "code_commit": environment.code_commit,
            "python_version": environment.python_version,
            "dependency_versions": environment.dependency_versions,
            "hardware_id": environment.hardware_id,
        },
        "counts": {
            "instance_count": len(entries),
            "family_count": len(summaries),
            "gap_available_count": sum(entry["gap_available"] for entry in entries),
            "positive_gap_count": sum(entry["positive_gap"] for entry in entries),
            "retained_family_count": retained_count,
        },
        "families": summaries,
        "instances": entries,
        "unmeasured_families": sorted(
            validated_unmeasured, key=lambda item: item["family_label"]
        ),
    }


def summarize_family_margin_entries(
    family_label: str,
    entries: tuple[dict[str, Any], ...],
    *,
    significant_positive_share: float = SIGNIFICANT_POSITIVE_SHARE,
) -> dict[str, Any]:
    """Aggregate measured entries of one family into its margin summary."""

    _require_text("family_label", family_label)
    if not 0 <= significant_positive_share <= 1:
        raise ValueError("significant_positive_share must lie within [0, 1]")
    available = [entry for entry in entries if entry["gap_available"]]
    positive_gaps = [entry["gap_bars"] for entry in available if entry["gap_bars"] > 0]
    reasons: dict[str, int] = {}
    for entry in entries:
        if not entry["gap_available"]:
            reasons[entry["gap_unavailable_reason"]] = (
                reasons.get(entry["gap_unavailable_reason"], 0) + 1
            )
    all_available = bool(entries) and len(available) == len(entries)
    positive_share = len(positive_gaps) / len(entries) if entries else 0.0
    return {
        "family_label": family_label,
        "instance_count": len(entries),
        "configuration": entries[0]["configuration"] if entries else None,
        "gap_available_count": len(available),
        "zero_gap_count": sum(bool(entry["zero_gap"]) for entry in available),
        "positive_gap_count": len(positive_gaps),
        "max_gap_bars": max((entry["gap_bars"] for entry in available), default=None),
        "all_gaps_available": all_available,
        "gap_unavailable_reasons": dict(sorted(reasons.items())),
        "positive_share_of_instances": positive_share,
        "retained": bool(all_available and positive_share >= significant_positive_share),
    }


def _entry(
    spec: FamilyMarginSpec,
    generator: SyntheticInstanceGenerator,
    *,
    environment: EnvironmentMetadata,
    reduced_cost_tolerance: float,
    integrality_tolerance: float,
    feasibility_tolerance: float,
    limits: MaximalPatternLimits | None,
    cross_check_with_enumeration: bool,
) -> dict[str, Any]:
    instance_id = generator.instance_id
    instance = generator.generate()
    result = ColumnGeneration(
        instance,
        reduced_cost_tolerance,
        instance_id=instance_id,
    ).solve()

    outcome, reference = solve_milp_exact_reference(
        instance_id,
        instance,
        environment=environment,
        integrality_tolerance=integrality_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        limits=limits,
    )
    verification_errors: list[str] = []
    verification_ran = False
    lp_bound_bars = None
    if outcome is not None and reference.status is ExactReferenceStatus.OPTIMAL:
        verification = verify_milp_exact_reference(
            instance_id,
            instance,
            outcome,
            reference,
            limits=limits,
            cross_check_with_enumeration=cross_check_with_enumeration,
        )
        verification_errors = list(verification.errors)
        lp_bound_bars = verification.lp_bound_bars
        verification_ran = True

    classical_bars = None
    plan_feasible = result.verification is not None and result.verification.feasible
    objective = (
        result.integer_master_result.objective_value if result.integer_master_result else None
    )
    if objective is not None:
        rounded = round(objective)
        if abs(objective - rounded) <= integrality_tolerance:
            classical_bars = rounded

    reason = _unavailability_reason(
        result.status,
        plan_feasible,
        classical_bars,
        reference,
        verification_errors,
    )
    gap_available = reason is None
    gap_bars = classical_bars - reference.integer_optimum_bars if gap_available else None
    return {
        "family_label": spec.family_label,
        "instance_id": instance_id,
        "seed": generator.seed,
        "configuration": spec.configuration,
        "stock_length": instance.stock_length,
        "kerf": instance.kerf,
        "number_of_piece_types": instance.number_of_types,
        "total_demand": sum(instance.demands),
        "classical_status": result.status,
        "classical_plan_feasible": plan_feasible,
        "classical_bars": classical_bars,
        "lp_bound_bars": lp_bound_bars,
        "reference_status": reference.status.value,
        "reference_method_limits": reference.method_limits,
        "integer_optimum_bars": reference.integer_optimum_bars,
        "pattern_count": outcome.number_of_patterns if outcome is not None else None,
        "verification_passed": (not verification_errors) if verification_ran else None,
        "verification_errors": verification_errors,
        "reference_error_message": reference.error_message,
        "gap_available": gap_available,
        "gap_unavailable_reason": reason,
        "gap_bars": gap_bars,
        "zero_gap": gap_bars == 0 if gap_available else None,
        "positive_gap": gap_bars > 0 if gap_available else None,
    }


def _unavailability_reason(
    classical_status: str,
    plan_feasible: bool,
    classical_bars: int | None,
    reference: Any,
    verification_errors: list[str],
) -> str | None:
    if classical_status != "converged":
        return f"classical_baseline_{classical_status}"
    if not plan_feasible:
        return "classical_plan_infeasible"
    if classical_bars is None:
        return "classical_baseline_without_integral_bar_count"
    if reference.status is not ExactReferenceStatus.OPTIMAL:
        return f"reference_not_{reference.status.value}"
    if verification_errors:
        return "reference_verification_failed"
    return None


def _validated_unmeasured(item: object) -> dict[str, str]:
    if not isinstance(item, dict):
        raise ValueError("each unmeasured family must be a mapping")
    for name in ("family_label", "reason"):
        value = item.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"each unmeasured family requires a non-empty {name}")
    return {"family_label": item["family_label"], "reason": item["reason"]}


def _entries_of(
    entries: list[dict[str, Any]], spec: FamilyMarginSpec
) -> tuple[dict[str, Any], ...]:
    return tuple(entry for entry in entries if entry["family_label"] == spec.family_label)


__all__ = [
    "FAMILY_MARGINS_SCHEMA_VERSION",
    "SIGNIFICANT_POSITIVE_SHARE",
    "FamilyMarginSpec",
    "measure_family_margins",
    "phase8_family_specs",
    "summarize_family_margin_entries",
]
