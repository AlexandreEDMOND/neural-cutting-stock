"""Run the mandatory Phase 9 ablations at equal refinement budget.

The trained quality checkpoint is evaluated alongside two non-learning
ablations — the deterministic greedy basis completion and a seeded random
search over the same capped action space — on one frozen quality partition.
All three agents share the exact same declared budget and flow through the
unchanged publication guardrails; the persisted report keeps each agent's
complete evaluation plus paired deltas of verified bars saved against the
learned reference. No duration enters the report: quality is the only
metric under comparison.
"""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import (
    collect_environment,
    read_quality_partition_manifest,
)
from neural_cutting_stock.learning import (
    DEFAULT_MAX_STEPS,
    GREEDY_ABLATION_IDENTIFIER,
    QUALITY_ABLATION_EVAL_SCHEMA_VERSION,
    RANDOM_SEARCH_ABLATION_IDENTIFIER,
    GreedyQualityAgent,
    NeuralQCBudget,
    RandomSearchQualityAgent,
    checkpoint_sha256,
    evaluate_quality_ablations_on_partition,
    quality_agent_from_checkpoint,
    summarize_ablation_deltas,
)

LEARNED_REFERENCE_AGENT = "learned_policy"
RANDOM_SEARCH_AGENT = "random_search"
GREEDY_AGENT = "greedy_completion"
DEFAULT_RANDOM_SEARCH_SEED = 2026


def main() -> int:
    args = _parse_args()
    manifest = read_quality_partition_manifest(args.manifest)
    learned_agent = quality_agent_from_checkpoint(args.checkpoint)
    print(
        f"evaluating {args.checkpoint} (sha256 {checkpoint_sha256(args.checkpoint)[:16]}...) "
        f"with its ablations on partition '{args.partition}' of {args.manifest}"
    )

    report = evaluate_quality_ablations_on_partition(
        manifest,
        args.partition,
        {
            LEARNED_REFERENCE_AGENT: learned_agent,
            RANDOM_SEARCH_AGENT: RandomSearchQualityAgent(args.random_search_seed),
            GREEDY_AGENT: GreedyQualityAgent(),
        },
        budget=NeuralQCBudget(args.max_steps, args.stall_patience),
        verification_tolerance=args.verification_tolerance,
    )
    report["provenance"] = {
        "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_sha256(args.checkpoint)},
        "random_search_seed": args.random_search_seed,
        "agent_keys": {
            "reference": LEARNED_REFERENCE_AGENT,
            "random_search_identifier": RANDOM_SEARCH_ABLATION_IDENTIFIER,
            "greedy_identifier": GREEDY_ABLATION_IDENTIFIER,
        },
    }
    environment = collect_environment(Path.cwd())
    report["environment"] = {
        "code_commit": environment.code_commit,
        "python_version": environment.python_version,
        "dependency_versions": environment.dependency_versions,
        "hardware_id": environment.hardware_id,
    }

    output_path = args.output_dir / f"phase-9-{args.partition}-ablations.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert report["schema_version"] == QUALITY_ABLATION_EVAL_SCHEMA_VERSION
    deltas = summarize_ablation_deltas(report, reference_agent=LEARNED_REFERENCE_AGENT)
    print(f"wrote {output_path}")
    for name in report["agent_names"]:
        counts = report["evaluations"][name]["counts"]
        overall = report["evaluations"][name]["overall"]
        mean = overall["mean_bars_saved"]
        mean_text = "n/a" if mean is None else f"{mean:.6f}"
        print(
            f"  agent {name}: instances={counts['instance_count']}, "
            f"solutions={counts['published_solution_count']}, "
            f"failures={counts['preserved_failure_count']}, "
            f"total_bars_saved={overall['total_bars_saved']}, mean_bars_saved={mean_text}"
        )
    for name, comparison in deltas["comparisons"].items():
        mean = comparison["delta_mean_bars_saved"]
        mean_text = "n/a" if mean is None else f"{mean:+.6f}"
        print(
            f"  delta {name} - {deltas['reference_agent']}: "
            f"paired={comparison['paired_instance_count']}, "
            f"excluded={len(comparison['excluded_instances'])}, "
            f"total_delta_bars_saved={comparison['delta_total_bars_saved']:+d}, "
            f"mean_delta_bars_saved={mean_text} "
            f"(reference more: {comparison['instances_where_reference_saves_more']}, "
            f"equal: {comparison['equal_instances']}, "
            f"candidate more: {comparison['instances_where_candidate_saves_more']})"
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/phase-8-partitions/manifest.json",
        help="frozen phase-8 quality partition manifest",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/phase-9-quality-policy.pt",
        help="trained quality policy checkpoint",
    )
    parser.add_argument("--partition", default="validation", help="partition to evaluate")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--stall-patience", type=int, default=1)
    parser.add_argument("--verification-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--random-seed",
        dest="random_search_seed",
        type=int,
        default=DEFAULT_RANDOM_SEARCH_SEED,
        help="seed of the random-search ablation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
