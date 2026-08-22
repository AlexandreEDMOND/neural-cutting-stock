"""Train the Phase 9 deep quality policy on a frozen partition of margin families.

The run is real end to end: instances are re-materialized from the validated
partition manifest, every classical starting point comes from an actual
column-generation solve, and training follows the documented REINFORCE
variant of docs/phase-9-rl-algorithm.md. The versioned checkpoint and the
complete experiment journal are persisted, and the printed summary reports
only numbers measured during this run.
"""

import argparse
import hashlib
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import (
    materialize_partition_instances,
    read_quality_partition_manifest,
)
from neural_cutting_stock.learning import (
    DEFAULT_BASELINE_MOMENTUM,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_WIDTH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_STEPS,
    save_checkpoint,
    train_quality_rl_policy,
    training_journal_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/phase-8-partitions/manifest.json",
        help="frozen phase-8 quality partition manifest",
    )
    parser.add_argument("--partition", default="train", help="partition to train on")
    parser.add_argument(
        "--checkpoint", default="models/phase-9-quality-policy.pt", help="checkpoint output path"
    )
    parser.add_argument(
        "--journal",
        default="results/phase-9-training-journal.json",
        help="experiment journal output path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden-width", type=int, default=DEFAULT_HIDDEN_WIDTH)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--baseline-momentum", type=float, default=DEFAULT_BASELINE_MOMENTUM)
    args = parser.parse_args()

    manifest = read_quality_partition_manifest(args.manifest)
    instances = materialize_partition_instances(manifest, args.partition)
    print(f"training instances: {len(instances)} from {args.manifest} ({args.partition})")

    policy = train_quality_rl_policy(
        instances,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_width=args.hidden_width,
        max_steps=args.max_steps,
        baseline_momentum=args.baseline_momentum,
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_metadata = save_checkpoint(
        checkpoint_path,
        module=policy.module,
        seed=policy.seed,
        config=dict(policy.config),
        curves=policy.curves,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    journal = training_journal_payload(
        policy,
        source={
            "partition_manifest": str(args.manifest),
            "plan_id": manifest["plan_id"],
            "partition": args.partition,
            "instance_ids": sorted(instances),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
        },
    )
    journal_path = Path(args.journal)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    totals = policy.totals
    print(f"episodes: {totals['episode_count']}  updates: {policy.curves.points[-1].step + 1}")
    print(f"steps: {totals['step_count']}  accepted: {totals['accepted_step_count']}  "
          f"invalid: {totals['invalid_step_count']}")
    print(f"bars saved over training: {totals['bars_saved_total']}")
    first = policy.curves.points[0].metrics["mean_episode_return"]
    last = policy.curves.points[-1].metrics["mean_episode_return"]
    print(f"mean episode return: {first:.3f} -> {last:.3f}")
    print(f"run id: {checkpoint_metadata['run_id']}")
    print(f"checkpoint: {checkpoint_path} (sha256 {checkpoint_sha256[:16]}...)")
    print(f"journal: {journal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
