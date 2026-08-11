"""Command-line entry point for classical and learned column generation."""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import ColumnGeneration


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "train":
        return _train_main(argv[1:])
    parser = _build_parser()
    args = parser.parse_args(argv)
    instance = CuttingStockInstance(
        stock_length=args.stock_length,
        kerf=args.kerf,
        piece_lengths=_parse_values(args.piece_lengths, float),
        demands=_parse_values(args.demands, int),
    )
    if args.solver == "classical":
        result = ColumnGeneration(
            instance,
            args.reduced_cost_tolerance,
            max_runtime_seconds=args.max_runtime_seconds,
            max_iterations=args.max_cg_iterations,
        ).solve()
    else:
        if args.model is None:
            parser.error("--model is required when --solver is neural")
        from neural_cutting_stock.learning import (
            LearnedColumnSelectionPolicy,
            NeuralColumnGeneration,
            load_training_artifact,
        )

        model = load_training_artifact(args.model)
        policy = LearnedColumnSelectionPolicy(model, candidate_budget=args.candidate_budget)
        result = NeuralColumnGeneration(
            instance,
            policy,
            candidate_budget=args.candidate_budget,
            reduced_cost_tolerance=args.reduced_cost_tolerance,
            max_runtime_seconds=args.max_runtime_seconds,
            max_iterations=args.max_cg_iterations,
        ).solve()
    output: dict[str, object] = {
        "solver": args.solver,
        "reduced_cost_tolerance": args.reduced_cost_tolerance,
        "resource_limits": {
            "max_runtime_seconds": args.max_runtime_seconds,
            "max_cg_iterations": args.max_cg_iterations,
        },
        "status": result.status,
        "termination_reason": result.termination_reason,
        "instance": {
            "stock_length": instance.stock_length,
            "kerf": instance.kerf,
            "piece_lengths": instance.piece_lengths,
            "demands": instance.demands,
        },
        "patterns": result.patterns,
        "iterations": result.iterations,
        "columns_added": result.columns_added,
        "duplicate_columns": result.duplicate_columns,
        "integrality_gap": result.integrality_gap,
        "integer_solution_guarantee": result.integer_solution_guarantee,
    }
    if result.rmp_result is not None:
        output["rmp"] = {
            "status": result.rmp_result.status,
            "objective_value": result.rmp_result.objective_value,
            "column_values": result.rmp_result.column_values,
            "dual_values": result.rmp_result.dual_values,
            "message": result.rmp_result.message,
        }
    if result.pricing_result is not None:
        output["pricing"] = {
            "status": result.pricing_result.status,
            "pattern": result.pricing_result.pattern,
            "dual_value": result.pricing_result.dual_value,
            "reduced_cost": result.pricing_result.reduced_cost,
            "message": result.pricing_result.message,
        }
    if result.integer_master_result is not None:
        integer_result = result.integer_master_result
        output["integer_master"] = {
            "status": integer_result.status,
            "objective_value": integer_result.objective_value,
            "column_values": integer_result.column_values,
            "message": integer_result.message,
        }
        if result.verification is not None:
            output["verification"] = asdict(result.verification)
    print(json.dumps(output, default=_json_default, sort_keys=True))
    return 0


def _train_main(argv: Sequence[str]) -> int:
    from neural_cutting_stock.learning import write_training_artifact

    parser = argparse.ArgumentParser(description="Train the linear column scoring model.")
    parser.add_argument("--manifest", required=True, help="Phase 3 corpus manifest")
    parser.add_argument("--output", required=True, help="Training artifact JSON path")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", help="JSON configuration object or file path", default="{}")
    args = parser.parse_args(argv)
    config = _parse_config(args.config)
    artifact = write_training_artifact(args.manifest, args.output, args.seed, config)
    print(
        json.dumps({"output": args.output, "example_count": artifact["metadata"]["example_count"]})
    )
    return 0


def _parse_config(value: str) -> dict[str, object]:
    try:
        path = Path(value)
        raw = path.read_text(encoding="utf-8") if path.is_file() else value
        config = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise argparse.ArgumentTypeError("config must be a JSON object or JSON file") from error
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("config must be a JSON object")
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve a 1D Cutting Stock instance.")
    parser.add_argument("--solver", choices=("classical", "neural"), required=True)
    parser.add_argument("--model", help="Training artifact JSON path (required for neural mode)")
    parser.add_argument("--candidate-budget", type=_positive_int, default=None)
    parser.add_argument("--stock-length", type=float, required=True)
    parser.add_argument("--kerf", type=float, default=0.0)
    parser.add_argument("--piece-lengths", required=True, help="Comma-separated lengths")
    parser.add_argument("--demands", required=True, help="Comma-separated positive integers")
    parser.add_argument("--reduced-cost-tolerance", type=float, default=1e-9)
    parser.add_argument("--max-runtime-seconds", type=_positive_float, default=None)
    parser.add_argument("--max-cg-iterations", type=_positive_int, default=None)
    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _parse_values(
    value: str, converter: type[float] | type[int]
) -> tuple[float, ...] | tuple[int, ...]:
    try:
        values = tuple(converter(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid comma-separated values: {value}") from error
    if not values or any(item == "" for item in value.split(",")):
        raise argparse.ArgumentTypeError("comma-separated values must not be empty")
    return values


def _json_default(value: object) -> object:
    if hasattr(value, "__dict__"):
        return value.__dict__
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
