"""Command-line entry point for the classical solver."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import ColumnGeneration, verify_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    instance = CuttingStockInstance(
        stock_length=args.stock_length,
        kerf=args.kerf,
        piece_lengths=_parse_values(args.piece_lengths, float),
        demands=_parse_values(args.demands, int),
    )
    result = ColumnGeneration(instance, args.reduced_cost_tolerance).solve()
    output: dict[str, object] = {
        "solver": args.solver,
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
    }
    if result.rmp_result is not None:
        output["rmp"] = {
            "status": result.rmp_result.status,
            "objective_value": result.rmp_result.objective_value,
            "column_values": result.rmp_result.column_values,
        }
    if result.pricing_result is not None:
        output["pricing"] = {
            "status": result.pricing_result.status,
            "pattern": result.pricing_result.pattern,
            "reduced_cost": result.pricing_result.reduced_cost,
        }
    if result.integer_master_result is not None:
        integer_result = result.integer_master_result
        output["integer_master"] = {
            "status": integer_result.status,
            "objective_value": integer_result.objective_value,
            "column_values": integer_result.column_values,
        }
        verification = verify_plan(
            instance, result.patterns, integer_result.column_values
        )
        output["verification"] = asdict(verification)
    print(json.dumps(output, default=_json_default, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve a 1D Cutting Stock instance.")
    parser.add_argument("--solver", choices=("classical",), required=True)
    parser.add_argument("--stock-length", type=float, required=True)
    parser.add_argument("--kerf", type=float, default=0.0)
    parser.add_argument("--piece-lengths", required=True, help="Comma-separated lengths")
    parser.add_argument("--demands", required=True, help="Comma-separated positive integers")
    parser.add_argument("--reduced-cost-tolerance", type=float, default=1e-9)
    return parser


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
