import json

from neural_cutting_stock.__main__ import main
from neural_cutting_stock.solver import (
    ColumnGeneration,
    ColumnGenerationResult,
    IntegerMasterResult,
)


def test_classical_cli_emits_structured_verified_result(capsys) -> None:
    exit_code = main(
        [
            "--solver",
            "classical",
            "--stock-length",
            "10",
            "--piece-lengths",
            "6,4",
            "--demands",
            "1,2",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "converged"
    assert output["termination_reason"] == "no_improving_column"
    assert output["reduced_cost_tolerance"] == 1e-9
    assert output["integrality_gap"] == 0.5
    assert output["rmp"]["dual_values"] == [0.5, 0.5]
    assert output["pricing"]["dual_value"] == 1.0
    assert output["integer_master"]["objective_value"] == 2
    assert output["verification"]["feasible"] is True


def test_classical_cli_preserves_integer_master_failure_status(
    monkeypatch, capsys
) -> None:
    failed_result = ColumnGenerationResult(
        status="limit_reached",
        patterns=((1,),),
        rmp_result=None,
        pricing_result=None,
        integer_master_result=IntegerMasterResult(1, None, (), "time limit"),
        iterations=1,
        columns_added=0,
        duplicate_columns=0,
        termination_reason="integer_master_failed",
    )
    monkeypatch.setattr(ColumnGeneration, "solve", lambda self: failed_result)

    exit_code = main(
        [
            "--solver",
            "classical",
            "--stock-length",
            "10",
            "--piece-lengths",
            "6",
            "--demands",
            "1",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "limit_reached"
    assert output["termination_reason"] == "integer_master_failed"
    assert output["integer_master"]["objective_value"] is None
    assert output["integer_master"]["message"] == "time limit"
    assert "verification" not in output
