import json

from neural_cutting_stock.__main__ import main


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
    assert output["integrality_gap"] == 0.5
    assert output["integer_master"]["objective_value"] == 2
    assert output["verification"]["feasible"] is True
