from pathlib import Path

ALTERNATIVE_SPEC = Path(__file__).parents[1] / "docs" / "phase-5-alternative.md"


def test_phase5_alternative_specifies_bounded_supervised_contract() -> None:
    specification = ALTERNATIVE_SPEC.read_text(encoding="utf-8")

    for required_section in ("## État", "## Action", "## Horizon", "## Portée et garde-fous"):
        assert required_section in specification

    assert "`bounded-column-selection-v1`" in specification
    assert "`PricingState`" in specification
    assert "`PatternCandidate`" in specification
    assert "`candidate_budget`" in specification
    assert "`H = 1`" in specification
    assert "pricing exact" in specification
    assert "La politique ne déclare jamais" in specification
    assert "la convergence" in specification
