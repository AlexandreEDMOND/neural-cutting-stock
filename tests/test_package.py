import unittest

from neural_cutting_stock import __version__


class PackageMetadataTests(unittest.TestCase):
    def test_package_version_is_exposed(self) -> None:
        self.assertEqual(__version__, "0.1.0")


class LearningInterfaceTests(unittest.TestCase):
    def test_state_and_decision_are_json_ready(self) -> None:
        from neural_cutting_stock.learning import (
            LEARNING_INTERFACE_SCHEMA_VERSION,
            ColumnSelectionDecision,
            PatternCandidate,
            PatternScore,
            PricingState,
        )

        state = PricingState(
            instance_id="instance-1",
            iteration_index=2,
            stock_length=100.0,
            kerf=1.0,
            piece_lengths=(20.0, 40.0),
            demands=(3, 2),
            dual_values=(0.5, 0.25),
            current_patterns=((5, 0), (0, 2)),
            rmp_objective_value=2.5,
        )
        candidate = PatternCandidate((1, 1), -0.1)
        decision = ColumnSelectionDecision(
            (PatternScore(candidate.pattern, 0.8),), (candidate.pattern,)
        )

        self.assertEqual(state.to_dict()["schema_version"], LEARNING_INTERFACE_SCHEMA_VERSION)
        self.assertEqual(state.to_dict()["dual_values"], [0.5, 0.25])
        self.assertEqual(decision.selected_patterns, ((1, 1),))

    def test_contract_rejects_inconsistent_shapes_and_decisions(self) -> None:
        import pytest

        from neural_cutting_stock.learning import (
            ColumnSelectionDecision,
            PatternScore,
            PricingState,
        )

        with pytest.raises(ValueError, match="dual_values must follow"):
            PricingState("instance-1", 1, 100.0, 0.0, (20.0, 40.0), (1, 1), (0.5,), ())
        with pytest.raises(ValueError, match="present in scored_candidates"):
            ColumnSelectionDecision((PatternScore((1, 0), 1.0),), ((0, 1),))


if __name__ == "__main__":
    unittest.main()
