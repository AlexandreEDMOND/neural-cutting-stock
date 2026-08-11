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

    def test_features_have_fixed_width_for_different_type_counts(self) -> None:
        from neural_cutting_stock.learning import PatternCandidate, PricingState, pricing_features

        two_types = PricingState(
            "instance-1", 1, 100.0, 1.0, (20.0, 40.0), (3, 2), (0.5, 0.25), ((3, 0),)
        )
        four_types = PricingState(
            "instance-2",
            1,
            100.0,
            1.0,
            (10.0, 20.0, 40.0, 50.0),
            (3, 2, 2, 1),
            (0.5, 0.25, 0.2, 0.1),
            ((3, 0, 0, 0),),
        )

        self.assertEqual(
            len(pricing_features(two_types, PatternCandidate((1, 1), -0.1))),
            len(pricing_features(four_types, PatternCandidate((1, 1, 0, 0), -0.1))),
        )

    def test_features_are_invariant_to_a_joint_type_permutation(self) -> None:
        from neural_cutting_stock.learning import PatternCandidate, PricingState, pricing_features

        state = PricingState(
            "instance-1",
            1,
            100.0,
            1.0,
            (20.0, 40.0, 60.0),
            (3, 2, 1),
            (0.5, 0.25, 0.1),
            ((3, 0, 0), (0, 1, 0)),
        )
        permuted = PricingState(
            "instance-1",
            1,
            100.0,
            1.0,
            (60.0, 20.0, 40.0),
            (1, 3, 2),
            (0.1, 0.5, 0.25),
            ((0, 3, 0), (0, 0, 1)),
        )
        original = pricing_features(state, PatternCandidate((1, 0, 1), -0.1))
        reordered = pricing_features(permuted, PatternCandidate((1, 1, 0), -0.1))

        self.assertEqual(original, reordered)


if __name__ == "__main__":
    unittest.main()
