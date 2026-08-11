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

    def test_candidate_pool_is_deterministic_and_excludes_existing_patterns(self) -> None:
        from neural_cutting_stock.learning import (
            CANDIDATE_POOL_SCHEMA_VERSION,
            deterministic_candidate_pool,
        )
        from neural_cutting_stock.problem import CuttingStockInstance

        instance = CuttingStockInstance(10, 0, [2, 3], [2, 2])
        first = deterministic_candidate_pool(instance, (0.4, 0.7), ((2, 0),))
        second = deterministic_candidate_pool(instance, (0.4, 0.7), ((2, 0),))

        self.assertEqual(CANDIDATE_POOL_SCHEMA_VERSION, "candidate-pool-v1")
        self.assertEqual(first, second)
        self.assertNotIn((2, 0), [candidate.pattern for candidate in first])
        self.assertEqual(
            list(first),
            sorted(first, key=lambda candidate: (candidate.reduced_cost, candidate.pattern)),
        )
        self.assertTrue(all(instance.capacity_used(item.pattern) <= 10 for item in first))

    def test_candidate_pool_limit_is_explicit_and_exact_pricing_remains_unchanged(self) -> None:
        from neural_cutting_stock.learning import deterministic_candidate_pool
        from neural_cutting_stock.problem import CuttingStockInstance
        from neural_cutting_stock.solver import ExactPricing

        instance = CuttingStockInstance(10, 0, [2, 3], [2, 2])
        pool = deterministic_candidate_pool(instance, (0.4, 0.7), max_candidates=2)

        self.assertEqual(len(pool), 2)
        exact = ExactPricing(instance).solve((0.4, 0.7))
        self.assertEqual(exact.pattern, (2, 2))
        self.assertAlmostEqual(exact.reduced_cost, -1.2)

    def test_linear_model_learns_and_scores_state_candidates(self) -> None:
        from neural_cutting_stock.learning import (
            LinearColumnScoringModel,
            PatternCandidate,
            PricingState,
            pricing_features,
        )

        state = PricingState(
            "instance-1", 1, 100.0, 1.0, (20.0, 40.0), (3, 2), (0.5, 0.25), ()
        )
        candidates = (PatternCandidate((1, 0), 0.2), PatternCandidate((0, 1), 0.4))
        rows = tuple(pricing_features(state, candidate) for candidate in candidates)
        model = LinearColumnScoringModel.fit(rows, (0.0, 1.0))

        scores = model.score(state, candidates)

        assert model.feature_width == len(rows[0])
        assert [score.pattern for score in scores] == [(1, 0), (0, 1)]
        assert scores[0].score < scores[1].score

    def test_linear_model_rejects_invalid_training_data(self) -> None:
        import pytest

        from neural_cutting_stock.learning import LinearColumnScoringModel

        with pytest.raises(ValueError, match="must not be empty"):
            LinearColumnScoringModel.fit((), ())
        with pytest.raises(ValueError, match="one value per feature row"):
            LinearColumnScoringModel.fit(((1.0,),), ())

    def test_linear_model_exposes_parameters_for_persistence(self) -> None:
        from neural_cutting_stock.learning import LinearColumnScoringModel

        model = LinearColumnScoringModel((1.0, 2.0), 3.0)

        assert model.weights == (1.0, 2.0)
        assert model.bias == 3.0

    def test_selection_policy_applies_a_deterministic_candidate_budget(self) -> None:
        from neural_cutting_stock.learning import (
            LearnedColumnSelectionPolicy,
            PatternCandidate,
            PatternScore,
            PricingState,
        )

        class FixedScorer:
            def score(self, state, candidates):
                del state
                values = {(1, 0): 0.5, (0, 1): 0.9, (1, 1): 0.9}
                return tuple(
                    PatternScore(candidate.pattern, values[candidate.pattern])
                    for candidate in candidates
                )

        state = PricingState("instance-1", 1, 10.0, 0.0, (2.0, 3.0), (2, 2), (0.5, 0.25), ())
        candidates = (
            PatternCandidate((1, 0), 0.1),
            PatternCandidate((1, 1), -0.1),
            PatternCandidate((0, 1), 0.0),
        )

        decision = LearnedColumnSelectionPolicy(FixedScorer(), candidate_budget=2).select(
            state, candidates
        )

        assert decision.selected_patterns == ((0, 1), (1, 1))
        assert tuple(score.pattern for score in decision.scored_candidates) == tuple(
            candidate.pattern for candidate in candidates
        )

    def test_selection_policy_rejects_invalid_budget(self) -> None:
        import pytest

        from neural_cutting_stock.learning import LearnedColumnSelectionPolicy

        with pytest.raises(ValueError, match="candidate_budget must be a positive integer"):
            LearnedColumnSelectionPolicy(object(), candidate_budget=0)


if __name__ == "__main__":
    unittest.main()
