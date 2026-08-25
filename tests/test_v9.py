import unittest

import numpy as np

from train_v8 import public_member_names
from train_v9 import (
    CandidateResult,
    advancement_gate,
    base_rank,
    build_dual_features,
    compressed_regime_features,
    family_core_features,
    family_index,
    full_regime_features,
    hierarchical_base_features,
    member_family,
    mix_predictions,
)


class V9PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = public_member_names()
        rng = np.random.default_rng(42)
        self.pool = rng.uniform(0.01, 0.99, size=(20, len(self.names))).astype("float32")
        self.base = rng.uniform(0.01, 0.99, size=20)
        self.complete = np.array([0, 1] * 10, dtype="float32")
        self.missing_many = np.array([1, 0, 0, 0] * 5, dtype="float32")

    def test_every_public_member_maps_to_one_family(self) -> None:
        groups = family_index(self.names)
        combined = np.concatenate(list(groups.values()))
        np.testing.assert_array_equal(np.sort(combined), np.arange(len(self.names)))
        self.assertEqual(member_family("bolt_x"), "bolt")
        self.assertEqual(member_family("kirill_o1"), "extra")

    def test_dual_features_have_rank_and_logit_halves(self) -> None:
        dual, rank_half = build_dual_features(self.pool, self.base)
        self.assertEqual(rank_half.shape, (20, 206))
        self.assertEqual(dual.shape, (20, 412))
        self.assertTrue(np.isfinite(dual).all())
        np.testing.assert_allclose(rank_half[:, -1], base_rank(self.base).ravel())

    def test_compressed_regime_shape(self) -> None:
        groups = family_index(self.names)
        dual, ranks = build_dual_features(self.pool, self.base)
        core = family_core_features(self.pool, self.base, groups)
        features, _, _ = compressed_regime_features(
            dual, ranks, core, self.complete, self.missing_many
        )
        self.assertEqual(core.shape[1], 2 * len(groups) + 2)
        self.assertEqual(features.shape[1], dual.shape[1] + 3 * core.shape[1] + 5)
        self.assertTrue(np.isfinite(features).all())

    def test_hierarchical_regime_shape(self) -> None:
        groups = family_index(self.names)
        base = hierarchical_base_features(self.pool, self.base, groups)
        features, _, _ = full_regime_features(
            base, self.complete, self.missing_many
        )
        self.assertEqual(base.shape[1], 5 * len(groups) + 2)
        self.assertEqual(features.shape[1], 4 * base.shape[1] + 5)

    def test_prediction_mix_is_ranked_and_bounded(self) -> None:
        mixed = mix_predictions(np.arange(20), np.arange(20)[::-1], 0.55)
        self.assertGreater(float(mixed.min()), 0.0)
        self.assertLessEqual(float(mixed.max()), 1.0)

    def test_advancement_gate_requires_overall_and_every_fold_gain(self) -> None:
        y = np.array([0, 1] * 10)
        base = np.random.default_rng(7).uniform(size=20)
        passing = CandidateResult("pass", y.astype(float), y.astype(float), 1.0, [1.0] * 5, [])
        self.assertTrue(advancement_gate(passing, base, y)["passed"])
        failing = CandidateResult("fail", y.astype(float), y.astype(float), 1.0, [1.0] * 4 + [0.0], [])
        self.assertFalse(advancement_gate(failing, base, y)["passed"])


if __name__ == "__main__":
    unittest.main()
