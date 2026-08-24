import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from train_v8 import (
    MIN_MEMBER_GAIN,
    StackResult,
    advancement_gate,
    percentile_rank,
    public_member_names,
    select_stack,
    validate_prediction,
)


def result(auc: float, folds: list[float]) -> StackResult:
    import pandas as pd

    return StackResult(
        oof=np.array([0.1, 0.2, 0.8, 0.9]),
        test=np.array([0.3, 0.7]),
        auc=auc,
        fold_auc=folds,
        coefficients=pd.DataFrame(),
    )


class V8PipelineTests(unittest.TestCase):
    def test_reference_registry_has_205_unique_members(self) -> None:
        names = public_member_names()
        self.assertEqual(len(names), 205)
        self.assertEqual(len(set(names)), 205)
        self.assertEqual(names[0], "naji07")
        self.assertEqual(names[-1], "cat_strall_d8")

    def test_percentile_rank_uses_average_ties(self) -> None:
        ranked = percentile_rank(np.array([10.0, 20.0, 20.0, 40.0]))
        np.testing.assert_allclose(ranked, [0.25, 0.625, 0.625, 1.0])

    def test_prediction_validation_rejects_wrong_length_and_nan(self) -> None:
        with self.assertRaises(ValueError):
            validate_prediction(np.array([0.1]), 2, "short")
        with self.assertRaises(ValueError):
            validate_prediction(np.array([0.1, np.nan]), 2, "nan")

    def test_v7_member_is_selected_only_for_material_gain(self) -> None:
        public = result(0.97020, [0.97] * 5)
        small = result(public.auc + MIN_MEMBER_GAIN / 2, [0.97] * 5)
        large = result(public.auc + MIN_MEMBER_GAIN, [0.97] * 5)
        self.assertEqual(select_stack(public, small)[0], "public")
        self.assertEqual(select_stack(public, large)[0], "public_plus_v7")

    def test_advancement_gate_requires_overall_and_every_fold_gain(self) -> None:
        y = np.array([0, 1] * 10)
        v7 = np.array([0.1, 0.9] * 10)
        perfect_auc = roc_auc_score(y, v7)
        candidate = result(perfect_auc + 0.001, [1.0] * 5)
        gate = advancement_gate(candidate, v7, y)
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
