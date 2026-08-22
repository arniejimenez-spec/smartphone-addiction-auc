import unittest

import numpy as np
import pandas as pd

from train_v7 import (
    NUM_COLS,
    TARGET_ENCODED_COLS,
    apply_fold_target_encoding,
    blend_grid,
    prepare_features,
)


class V7PipelineTests(unittest.TestCase):
    def sample_frame(self, ids: list[int], include_target: bool = False) -> pd.DataFrame:
        size = len(ids)
        frame = pd.DataFrame({
            "id": ids,
            "age": [20.0, 20.0, 30.0, 30.0][:size],
            "daily_screen_time_hours": [4.0] * size,
            "social_media_hours": [1.0] * size,
            "gaming_hours": [0.5] * size,
            "work_study_hours": [2.0] * size,
            "sleep_hours": [8.0] * size,
            "notifications_per_day": [40.0] * size,
            "app_opens_per_day": [20.0] * size,
            "weekend_screen_time": [5.0] * size,
            "gender": ["Female"] * size,
            "stress_level": ["Low"] * size,
            "academic_work_impact": ["No"] * size,
        })
        if include_target:
            frame["addicted_label"] = [0, 1, 0, 1][:size]
        return frame

    def test_prepare_features_matches_selected_notebook_shape(self) -> None:
        train = self.sample_frame([1, 2, 3, 4], include_target=True)
        test = self.sample_frame([5, 6])
        train_x, test_x = prepare_features(train, test)
        self.assertEqual(train_x.shape[1], 44)
        self.assertListEqual(list(train_x.columns), list(test_x.columns))
        self.assertNotIn("gender", train_x)
        self.assertIn("age_freq", train_x)
        self.assertTrue(all(col in train_x for col in TARGET_ENCODED_COLS))

    def test_frequency_encoding_uses_combined_covariates(self) -> None:
        train = self.sample_frame([1, 2], include_target=True)
        test = self.sample_frame([3, 4])
        train_x, _ = prepare_features(train, test)
        self.assertEqual(train_x.loc[0, "age_freq"], 4)
        self.assertEqual(train_x.loc[1, "age_freq"], 4)

    def test_validation_target_encoding_ignores_validation_labels(self) -> None:
        train = self.sample_frame([1, 2, 3, 4], include_target=True)
        test = self.sample_frame([5, 6])
        train_x, test_x = prepare_features(train, test)
        train_idx = np.array([0, 1])
        valid_idx = np.array([2, 3])
        y_a = pd.Series([0, 1, 0, 1])
        y_b = pd.Series([0, 1, 1, 0])
        _, valid_a, _ = apply_fold_target_encoding(
            train_x, test_x, y_a, train_idx, valid_idx
        )
        _, valid_b, _ = apply_fold_target_encoding(
            train_x, test_x, y_b, train_idx, valid_idx
        )
        np.testing.assert_allclose(
            valid_a[TARGET_ENCODED_COLS], valid_b[TARGET_ENCODED_COLS]
        )

    def test_blend_grid_includes_both_standalones(self) -> None:
        y = pd.Series([0, 0, 1, 1])
        v5 = np.array([0.1, 0.3, 0.8, 0.9])
        v7 = np.array([0.2, 0.1, 0.9, 0.8])
        records = blend_grid(y, v5, v7, step=0.5)
        self.assertEqual(records[0]["v7_weight"], 0.0)
        self.assertEqual(records[-1]["v7_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
