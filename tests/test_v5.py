import unittest

import numpy as np
import pandas as pd

from train_v5 import add_v5_features, gate_passes, rank_blend_grid


class V5PipelineTests(unittest.TestCase):
    def sample_frame(self, ids: list[int]) -> pd.DataFrame:
        size = len(ids)
        return pd.DataFrame({
            "id": ids,
            "age": [20.0] * size,
            "daily_screen_time_hours": [4.0] * size,
            "social_media_hours": [1.0] * size,
            "gaming_hours": [0.5] * size,
            "work_study_hours": [3.0] * size,
            "sleep_hours": [8.0] * size,
            "notifications_per_day": [40.0] * size,
            "app_opens_per_day": [20.0] * size,
            "weekend_screen_time": [5.0] * size,
            "gender": ["Female"] * size,
            "stress_level": ["Low"] * size,
            "academic_work_impact": ["No"] * size,
        })

    def test_v5_features_are_numeric_aligned_and_drop_pattern(self) -> None:
        train = self.sample_frame([1, 2])
        test = self.sample_frame([3])
        test.loc[0, "gender"] = "Other"
        train_x, test_x = add_v5_features(train, test)
        self.assertListEqual(list(train_x.columns), list(test_x.columns))
        self.assertNotIn("missing_pattern", train_x.columns)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(train_x[col]) for col in train_x))
        self.assertIn("gender_Other", train_x.columns)

    def test_gate_requires_every_seed_and_mean_gain(self) -> None:
        self.assertTrue(gate_passes([
            {"gain": 0.0011}, {"gain": 0.0010}, {"gain": 0.0009},
        ]))
        self.assertFalse(gate_passes([
            {"gain": 0.0015}, {"gain": 0.0015}, {"gain": 0.0004},
        ]))
        self.assertFalse(gate_passes([{"gain": 0.0020}]))

    def test_rank_blend_grid_includes_both_standalones(self) -> None:
        y = pd.Series([0, 0, 1, 1])
        baseline = np.array([0.1, 0.3, 0.8, 0.9])
        challenger = np.array([0.2, 0.1, 0.9, 0.8])
        records = rank_blend_grid(y, baseline, challenger, step=0.5)
        self.assertEqual(records[0]["xgboost_weight"], 0.0)
        self.assertEqual(records[-1]["xgboost_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
