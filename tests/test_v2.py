import unittest

import numpy as np
import pandas as pd

from train_v2 import BASE_FEATURES, V2_CAT_COLS, add_v2_features, normalized_weights, percentile_rank


class V2PipelineTests(unittest.TestCase):
    def sample_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": [1, 2, 3],
            "age": [20.0, np.nan, 30.0],
            "daily_screen_time_hours": [4.0, 8.0, 6.0],
            "social_media_hours": [1.0, np.nan, 2.0],
            "gaming_hours": [0.5, 2.0, 1.0],
            "work_study_hours": [3.0, 1.0, np.nan],
            "sleep_hours": [8.0, 6.0, 7.0],
            "notifications_per_day": [40.0, 160.0, 90.0],
            "app_opens_per_day": [20.0, 80.0, 45.0],
            "weekend_screen_time": [5.0, 10.0, 7.0],
            "gender": ["Female", None, "Male"],
            "stress_level": ["Low", "High", None],
            "academic_work_impact": ["No", "Yes", None],
            "addicted_label": [0, 1, 0],
        })

    def test_v2_features_include_all_missing_indicators(self) -> None:
        features = add_v2_features(self.sample_frame())
        for col in BASE_FEATURES:
            self.assertIn(f"{col}__missing", features.columns)
        self.assertIn("missing_pattern", features.columns)
        self.assertTrue(set(V2_CAT_COLS).issubset(features.columns))
        self.assertNotIn("id", features.columns)
        self.assertNotIn("addicted_label", features.columns)

    def test_percentile_rank_is_monotonic_and_bounded(self) -> None:
        ranked = percentile_rank(np.array([10.0, 1.0, 5.0]))
        self.assertTrue(((ranked > 0) & (ranked < 1)).all())
        self.assertGreater(ranked[0], ranked[2])
        self.assertGreater(ranked[2], ranked[1])

    def test_weights_sum_to_one_for_subsets(self) -> None:
        weights = normalized_weights(["lgbm_a", "catboost"], "diversity")
        self.assertAlmostEqual(sum(weights.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
