import unittest

import numpy as np
import pandas as pd

from train_v3 import grid_blends, missing_bucket


class V3PipelineTests(unittest.TestCase):
    def test_missing_bucket(self) -> None:
        frame = pd.DataFrame({
            "age": [20.0, np.nan, np.nan],
            "daily_screen_time_hours": [5.0, 5.0, np.nan],
            "social_media_hours": [1.0, 1.0, 1.0],
            "gaming_hours": [1.0, 1.0, 1.0],
            "work_study_hours": [2.0, 2.0, 2.0],
            "sleep_hours": [7.0, 7.0, 7.0],
            "notifications_per_day": [50.0, 50.0, 50.0],
            "app_opens_per_day": [30.0, 30.0, 30.0],
            "weekend_screen_time": [7.0, 7.0, 7.0],
            "gender": ["Female", "Female", "Female"],
            "stress_level": ["Low", "Low", "Low"],
            "academic_work_impact": ["No", "No", "No"],
        })
        np.testing.assert_array_equal(missing_bucket(frame), np.array([0, 1, 2]))

    def test_grid_blend_keeps_material_improvement(self) -> None:
        y = pd.Series([0, 0, 0, 1, 1, 1])
        oof = {
            "a": np.array([0.1, 0.2, 0.8, 0.4, 0.9, 0.7]),
            "b": np.array([0.3, 0.1, 0.4, 0.9, 0.6, 0.8]),
            "c": np.array([0.2, 0.3, 0.7, 0.5, 0.8, 0.6]),
        }
        selection, _, _ = grid_blends(y, oof, oof, step=50)
        self.assertIn("selected", selection)
        self.assertGreaterEqual(selection["selected"]["auc"], selection["best_single"]["auc"])

    def test_two_candidate_grid_includes_blends(self) -> None:
        y = pd.Series([0, 0, 1, 1, 0, 1])
        oof = {
            "new": np.array([0.1, 0.6, 0.7, 0.8, 0.3, 0.5]),
            "old": np.array([0.2, 0.4, 0.9, 0.6, 0.1, 0.8]),
        }
        selection, blend_oof, _ = grid_blends(y, oof, oof, step=20)
        self.assertTrue(any(name.startswith("blend_") for name in blend_oof))
        self.assertIn("best_grid_result", selection)


if __name__ == "__main__":
    unittest.main()
