import unittest

import numpy as np
import pandas as pd

from train_v6 import RAW_FEATURES, ablation_passes, hard_test_masks, make_masked_copies


class V6PipelineTests(unittest.TestCase):
    def sample_frame(self, rows: int = 8) -> pd.DataFrame:
        return pd.DataFrame({
            "id": np.arange(rows),
            "age": [20.0] * rows,
            "daily_screen_time_hours": [4.0] * rows,
            "social_media_hours": [1.0] * rows,
            "gaming_hours": [0.5] * rows,
            "work_study_hours": [3.0] * rows,
            "sleep_hours": [8.0] * rows,
            "notifications_per_day": [40.0] * rows,
            "app_opens_per_day": [20.0] * rows,
            "weekend_screen_time": [5.0] * rows,
            "gender": ["Female"] * rows,
            "stress_level": ["Low"] * rows,
            "academic_work_impact": ["No"] * rows,
            "addicted_label": [0, 1] * (rows // 2),
        })

    def test_masked_copies_are_hard_and_preserve_labels(self) -> None:
        frame = self.sample_frame()
        masks = np.zeros((2, len(RAW_FEATURES)), dtype=bool)
        masks[0, :2] = True
        masks[1, 2:5] = True
        copies = make_masked_copies(frame, masks, ratio=0.5, seed=42)
        self.assertTrue((copies[RAW_FEATURES].isna().sum(axis=1) >= 2).all())
        self.assertTrue(set(copies["addicted_label"]).issubset({0, 1}))

    def test_hard_test_masks_filters_easy_patterns(self) -> None:
        frame = self.sample_frame(4)
        frame.loc[0, "age"] = np.nan
        frame.loc[1, ["age", "sleep_hours"]] = np.nan
        masks = hard_test_masks(frame)
        self.assertEqual(len(masks), 1)
        self.assertGreaterEqual(masks[0].sum(), 2)

    def test_ablation_gate_checks_global_hard_and_easy_slices(self) -> None:
        record = {
            "global_gain": 0.0004,
            "slices": {
                "0": {"gain": -0.0001},
                "1": {"gain": 0.0},
                "2+": {"gain": 0.0012},
            },
        }
        self.assertTrue(ablation_passes(record))
        record["slices"]["0"]["gain"] = -0.0003
        self.assertFalse(ablation_passes(record))


if __name__ == "__main__":
    unittest.main()
