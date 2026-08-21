import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from train_v4 import eligible_blends, load_density_weights
from validate_v4 import density_ratio


class V4PipelineTests(unittest.TestCase):
    def test_density_ratio_is_positive_normalized_and_ordered(self) -> None:
        result = density_ratio(np.array([0.1, 0.2, 0.4]), 6, 3)
        self.assertTrue((result > 0).all())
        self.assertAlmostEqual(result.mean(), 1.0)
        self.assertTrue(np.all(np.diff(result) > 0))

    def test_weight_loading_aligns_by_id(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weights.csv"
            pd.DataFrame({"id": [2, 1], "density_weight": [0.5, 1.5]}).to_csv(path, index=False)
            result = load_density_weights(path, pd.Series([1, 2]))
        np.testing.assert_allclose(result, [1.5, 0.5])

    def test_blend_gate_requires_both_metrics_to_improve(self) -> None:
        y = pd.Series([0, 0, 1, 1, 0, 1])
        baseline = np.array([0.1, 0.8, 0.7, 0.6, 0.2, 0.9])
        challenger = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
        records = eligible_blends(y, baseline, challenger, np.ones(6), step=1.0)
        self.assertFalse(records[0]["eligible"])
        self.assertTrue(records[1]["eligible"])

    def test_blend_gate_allows_only_numerical_threshold_tolerance(self) -> None:
        y = pd.Series([0, 0, 1, 1, 0, 1])
        baseline = np.array([0.1, 0.8, 0.7, 0.6, 0.2, 0.9])
        challenger = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
        strict = eligible_blends(
            y, baseline, challenger, np.ones(6), step=1.0,
            minimum_gain=1.0, comparison_tolerance=0.0,
        )
        tolerant = eligible_blends(
            y, baseline, challenger, np.ones(6), step=1.0,
            minimum_gain=1.0, comparison_tolerance=1.0,
        )
        self.assertFalse(strict[1]["eligible"])
        self.assertTrue(tolerant[1]["eligible"])


if __name__ == "__main__":
    unittest.main()
