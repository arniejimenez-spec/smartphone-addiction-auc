import json
import unittest
from pathlib import Path

import numpy as np

from train_v10_gpu import (
    EXPECTED_MEMBERS,
    build_dual,
    build_regime_features,
    clipped_logit,
    parse_args,
    rank01,
    standardize_in_place,
)


ROOT = Path(__file__).resolve().parents[1]


class V10PipelineTests(unittest.TestCase):
    def test_rank01_matches_reference_half_ranks(self) -> None:
        ranked = rank01(np.array([0.4, 0.1, 0.9, 0.2]))
        np.testing.assert_allclose(ranked, [0.625, 0.125, 0.875, 0.375])

    def test_logit_is_finite_at_probability_boundaries(self) -> None:
        transformed = clipped_logit(np.array([0.0, 0.5, 1.0]))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertAlmostEqual(float(transformed[1]), 0.0)

    def test_exact_dual_and_regime_widths(self) -> None:
        rng = np.random.default_rng(42)
        pool = rng.uniform(0.01, 0.99, size=(12, EXPECTED_MEMBERS))
        base = rng.uniform(0.01, 0.99, size=12)
        dual, ranks = build_dual(pool, base)
        regime = build_regime_features(
            ranks,
            dual,
            np.array([0, 1] * 6),
            np.array([1, 0, 0] * 4),
        )
        self.assertEqual(dual.shape, (12, 412))
        self.assertEqual(ranks.shape, (12, 206))
        self.assertEqual(regime.shape, (12, 1653))
        self.assertTrue(np.isfinite(regime).all())

    def test_standardization_uses_train_statistics(self) -> None:
        train = np.array([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]])
        test = np.array([[7.0, 15.0]])
        standardize_in_place(train, test)
        np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(train.std(axis=0), 1.0, atol=1e-12)
        self.assertTrue(np.isfinite(test).all())

    def test_iteration_arguments_are_validated(self) -> None:
        self.assertFalse(parse_args([]).source_closure)
        self.assertEqual(parse_args([]).max_total_iter, 20000)
        with self.assertRaises(SystemExit):
            parse_args(["--max-total-iter", "100", "--block-iter", "200"])

    def test_generated_notebook_is_gpu_enabled_and_self_contained(self) -> None:
        notebook = json.loads(
            (ROOT / "kaggle" / "v10_exact_gpu_fusion.ipynb").read_text(encoding="utf-8")
        )
        self.assertTrue(notebook["metadata"]["kaggle"]["isGpuEnabled"])
        source = "".join(notebook["cells"][1]["source"])
        self.assertIn("def fit_gpu_logistic", source)
        self.assertIn("'/kaggle/working'", source)
        self.assertIn("'--max-total-iter', '20000'", source)
        script = (ROOT / "train_v10_gpu.py").read_text(encoding="utf-8")
        script_prefix = script.split('\nif __name__ == "__main__":\n', 1)[0]
        notebook_prefix = source.split("\n\n# Kaggle entrypoint:", 1)[0]
        self.assertEqual(notebook_prefix, script_prefix)


if __name__ == "__main__":
    unittest.main()
