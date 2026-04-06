from __future__ import annotations

import importlib.util
import unittest


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy is not installed in the active Python environment")
class ComputeFisherRatioTest(unittest.TestCase):
    def test_compute_fisher_ratio_matches_manual_values(self) -> None:
        import numpy as np

        from shared.metrics import compute_fisher_ratio

        positive = np.array([[2.0, 0.0], [4.0, 0.0]])
        negative = np.array([[0.0, 0.0], [2.0, 0.0]])

        stats = compute_fisher_ratio(positive, negative, epsilon=1e-10)

        self.assertAlmostEqual(stats["between_class_variance"], 4.0)
        self.assertAlmostEqual(stats["within_class_variance"], 2.0)
        self.assertAlmostEqual(stats["fisher_ratio"], 2.0, places=6)
        self.assertEqual(stats["num_positive"], 2)
        self.assertEqual(stats["num_negative"], 2)

    def test_zero_within_and_zero_between_returns_zero(self) -> None:
        import numpy as np

        from shared.metrics import compute_fisher_ratio

        positive = np.array([[1.0, 1.0], [1.0, 1.0]])
        negative = np.array([[1.0, 1.0], [1.0, 1.0]])

        stats = compute_fisher_ratio(positive, negative, epsilon=1e-10)

        self.assertEqual(stats["within_class_variance"], 0.0)
        self.assertEqual(stats["between_class_variance"], 0.0)
        self.assertEqual(stats["fisher_ratio"], 0.0)

    def test_zero_within_uses_epsilon_stabilization(self) -> None:
        import numpy as np

        from shared.metrics import compute_fisher_ratio

        positive = np.array([[3.0, 0.0], [3.0, 0.0]])
        negative = np.array([[0.0, 0.0], [0.0, 0.0]])

        stats = compute_fisher_ratio(positive, negative, epsilon=1e-6)

        self.assertEqual(stats["within_class_variance"], 0.0)
        self.assertAlmostEqual(stats["between_class_variance"], 9.0)
        self.assertAlmostEqual(stats["fisher_ratio"], 9.0 / 1e-6, places=2)


if __name__ == "__main__":
    unittest.main()
