from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy is not installed in the active Python environment")
class FisherRatioConfigTest(unittest.TestCase):
    def test_fisher_ratio_config_builds_experiment(self) -> None:
        from run_experiment import build_experiment, load_yaml_config

        repo_root = Path(__file__).resolve().parents[1]
        config, _ = load_yaml_config(repo_root / "configs" / "fisher_ratio_encoder.yaml")

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = build_experiment(config, Path(tmpdir))

        self.assertEqual(type(experiment).__name__, "FisherRatioExperiment")
        self.assertEqual(experiment.components, ["encoder", "decoder"])


if __name__ == "__main__":
    unittest.main()
