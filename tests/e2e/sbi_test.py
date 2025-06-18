import unittest
from pathlib import Path

import numpy as np

from src.cli.config import Config
from src.cli.generate import generate
from src.cli.inference import inference
from tests.constants import MOCK_DIR

INFERENCE_YAML_YULE_SBI = MOCK_DIR.joinpath("yule_sbi_inference_config.yml")
INFERENCE_YAML_BD_SBI = MOCK_DIR.joinpath("bd_sbi_inference_config.yml")
SIMULATION_DATA = MOCK_DIR.joinpath("simulation_data.csv")


class End2EndSBITest(unittest.TestCase):
    def setUp(self):
        self.yule_config = Config(config_path=INFERENCE_YAML_YULE_SBI)
        self.outdir = Path(__file__).parent.joinpath("results")
        self.outdir.mkdir(exist_ok=True)
        return super().setUp()

    def test_yule_generate(self):
        """Test generation with Yule model"""
        result = generate(
            data_path=str(self.outdir / "test_yule_data.csv"),
            generator=self.yule_config.generator,
            parameters=self.yule_config.params,
            stats=self.yule_config.stats,
            seed=42,
        )
        self.assertTrue(result)

        # Verify file was created and has content
        generated_file = self.outdir / "test_yule_data.csv"
        self.assertTrue(generated_file.exists())

        # Load and verify data structure
        pop = self.yule_config.generator.load_data(str(generated_file))
        self.assertIsInstance(pop, list)
        self.assertGreater(len(pop), 0)

    def test_yule_inference(self):
        """Test inference with Yule model"""
        generator = self.yule_config.generator
        stats = self.yule_config.stats
        prior = self.yule_config.prior
        backend = self.yule_config.backend

        inference_data = inference(
            csv_file=str(SIMULATION_DATA),
            generator=generator,
            stats=stats,
            prior=prior,
            backend=backend,
            dir=self.outdir,
        )

        # Verify inference results
        self.assertIsNotNone(inference_data)
        self.assertIn("posterior_samples", inference_data)
        self.assertIn("posterior_predictive_stats", inference_data)

        # Check output files were created
        expected_files = [
            "posterior_samples.npy",
            "posterior_summary.csv",
            "posterior_predictive.npy",
            "pp_summaries.png",
        ]

        for filename in expected_files:
            file_path = self.outdir / filename
            self.assertTrue(file_path.exists(), f"Expected file {filename} not found")

    def test_parameter_recovery(self):
        """Test that inference can recover known parameters"""
        # Generate data with known parameters
        test_data_path = self.outdir / "parameter_recovery_data.csv"

        generate(
            data_path=str(test_data_path),
            generator=self.yule_config.generator,
            parameters=self.yule_config.params,
            seed=42,
        )

        # Run inference on generated data
        inference_data = inference(
            csv_file=str(test_data_path),
            generator=self.yule_config.generator,
            stats=self.yule_config.stats,
            prior=self.yule_config.prior,
            backend=self.yule_config.backend,
            dir=self.outdir,
        )

        # Check that posterior samples have the right shape
        samples = inference_data["posterior_samples"]
        expected_n_params = len(
            [v for v in self.yule_config.params.values() if v is not None]
        )
        self.assertEqual(samples.shape[1], expected_n_params)

        # Check that posterior means are reasonable (within prior bounds)
        posterior_means = np.mean(samples, axis=0)
        prior_low = self.yule_config.prior._low.numpy()
        prior_high = self.yule_config.prior._high.numpy()

        for i, mean in enumerate(posterior_means):
            self.assertGreaterEqual(
                mean,
                prior_low[i],
                f"Posterior mean {mean} below prior lower bound {prior_low[i]}",
            )
            self.assertLessEqual(
                mean,
                prior_high[i],
                f"Posterior mean {mean} above prior upper bound {prior_high[i]}",
            )

    def tearDown(self):
        # Clean up generated files
        for f in self.outdir.iterdir():
            if f.is_file():
                f.unlink()
        if self.outdir.exists():
            self.outdir.rmdir()
        return super().tearDown()


class End2EndBirthDeathTest(unittest.TestCase):
    """Separate test class for Birth-Death model if config exists"""

    def setUp(self):
        # Create a simple Birth-Death config for testing
        if not INFERENCE_YAML_BD_SBI.exists():
            self.skipTest("Birth-Death SBI config not available")

        self.bd_config = Config(config_path=INFERENCE_YAML_BD_SBI)
        self.outdir = Path(__file__).parent.joinpath("bd_results")
        self.outdir.mkdir(exist_ok=True)

    def test_bd_full_pipeline(self):
        """Test complete Birth-Death pipeline"""
        test_data_path = self.outdir / "bd_test_data.csv"

        # Generate
        generate(
            data_path=str(test_data_path),
            generator=self.bd_config.generator,
            parameters=self.bd_config.params,
            seed=42,
        )

        # Infer
        inference_data = inference(
            csv_file=str(test_data_path),
            generator=self.bd_config.generator,
            stats=self.bd_config.stats,
            prior=self.bd_config.prior,
            backend=self.bd_config.backend,
            dir=self.outdir,
        )

        # Verify results
        self.assertIsNotNone(inference_data)
        samples = inference_data["posterior_samples"]
        self.assertEqual(samples.shape[1], 2)  # Birth-Death has 2 parameters

    def tearDown(self):
        if hasattr(self, "outdir") and self.outdir.exists():
            for f in self.outdir.iterdir():
                if f.is_file():
                    f.unlink()
            self.outdir.rmdir()


if __name__ == "__main__":
    unittest.main()
