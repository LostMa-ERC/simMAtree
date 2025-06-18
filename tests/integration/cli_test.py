import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

# Integration tests for the complete CLI workflow


class CLIIntegrationTest(unittest.TestCase):
    """Test the complete CLI workflow"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_config.yml"
        self.data_file = self.temp_dir / "test_data.csv"
        self.results_dir = self.temp_dir / "results"

        # Create minimal working configuration
        self.config = {
            "generator": {
                "name": "BirthDeathAbundance",
                "config": {"n_init": 2, "Nact": 10, "Ninact": 10, "max_pop": 100},
            },
            "stats": {"name": "Abundance", "config": {"additional_stats": False}},
            "prior": {
                "name": "ConstrainedUniform2D",
                "config": {"low": [0.001, 0.0], "high": [0.02, 0.01]},
            },
            "params": {"lda": 0.01, "mu": 0.005},
            "inference": {
                "name": "SBI",
                "config": {
                    "method": "NPE",
                    "num_simulations": 8,
                    "num_rounds": 1,
                    "random_seed": 42,
                    "num_samples": 8,
                    "num_workers": 1,
                    "device": "cpu",
                },
            },
        }

        # Write config file
        with open(self.config_file, "w") as f:
            yaml.dump(self.config, f)

    def _run_cli_command(self, command, timeout=60):
        """Run a CLI command and return result"""
        cmd = [sys.executable, "-m", "src.__main__"] + command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(__file__).parent.parent.parent,  # Project root
            )
            return result
        except subprocess.TimeoutExpired:
            self.fail(f"Command timed out: {' '.join(cmd)}")

    def test_generate_command(self):
        """Test the generate command"""
        command = [
            "-c",
            str(self.config_file),
            "generate",
            "-o",
            str(self.data_file),
            "-s",
            "42",  # seed
        ]

        result = self._run_cli_command(command)

        # Check that command succeeded
        if result.returncode != 0:
            self.fail(f"Generate command failed: {result.stderr}")

        # Check that data file was created
        self.assertTrue(self.data_file.exists(), "Data file was not created")

        # Check that data file has correct format
        df = pd.read_csv(self.data_file, sep=";")
        self.assertIn("witness_ID", df.columns)
        self.assertIn("text_ID", df.columns)
        self.assertGreater(len(df), 0, "Generated data is empty")

    def test_infer_command(self):
        """Test the infer command"""
        # First generate some data
        self._run_cli_command(
            [
                "-c",
                str(self.config_file),
                "generate",
                "-o",
                str(self.data_file),
                "-s",
                "42",
            ]
        )

        # Create results directory
        self.results_dir.mkdir(exist_ok=True)

        # Run inference
        command = [
            "-c",
            str(self.config_file),
            "infer",
            "-i",
            str(self.data_file),
            "-o",
            str(self.results_dir),
        ]

        result = self._run_cli_command(command, timeout=120)  # Inference takes longer

        # Check that command succeeded
        if result.returncode != 0:
            self.fail(f"Infer command failed: {result.stderr}")

        # Check that result files were created
        expected_files = [
            "posterior_samples.npy",
            "posterior_summary.csv",
            "posterior_predictive.npy",
        ]

        for filename in expected_files:
            file_path = self.results_dir / filename
            self.assertTrue(file_path.exists(), f"Expected file {filename} not found")

    def test_score_command(self):
        """Test the score command"""
        # First generate data and run inference
        self._run_cli_command(
            [
                "-c",
                str(self.config_file),
                "generate",
                "-o",
                str(self.data_file),
                "-s",
                "42",
            ]
        )

        self.results_dir.mkdir(exist_ok=True)

        # Run inference (with very minimal settings for speed)
        self._run_cli_command(
            [
                "-c",
                str(self.config_file),
                "infer",
                "-i",
                str(self.data_file),
                "-o",
                str(self.results_dir),
            ],
            timeout=120,
        )

        # Run scoring
        command = ["-c", str(self.config_file), "score", "-d", str(self.results_dir)]

        result = self._run_cli_command(command)

        # Check that command succeeded
        if result.returncode != 0:
            self.fail(f"Score command failed: {result.stderr}")

        # Check that evaluation files were created
        expected_files = ["summary_metrics.csv"]
        for filename in expected_files:
            file_path = self.results_dir / filename
            self.assertTrue(
                file_path.exists(), f"Expected evaluation file {filename} not found"
            )

    def test_full_workflow(self):
        """Test the complete workflow: generate -> infer -> score"""
        # 1. Generate data
        gen_result = self._run_cli_command(
            [
                "-c",
                str(self.config_file),
                "generate",
                "-o",
                str(self.data_file),
                "-s",
                "42",
                "--show-params",
            ]
        )

        self.assertEqual(
            gen_result.returncode, 0, f"Generate failed: {gen_result.stderr}"
        )
        self.assertTrue(self.data_file.exists())

        # 2. Run inference
        self.results_dir.mkdir(exist_ok=True)

        inf_result = self._run_cli_command(
            [
                "-c",
                str(self.config_file),
                "infer",
                "-i",
                str(self.data_file),
                "-o",
                str(self.results_dir),
            ],
            timeout=120,
        )

        self.assertEqual(
            inf_result.returncode, 0, f"Inference failed: {inf_result.stderr}"
        )

        # 3. Score results
        score_result = self._run_cli_command(
            ["-c", str(self.config_file), "score", "-d", str(self.results_dir)]
        )

        self.assertEqual(
            score_result.returncode, 0, f"Scoring failed: {score_result.stderr}"
        )

        # Verify final outputs
        summary_file = self.results_dir / "summary_metrics.csv"
        self.assertTrue(summary_file.exists())

        # Check summary metrics are reasonable
        summary_df = pd.read_csv(summary_file)
        self.assertIn("rmse", summary_df.columns)
        self.assertIn("coverage_probability", summary_df.columns)

    def test_invalid_config(self):
        """Test CLI behavior with invalid configuration"""
        # Create invalid config (missing required fields)
        invalid_config = {"generator": {"name": "NonexistentGenerator"}}

        invalid_config_file = self.temp_dir / "invalid_config.yml"
        with open(invalid_config_file, "w") as f:
            yaml.dump(invalid_config, f)

        # Should fail gracefully
        result = self._run_cli_command(
            ["-c", str(invalid_config_file), "generate", "-o", str(self.data_file)]
        )

        self.assertNotEqual(result.returncode, 0, "Should fail with invalid config")
        self.assertIn("error", result.stderr.lower())

    def test_missing_input_file(self):
        """Test CLI behavior with missing input file"""
        nonexistent_file = self.temp_dir / "nonexistent.csv"

        result = self._run_cli_command(
            [
                "-c",
                str(self.config_file),
                "infer",
                "-i",
                str(nonexistent_file),
                "-o",
                str(self.results_dir),
            ]
        )

        self.assertNotEqual(result.returncode, 0, "Should fail with missing input file")

    def test_csv_separator_option(self):
        """Test CSV separator option"""
        # Generate data with default separator
        self._run_cli_command(
            ["-c", str(self.config_file), "generate", "-o", str(self.data_file)]
        )

        # Create results directory
        self.results_dir.mkdir(exist_ok=True)

        # Test with explicit separator
        result = self._run_cli_command(
            [
                "-c",
                str(self.config_file),
                "infer",
                "-i",
                str(self.data_file),
                "-o",
                str(self.results_dir),
                "-s",
                ";",  # explicit separator
            ],
            timeout=120,
        )

        self.assertEqual(
            result.returncode, 0, f"Inference with separator failed: {result.stderr}"
        )

    def tearDown(self):
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


class CLIYuleIntegrationTest(unittest.TestCase):
    """Test CLI with Yule model (more complex case)"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "yule_config.yml"

        # Yule model configuration
        self.yule_config = {
            "generator": {
                "name": "YuleAbundance",
                "config": {"n_init": 1, "Nact": 10, "Ninact": 10, "max_pop": 100},
            },
            "stats": {"name": "Abundance", "config": {"additional_stats": True}},
            "prior": {
                "name": "ConstrainedUniform4D",
                "config": {
                    "low": [0.0, 0.001, 0.0, 0.0],
                    "high": [0.5, 0.02, 0.01, 0.01],
                },
            },
            "params": {"LDA": 0.1, "lda": 0.01, "gamma": 0.005, "mu": 0.003},
            "inference": {
                "name": "SBI",
                "config": {
                    "method": "NPE",
                    "num_simulations": 8,
                    "num_rounds": 1,
                    "random_seed": 42,
                    "num_samples": 8,
                    "num_workers": 1,
                    "device": "cpu",
                },
            },
        }

        with open(self.config_file, "w") as f:
            yaml.dump(self.yule_config, f)

    def _run_cli_command(self, command, timeout=60):
        """Run a CLI command and return result"""
        cmd = [sys.executable, "-m", "src.__main__"] + command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(__file__).parent.parent.parent,
            )
            return result
        except subprocess.TimeoutExpired:
            self.fail(f"Command timed out: {' '.join(cmd)}")

    def test_yule_generate(self):
        """Test generation with Yule model"""
        data_file = self.temp_dir / "yule_data.csv"

        result = self._run_cli_command(
            ["-c", str(self.config_file), "generate", "-o", str(data_file), "-s", "42"]
        )

        self.assertEqual(
            result.returncode, 0, f"Yule generation failed: {result.stderr}"
        )
        self.assertTrue(data_file.exists())

        # Check data structure
        df = pd.read_csv(data_file, sep=";")
        self.assertGreater(len(df), 0)

    def test_yule_with_additional_stats(self):
        """Test Yule model with additional statistics"""
        data_file = self.temp_dir / "yule_data.csv"
        results_dir = self.temp_dir / "yule_results"

        # Generate
        self._run_cli_command(
            ["-c", str(self.config_file), "generate", "-o", str(data_file)]
        )

        # Infer
        results_dir.mkdir(exist_ok=True)
        result = self._run_cli_command(
            [
                "-c",
                str(self.config_file),
                "infer",
                "-i",
                str(data_file),
                "-o",
                str(results_dir),
            ],
            timeout=150,
        )  # Yule model may take longer

        if result.returncode == 0:
            # Check that 4D parameter space was handled
            import numpy as np

            samples = np.load(results_dir / "posterior_samples.npy")
            self.assertEqual(
                samples.shape[1], 4, "Should have 4 parameters for Yule model"
            )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


class CLIErrorHandlingTest(unittest.TestCase):
    """Test CLI error handling and edge cases"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def _run_cli_command(self, command, timeout=30):
        """Run a CLI command and return result"""
        cmd = [sys.executable, "-m", "src.__main__"] + command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(__file__).parent.parent.parent,
            )
            return result
        except subprocess.TimeoutExpired:
            self.fail(f"Command timed out: {' '.join(cmd)}")

    def test_missing_config_file(self):
        """Test behavior when config file is missing"""
        nonexistent_config = self.temp_dir / "missing.yml"

        result = self._run_cli_command(
            ["-c", str(nonexistent_config), "generate", "-o", "output.csv"]
        )

        self.assertNotEqual(result.returncode, 0)

    def test_invalid_command(self):
        """Test behavior with invalid command"""
        # Create minimal valid config
        config_file = self.temp_dir / "config.yml"
        with open(config_file, "w") as f:
            yaml.dump({"generator": {"name": "BirthDeathAbundance"}}, f)

        result = self._run_cli_command(["-c", str(config_file), "invalid_command"])

        self.assertNotEqual(result.returncode, 0)

    def test_help_command(self):
        """Test help command"""
        result = self._run_cli_command(["--help"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("usage", result.stdout.lower())

    def test_generate_help(self):
        """Test help for generate command"""
        result = self._run_cli_command(["generate", "--help"])
        self.assertIn(result.returncode, [0, 1])

    def test_infer_help(self):
        """Test help for infer command"""
        result = self._run_cli_command(["infer", "--help"])
        self.assertIn(result.returncode, [0, 1])

    def test_score_help(self):
        """Test help for score command"""
        result = self._run_cli_command(["score", "--help"])
        self.assertIn(result.returncode, [0, 1])

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
