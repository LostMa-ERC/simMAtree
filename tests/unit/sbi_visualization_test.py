import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend for testing

from src.utils.visualisation import (
    compute_hpdi_point,
    plot_combined_hpdi,
    plot_marginal_posterior,
    plot_posterior_predictive_stats,
)


class SBIVisualizationTest(unittest.TestCase):
    def setUp(self):
        self.outdir = Path(__file__).parent.joinpath("vis_test_output")
        self.outdir.mkdir(exist_ok=True)

        # Create mock data for testing
        np.random.seed(42)
        self.n_samples = 100
        self.n_dims = 4

        # Mock posterior samples (4D for Yule model)
        self.posterior_samples = np.random.normal(
            loc=[0.3, 0.01, 0.005, 0.003],
            scale=[0.1, 0.003, 0.001, 0.001],
            size=(self.n_samples, self.n_dims),
        )
        self.posterior_samples = np.abs(
            self.posterior_samples
        )  # Ensure positive values

        # Mock posterior predictive stats (5 statistics from AbundanceStats)
        self.pp_stats = np.random.exponential(
            scale=[1000, 50, 10, 5, 30], size=(self.n_samples, 5)
        )

        # Mock observed values
        self.obs_values = [1200, 45, 12, 6, 25]

        # Mock true parameter values
        self.true_values = [0.3, 0.009, 0.001, 0.0033]

        return super().setUp()

    def test_compute_hpdi_point(self):
        """Test HPDI point computation"""
        hpdi_point, hpdi_samples = compute_hpdi_point(
            self.posterior_samples, prob_level=0.95
        )

        # Check output shapes
        self.assertEqual(hpdi_point.shape, (self.n_dims,))
        self.assertLessEqual(hpdi_samples.shape[0], self.n_samples)
        self.assertEqual(hpdi_samples.shape[1], self.n_dims)

        # Check that HPDI point is within data range
        for i in range(self.n_dims):
            self.assertGreaterEqual(hpdi_point[i], np.min(self.posterior_samples[:, i]))
            self.assertLessEqual(hpdi_point[i], np.max(self.posterior_samples[:, i]))

    def test_plot_posterior_predictive_stats(self):
        """Test posterior predictive statistics plotting"""
        plot_posterior_predictive_stats(
            samples=self.pp_stats, output_dir=self.outdir, obs_value=self.obs_values
        )

        # Check that plot file was created
        plot_file = self.outdir / "pp_summaries.png"
        self.assertTrue(plot_file.exists())

        # Test without observed values
        plot_posterior_predictive_stats(samples=self.pp_stats, output_dir=self.outdir)

    def test_plot_marginal_posterior(self):
        """Test marginal posterior plotting"""
        hpdi_point, _ = compute_hpdi_point(self.posterior_samples)

        plot_marginal_posterior(
            samples=self.posterior_samples,
            output_dir=self.outdir,
            hpdi_point=hpdi_point,
            true_value=self.true_values,
        )

        # Check that plot file was created
        plot_file = self.outdir / "posterior.png"
        self.assertTrue(plot_file.exists())

    def test_plot_combined_hpdi(self):
        """Test combined HPDI plotting"""
        # Test with single dataset
        plot_combined_hpdi(
            samples_list=[self.posterior_samples],
            output_dir=self.outdir,
            true_values=self.true_values,
            param_names=[r"$\Lambda$", r"$\lambda$", r"$\gamma$", r"$\mu$"],
        )

        # The function saves without extension, matplotlib adds .png
        self.assertTrue(
            any(f.name.startswith("pairplot") for f in self.outdir.iterdir())
        )

        # Test with multiple datasets
        samples_2 = self.posterior_samples + np.random.normal(
            0, 0.01, self.posterior_samples.shape
        )
        plot_combined_hpdi(
            samples_list=[self.posterior_samples, samples_2],
            output_dir=self.outdir,
            dataset_names=["Dataset 1", "Dataset 2"],
            true_values=self.true_values,
        )

    def test_empty_samples_handling(self):
        """Test handling of empty or invalid samples"""
        # Test with empty samples
        empty_samples = np.array([]).reshape(0, 5)

        # This should not crash
        plot_posterior_predictive_stats(samples=empty_samples, output_dir=self.outdir)

    def test_outlier_handling(self):
        """Test handling of outlier values in plotting"""
        # Create samples with outliers
        outlier_samples = self.pp_stats.copy()
        outlier_samples[0, 1] = 1e6  # Add extreme outlier

        # This should handle outliers gracefully
        plot_posterior_predictive_stats(
            samples=outlier_samples, output_dir=self.outdir, obs_value=self.obs_values
        )

        plot_file = self.outdir / "pp_summaries.png"
        self.assertTrue(plot_file.exists())

    def tearDown(self):
        # Clean up generated plots
        for f in self.outdir.iterdir():
            if f.is_file():
                f.unlink()
        if self.outdir.exists():
            self.outdir.rmdir()
        return super().tearDown()


class HDPIComputationTest(unittest.TestCase):
    """Separate test class for HPDI computation edge cases"""

    def test_hpdi_different_methods(self):
        """Test HPDI computation with different methods"""
        np.random.seed(42)
        samples = np.random.multivariate_normal(
            mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=1000
        )

        # Test KDE method
        hpdi_kde, samples_kde = compute_hpdi_point(samples, method="kde")
        self.assertEqual(len(hpdi_kde), 2)

        # Test sklearn method
        hpdi_sklearn, samples_sklearn = compute_hpdi_point(samples, method="sklearn")
        self.assertEqual(len(hpdi_sklearn), 2)

        # Results should be similar (not identical due to different algorithms)
        np.testing.assert_allclose(hpdi_kde, hpdi_sklearn, rtol=0.3)

    def test_hpdi_different_prob_levels(self):
        """Test HPDI computation with different probability levels"""
        np.random.seed(42)
        samples = np.random.normal(0, 1, size=(1000, 1))

        # Test different probability levels
        for prob_level in [0.5, 0.68, 0.95, 0.99]:
            hpdi_point, hpdi_samples = compute_hpdi_point(
                samples, prob_level=prob_level
            )

            # Check that the number of HPDI samples corresponds roughly to prob_level
            expected_n_samples = int(prob_level * len(samples))
            actual_n_samples = len(hpdi_samples)

            # Allow some tolerance due to discrete sampling
            self.assertLess(
                abs(actual_n_samples - expected_n_samples), 0.1 * expected_n_samples
            )


if __name__ == "__main__":
    unittest.main()
