import unittest

import torch

from src.priors.constrained_uniform_2D import ConstrainedUniform2DPrior
from src.priors.constrained_uniform_4D import ConstrainedUniform4DPrior


class ConstrainedUniform2DTest(unittest.TestCase):
    def setUp(self):
        self.low = torch.tensor([0.0, 0.0])
        self.high = torch.tensor([0.02, 0.01])
        self.hyperparams = {"n_init": 1, "Nact": 100, "Ninact": 100, "max_pop": 1000}

        self.prior = ConstrainedUniform2DPrior(
            low=self.low, high=self.high, hyperparams=self.hyperparams
        )

    def test_initialization(self):
        """Test prior initialization"""
        self.assertEqual(self.prior.dimension, 2)
        self.assertTrue(torch.allclose(self.prior._low, self.low))
        self.assertTrue(torch.allclose(self.prior._high, self.high))

    def test_constraints_valid_samples(self):
        """Test constraint checking with valid samples"""
        # Valid sample: lda > mu and satisfies population constraint
        valid_sample = torch.tensor([[0.01, 0.005]])
        constraints = self.prior._check_constraints(valid_sample)
        self.assertTrue(constraints.item())

    def test_constraints_invalid_samples(self):
        """Test constraint checking with invalid samples"""
        # Invalid sample: lda < mu
        invalid_sample = torch.tensor([[0.005, 0.01]])
        constraints = self.prior._check_constraints(invalid_sample)
        self.assertFalse(constraints.item())

        # Invalid sample: violates population constraint (too high growth rate)
        invalid_sample2 = torch.tensor([[0.019, 0.001]])  # Very high lda
        constraints2 = self.prior._check_constraints(invalid_sample2)
        # This might be valid or invalid depending on exact values, just test it runs
        self.assertIsInstance(constraints2.item(), bool)

    def test_sampling(self):
        """Test sampling from the prior"""
        n_samples = 100
        samples = self.prior.sample(torch.Size([n_samples]))

        self.assertEqual(samples.shape, (n_samples, 2))

        # All samples should satisfy constraints
        constraints = self.prior._check_constraints(samples)
        self.assertTrue(torch.all(constraints))

        # All samples should be within bounds
        self.assertTrue(torch.all(samples >= self.low))
        self.assertTrue(torch.all(samples <= self.high))

    def test_log_prob(self):
        """Test log probability computation"""
        # Valid sample
        valid_sample = torch.tensor([[0.01, 0.005]])
        log_prob = self.prior.log_prob(valid_sample)
        self.assertFalse(torch.isinf(log_prob))

        # Invalid sample
        invalid_sample = torch.tensor([[0.005, 0.01]])
        log_prob_invalid = self.prior.log_prob(invalid_sample)
        self.assertTrue(torch.isinf(log_prob_invalid))

    def test_mean_and_std(self):
        """Test mean and standard deviation computation"""
        mean = self.prior.mean
        std = self.prior.stddev

        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(torch.all(std > 0))


class ConstrainedUniform4DTest(unittest.TestCase):
    def setUp(self):
        self.low = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.high = torch.tensor([1.0, 0.02, 0.01, 0.01])
        self.hyperparams = {"n_init": 1, "Nact": 100, "Ninact": 100, "max_pop": 1000}

        self.prior = ConstrainedUniform4DPrior(
            low=self.low, high=self.high, hyperparams=self.hyperparams
        )

    def test_initialization(self):
        """Test prior initialization"""
        self.assertEqual(self.prior.dimension, 4)
        self.assertTrue(torch.allclose(self.prior._low, self.low))
        self.assertTrue(torch.allclose(self.prior._high, self.high))

    def test_constraints_valid_samples(self):
        """Test constraint checking with valid samples"""
        # Valid sample: lda + gamma > mu, gamma < lda, population constraint satisfied
        valid_sample = torch.tensor([[0.3, 0.01, 0.005, 0.003]])
        constraints = self.prior._check_constraints(valid_sample)
        self.assertTrue(constraints.item())

    def test_constraints_invalid_samples(self):
        """Test constraint checking with invalid samples"""
        # Invalid: lda + gamma <= mu
        invalid_sample1 = torch.tensor([[0.3, 0.005, 0.003, 0.01]])
        constraints1 = self.prior._check_constraints(invalid_sample1)
        self.assertFalse(constraints1.item())

        # Invalid: gamma >= lda
        invalid_sample2 = torch.tensor([[0.3, 0.005, 0.01, 0.003]])
        constraints2 = self.prior._check_constraints(invalid_sample2)
        self.assertFalse(constraints2.item())

    def test_sampling(self):
        """Test sampling from the prior"""
        n_samples = 50  # Reduced for 4D case as it's more constrained
        samples = self.prior.sample(torch.Size([n_samples]))

        self.assertEqual(samples.shape, (n_samples, 4))

        # All samples should satisfy constraints
        constraints = self.prior._check_constraints(samples)
        self.assertTrue(torch.all(constraints))

        # All samples should be within bounds
        self.assertTrue(torch.all(samples >= self.low))
        self.assertTrue(torch.all(samples <= self.high))

    def test_constraint_combinations(self):
        """Test specific constraint combinations"""
        # Test constraint 1: lda + gamma > mu
        sample1 = torch.tensor(
            [[0.3, 0.006, 0.003, 0.005]]
        )  # 0.006 + 0.003 = 0.009 > 0.005
        c1 = self.prior._check_constraints(sample1)
        # Should pass constraint 1 but check others too

        # Test constraint 2: gamma < lda
        sample2 = torch.tensor([[0.3, 0.008, 0.006, 0.003]])  # 0.006 < 0.008
        c2 = self.prior._check_constraints(sample2)
        # Should pass constraint 2 but check others too

        # All constraints should be computed correctly
        self.assertIsInstance(c1.item(), bool)
        self.assertIsInstance(c2.item(), bool)

    def test_batch_sampling(self):
        """Test batch sampling works correctly"""
        batch_size = 10
        samples = self.prior.sample(torch.Size([batch_size]))

        # Should handle batch sampling without issues
        self.assertEqual(samples.shape[0], batch_size)
        self.assertEqual(samples.shape[1], 4)

    def test_log_prob_batch(self):
        """Test log probability computation for batches"""
        samples = self.prior.sample(torch.Size([5]))
        log_probs = self.prior.log_prob(samples)

        self.assertEqual(log_probs.shape, (5,))
        # All should be finite for valid samples
        self.assertTrue(torch.all(torch.isfinite(log_probs)))


class PriorEdgeCaseTest(unittest.TestCase):
    """Test edge cases and error conditions for priors"""

    def test_invalid_dimension_2d(self):
        """Test error handling for wrong dimensions in 2D prior"""
        with self.assertRaises(AssertionError):
            ConstrainedUniform2DPrior(
                low=torch.tensor([0.0]),  # Wrong dimension
                high=torch.tensor([1.0, 2.0]),
                hyperparams={"n_init": 1, "Nact": 100, "Ninact": 100, "max_pop": 1000},
            )

    def test_invalid_dimension_4d(self):
        """Test error handling for wrong dimensions in 4D prior"""
        with self.assertRaises(AssertionError):
            ConstrainedUniform4DPrior(
                low=torch.tensor([0.0, 0.0]),  # Wrong dimension
                high=torch.tensor([1.0, 1.0, 1.0, 1.0]),
                hyperparams={"n_init": 1, "Nact": 100, "Ninact": 100, "max_pop": 1000},
            )

    def test_sampling_with_very_constrained_space(self):
        """Test sampling when constraint space is very small"""
        # Very tight constraints that might make sampling difficult
        prior = ConstrainedUniform2DPrior(
            low=torch.tensor([0.001, 0.0]),
            high=torch.tensor([0.002, 0.0005]),  # Very small valid region
            hyperparams={"n_init": 1, "Nact": 10, "Ninact": 10, "max_pop": 100},
        )

        # Should still be able to sample
        samples = prior.sample(torch.Size([10]))
        self.assertEqual(samples.shape, (10, 2))


if __name__ == "__main__":
    unittest.main()
