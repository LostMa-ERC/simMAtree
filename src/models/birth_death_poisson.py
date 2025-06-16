import numpy as np
import torch
from sbi.utils.user_input_checks import process_prior

from src.generator.birth_death_witness import BirthDeathWitness
from src.utils.stats import compute_stat_witness

from .base_model import AbstractModelClass
from .base_prior import ConstrainedUniform


class ConstrainedUniform2DPrior(ConstrainedUniform):
    """
    Uniform prior on lambda and mu, with the constraint :
        - lda > mu

    Args:
        low: lower bound on [lda, mu]
        high: upper bound on [lda, mu]
        device: tensor device
    """

    def __init__(self, low, high, constraints_params=None, device=None):
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low, dtype=torch.float32)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high, dtype=torch.float32)

        # Dimension du prior est 2: [lda, mu]
        assert low.shape[-1] == 2 and high.shape[-1] == 2, (
            "Parameters must be 2-dimensional."
        )

        super().__init__(low, high, constraints_params, device)

    def _check_constraints(self, x):
        # Constraint 1: lda > mu
        constraint1 = x[..., 0] > x[..., 1]

        # Constraint 2: E[population of a tree] < max_pop/n_init
        constraint2 = (
            torch.exp(
                (x[..., 0] - x[..., 1]) * self.constraints_params[1]
                - x[..., 1] * self.constraints_params[2]
            )
            <= self.constraints_params[3] / self.constraints_params[0]
        )

        return constraint1 & constraint2


class BirthDeath(AbstractModelClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator = BirthDeathWitness(
            self.n_init, self.Nact, self.Ninact, self.max_pop
        )

    def get_simulator(self, rng, params, size=None, additional_stats=True):
        # Required method, inherited from AbstractBaseClass
        witness_nb = self.generator.generate(rng, params)
        try:
            if not witness_nb:
                return np.zeros(13) if additional_stats else np.zeros(6)
            stats = compute_stat_witness(witness_nb, additional_stats)
            if not np.all(np.isfinite(stats)):
                return np.zeros(13) if additional_stats else np.zeros(6)
            return stats
        except Exception as e:
            print(f"Error in simulate_tree_stats: {e}")
            print()
            return np.zeros(6)

    def get_sbi_priors(self, device="cpu"):
        lower_bounds = torch.tensor([0.0, 0.0], device=device)
        upper_bounds = torch.tensor([0.005, 0.005], device=device)

        prior = ConstrainedUniform2DPrior(
            lower_bounds,
            upper_bounds,
            constraints_params=[self.n_init, self.Nact, self.Ninact, self.max_pop],
            device=device,
        )
        prior, _, _ = process_prior(prior)
        return prior
