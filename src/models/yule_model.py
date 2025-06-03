import numpy as np
import torch
from sbi.utils.user_input_checks import process_prior

from src.generator.yule_witness import YuleWitness
from src.utils.stats import compute_stat_witness

from .base_model import AbstractModelClass
from .base_prior import ConstrainedUniform


class ConstrainedUniform4DPrior(ConstrainedUniform):
    """
    Uniform prior on LDA, lda, gamma, mu, with the constraints :
        - lda + gamma > mu
        - gamma < lda
        - The expected yule population of a tree lower than max_pop/n_init

    Args:
        low: lower bound on [LDA, lda, gamma, mu]
        high: upper bound on [LDA, lda, gamma, mu]
        device: tensor device
    """

    def __init__(self, low, high, constraints_params=None, device=None):
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low, dtype=torch.float32)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high, dtype=torch.float32)

        # Dimension du prior est 2: [lda, mu]
        assert low.shape[-1] == 4 and high.shape[-1] == 4, (
            "Parameters must be 4-dimensional."
        )

        super().__init__(low, high, constraints_params, device)

    def _check_constraints(self, x):
        # Contrainte 1: lda + gamma > mu (indices 1, 2, 3)
        constraint1 = x[..., 1] + x[..., 2] > x[..., 3]

        # Contrainte 2: gamma < lda (indices 1, 2)
        constraint2 = x[..., 2] < x[..., 1]

        # Contrainte 3: E[population d'un arbre] < max_pop
        constraint3 = (
            torch.exp(
                (x[..., 1] + x[..., 2] - x[..., 3]) * self.constraints_params[1]
                - x[..., 3] * self.constraints_params[2]
            )
            <= self.constraints_params[3] / self.constraints_params[0]
        )

        return constraint1 & constraint2 & constraint3


class YuleModel(AbstractModelClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator = YuleWitness(self.n_init, self.Nact, self.Ninact, self.max_pop)

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
        # LDA, lda, gamma, mu
        lower_bounds = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
        upper_bounds = torch.tensor([2, 0.015, 0.01, 0.01], device=device)

        prior = ConstrainedUniform4DPrior(
            lower_bounds,
            upper_bounds,
            constraints_params=[self.n_init, self.Nact, self.Ninact, self.max_pop],
            device=device,
        )
        prior, _, _ = process_prior(prior)
        return prior
