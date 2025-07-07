import torch

from ..utils.stats import expected_yule_tree_size
from .constrained_uniform import ConstrainedUniform


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

    def __init__(self, low, high, hyperparams=None, device=None):
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low, dtype=torch.float32)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high, dtype=torch.float32)

        # Dimension du prior est 2: [lda, mu]
        assert low.shape[-1] == 4 and high.shape[-1] == 4, (
            "Parameters must be 4-dimensional."
        )

        self.dimension = 4

        super().__init__(low, high, hyperparams, device)

    def _check_constraints(self, x):
        # Constraint 1: lda + gamma > mu (indices 1, 2, 3)
        constraint1 = x[..., 1] + x[..., 2] > x[..., 3]

        # Constraint 2: gamma < lda (indices 1, 2)
        constraint2 = x[..., 2] < x[..., 1]

        # Constraint 3: E[population of a tree at Nact] < max_pop
        constraint3 = (
            expected_yule_tree_size(
                x[..., 0],
                x[..., 1],
                x[..., 2],
                x[..., 3],
                self.hyperparams["n_init"],
                self.hyperparams["Nact"],
            )
            <= self.hyperparams["max_pop"]
        )

        # Constraint 4: E[population of a tree at Ninact] > 1
        constraint4 = (
            expected_yule_tree_size(
                x[..., 0],
                x[..., 1],
                x[..., 2],
                x[..., 3],
                self.hyperparams["n_init"],
                self.hyperparams["Nact"],
                self.hyperparams["Ninact"],
            )
            >= 1
        )

        return constraint1 & constraint2 & constraint3 & constraint4
