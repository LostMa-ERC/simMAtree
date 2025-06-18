import torch

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
        # Contrainte 1: lda + gamma > mu (indices 1, 2, 3)
        constraint1 = x[..., 1] + x[..., 2] > x[..., 3]

        # Contrainte 2: gamma < lda (indices 1, 2)
        constraint2 = x[..., 2] < x[..., 1]

        # Contrainte 3: E[population d'un arbre] < max_pop
        constraint3 = (
            torch.exp(
                (x[..., 1] + x[..., 2] - x[..., 3]) * self.hyperparams["Nact"]
                - x[..., 3] * self.hyperparams["Ninact"]
            )
            <= self.hyperparams["max_pop"] / self.hyperparams["n_init"]
        )

        return constraint1 & constraint2 & constraint3
