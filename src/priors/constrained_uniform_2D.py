import torch

from .constrained_uniform import ConstrainedUniform


class ConstrainedUniform2DPrior(ConstrainedUniform):
    """
    Uniform prior on lambda and mu, with the constraint :
        - lda > mu

    Args:
        low: lower bound on [lda, mu]
        high: upper bound on [lda, mu]
        device: tensor device
    """

    def __init__(self, low, high, hyperparams=None, device=None):
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low, dtype=torch.float32)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high, dtype=torch.float32)

        # Dimension du prior est 2: [lda, mu]
        assert low.shape[-1] == 2 and high.shape[-1] == 2, (
            "Parameters must be 2-dimensional."
        )

        super().__init__(low, high, hyperparams, device)

    def _check_constraints(self, x):
        # Constraint 1: lda > mu
        constraint1 = x[..., 0] > x[..., 1]

        # Constraint 2: E[population of a tree] < max_pop/n_init
        constraint2 = (
            torch.exp(
                (x[..., 0] - x[..., 1]) * self.hyperparams["Nact"]
                - x[..., 1] * self.hyperparams["Ninact"]
            )
            <= self.hyperparams["max_pop"] / self.hyperparams["n_init"]
        )

        return constraint1 & constraint2
