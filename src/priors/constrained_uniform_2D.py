import torch

from ..utils.survival_rate import expected_number_witness
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

        self.dimension = 2

        super().__init__(low, high, hyperparams, device)

    def _check_constraints(self, x):
        # Constraint 1: lda > mu
        constraint1 = x[..., 0] > x[..., 1]

        # Constraint 2: E[population of a tree] < max_pop/n_init
        constraint2 = (
            expected_number_witness(
                0,
                x[..., 0],
                0,
                x[..., 1],
                self.hyperparams["n_init"],
                self.hyperparams["Nact"],
            )
            <= self.hyperparams["max_pop"]
        )

        # Constraint 3: E[population of a tree at Ninact] > 1
        constraint3 = (
            expected_number_witness(
                LDA=0,
                lda=x[..., 0],
                gamma=0,
                mu=x[..., 1],
                n_init=self.hyperparams["n_init"],
                Nact=self.hyperparams["Nact"],
                Ninact=self.hyperparams["Ninact"],
            )
            >= 1
        )

        return constraint1 & constraint2 & constraint3
