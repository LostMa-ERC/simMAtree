import torch

from ..utils.survival_rate import expected_number_witness
from .constrained_uniform import ConstrainedUniform


class HybridConstrainedUniform3DPrior(ConstrainedUniform):
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
        assert low.shape[-1] == 3 and high.shape[-1] == 3, (
            "Parameters must be 3-dimensional."
        )

        self.dimension = 3

        super().__init__(low, high, hyperparams, device)

    def _check_constraints(self, x):
        # Constraint 1: lda > mu
        # constraint1 = x[..., 1] > x[..., 2]
        constraint1 = True

        expected_size_spec = expected_number_witness(
            x[..., 0],
            x[..., 1],
            0,
            x[..., 2],
            self.hyperparams["n_init"],
            self.hyperparams["Nino"],
        )

        constraint3 = (
            expected_number_witness(
                0,
                x[..., 1],
                0,
                x[..., 2],
                expected_size_spec,
                self.hyperparams["Nact"],
            )
            <= self.hyperparams["max_pop"]
        )

        # Constraint 4: E[population of a tree at Ninact] > 1
        constraint4 = (
            expected_number_witness(
                0,
                x[..., 1],
                0,
                x[..., 2],
                expected_size_spec,
                self.hyperparams["Nact"],
                self.hyperparams["Ninact"],
            )
            >= 1
        )

        return constraint1 & constraint3 & constraint4
