import torch

from .constrained_uniform import ConstrainedUniform


class BoxUniform4DPrior(ConstrainedUniform):
    # [lda, mu, decay, decim]
    def __init__(self, low, high, hyperparams=None, device=None):
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low, dtype=torch.float32)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high, dtype=torch.float32)

        assert low.shape[-1] == 4 and high.shape[-1] == 4, (
            "Parameters must be 4-dimensional."
        )

        self.dimension = 4

        super().__init__(low, high, hyperparams, device)

    def _check_constraints(self, x):
        # Constraint: lda > mu

        return x[..., 0] > x[..., 1]
