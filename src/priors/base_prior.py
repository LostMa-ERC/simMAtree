from abc import abstractmethod

import torch
from torch.distributions import Distribution


class BasePrior(Distribution):
    def __init__(self, params, batch_shape, event_shape, hyperparams=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        for p in params:
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p, dtype=torch.float32, device=device)
            else:
                p = p.to(device)

        self.hyperparams = hyperparams
        self.params = params

        super().__init__(batch_shape, event_shape)

    @property
    def support(self):
        pass

    @property
    def mean(self):
        pass

    @property
    def stddev(self):
        pass

    def _check_constraints(self, x):
        pass

    @abstractmethod
    def sample(self, sample_shape=torch.Size()):
        pass
