from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class AbstractModelClass(ABC):
    """Classe abstraite pour définir l'interface d'un modèle"""

    def __init__(
        self,
        n_init: int,
        Nact: int,
        Ninact: int,
        max_pop: int,
        LDA: float = None,
        lda: float = None,
        gamma: float = None,
        mu: float = None,
    ):
        # Model configuration
        self.n_init = n_init
        self.Nact = Nact
        self.Ninact = Ninact
        self.max_pop = max_pop

        # Optional parameters
        self.LDA = LDA
        self.lda = lda
        self.gamma = gamma
        self.mu = mu

    @abstractmethod
    def get_simulator(
        self, rng: np.random.default_rng, params: Union[list, tuple, dict]
    ):
        """
        Définit et retourne les priors du modèle
        """
        pass

    def sample_from_prior(self, n_samples=1, device="cpu", param_names=None):
        if not hasattr(self, "get_sbi_priors"):
            raise NotImplementedError("La méthode get_sbi_priors doit être implémentée")

        prior = self.get_sbi_priors(device=device)
        sample_shape = torch.Size([n_samples])
        samples = prior.sample(sample_shape)

        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()

        if param_names is None:
            param_names = [f"param_{i}" for i in range(samples.shape[1])]

        json_samples = []
        for i in range(n_samples):
            sample_dict = {
                name: float(value) for name, value in zip(param_names, samples[i])
            }
            json_samples.append(sample_dict)

        return json_samples
