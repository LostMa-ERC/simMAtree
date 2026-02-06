from typing import Dict, Union

import numpy as np
import torch

from src.generator.generalized_stemma import GeneralizedStemmaGenerator


class BirthDeathStemmaGenerator(GeneralizedStemmaGenerator):
    """
    Stemma generator for simple birth-death model

    Parameters:
    - lda: Birth rate (manuscript copying rate)
    - mu: Death rate (manuscript destruction rate)

    Fixed parameters:
    - decay: 0 (constant birth rate)
    - decim: 0 (no decimation crisis)

    Constraint: lda > mu (for non-trivial dynamics)
    """

    def __init__(
        self,
        n_init: int = 1,
        Nact: int = 1000,
        Ninact: int = 1000,
        max_pop: int = 50000,
    ):
        """
        Initialize the birth-death stemma generator

        Parameters
        ----------
        n_init : int
            Number of initial manuscripts (typically 1)
        Nact : int
            Duration of active transmission phase
        Ninact : int
            Duration of inactive phase (pure death)
        Ncrisis : int
            Time when decimation would occur (unused, kept for compatibility)
        max_pop : int
            Maximum population size to prevent memory issues
        """
        super().__init__(
            n_init=n_init,
            Nact=Nact,
            Ninact=Ninact,
            Ncrisis=Nact + Ninact,
            max_pop=max_pop,
        )
        self.param_count = 2

    def _extract_params(self, params: Union[list, tuple, dict]) -> Dict[str, float]:
        """
        Extract parameters and set decay=0, decim=0

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Model parameters [lda, mu] or {"lda": ..., "mu": ...}

        Returns
        -------
        Dict[str, float]
            Dictionary with keys: lda, mu, decay=0, decim=0
        """
        if isinstance(params, torch.Tensor):
            lda = params[0].item()
            mu = params[1].item()
        elif isinstance(params, dict):
            lda = params.get("lda", 0)
            mu = params.get("mu", 0)
        else:
            # List or tuple
            lda = params[0]
            mu = params[1]
            # Handle nested arrays
            try:
                if not isinstance(lda, float) and hasattr(lda, "__getitem__"):
                    lda = lda[0]
                    mu = mu[0]
            except Exception:
                pass

        # Return with decay and decim fixed to 0
        return {"lda": lda, "mu": mu, "decay": 0.0, "decim": 0.0}

    def validate_params(self, params: Union[list, tuple, dict]) -> bool:
        """
        Validate Birth-Death parameters

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Parameters to validate

        Returns
        -------
        bool
            True if lda >= 0, mu >= 0, and lda > mu
        """
        try:
            extracted = self._extract_params(params)
            lda = extracted["lda"]
            mu = extracted["mu"]

            # Check non-negativity and finiteness
            if lda < 0 or mu < 0:
                return False
            if not (np.isfinite(lda) and np.isfinite(mu)):
                return False

            # Constraint: lda > mu for non-trivial dynamics
            return lda > mu

        except Exception:
            return False
