"""
Unified Stemma Generator

Full 4-parameter model with temporal decay and decimation.
Inherits from GeneralizedStemmaGenerator and uses all parameters.
"""

from typing import Dict, Union

import numpy as np
import torch

from src.generator.generalized_stemma import GeneralizedStemmaGenerator


class UnifiedStemmaGenerator(GeneralizedStemmaGenerator):
    """
    Stemma generator with full model parameters

    Parameters:
    - lda: Initial birth rate (manuscript copying rate)
    - mu: Death rate (manuscript destruction rate)
    - decay: Temporal decay factor ∈ [0, 1]
    - decim: Decimation rate at crisis ∈ [0, 1]

    This is the most general model, implementing:
    - Time-varying birth rate with decay
    - Punctual decimation crisis
    - Continuous-time exponential events
    """

    def __init__(
        self,
        n_init: int = 1,
        Nact: int = 1000,
        Ninact: int = 1000,
        Ncrisis: int = 500,
        max_pop: int = 50000,
    ):
        """
        Initialize the unified stemma generator

        Parameters
        ----------
        n_init : int
            Number of initial manuscripts (typically 1)
        Nact : int
            Duration of active transmission phase
        Ninact : int
            Duration of inactive phase (pure death)
        Ncrisis : int
            Time when decimation crisis occurs
        max_pop : int
            Maximum population size to prevent memory issues
        """
        super().__init__(
            n_init=n_init,
            Nact=Nact,
            Ninact=Ninact,
            Ncrisis=Ncrisis,
            max_pop=max_pop,
        )
        self.param_count = 4

    def _extract_params(self, params: Union[list, tuple, dict]) -> Dict[str, float]:
        """
        Extract all four model parameters

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Model parameters [lda, mu, decay, decim] or dict

        Returns
        -------
        Dict[str, float]
            Dictionary with keys: lda, mu, decay, decim
        """
        if isinstance(params, torch.Tensor):
            lda = params[0].item()
            mu = params[1].item()
            decay = params[2].item()
            decim = params[3].item()
        elif isinstance(params, dict):
            lda = params.get("lda", params.get("lda", 0))
            mu = params.get("mu", 0)
            decay = params.get("decay", 0)
            decim = params.get("decim", 0)
        else:
            # List or tuple
            lda = params[0]
            mu = params[1]
            decay = params[2]
            decim = params[3]
            # Handle nested arrays
            try:
                if not isinstance(lda, float) and hasattr(lda, "__getitem__"):
                    lda = lda[0]
                    mu = mu[0]
                    decay = decay[0]
                    decim = decim[0]
            except Exception:
                pass

        return {"lda": lda, "mu": mu, "decay": decay, "decim": decim}

    def validate_params(self, params: Union[list, tuple, dict]) -> bool:
        """
        Validate all model parameters

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Parameters to validate

        Returns
        -------
        bool
            True if all parameters are valid
        """
        try:
            p = self._extract_params(params)

            # Check non-negativity and finiteness
            for key, value in p.items():
                if value < 0 or not np.isfinite(value):
                    return False

            # Check decay and decim are in [0, 1]
            if not (0 <= p["decay"] <= 1):
                return False
            if not (0 <= p["decim"] <= 1):
                return False

            return True

        except Exception:
            return False
