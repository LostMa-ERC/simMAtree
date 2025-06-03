from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np


class BaseGenerator(ABC):
    """
    Abstract class for a generation of a population (typically trees/forest, etc)
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def generate(
        self,
        rng: np.random.Generator,
        params: Union[list, tuple, dict],
        hyperparams: Union[list, tuple, dict],
    ) -> Any:
        """
        Generate a population according to a model

        Parameters
        ----------
        rng : np.random.Generator
            Random generator
        params : Union[list, tuple, dict]
            Parameters of the model (lda, mu, gamma, etc.)
        hyperparams : Union[list, tuple, dict]
            Parameters of the model (lda, mu, gamma, etc.)

        Returns
        -------
        Any
            Generated structure
        """
        pass

    @abstractmethod
    def validate_params(self, params: Union[list, tuple, dict]) -> bool:
        """
        Validate the parameters are valid for the generator

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Parameters to validate

        Returns
        -------
        bool
            True if parameter are valid
        """
        pass

    @abstractmethod
    def _extract_params(self, params: Union[list, tuple, dict]) -> tuple:
        """
        Extract parameter in a standard format.

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Parameters to extract

        Returns
        -------
        tuple
            the tuple of parameters
        """
        pass
