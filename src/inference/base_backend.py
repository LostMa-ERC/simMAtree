from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from torch.distributions import Distribution

from src.generator import BaseGenerator
from src.stats import AbstractStatsClass


class AbstractInferenceClass(ABC):
    """
    Abstract class to define an inference backend
    """

    def __init__(self, random_seed: int):
        self.rng = np.random.default_rng(random_seed)

    @abstractmethod
    def run_inference(
        self,
        generator: BaseGenerator,
        stats: AbstractStatsClass,
        data: np.ndarray,
        prior: Distribution,
    ):
        """
        Run the inference
        """
        pass

    @abstractmethod
    def save_results(self, observed_values: np.ndarray, output_dir: Path):
        """
        Save the results of the inference
        """
        pass

    @abstractmethod
    def plot_results(self, data, observed_values, output_dir: Path):
        """
        Visualize the results of the inference
        """
        pass

    def save_model(self, output_dir: Path):
        """
        Save the trained model to the specified directory

        Parameters
        ----------
        output_dir : Path
            Directory where to save the model
        """
        pass
