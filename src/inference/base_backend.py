from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.models import AbstractModelClass


class AbstractInferenceClass(ABC):
    """Classe abstraite définissant l'interface pour un backend d'inférence"""

    def __init__(self, random_seed: int):
        self.rng = np.random.default_rng(random_seed)

    @abstractmethod
    def run_inference(self, model: AbstractModelClass, data: np.ndarray):
        """Exécute l'inférence avec le backend spécifique"""
        pass

    @abstractmethod
    def save_results(self, observed_values: np.ndarray, output_dir: Path):
        """Sauvegarde les résultats de l'inférence"""
        pass

    @abstractmethod
    def plot_results(self, data, observed_values, output_dir: Path):
        """Visualise les résultats de l'inférence"""
        pass
