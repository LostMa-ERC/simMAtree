import numpy as np


class InferenceBackend:
    """Classe abstraite définissant l'interface pour un backend d'inférence"""

    def __init__(self, random_seed: int):
        self.results = None
        self.rng = np.random.default_rng(random_seed)

    def run_inference(self, model, data, config):
        """Exécute l'inférence avec le backend spécifique"""
        pass

    def save_results(self, results, output_dir):
        """Sauvegarde les résultats de l'inférence"""
        pass

    def plot_results(self, results, true_values, output_dir):
        """Visualise les résultats de l'inférence"""
        pass
