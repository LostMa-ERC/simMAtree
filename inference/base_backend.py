import numpy as np
import json

class InferenceBackend:
    """Classe abstraite définissant l'interface pour un backend d'inférence"""
    
    def __init__(self, config_file):
        self.results = None
        with open(config_file) as f:
            self.inference_params = json.load(f)
        self.rng = np.random.default_rng(self.inference_params["random_seed"])

    def run_inference(self, 
                     model,
                     data,
                     config):
        """Exécute l'inférence avec le backend spécifique"""
        pass
    
    def save_results(self,
                    results,
                    output_dir):
        """Sauvegarde les résultats de l'inférence"""
        pass
    
    def plot_results(self,
                    results,
                    true_values,
                    output_dir):
        """Visualise les résultats de l'inférence"""
        pass