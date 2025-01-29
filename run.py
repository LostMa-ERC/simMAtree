import pandas as pd
import numpy as np
import argparse
import json

from utils.stats import compute_stat_witness, inverse_compute_stat_witness
from models.yule_model import YuleModel
from inference.pymc_backend import PymcBackend


def run(data_path, model_config, inference_config, results_dir):
    
    # Chargement des données
    df_data = pd.read_csv(data_path, sep=";")
    witness_counts = df_data.groupby('text_ID')['witness_ID'].count()
    data = compute_stat_witness(list(witness_counts))
    
    with open(model_config) as f:
        config_param = json.load(f)

    with open(inference_config) as f:
        inference_param = json.load(f)

    if config_param["type"] == "yule":
        model = YuleModel(model_config)
    else:
        raise ValueError(f"Unknown model type")
    
    if inference_param["backend"] == "pymc":
        backend = PymcBackend(inference_config)
    elif inference_param["backend"] == "sbi":
        pass
    else:
        raise ValueError(f"Unknown backend type")
    
    results = backend.run_inference(model, data)
    
    # Sauvegarde et visualisation
    backend.save_results(data, results_dir)
    # backend.plot_results(results, data, results_dir)

def main():
    parser = argparse.ArgumentParser(description='Exécuter l\'inférence sur les données')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Chemin vers le fichier de données CSV')
    parser.add_argument('--model_config', type=str, required=True,
                      help='JSON file')
    parser.add_argument('--inference_config', type=str, required=True,
                      help='JSON file')
    parser.add_argument('--results_dir', type=str, default="results/",
                      help='Dossier de sortie')
    
    args = parser.parse_args()
    
    run(
        data_path=args.data_path,
        model_config=args.model_config,
        inference_config=args.inference_config,
        results_dir=args.results_dir
    )

if __name__ == "__main__":
    main()
