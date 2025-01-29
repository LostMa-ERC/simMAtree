import pandas as pd
import argparse
import json

from importlib import import_module
from utils.stats import compute_stat_witness, inverse_compute_stat_witness
from inference.pymc_backend import PymcBackend


def run(data_path, model_config, inference_config, results_dir):
    
    # Chargement des données
    df_data = pd.read_csv(data_path, sep=";")
    witness_counts = df_data.groupby('text_ID')['witness_ID'].count()
    data = compute_stat_witness(list(witness_counts))
    
    with open(model_config) as f:
        model_param = json.load(f)

    with open(inference_config) as f:
        inference_param = json.load(f)


    Model_module = import_module(model_param["module_name"])
    Model_class = getattr(Model_module, model_param["class_name"])
    try:
        model = Model_class(model_config)
    except:
        print("Unknown model type")
    
    Inference_module = import_module(inference_param["module_name"])
    Inference_class = getattr(Inference_module, inference_param["class_name"])
    try:
        backend = Inference_class(inference_config)
    except:
        print("Unknown backend type")
    
    results = backend.run_inference(model, data)
    
    # Sauvegarde et visualisation
    obs_values = inverse_compute_stat_witness(data)
    backend.save_results(obs_values, results_dir)

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
