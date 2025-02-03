import numpy as np
import pandas as pd
import argparse
import json

from importlib import import_module
from utils.stats import compute_stat_witness, inverse_compute_stat_witness



def run(data_path, model, backend, results_dir):
    
    # Chargement des données
    df_data = pd.read_csv(data_path, sep=";")
    witness_counts = df_data.groupby('text_ID')['witness_ID'].count()
    data = compute_stat_witness(list(witness_counts))
    
    results = backend.run_inference(model, data)
    
    # Sauvegarde et visualisation
    obs_values = inverse_compute_stat_witness(data)
    backend.save_results(obs_values, results_dir)
    return results

def generate_dataset(data_path, model, model_param):
    rng = np.random.default_rng(42)

    pop = model.simulate_pop(rng, list(model_param.values()))
    if pop == []:
        print("No survivors in the simulation!")
        return 
    if pop == "BREAK":
        print("The estimation hit the maximum size during simulation...")
        print("Estimation not saved.")
        return
    
    text_val = []
    witness_val = []
    
    for i, num in enumerate(pop):
        for j in range(num):
            text_val.append(f'T{i}')
            witness_val.append(f'W{i}-{j+1}')
    
    # Créer le dataframe
    df = pd.DataFrame({
        'witness_ID': witness_val,
        'text_ID': text_val
    })
    
    df.to_csv(data_path, sep=";", index=False)

    witness_counts = list(df.groupby('text_ID')['witness_ID'].count().sort_values(ascending=False))
    s = inverse_compute_stat_witness(compute_stat_witness(witness_counts))

    print("DONE!\n")
    print(f"Witness Number: {s[0]}")
    print(f"Works Number: {s[1]}")
    print(f"Max Witnesses: {s[2]}")
    print(f"Number of 1: {s[4]}")


def main():
    parser = argparse.ArgumentParser(description="run the inference or the generation method")

    parser.add_argument('--task', type=str, required=True, choices=['inference', 'generate'],
                       help='Choose the task')

    parser.add_argument('--data_path', type=str, required=False,
                      help='Chemin vers le fichier de données CSV')
    parser.add_argument('--model_config', type=str, required=True,
                      help='JSON file')
    parser.add_argument('--inference_config', type=str, required=False,
                      help='JSON file')
    parser.add_argument('--results_dir', type=str, default="results/",
                      help='Dossier de sortie')
    
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_param = json.load(f)

    Model_module = import_module(model_param["module_name"])
    Model_class = getattr(Model_module, model_param["class_name"])
    try:
        model = Model_class(args.model_config)
    except:
        print("Unknown model type")
    
    if args.inference_config is not None:
        with open(args.inference_config) as f:
            inference_param = json.load(f)

        Inference_module = import_module(inference_param["module_name"])
        Inference_class = getattr(Inference_module, inference_param["class_name"])
        try:
            backend = Inference_class(args.inference_config)
        except:
            print("Unknown backend type")
    
    if args.task == 'inference':
        results = run(
                        data_path=args.data_path,
                        model=model,
                        backend=backend,
                        results_dir=args.results_dir
                    )
    elif args.task == 'generate':
        generate_dataset(
            data_path=args.data_path,
            model=model,
            model_param=model_param["params"]
        )

if __name__ == "__main__":
    main()
