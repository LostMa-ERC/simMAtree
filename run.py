import numpy as np
import pandas as pd
import argparse
import json

from importlib import import_module
from utils.stats import compute_stat_witness, inverse_compute_stat_witness
from utils.evaluation import evaluate_inference


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

def generate_dataset(data_path, model, model_param = None, seed=42):
    rng = np.random.default_rng(seed)

    if model_param is None:
        parameters = model.sample_from_prior()
    else:
        parameters = model_param
    
    print("\nPARAMETERS_JSON_START")
    print(json.dumps(parameters))
    print("PARAMETERS_JSON_END")

    pop = model.simulate_pop(rng, list(parameters.values()))
    if pop == []:
        print("No survivors in the simulation!")
        return False
    if pop == "BREAK":
        print("The estimation hit the maximum size during simulation...")
        print("Estimation not saved.")
        return False
    
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

    print(f"Witness Number: {s[0]}")
    print(f"Works Number: {s[1]}")
    print(f"Max Witnesses: {s[2]}")
    print(f"Number of 1: {s[4]}")

    return True
    


def main():
    parser = argparse.ArgumentParser(description="run the inference or the generation method")

    parser.add_argument('--task', type=str, required=True, choices=['inference', 'generate', 'score'],
                       help='Choose the task')

    parser.add_argument('--data_path', type=str, required=False,
                      help='Chemin vers le fichier de données CSV')
    parser.add_argument('--model_config', type=str, required=True,
                      help='JSON file')
    parser.add_argument('--inference_config', type=str, required=False,
                      help='JSON file')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--results_dir', type=str, default="results/",
                      help='Dossier de sortie')
    parser.add_argument('--true_params', type=str, required=False,
                    help='Valeurs réelles des paramètres au format JSON (pour la tâche score)')

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
        generated = False
        seed = args.seed
        while not generated:
            seed = seed*10
            generated = generate_dataset(data_path=args.data_path,
                                         model=model,
                                         model_param=model_param["params"],
                                         seed=seed
                                        )
    elif args.task == 'score':
        if args.true_params is None:
            true_params = model_param["params"] 
        else:
            # Charger true_params depuis le fichier JSON fourni
            with open(args.true_params) as f:
                true_params = json.load(f)
        
        # Déterminer les noms de paramètres en fonction du modèle
        if model_param["class_name"] == "YuleModel":
            param_names = ["LDA", "lda", "gamma", "mu"]
        elif model_param["class_name"] == "BirthDeathPoisson":
            param_names = ["LDA", "lda", "mu"]
        else:
            param_names = list(true_params.keys())
        
        # Exécuter l'évaluation
        summary, param_summary = evaluate_inference(
            true_params=true_params if isinstance(true_params, list) else list(true_params.values()),
            results_dir=args.results_dir,
            param_names=param_names
        )
        
        # Afficher le résumé
        print("\n=== Summary of Evaluation Metrics ===")
        for metric, value in summary.items():
            print(f"{metric}: {value}")
        
        print("\n=== Parameter-Specific Metrics ===")
        for param, metrics in param_summary.items():
            print(f"\n{param}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
