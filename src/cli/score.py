from pathlib import Path

from src.utils.evaluation import evaluate_inference
from src.config.types import ExperimentParamters


def score(true_params: ExperimentParamters, results_dir: str):
    dir = Path(results_dir)
    dir.mkdir(exist_ok=True)

    # Déterminer les noms de paramètres en fonction du modèle
    param_dict = true_params.model_dump()
    param_names = list(param_dict.keys())

    # Exécuter l'évaluation
    summary, param_summary = evaluate_inference(
        true_params=param_dict,
        results_dir=dir,
        param_names=param_names,
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
