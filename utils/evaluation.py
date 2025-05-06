import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def evaluate_inference(true_params, results_dir, param_names=None):
    """
    Évalue les performances d'inférence en comparant les valeurs HPDI avec les vrais paramètres.
    
    Parameters:
    ----------
    true_params : list ou dict
        Valeurs réelles des paramètres à comparer
    results_dir : str
        Répertoire contenant les résultats d'inférence
    param_names : list, optional
        Noms des paramètres pour l'affichage
    
    Returns:
    -------
    dict
        Dictionnaire contenant les métriques d'évaluation
    """
    # Convertir true_params en liste si c'est un dictionnaire
    if isinstance(true_params, dict):
        if param_names is None:
            param_names = list(true_params.keys())
        true_params = list(true_params.values())
    
    if param_names is None:
        param_names = [f"param_{i}" for i in range(len(true_params))]
    
    # Charger le résumé des résultats qui contient hpdi_point
    try:
        results_summary = pd.read_csv(f"{results_dir}/results_summary.csv")
        hpdi_values = results_summary['hpdi_95%']
        
    except (FileNotFoundError, KeyError):
        print(f"Erreur: Impossible de trouver les valeurs HPDI dans {results_dir}/results_summary.csv")
        return None
    
    hpdi_point = np.array(hpdi_values)
    true_params = np.array(true_params)
    
    errors = hpdi_point - true_params
    squared_errors = errors**2
    
    param_metrics = {}
    for i, (name, true_val) in enumerate(zip(param_names, true_params)):
        # Erreur absolue
        abs_error = abs(errors[i])
        # Erreur relative (en pourcentage)
        rel_error = abs_error / abs(true_val) * 100 if true_val != 0 else float('inf')
        # Erreur quadratique normalisée
        norm_squared_error = squared_errors[i] / (true_val**2) if true_val != 0 else squared_errors[i]
        
        param_metrics[name] = {
            "true_value": true_val,
            "hpdi_point": hpdi_point[i],
            "abs_error": abs_error,
            "rel_error_pct": rel_error,
            "norm_squared_error": norm_squared_error,
            "bias": errors[i],
            "in_hpdi_95": coverage_by_param[i]
        }

    # Calculer les métriques agrégées basées sur les erreurs relatives
    mean_abs_error = np.mean([m["abs_error"] for m in param_metrics.values()])
    mean_rel_error = np.mean([m["rel_error_pct"] for m in param_metrics.values() if m["rel_error_pct"] != float('inf')])
    rmse = np.sqrt(np.mean(squared_errors))  # Conserver RMSE original
    nrmse = np.sqrt(np.mean([m["norm_squared_error"] for m in param_metrics.values() if m["norm_squared_error"] != float('inf')]))

    # Résumé des métriques
    summary = {
        "rmse": rmse,
        "nrmse": nrmse,  # RMSE normalisé
        "mean_abs_error": mean_abs_error,
        "mean_rel_error_pct": mean_rel_error,
        "coverage_probability": coverage_prob
    }
    
    # Calculer la coverage probability
    # Charger les échantillons postérieurs
    try:
        posterior_samples = np.load(f"{results_dir}/posterior_samples.npy")
        
        # Calculer les bornes HPDI à 95%
        # Trions les échantillons par densité décroissante (normalement déjà fait pour HPDI)
        coverage = []
        
        for i, true_val in enumerate(true_params):
            # Calculer les quantiles 2.5% et 97.5% pour une approximation simple de l'HPDI
            lower, upper = np.percentile(posterior_samples[:, i], [2.5, 97.5])
            in_interval = (true_val >= lower) and (true_val <= upper)
            coverage.append(in_interval)
        
        coverage_prob = np.mean(coverage)
        coverage_by_param = coverage
    except:
        coverage_prob = np.nan
        coverage_by_param = [np.nan] * len(true_params)

    
    # Créer un rapport
    if not Path(f"{results_dir}/evaluation").exists():
        Path(f"{results_dir}/evaluation").mkdir(parents=True)
    
    # Sauvegarder les métriques
    pd.DataFrame([summary]).to_csv(f"{results_dir}/evaluation/summary_metrics.csv", index=False)
    
    # Créer un graphique de comparaison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(param_names))
    width = 0.35
    
    plt.bar(x - width/2, true_params, width, label='True Values')
    plt.bar(x + width/2, hpdi_point, width, label='HPDI Point')
    
    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.title('Comparison of True Parameters vs HPDI Point Estimates')
    plt.xticks(x, param_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/evaluation/param_comparison.png")
    plt.close()
    
    # Créer un graphique de barres pour le biais
    plt.figure(figsize=(10, 6))
    plt.bar(param_names, errors)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Parameters')
    plt.ylabel('Bias')
    plt.title('Bias of HPDI Point Estimates')
        
    plt.figure(figsize=(10, 6))
    rel_errors = [param_metrics[name]["rel_error_pct"] for name in param_names if param_metrics[name]["rel_error_pct"] != float('inf')]
    rel_error_names = [name for name in param_names if param_metrics[name]["rel_error_pct"] != float('inf')]
    plt.bar(rel_error_names, rel_errors)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Parameters')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error of HPDI Point Estimates')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/evaluation/relative_error.png")
    plt.close()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/evaluation/bias.png")
    plt.close()
    
    return summary, param_metrics
