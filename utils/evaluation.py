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
    
    metrics = {
        "parameter": param_names,
        "true_value": true_params,
        "hpdi_point": hpdi_point,
        "bias": errors,
        "squared_error": squared_errors
    }
    
    # Calculer les métriques agrégées
    rmse = np.sqrt(np.mean(squared_errors))
    bias = np.mean(errors)
    std_dev = np.std(errors)
    
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
    
    metrics["in_hpdi_95"] = coverage_by_param
    
    # Résumé des métriques
    summary = {
        "rmse": rmse,
        "bias": bias,
        "std_dev": std_dev,
        "coverage_probability": coverage_prob
    }
    
    # Ajouter un résumé par paramètre
    param_summary = {}
    for i, name in enumerate(param_names):
        param_summary[name] = {
            "true_value": true_params[i],
            "hpdi_point": hpdi_point[i],
            "bias": errors[i],
            "squared_error": squared_errors[i],
            "in_hpdi_95": coverage_by_param[i]
        }
    
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
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/evaluation/bias.png")
    plt.close()
    
    return summary, param_summary
