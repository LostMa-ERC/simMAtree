import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import xarray as xr
from utils.visualisation import plot_posterior_predictive_stats, plot_marginal_posterior, plot_combined_hpdi

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
        posterior_mean = results_summary['mean']
        
    except (FileNotFoundError, KeyError):
        print(f"Erreur: Impossible de trouver les valeurs HPDI dans {results_dir}/results_summary.csv")
        return None
    
    try:
        obs_values = pd.read_csv(f"{results_dir}/obs_values.npy")
        
    except:
        obs_values = None
    
    hpdi_point = np.array(hpdi_values)
    true_params = np.array(true_params)
    
    errors = hpdi_point - true_params
    squared_errors = errors**2
    
    # Calculer la coverage probability
    # Charger les échantillons postérieurs
    try:
        posterior_samples = np.load(f"{results_dir}/posterior_samples.npy")
        posterior_pred_samples = np.load(f"{results_dir}/posterior_predictive.npy")
        pp_samples_xr = xr.DataArray(posterior_pred_samples,
                                    dims=["sample", "stat"])
        
        # Calculer les bornes HPDI à 95%
        # Trions les échantillons par densité décroissante (normalement déjà fait pour HPDI)
        coverage = []
        hpdi_interval = []

        for i, true_val in enumerate(true_params):
            # Calculer les quantiles 2.5% et 97.5% pour une approximation simple de l'HPDI
            lower, upper = np.percentile(posterior_samples[:, i], [2.5, 97.5])
            in_interval = (true_val >= lower) and (true_val <= upper)
            coverage.append(in_interval)
            hpdi_interval.append((lower, upper))
        
        coverage_prob = np.mean(coverage)
        coverage_by_param = coverage
    except:
        coverage_prob = np.nan
        coverage_by_param = [np.nan] * len(true_params)

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
            "post. mean": posterior_mean[i],
            "rel_error_pct": rel_error,
            "norm_squared_error": norm_squared_error,
            "bias": errors[i],
            "hpdi_interval": hpdi_interval[i],
            "in_hpdi_95": coverage_by_param[i]
        }

    # Calculer les métriques agrégées basées sur les erreurs relatives
    mean_rel_error = np.mean([m["rel_error_pct"] for m in param_metrics.values() if m["rel_error_pct"] != float('inf')])
    rmse = np.sqrt(np.mean(squared_errors))  # Conserver RMSE original
    nrmse = np.sqrt(np.mean([m["norm_squared_error"] for m in param_metrics.values() if m["norm_squared_error"] != float('inf')]))

    # Résumé des métriques
    summary = {
        "rmse": rmse,
        "nrmse": nrmse,  # RMSE normalisé
        "mean_rel_error_pct": mean_rel_error,
        "coverage_probability": coverage_prob
    }
    
    # Créer un rapport
    if not Path(f"{results_dir}").exists():
        Path(f"{results_dir}").mkdir(parents=True)
    
    # Sauvegarder les métriques
    pd.DataFrame([summary]).to_csv(f"{results_dir}/summary_metrics.csv", index=False)

    plt.figure(figsize=(10, 6))
    rel_errors = [param_metrics[name]["rel_error_pct"] for name in param_names if param_metrics[name]["rel_error_pct"] != float('inf')]
    rel_error_names = [name for name in param_names if param_metrics[name]["rel_error_pct"] != float('inf')]
    plt.bar(rel_error_names, rel_errors)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Parameters')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error of HPDI Point Estimates')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/relative_error.png")
    plt.close()


    plot_posterior_predictive_stats(
        pp_samples_xr,
        obs_value = obs_values,
        output_dir = results_dir
    )

    plot_combined_hpdi(
        [posterior_samples], 
        output_dir = results_dir, 
        true_values = true_params
    )
    
    plot_marginal_posterior(
        posterior_samples, 
        output_dir = results_dir
    )
    
    return summary, param_metrics

