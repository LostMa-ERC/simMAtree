import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import sys
import gc
import argparse
from src.stats import compute_stat_witness, inverse_compute_stat_witness
from src.yule import simulate_tree_stats
from src.visualisation import plot_posterior_predictive_stats, plot_inference_checks

HYPERPARAMS = {
    "n_init" : 6000,
    "Nact" : 1000,
    "Ninact" : 1000,
    "max_pop" : 100000
}

def main():
    # Parser les arguments
    parser = argparse.ArgumentParser(description='Exécuter l\'inférence sur les données de témoins')
    parser.add_argument('--data_path', type=str, required=True, help='Chemin vers le fichier de données CSV')
    parser.add_argument('--draws', type=int, default=10, help='Nombre d\'échantillons')
    parser.add_argument('--chains', type=int, default=4, help='Nombre de chaînes')
    parser.add_argument('--results_dir', type=str, default="results/", help='Chemin vers le dossier où sauvegarder les résultats')
    args = parser.parse_args()

    # Configuration
    sys.setrecursionlimit(20000)
    rng = np.random.default_rng(42)

    # Chargement des données
    df_data = pd.read_csv(args.data_path, sep=";")
    witness_counts = df_data.groupby('text_ID')['witness_ID'].count().sort_values(ascending=False)
    data = compute_stat_witness(list(witness_counts))

    print("Démarrage de l'inférence...")
    gc.collect()

    # Configuration du modèle
    with pm.Model() as model:
        # Priors
        LDA = pm.Uniform('LDA', lower=0, upper=0.01)
        lda = pm.Uniform('lda', lower=0, upper=0.05)
        gamma = pm.Uniform('gamma', lower=0, upper=0.01)
        mu = pm.Uniform('mu', lower=0, upper=0.01)

        # Contraintes
        constraints = pm.Potential('constraints',
            pm.math.switch((lda + gamma > mu) & (gamma < lda), 0, -np.inf))

        # Simulateur
        s = pm.Simulator("s",
            lambda rng, LDA, lda, gamma, mu, size : simulate_tree_stats(rng, LDA, lda, gamma, mu, HYPERPARAMS, size=None),
            params=(LDA, lda, gamma, mu),
            distance="gaussian",
            sum_stat="identity",
            epsilon=1,
            observed=data)

        # Échantillonnage
        idata = pm.sample_smc(
            draws=args.draws,
            start=None,
            random_seed=42,
            chains=args.chains,
            return_inferencedata=True,
            progressbar=True
        )

        # Extension avec prédictions postérieures
        idata.extend(pm.sample_posterior_predictive(idata))

    # Tracé et sauvegarde des résultats
    true_values = inverse_compute_stat_witness(data)
    plot_inference_checks(idata, args.results_dir)
    plot_posterior_predictive_stats(idata.posterior_predictive.s,
                                  true_values,
                                  args.results_dir)
    
    

    # Sauvegarde des statistiques
    summary = az.summary(idata)
    summary.to_csv(args.results_dir+"results_summary.csv")

    print("Analyse terminée !")

if __name__ == "__main__":
    main()