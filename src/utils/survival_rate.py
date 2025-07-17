import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.integrate import quad


def expected_number_witness(LDA, lda, gamma, mu, n_init, Nact, Ninact=0, Nino=None):
    if not Nino:
        Nino = Nact

    expected_size_ino = (n_init + (LDA / (lda + gamma - mu))) * torch.exp(
        (lda + gamma - mu) * Nino
    ) - (LDA / (lda + gamma - mu))
    expected_size_act = torch.exp((lda - mu) * (Nact - Nino))
    expected_loss_inact = torch.exp(-mu * Ninact)

    return expected_size_ino * expected_size_act * expected_loss_inact


def expected_number_witness_np(LDA, lda, gamma, mu, n_init, Nact, Ninact=0, Nino=None):
    if not Nino:
        Nino = Nact

    expected_size_ino = (n_init + (LDA / (lda + gamma - mu))) * np.exp(
        (lda + gamma - mu) * Nino
    ) - (LDA / (lda + gamma - mu))
    expected_size_act = np.exp((lda - mu) * (Nact - Nino))
    expected_loss_inact = np.exp(-mu * Ninact)

    return expected_size_ino * expected_size_act * expected_loss_inact


def expected_number_trees_analytical(
    LDA, lda, gamma, mu, Nact, Ninact, Nino=None, n_init=1
):
    if not Nino:
        Nino = Nact

    alpha = lda + gamma - mu
    active_part = (LDA / (lda + gamma)) * np.log(
        ((lda + gamma) * np.exp(alpha * (Nact)) - mu)
        / ((lda + gamma) * np.exp(alpha * (Nact - Nino)) - mu)
    )

    # mean_wit_per_tree = expected_number_witness_np(LDA, lda, gamma, mu, Nact, 0, Nino) / active_part
    decrease_rate = np.exp(-mu * Ninact)

    return (n_init + active_part) * decrease_rate


def _extinction_probability_single_tree(t_a, t_e, lda, mu):
    """
    Calcule la probabilité d'extinction à l'instant `t` pour un seul arbre
    né à l'instant `tau`.
    """

    if t_a < 0 or t_e < 0:
        return 1.0

    decay = 1 - np.exp(-mu * (t_e - t_a))
    p0_ta = mu * (np.exp((lda - mu) * t_a) - 1) / (lda * np.exp((lda - mu) * t_a) - mu)

    numerator = decay * (1 - p0_ta) * (1 - p0_ta * (lda / mu))
    denominator = 1 - p0_ta * decay * (lda / mu)

    result = p0_ta + numerator / denominator

    return result


def expected_number_trees(LDA, lda, gamma, mu, Nact, Ninact, Nino=None):
    """
    Calcule la fraction de survie des arbres à l'instant t en utilisant
    une intégration numérique précise de la probabilité de survie sur 3 phases.

    Returns:
        float: Fraction des arbres initiés qui survivent à l'instant t.
    """
    if not Nino:
        Nino = Nact
    t1 = Nino
    t2 = Nact
    t = Nact + Ninact

    if t1 <= 0 or LDA <= 0:
        return 0.0
    if t < t2:
        raise ValueError("Le temps final 't' doit être supérieur ou égal à 't2'.")

    survival_integral, _ = quad(
        lambda tau: 1.0
        - _extinction_probability_single_tree(t2 - tau, t - tau, lda, mu),
        0,
        t1,
        limit=500,
    )
    return LDA * survival_integral


def expected_number_texts(
    LDA: float, lda: float, gamma: float, mu: float, t: int, n_init: int = 1
) -> float:
    alpha = lda + gamma - mu
    rho = 1 + gamma / (lda - mu)
    K = lda + mu - 2 * mu / (rho + 2)
    omega = mu / (alpha + K)

    exp_term = (
        (gamma / alpha)
        * (1 - omega)
        * (
            expected_number_witness_np(LDA, lda, gamma, mu, t, n_init)
            - expected_number_witness_np(LDA, lda, gamma, mu, 0, n_init)
        )
    )
    linear_term = LDA * (1 - (gamma / alpha) - (mu / K) + (gamma * mu / (alpha * K)))

    return exp_term + linear_term * t + n_init


def tree_survival_rate(LDA, lda, gamma, mu, Nact, Ninact, Nino=None):
    if not Nino:
        Nino = Nact

    if expected_number_trees(LDA, lda, gamma, mu, Nact, Ninact, Nino) <= 1e-9:
        print("#" * 50)
        print("Parameters:")
        print(f"LDA: {LDA}, lda: {lda}, gamma: {gamma}, mu: {mu}\n")
        print("Numerical solution:")
        print(
            expected_number_trees(LDA, lda, gamma, mu, Nact, Ninact, Nino)
            / (LDA * Nino)
        )

    return expected_number_trees(LDA, lda, gamma, mu, Nact, Ninact, Nino) / (LDA * Nino)


def text_survival_rate(LDA, lda, gamma, mu, Nact, Ninact, Nino=None, n_init=1):
    if not Nino:
        Nino = Nact

    alpha = lda + gamma - mu
    S_cum = (
        n_init
        + LDA * Nino
        + ((lda + gamma) / alpha)
        * (
            expected_number_witness_np(LDA, lda, gamma, mu, n_init, Nact, 0, Nino)
            - n_init
            - LDA * Nino
        )
    )

    return (
        expected_number_witness_np(LDA, lda, gamma, mu, n_init, Nact, Ninact, Nino)
        / S_cum
    )


def plot_richness_posterior(posterior_samples, hyperparams, output_dir):
    """
    Plot posterior distributions of richness derived from model parameters

    Parameters
    ----------
    posterior_samples : np.ndarray
        Posterior samples with shape (n_samples, n_dims)
    hyperparams : dict
        Model hyperparameters (n_init, Nact, Ninact, max_pop)
    output_dir : Path
        Output directory for saving plots
    """

    Nact = hyperparams["Nact"]
    Nino = hyperparams.get("Nino", Nact)

    # Calculate survival rates for each posterior sample
    n_samples = posterior_samples.shape[0]
    n_dims = posterior_samples.shape[1]

    # Initialize arrays for survival rates
    tree_cumulative_sum = np.zeros(n_samples)
    text_cumulative_sum = np.zeros(n_samples)
    wit_cumulative_sum = np.zeros(n_samples)

    # Calculate survival rates for each sample
    for i in range(n_samples):
        LDA, lda, gamma, mu = 0, 0, 0, 0
        if n_dims == 2:
            lda, mu = posterior_samples[i]
        if n_dims == 3:
            LDA, lda, mu = posterior_samples[i]
        elif n_dims == 4:
            LDA, lda, gamma, mu = posterior_samples[i]
        else:
            raise ValueError(f"Unsupported parameter dimension: {n_dims}")

        alpha = lda + gamma - mu
        rho = 1 + gamma / (lda - mu)

        tree_cumulative_sum[i] = 1 + LDA * Nino
        text_cumulative_sum[i] = (
            1
            + (gamma / alpha)
            * expected_number_witness_np(LDA, lda, gamma, mu, 0, Nact, 0, Nino)
            + LDA * Nino / rho
        )
        wit_cumulative_sum[i] = LDA * Nino + ((lda + gamma) / alpha) * (
            expected_number_witness_np(LDA, lda, gamma, mu, 0, Nact, 0, Nino)
            - LDA * Nino
        )

    # Create plots
    survival_data = {
        "Story Richness": tree_cumulative_sum,
        "Text Richness": text_cumulative_sum,
        "Witness Richness": wit_cumulative_sum,
    }

    fig, axes = plt.subplots(1, len(survival_data), figsize=(15, 6), dpi=300)
    axes = axes.ravel()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (name, rates) in enumerate(survival_data.items()):
        ax = axes[idx]

        # Plot posterior distribution
        sns.histplot(
            rates,
            ax=ax,
            kde=True,
            color=colors[idx],
            alpha=0.7,
            stat="density",
            bins=30,
        )

        # Add summary statistics
        median_rate = np.median(rates)
        # ci_lower = np.percentile(rates, 2.5)
        # ci_upper = np.percentile(rates, 97.5)
        hdi_interval = az.hdi(rates, hdi_prob=0.95)

        ax.axvline(
            median_rate,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_rate:.0f}",
        )
        ax.axvline(hdi_interval[0], color="gray", linestyle=":", alpha=0.7)
        ax.axvline(
            hdi_interval[1],
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"95% HDI: [{hdi_interval[0]:.0f}, {hdi_interval[1]:.0f}]",
        )

        ax.set_title(name, fontsize=17, fontweight="bold")
        ax.set_xlabel("Richness", fontsize=15)
        ax.set_ylabel("Density", fontsize=15)
        ax.legend(fontsize=15, loc="upper right")
        ax.grid(alpha=0.3)

        # Set x-axis to [0, 1] for survival rates
        # ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.suptitle(
        "Posterior Distributions of Richness", fontsize=20, fontweight="bold", y=1.02
    )
    plt.savefig(output_dir / "richness_posterior.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save summary statistics
    summary_stats = {}
    for name, rates in survival_data.items():
        summary_stats[name] = {
            "median": np.median(rates),
            "std": np.std(rates),
            "ci_2.5": np.percentile(rates, 2.5),
            "ci_97.5": np.percentile(rates, 97.5),
        }

    summary_df = pd.DataFrame(summary_stats).T
    summary_df.to_csv(output_dir / "richness_summary.csv")

    return summary_stats


def plot_survival_rates_posterior(posterior_samples, hyperparams, output_dir):
    """
    Plot posterior distributions of survival rates derived from model parameters

    Parameters
    ----------
    posterior_samples : np.ndarray
        Posterior samples with shape (n_samples, n_dims)
    hyperparams : dict
        Model hyperparameters (n_init, Nact, Ninact, max_pop)
    output_dir : Path
        Output directory for saving plots
    """

    Nact = hyperparams["Nact"]
    Ninact = hyperparams["Ninact"]
    Nino = hyperparams.get("Nino", Nact)

    # Calculate survival rates for each posterior sample
    n_samples = posterior_samples.shape[0]
    n_dims = posterior_samples.shape[1]

    # Initialize arrays for survival rates
    tree_survival_rates = np.zeros(n_samples)
    text_survival_rates = np.zeros(n_samples)

    # Calculate survival rates for each sample
    for i in range(n_samples):
        LDA, lda, gamma, mu = 0, 0, 0, 0
        if n_dims == 2:
            lda, mu = posterior_samples[i]
        if n_dims == 3:
            LDA, lda, mu = posterior_samples[i]
        elif n_dims == 4:
            # THE FORMULAS ARE WRONG WITH GAMMA != 0 !
            return
            LDA, lda, gamma, mu = posterior_samples[i]
        else:
            raise ValueError(f"Unsupported parameter dimension: {n_dims}")

        tree_survival_rates[i] = tree_survival_rate(
            LDA, lda, gamma, mu, Nact, Ninact, Nino
        )

        text_survival_rates[i] = text_survival_rate(
            LDA, lda, gamma, mu, Nact, Ninact, Nino
        )

    # Create plots
    survival_data = {
        "Text Survival": tree_survival_rates,
        "Witness Survival": text_survival_rates,
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    axes = axes.ravel()

    colors = ["#1f77b4", "#ff7f0e"]

    for idx, (name, rates) in enumerate(survival_data.items()):
        ax = axes[idx]

        # Plot posterior distribution
        sns.histplot(
            rates,
            ax=ax,
            kde=True,
            color=colors[idx],
            alpha=0.7,
            stat="density",
            bins=30,
        )

        # Add summary statistics
        median_rate = np.median(rates)
        # ci_lower = np.percentile(rates, 2.5)
        # ci_upper = np.percentile(rates, 97.5)
        hdi_interval = az.hdi(rates, hdi_prob=0.95)

        ax.axvline(
            median_rate,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_rate * 100:.2f}%",
        )
        ax.axvline(hdi_interval[0], color="gray", linestyle=":", alpha=0.7)
        ax.axvline(
            hdi_interval[1],
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"95% HDI: [{hdi_interval[0] * 100:.2f}%, {hdi_interval[1] * 100:.2f}%]",
        )

        ax.set_title(name, fontsize=17, fontweight="bold")
        ax.set_xlabel("Survival Fraction", fontsize=15)
        ax.set_ylabel("Density", fontsize=15)
        ax.legend(fontsize=15)
        ax.grid(alpha=0.3)

        # Set x-axis to [0, 1] for survival rates
        # ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.suptitle(
        "Posterior Distributions of Survival Rates",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(
        output_dir / "survival_rates_posterior.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    # Save summary statistics
    summary_stats = {}
    for name, rates in survival_data.items():
        summary_stats[name] = {
            "median": np.median(rates),
            "std": np.std(rates),
            "ci_2.5": np.percentile(rates, 2.5),
            "ci_97.5": np.percentile(rates, 97.5),
        }

    summary_df = pd.DataFrame(summary_stats).T
    summary_df.to_csv(output_dir / "survival_rates_summary.csv")

    return summary_stats
