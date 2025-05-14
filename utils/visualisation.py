import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import torch 
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from matplotlib.colors import to_rgba

from sbi.analysis import pairplot


from utils.stats import inverse_compute_stat_witness

def plot_sbi_pairplot(samples, output_dir, hpdi_point=None):
       
    param_samples = torch.tensor(samples)
    fig, axes = pairplot(param_samples,
             figsize=(8, 8),
             diag="kde",
             upper="kde",
             labels=[r"$\Lambda$", r"$\lambda$", r"$\gamma$", r"$\mu$"])
    
    if hpdi_point is not None:
        n_dims = len(hpdi_point)
        
        # Tracer le point HPDI sur chaque sous-graphique
        for i in range(n_dims):
            for j in range(i+1, n_dims):
                ax_upper = axes[i, j]
                # Tracer le point HPDI
                ax_upper.scatter(hpdi_point[j], hpdi_point[i], color='red', marker='*', s=100, label="HPDI 95%", zorder=10)
                
                # Ne mettre la légende que sur un seul graphique
                if i == 1 and j == 0:
                    ax_upper.legend(fontsize=10)
                
            # Ajouter le point à la diagonale
            ax_diag = axes[i, i]
            if hasattr(ax_diag, 'axvline'):  # Si c'est un axe de plot
                ax_diag.axvline(hpdi_point[i], color='red', linestyle='--', linewidth=2, label="HPDI 95%" if i == 0 else None)
                if i == 0:
                    ax_diag.legend(fontsize=10)
    

    plt.tight_layout()
    plt.savefig(f"{output_dir}/posterior_pairs.png")
    plt.close()


def plot_combined_hpdi(samples_list, output_dir, dataset_names=None, true_values=None):

    n_datasets = len(samples_list)
    
    # colors = sns.color_palette("tab10", n_datasets)
    colors = sns.color_palette("pastel", n_datasets)
    
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X'][:n_datasets]
    
    # Calculer les points HPDI pour chaque jeu de données
    hpdi_points = []
    hpdi_samples_list = []
    for samples in samples_list:
        hpdi_point, hpdi_samples = compute_hpdi_point(samples)
        hpdi_points.append(hpdi_point)
        hpdi_samples_list.append(hpdi_samples)
    
    # Déterminer les dimensions
    n_dims = samples_list[0].shape[1]
    
    param_names = [r"$\Lambda$", r"$\lambda$", r"$\gamma$", r"$\mu$"]
    
    fig = plt.figure(figsize=(3*n_dims, 3*n_dims), dpi=100)
    gs = fig.add_gridspec(n_dims, n_dims, wspace=0.3, hspace=0.3)
    
    # Créer des axes pour chaque paire de paramètres
    axes = np.empty((n_dims, n_dims), dtype=object)
    
    # Limites globales pour chaque paramètre
    global_limits = []
    for i in range(n_dims):
        all_values = np.concatenate([samples[:, i] for samples in samples_list])
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        padding = (max_val - min_val) * 0.1
        global_limits.append((min_val - padding, max_val + padding))
    
    for i in range(n_dims):
        for j in range(n_dims):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            
            # Améliorer l'apparence des axes
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=9)
            
            if i == j:  # Diagonale: distributions marginales
                for k, samples in enumerate(samples_list):
                    # Utiliser KDE avec fill pour un aspect plus moderne
                    color = colors[k]
                    color_with_alpha = to_rgba(color, alpha=0.3)
                    
                    label_name = None
                    if dataset_names is not None and j == 3:
                        label_name = dataset_names[k]

                    # Tracer la distribution avec un remplissage
                    sns.kdeplot(
                        samples[:, i],
                        ax=ax,
                        color=color,
                        fill=True,
                        alpha=0.3,
                        linewidth=2,
                        label=label_name
                    )
                    
                    # Ajouter une ligne verticale pour le point HPDI
                    ax.axvline(
                        hpdi_points[k][i],
                        color=color,
                        linestyle='-',
                        linewidth=2,
                        alpha=0.8
                    )
                    
                    # Ajouter un marqueur au point HPDI sur l'axe x
                    ax.scatter(
                        hpdi_points[k][i],
                        0,  # Bas de l'axe
                        color=color,
                        edgecolor='black',
                        s=100,
                        marker=markers[k],
                        zorder=10
                    )
                if true_values is not None:
                    ax.axvline(
                        true_values[i],
                        color='black',
                        linestyle='--',
                        linewidth=2,
                        alpha=0.8,
                        label='True value' if i == 3 else None
                    )
                    
                    # Ajouter un marqueur pour la vraie valeur
                    ax.scatter(
                        true_values[i],
                        0,  # Bas de l'axe
                        color='black',
                        edgecolor='white',
                        s=120,
                        marker='*',
                        zorder=15
                    )
                
                # Configurer les axes
                ax.set_xlabel(param_names[i], fontsize=12, fontweight='bold')
                ax.set_xlim(global_limits[i])
                ax.set_yticks([])
                ax.set_ylabel('Density', fontsize=10)
                
                # Ajouter une grille horizontale subtile
                ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            elif i < j:  # Triangle supérieur: scatter plots
                for k, samples in enumerate(samples_list):
                    # Scatter avec un contour de densité pour un look moderne
                    color = colors[k]
                    
                    # Tracer un contour de densité pour rendre le scatter plus esthétique
                    try:
                        sns.kdeplot(
                            x=samples[:, j], 
                            y=samples[:, i],
                            ax=ax,
                            levels=5,
                            color=color,
                            alpha=0.6,
                            fill=True,
                            thresh=0.05
                        )
                    except:
                        pass
                    
                    # Scatter léger des points HPDI
                    ax.scatter(
                        hpdi_samples_list[k][:, j],
                        hpdi_samples_list[k][:, i],
                        color=color,
                        alpha=0.3,
                        s=5,
                        edgecolor=None
                    )
                    
                    # Point HPDI principal avec bordure noire
                    ax.scatter(
                        hpdi_points[k][j],
                        hpdi_points[k][i],
                        color=color,
                        marker=markers[k],
                        s=120,
                        edgecolor='black',
                        linewidth=1.5,
                        zorder=100,
                        label=dataset_names[k] if i == n_dims-1 and j == 0 else None
                    )
                if true_values is not None and i < j:
                    ax.scatter(
                        true_values[j],
                        true_values[i],
                        color='black',
                        marker='*',
                        s=180,
                        edgecolor='white',
                        linewidth=1.5,
                        zorder=200
                    )
                # Configurer les axes
                ax.set_xlabel(param_names[j], fontsize=12, fontweight='bold')
                ax.set_ylabel(param_names[i], fontsize=12, fontweight='bold')
                ax.set_xlim(global_limits[j])
                ax.set_ylim(global_limits[i])
                
                # Ajouter une grille subtile
                ax.grid(alpha=0.2, linestyle='--')
            else:  # Triangle supérieur: laisser vide
                ax.set_visible(False)
    axes[3,3].legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.savefig(output_dir+"/pairplot", dpi=300, bbox_inches='tight')


def plot_marginal_posterior(samples, output_dir, hpdi_point=None):
    summary_stats = {
        'mean': np.mean(samples, axis=0),
        'std': np.std(samples, axis=0),
        '5%': np.percentile(samples, 5, axis=0),
        '50%': np.percentile(samples, 50, axis=0),
        '95%': np.percentile(samples, 95, axis=0)
    }

    fig, axes = plt.subplots(1, samples.shape[1], figsize=(4*samples.shape[1], 4))
    if samples.shape[1] == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        if i < samples.shape[1]:
            param_data = samples[:, i]
            sns.kdeplot(param_data, ax=ax, fill=True)
            ax.axvline(summary_stats['mean'][i], color='r', linestyle='-', label='Mean')
            ax.axvline(summary_stats['50%'][i], color='g', linestyle='--', label='Median')
            ax.axvline(summary_stats['5%'][i], color='b', linestyle=':', label='5%')
            ax.axvline(summary_stats['95%'][i], color='b', linestyle=':', label='95%')
            if hpdi_point is not None:
                ax.axvline(hpdi_point[i], color='purple', linestyle='-.',
                          linewidth=2, label='HPDI 95%')
                
            if i == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/posterior.png")
    plt.close()

def plot_posterior_predictive_stats(samples, output_dir, obs_value=None):
    """Crée et sauvegarde les distributions postérieures"""
    flat_samples = samples.values.reshape(-1, samples.shape[-1])
    processed = np.array([inverse_compute_stat_witness(s) for s in flat_samples]) 

    if len(processed)==0:
        print("No samples to plot")
        return 
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    metric_names = [
        "Number of witnesses",
        "Number of works",
        "Max. number of witnesses per work",
        "Med. number of witnesses per work",
        "Number of witnesses with one work"
    ]
    
    colors = sns.color_palette("husl", 5)
    
    for i, (name, color) in enumerate(zip(metric_names, colors)):
        if i < processed.shape[1]:
            sns.histplot(
                data=processed[:, i],
                ax=axes[i],
                color=color,
                stat='count',
                bins=30,
                kde=True
            )
            
            axes[i].set_title(name, fontsize=12, pad=10)
            axes[i].set_xlabel("Valeur")
            axes[i].set_ylabel("Count")
            
            median_val = np.median(processed[:, i])
            
            axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.5)
            
            if obs_value is not None:
                true_val = obs_value[i]
                axes[i].axvline(true_val, color='red', linestyle='-', alpha=0.8, label='Observation')
                axes[i].text(0.95, 0.95,
                    f'Observation: {true_val:.2e}\nMédiane: {median_val:.2e}',
                    transform=axes[i].transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8))
    
    if len(metric_names) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(output_dir+"/pp_summaries.png")
    plt.close()



def plot_inference_checks(idata, output_dir):
    """
    Visualise les résultats de l'inférence
    """
    # Tracer les chaînes MCMC
    az.plot_trace(idata)
    plt.savefig(output_dir+"traces.png")
    plt.tight_layout()
    
    # Tracer les distributions postérieures
    az.plot_posterior(idata)
    plt.savefig(output_dir+"posterior.png")
    plt.tight_layout()
    
    # Tracer les corrélations entre paramètres
    az.plot_pair(idata)

    plt.savefig(output_dir+"posterior_pairs.png")
    plt.tight_layout()
    

def compute_hpdi_point(samples, prob_level=0.95, method='kde'):
    """
    Calcule le point médian de la région de plus haute densité de probabilité (HPDI)
    pour un ensemble d'échantillons multivariés.
    
    Parameters:
    -----------
    samples : numpy.ndarray
        Échantillons de la distribution postérieure, shape (n_samples, n_dimensions)
    prob_level : float, optional (default=0.95)
        Niveau de probabilité pour le HPDI (ex: 0.95 pour 95% HPDI)
    method : str, optional (default='kde')
        Méthode pour estimer la densité: 'kde' utilise scipy.stats.gaussian_kde,
        'sklearn' utilise sklearn.neighbors.KernelDensity
        
    Returns:
    --------
    hpdi_point : numpy.ndarray
        Point médian du HPDI, shape (n_dimensions,)
    hpdi_samples : numpy.ndarray
        Échantillons qui se trouvent dans la région HPDI
    """
    n_samples, n_dims = samples.shape
    
    if method == 'kde':
        kde = gaussian_kde(samples.T)
        densities = kde(samples.T)
    elif method == 'sklearn':
        # Utiliser sklearn.neighbors.KernelDensity (plus rapide pour grandes dimensions)
        kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(samples)
        log_densities = kde.score_samples(samples)
        densities = np.exp(log_densities)
    else:
        raise ValueError(f"Méthode '{method}' non reconnue.")
    
    # Trier les échantillons par densité décroissante
    sorted_indices = np.argsort(-densities)
    sorted_samples = samples[sorted_indices]
    sorted_densities = densities[sorted_indices]
    
    # Calculer la densité cumulative normalisée
    cumulative_densities = np.cumsum(sorted_densities) / np.sum(sorted_densities)
    
    # Déterminer le seuil pour le niveau de probabilité spécifié
    threshold_idx = np.searchsorted(cumulative_densities, prob_level)
    threshold_idx = min(threshold_idx, len(cumulative_densities) - 1)
    
    # Extraire les échantillons dans la région HPDI
    hpdi_samples = sorted_samples[:threshold_idx + 1]
    
    # Calculer le point médian de la région HPDI
    # Option 1: Moyenne des échantillons dans HPDI (centre de masse)
    # hpdi_point = np.mean(hpdi_samples, axis=0)
    
    # Option 2: Médiane géométrique (plus robuste)
    hpdi_point = np.median(hpdi_samples, axis=0)
    
    # Option 3: échantillon avec la densité maximale (MAP)
    # hpdi_point = sorted_samples[0]
    
    return hpdi_point, hpdi_samples

def plot_hpdi_samples(results, output_dir):
    """Visualiser les échantillons inclus dans la région HPDI et le point HPDI"""
    samples = results['posterior_samples']
    hpdi_samples = results['hpdi_samples']
    hpdi_point = results['hpdi_point']
    
    n_params = samples.shape[1]
    param_names = ["LDA", "lda", "gamma", "mu"]
    if len(param_names) != n_params:
        param_names = [f"param_{i}" for i in range(n_params)]
    
    # Créer une matrice de scatter plots pour visualiser les paires de paramètres
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonale: histogramme
                ax.hist(samples[:, i], bins=30, alpha=0.3, color='blue', density=True)
                ax.hist(hpdi_samples[:, i], bins=30, alpha=0.6, color='red', density=True)
                ax.axvline(hpdi_point[i], color='black', linestyle='--', linewidth=2)
                ax.set_xlabel(param_names[i])
                ax.set_ylabel('Densité')
            else:
                # Scatter plot pour les paires de paramètres
                ax.scatter(samples[:, j], samples[:, i], alpha=0.1, s=5, color='blue')
                ax.scatter(hpdi_samples[:, j], hpdi_samples[:, i], alpha=0.3, s=5, color='red')
                ax.scatter(hpdi_point[j], hpdi_point[i], color='black', s=100, marker='*')
                ax.set_xlabel(param_names[j])
                ax.set_ylabel(param_names[i])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hpdi_visualization.png")
    plt.close()
