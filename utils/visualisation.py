import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az


from utils.stats import inverse_compute_stat_witness

def plot_posterior_predictive_stats(samples, obs_value, output_dir):
    """Crée et sauvegarde les distributions postérieures"""
    flat_samples = samples.values.reshape(-1, samples.shape[-1])
    processed = np.array([inverse_compute_stat_witness(s) for s in flat_samples])
    
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
            
            true_val = obs_value[i]
            median_val = np.median(processed[:, i])
            
            axes[i].axvline(true_val, color='red', linestyle='-', alpha=0.8, label='Jonas')
            axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.5)
            
            axes[i].text(0.95, 0.95,
                f'Jonas: {true_val:.2e}\nMédiane: {median_val:.2e}',
                transform=axes[i].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8))
    
    if len(metric_names) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(output_dir+"pp_summaries.png")
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
    


