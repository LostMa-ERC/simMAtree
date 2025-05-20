import numpy as np


def compute_stat_witness(
    witness_nb: list,
    additional_stats: bool = True,
) -> np.ndarray:
    """
    Compute stats on witness distribution.
    """
    nb_stats = 13 if additional_stats else 6

    if not witness_nb:
        return np.zeros(nb_stats)
    elif witness_nb == "BREAK":
        return np.ones(nb_stats)

    witness_nb = np.array(witness_nb, dtype=np.float64)
    nb_texts = witness_nb.size
    nb_witnesses = np.sum(witness_nb)

    stats = []

    # Calcul de chaque statistique
    stats.append(nb_witnesses.item() / 1e6)
    stats.append(nb_texts / 1e6)
    stats.append(nb_texts / nb_witnesses.item())
    stats.append(np.max(witness_nb).item() / nb_witnesses.item())
    stats.append(np.median(witness_nb).item() / np.max(witness_nb).item())
    stats.append(np.sum(witness_nb == 1).item() / nb_texts)

    if additional_stats:
        stats.append(np.sum(witness_nb == 2).item() / nb_texts)
        stats.append(np.sum(witness_nb == 3).item() / nb_texts)
        stats.append(np.sum(witness_nb == 4).item() / nb_texts)
        stats.append(
            np.quantile(witness_nb, 0.75).item() / np.max(witness_nb).item(),
        )
        stats.append(
            np.quantile(witness_nb, 0.85).item() / np.max(witness_nb).item(),
        )
        stats.append(
            np.partition(witness_nb, -2)[-2].item() / nb_witnesses.item()
            if len(witness_nb) > 1
            else 0
        )
        stats.append(
            np.partition(witness_nb, -2)[-2].item() / np.max(witness_nb).item()
            if len(witness_nb) > 1
            else 0
        )

    # Construction du tableau avec des valeurs scalaires
    stats = np.array(stats, dtype=np.float64)

    return stats


def inverse_compute_stat_witness(stats):
    """Inverse les statistiques pour retrouver les valeurs d'origine"""

    nb_temoins = int(stats[0] * 1e6)  # number of witnesses
    nb_oeuvre = int(stats[1] * 1e6)  # number of works
    max_wit = int(stats[3] * stats[0] * 1e6)  # maximum number of witnesses for 1 work
    med_wit = int(
        stats[4] * stats[3] * stats[0] * 1e6
    )  # median of witness number per work
    nb_one = int(stats[5] * stats[1] * 1e6)  # works with 1 witness

    return [nb_temoins, nb_oeuvre, max_wit, med_wit, nb_one]


def compute_hist_stats(witness_nb, max_bins=200):
    """
    Compute complete histogram of witness distribution.
    """
    if not witness_nb:
        return np.zeros(max_bins)
    elif witness_nb == "BREAK":
        return np.ones(max_bins)

    witness_nb = np.array(witness_nb, dtype=np.float64)

    # Calculer l'histogramme
    max_count = min(int(np.max(witness_nb)), max_bins)
    hist = np.zeros(max_bins)

    # Remplir l'histogramme
    for i in range(1, max_count + 1):
        hist[i - 1] = np.sum(witness_nb == i)

    # Normaliser pour éviter les problèmes d'échelle
    hist = hist / np.sum(witness_nb)

    return hist
