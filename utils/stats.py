import numpy as np

def compute_stat_witness(witness_nb):
    """Calcule les statistiques sur la distribution des témoins"""
    if not witness_nb:
        return np.zeros(6)
    elif witness_nb == "BREAK":
        return np.ones(6)
    
    witness_nb = np.array(witness_nb, dtype=np.float64)
    nb_oeuvre = witness_nb.size
    nb_temoins = np.sum(witness_nb)
    
    # Calcul de chaque statistique avec les méthodes NumPy appropriées
    stat1 = nb_temoins.item()/1e6
    stat2 = nb_oeuvre/1e6
    stat3 = nb_oeuvre/nb_temoins.item()
    stat4 = np.max(witness_nb).item()/nb_temoins.item()
    stat5 = np.median(witness_nb).item()/np.max(witness_nb).item()
    stat6 = np.sum(witness_nb == 1).item()/nb_oeuvre
    
    # Construction du tableau avec des valeurs scalaires
    stats = np.array([stat1, stat2, stat3, stat4, stat5, stat6], dtype=np.float64)
    
    return stats

def inverse_compute_stat_witness(stats):
    """Inverse les statistiques pour retrouver les valeurs d'origine"""
    # if np.sum(stats) == 0:
    #     return np.zeros(5)
    # elif np.sum(stats) == 6:
    #     return None

    nb_temoins = int(stats[0]*1e6)
    nb_oeuvre = int(stats[1]*1e6)
    max_wit = int(stats[3]*stats[0]*1e6)
    med_wit = int(stats[4]*stats[3]*stats[0]*1e6)
    nb_one = int(stats[5]*stats[1]*1e6)
    
    return [nb_temoins, nb_oeuvre, max_wit, med_wit, nb_one]

