import numpy as np
import pymc as pm
import json

from utils.stats import compute_stat_witness

from models.base_model import BaseModel


class BirthDeathPoisson(BaseModel):
    def __init__(self, hyperparams_file):
        with open(hyperparams_file) as f:
            hyperparams = json.load(f)

        self.n_init = hyperparams["n_init"]
        self.Nact = hyperparams["Nact"]
        self.Ninact = hyperparams["Ninact"]
        self.max_pop = hyperparams["max_pop"]
    
    def simulate_pop(self, rng, LDA, lda, mu):
        """
        Generate a Yule population
        """
        

        species_count = {i: 1 for i in range(self.n_init)} if self.n_init > 0 else {}
        next_species_id = self.n_init
        
        t = 0
        total_pop = sum(species_count.values())
        
        while t < self.Nact and total_pop <= self.max_pop:
            n_new_trees = rng.poisson(lam=LDA) if total_pop < self.max_pop else 0
            
            if n_new_trees:
                next_species_id += 1
                species_count[next_species_id] = 1
                total_pop += 1
            
            new_species_count = species_count.copy()
            
            for species, count in species_count.items():
                if count == 0:
                    continue
                
                if total_pop < self.max_pop:
                    # Copie simple
                    n_copies = rng.binomial(count, lda)
                    if n_copies:
                        new_copies = min(n_copies, self.max_pop - total_pop)
                        new_species_count[species] += new_copies
                        total_pop += new_copies

                # Mort
                n_deaths = rng.binomial(count, mu)
                if n_deaths:
                    new_species_count[species] = max(0, new_species_count[species] - n_deaths)
                    total_pop -= min(n_deaths, new_species_count[species])

            # Nettoyage des espèces éteintes
            species_count = {k: v for k, v in new_species_count.items() if v > 0}
            if not species_count:
                return []
            
            total_pop = sum(species_count.values())
            t += 1
        
        # Phase de décimation optimisée
        survival_rate = (1 - mu) ** self.Ninact
        
        # Application directe du taux de survie
        final_species_count = {}
        for species, count in species_count.items():
            n_survivors = rng.binomial(count, survival_rate)
            if n_survivors > 0:
                final_species_count[species] = n_survivors
        
        return list(final_species_count.values()) if final_species_count else []

    def get_simulator(self, rng, params, size=None):
        LDA = params[0]
        lda = params[1]
        mu = params[2]

        witness_nb = self.simulate_pop(rng, LDA, lda, mu)
        try:
            if not witness_nb:
                return np.zeros(6)
            stats = compute_stat_witness(witness_nb)
            if not np.all(np.isfinite(stats)):
                return np.zeros(6)
            return stats
        except Exception as e:
            print(f"Error in simulate_tree_stats: {e}")
            print()
            return np.zeros(6)
    

    def get_pymc_priors(self, model):
        with model:
            LDA = pm.Gamma('LDA', alpha=1, beta=1)
            lda = pm.Uniform('lda', lower=0, upper=0.05)
            mu = pm.Uniform('mu', lower=0, upper=0.01)
        return LDA, lda, mu
    
    def get_constraints(self, model, params):
        LDA, lda, mu = params
        with model:
            constraints = pm.Potential('constraints',
                pm.math.switch((lda > mu), 0, -np.inf))
        return constraints
    
    

