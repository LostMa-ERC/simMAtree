import numpy as np
import pymc as pm
import json

from utils.stats import compute_stat_witness

from models.base_model import BaseModel


class YuleModel(BaseModel):
    def __init__(self, hyperparams_file):
        with open(hyperparams_file) as f:
            hyperparams = json.load(f)

        self.n_init = hyperparams["n_init"]
        self.Nact = hyperparams["Nact"]
        self.Ninact = hyperparams["Ninact"]
        self.max_pop = hyperparams["max_pop"]

    def get_pymc_priors(self, model):
        with model:
            LDA = pm.Uniform('LDA', lower=0, upper=0.05)
            lda = pm.Uniform('lda', lower=0, upper=0.05)
            gamma = pm.Uniform('gamma', lower=0, upper=0.01)
            mu = pm.Uniform('mu', lower=0, upper=0.01)
        return LDA, lda, gamma, mu
    
    def get_constraints(self, model, params):
        LDA, lda, gamma, mu = params
        with model:
            constraints = pm.Potential('constraints',
                pm.math.switch((lda + gamma > mu) & (gamma < lda), 0, -np.inf))
        return constraints
    
    def simulate_pop(self, rng, params):
        
        LDA = params[0]
        lda = params[1]
        gamma = params[2]
        mu = params[3]
        if not isinstance(LDA, float):
            LDA = LDA[0]
            lda = lda[0]
            gamma = gamma[0]
            mu = mu[0]
            

        species_count = {i: 1 for i in range(self.n_init)} if self.n_init > 0 else {}
        next_species_id = self.n_init
        
        t = 0
        total_pop = sum(species_count.values())
        
        while t < self.Nact and total_pop <= self.max_pop:
            n_new_trees = rng.poisson(LDA)
            
            if n_new_trees:
                next_species_id += 1
                species_count[next_species_id] = 1
                total_pop += 1
            
            new_species_count = species_count.copy()
            for species, count in species_count.items():
                if count == 0:
                    continue

                probs = [lda, gamma, mu, 1-(lda+gamma+mu)]
                n_event = rng.multinomial(count, probs)

                if total_pop + n_event[0] + n_event[1] > self.max_pop:
                    return "BREAK"

                # Copy
                new_species_count[species] += n_event[0]
                total_pop += n_event[0]

                # Speciation
                for _ in range(n_event[1]):
                    next_species_id += 1
                    new_species_count[next_species_id] = 1
                total_pop += n_event[1]

                # Death
                new_species_count[species] = max(0, new_species_count[species] - n_event[2])
                total_pop -= min(n_event[2], new_species_count[species])

            if total_pop > self.max_pop:
                return "BREAK"

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
        witness_nb = self.simulate_pop(rng, params)        
        stats = compute_stat_witness(witness_nb)

        if not np.all(np.isfinite(stats)):
            return np.zeros(6)

        return stats



