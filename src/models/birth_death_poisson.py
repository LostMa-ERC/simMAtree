import numpy as np
import pymc as pm

from src.models.base_model import BaseModel
from src.utils.stats import compute_stat_witness


class BirthDeathPoisson(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def simulate_pop(self, rng, params):
        """
        Generate a Yule population
        """

        LDA = params[0]
        lda = params[1]
        mu = params[2]

        if self.n_init > 0:
            species_counter = {i: 1 for i in range(self.n_init)}
        else:
            species_counter = {}

        next_species_id = self.n_init

        t = 0
        total_pop = sum(species_counter.values())

        while t < self.Nact and total_pop <= self.max_pop:
            if total_pop < self.max_pop:
                n_new_trees = rng.poisson(lam=LDA)
            else:
                n_new_trees = 0

            if n_new_trees:
                next_species_id += 1
                species_counter[next_species_id] = 1
                total_pop += 1

            new_species_count = species_counter.copy()

            for species, count in species_counter.items():
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
                    new_species_count[species] = max(
                        0, new_species_count[species] - n_deaths
                    )
                    total_pop -= min(n_deaths, new_species_count[species])

            # Nettoyage des espèces éteintes
            nsc = new_species_count
            species_counter = {k: v for k, v in nsc.items() if v > 0}
            if not species_counter:
                return []

            total_pop = sum(species_counter.values())
            t += 1

        # Phase de décimation optimisée
        survival_rate = (1 - mu) ** self.Ninact

        # Application directe du taux de survie
        final_species_count = {}
        for species, count in species_counter.items():
            n_survivors = rng.binomial(count, survival_rate)
            if n_survivors > 0:
                final_species_count[species] = n_survivors

        if final_species_count is not None:
            return list(final_species_count.values())
        else:
            return []

    def get_simulator(self, rng, params, size=None):

        witness_nb = self.simulate_pop(rng, params)
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
            LDA = pm.Gamma("LDA", alpha=1, beta=1)
            lda = pm.Uniform("lda", lower=0, upper=0.05)
            mu = pm.Uniform("mu", lower=0, upper=0.01)
        return LDA, lda, mu

    def get_constraints(self, model, params):
        LDA, lda, mu = params
        with model:
            constraints = pm.Potential(
                "constraints", pm.math.switch((lda > mu), 0, -np.inf)
            )
        return constraints
