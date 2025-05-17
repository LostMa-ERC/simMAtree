import numpy as np
import pymc as pm
import torch
from sbi.utils.user_input_checks import process_prior

from src.models.base_model import BaseModel
from src.utils.stats import compute_stat_witness


class YuleModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate_params(self, params):
        """Valide les paramètres pour le modèle Yule"""
        try:
            LDA, lda, gamma, mu = self.process_params(params)

            if (
                LDA < 0
                or np.isnan(LDA)
                or lda < 0
                or np.isnan(lda)
                or gamma < 0
                or np.isnan(gamma)
                or mu < 0
                or np.isnan(mu)
            ):
                raise ValueError("Paramètres invalides détectés")

            if not (lda + gamma > mu and gamma < lda):
                raise ValueError("Contraintes du modèle non respectées")

            return params
        except ValueError:
            raise

    def get_pymc_priors(self, model):
        with model:
            LDA = pm.Uniform("LDA", lower=0, upper=0.05)
            lda = pm.Uniform("lda", lower=0, upper=0.05)
            gamma = pm.Uniform("gamma", lower=0, upper=0.01)
            mu = pm.Uniform("mu", lower=0, upper=0.01)
        return LDA, lda, gamma, mu

    def get_sbi_priors(self, device="cpu"):
        from src.models.distribution import ConstrainedUniform

        # LDA, lda, gamma, mu
        lower_bounds = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
        upper_bounds = torch.tensor([2, 0.015, 0.01, 0.01], device=device)

        prior = ConstrainedUniform(lower_bounds, upper_bounds, device=device)
        prior, num_parameters, prior_returns_numpy = process_prior(prior)
        return prior

    def get_constraints(self, model, params):
        LDA, lda, gamma, mu = params
        with model:
            constraints = pm.Potential(
                "constraints",
                pm.math.switch((lda + gamma > mu) & (gamma < lda), 0, -np.inf),
            )
        return constraints

    def process_params(self, params):
        if isinstance(params, torch.Tensor):
            LDA = params[0].item()
            lda = params[1].item()
            gamma = params[2].item()
            mu = params[3].item()
        else:
            LDA = params[0]
            lda = params[1]
            gamma = params[2]
            mu = params[3]
            try:
                if not isinstance(LDA, float) and hasattr(LDA, "__getitem__"):
                    LDA = LDA[0]
                    lda = lda[0]
                    gamma = gamma[0]
                    mu = mu[0]
            except Exception:
                pass
        return LDA, lda, gamma, mu

    def simulate_pop(self, rng, params):

        LDA, lda, gamma, mu = self.process_params(params)

        if self.n_init > 0:
            species_counter = {i: 1 for i in range(self.n_init)}
        else:
            species_counter = {}
        next_species_id = self.n_init

        t = 0
        total_pop = sum(species_counter.values())

        while t < self.Nact and total_pop <= self.max_pop:
            n_new_trees = rng.poisson(LDA)

            if n_new_trees:
                next_species_id += 1
                species_counter[next_species_id] = 1
                total_pop += 1

            new_species_count = species_counter.copy()
            for species, count in species_counter.items():
                if count == 0:
                    continue

                probs = [lda, gamma, mu, 1 - (lda + gamma + mu)]
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
                new_species_count[species] = max(
                    0, new_species_count[species] - n_event[2]
                )
                total_pop -= min(n_event[2], new_species_count[species])

            if total_pop > self.max_pop:
                return "BREAK"

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

    def get_simulator(self, rng, params, additional_stats=True, size=None):
        witness_nb = self.simulate_pop(rng, params)
        stats = compute_stat_witness(witness_nb, additional_stats)

        return stats
