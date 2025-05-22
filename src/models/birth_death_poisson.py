import numpy as np
import pymc as pm
from pytensor.tensor.variable import TensorVariable
import torch
from sbi.utils.user_input_checks import process_prior


from src.models.base_model import AbstractModelClass
from src.models.constants import PyMCPriors
from src.utils.stats import compute_stat_witness


class BirthDeathPoisson(AbstractModelClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_pymc_priors(self, model):
        # Required method, inherited from AbstractBaseClass
        with model:
            LDA = pm.Gamma("LDA", alpha=1, beta=1)
            lda = pm.Uniform("lda", lower=0, upper=0.05)
            mu = pm.Uniform("mu", lower=0, upper=0.01)
        return PyMCPriors(LDA=LDA, lda=lda, mu=mu)

    def get_constraints(self, model, params) -> TensorVariable:
        # Required method, inherited from AbstractBaseClass
        with model:
            constraints = pm.Potential(
                "constraints", pm.math.switch((params.lda > params.mu), 0, -np.inf)
            )
        return constraints

    def get_simulator(self, rng, params, size=None, additional_stats=True):
        # Required method, inherited from AbstractBaseClass
        witness_nb = self.simulate_pop(rng, params)
        try:
            if not witness_nb:
                return np.zeros(13) if additional_stats else np.zeros(6)
            stats = compute_stat_witness(witness_nb, additional_stats)
            if not np.all(np.isfinite(stats)):
                return np.zeros(13) if additional_stats else np.zeros(6)
            return stats
        except Exception as e:
            print(f"Error in simulate_tree_stats: {e}")
            print()
            return np.zeros(6)

    def simulate_pop(self, rng, params):
        # Required method, inherited from AbstractBaseClass
        LDA, lda, gamma, mu = self.process_params(params)

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

    def get_sbi_priors(self, device="cpu"):
        from src.models.distribution import ConstrainedUniform

        # LDA, lda, gamma, mu
        lower_bounds = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
        upper_bounds = torch.tensor([2, 0.015, 0.01, 0.01], device=device)

        prior = ConstrainedUniform(lower_bounds, upper_bounds, device=device)
        prior, num_parameters, prior_returns_numpy = process_prior(prior)
        return prior

    def process_params(self, params):
        # TODO: Check where the params are generated and see if the data type
        # can be replaced by something more stable (dataclass, Pydantic model,
        # namedtuple, etc.)
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
