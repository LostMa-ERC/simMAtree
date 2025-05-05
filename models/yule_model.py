import numpy as np
import pymc as pm
import json
import torch
from torch.distributions.constraints import independent, interval

from sbi.utils.user_input_checks import process_prior

from utils.stats import compute_stat_witness
from utils.yule import avg_yule_pop
from models.base_model import BaseModel

import torch
from torch.distributions import Distribution, Independent, Uniform

class ConstrainedUniform(Distribution):
    def __init__(self, low, high, device=None):
        """
        Prior uniforme pour paramètres LDA, lda, gamma, mu avec contraintes:
        - lda + gamma > mu
        - gamma < lda
        
        Args:
            low: borne inférieure [LDA, lda, gamma, mu]
            high: borne supérieure [LDA, lda, gamma, mu]
            device: périphérique pour les tenseurs
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low, dtype=torch.float32, device=device)
        else:
            low = low.to(device)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high, dtype=torch.float32, device=device)
        else:
            high = high.to(device)
            
        self.base_dist = Independent(Uniform(low, high), 1)
        
        # Dimension du prior est 4: [LDA, lda, gamma, mu]
        assert low.shape[-1] == 4 and high.shape[-1] == 4, "Les paramètres doivent être de dimension 4"
        
        batch_shape = self.base_dist.batch_shape
        event_shape = self.base_dist.event_shape
        super().__init__(batch_shape, event_shape)
    
    @property
    def support(self):
        return independent(interval(self._low, self._high), 1)
        
    @property
    def mean(self):
        return (self._low + self._high) / 2.0
        
    @property
    def stddev(self):
        return (self._high - self._low) / (2.0 * np.sqrt(3.0))
    
    def _check_constraints(self, x):
        # Contrainte 1: lda + gamma > mu (indices 1, 2, 3)
        constraint1 = x[..., 1] + x[..., 2] > x[..., 3]
        
        # Contrainte 2: gamma < lda (indices 1, 2)
        constraint2 = x[..., 2] < x[..., 1]

        # Contrainte 3: E[population d'un arbre] < 10^4
        constraint3 = avg_yule_pop(x[...,1], x[...,2], x[...,3], 1000,1000) <= 10**5
        
        return constraint1 & constraint2 & constraint3
    
    def sample(self, sample_shape=torch.Size()):
        """Échantillonnage par rejet pour respecter les contraintes"""
        samples = self.base_dist.sample(sample_shape)
        valid = self._check_constraints(samples)
        
        # Continuer à échantillonner jusqu'à ce que tous les échantillons soient valides
        max_attempts = 200
        attempt = 0
        
        while not torch.all(valid) and attempt < max_attempts:
            # Générer de nouveaux échantillons pour les points non valides
            new_samples = self.base_dist.sample(sample_shape)
            
            # Remplacer uniquement les échantillons non valides
            if samples.dim() > 1:
                invalid_indices = torch.where(~valid)[0]
                samples[invalid_indices] = new_samples[invalid_indices]
            else:
                samples = new_samples if not valid else samples
                
            valid = self._check_constraints(samples)
            attempt += 1
        
        if attempt == max_attempts and not torch.all(valid):
            raise ValueError("Impossible de générer suffisamment d'échantillons valides.")
        
        return samples
    
    def log_prob(self, value):
        """Calcule le log de la probabilité pour la valeur donnée"""
        # Vérifier si la valeur est dans la boîte et respecte les contraintes
        base_log_prob = self.base_dist.log_prob(value)
        valid = self._check_constraints(value)
        
        # Si la valeur ne respecte pas les contraintes, retourner -inf
        return torch.where(valid, base_log_prob, torch.tensor(-float('inf'), device=value.device))



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

    def get_sbi_priors(self, device='cpu'):
        # LDA, lda, gamma, mu
        lower_bounds = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device) 
        upper_bounds = torch.tensor([1, 0.015, 0.01, 0.01], device=device)  

        prior = ConstrainedUniform(lower_bounds, upper_bounds, device=device)
        prior, num_parameters, prior_returns_numpy = process_prior(prior)
        return prior

    def get_constraints(self, model, params):
        LDA, lda, gamma, mu = params
        with model:
            constraints = pm.Potential('constraints',
                pm.math.switch((lda + gamma > mu) & (gamma < lda), 0, -np.inf))
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
                if not isinstance(LDA, float) and hasattr(LDA, '__getitem__'):
                    LDA = LDA[0]
                    lda = lda[0]
                    gamma = gamma[0]
                    mu = mu[0]
            except:
                pass
        return LDA, lda, gamma, mu
            

    def simulate_pop(self, rng, params):
        
        LDA, lda, gamma, mu = self.process_params(params)

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

    def get_simulator(self, rng, params,  additional_stats = True, size=None):
        witness_nb = self.simulate_pop(rng, params)        
        stats = compute_stat_witness(witness_nb, additional_stats)

        if not np.all(np.isfinite(stats)):
            return np.zeros(6)

        return stats



