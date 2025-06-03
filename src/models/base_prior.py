from abc import abstractmethod

import torch
from torch.distributions import Distribution, Independent, Uniform
from torch.distributions.constraints import independent, interval


class ConstrainedUniform(Distribution):
    def __init__(self, low, high, constraints_params=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        if constraints_params is None:
            self.constraints_params = [1, 1000, 1000, 10**4]
        else:
            self.constraints_params = constraints_params

        self._low = low
        self._high = high

        self.base_dist = Independent(Uniform(low, high), 1)

        with torch.no_grad():
            n_samples = 10000

            samples_base = self.base_dist.sample(torch.Size([n_samples]))
            valid_mask = self._check_constraints(samples_base)
            self.valid_proportion = torch.mean(valid_mask.float())
            self.log_normalizing_constant = -torch.log(self.valid_proportion)

            samples = self.sample(torch.Size([n_samples]))
            self.mean_val = torch.mean(samples, dim=0)
            self.std_val = torch.std(samples, dim=0)

        batch_shape = self.base_dist.batch_shape
        event_shape = self.base_dist.event_shape
        super().__init__(batch_shape, event_shape)

    @property
    def support(self):
        return independent(interval(self._low, self._high), 1)

    @property
    def mean(self):
        return self.mean_val

    @property
    def stddev(self):
        return self.std_val

    def log_prob(self, value):
        base_log_prob = self.base_dist.log_prob(value)
        valid = self._check_constraints(value)

        normalized_log_prob = base_log_prob + self.log_normalizing_constant

        return torch.where(
            valid, normalized_log_prob, torch.tensor(-float("inf"), device=value.device)
        )

    @abstractmethod
    def _check_constraints(self, x):
        pass

    def sample(self, sample_shape=torch.Size()):
        """
        Rejection sampling to respect the constraints
        """
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
            raise ValueError("Unable to generate enough valid samples.")

        return samples
