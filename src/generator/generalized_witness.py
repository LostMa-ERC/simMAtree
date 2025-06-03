from abc import abstractmethod
from typing import Dict, Union

import numpy as np

from .base_generator import BaseGenerator


class GeneralizedWitnessGenerator(BaseGenerator):
    """
    Generalized witness generator implementing the full 4-parameter model

    Implements the complete simulation logic for models with parameters:
    - LDA: Rate of new independent trees
    - lda: Probability of copying/reproduction
    - gamma: Probability of speciation
    - mu: Probability of death

    Subclasses can specialize by fixing certain parameters to zero.
    """

    def __init__(self, n_init: int, Nact: int, Ninact: int, max_pop: int):
        """
        Initialize the generalized witness generator

        Parameters
        ----------
        **kwargs : parameters passed to BaseGenerator
        """
        self.n_init = n_init
        self.Nact = Nact
        self.Ninact = Ninact
        self.max_pop = max_pop

        self.infer_n_trees = False

    def generate(
        self, rng: np.random.Generator, params: Union[list, tuple, dict]
    ) -> Union[list, str]:
        """
        Generate survivor counts according to the generalized model

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
        params : Union[list, tuple, dict]
            Model parameters

        Returns
        -------
        Union[list, str]
            List of survivor counts per species, or "BREAK" if limit reached
        """
        # Extract and validate parameters
        model_params = self._extract_params(params)

        if not self.validate_params(params):
            raise ValueError(f"Invalid parameters for {self.__class__.__name__}")

        return self._simulate_population(rng, **model_params)

    @abstractmethod
    def _extract_params(self, params: Union[list, tuple, dict]) -> Dict[str, float]:
        """
        Extract model parameters according to the specific model

        Must be implemented by subclasses to define which parameters are used
        and how they are extracted from the input.

        Returns
        -------
        Dict[str, float]
            Dictionary with keys: 'LDA', 'lda', 'gamma', 'mu'
        """
        pass

    def validate_params(self, params: Union[list, tuple, dict]) -> bool:
        """
        Validate parameters according to model-specific constraints

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Parameters to validate

        Returns
        -------
        bool
            True if parameters are valid
        """
        try:
            model_params = self._extract_params(params)

            # Check basic non-negativity and no NaN
            for value in model_params.values():
                if value < 0 or np.isnan(value):
                    return False

            return True

        except (ValueError, KeyError, IndexError):
            return False

    def _simulate_population(
        self, rng: np.random.Generator, LDA: float, lda: float, gamma: float, mu: float
    ) -> Union[list, str]:
        """
        Simulate population according to the generalized algorithm

        This is the core simulation logic that works for all models
        """
        # Initialization
        if self.n_init > 0:
            species_counter = {i: 1 for i in range(self.n_init)}
        else:
            species_counter = {}

        next_species_id = self.n_init

        t = 0
        total_pop = sum(species_counter.values())

        # Active phase
        while t < self.Nact and total_pop <= self.max_pop:
            # New independent trees (LDA > 0 for Yule models)
            if total_pop < self.max_pop and LDA > 0:
                n_new_trees = rng.poisson(lam=LDA)
                if n_new_trees:
                    for _ in range(min(n_new_trees, self.max_pop - total_pop)):
                        next_species_id += 1
                        species_counter[next_species_id] = 1
                        total_pop += 1

            new_species_count = species_counter.copy()
            for species, count in species_counter.items():
                if count == 0:
                    continue

                if total_pop < self.max_pop:
                    # Reproduction and speciation events
                    if gamma > 0:
                        # Yule model: multinomial choice between lda, gamma, mu, nothing
                        probs = [lda, gamma, mu, 1 - (lda + gamma + mu)]
                        if sum(probs[:3]) <= 1:  # Valid probability
                            n_events = rng.multinomial(count, probs)

                            # Reproduction
                            if n_events[0] > 0:
                                new_copies = min(n_events[0], self.max_pop - total_pop)
                                new_species_count[species] += new_copies
                                total_pop += new_copies

                            # Speciation
                            if n_events[1] > 0:
                                new_species = min(n_events[1], self.max_pop - total_pop)
                                for _ in range(new_species):
                                    next_species_id += 1
                                    new_species_count[next_species_id] = 1
                                total_pop += new_species

                            # Death
                            if n_events[2] > 0:
                                deaths = min(n_events[2], new_species_count[species])
                                new_species_count[species] -= deaths
                                total_pop -= deaths
                    else:
                        # Birth-Death model: separate binomial events
                        # Reproduction
                        n_copies = rng.binomial(count, lda)
                        if n_copies:
                            new_copies = min(n_copies, self.max_pop - total_pop)
                            new_species_count[species] += new_copies
                            total_pop += new_copies

                        # Death
                        n_deaths = rng.binomial(count, mu)
                        if n_deaths:
                            deaths = min(n_deaths, new_species_count[species])
                            new_species_count[species] -= deaths
                            total_pop -= deaths

            # Check population limit
            if total_pop >= self.max_pop:
                return "BREAK"

            # Clean up extinct species
            species_counter = {k: v for k, v in new_species_count.items() if v > 0}
            if not species_counter:
                return []

            total_pop = sum(species_counter.values())
            t += 1

        # Decimation phase (Ninact)
        survival_rate = (1 - mu) ** self.Ninact

        final_species_count = {}
        for species, count in species_counter.items():
            n_survivors = rng.binomial(count, survival_rate)
            if n_survivors > 0:
                final_species_count[species] = n_survivors

        return list(final_species_count.values()) if final_species_count else []
