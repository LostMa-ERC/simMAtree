from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch

from .base_generator import BaseGenerator


class TwoStatesBDGenerator(BaseGenerator):
    """
    Two-States Birth-Death generator for manuscript transmission with temporal parameters

    This generator models manuscripts in two states:
    - Living state: manuscripts can be copied, die, or transition to conserved state
    - Conserved state: manuscripts can be copied or die, but cannot return to living state

    Parameters:
    - LDA (Λ): Rate of new independent trees
    - lambda_v: Time-varying copying rates for living manuscripts (array of size k)
    - mu_v: Time-varying death rates for living manuscripts (array of size k)
    - r: Transition rate from living to conserved state
    - lda: Constant copying rate for conserved manuscripts
    - mu: Constant death rate for conserved manuscripts
    """

    def __init__(self, n_init: int, N: int, max_pop: int, params_csv: str):
        """
        Initialize the Two-States Birth-Death generator

        Parameters
        ----------
        n_init : int
            Number of initial trees
        N : int
            Total number of iterations (replaces Nact + Ninact)
        params_csv : str
            Path to CSV file containing temporal parameters (Lda, lda, mu columns)
        max_pop : int
            Maximum population size
        """
        self.n_init = n_init
        self.N = N
        self.max_pop = max_pop

        self.params_csv = params_csv

        # Load temporal parameters
        self.temporal_params = self._load_temporal_params()
        self.k = len(self.temporal_params)

        # Parameters: r + lda + mu (temporal params come from CSV)
        self.param_count = 3

    def _load_temporal_params(self) -> pd.DataFrame:
        """
        Load temporal parameters from CSV file
        """
        df = pd.read_csv(self.params_csv, sep=";")
        expected_columns = ["LDA", "lda", "mu"]
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"CSV must contain columns: {expected_columns}")
        return df[expected_columns]

    def generate(
        self,
        rng: np.random.Generator,
        params: Union[list, tuple, dict],
        verbose: bool = False,
    ) -> Union[List[int], str]:
        """
        Generate manuscript population according to the Two-States Birth-Death model

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
        params : Union[list, tuple, dict]
            Model parameters
        verbose : bool
            Whether to print verbose output

        Returns
        -------
        Union[List[int], str]
            List of manuscript counts per text, or "BREAK" if population limit reached
        """
        # Extract and validate parameters
        model_params = self._extract_params(params)

        if not self.validate_params(params):
            raise ValueError(f"Invalid parameters for {self.__class__.__name__}")

        pop = self._simulate_population(rng, **model_params)

        if pop == []:
            if verbose:
                print("No survivors in the simulation!")
        elif pop == "BREAK":
            if verbose:
                print("The estimation hit the maximum size during simulation...")
                print("Estimation not saved.")

        return pop

    def _extract_params(
        self, params: Union[list, tuple, dict]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract parameters from input format

        Expected parameters: [r, lda, mu]
        Temporal parameters (LDA, lambda_v, mu_v) come from CSV
        """
        if isinstance(params, dict):
            r = params.get("r", 0)
            lda = params.get("lda", 0)
            mu = params.get("mu", 0)
        elif isinstance(params, torch.Tensor):
            params_array = params.numpy()
            r = params_array[0]
            lda = params_array[1]
            mu = params_array[2]
        elif isinstance(params, (list, tuple)):
            r = params[0]
            lda = params[1]
            mu = params[2]
        else:
            raise ValueError(f"Unsupported parameter type: {type(params)}")

        # Get temporal parameters from CSV
        LDA_temporal = self.temporal_params["LDA"].values
        lambda_v_temporal = self.temporal_params["lda"].values
        mu_v_temporal = self.temporal_params["mu"].values

        return {
            "LDA": LDA_temporal,
            "lambda_v": lambda_v_temporal,
            "mu_v": mu_v_temporal,
            "r": r,
            "lda": lda,
            "mu": mu,
        }

    def validate_params(self, params: Union[list, tuple, dict]) -> bool:
        """
        Validate Two-States Birth-Death parameters

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Parameters to validate

        Returns
        -------
        bool
            True if all parameters are valid
        """
        try:
            model_params = self._extract_params(params)

            # Check that all parameters are non-negative and finite
            if model_params["r"] < 0 or np.isnan(model_params["r"]):
                return False
            if model_params["lda"] < 0 or np.isnan(model_params["lda"]):
                return False
            if model_params["mu"] < 0 or np.isnan(model_params["mu"]):
                return False

            # Check temporal parameters
            if np.any(model_params["LDA"] < 0) or np.any(np.isnan(model_params["LDA"])):
                return False
            if np.any(model_params["lambda_v"] < 0) or np.any(
                np.isnan(model_params["lambda_v"])
            ):
                return False
            if np.any(model_params["mu_v"] < 0) or np.any(
                np.isnan(model_params["mu_v"])
            ):
                return False

            # Check that living state parameters are reasonable
            # (lambda_v + r > mu_v for population growth)
            if np.any(
                model_params["lambda_v"] + model_params["r"] <= model_params["mu_v"]
            ):
                return False

            return True

        except (ValueError, KeyError, IndexError):
            return False

    def _simulate_population(
        self,
        rng: np.random.Generator,
        LDA: np.ndarray,
        lambda_v: np.ndarray,
        mu_v: np.ndarray,
        r: float,
        lda: float,
        mu: float,
    ) -> Union[List[int], str]:
        """
        Simulate population evolution with two states

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
        LDA : np.ndarray
            Time varying tree creation rate
        lambda_v : np.ndarray
            Time-varying copying rates for living manuscripts
        mu_v : np.ndarray
            Time-varying death rates for living manuscripts
        r : float
            Transition rate from living to conserved state
        lda : float
            Constant copying rate for conserved manuscripts
        mu : float
            Constant death rate for conserved manuscripts

        Returns
        -------
        Union[List[int], str]
            List of manuscript counts or "BREAK"
        """
        # Initialize populations
        # Each text has two states: living and conserved
        living_counts = {i: 1 for i in range(self.n_init)}
        conserved_counts = {i: 0 for i in range(self.n_init)}

        next_text_id = self.n_init

        # Simulate over N iterations
        for t in range(self.N):
            # Determine which time period we're in for temporal parameters
            period_idx = min(t * self.k // self.N, self.k - 1)
            current_LDA = LDA[period_idx]
            current_lambda_v = lambda_v[period_idx]
            current_mu_v = mu_v[period_idx]

            # Calculate total population
            total_pop = sum(living_counts.values()) + sum(conserved_counts.values())

            # Check population limit
            if total_pop >= self.max_pop:
                return "BREAK"

            # Create new trees (Poisson process)
            if current_LDA > 0 and total_pop < self.max_pop:
                n_new_trees = rng.poisson(current_LDA)
                for _ in range(min(n_new_trees, self.max_pop - total_pop)):
                    living_counts[next_text_id] = 1
                    conserved_counts[next_text_id] = 0
                    next_text_id += 1
                    total_pop += 1

            # Process each text
            new_living_counts = living_counts.copy()
            new_conserved_counts = conserved_counts.copy()

            for text_id in list(living_counts.keys()):
                living_count = living_counts[text_id]
                conserved_count = conserved_counts[text_id]

                # Process living manuscripts
                if living_count > 0:
                    # Copying events
                    n_copies = rng.binomial(living_count, current_lambda_v)
                    if n_copies > 0 and total_pop < self.max_pop:
                        new_copies = min(n_copies, self.max_pop - total_pop)
                        new_living_counts[text_id] += new_copies
                        total_pop += new_copies

                    # Death events
                    n_deaths = rng.binomial(living_count, current_mu_v)
                    if n_deaths > 0:
                        deaths = min(n_deaths, new_living_counts[text_id])
                        new_living_counts[text_id] -= deaths
                        total_pop -= deaths

                    # Copying + transition to conserved state
                    n_transitions = rng.binomial(
                        new_living_counts[text_id], current_lambda_v / r
                    )
                    if n_transitions > 0:
                        new_conserved_counts[text_id] += n_transitions

                # Process conserved manuscripts
                if conserved_count > 0:
                    # Copying events
                    n_copies = rng.binomial(conserved_count, lda)
                    if n_copies > 0 and total_pop < self.max_pop:
                        new_copies = min(n_copies, self.max_pop - total_pop)
                        new_conserved_counts[text_id] += new_copies
                        total_pop += new_copies

                    # Death events
                    n_deaths = rng.binomial(conserved_count, mu)
                    if n_deaths > 0:
                        deaths = min(n_deaths, new_conserved_counts[text_id])
                        new_conserved_counts[text_id] -= deaths
                        total_pop -= deaths

            # Update counts
            living_counts = new_living_counts
            conserved_counts = new_conserved_counts

            # Remove extinct texts
            extinct_texts = [
                text_id
                for text_id in living_counts
                if living_counts[text_id] == 0 and conserved_counts[text_id] == 0
            ]
            for text_id in extinct_texts:
                del living_counts[text_id]
                del conserved_counts[text_id]

            # Check if population is extinct
            if not living_counts and not conserved_counts:
                return []

        # Return total manuscript counts per text
        final_counts = []
        for text_id in living_counts:
            total_count = living_counts[text_id] + conserved_counts[text_id]
            if total_count > 0:
                final_counts.append(total_count)

        return final_counts if final_counts else []

    def save_simul(self, pop: List[int], data_path: str) -> pd.DataFrame:
        """
        Save simulation results to CSV file

        Parameters
        ----------
        pop : List[int]
            List of manuscript counts per text
        data_path : str
            Path to save the CSV file

        Returns
        -------
        pd.DataFrame
            DataFrame that was saved
        """
        if not pop or pop == "BREAK":
            print("An empty population is given so nothing is saved.")
            return None

        witness_ids = []
        text_ids = []

        for text_idx, count in enumerate(pop):
            for manuscript_idx in range(count):
                witness_ids.append(f"W{text_idx}-{manuscript_idx + 1}")
                text_ids.append(f"T{text_idx}")

        df = pd.DataFrame({"witness_ID": witness_ids, "text_ID": text_ids})

        df.to_csv(data_path, sep=";", index=False)
        return df

    def load_data(self, csv_file: str, csv_sep: str = ";") -> List[int]:
        """
        Load data from CSV file

        Parameters
        ----------
        csv_file : str
            Path to CSV file
        csv_sep : str
            CSV separator

        Returns
        -------
        List[int]
            List of manuscript counts per text
        """
        df = pd.read_csv(csv_file, sep=csv_sep, engine="python")
        witness_counts = df.groupby("text_ID")["witness_ID"].count()
        return list(witness_counts)
