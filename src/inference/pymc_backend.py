from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm

from src.inference.base_backend import InferenceBackend
from src.models import BaseModel


class PymcBackend(InferenceBackend):

    def __init__(
        self,
        draws: int,
        chains: int,
        random_seed: int,
        epsilon: int,
        sum_stat: str,
        distance: str,
    ):
        self.draws = draws
        self.chains = chains
        self.epsilon = epsilon
        self.sum_stat = sum_stat
        self.distance = distance
        self.random_seed = random_seed
        super().__init__(random_seed=random_seed)

    def run_inference(
        self,
        model: BaseModel,
        data: np.ndarray,
    ) -> az.InferenceData:
        """
        run_inference _summary_

        Args:
            model (BaseModel): Yule or SBI model.
            data (np.ndarray): Observed data / ground truth.

        Returns:
            az.InferenceData: Inference data produced by SMC sampling.
        """

        with pm.Model() as pymc_model:
            params = model.get_pymc_priors(pymc_model)
            nb_param = len(params)
            _ = model.get_constraints(pymc_model, params)
            _ = pm.Simulator(
                "s",
                lambda rng, *params, size=None: model.get_simulator(
                    rng, params, params[-1] if len(params) > nb_param else size
                ),
                params=params,
                distance=self.distance,
                sum_stat=self.sum_stat,
                epsilon=self.epsilon,
                observed=data,
            )

            inference_data = pm.sample_smc(
                draws=self.draws,
                chains=self.chains,
                random_seed=self.random_seed,
                return_inferencedata=True,
                progressbar=True,
            )

            inference_data.extend(
                pm.sample_posterior_predictive(inference_data),
            )

            samples = inference_data.posterior_predictive.s
            pp_samples = samples.values.reshape(-1, samples.shape[-1])
            pp_no_survivors = len([p for p in pp_samples if sum(p) == 0])
            pp_max_population = len(
                [
                    p
                    for p in pp_samples
                    if sum(
                        np.all(p == 1),
                    )
                ]
            )
            # Originally [p for p in pp_samples if sum(np.all(p==1)) == 1], but
            # that fails because np.all(p==1) evaluates to a boolean, and you
            # can't sum a boolean.

            print("Number of PP samples :", len(pp_samples))
            print("Number of PP samples without survivors: ", pp_no_survivors)
            print(
                "Number of PP samples with population > MAX: ",
                pp_max_population,
            )

        return inference_data

    @classmethod
    def save_results(
        cls,
        inference_data: az.InferenceData,
        output_dir: Path,
    ) -> None:

        if inference_data is None:
            print("Inference needs to be done before saving the results")

        # Write the inference data to a binary file
        trace_file = output_dir.joinpath("inference_data.nc")
        az.to_netcdf(data=inference_data, filename=trace_file)
        # Write the summary to a CSV file
        summary_file = output_dir.joinpath("results_summary.csv")
        summary = az.summary(inference_data)
        summary.to_csv(summary_file)
