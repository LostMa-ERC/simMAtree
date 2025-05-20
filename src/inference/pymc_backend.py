import arviz as az
import numpy as np
import pymc as pm

from src.inference.base_backend import AbstractInferenceClass
from src.utils import visualisation
from src.utils.visualisation import (
    plot_inference_checks,
    plot_posterior_predictive_stats,
)


class PymcBackend(AbstractInferenceClass):
    def __init__(
        self,
        draws: int,
        chains: int,
        random_seed: int,
        epsilon: int,
        sum_stat: str,
        distance: str,
    ):
        super().__init__(random_seed=random_seed)
        self.draws = draws
        self.chains = chains
        self.epsilon = epsilon
        self.sum_stat = sum_stat
        self.distance = distance
        self.random_seed = random_seed

    def run_inference(self, model, data):
        with pm.Model() as pymc_model:
            param_data = model.get_pymc_priors(pymc_model)
            params = param_data.list()
            nb_param = len(params)

            # TODO: constraints variable is created but never used
            _ = model.get_constraints(pymc_model, param_data)

            # TODO: simulator variable is created but never used.
            _ = pm.Simulator(
                "s",
                # TODO: I don't understand this lambda (can it be rewritten as a
                # function?) len(params) is always == nb_param, so what
                # is this if-condition doing?
                lambda rng, *params, size=None: model.get_simulator(
                    rng,
                    params,
                    params[-1] if len(params) > nb_param else size,
                ),
                params=params,
                distance=self.distance,
                sum_stat=self.sum_stat,
                epsilon=self.epsilon,
                observed=data,
            )

            idata = pm.sample_smc(
                draws=self.draws,
                chains=self.chains,
                random_seed=self.random_seed,
                return_inferencedata=True,
                progressbar=True,
            )

            idata.extend(pm.sample_posterior_predictive(idata))
            self.results = idata

            samples = idata.posterior_predictive.s
            pp_samples = samples.values.reshape(-1, samples.shape[-1])

            print(f"Number of PP samples :{len(pp_samples)}")
            print(
                f"Number of PP samples without survivors: {len([p for p in pp_samples if sum(p) == 0])}"
            )
            print(
                f"Number of PP samples with population > MAX: {len([p for p in pp_samples if sum(np.all(p == 1)) == 1])}"
            )

        return self.results

    def save_results(self, observed_values, output_dir):
        if self.results is None:
            print("Inference needs to be done before saving the results")

        summary = az.summary(self.results)
        summary.to_csv(f"{output_dir}/results_summary.csv")
        plot_inference_checks(self.results, output_dir)
        plot_posterior_predictive_stats(
            self.results.posterior_predictive.s, observed_values, output_dir
        )

    def plot_results(self, data: az.InferenceData, observed_values, output_dir):
        # Plot visualisations to results directory
        visualisation.plot_inference_checks(data, output_dir)

        try:
            visualisation.plot_posterior_predictive_stats(
                data.posterior_predictive.s,
                observed_values,
                dir,
            )
        except Exception as e:
            # TODO: Predictive stats isn't working.
            # See tests/vis_inference_test.py
            raise e("This is a known error. See tests/vis_inference_test.py")
