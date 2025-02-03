import numpy as np
import pymc as pm
import arviz as az
from inference.base_backend import InferenceBackend
from utils.visualisation import plot_inference_checks, plot_posterior_predictive_stats

class PymcBackend(InferenceBackend):
    def __init__(self, config_file):
        super(PymcBackend, self).__init__(config_file)

    def run_inference(self, model, data):
        with pm.Model() as pymc_model:
            params = model.get_pymc_priors(pymc_model)
            nb_param = len(params)
            constraints = model.get_constraints(pymc_model, params)
            simulator = pm.Simulator("s",
                                    lambda rng, *params, size=None: model.get_simulator(rng, params, params[-1] if len(params) > nb_param else size),
                                    params=params,
                                    distance=self.inference_params["distance"],
                                    sum_stat=self.inference_params["sum_stat"],
                                    epsilon=self.inference_params["epsilon"],
                                    observed=data)

            idata = pm.sample_smc(
                draws=self.inference_params["draws"],
                chains=self.inference_params["chains"],
                random_seed=self.inference_params["random_seed"],
                return_inferencedata=True,
                progressbar=True
            )
            
            idata.extend(pm.sample_posterior_predictive(idata))
            self.results = idata

        return self.results
    
    def save_results(self, obs_values, output_dir):

        if self.results is None:
            print("Inference needs to be done before saving the results")

        summary = az.summary(self.results)
        summary.to_csv(f"{output_dir}/results_summary.csv")
        plot_inference_checks(self.results, output_dir)
        plot_posterior_predictive_stats(
            self.results.posterior_predictive.s,
            obs_values,
            output_dir
        )