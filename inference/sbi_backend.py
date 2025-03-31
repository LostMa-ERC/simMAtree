import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sbi.inference import SNPE, NPE, NRE, SNLE, BNRE, simulate_for_sbi
from sbi.utils.user_input_checks import check_sbi_inputs, process_simulator
from sbi.analysis import pairplot
from inference.base_backend import InferenceBackend
from utils.visualisation import plot_posterior_predictive_stats

class SbiBackend(InferenceBackend):
    def __init__(self, config_file):
        super(SbiBackend, self).__init__(config_file)

    def run_inference(self, model, data):
        # Convert data to torch tensor if it's not already
        x_o = torch.tensor(data, dtype=torch.float32)
        
        # Get prior from model
        prior = model.get_sbi_priors()
        
        # Define simulator wrapper
        def sbi_simulator_wrapper(params):
            result = model.get_simulator(self.rng, params)
            return torch.tensor(result, dtype=torch.float32)
        
        # Process simulator for SBI
        simulator = process_simulator(sbi_simulator_wrapper, prior, False)
        
        # Check SBI inputs
        check_sbi_inputs(simulator, prior)
        
        # Choose inference method
        method = self.inference_params["method"].upper()
        if method == "NPE":
            inference = NPE(prior=prior)
        elif method == "SNPE":
            inference = SNPE(prior=prior)
        elif method == "NRE":
            inference = NRE(prior=prior)
        elif method == "SNLE":
            inference = SNLE(prior=prior)
        elif method == "BNRE":
            inference = BNRE(prior=prior)
        else:
            raise ValueError(f"Unknown SBI method: {method}")
        
        # Perform inference with multiple rounds
        num_simulations = self.inference_params["num_simulations"]
        num_rounds = self.inference_params["num_rounds"]
        
        posteriors = []
        proposal = prior
        
        for i in range(num_rounds):
            print(f"ROUND {i+1}")
            params, x = simulate_for_sbi(simulator, proposal, num_simulations)
            density_estimator = inference.append_simulations(params, x, proposal=proposal).train()
            posterior = inference.build_posterior(density_estimator)
            posteriors.append(posterior)
            proposal = posterior.set_default_x(x_o)
        
        # Get samples from the posterior
        num_samples = self.inference_params["num_samples"]
        samples = posterior.sample((num_samples,), x=x_o)
        
        # Create posterior predictive samples
        samples_np = samples.cpu().numpy()
        num_pp = 100
        pp_samples = []
        for i in range(min(num_pp, len(samples_np))):
            param = samples_np[i]
            sim = model.get_simulator(self.rng, param)
            pp_samples.append(sim)
        
        pp_samples = np.array(pp_samples)
        
        # Create a structure similar to PyMC's InferenceData
        import xarray as xr
        import arviz as az
        
        # Structure samples to mimic PyMC's chains and draws
        n_chains = 4  # Match PyMC's default chains
        chain_length = num_samples // n_chains
        reshaped_samples = samples_np[:n_chains * chain_length].reshape(n_chains, chain_length, -1)
        
        # Create xarray datasets
        posterior_coords = {
            "chain": np.arange(n_chains),
            "draw": np.arange(chain_length),
            "param": [f"param_{i}" for i in range(samples_np.shape[1])]
        }
        
        posterior_data = xr.DataArray(
            reshaped_samples,
            coords=posterior_coords,
            dims=["chain", "draw", "param"]
        ).to_dataset(dim="param")
        
        # Structure posterior predictive samples
        pp_chains = min(n_chains, pp_samples.shape[0] // chain_length)
        pp_draws = min(chain_length, pp_samples.shape[0] // pp_chains)
        
        # Ensure we have at least one sample
        if pp_chains == 0 or pp_draws == 0:
            pp_chains = 1
            pp_draws = pp_samples.shape[0]
        
        pp_reshaped = pp_samples[:pp_chains * pp_draws].reshape(pp_chains, pp_draws, -1)
        
        pp_coords = {
            "chain": np.arange(pp_chains),
            "draw": np.arange(pp_draws),
            "stat": np.arange(pp_samples.shape[1])
        }
        
        pp_data = xr.Dataset(
            {"s": (["chain", "draw", "stat"], pp_reshaped)}
        )
        
        # Create InferenceData object
        self.results = az.InferenceData(
            posterior=posterior_data,
            posterior_predictive=pp_data
        )
        
        return self.results
    
    def save_results(self, obs_values, output_dir):
        if self.results is None:
            print("Inference needs to be done before saving the results")
            return
        
        import arviz as az
        
        # Summary statistics
        summary = az.summary(self.results)
        summary.to_csv(f"{output_dir}/results_summary.csv")
        
        # Plot inference checks using the same functions as for PyMC
        from utils.visualisation import plot_inference_checks
        plot_inference_checks(self.results, output_dir)
        
        # Plot posterior predictive checks
        plot_posterior_predictive_stats(
            self.results.posterior_predictive.s,
            obs_values,
            output_dir
        )