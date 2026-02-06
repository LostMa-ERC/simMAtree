import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sbi.inference
import torch
from sbi.inference import simulate_for_sbi
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from torch.distributions import Distribution

from src.generator import BaseGenerator
from src.inference.base_backend import AbstractInferenceClass
from src.stats import AbstractStatsClass
from src.utils.survival_rate import (
    plot_richness_posterior,
    plot_survival_rates_posterior,
)
from src.utils.visualisation import (
    compute_hpdi_point,
    plot_combined_hpdi,
    plot_marginal_posterior,
    plot_posterior_predictive_stats,
    plot_prior_posterior_comparison,
)


class SbiBackend(AbstractInferenceClass):
    def __init__(
        self,
        device: str = "cpu",
        method: str = None,
        num_simulations: int = 10,
        num_rounds: int = 1,
        random_seed: int = 42,
        num_samples: int = 10,
        num_workers: int = 1,
        posterior_path: str = None,
    ):
        super().__init__(random_seed=random_seed)
        self.method = method
        self.num_simulations = num_simulations
        self.num_rounds = num_rounds
        self.randon_seed = random_seed
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.device = device
        self.results = None
        self.posterior_path = posterior_path
        self.trained_posterior = None

    def run_inference(
        self,
        generator: BaseGenerator,
        stats: AbstractStatsClass,
        data: np.ndarray,
        prior: Distribution,
    ):
        self.prior_for_plotting = prior

        if hasattr(prior, "hyperparams"):
            self.hyperparams = prior.hyperparams
        else:
            self.hyperparams = None

        # Convert data to torch tensor
        x_o = torch.tensor(data, dtype=torch.float32).to(self.device)

        # Prior checking using SBI util function
        prior, _, _ = process_prior(prior)

        # Define simulator wrapper
        def sbi_simulator_wrapper(params):
            try:
                if not generator.validate_params(params):
                    raise ValueError(f"Unvalidated parameters: {params}")
                pop = generator.generate(self.rng, params)
                result = stats.compute_stats(pop)
                return torch.tensor(result, dtype=torch.float32)
            except ValueError:
                return torch.zeros(13, dtype=torch.float32)

        # Process simulator for SBI
        simulator = process_simulator(sbi_simulator_wrapper, prior, False)

        # Check SBI inputs
        check_sbi_inputs(simulator, prior)

        if self.posterior_path:
            print(f"Loading pre-trained model from {self.posterior_path}")

            with open(self.posterior_path, "rb") as f:
                posterior = pickle.load(f)

        else:
            print(f"Training device: {self.device}")
            if self.method is None:
                raise ValueError(
                    "SBI Method is required in the config for training (example: NPE)"
                )

            # Choose inference method
            Model_class = getattr(sbi.inference, self.method.upper())
            try:
                inference = Model_class(prior=prior, device=self.device)
            except Exception as e:
                raise e("Unknown SBI method")

            # Perform inference with multiple rounds
            num_simulations = self.num_simulations
            num_rounds = self.num_rounds

            posteriors = []
            proposal = prior
            # TODO : This part must be change if self.device == "cuda" (alternate between cpu and gpu for simulation and learning)

            print("Running simulations...")

            for i in range(num_rounds):
                print(f"ROUND {i + 1}")

                params, x = simulate_for_sbi(
                    simulator, proposal, num_simulations, num_workers=self.num_workers
                )
                zero_counter = torch.sum(torch.all(x == 0, dim=1)).item()
                break_counter = torch.sum(torch.all(x == 1, dim=1)).item()
                print(
                    f"\n{zero_counter} zero occurrences out of {num_simulations} simulations ({zero_counter / num_simulations * 100:.2f}%)"
                )
                print(
                    f"{break_counter} BREAK occurrences out of {num_simulations} simulations ({break_counter / num_simulations * 100:.2f}%)\n"
                )

                if num_rounds == 1:
                    density_estimator = inference.append_simulations(params, x).train()
                else:
                    density_estimator = inference.append_simulations(
                        params, x, proposal
                    ).train(force_first_round_loss=True)
                posterior = inference.build_posterior(
                    density_estimator, sample_with="mcmc"
                ).set_default_x(x_o)
                posteriors.append(posterior)

                accept_reject_fn = get_density_thresholder(
                    posterior, quantile=1e-4, num_samples_to_estimate_support=100000
                )
                proposal = RestrictedPrior(
                    prior, accept_reject_fn, sample_with="rejection"
                )

            self.trained_posterior = posterior

        # Get samples from the posterior
        num_samples = 1000
        samples = posterior.sample((num_samples,), x=x_o)

        # Create posterior predictive samples
        samples_np = samples.cpu().numpy()
        hpdi_point, hpdi_samples = compute_hpdi_point(samples_np, prob_level=0.95)

        print("Running posterior predictive checks...")
        _, pp_samples = simulate_for_sbi(
            simulator, posterior, self.num_samples, num_workers=self.num_workers
        )

        flat_samples = pp_samples.reshape(-1, pp_samples.shape[-1])
        pp_samples_stats = np.array(
            [stats.rescaled_stats(s) for s in flat_samples],
        )

        self.results = {
            "posterior_samples": samples_np,
            "posterior_predictive_stats": pp_samples_stats,
            "observed_data": x_o.numpy(),
            "parameter_names": [f"param_{i}" for i in range(samples_np.shape[1])],
            "hpdi_point": hpdi_point,
            "hpdi_samples": hpdi_samples,
        }

        return self.results

    def save_results(self, observed_values: list, output_dir: Path):
        if not hasattr(self, "results") or self.results is None:
            print("Inference needs to be done before saving the results")
            return

        # Create summary statistics for parameters
        samples = self.results["posterior_samples"]

        np.save(output_dir.joinpath("posterior_samples.npy"), samples)
        np.save(output_dir.joinpath("obs_values.npy"), observed_values)
        np.save(
            output_dir.joinpath("posterior_predictive.npy"),
            self.results["posterior_predictive_stats"],
        )

        # Calculate summary statistics
        summary_stats = {
            "mean": np.mean(samples, axis=0),
            "std": np.std(samples, axis=0),
            "5%": np.percentile(samples, 5, axis=0),
            "50%": np.percentile(samples, 50, axis=0),
            "95%": np.percentile(samples, 95, axis=0),
            "hpdi_95%": self.results["hpdi_point"],
        }

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(output_dir.joinpath("posterior_summary.csv"))

    def plot_results(self, data, observed_values, output_dir):
        plot_posterior_predictive_stats(
            data["posterior_predictive_stats"],
            obs_value=observed_values,
            output_dir=output_dir,
        )
        plot_combined_hpdi([data["posterior_samples"]], output_dir=output_dir)
        plot_marginal_posterior(data["posterior_samples"], output_dir=output_dir)

        if hasattr(self, "prior_for_plotting"):
            plot_prior_posterior_comparison(
                data["posterior_samples"],
                self.prior_for_plotting,
                output_dir=output_dir,
            )

        if hasattr(self, "hyperparams") and self.hyperparams is not None:
            plot_survival_rates_posterior(
                data["posterior_samples"],
                self.hyperparams,
                output_dir=output_dir,
            )
            plot_richness_posterior(
                data["posterior_samples"],
                self.hyperparams,
                output_dir=output_dir,
            )

    def save_model(self, output_dir: Path):
        """
        Save the trained SBI posterior
        """
        if self.trained_posterior is None:
            print("No trained model to save. Run inference first.")
            return

        try:
            posterior_path = output_dir / "posterior.pkl"
            with open(posterior_path, "wb") as f:
                pickle.dump(self.trained_posterior, f)
            print(f"Posterior saved to {posterior_path}")

        except Exception as e:
            print(f"Error saving model: {e}")
