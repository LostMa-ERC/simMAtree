import numpy as np
import pandas as pd
import sbi.inference
import torch
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import check_sbi_inputs, process_simulator

from src.inference.base_backend import InferenceBackend
from src.utils.visualisation import compute_hpdi_point


class SbiBackend(InferenceBackend):
    def __init__(self, **kwargs):
        # self.device = self.inference_params.get("device", "cpu")
        super().__init__(**kwargs)
        print(self.device)
        print(self.cpu)

    def run_inference(self, model, data):

        print(f"Training device: {self.device}")
        simulation_device = torch.device("cpu")
        print(f"Simulation device: {simulation_device}")

        # Convert data to torch tensor
        x_o = torch.tensor(data, dtype=torch.float32).to(self.device)

        # Get prior from model
        prior = model.get_sbi_priors(device=self.device)

        # Define simulator wrapper
        def sbi_simulator_wrapper(params):
            try:
                validated_params = model.validate_params(params)
                result = model.get_simulator(self.rng, validated_params)
                return torch.tensor(result, dtype=torch.float32)
            except ValueError:
                return torch.zeros(13, dtype=torch.float32)

        # Process simulator for SBI
        simulator = process_simulator(sbi_simulator_wrapper, prior, False)

        # Check SBI inputs
        check_sbi_inputs(simulator, prior)

        # Choose inference method
        Model_class = getattr(sbi.inference, self.inference_params["method"].upper())
        try:
            inference = Model_class(prior=prior, device=self.device)
        except Exception as e:
            raise e("Unknown SBI method")

        # Perform inference with multiple rounds
        num_simulations = self.inference_params["num_simulations"]
        num_rounds = self.inference_params["num_rounds"]

        posteriors = []
        proposal = prior
        # TODO : This part must be change if self.device == "cuda" (alternate between cpu and gpu for simulation and learning)

        for i in range(num_rounds):
            print(f"ROUND {i+1}")
            params, x = simulate_for_sbi(
                simulator,
                proposal,
                num_simulations,
                num_workers=self.inference_params["num_workers"],
            )
            zero_counter = torch.sum(torch.all(x == 0, dim=1)).item()
            break_counter = torch.sum(torch.all(x == 1, dim=1)).item()
            print(
                f"\n{zero_counter} zero occurrences out of {num_simulations} simulations ({zero_counter/num_simulations*100:.2f}%)"
            )
            print(
                f"{break_counter} BREAK occurrences out of {num_simulations} simulations ({break_counter/num_simulations*100:.2f}%)\n"
            )
            if num_rounds == 1:
                density_estimator = inference.append_simulations(params, x).train()
            else:
                density_estimator = inference.append_simulations(
                    params, x, proposal=proposal
                ).train()
            posterior = inference.build_posterior(density_estimator)
            posteriors.append(posterior)
            proposal = posterior.set_default_x(x_o)

        # Get samples from the posterior
        num_samples = 5000
        samples = posterior.sample((num_samples,), x=x_o)

        # Create posterior predictive samples
        samples_np = samples.cpu().numpy()
        hpdi_point, hpdi_samples = compute_hpdi_point(samples_np, prob_level=0.95)

        _, pp_samples = simulate_for_sbi(
            simulator,
            proposal,
            self.inference_params["num_samples"],
            num_workers=self.inference_params["num_workers"],
        )

        # num_pp = 100
        # pp_samples = []
        # for i in tqdm(range(min(num_pp, len(samples_np))), desc="Generating posterior predictive samples"):
        #     param = samples_np[i]
        #     sim = model.get_simulator(self.rng, param)
        #     pp_samples.append(sim)

        pp_samples = np.array(pp_samples)

        self.results = {
            "posterior_samples": samples_np,
            "posterior_predictive": pp_samples,
            "observed_data": x_o.numpy(),
            "parameter_names": [f"param_{i}" for i in range(samples_np.shape[1])],
            "hpdi_point": hpdi_point,
            "hpdi_samples": hpdi_samples,
        }

        return self.results

    def save_results(self, obs_values, output_dir):
        if self.results is None:
            print("Inference needs to be done before saving the results")
            return

        # Create summary statistics for parameters
        samples = self.results["posterior_samples"]

        np.save(f"{output_dir}/posterior_samples.npy", samples)
        np.save(f"{output_dir}/obs_values.npy", obs_values)
        np.save(
            f"{output_dir}/posterior_predictive.npy",
            self.results["posterior_predictive"],
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
        summary_df.to_csv(f"{output_dir}/posterior_summary.csv")
