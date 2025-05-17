from pathlib import Path

import pandas as pd
from rich.console import Console

from src.inference.base_backend import InferenceBackend
from src.models.base_model import BaseModel
from src.utils import visualisation
from src.utils.stats import compute_stat_witness, inverse_compute_stat_witness


def inference(
    csv_file: str,
    model: BaseModel,
    backend: InferenceBackend,
    dir: Path,
    csv_separator: str = ";",
):
    console = Console()
    console.clear()

    # Load data
    console.rule("Loading dataframe")
    console.print(f"Data: {csv_file}")
    df = pd.read_csv(csv_file, sep=csv_separator, engine="python")
    witness_counts = df.groupby("text_ID")["witness_ID"].count()

    # Compute statistics
    console.rule("Computing stats")
    witness_nb = list(witness_counts)
    stats = compute_stat_witness(witness_nb=witness_nb)
    console.print(stats)

    # Run inference
    console.rule("Running inference")
    console.print(model.name, style="cyan")
    inference_data = backend.run_inference(model=model, data=stats)
    console.print(inference_data)

    # Compute results
    observed_values = inverse_compute_stat_witness(stats=stats)
    console.print(observed_values)

    # Save the inference data to results directory
    console.rule("Writing results")
    console.print("Output directory: ", dir.absolute())
    backend.save_results(inference_data=inference_data, output_dir=dir)

    # Plot visualisations to results directory
    visualisation.plot_inference_checks(inference_data, dir)

    try:
        visualisation.plot_posterior_predictive_stats(
            inference_data.posterior_predictive.s,
            observed_values,
            dir,
        )
    except Exception as e:
        # Predictive stats isn't currently working.
        # See tests/vis_inference_test.py
        raise e("This is a known error. See tests/vis_inference_test.py")

    return inference_data
