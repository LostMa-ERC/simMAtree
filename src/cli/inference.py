from pathlib import Path

from rich.console import Console

from src.generator import BaseGenerator
from src.inference.base_backend import AbstractInferenceClass
from src.priors.base_prior import BasePrior
from src.stats.base_stats import AbstractStatsClass


def inference(
    csv_file: str,
    generator: BaseGenerator,
    stats: AbstractStatsClass,
    prior: BasePrior,
    backend: AbstractInferenceClass,
    dir: Path,
    csv_separator: str = ";",
    save_model: bool = False,
):
    console = Console()

    # Load data
    console.rule("Dataset")
    console.print(f"Data: {csv_file}")
    pop = generator.load_data(csv_file, csv_separator)

    # Compute statistics
    witness_stats = stats.compute_stats(pop)

    # Print statistics for the user
    stats.print_stats(pop)

    # Run inference
    console.rule("Running inference")
    console.print(type(generator).__name__, style="cyan")
    inference_data = backend.run_inference(
        generator=generator, data=witness_stats, stats=stats, prior=prior
    )

    # Save the inference data to results directory
    console.rule("Writing results")
    console.print("Output directory: ", dir.absolute())
    observed_values = stats.get_rescaled_stats(pop)

    backend.save_results(observed_values=observed_values, output_dir=dir)
    backend.plot_results(
        data=inference_data, observed_values=observed_values, output_dir=dir
    )

    # Save the model if option is true
    if save_model:
        console.rule("Saving trained model")
        backend.save_model(output_dir=dir)

    return inference_data
