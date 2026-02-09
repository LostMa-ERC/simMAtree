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

    # Run inference
    console.rule("Running inference")
    console.print(type(generator).__name__, style="cyan")
    inference_data = backend.run_inference(
        generator=generator, data=pop, stats=stats, prior=prior
    )

    # Save the inference data to results directory
    console.rule("Writing results")
    console.print("Output directory: ", dir.absolute())

    backend.save_results(stats=stats, data=pop, output_dir=dir)
    backend.plot_results(stats=stats, data=inference_data, pop=pop, output_dir=dir)

    if save_model:
        console.rule("Saving trained model")
        backend.save_model(output_dir=dir)

    return inference_data
