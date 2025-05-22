from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.inference.base_backend import AbstractInferenceClass
from src.models.base_model import AbstractModelClass
from src.utils.stats import compute_stat_witness, inverse_compute_stat_witness


def inference(
    csv_file: str,
    model: AbstractModelClass,
    backend: AbstractInferenceClass,
    dir: Path,
    csv_separator: str = ";",
):
    console = Console()

    # Load data
    console.rule("Dataset")
    console.print(f"Data: {csv_file}")
    df = pd.read_csv(csv_file, sep=csv_separator, engine="python")
    witness_counts = df.groupby("text_ID")["witness_ID"].count()

    # Compute statistics
    witness_nb = list(witness_counts)
    stats = compute_stat_witness(witness_nb=witness_nb)

    # Print statistics for the user
    summary = inverse_compute_stat_witness(stats)
    nb_witnesses, nb_texts, max_wits, med_wits, text_with_one_wit = summary
    table = Table(title="Data observation")
    table.add_column("statistics")
    table.add_column("value")
    table.add_row("Number of witnesses", str(nb_witnesses))
    table.add_row("Number of texts", str(nb_texts))
    table.add_row("Max witnesses for 1 text", str(max_wits))
    table.add_row("Median witnesses per text", str(med_wits))
    table.add_row("Number of texts w/ 1 witness", str(text_with_one_wit))
    console.print(table)

    # Run inference
    console.rule("Running inference")
    console.print(type(model).__name__, style="cyan")
    inference_data = backend.run_inference(model=model, data=stats)
    # console.print(inference_data)

    # Compute results
    observed_values = inverse_compute_stat_witness(stats=stats)
    # console.print(observed_values)

    # Save the inference data to results directory
    console.rule("Writing results")
    console.print("Output directory: ", dir.absolute())
    backend.save_results(observed_values=observed_values, output_dir=dir)

    backend.plot_results(
        data=inference_data, observed_values=observed_values, output_dir=dir
    )

    return inference_data
