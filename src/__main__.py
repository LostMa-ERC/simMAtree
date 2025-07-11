from pathlib import Path

import click

from src.cli.config import Config
from src.cli.generate import generate
from src.cli.inference import inference
from src.cli.score import score


@click.group
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.pass_context
def cli(ctx, config):
    ctx.obj = Config(config)


@click.command
def test_installation():
    print("Looks good!")


@cli.command("infer")
@click.option(
    "-i",
    "--infile",
    required=True,
    type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=False),
)
@click.option(
    "-o",
    "--outdir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option("-s", "--separator", required=False, default=";", type=click.STRING)
@click.option("--save", is_flag=True, help="Save the trained model")
@click.pass_obj
def infer_command(config: Config, infile: str, outdir: str, separator: str, save: str):
    generator = config.generator
    stats = config.stats
    prior = config.prior
    backend = config.backend
    dir = Path(outdir)
    dir.mkdir(exist_ok=True)
    inference(
        csv_file=infile,
        generator=generator,
        stats=stats,
        prior=prior,
        backend=backend,
        dir=dir,
        csv_separator=separator,
        save_model=save,
    )


@cli.command("generate")
@click.option(
    "-o",
    "--outfile",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
)
@click.option(
    "-s",
    "--seed",
    required=False,
    default=42,
    type=click.INT,
    help="Random seed for generation",
)
@click.option("--show-params", is_flag=True, help="Display parameters in JSON format")
@click.pass_obj
def generate_command(config: Config, outfile: str, seed: int, show_params: bool):
    generator = config.generator
    params = config.params
    stats = config.stats
    generate(
        data_path=outfile,
        generator=generator,
        parameters=params,
        stats=stats,
        seed=seed,
        show_params=show_params,
    )


@cli.command("score")
@click.option(
    "-d",
    "--directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.pass_obj
def score_command(config: Config, directory: str):
    true_params = config.params
    score(param_dict=true_params, results_dir=directory)


if __name__ == "__main__":
    cli()
