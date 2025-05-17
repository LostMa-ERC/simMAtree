from pathlib import Path

import click

from src.cli.config_parser import Config
from src.cli.generate import generate
from src.cli.inference import inference
from src.cli.score import score


@click.group
@click.option("-c", "--config")
@click.pass_context
def cli(ctx, config):
    ctx.obj = Config(config)


@cli.command("infer")
@click.option("-i", "--infile", required=True)
@click.option("-o", "--outdir", required=True)
@click.option("-s", "--separator", required=False, default=";")
@click.pass_obj
def infer_command(config: Config, infile: str, outdir: str, separator: str):
    model = config.parse_model_config()
    backend = config.parse_backend_config()
    dir = Path(outdir)
    dir.mkdir(exist_ok=True)
    inference(
        csv_file=infile,
        model=model,
        backend=backend,
        dir=dir,
        csv_separator=separator,
    )


@cli.command("generate")
@click.option("-i", "--infile", required=True)
@click.pass_obj
def generate_command(config: Config, infile: str):
    model = config.parse_model_config()
    generate(data_path=infile, model=model)


@cli.command("score")
@click.option("-o", "--outdir", required=True)
@click.pass_obj
def score_command(config: Config, outdir: str):
    true_params = config.parse_experiment_parameters()
    score(true_params=true_params, results_dir=outdir)


if __name__ == "__main__":
    cli()
