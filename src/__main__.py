import click

from src.cli.generate import generate
from src.cli.inference import inference
from src.cli.score import score


@click.group
def cli():
    pass


@cli.command("infer")
def infer_command():
    inference()


@cli.command("generate")
def generate_command():
    generate()


@cli.command("score")
def score_command():
    score()


if __name__ == "__main__":
    cli()
