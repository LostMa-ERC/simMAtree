# Bayesian Inference for Witness Analysis
[![DOI](https://zenodo.org/badge/900725844.svg)](https://doi.org/10.5281/zenodo.15350477)

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Python package](https://github.com/LostMa-ERC/simMAtree/actions/workflows/ci.yml/badge.svg)](https://github.com/LostMa-ERC/simMAtree/actions/workflows/ci.yml)
[![Tests](https://github.com/LostMa-ERC/simMAtree/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/LostMa-ERC/simMAtree/actions/workflows/unit-tests.yml)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

## Description
This repository performs some Simulation Based Algorithm (SBI) on abundance distribution data. 
One application done for the [LostMa](https://lostma-erc.github.io/) project consist in modelling the transmission and survival of textual witnesses through time, enabling researchers to infer model parameters from observed data.

## Contributing 🔧

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

To develop and/or contribute to the project, see more detailed instructions [here](./CONTRIBUTING.md).

## Installation 📦️

1. Have Python installed on your computer or in your virtual environment manager, i.e. [`pyenv`](https://github.com/pyenv/pyenv/blob/master/README.md). For this project, you'll need version 3.12 of Python.

2. Create a new virtual Python environment (version 3.12) and activate it.

3. Install this package with `pip`. Because it depends on several "heavy" Python libraries (i.e. `torch`), the installation may take several minutes. ☕

    a. _Option 1_: Install directly from the project's GitHub repository URL.

    b. _Option 2_: Download ("clone") the repository using `git` [(must be installed)](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), then install the downloaded files in your virtual Python environment.

### Option 1:

```shell
pip install git+https://github.com/LostMa-ERC/simMAtree.git
```

### Option 2:

> Note: Requires that you have [`git` installed](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) on your computer.

```shell
git clone https://github.com/LostMa-ERC/simMAtree.git
cd simMAtree
pip install .
```

4. Test the installation.

```console
$ simmatree-test
Looks good!
```

> Note: It's normal for the command to take a while. Some of the Python dependencies are very "heavy" and, when starting up, importing everything in the library can be slow.

## Usage ▶️

The script supports three tasks: `inference`, `generate` and `score`.

No matter the task in your experiment, prepare a configuration YAML file. Follow the [model here](./example.config_BD.yml).

When running any of the `simmatree` tasks, you'll need to provide your experiment's configuration file.

## Quick Start

### 1. Create a Configuration File

Create `experiment.yml`:

```yaml
generator:
  name: YuleAbundance  # or BirthDeathAbundance
  config:
    n_init: 1
    Nact: 1000
    Ninact: 1000
    max_pop: 50000

stats:
  name: Abundance
  config:
    additional_stats: true

prior:
  name: ConstrainedUniform4D  # or ConstrainedUniform2D for Birth-Death
  config:
    low: [0.0, 0.0, 0.0, 0.0]
    high: [1.0, 0.015, 0.01, 0.01]

params:
  LDA: 0.3      # Rate of new independent trees (Yule only)
  lda: 0.009    # Probability of copying/reproduction
  gamma: 0.001  # Probability of speciation (Yule only)
  mu: 0.0033    # Probability of death

inference:
  name: SBI
  config:
    method: NPE
    num_simulations: 500
    num_rounds: 2
    random_seed: 42
    num_samples: 500
    num_workers: 10
    device: cpu
```

This example performs all three simmatree tasks (`generate`, `score` and `infer`). 
Certain blocks of information need not be provided if only one of the three tasks is to be performed (e.g. params if you only wish to perform inference and have no ground truth).

### 2. Generate Synthetic Data

```bash
simmatree -c experiment.yml generate -o synthetic_data.csv -s 42
```

### 3. Run Inference

```bash
simmatree -c experiment.yml infer -i synthetic_data.csv -o results/
```

### 4. Evaluate Results

```bash
simmatree -c experiment.yml score -d results/
```
## Architecture

### Core Components

- **Generators** (`src/generator/`): Implement stochastic evolutionary models
  - `YuleWitness`: Full 4-parameter Yule process
  - `BirthDeathWitness`: Simplified 2-parameter Birth-Death process
  - `GeneralizedWitnessGenerator`: Base class with shared simulation logic

- **Statistics** (`src/stats/`): Extract summary statistics from simulated data
  - `AbundanceStats`: Witness count distributions and derived metrics

- **Priors** (`src/priors/`): Constrained uniform distributions
  - `ConstrainedUniform4D`: For Yule model with biological constraints
  - `ConstrainedUniform2D`: For Birth-Death model

- **Inference** (`src/inference/`): SBI backends
  - `SbiBackend`: Neural Posterior Estimation and related methods

- **CLI** (`src/cli/`): Command-line interface and configuration management## Outputs

## Output Files

### Inference Results
- `posterior_samples.npy`: Raw posterior samples
- `posterior_summary.csv`: Summary statistics (mean, quantiles, HPDI)
- `posterior_predictive.npy`: Posterior predictive samples
- `pp_summaries.png`: Posterior predictive check visualizations
- `posterior.png`: Marginal posterior distributions
- `pairplot.png`: Parameter correlation plots

### Evaluation Results
- `summary_metrics.csv`: RMSE, coverage probability, relative errors
- `relative_error.png`: Parameter-wise relative error analysis
- Additional diagnostic plots

## Testing

The project includes comprehensive tests:

```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test categories
python tests/run_all_tests.py --category unit
python tests/run_all_tests.py --category integration
python tests/run_all_tests.py --category e2e
```

## Contributing

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed development instructions, including:
- Setting up the development environment
- Code formatting with `ruff` and `isort`
- Pre-commit hooks
- Testing guidelines

## Acknowledgements

Funded by the European Union (ERC, LostMA, 101117408). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
