# Bayesian Inference for Witness Analysis
[![DOI](https://zenodo.org/badge/900725844.svg)](https://doi.org/10.5281/zenodo.15350477)

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Python package](https://github.com/LostMa-ERC/BayesYule/actions/workflows/ci.yml/badge.svg)](https://github.com/LostMa-ERC/BayesYule/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://www.gnu.org/licenses/MIT)

## Description
This repository performs an Approximate Bayesian Computation algorithm on witness distribution data modeled by a Yule process, using PyMC library.

## Contributing 🔧

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

To develop and/or contribute to the project, see more detailed instructions [here](./CONTRIBUTING.md).

## Installation 📦️

1. Have Python installed on your computer or in your virtual environment manager, i.e. [`pyenv`](https://github.com/pyenv/pyenv/blob/master/README.md). For this project, you'll need version 3.12 or 3.13 of Python.

2. Create a new virtual Python environment (version 3.12 or 3.13) and activate it.

3. Install this package with `pip`. Because it depends on several "heavy" Python libraries (i.e. `torch`), the installation may take several minutes. ☕

    a. _Option 1_: Install directly from the project's GitHub repository URL.

    b. _Option 2_: Download ("clone") the repository using `git` [(must be installed)](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), then install the downloaded files in your virtual Python environment.

### Option 1:

```shell
pip install git+https://github.com/LostMa-ERC/BayesYule.git
```

### Option 2:

> Note: Requires that you have [`git` installed](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) on your computer.

```shell
git clone https://github.com/LostMa-ERC/BayesYule.git
cd simetree
pip install .
```

4. Test the installation.

```console
$ simetree-test
Looks good!
```

> Note: It's normal for the command to take a while. Some of the Python dependencies are very "heavy" and, when starting up, importing everything in the library can be slow.

## Usage ▶️

The script supports three tasks: `inference`, `generate` and `score`.

No matter the task in your experiment, prepare a configuration YAML file. Follow the [model here](./example.config.yml).

When running any of the `simetree` tasks, you'll need to provide your experiment's configuration file.

### Data Generation

Generate synthetic witness data and write it to a CSV file.

1. Set up the relevant parts of your experiment's configuration file.

```yaml
model:
  name: Yule
  config:
    n_init: 1
    Nact: 1000
    Ninact: 1000
    max_pop: 500000
```

2. Run the generate task.

```shell
simetree -c <CONFIG FILE> generate -o <OUTPUT FILE>
```

### Inference

Run inference on witness data.

1. Set up the relevant parts of your experiment's configuration file.

```yaml
model:
  name: Yule
  config:
    n_init: 1
    Nact: 1000
    Ninact: 1000
    max_pop: 500000

inference:
  name: SBI
  config:
    method : NPE
    num_simulations : 10
    num_rounds : 1
    random_seed : 42
    num_samples : 10
    num_workers : 10
    device : cpu
```

2. Run the inference task.

```shell
simetree -c <CONFIG FILE> infer -i <DATA FILE> -o <OUTPUT DIRECTORY>
```

### Score

Evaluate inference results against known parameters.

1. Set up the relevant parts of your configuration file.

```yaml
model:
  name: Yule
  config:
    n_init: 1
    Nact: 1000
    Ninact: 1000
    max_pop: 500000

# Ground truth for scoring and generation
params:
  LDA: 0.3
  lda: 0.012
  gamma: 0.001
  mu: 0.0033
```

2. Run the score task.

```shell
simetree -c <CONFIG FILE> score -o <OUTPUT DIRECTORY>
```

### Example Workflow

1. Set up a configuration file, i.e. `experiment_1.yml`.

```yaml
model:
  name: Yule
  config:
    n_init: 1
    Nact: 1000
    Ninact: 1000
    max_pop: 500000

params:
  LDA: 0.3
  lda: 0.012
  gamma: 0.001
  mu: 0.0033

inference:
  name: SBI
  config:
    method : NPE
    num_simulations : 10
    num_rounds : 1
    random_seed : 42
    num_samples : 10
    num_workers : 10
    device : cpu
```

2. Generate synthetic data with known parameters.

```shell
simetree -c experiment_1.yml generate -o synthetic_data.csv
```

3. Run inference on the synthetic data.

```shell
simetree -c experiment_1.yml infer -i synthetic_data.csv -o results/
```

4. Evaluate inference quality.

```shell
simetree -c experiment_1.yml score -d results/
```

## Models

### YuleModel
A stochastic model based on the Yule process, with birth, death, and speciation events.

Parameters:
- `LDA`: Probability of new independent trees
- `lda`: Probability of copying (reproduction)
- `gamma`: Probability of speciation
- `mu`: Probability of death

### BirthDeathPoisson
A simpler model with Poisson-distributed events.

Parameters:
- `LDA`: Rate of new independent populations
- `lda`: Probability of copying
- `mu`: Probability of death

## Inference Backends

### PyMC Backend
Uses Approximate Bayesian Computation with Sequential Monte Carlo through PyMC.

Configuration parameters:
- `draws`: Number of samples to draw
- `chains`: Number of MCMC chains
- `random_seed`: Random seed for reproducibility
- `epsilon`: Epsilon value for ABC inference
- `sum_stat`: Summary statistics type
- `distance`: Distance metric for ABC

### SBI Backend
Uses simulation-based inference with neural networks through the SBI library.

Configuration parameters:
- `method`: Method to use (NPE, SNPE, etc.)
- `num_simulations`: Number of simulations per round
- `num_rounds`: Number of training rounds
- `random_seed`: Random seed for reproducibility
- `num_samples`: Number of posterior samples
- `num_workers`: Number of parallel workers
- `device`: Computation device (cpu/cuda)

## Statistics Computation

The project uses several summary statistics to analyze witness distributions:
- Total number of witnesses
- Number of unique works
- Maximum number of witnesses per work
- Median number of witnesses per work
- Proportion of works with single witnesses
- Additional quantile-based statistics

## Outputs

### For Inference Task
The script generates:
1. `pp_summaries.png`: Visualization of posterior predictive checks of different statistics
2. `results_summary.csv`: Statistical summary of inference results
3. `traces.png`: MCMC trace plots (PyMC backend only)
4. `posterior.png`: Posterior distribution plots
5. `posterior_pairs.png`: Pair plots showing parameter correlations
6. Posterior samples saved as NumPy arrays

### For Evaluation Task
The script generates:
1. `param_comparison.png`: Comparison of true vs. inferred parameters
2. `bias.png`: Bias analysis for each parameter
3. `relative_error.png`: Relative error analysis
4. `summary_metrics.csv`: Summary of evaluation metrics


## Acknowledgements

Funded by the European Union (ERC, LostMA, 101117408). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
