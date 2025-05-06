# Bayesian Inference for Witness Analysis
[![DOI](https://zenodo.org/badge/900725844.svg)](https://doi.org/10.5281/zenodo.15350477)

## Description
This repository performs an Approximate Bayesian Computation algorithm on witness distribution data modeled by a Yule process, using PyMC library. 

## Prerequisites
- Required Python libraries:
  ```
  pandas
  numpy
  pymc
  arviz
  torch
  sbi
  tqdm
  ```

A requirements.txt file is provided is provided for an example of a functional environment.

## Usage


The script supports two tasks: `inference`, `generate` and `score`

### Data Generation

Generate synthetic witness data using a Yule model:

```bash
python run.py --task generate \
              --data_path output/synthetic_data.csv \
              --model_config configs/yule_param_simul.json
```

### Inference

Run Bayesian inference on witness data:

```bash
python run.py --task inference \
              --data_path path/to/your/data.csv \
              --model_config configs/yule_param_simul.json \
              --inference_config configs/pymc_config.json \
              --results_dir results/
```

### Score

Evaluate inference results against known parameters:

```bash
python run.py --task score \
              --model_config params/yule.json \
              --results_dir results/ \
              --true_params path/to/true_params.json
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

## Configuration Files

### Model configuration (e.g., `yule_params.json`):
```json
{
    "module_name": "models.yule_model",
    "class_name": "YuleModel",
    "n_init": 1,
    "Nact": 1000,
    "Ninact": 1000,
    "max_pop": 500000,
    "params": {
        "LDA": 0.3,
        "lda": 0.012,
        "gamma": 0.001,
        "mu": 0.0033
    }
}
```

Common parameters for all models:
- `n_init`: Number of initially living independent nodes at t=0
- `Nact`: Duration of the active reproduction phase
- `Ninact`: Duration of the decimation (pure death phase)
- `max_pop`: Maximum number of active population

### Inference configuration (e.g., `pymc_config.json`):
```json
{
    "module_name": "inference.pymc_backend",
    "class_name": "PymcBackend",
    "draws": 5,
    "chains": 4,
    "random_seed": 42,
    "epsilon": 1,
    "sum_stat": "identity",
    "distance": "gaussian"
}
```

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

## Example Workflow

1. Generate synthetic data with known parameters:
```bash
python run.py --task generate \
              --data_path synthetic/data.csv \
              --model_config params/birth_death_poisson.json
```

2. Run inference on the synthetic data:
```bash
python run.py --task inference \
              --data_path synthetic/data.csv \
              --model_config params/birth_death_poisson.json \
              --inference_config params/sbi_config.json \
              --results_dir results/birth_death/
```

3. Evaluate inference quality:
```bash
python run.py --task score \
              --model_config params/birth_death_poisson.json \
              --results_dir results/birth_death/ \
              --true_params params/birth_death_poisson.json
```

For multiple replicates and comprehensive evaluation:
```bash
./eval_on_sim.sh params/birth_death_poisson.json params/sbi_config.json results/evaluation/ 10
```
