# Bayesian Inference for Witness Analysis

## Description
This repository performs an Approximate Bayesian Computation algorithm on witness distribution data modeled by a Yule process, using PyMC library. 

## Prerequisites
- Required Python libraries:
  ```
  pandas
  numpy
  pymc
  arviz
  ```

## Usage

### Basic Usage

1. Prepare your data in CSV format with columns 'text_ID' and 'witness_ID'

2. Configure your model parameters in a JSON file (e.g., `yule_param_simul.json`):
```json
{
    "type": "yule",
    "n_init": 1,
    "Nact": 1000,
    "Ninact": 1000,
    "max_pop": 100000
}
```

3. Configure your inference parameters in a JSON file (e.g., `pymc_config.json`):
```json
{
    "backend": "pymc",
    "draws": 10,
    "chains": 4,
    "random_seed": 42,
    "epsilon": 1,
    "sum_stat": "identity",
    "distance": "gaussian"
}
```

4. Run the inference:
```bash
python run.py --data_path path/to/your/data.csv \
              --model_config path/to/yule_param_simul.json \
              --inference_config path/to/pymc_config.json \
              --results_dir results/
```

The script will:
1. Load and process your manuscript data
2. Set up the Yule model with specified parameters
3. Run Bayesian inference using PyMC
4. Save results and generate visualizations in the specified output directory

## Model Parameters

The Yule model requires the following parameters:
- `Nact`: duration of the active reproduction phase
- `Ninact`: duration of the decimation (pure death phase)
- `n_init`: number of initialy living independant nodes at t=0
- `max_pop`: maximum number of active population


## Inference Parameters

The PyMC backend require these configuration parameters 
- `draws`: Number of samples to draw
- `chains`: Number of MCMC chains
- `random_seed`: Random seed for reproducibility
- `epsilon`: Epsilon value for ABC inference
- `sum_stat`: Summary statistics type
- `distance`: Distance metric for ABC

See the corresponding PyMC documentation for more details: [here](https://www.pymc.io/projects/docs/en/latest/api/generated/pymc.smc.sample_smc.html) and [here](https://www.pymc.io/projects/docs/en/stable/api/distributions/simulator.html)


## Outputs
The script generates two output files:
1. `pp_summaries.png`: Visualization of posterior predictives checks of differents statistics
2. `results_summary.csv`: Statistical summary of results
3. `traces.png`, `posterior.png`, `posterior_pairs.png`: different visulations of the inference result

