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
The script can be run from the command line with various parameters:

```bash
python main.py --data_path path/to/data.csv --draws 100 --chains 4
```

### Parameters
- `--data_path` (required): Path to the CSV file containing witness data
- `--draws` (optional): Number of samples to generate (default: 10)
- `--chains` (optional): Number of MCMC chains (default: 4)
- `--model_path` (optional): Path to a custom model file

### Data Format
The input CSV file must contain the following columns:
- `text_ID`: Text identifier
- `witness_ID`: Witness identifier

## Outputs
The script generates two output files:
1. `pp_summaries.png`: Visualization of posterior predictives checks of differents statistics
2. `results_summary.csv`: Statistical summary of results
3. `traces.png`, `posterior.png`, `posterior_pairs.png`: different visulations of the inference result

## Model

The model uses four main parameters:
- `LDA`: birth rate of independant trees/metatradition
- `lda`: birth rate of nodes (simple copy)
- `gamma`: birth rate of nodes with different species than parent (speciation rate)
- `mu`: death rate of nodes

Along with others hyperparameters:
- `Nact`: duration of the active reproduction phase
- `Ninact`: duration of the decimation (pure death phase)
- `n_init`: number of initialy living independant nodes at t=0
- `max_pop`: maximum number of active population

