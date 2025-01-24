import sys
import numpy as np
import json

from src.yule import generate_yule_dataset
from src.stats import compute_stat_witness, inverse_compute_stat_witness
from run_yule import HYPERPARAMS


sys.setrecursionlimit(20000)
rng = np.random.default_rng(42)

output_path = "data/simulation.csv"
params_path = "params/yule_params.json"


with open(params_path) as f:
    params = json.load(f)

result = generate_yule_dataset(rng, params, HYPERPARAMS, output_path)

witness_counts = list(result.groupby('text_ID')['witness_ID'].count().sort_values(ascending=False))
s = inverse_compute_stat_witness(compute_stat_witness(witness_counts))

print("DONE!\n")
print(f"\nNombre Témoins: {s[0]}")
print(f"Nombre Oeuvres: {s[1]}")
print(f"Max Témoins: {s[2]}")
print(f"Nombre de 1: {s[4]}")
