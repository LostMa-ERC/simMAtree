import sys
import numpy as np

from src.yule import generate_yule_dataset
from src.stats import compute_stat_witness, inverse_compute_stat_witness
from main import HYPERPARAMS


sys.setrecursionlimit(20000)
rng = np.random.default_rng(42)

output_path = "data/simulation.csv"

PARAMS = {
    "LDA" : 0.003,
    "lda" : 0.012,
    "gamma" : 0.001,
    "mu" : 0.0033
}

result = generate_yule_dataset(rng, PARAMS, HYPERPARAMS, output_path)

witness_counts = list(result.groupby('text_ID')['witness_ID'].count().sort_values(ascending=False))
s = inverse_compute_stat_witness(compute_stat_witness(witness_counts))

print("DONE!\n")
print(f"\nNombre Témoins: {s[0]}")
print(f"Nombre Oeuvres: {s[1]}")
print(f"Max Témoins: {s[2]}")
print(f"Nombre de 1: {s[4]}")
