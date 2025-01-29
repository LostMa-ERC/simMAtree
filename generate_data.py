import sys
import numpy as np
import pandas as pd
import json

from models.yule_model import YuleModel
from utils.stats import compute_stat_witness, inverse_compute_stat_witness


sys.setrecursionlimit(20000)
rng = np.random.default_rng(42)

output_path = "data/simulation.csv"
params = {
        "LDA" : 0.003,
        "lda" : 0.012,
        "gamma" : 0.001,
        "mu" : 0.0033
}


def generate_yule_dataset(rng, params, hyperparams, output_path):
    LDA, lda, gamma, mu = params.values()
    model = YuleModel(hyperparams)
    yule_pop = model.simulate_pop(rng, LDA, lda, gamma, mu)
    text_val = []
    witness_val = []
    
    for i, num in enumerate(yule_pop):
        for j in range(num):
            text_val.append(f'T{i}')
            witness_val.append(f'W{i}-{j+1}')
    
    # Créer le dataframe
    df = pd.DataFrame({
        'witness_ID': witness_val,
        'text_ID': text_val
    })
    
    df.to_csv(output_path, sep=";", index=False)
    
    return df

HYPERPARAMS = "params/yule_param_simul.json"
result = generate_yule_dataset(rng, params, HYPERPARAMS, output_path)

witness_counts = list(result.groupby('text_ID')['witness_ID'].count().sort_values(ascending=False))
s = inverse_compute_stat_witness(compute_stat_witness(witness_counts))

print("DONE!\n")
print(f"\nNombre Témoins: {s[0]}")
print(f"Nombre Oeuvres: {s[1]}")
print(f"Max Témoins: {s[2]}")
print(f"Nombre de 1: {s[4]}")

