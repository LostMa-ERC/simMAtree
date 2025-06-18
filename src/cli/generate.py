import json
from typing import Union

import numpy as np

from src.generator import BaseGenerator
from src.stats import AbstractStatsClass


def generate(
    data_path: str,
    generator: BaseGenerator,
    parameters: Union[list, tuple, dict],
    stats: AbstractStatsClass | None = None,
    seed: int = 42,
    show_params: bool = False,
):
    rng = np.random.default_rng(seed)

    if show_params:
        print("\nPARAMETERS_JSON_START")
        print(json.dumps(parameters))
        print("PARAMETERS_JSON_END")

    print("Generating population...\n")
    pop = generator.generate(rng, parameters, verbose=True)
    _ = generator.save_simul(pop, data_path)

    if stats is not None:
        stats.print_stats(pop)

    print("\nDone!")

    return True
