import torch

from .generalized_witness import GeneralizedWitnessGenerator


class BirthDeathWitness(GeneralizedWitnessGenerator):
    """
    Optimized generator for witness counts according to Birth-Death model

    - lda: Probability of copying/reproduction
    - mu: Probability of death
    - LDA: Fixed at 0 (no new independent trees)
    - gamma: Fixed at 0 (no speciation)

    Constraints: lda > mu
    """

    def _extract_params(self, params):
        # TODO: Check where the params are generated and see if the data type
        # can be replaced by something more stable (dataclass, Pydantic model,
        # namedtuple, etc.)
        if isinstance(params, torch.Tensor):
            lda = params[0].item()
            mu = params[1].item()
        elif isinstance(params, dict):
            lda = params["lda"]
            mu = params["mu"]
        else:
            lda = params[0]
            mu = params[1]
            try:
                if not isinstance(lda, float) and hasattr(lda, "__getitem__"):
                    lda = lda[0]
                    mu = mu[1]
            except Exception:
                pass
        return {"LDA": 0, "lda": lda, "mu": mu, "gamma": 0}
