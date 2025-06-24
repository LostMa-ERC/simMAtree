import torch

from .generalized_abundance import GeneralizedAbundanceGenerator


class BirthDeathAbundanceSingleTree(GeneralizedAbundanceGenerator):
    """
    Optimized generator for abundance counts according to Birth-Death model

    - lda: Probability of copying/reproduction
    - mu: Probability of death

    Constraints: lda > mu
    """

    def __init__(self, n_init, Nact, Ninact, max_pop):
        self.param_count = 2
        super().__init__(n_init, Nact, Ninact, max_pop)

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
