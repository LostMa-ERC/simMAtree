import torch

from .generalized_abundance import GeneralizedAbundanceGenerator


class BirthDeathAbundance(GeneralizedAbundanceGenerator):
    """
    Optimized generator for abundance counts according to Birth-Death model

    - lda: Probability of copying/reproduction
    - mu: Probability of death
    - LDA: Poisson probability of new independent trees

    Constraints: lda > mu
    """

    def __init__(self, n_init, Nact, Ninact, max_pop):
        self.param_count = 3
        super().__init__(n_init, Nact, Ninact, max_pop)

    def _extract_params(self, params):
        # TODO: Check where the params are generated and see if the data type
        # can be replaced by something more stable (dataclass, Pydantic model,
        # namedtuple, etc.)
        if isinstance(params, torch.Tensor):
            LDA = params[0].item()
            lda = params[1].item()
            mu = params[2].item()
        elif isinstance(params, dict):
            LDA = params["LDA"]
            mu = params["mu"]
            lda = params["lda"]
        else:
            LDA = params[0]
            lda = params[1]
            mu = params[2]
            try:
                if not isinstance(LDA, float) and hasattr(LDA, "__getitem__"):
                    LDA = LDA[0]
                    lda = lda[1]
                    mu = mu[2]
            except Exception:
                pass
        return {"LDA": LDA, "lda": lda, "mu": mu, "gamma": 0}
