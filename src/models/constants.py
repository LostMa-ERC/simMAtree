from pydantic import BaseModel, ConfigDict, Field
from pytensor.tensor.variable import TensorVariable


class PyMCPriors(BaseModel):
    LDA: TensorVariable
    lda: TensorVariable
    mu: TensorVariable
    gamma: TensorVariable | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def list(self) -> list:
        _all = [self.LDA, self.lda, self.mu, self.gamma]
        return [i for i in _all if i is not None]
