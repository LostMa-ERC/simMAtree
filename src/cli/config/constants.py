from enum import Enum

from pydantic import BaseModel, Field


class ModelImports(Enum):
    YULE = ("src.models.yule_model", "YuleModel")
    BIRTHDEATH = ("src.models.birth_death_poisson", "BirthDeath")
    PYMC = ("src.inference.pymc_backend", "PymcBackend")
    SBI = ("src.inference.sbi_backend", "SbiBackend")


class ExperimentParamters(BaseModel):
    LDA: float
    lda: float
    gamma: float | None = Field(default=None)
    mu: float
