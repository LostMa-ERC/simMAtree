from enum import Enum

from pydantic import BaseModel, Field


class ModelImports(Enum):
    # Generators
    YULEABUNDANCE = ("src.generator.yule_witness", "YuleWitness")
    BIRTHDEATHABUNDANCE = ("src.generator.birth_death_witness", "BirthDeathWitness")

    # Backends
    SBI = ("src.inference.sbi_backend", "SbiBackend")

    # Stats
    ABUNDANCE = ("src.stats.abundance_stats", "AbundanceStats")

    # Priors
    CONSTRAINEDUNIFORM2D = (
        "src.priors.constrained_uniform_2D",
        "ConstrainedUniform2DPrior",
    )
    CONSTRAINEDUNIFORM4D = (
        "src.priors.constrained_uniform_4D",
        "ConstrainedUniform4DPrior",
    )


class ExperimentParamters(BaseModel):
    LDA: float | None = Field(default=None)
    lda: float
    gamma: float | None = Field(default=None)
    mu: float
