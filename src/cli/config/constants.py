from enum import Enum

from pydantic import BaseModel, Field


class ModelImports(Enum):
    # Generators
    YULEABUNDANCE = ("src.generator.yule_abundance", "YuleAbundance")
    YULEBDABUNDANCE = ("src.generator.hybrid_yuleBD_abundance", "YuleBDAbundance")
    BIRTHDEATHABUNDANCE = ("src.generator.birth_death_abundance", "BirthDeathAbundance")
    BIRTHDEATHABUNDANCESINGLETREE = (
        "src.generator.birth_death_abundance_single_tree",
        "BirthDeathAbundanceSingleTree",
    )
    BIRTHDEATHTREE = ("src.generator.birth_death_tree", "BirthDeathTree")
    TWOSTATESBDGENERATOR = (
        "src.generator.twostates_BD_generator",
        "TwoStatesBDGenerator",
    )

    # Backends
    SBI = ("src.inference.sbi_backend", "SbiBackend")

    # Stats
    ABUNDANCE = ("src.stats.abundance_stats", "AbundanceStats")
    TOPOLOGY = ("src.stats.topology_stats", "TreeTopologyStats")

    # Priors
    CONSTRAINEDUNIFORM2D = (
        "src.priors.constrained_uniform_2D",
        "ConstrainedUniform2DPrior",
    )
    CONSTRAINEDUNIFORM3D = (
        "src.priors.constrained_uniform_3D",
        "ConstrainedUniform3DPrior",
    )
    CONSTRAINEDUNIFORM4D = (
        "src.priors.constrained_uniform_4D",
        "ConstrainedUniform4DPrior",
    )
    HYBRIDCONSTRAINEDUNIFORM3D = (
        "src.priors.hybrid_constrained_uniform_3D",
        "HybridConstrainedUniform3DPrior",
    )


class ExperimentParamters(BaseModel):
    LDA: float | None = Field(default=None)
    lda: float | None = Field(default=None)
    gamma: float | None = Field(default=None)
    mu: float | None = Field(default=None)
    r: float | None = Field(default=None)
