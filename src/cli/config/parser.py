from importlib import import_module

from yaml import safe_load

from src.generator import BaseGenerator
from src.inference import AbstractInferenceClass
from src.priors import BasePrior
from src.priors.pytorch_priors import PyTorchPrior
from src.stats import AbstractStatsClass

from .constants import ExperimentParamters, ModelImports
from .exceptions import InvalidConfigValue


class Config(object):
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.yaml: dict = safe_load(f)

        self._validate_parameter_consistency()

    def _validate_parameter_consistency(self) -> None:
        """
        Validate that generator, prior, and params have consistent parameter counts
        """
        actual_param_count = len([v for v in self.params.values() if v is not None])

        if self.yaml.get("inference", {}).get("config", {}).get("posterior_path"):
            return

        if "generator" in self.yaml:
            generator_param_count = self.generator.param_count

            # Check generator vs params consistency
            if actual_param_count != 0 and generator_param_count != actual_param_count:
                raise ValueError(
                    f"Parameter count mismatch: generator '{self.yaml['generator']['name']}' "
                    f"expects {generator_param_count} parameters, "
                    f"but {actual_param_count} parameters provided in params."
                )

        if "inference" in self.yaml:
            prior = self.prior

            # Get dimension differently for PyTorch vs custom distributions
            if hasattr(prior, "dimension"):
                # Custom distribution
                prior_dimension = prior.dimension
            else:
                # PyTorch distribution
                prior_dimension = PyTorchPrior.get_dimension(prior)

            if actual_param_count != 0 and prior_dimension != actual_param_count:
                raise ValueError(
                    f"Parameter count mismatch: prior"
                    f"expects {prior_dimension} parameters, "
                    f"but {actual_param_count} parameters provided in params."
                )

    @property
    def stats(self) -> AbstractStatsClass:
        if not self.yaml.get("stats"):
            return None
        name = self.yaml["stats"]["name"]
        config = self.yaml["stats"]["config"]
        return self.create_class(code_name=name, params=config)

    @property
    def generator(self) -> BaseGenerator:
        name = self.yaml["generator"]["name"]
        config = self.yaml["generator"]["config"]
        return self.create_class(code_name=name, params=config)

    @property
    def prior(self) -> BasePrior:
        prior_config = self.yaml["prior"]
        name = prior_config["name"]
        config = prior_config["config"]

        # Check if it's a PyTorch distribution first
        if PyTorchPrior.is_pytorch_distribution(name):
            return PyTorchPrior.create_distribution(name, config)

        # Fallback to existing custom distributions
        else:
            params = config | {"hyperparams": self.yaml["generator"]["config"]}
            return self.create_class(code_name=name, params=params)

    @property
    def backend(self) -> AbstractInferenceClass | None:
        if not self.yaml.get("inference"):
            return
        name = self.yaml["inference"]["name"]
        config = self.yaml["inference"]["config"]
        return self.create_class(code_name=name, params=config)

    @property
    def params(self) -> dict:
        if not self.yaml.get("params"):
            return {}
        conf = self.yaml["params"]
        validated_data = ExperimentParamters.model_validate(conf)
        return validated_data.model_dump()

    @classmethod
    def import_class(cls, name: str) -> object:
        try:
            module_name, _class = ModelImports[name.upper()].value
        except ValueError:
            raise InvalidConfigValue(name)
        module = import_module(module_name)
        return getattr(module, _class)

    @classmethod
    def create_class(cls, code_name: str, params: dict) -> object:
        _class = cls.import_class(name=code_name)
        return _class(**params)
