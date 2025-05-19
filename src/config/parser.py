from importlib import import_module

from yaml import safe_load

from .exceptions import InvalidConfigValue
from .types import ModelImports, ExperimentParamters


class Config(object):
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.yaml: dict = safe_load(f)

    @property
    def model(self) -> object:
        name = self.yaml["model"]["name"]
        config = self.yaml["model"]["config"]
        return self.create_class(code_name=name, params=config)

    @property
    def backend(self) -> object | None:
        if not self.yaml.get("inference"):
            return
        name = self.yaml["inference"]["name"]
        config = self.yaml["inference"]["config"]
        return self.create_class(code_name=name, params=config)

    @property
    def params(self) -> ExperimentParamters:
        conf = self.yaml["params"]
        return ExperimentParamters.model_validate(conf)

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
