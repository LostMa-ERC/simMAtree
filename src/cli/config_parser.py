from importlib import import_module

from yaml import safe_load

from src.inference.base_backend import InferenceBackend
from src.models.base_model import BaseModel
from src.utils.model_param_type import ModelParam


class Config(object):
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.yaml = safe_load(f)

    @classmethod
    def create_class_instance(cls, conf: dict) -> BaseModel | InferenceBackend:
        module_name = conf["module"]
        class_name = conf["class"]
        params = conf["config"]

        module_path = f"src.{module_name}"
        module = import_module(module_path)

        imported_class = getattr(module, class_name)
        return imported_class(**params)

    def parse_model_config(self) -> BaseModel:
        if not self.yaml.get("model"):
            raise KeyError("Missing model config")
        else:
            model_conf = self.yaml["model"]

        return self.create_class_instance(conf=model_conf)

    def parse_backend_config(self) -> InferenceBackend:
        if not self.yaml.get("inference"):
            raise KeyError("Missing inference config")
        else:
            inference_conf = self.yaml["inference"]

        return self.create_class_instance(conf=inference_conf)

    def parse_experiment_parameters(self) -> ModelParam:
        return ModelParam.load_from_yaml(yaml_dict=self.yaml)
