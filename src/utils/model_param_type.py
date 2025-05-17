from dataclasses import dataclass


@dataclass
class ModelParam:
    LDA: int
    lda: int
    mu: int
    gamma: int | None = None

    @classmethod
    def load_from_yaml(cls, yaml_dict: dict) -> "ModelParam":
        params = yaml_dict["model"]["params"]
        return ModelParam(**params)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}

    def list_names(self) -> list:
        return list(self.to_dict().keys())

    def list_values(self) -> list:
        return list(self.to_dict().values())
