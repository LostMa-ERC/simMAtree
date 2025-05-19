from enum import Enum

from .types import ModelImports


class InvalidConfigValue(Exception):
    def __init__(self, input_value: str, valid_values=ModelImports):
        # Call the base class constructor with the parameters it needs
        super().__init__(
            self.make_message(input_value=input_value, valid_values=valid_values)
        )

    @classmethod
    def make_message(cls, input_value: str, valid_values: Enum) -> str:
        vs = ", ".join([i.name for i in valid_values])
        return f"\nInvalid value: {input_value}.\nValid values: {vs}"
