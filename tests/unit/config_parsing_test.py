import unittest

from src.config.parser import Config
from tests.constants import EXAMPLE_CONFIG


class YamlParsingTest(unittest.TestCase):
    def setUp(self):
        self.conf = Config(EXAMPLE_CONFIG)

    def test_example_yaml(self):
        from src.inference.sbi_backend import SbiBackend
        from src.models.yule_model import YuleModel

        self.assertIsInstance(self.conf.model, YuleModel)
        self.assertIsInstance(self.conf.backend, SbiBackend)
        self.assertIsInstance(self.conf.params, dict)


class ConfigImportsTest(unittest.TestCase):
    def test_yule_model_import(self):
        from src.models.yule_model import YuleModel

        model = Config.import_class(name="Yule")
        self.assertEqual(model, YuleModel)

    def test_birth_death_poisson_model_import(self):
        from src.models.birth_death_poisson import BirthDeathPoisson

        model = Config.import_class(name="BirthDeathPoisson")
        self.assertEqual(model, BirthDeathPoisson)

    def test_pymc_model_import(self):
        from src.inference.pymc_backend import PymcBackend

        model = Config.import_class(name="PYMC")
        self.assertEqual(model, PymcBackend)

    def test_sbi_model_import(self):
        from src.inference.sbi_backend import SbiBackend

        model = Config.import_class(name="SBI")
        self.assertEqual(model, SbiBackend)


if __name__ == "__main__":
    unittest.main()
