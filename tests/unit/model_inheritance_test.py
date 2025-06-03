import unittest

from src.models.birth_death_poisson import BirthDeath
from src.models.yule_model import YuleModel

BASIC_MODEL_CONFIG = {
    "n_init": 1,
    "Nact": 1,
    "Ninact": 1,
    "max_pop": 1,
}

FULL_MODEL_CONFIG = BASIC_MODEL_CONFIG | {
    "LDA": 0.0,
    "lda": 0.0,
    "gamma": 0.0,
    "mu": 0.0,
}


class ModelInheritanceTest(unittest.TestCase):
    def test_yule_with_params(self):
        model = YuleModel(**FULL_MODEL_CONFIG)
        self.assertIsInstance(model, YuleModel)

    def test_yule_without_params(self):
        model = YuleModel(**BASIC_MODEL_CONFIG)
        self.assertIsInstance(model, YuleModel)

    def test_bdp_with_params(self):
        model = BirthDeath(**FULL_MODEL_CONFIG)
        self.assertIsInstance(model, BirthDeath)

    def test_bdp_without_params(self):
        model = BirthDeath(**BASIC_MODEL_CONFIG)
        self.assertIsInstance(model, BirthDeath)


if __name__ == "__main__":
    unittest.main()
