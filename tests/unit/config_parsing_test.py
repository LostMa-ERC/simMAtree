import tempfile
import unittest
from pathlib import Path

import pytest
import yaml

from generator.birth_death_abundance import BirthDeathAbundance
from generator.yule_abundance import YuleAbundance
from src.cli.config import Config
from src.inference.sbi_backend import SbiBackend
from src.priors.constrained_uniform_2D import ConstrainedUniform2DPrior
from src.priors.constrained_uniform_4D import ConstrainedUniform4DPrior
from src.stats.abundance_stats import AbundanceStats


@pytest.mark.unit
class YamlParsingTest(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file for testing
        self.yule_config = {
            "generator": {
                "name": "YuleAbundance",
                "config": {"n_init": 1, "Nact": 100, "Ninact": 100, "max_pop": 1000},
            },
            "stats": {"name": "Abundance", "config": {"additional_stats": True}},
            "prior": {
                "name": "ConstrainedUniform4D",
                "config": {
                    "low": [0.0, 0.0, 0.0, 0.0],
                    "high": [1.0, 0.015, 0.01, 0.01],
                },
            },
            "params": {"LDA": 0.3, "lda": 0.009, "gamma": 0.001, "mu": 0.0033},
            "inference": {
                "name": "SBI",
                "config": {
                    "method": "NPE",
                    "num_simulations": 10,
                    "num_rounds": 1,
                    "random_seed": 42,
                    "num_samples": 10,
                    "num_workers": 1,
                    "device": "cpu",
                },
            },
        }

        self.bd_config = {
            "generator": {
                "name": "BirthDeathAbundance",
                "config": {"n_init": 1, "Nact": 100, "Ninact": 100, "max_pop": 1000},
            },
            "stats": {"name": "Abundance", "config": {"additional_stats": True}},
            "prior": {
                "name": "ConstrainedUniform2D",
                "config": {"low": [0.0, 0.0], "high": [0.01, 0.01]},
            },
            "params": {"lda": 0.009, "mu": 0.0033},
            "inference": {
                "name": "SBI",
                "config": {
                    "method": "NPE",
                    "num_simulations": 10,
                    "num_rounds": 1,
                    "random_seed": 42,
                    "num_samples": 10,
                    "num_workers": 1,
                    "device": "cpu",
                },
            },
        }

    def _create_temp_config(self, config_dict):
        """Create a temporary config file"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name

    def test_yule_yaml_parsing(self):
        config_path = self._create_temp_config(self.yule_config)
        try:
            conf = Config(config_path)
            self.assertIsInstance(conf.generator, YuleAbundance)
            self.assertIsInstance(conf.backend, SbiBackend)
            self.assertIsInstance(conf.stats, AbundanceStats)
            self.assertIsInstance(conf.prior, ConstrainedUniform4DPrior)
            self.assertEqual(len(conf.params), 4)
        finally:
            Path(config_path).unlink()

    def test_birth_death_yaml_parsing(self):
        config_path = self._create_temp_config(self.bd_config)
        try:
            conf = Config(config_path)
            self.assertIsInstance(conf.generator, BirthDeathAbundance)
            self.assertIsInstance(conf.backend, SbiBackend)
            self.assertIsInstance(conf.stats, AbundanceStats)
            self.assertIsInstance(conf.prior, ConstrainedUniform2DPrior)
            self.assertEqual(len([v for v in conf.params.values() if v is not None]), 2)
        finally:
            Path(config_path).unlink()

    def test_parameter_consistency_validation(self):
        """Test that config validates parameter count consistency"""
        # This should raise an error: 4D prior with 2 parameters
        bad_config = self.yule_config.copy()
        bad_config["params"] = {"lda": 0.009, "mu": 0.0033}  # Only 2 params

        config_path = self._create_temp_config(bad_config)
        try:
            with self.assertRaises(ValueError):
                Config(config_path)
        finally:
            Path(config_path).unlink()


@pytest.mark.unit
class ConfigImportsTest(unittest.TestCase):
    def test_yule_generator_import(self):
        model = Config.import_class(name="YuleAbundance")
        self.assertEqual(model, YuleAbundance)

    def test_birth_death_generator_import(self):
        model = Config.import_class(name="BirthDeathAbundance")
        self.assertEqual(model, BirthDeathAbundance)

    def test_sbi_backend_import(self):
        backend = Config.import_class(name="SBI")
        self.assertEqual(backend, SbiBackend)

    def test_stats_import(self):
        stats = Config.import_class(name="Abundance")
        self.assertEqual(stats, AbundanceStats)

    def test_prior_imports(self):
        prior_2d = Config.import_class(name="ConstrainedUniform2D")
        self.assertEqual(prior_2d, ConstrainedUniform2DPrior)

        prior_4d = Config.import_class(name="ConstrainedUniform4D")
        self.assertEqual(prior_4d, ConstrainedUniform4DPrior)


if __name__ == "__main__":
    unittest.main()
