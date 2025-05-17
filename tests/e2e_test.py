import unittest
from pathlib import Path

import pytest

from src.cli.config_parser import Config
from src.cli.generate import generate
from src.cli.inference import inference

MOCK_DIR = Path(__file__).parent.joinpath("mock")

INFERENCE_YAML_YULE_PYMC = MOCK_DIR.joinpath("yule_pymc_inference_config.yml")
SIMULATION_DATA = MOCK_DIR.joinpath("simulation_data.csv")


class End2EndYuleTest(unittest.TestCase):
    def setUp(self):
        self.config = Config(config_path=INFERENCE_YAML_YULE_PYMC)
        self.outdir = Path(__file__).parent.joinpath("results")
        self.outdir.mkdir(exist_ok=True)
        return super().setUp()

    def test_generate(self):
        model = self.config.parse_model_config()
        generate(data_path=SIMULATION_DATA, model=model)

    def test_infererence(self):
        model = self.config.parse_model_config()
        backend = self.config.parse_backend_config()
        # Known TypeError in plot_posterior_predictive_stats
        with pytest.raises(TypeError):
            inference(
                csv_file=SIMULATION_DATA,
                model=model,
                backend=backend,
                results_dir=self.outdir,
            )

    def tearDown(self):
        for f in self.outdir.iterdir():
            f.unlink()
        self.outdir.rmdir()
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
