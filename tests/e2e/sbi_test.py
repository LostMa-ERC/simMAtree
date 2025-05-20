import unittest
from pathlib import Path

from src.cli.config import Config
from src.cli.generate import generate
from src.cli.inference import inference
from tests.constants import MOCK_DIR

INFERENCE_YAML_YULE_SBI = MOCK_DIR.joinpath("yule_sbi_inference_config.yml")
SIMULATION_DATA = MOCK_DIR.joinpath("simulation_data.csv")


class End2EndSBITest(unittest.TestCase):
    def setUp(self):
        self.config = Config(config_path=INFERENCE_YAML_YULE_SBI)
        self.outdir = Path(__file__).parent.joinpath("results")
        self.outdir.mkdir(exist_ok=True)
        return super().setUp()

    def test_generate(self):
        generate(data_path=SIMULATION_DATA, model=self.config.model)

    def test_infererence(self):
        model = self.config.model
        backend = self.config.backend
        inference(
            csv_file=SIMULATION_DATA,
            model=model,
            backend=backend,
            dir=self.outdir,
        )

    def tearDown(self):
        for f in self.outdir.iterdir():
            f.unlink()
        self.outdir.rmdir()
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
