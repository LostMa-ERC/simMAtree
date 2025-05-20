import unittest
from pathlib import Path

import arviz

from src.utils import visualisation
from tests.constants import MOCK_DIR

MOCK_DATA = MOCK_DIR.joinpath("pymc_inference_data.nc")


class PymcVisTest(unittest.TestCase):
    def setUp(self):
        self.data = arviz.from_netcdf(MOCK_DATA)
        self.outdir = Path(__file__).parent.joinpath("vis")
        self.outdir.mkdir(exist_ok=True)
        return super().setUp()

    def test_inference_checks(self):
        visualisation.plot_inference_checks(
            idata=self.data,
            output_dir=self.outdir,
        )

    @unittest.skip("Known problem")
    def test_predictive_stats(self):
        # TODO: See line 302
        # flat_samples = samples.values.reshape(-1, samples.shape[-1])
        # AttributeError: 'function' object has no attribute 'reshape'
        visualisation.plot_posterior_predictive_stats(
            samples=self.data,
            output_dir=self.outdir,
        )

    def tearDown(self):
        for f in self.outdir.iterdir():
            f.unlink()
        self.outdir.rmdir()
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
