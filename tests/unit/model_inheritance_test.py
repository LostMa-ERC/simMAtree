import unittest

from src.generator.birth_death_witness import BirthDeathWitness
from src.generator.yule_witness import YuleWitness
from src.stats.abundance_stats import AbundanceStats

BASIC_GENERATOR_CONFIG = {
    "n_init": 1,
    "Nact": 1,
    "Ninact": 1,
    "max_pop": 1000,
}

FULL_YULE_PARAMS = {
    "LDA": 0.0,
    "lda": 0.0,
    "gamma": 0.0,
    "mu": 0.0,
}

FULL_BD_PARAMS = {
    "lda": 0.0,
    "mu": 0.0,
}


class GeneratorInheritanceTest(unittest.TestCase):
    def test_yule_generator_creation(self):
        generator = YuleWitness(**BASIC_GENERATOR_CONFIG)
        self.assertIsInstance(generator, YuleWitness)
        self.assertEqual(generator.param_count, 4)

    def test_birth_death_generator_creation(self):
        generator = BirthDeathWitness(**BASIC_GENERATOR_CONFIG)
        self.assertIsInstance(generator, BirthDeathWitness)
        self.assertEqual(generator.param_count, 2)

    def test_yule_parameter_extraction(self):
        generator = YuleWitness(**BASIC_GENERATOR_CONFIG)

        # Test dict format
        params = generator._extract_params(FULL_YULE_PARAMS)
        expected = {"LDA": 0.0, "lda": 0.0, "gamma": 0.0, "mu": 0.0}
        self.assertEqual(params, expected)

        # Test list format
        params = generator._extract_params([0.1, 0.2, 0.3, 0.4])
        expected = {"LDA": 0.1, "lda": 0.2, "gamma": 0.3, "mu": 0.4}
        self.assertEqual(params, expected)

    def test_birth_death_parameter_extraction(self):
        generator = BirthDeathWitness(**BASIC_GENERATOR_CONFIG)

        # Test dict format
        params = generator._extract_params(FULL_BD_PARAMS)
        expected = {"LDA": 0, "lda": 0.0, "gamma": 0, "mu": 0.0}
        self.assertEqual(params, expected)

        # Test list format
        params = generator._extract_params([0.1, 0.2])
        expected = {"LDA": 0, "lda": 0.1, "gamma": 0, "mu": 0.2}
        self.assertEqual(params, expected)

    def test_parameter_validation(self):
        yule_gen = YuleWitness(**BASIC_GENERATOR_CONFIG)
        bd_gen = BirthDeathWitness(**BASIC_GENERATOR_CONFIG)

        # Valid parameters
        self.assertTrue(yule_gen.validate_params([0.1, 0.05, 0.02, 0.01]))
        self.assertTrue(bd_gen.validate_params([0.05, 0.01]))

        # Invalid parameters (negative values)
        self.assertFalse(yule_gen.validate_params([0.1, -0.05, 0.02, 0.01]))
        self.assertFalse(bd_gen.validate_params([-0.05, 0.01]))

        # Invalid parameters (NaN values)
        self.assertFalse(yule_gen.validate_params([0.1, float("nan"), 0.02, 0.01]))
        self.assertFalse(bd_gen.validate_params([float("nan"), 0.01]))


class StatsInheritanceTest(unittest.TestCase):
    def test_abundance_stats_creation(self):
        stats = AbundanceStats()
        self.assertIsInstance(stats, AbundanceStats)

        stats_with_additional = AbundanceStats(additional_stats=True)
        self.assertIsInstance(stats_with_additional, AbundanceStats)

        # Test that additional stats changes the number of computed stats
        self.assertGreater(
            stats_with_additional.get_num_stats(),
            AbundanceStats(additional_stats=False).get_num_stats(),
        )

    def test_stats_computation(self):
        stats = AbundanceStats(additional_stats=False)

        # Test with empty data
        result = stats.compute_stats([])
        self.assertEqual(len(result), stats.get_num_stats())

        # Test with BREAK condition
        result = stats.compute_stats("BREAK")
        self.assertEqual(len(result), stats.get_num_stats())
        self.assertTrue(all(x == 1 for x in result))

        # Test with actual data
        witness_counts = [1, 2, 3, 1, 1, 5]
        result = stats.compute_stats(witness_counts)
        self.assertEqual(len(result), stats.get_num_stats())
        self.assertTrue(all(isinstance(x, (int, float)) for x in result))

    def test_stats_rescaling(self):
        stats = AbundanceStats(additional_stats=False)
        witness_counts = [1, 2, 3, 1, 1, 5]

        # Compute stats and rescale
        computed_stats = stats.compute_stats(witness_counts)
        rescaled = stats.rescaled_stats(computed_stats)

        # Check that rescaled stats are interpretable integers
        self.assertTrue(all(isinstance(x, int) for x in rescaled))
        self.assertEqual(len(rescaled), 5)  # Number of rescaled stats


if __name__ == "__main__":
    unittest.main()
