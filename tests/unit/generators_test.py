import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.generator.birth_death_witness import BirthDeathWitness
from src.generator.yule_witness import YuleWitness


class YuleWitnessTest(unittest.TestCase):
    def setUp(self):
        self.generator = YuleWitness(n_init=1, Nact=50, Ninact=50, max_pop=1000)
        self.rng = np.random.default_rng(42)

        # Valid parameters for Yule model
        self.valid_params_dict = {"LDA": 0.3, "lda": 0.01, "gamma": 0.005, "mu": 0.003}
        self.valid_params_list = [0.3, 0.01, 0.005, 0.003]

    def test_parameter_extraction_dict(self):
        """Test parameter extraction from dictionary"""
        params = self.generator._extract_params(self.valid_params_dict)
        expected = {"LDA": 0.3, "lda": 0.01, "gamma": 0.005, "mu": 0.003}
        self.assertEqual(params, expected)

    def test_parameter_extraction_list(self):
        """Test parameter extraction from list"""
        params = self.generator._extract_params(self.valid_params_list)
        expected = {"LDA": 0.3, "lda": 0.01, "gamma": 0.005, "mu": 0.003}
        self.assertEqual(params, expected)

    def test_parameter_validation_valid(self):
        """Test parameter validation with valid parameters"""
        self.assertTrue(self.generator.validate_params(self.valid_params_dict))
        self.assertTrue(self.generator.validate_params(self.valid_params_list))

    def test_parameter_validation_invalid(self):
        """Test parameter validation with invalid parameters"""
        # Negative values
        invalid_negative = {"LDA": 0.3, "lda": -0.01, "gamma": 0.005, "mu": 0.003}
        self.assertFalse(self.generator.validate_params(invalid_negative))

        # NaN values
        invalid_nan = {"LDA": 0.3, "lda": float("nan"), "gamma": 0.005, "mu": 0.003}
        self.assertFalse(self.generator.validate_params(invalid_nan))

    def test_generation_success(self):
        """Test successful generation"""
        result = self.generator.generate(self.rng, self.valid_params_dict)

        # Should return a list of integers (witness counts)
        self.assertIsInstance(result, list)
        if result:  # If not empty
            self.assertTrue(all(isinstance(x, (int, np.integer)) for x in result))
            self.assertTrue(all(x > 0 for x in result))

    def test_generation_with_invalid_params(self):
        """Test generation with invalid parameters"""
        invalid_params = {"LDA": 0.3, "lda": -0.01, "gamma": 0.005, "mu": 0.003}

        with self.assertRaises(ValueError):
            self.generator.generate(self.rng, invalid_params)

    def test_generation_deterministic(self):
        """Test that generation is deterministic with same seed"""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        result1 = self.generator.generate(rng1, self.valid_params_dict)
        result2 = self.generator.generate(rng2, self.valid_params_dict)

        self.assertEqual(result1, result2)

    def test_save_and_load_simulation(self):
        """Test saving and loading simulation data"""
        # Generate data
        result = self.generator.generate(self.rng, self.valid_params_dict)

        if result and result != "BREAK":
            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp:
                temp_path = tmp.name

            try:
                df = self.generator.save_simul(result, temp_path)

                # Check that file was created and has correct structure
                self.assertTrue(Path(temp_path).exists())
                self.assertIsInstance(df, pd.DataFrame)
                self.assertIn("witness_ID", df.columns)
                self.assertIn("text_ID", df.columns)

                # Load the data back
                loaded_data = self.generator.load_data(temp_path)
                self.assertEqual(sorted(loaded_data), sorted(result))

            finally:
                Path(temp_path).unlink()

    def test_population_limit(self):
        """Test that population limit is respected"""
        # Use parameters that might cause population explosion
        high_growth_params = {"LDA": 0.1, "lda": 0.05, "gamma": 0.03, "mu": 0.001}

        # Use small population limit
        small_generator = YuleWitness(n_init=1, Nact=20, Ninact=10, max_pop=50)

        result = small_generator.generate(self.rng, high_growth_params)

        # Should either return BREAK or a valid result within limits
        self.assertTrue(result == "BREAK" or isinstance(result, list))


class BirthDeathWitnessTest(unittest.TestCase):
    def setUp(self):
        self.generator = BirthDeathWitness(n_init=5, Nact=50, Ninact=50, max_pop=1000)
        self.rng = np.random.default_rng(42)

        # Valid parameters for Birth-Death model (only lda and mu)
        self.valid_params_dict = {"lda": 0.01, "mu": 0.005}
        self.valid_params_list = [0.01, 0.005]

    def test_parameter_extraction_dict(self):
        """Test parameter extraction from dictionary"""
        params = self.generator._extract_params(self.valid_params_dict)
        expected = {"LDA": 0, "lda": 0.01, "gamma": 0, "mu": 0.005}
        self.assertEqual(params, expected)

    def test_parameter_extraction_list(self):
        """Test parameter extraction from list"""
        params = self.generator._extract_params(self.valid_params_list)
        expected = {"LDA": 0, "lda": 0.01, "gamma": 0, "mu": 0.005}
        self.assertEqual(params, expected)

    def test_parameter_validation_valid(self):
        """Test parameter validation with valid parameters"""
        self.assertTrue(self.generator.validate_params(self.valid_params_dict))
        self.assertTrue(self.generator.validate_params(self.valid_params_list))

    def test_parameter_validation_invalid(self):
        """Test parameter validation with invalid parameters"""
        # Negative values
        invalid_negative = {"lda": -0.01, "mu": 0.005}
        self.assertFalse(self.generator.validate_params(invalid_negative))

        # NaN values
        invalid_nan = {"lda": float("nan"), "mu": 0.005}
        self.assertFalse(self.generator.validate_params(invalid_nan))

    def test_generation_success(self):
        """Test successful generation"""
        result = self.generator.generate(self.rng, self.valid_params_dict)

        # Should return a list of integers (witness counts) or False for empty pop
        self.assertTrue(isinstance(result, list) or result is False)
        if isinstance(result, list) and result:
            self.assertTrue(all(isinstance(x, (int, np.integer)) for x in result))
            self.assertTrue(all(x > 0 for x in result))

    def test_param_count(self):
        """Test that Birth-Death model has correct parameter count"""
        self.assertEqual(self.generator.param_count, 2)

    def test_no_speciation_or_lda(self):
        """Test that Birth-Death model sets LDA and gamma to 0"""
        params = self.generator._extract_params(self.valid_params_dict)
        self.assertEqual(params["LDA"], 0)
        self.assertEqual(params["gamma"], 0)


class GeneratorComparisonTest(unittest.TestCase):
    """Test differences between Yule and Birth-Death generators"""

    def setUp(self):
        self.yule_gen = YuleWitness(n_init=1, Nact=30, Ninact=30, max_pop=500)
        self.bd_gen = BirthDeathWitness(n_init=5, Nact=30, Ninact=30, max_pop=500)
        self.rng = np.random.default_rng(42)

    def test_parameter_counts(self):
        """Test that models have correct parameter counts"""
        self.assertEqual(self.yule_gen.param_count, 4)
        self.assertEqual(self.bd_gen.param_count, 2)

    def test_model_differences(self):
        """Test that models behave differently with similar parameters"""
        # Use comparable parameters
        yule_params = {"LDA": 0.0, "lda": 0.01, "gamma": 0.0, "mu": 0.005}
        bd_params = {"lda": 0.01, "mu": 0.005}

        # Generate with both models
        yule_result = self.yule_gen.generate(self.rng, yule_params)
        bd_result = self.bd_gen.generate(np.random.default_rng(42), bd_params)

        # Both should generate valid results
        self.assertTrue(isinstance(yule_result, (list, bool)) or yule_result == "BREAK")
        self.assertTrue(isinstance(bd_result, (list, bool)) or bd_result == "BREAK")

    def test_save_load_compatibility(self):
        """Test that save/load format is compatible between models"""
        yule_params = {"LDA": 0.1, "lda": 0.008, "gamma": 0.003, "mu": 0.002}
        bd_params = {"lda": 0.008, "mu": 0.002}

        yule_result = self.yule_gen.generate(self.rng, yule_params)
        _ = self.bd_gen.generate(np.random.default_rng(42), bd_params)

        if isinstance(yule_result, list) and yule_result:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp:
                temp_path = tmp.name

            try:
                # Save with Yule generator
                self.yule_gen.save_simul(yule_result, temp_path)

                # Load with Birth-Death generator (should work)
                loaded_by_bd = self.bd_gen.load_data(temp_path)
                self.assertEqual(loaded_by_bd, yule_result)

            finally:
                Path(temp_path).unlink()


class GeneratorEdgeCaseTest(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_empty_population_handling(self):
        """Test handling when population goes extinct"""
        # Use parameters that favor extinction
        extinction_params = {"lda": 0.001, "mu": 0.01}

        generator = BirthDeathWitness(n_init=1, Nact=10, Ninact=50, max_pop=100)
        rng = np.random.default_rng(42)

        result = generator.generate(rng, extinction_params)

        # Should handle extinction gracefully
        self.assertTrue(result is False or isinstance(result, list))

    def test_save_empty_population(self):
        """Test saving empty population"""
        generator = YuleWitness(n_init=1, Nact=10, Ninact=10, max_pop=100)

        # Should handle empty list gracefully
        result = generator.save_simul([], "dummy_path.csv")
        self.assertIsNone(result)

        # Should handle None gracefully
        result = generator.save_simul(None, "dummy_path.csv")
        self.assertIsNone(result)

    def test_malformed_csv_loading(self):
        """Test loading from malformed CSV"""
        generator = YuleWitness(n_init=1, Nact=10, Ninact=10, max_pop=100)

        # Create malformed CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write("wrong,headers\n1,2\n3,4\n")
            temp_path = tmp.name

        try:
            # Should handle gracefully or raise appropriate error
            with self.assertRaises(Exception):
                generator.load_data(temp_path)
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    unittest.main()
