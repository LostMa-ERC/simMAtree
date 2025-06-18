import unittest

import numpy as np

from src.stats.abundance_stats import AbundanceStats


class AbundanceStatsTest(unittest.TestCase):
    def setUp(self):
        self.stats = AbundanceStats(additional_stats=False)
        self.stats_with_additional = AbundanceStats(additional_stats=True)

        # Test data: witness counts per text
        self.test_data = [1, 2, 3, 1, 1, 5, 2, 1, 4, 1]
        self.single_text = [10]
        self.many_ones = [1] * 20
        self.varied_data = [1, 1, 2, 2, 3, 5, 8, 13, 21]

    def test_initialization(self):
        """Test stats object initialization"""
        self.assertIsInstance(self.stats, AbundanceStats)
        self.assertIsInstance(self.stats_with_additional, AbundanceStats)

        # Basic stats should have 6 statistics
        self.assertEqual(self.stats.get_num_stats(), 6)

        # With additional stats should have more
        self.assertGreater(
            self.stats_with_additional.get_num_stats(), self.stats.get_num_stats()
        )

    def test_stats_names(self):
        """Test that stats names are correctly defined"""
        names = self.stats.get_stats_names()
        expected_names = [
            "Total witnesses (scaled)",
            "Total texts (scaled)",
            "Texts per witness ratio",
            "Max witnesses proportion",
            "Median/max ratio",
            "Proportion with 1 witness",
        ]

        self.assertEqual(names, expected_names)

        # Additional stats should have more names
        additional_names = self.stats_with_additional.get_stats_names()
        self.assertGreater(len(additional_names), len(names))

    def test_rescaled_stats_names(self):
        """Test rescaled stats names"""
        rescaled_names = self.stats.get_rescaled_stats_names()
        expected_rescaled = [
            "Number of witnesses",
            "Number of texts",
            "Max. number of witnesses per text",
            "Med. number of witnesses per text",
            "Number of text with one witness",
        ]

        self.assertEqual(rescaled_names, expected_rescaled)

    def test_empty_data(self):
        """Test handling of empty data"""
        result = self.stats.compute_stats([])

        self.assertEqual(len(result), self.stats.get_num_stats())
        self.assertTrue(all(x == 0 for x in result))

    def test_break_condition(self):
        """Test handling of BREAK condition"""
        result = self.stats.compute_stats("BREAK")

        self.assertEqual(len(result), self.stats.get_num_stats())
        self.assertTrue(all(x == 1 for x in result))

    def test_basic_stats_computation(self):
        """Test computation of basic statistics"""
        result = self.stats.compute_stats(self.test_data)

        # Should return correct number of stats
        self.assertEqual(len(result), 6)

        # All results should be finite numbers
        self.assertTrue(all(np.isfinite(x) for x in result))

        # All results should be non-negative
        self.assertTrue(all(x >= 0 for x in result))

    def test_stats_interpretation(self):
        """Test that computed stats make sense"""
        witness_counts = [1, 2, 3, 1, 1]  # 5 texts, 8 witnesses total
        result = self.stats.compute_stats(witness_counts)

        # Total witnesses (scaled) = 8 / 1e6
        self.assertAlmostEqual(result[0], 8 / 1e6, places=8)

        # Total texts (scaled) = 5 / 1e6
        self.assertAlmostEqual(result[1], 5 / 1e6, places=8)

        # Texts per witness ratio = 5/8
        self.assertAlmostEqual(result[2], 5 / 8, places=8)

        # Max witnesses proportion = 3/8
        self.assertAlmostEqual(result[3], 3 / 8, places=8)

        # Median/max ratio = 1/3 (median=1, max=3)
        self.assertAlmostEqual(result[4], 1 / 3, places=8)

        # Proportion with 1 witness = 3/5
        self.assertAlmostEqual(result[5], 3 / 5, places=8)

    def test_additional_stats_computation(self):
        """Test computation with additional statistics"""
        result = self.stats_with_additional.compute_stats(self.test_data)

        # Should have more stats than basic version
        self.assertGreater(len(result), 6)

        # All results should be finite
        self.assertTrue(all(np.isfinite(x) for x in result))

    def test_rescaling(self):
        """Test rescaling of statistics back to interpretable values"""
        result = self.stats.compute_stats(self.test_data)
        rescaled = self.stats.rescaled_stats(result)

        # Should return list of integers
        self.assertIsInstance(rescaled, list)
        self.assertEqual(len(rescaled), 5)
        self.assertTrue(all(isinstance(x, int) for x in rescaled))

        # Values should make sense
        num_witnesses, num_texts, max_witnesses, median_witnesses, texts_with_one = (
            rescaled
        )

        self.assertEqual(num_witnesses, sum(self.test_data))
        self.assertEqual(num_texts, len(self.test_data))
        self.assertEqual(max_witnesses, max(self.test_data))
        self.assertGreater(texts_with_one, 0)  # Should have some texts with 1 witness

    def test_single_text_case(self):
        """Test with single text having many witnesses"""
        result = self.stats.compute_stats(self.single_text)
        rescaled = self.stats.rescaled_stats(result)

        num_witnesses, num_texts, max_witnesses, median_witnesses, texts_with_one = (
            rescaled
        )

        self.assertEqual(num_witnesses, 10)
        self.assertEqual(num_texts, 1)
        self.assertEqual(max_witnesses, 10)
        self.assertEqual(median_witnesses, 10)
        self.assertEqual(texts_with_one, 0)

    def test_many_ones_case(self):
        """Test with many texts having single witness"""
        result = self.stats.compute_stats(self.many_ones)
        rescaled = self.stats.rescaled_stats(result)

        num_witnesses, num_texts, max_witnesses, median_witnesses, texts_with_one = (
            rescaled
        )

        self.assertEqual(num_witnesses, 20)
        self.assertEqual(num_texts, 20)
        self.assertEqual(max_witnesses, 1)
        self.assertEqual(median_witnesses, 1)
        self.assertEqual(texts_with_one, 20)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Single witness, single text
        result = self.stats.compute_stats([1])
        self.assertEqual(len(result), self.stats.get_num_stats())

        # Large numbers
        large_data = [100, 200, 300]
        result = self.stats.compute_stats(large_data)
        self.assertTrue(all(np.isfinite(x) for x in result))

        # All same values
        same_values = [5, 5, 5, 5]
        result = self.stats.compute_stats(same_values)
        self.assertTrue(all(np.isfinite(x) for x in result))

    def test_print_stats(self):
        """Test print_stats method (should not crash)"""
        # Test with valid data
        try:
            self.stats.print_stats(self.test_data)
        except Exception as e:
            self.fail(f"print_stats raised an exception: {e}")

        # Test with empty data
        try:
            self.stats.print_stats([])
        except Exception as e:
            self.fail(f"print_stats with empty data raised an exception: {e}")

        # Test with None
        try:
            self.stats.print_stats(None)
        except Exception as e:
            self.fail(f"print_stats with None raised an exception: {e}")

    def test_get_rescaled_stats(self):
        """Test get_rescaled_stats method"""
        rescaled = self.stats.get_rescaled_stats(self.test_data)

        # Should be same as compute_stats -> rescaled_stats
        computed = self.stats.compute_stats(self.test_data)
        manual_rescaled = self.stats.rescaled_stats(computed)

        np.testing.assert_array_equal(rescaled, manual_rescaled)

    def test_consistency_between_methods(self):
        """Test consistency between different methods"""
        # Test that get_num_stats matches actual number of stats computed
        result = self.stats.compute_stats(self.test_data)
        self.assertEqual(len(result), self.stats.get_num_stats())

        # Test that names match number of stats
        names = self.stats.get_stats_names()
        self.assertEqual(len(names), self.stats.get_num_stats())

        # Test rescaled names length
        rescaled_names = self.stats.get_rescaled_stats_names()
        rescaled_values = self.stats.rescaled_stats(result)
        self.assertEqual(len(rescaled_names), len(rescaled_values))


class AbundanceStatsAdditionalTest(unittest.TestCase):
    """Test additional statistics in detail"""

    def setUp(self):
        self.stats = AbundanceStats(additional_stats=True)
        # Data designed to test specific additional stats
        self.test_data = [1, 1, 2, 2, 3, 3, 4, 4, 5]  # Various counts

    def test_additional_stats_count(self):
        """Test that additional stats are correctly counted"""
        basic_stats = AbundanceStats(additional_stats=False)
        additional_stats = AbundanceStats(additional_stats=True)

        self.assertEqual(basic_stats.get_num_stats(), 6)
        self.assertEqual(additional_stats.get_num_stats(), 13)  # 6 + 7 additional

    def test_proportion_stats(self):
        """Test proportion statistics (2, 3, 4 witnesses)"""
        # Create data with known proportions
        data_with_proportions = [1] * 10 + [2] * 5 + [3] * 3 + [4] * 2  # 20 texts total
        result = self.stats.compute_stats(data_with_proportions)

        # Extract additional stats (indices 6, 7, 8 are proportions with 2, 3, 4)
        prop_2_witnesses = result[6]  # Should be 5/20 = 0.25
        prop_3_witnesses = result[7]  # Should be 3/20 = 0.15
        prop_4_witnesses = result[8]  # Should be 2/20 = 0.10

        self.assertAlmostEqual(prop_2_witnesses, 0.25, places=6)
        self.assertAlmostEqual(prop_3_witnesses, 0.15, places=6)
        self.assertAlmostEqual(prop_4_witnesses, 0.10, places=6)

    def test_quantile_ratios(self):
        """Test quantile ratio statistics"""
        # Use data where quantiles are predictable
        sorted_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.stats.compute_stats(sorted_data)

        # Quantile ratios should be reasonable
        q75_ratio = result[9]  # Index 9: Q75/max ratio
        q85_ratio = result[10]  # Index 10: Q85/max ratio

        # Both should be between 0 and 1
        self.assertGreater(q75_ratio, 0)
        self.assertLess(q75_ratio, 1)
        self.assertGreater(q85_ratio, 0)
        self.assertLess(q85_ratio, 1)

        # Q85 should be >= Q75
        self.assertGreaterEqual(q85_ratio, q75_ratio)

    def test_second_largest_stats(self):
        """Test second largest statistics"""
        # Data with clear largest and second largest
        data_with_max = [1, 1, 1, 5, 10]  # max=10, second_largest=5
        result = self.stats.compute_stats(data_with_max)

        second_largest_prop = result[11]  # Index 11: second_largest/total_witnesses
        second_largest_max_ratio = result[12]  # Index 12: second_largest/max

        total_witnesses = sum(data_with_max)  # 18
        expected_prop = 5 / total_witnesses
        expected_ratio = 5 / 10  # 0.5

        self.assertAlmostEqual(second_largest_prop, expected_prop, places=6)
        self.assertAlmostEqual(second_largest_max_ratio, expected_ratio, places=6)

    def test_single_text_additional_stats(self):
        """Test additional stats with single text"""
        result = self.stats.compute_stats([10])

        # With only one text, second largest stats should be 0
        second_largest_prop = result[11]
        second_largest_max_ratio = result[12]

        self.assertEqual(second_largest_prop, 0.0)
        self.assertEqual(second_largest_max_ratio, 0.0)


class AbundanceStatsErrorHandlingTest(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_invalid_rescaling_input(self):
        """Test rescaling with invalid input"""
        stats = AbundanceStats()

        # Too few statistics
        with self.assertRaises(ValueError):
            stats.rescaled_stats([1, 2, 3])  # Only 3 stats, need at least 6

    def test_numeric_stability(self):
        """Test numeric stability with extreme values"""
        stats = AbundanceStats()

        # Very large numbers
        large_data = [1000000] * 100
        result = stats.compute_stats(large_data)
        self.assertTrue(all(np.isfinite(x) for x in result))

        # Very small numbers (single witnesses)
        small_data = [1] * 1000000  # Many texts with 1 witness each
        result = stats.compute_stats(small_data)
        self.assertTrue(all(np.isfinite(x) for x in result))

    def test_data_type_handling(self):
        """Test handling of different data types"""
        stats = AbundanceStats()

        # NumPy array
        np_data = np.array([1, 2, 3, 4, 5])
        result_np = stats.compute_stats(np_data)

        # Python list
        list_data = [1, 2, 3, 4, 5]
        result_list = stats.compute_stats(list_data)

        # Should give same results
        np.testing.assert_array_almost_equal(result_np, result_list)


if __name__ == "__main__":
    unittest.main()
