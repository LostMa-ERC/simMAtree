#!/usr/bin/env python3
# ruff: noqa: F403, F401
"""
Test runner for simmatree project.

This script runs all tests and provides detailed reporting.
"""

import sys
import time
import unittest
from io import StringIO
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import all test modules
try:
    from tests.e2e.sbi_test import *
    from tests.integration.cli_test import *
    from tests.unit.config_parsing_test import *
    from tests.unit.evaluation_test import *
    from tests.unit.generators_test import *
    from tests.unit.model_inheritance_test import *
    from tests.unit.priors_test import *
    from tests.unit.sbi_backend_test import *
    from tests.unit.sbi_visualization_test import *
    from tests.unit.stats_test import *

    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Some test modules could not be imported: {e}")
    IMPORT_SUCCESS = False


class VerboseTestResult(unittest.TextTestResult):
    """Custom test result class for more detailed output"""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
        self.start_time = None

    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()

    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.test_results.append(
            {"test": str(test), "status": "PASS", "time": elapsed, "message": None}
        )

    def addError(self, test, err):
        super().addError(test, err)
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.test_results.append(
            {
                "test": str(test),
                "status": "ERROR",
                "time": elapsed,
                "message": self._exc_info_to_string(err, test),
            }
        )

    def addFailure(self, test, err):
        super().addFailure(test, err)
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.test_results.append(
            {
                "test": str(test),
                "status": "FAIL",
                "time": elapsed,
                "message": self._exc_info_to_string(err, test),
            }
        )

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.test_results.append(
            {"test": str(test), "status": "SKIP", "time": elapsed, "message": reason}
        )


class TestCategories:
    """Organize tests by category"""

    UNIT_TESTS = [
        "config_parsing_test",
        "model_inheritance_test",
        "sbi_visualization_test",
        "priors_test",
        "generators_test",
        "stats_test",
        "sbi_backend_test",
        "evaluation_test",
    ]

    E2E_TESTS = ["sbi_test"]

    INTEGRATION_TESTS = ["cli_test"]

    FAST_TESTS = [
        "config_parsing_test",
        "model_inheritance_test",
        "priors_test",
        "stats_test",
    ]

    SLOW_TESTS = ["sbi_backend_test", "sbi_test", "evaluation_test"]

    CLI_TESTS = ["cli_test"]


def run_test_category(category_name, test_modules, verbosity=2):
    """Run a specific category of tests"""
    print(f"\n{'=' * 60}")
    print(f"Running {category_name}")
    print(f"{'=' * 60}")

    suite = unittest.TestSuite()

    for module_name in test_modules:
        try:
            # Get all test classes from the module
            if module_name in TestCategories.INTEGRATION_TESTS:
                module = sys.modules.get(f"tests.integration.{module_name}")
            elif module_name in TestCategories.E2E_TESTS:
                module = sys.modules.get(f"tests.e2e.{module_name}")
            else:
                module = sys.modules.get(f"tests.unit.{module_name}")

            if module:
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if (
                        isinstance(item, type)
                        and issubclass(item, unittest.TestCase)
                        and item != unittest.TestCase
                    ):
                        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(item))
        except Exception as e:
            print(f"Warning: Could not load tests from {module_name}: {e}")

    if suite.countTestCases() == 0:
        print(f"No tests found in {category_name}")
        return True

    # Run tests with custom result class
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream, verbosity=verbosity, resultclass=VerboseTestResult
    )

    result = runner.run(suite)

    # Print summary
    print(f"\nResults for {category_name}:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    # Print details for failed tests
    if result.failures or result.errors:
        print(f"\nFailed tests in {category_name}:")
        for test, traceback in result.failures + result.errors:
            print(f"- {test}")

    # Print timing info if available
    if hasattr(result, "test_results"):
        slow_tests = [t for t in result.test_results if t["time"] > 1.0]
        if slow_tests:
            print(f"\nSlow tests in {category_name} (>1s):")
            for test in sorted(slow_tests, key=lambda x: x["time"], reverse=True):
                print(f"- {test['test']}: {test['time']:.2f}s")

    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """Main test runner"""
    if not IMPORT_SUCCESS:
        print("Cannot run tests due to import errors.")
        sys.exit(1)

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Run simmatree tests")
    parser.add_argument(
        "--category",
        choices=["unit", "e2e", "integration", "fast", "slow", "cli", "all"],
        default="all",
        help="Test category to run",
    )
    parser.add_argument(
        "--verbosity", type=int, default=2, help="Test verbosity level (0-2)"
    )
    parser.add_argument("--failfast", action="store_true", help="Stop on first failure")

    args = parser.parse_args()

    print("SimMAtree Test Runner")
    print("====================")

    # Check if we can import key dependencies
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sbi  # noqa: F401
        import torch  # noqa: F401

        print("✓ All key dependencies available")
    except ImportError as e:
        print(f"⚠ Warning: Missing dependency: {e}")

        print("✓ All key dependencies available")
    except ImportError as e:
        print(f"⚠ Warning: Missing dependency: {e}")

    success = True

    if args.category == "all":
        # Run all test categories
        success &= run_test_category(
            "Unit Tests", TestCategories.UNIT_TESTS, args.verbosity
        )
        success &= run_test_category(
            "End-to-End Tests", TestCategories.E2E_TESTS, args.verbosity
        )
        success &= run_test_category(
            "Integration Tests", TestCategories.INTEGRATION_TESTS, args.verbosity
        )

    elif args.category == "unit":
        success &= run_test_category(
            "Unit Tests", TestCategories.UNIT_TESTS, args.verbosity
        )

    elif args.category == "e2e":
        success &= run_test_category(
            "End-to-End Tests", TestCategories.E2E_TESTS, args.verbosity
        )

    elif args.category == "integration":
        success &= run_test_category(
            "Integration Tests", TestCategories.INTEGRATION_TESTS, args.verbosity
        )

    elif args.category == "fast":
        success &= run_test_category(
            "Fast Tests", TestCategories.FAST_TESTS, args.verbosity
        )

    elif args.category == "slow":
        success &= run_test_category(
            "Slow Tests", TestCategories.SLOW_TESTS, args.verbosity
        )

    elif args.category == "cli":
        success &= run_test_category(
            "CLI Tests", TestCategories.CLI_TESTS, args.verbosity
        )

    # Final summary
    print(f"\n{'=' * 60}")
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
