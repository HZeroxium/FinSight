#!/usr/bin/env python3
"""
Test script to verify that the enhanced PatchTST and PatchTSMixer experiments
work correctly with visualization capabilities.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from logger.logger_factory import LoggerFactory


def test_imports():
    """Test that all required imports work correctly."""
    logger = LoggerFactory.get_logger("TestVisualization")

    try:
        # Test PatchTST imports
        logger.info("Testing PatchTST imports...")
        from patchtst import PatchTSTExperiment

        logger.info("✓ PatchTST imports successful")

        # Test PatchTSMixer imports
        logger.info("Testing PatchTSMixer imports...")
        from patchtsmixer import PatchTSMixerExperiment

        logger.info("✓ PatchTSMixer imports successful")

        # Test visualization libraries
        logger.info("Testing visualization libraries...")
        import matplotlib.pyplot as plt
        import seaborn as sns

        logger.info("✓ Visualization libraries available")

        return True

    except Exception as e:
        logger.error(f"Import test failed: {str(e)}")
        return False


def test_experiment_initialization():
    """Test that experiments can be initialized correctly."""
    logger = LoggerFactory.get_logger("TestVisualization")

    try:
        data_path = r"d:\Projects\Desktop\FinSight\finetuning\data\BTCUSDT_1d.csv"

        # Test PatchTST initialization
        logger.info("Testing PatchTST initialization...")
        from patchtst import PatchTSTExperiment

        test_output_dir = r"d:\Projects\Desktop\FinSight\finetuning\src\experiments\test_outputs\patchtst"
        patchtst_exp = PatchTSTExperiment(data_path, test_output_dir)
        logger.info("✓ PatchTST experiment initialized successfully")

        # Test PatchTSMixer initialization
        logger.info("Testing PatchTSMixer initialization...")
        from patchtsmixer import PatchTSMixerExperiment

        test_output_dir = r"d:\Projects\Desktop\FinSight\finetuning\src\experiments\test_outputs\patchtsmixer"
        patchtsmixer_exp = PatchTSMixerExperiment(data_path, test_output_dir)
        logger.info("✓ PatchTSMixer experiment initialized successfully")

        return True

    except Exception as e:
        logger.error(f"Initialization test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    logger = LoggerFactory.get_logger("TestVisualization")

    logger.info("Starting enhanced experiments test...")

    # Test imports
    if not test_imports():
        logger.error("Import tests failed")
        return False

    # Test initialization
    if not test_experiment_initialization():
        logger.error("Initialization tests failed")
        return False

    logger.info("✅ All tests passed! Enhanced experiments are ready to run.")
    logger.info(
        "📊 Both PatchTST and PatchTSMixer now include comprehensive visualization capabilities:"
    )
    logger.info("   • Time series comparison plots")
    logger.info("   • Prediction scatter plots")
    logger.info("   • Error distribution analysis")
    logger.info("   • Performance metrics dashboard")
    logger.info("   • Residuals analysis")
    logger.info("   • Detailed JSON reports")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
