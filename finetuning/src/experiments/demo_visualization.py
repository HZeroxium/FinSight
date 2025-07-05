#!/usr/bin/env python3
"""
Demo script showing how to run the enhanced PatchTST and PatchTSMixer experiments
with comprehensive visualization capabilities.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from logger.logger_factory import LoggerFactory


def demo_patchtst():
    """Demonstrate PatchTST experiment with visualization."""
    logger = LoggerFactory.get_logger("DemoVisualization")

    try:
        logger.info("🚀 Starting PatchTST experiment demo...")

        from patchtst import PatchTSTExperiment

        data_path = r"d:\Projects\Desktop\FinSight\finetuning\data\BTCUSDT_1d.csv"
        output_dir = r"d:\Projects\Desktop\FinSight\finetuning\src\experiments\outputs\patchtst_demo"

        # Create experiment
        experiment = PatchTSTExperiment(data_path, output_dir)

        # Run experiment (this will take some time)
        logger.info("Running PatchTST experiment with visualization...")
        results = experiment.run_experiment()

        logger.info("✅ PatchTST experiment completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info(f"📊 Visualizations saved to: {output_dir}/visualizations/")
        logger.info(
            f"📈 Direction accuracy: {results['backtest_results'].get('direction_accuracy_pct', 0):.2f}%"
        )

        return True

    except Exception as e:
        logger.error(f"PatchTST demo failed: {str(e)}")
        return False


def demo_patchtsmixer():
    """Demonstrate PatchTSMixer experiment with visualization."""
    logger = LoggerFactory.get_logger("DemoVisualization")

    try:
        logger.info("🚀 Starting PatchTSMixer experiment demo...")

        from patchtsmixer import PatchTSMixerExperiment

        data_path = r"d:\Projects\Desktop\FinSight\finetuning\data\BTCUSDT_1d.csv"
        output_dir = r"d:\Projects\Desktop\FinSight\finetuning\src\experiments\outputs\patchtsmixer_demo"

        # Create experiment
        experiment = PatchTSMixerExperiment(data_path, output_dir)

        # Run experiment (this will take some time)
        logger.info("Running PatchTSMixer experiment with visualization...")
        results = experiment.run_experiment()

        logger.info("✅ PatchTSMixer experiment completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info(f"📊 Visualizations saved to: {output_dir}/visualizations/")
        logger.info(
            f"📈 Direction accuracy: {results['backtest_results'].get('direction_accuracy_pct', 0):.2f}%"
        )

        return True

    except Exception as e:
        logger.error(f"PatchTSMixer demo failed: {str(e)}")
        return False


def main():
    """Run demonstration of enhanced experiments."""
    logger = LoggerFactory.get_logger("DemoVisualization")

    logger.info("🎯 Enhanced Time Series Model Experiments Demo")
    logger.info("=" * 50)
    logger.info("This demo will run both PatchTST and PatchTSMixer experiments")
    logger.info("with comprehensive visualization and analysis capabilities.")
    logger.info("")
    logger.info("⚠️  Note: Each experiment may take several minutes to complete.")
    logger.info(
        "⚠️  Ensure you have sufficient disk space for outputs and visualizations."
    )
    logger.info("")

    # Ask user which demo to run
    choice = (
        input("Choose demo: [1] PatchTST, [2] PatchTSMixer, [3] Both, [q] Quit: ")
        .strip()
        .lower()
    )

    if choice == "q":
        logger.info("Demo cancelled by user.")
        return True

    success = True

    if choice in ["1", "3"]:
        success &= demo_patchtst()

    if choice in ["2", "3"]:
        success &= demo_patchtsmixer()

    if choice not in ["1", "2", "3"]:
        logger.error("Invalid choice. Please select 1, 2, 3, or q.")
        return False

    if success:
        logger.info("🎉 All selected demos completed successfully!")
        logger.info("")
        logger.info("📋 Generated outputs include:")
        logger.info("   • Time series comparison plots (actual vs predicted)")
        logger.info("   • Prediction accuracy scatter plots")
        logger.info("   • Error distribution analysis")
        logger.info("   • Comprehensive metrics dashboard")
        logger.info("   • Residuals analysis for model diagnostics")
        logger.info("   • Detailed JSON reports with all metrics")
        logger.info("")
        logger.info("🔍 Check the output directories for all visualization files!")
    else:
        logger.error("❌ Some demos failed. Check logs for details.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
