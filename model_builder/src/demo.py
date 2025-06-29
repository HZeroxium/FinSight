# streamlined_demo.py

"""
Streamlined AI Prediction Demo using refactored utilities.
This provides the same functionality as main_demo.py but with significantly less code.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from .core.config import Config, create_development_config
from .utils import (
    DemoUtils,
    TrainingUtils,
    EvaluationUtils,
    PredictionUtils,
    VisualizationUtils,
    CommonUtils,
    FileUtils,
    DeviceUtils,
)
from .common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel


class StreamlinedAIDemo:
    """
    Streamlined AI Demo that leverages utilities for a clean, maintainable implementation
    """

    def __init__(self, config: Optional[Config] = None, verbose: bool = True):
        """Initialize the streamlined demo"""
        self.config = config or create_development_config()

        # Setup logging
        log_level = LogLevel.DEBUG if verbose else LogLevel.INFO
        self.logger = LoggerFactory.get_logger(
            name=self.__class__.__name__,
            logger_type=LoggerType.STANDARD,
            level=log_level,
            use_colors=True,
        )

        # Demo results storage
        self.demo_results: Dict[str, Any] = {}

        # Components
        self.device = None
        self.data_loaders = None
        self.models = {}

    def run_full_demo(self) -> Dict[str, Any]:
        """Run the complete streamlined demo"""
        self.logger.info("🚀" * 20)
        self.logger.info("STARTING STREAMLINED AI DEMO")
        self.logger.info("🚀" * 20)

        demo_start_time = time.time()

        try:
            # 1. Setup environment
            self._log_section("SETUP", "🔧")
            setup_info = DemoUtils.setup_demo_environment(self.config)
            self.demo_results["setup_info"] = setup_info

            # Extract device from setup_info string and convert to torch.device
            device_str = setup_info["device"]
            self.device = DeviceUtils.get_device(
                prefer_gpu=self.config.model.use_gpu, gpu_id=self.config.model.gpu_id
            )

            # 2. Data processing
            self._log_section("DATA PROCESSING", "📊")
            self.data_loaders = DemoUtils.prepare_data_pipeline(self.config)

            # 3. Model creation
            self._log_section("MODEL CREATION", "🏗️")
            self.models = DemoUtils.create_model_variants(self.config, self.device)

            # 4. Training (single model for demo)
            self._log_section("MODEL TRAINING", "🏋️")
            training_results = TrainingUtils.train_single_model(
                model=self.models["transformer"],
                train_loader=self.data_loaders[0],
                val_loader=self.data_loaders[1],
                config=self.config,
                device=self.device,
                model_name="transformer",
            )
            self.demo_results["training_results"] = {"transformer": training_results}

            # 5. Evaluation
            self._log_section("MODEL EVALUATION", "📈")
            evaluation_results = EvaluationUtils.evaluate_single_model(
                model=self.models["transformer"],
                test_loader=self.data_loaders[2],
                config=self.config,
                device=self.device,
                model_name="transformer",
            )
            self.demo_results["evaluation_results"] = {
                "transformer": evaluation_results
            }

            # 6. Predictions
            self._log_section("PREDICTION GENERATION", "🔮")
            prediction_results = PredictionUtils.generate_predictions(
                model=self.models["transformer"],
                test_loader=self.data_loaders[2],
                device=self.device,
                model_name="transformer",
            )
            self.demo_results["predictions"] = prediction_results

            # 7. Visualizations
            self._log_section("CREATING VISUALIZATIONS", "📈")
            viz_dir = Path("demo_visualizations")
            FileUtils.ensure_dir(viz_dir)

            visualization_paths = (
                VisualizationUtils.create_comprehensive_visualizations(
                    training_results=self.demo_results["training_results"],
                    evaluation_results=self.demo_results["evaluation_results"],
                    predictions_data=self.demo_results["predictions"],
                    feature_names=self.config.model.features_to_use,
                    output_dir=viz_dir,
                )
            )
            self.demo_results["visualizations"] = visualization_paths

            # 8. Save results
            demo_time = time.time() - demo_start_time
            self.demo_results["demo_metadata"] = {
                "total_time": demo_time,
                "completion_status": "success",
                "timestamp": CommonUtils.get_readable_timestamp(),
            }

            results_path = DemoUtils.save_demo_results(self.demo_results)

            # Final summary
            self.logger.info("✅" * 20)
            self.logger.info("DEMO COMPLETED SUCCESSFULLY!")
            self.logger.info("✅" * 20)
            self.logger.info(f"Total time: {CommonUtils.format_duration(demo_time)}")
            self.logger.info(f"Results saved to: {results_path}")

            return self.demo_results

        except Exception as e:
            self.logger.error(f"Demo failed: {str(e)}")
            raise

    def _log_section(self, title: str, emoji: str = "🔹") -> None:
        """Log a section header"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"{emoji} {title}")
        self.logger.info("=" * 60)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary using utilities"""
        if "training_results" not in self.demo_results:
            return {}

        return TrainingUtils.get_training_summary(self.demo_results["training_results"])

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get evaluation summary using utilities"""
        if "evaluation_results" not in self.demo_results:
            return {}

        return EvaluationUtils.create_evaluation_report(
            self.demo_results["evaluation_results"]
        )


def main():
    """Main function for streamlined demo"""
    import argparse

    parser = argparse.ArgumentParser(description="Streamlined AI Prediction Demo")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Create and run demo
    demo = StreamlinedAIDemo(verbose=args.verbose)
    results = demo.run_full_demo()

    return results


if __name__ == "__main__":
    main()
