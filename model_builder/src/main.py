# main.py

"""
Streamlined AI Prediction Demo using refactored utilities.
This provides the same functionality as main_demo.py but with significantly less code.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

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
            log_file="logs/streamlined_demo.log",
            use_colors=True,
        )

        # Demo results storage
        self.demo_results: Dict[str, Any] = {}

        # Components
        self.device = None
        self.data_loaders = None
        self.models = {}
        self.raw_data = None
        self.processed_data = None

    def run_full_demo(self) -> Dict[str, Any]:
        """Run the complete streamlined demo"""
        self.logger.info("=" * 20)
        self.logger.info("STARTING STREAMLINED AI DEMO")
        self.logger.info("=" * 20)

        demo_start_time = time.time()

        try:
            # 1. Setup environment
            self._log_section("SETUP", "🔧")
            setup_info = DemoUtils.setup_demo_environment(self.config)
            self.demo_results["setup_info"] = setup_info

            # Extract device from setup_info string and convert to torch.device
            self.device = DeviceUtils.get_device(
                prefer_gpu=self.config.model.use_gpu, gpu_id=self.config.model.gpu_id
            )

            # 2. Data processing
            self._log_section("DATA PROCESSING", "📊")
            self.data_loaders, self.raw_data, self.processed_data = (
                DemoUtils.prepare_data_pipeline_with_data(self.config)
            )
            self.demo_results["data_info"] = {
                "raw_shape": (
                    (self.raw_data.shape[0], self.raw_data.shape[1])
                    if self.raw_data is not None
                    else None
                ),
                "processed_shape": (
                    (self.processed_data.shape[0], self.processed_data.shape[1])
                    if self.processed_data is not None
                    else None
                ),
                "features": self.config.model.features_to_use,
            }

            # 3. Model creation
            self._log_section("MODEL CREATION", "🏗️")
            self.models = DemoUtils.create_model_variants(self.config, self.device)

            # 4. Training (all model variants for comprehensive demo)
            self._log_section("MODEL TRAINING", "🏋️")
            training_results = {}

            # Train primary model (transformer)
            transformer_results = TrainingUtils.train_single_model(
                model=self.models["transformer"],
                train_loader=self.data_loaders[0],
                val_loader=self.data_loaders[1],
                config=self.config,
                device=self.device,
                model_name="transformer",
            )
            training_results["transformer"] = transformer_results

            # Train additional models if time permits (simplified for demo)
            if self.config:
                for model_name in ["lightweight", "hybrid"]:
                    if model_name in self.models:
                        results = TrainingUtils.train_single_model(
                            model=self.models[model_name],
                            train_loader=self.data_loaders[0],
                            val_loader=self.data_loaders[1],
                            config=self.config,
                            device=self.device,
                            model_name=model_name,
                        )
                        training_results[model_name] = results

            self.demo_results["training_results"] = training_results

            # 5. Evaluation
            self._log_section("MODEL EVALUATION", "📈")
            evaluation_results = {}

            # Evaluate primary model
            transformer_eval = EvaluationUtils.evaluate_single_model(
                model=self.models["transformer"],
                test_loader=self.data_loaders[2],
                config=self.config,
                device=self.device,
                model_name="transformer",
            )
            evaluation_results["transformer"] = transformer_eval

            # Evaluate additional models if they were trained
            for model_name, training_result in training_results.items():
                if model_name != "transformer" and model_name in self.models:
                    eval_result = EvaluationUtils.evaluate_single_model(
                        model=self.models[model_name],
                        test_loader=self.data_loaders[2],
                        config=self.config,
                        device=self.device,
                        model_name=model_name,
                    )
                    evaluation_results[model_name] = eval_result

            # Compare models
            if len(evaluation_results) > 1:
                model_comparison = DemoUtils.compare_models(evaluation_results)
                evaluation_results["model_comparison"] = model_comparison

            self.demo_results["evaluation_results"] = evaluation_results

            # 6. Predictions
            self._log_section("PREDICTION GENERATION", "🔮")
            prediction_results = {}

            # Generate predictions for the primary model
            transformer_preds = PredictionUtils.generate_predictions(
                model=self.models["transformer"],
                test_loader=self.data_loaders[2],
                device=self.device,
                model_name="transformer",
            )

            # Store with consistent structure
            prediction_results = {
                "predictions": transformer_preds["predictions"],
                "targets": transformer_preds["targets"],
                "analysis": transformer_preds["analysis"],
                "future_predictions": transformer_preds["future_predictions"],
                "model_used": transformer_preds["model_used"],
            }

            # Generate predictions for additional models if they were trained
            all_model_predictions = {}
            for model_name in training_results.keys():
                if model_name in self.models:
                    preds = PredictionUtils.generate_predictions(
                        model=self.models[model_name],
                        test_loader=self.data_loaders[2],
                        device=self.device,
                        model_name=model_name,
                    )
                    # Extract just the predictions array for comparison
                    all_model_predictions[model_name] = preds["predictions"]

            if all_model_predictions:
                prediction_results["all_model_predictions"] = all_model_predictions

            # Extract attention weights for explainability (if model supports it)
            if hasattr(self.models["transformer"], "get_attention_weights"):
                try:
                    # Get a sample from the test loader
                    sample_batch = next(iter(self.data_loaders[2]))
                    sample_input = sample_batch[0][:1]  # Take first sample
                    attention_weights = self.models[
                        "transformer"
                    ].get_attention_weights(sample_input)
                    prediction_results["attention_weights"] = attention_weights
                except Exception as e:
                    self.logger.warning(
                        f"Could not extract attention weights: {str(e)}"
                    )

            # Extract feature importance
            try:
                feature_importance = PredictionUtils.estimate_feature_importance(
                    model=self.models["transformer"],
                    test_loader=self.data_loaders[2],
                    feature_names=self.config.model.features_to_use,
                    device=self.device,
                )
                prediction_results["feature_importance"] = feature_importance
            except Exception as e:
                self.logger.warning(f"Could not estimate feature importance: {str(e)}")

            self.demo_results["predictions"] = prediction_results

            # 7. Visualizations
            self._log_section("CREATING VISUALIZATIONS", "📊")
            viz_dir = Path("demo_visualizations")
            FileUtils.ensure_dir(viz_dir)

            # Create comprehensive visualizations using our enhanced utilities
            try:
                # Validate prediction data before visualization
                predictions_for_viz = self.demo_results["predictions"].copy()

                # Ensure prediction arrays are properly formatted
                if "predictions" in predictions_for_viz:
                    pred_array = predictions_for_viz["predictions"]
                    if isinstance(pred_array, np.ndarray) and pred_array.ndim > 1:
                        predictions_for_viz["predictions"] = pred_array.flatten()

                if "targets" in predictions_for_viz:
                    target_array = predictions_for_viz["targets"]
                    if isinstance(target_array, np.ndarray) and target_array.ndim > 1:
                        predictions_for_viz["targets"] = target_array.flatten()

                # Validate all model predictions if they exist
                if "all_model_predictions" in predictions_for_viz:
                    all_preds = predictions_for_viz["all_model_predictions"]
                    for model_name, preds in all_preds.items():
                        if isinstance(preds, np.ndarray) and preds.ndim > 1:
                            all_preds[model_name] = preds.flatten()

                visualization_paths = (
                    VisualizationUtils.create_comprehensive_visualizations(
                        training_results=self.demo_results["training_results"],
                        evaluation_results=self.demo_results["evaluation_results"],
                        predictions_data=predictions_for_viz,
                        feature_names=self.config.model.features_to_use,
                        output_dir=viz_dir,
                        raw_data=self.raw_data,
                        processed_data=self.processed_data,
                        show_advanced=True,
                    )
                )
                self.demo_results["visualizations"] = visualization_paths

                # Log successful visualizations
                if visualization_paths:
                    self.logger.info("\nVisualizations created successfully:")
                    for viz_type, path in visualization_paths.items():
                        self.logger.info(f"  ✓ {viz_type}: {Path(path).name}")
                else:
                    self.logger.warning("No visualizations were created")
            except Exception as e:
                self.logger.error(f"Error during visualization creation: {str(e)}")
                # Continue with demo even if visualizations fail
                self.demo_results["visualizations"] = {}
                self.demo_results["visualization_error"] = str(e)

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

            # Summary of visualizations created
            if visualization_paths:
                self.logger.info("\nVisualizations created:")
                for viz_type, path in visualization_paths.items():
                    self.logger.info(f"  - {viz_type}: {Path(path).name}")
            else:
                self.logger.warning("No visualizations were created due to errors")

            return self.demo_results

        except Exception as e:
            self.logger.error(f"Demo failed: {str(e)}", exc_info=True)
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

    def visualize_specific_aspect(self, aspect_type: str, **kwargs) -> Optional[str]:
        """
        Generate a specific visualization on demand

        Args:
            aspect_type: Type of visualization to generate
            **kwargs: Additional arguments for the visualization

        Returns:
            Optional[str]: Path to the generated visualization or None if failed
        """
        try:
            viz_dir = Path("custom_visualizations")
            FileUtils.ensure_dir(viz_dir)

            if aspect_type == "feature_distributions":
                return VisualizationUtils.plot_feature_distributions(
                    data=self.processed_data,
                    save_path=viz_dir / "feature_distributions.png",
                    **kwargs,
                )

            elif aspect_type == "correlation_matrix":
                return VisualizationUtils.plot_correlation_matrix(
                    data=self.processed_data,
                    save_path=viz_dir / "correlation_matrix.png",
                    **kwargs,
                )

            elif aspect_type == "price_series":
                return VisualizationUtils.plot_price_series(
                    data=self.raw_data, save_path=viz_dir / "price_series.png", **kwargs
                )

            elif aspect_type == "trading_simulation":
                if "predictions" in self.demo_results:
                    test_preds = self.demo_results["predictions"]["test_predictions"]
                    return VisualizationUtils.plot_trading_simulation(
                        test_preds["predictions"],
                        test_preds["targets"],
                        save_path=viz_dir / "trading_simulation.png",
                        **kwargs,
                    )

            elif aspect_type == "feature_importance":
                if "feature_importance" in self.demo_results.get("predictions", {}):
                    return VisualizationUtils.plot_feature_importance(
                        self.demo_results["predictions"]["feature_importance"],
                        self.config.model.features_to_use,
                        save_path=viz_dir / "feature_importance.png",
                        **kwargs,
                    )

            elif aspect_type == "attention_weights":
                if "attention_weights" in self.demo_results.get("predictions", {}):
                    return VisualizationUtils.plot_attention_weights(
                        self.demo_results["predictions"]["attention_weights"],
                        sequence_length=self.config.model.sequence_length,
                        save_path=viz_dir / "attention_weights.png",
                        **kwargs,
                    )

            elif aspect_type == "candlestick_chart":
                if (
                    "Open" in self.raw_data.columns
                    and "predictions" in self.demo_results
                ):
                    test_preds = self.demo_results["predictions"]["test_predictions"]
                    return VisualizationUtils.plot_candlestick_chart(
                        data=self.raw_data.tail(30),
                        overlay_predictions=(
                            test_preds["predictions"][-30:]
                            if len(test_preds["predictions"]) >= 30
                            else None
                        ),
                        save_path=viz_dir / "candlestick_chart.png",
                        **kwargs,
                    )

            elif aspect_type == "forecast_comparison":
                if "all_model_predictions" in self.demo_results.get("predictions", {}):
                    all_preds = self.demo_results["predictions"][
                        "all_model_predictions"
                    ]
                    test_preds = self.demo_results["predictions"]["test_predictions"]
                    return VisualizationUtils.plot_forecast_comparison(
                        all_preds,
                        test_preds["targets"],
                        save_path=viz_dir / "forecast_comparison.png",
                        **kwargs,
                    )

            self.logger.warning(
                f"Visualization type '{aspect_type}' not recognized or data not available"
            )
            return None

        except Exception as e:
            self.logger.error(f"Error creating {aspect_type} visualization: {str(e)}")
            return None


def main():
    """Main function for streamlined demo"""
    import argparse

    parser = argparse.ArgumentParser(description="Streamlined AI Prediction Demo")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Skip training and only visualize results",
    )
    parser.add_argument(
        "--visualization",
        type=str,
        choices=[
            "feature_distributions",
            "correlation_matrix",
            "price_series",
            "trading_simulation",
            "feature_importance",
            "attention_weights",
            "candlestick_chart",
            "forecast_comparison",
        ],
        help="Generate a specific visualization",
    )
    args = parser.parse_args()

    # Create demo instance
    demo = StreamlinedAIDemo(verbose=args.verbose)

    if args.visualize_only:
        # Load previous results and generate visualizations
        results = DemoUtils.load_latest_demo_results()
        if not results:
            print("No previous demo results found. Running full demo instead.")
            results = demo.run_full_demo()
        else:
            demo.demo_results = results
            viz_dir = Path("demo_visualizations")
            FileUtils.ensure_dir(viz_dir)
            visualization_paths = (
                VisualizationUtils.create_comprehensive_visualizations(
                    training_results=results.get("training_results", {}),
                    evaluation_results=results.get("evaluation_results", {}),
                    predictions_data=results.get("predictions", {}),
                    feature_names=results.get("data_info", {}).get("features", []),
                    output_dir=viz_dir,
                    show_advanced=True,
                )
            )
            print("Visualizations generated:")
            for viz_type, path in visualization_paths.items():
                print(f"  - {viz_type}: {Path(path).name}")
    elif args.visualization:
        # Generate a specific visualization
        results = DemoUtils.load_latest_demo_results()
        if not results:
            print("No previous demo results found. Running full demo instead.")
            results = demo.run_full_demo()
        else:
            demo.demo_results = results
            # Need to load raw and processed data for visualizations
            if "data_info" in results:
                # We need to load data again since it's not saved in results
                try:
                    data_loaders, raw_data, processed_data = (
                        DemoUtils.prepare_data_pipeline_with_data(demo.config)
                    )
                    demo.data_loaders = data_loaders
                    demo.raw_data = raw_data
                    demo.processed_data = processed_data
                except Exception as e:
                    print(f"Warning: Could not reload data: {str(e)}")

            viz_path = demo.visualize_specific_aspect(args.visualization)
            if viz_path:
                print(f"Visualization generated: {viz_path}")
            else:
                print(f"Failed to generate {args.visualization} visualization")
    else:
        # Run full demo
        results = demo.run_full_demo()

    return results


if __name__ == "__main__":
    main()
