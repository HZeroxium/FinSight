# models/model_example_usage.py

"""
Simplified and working example usage of the        # 2. Test models
        models_to_test = [
            ModelType.PATCHTST,
            ModelType.PATCHTSMIXER,
            ModelType.PYTORCH_TRANSFORMER,
        ]ght Model Builder system

This module demonstrates the core functionality:
- Training PatchTST and PatchTSMixer models
- Making predictions
- Basic evaluation
- Visualization and backtesting utilities
"""
import json
import asyncio
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

from .model_facade import ModelFacade
from ..data.data_loader import CSVDataLoader
from ..schemas.enums import ModelType, TimeFrame
from ..logger.logger_factory import LoggerFactory

# Import utilities with fallbacks
try:
    from ..utils.visualization_utils import VisualizationUtils
except ImportError:

    class VisualizationUtils:
        @staticmethod
        def plot_training_metrics(*args, **kwargs):
            print("VisualizationUtils not available - skipping visualization")
            return None

        @staticmethod
        def plot_prediction_analysis(*args, **kwargs):
            print("VisualizationUtils not available - skipping visualization")
            return None


try:
    from ..utils.backtest_strategy_utils import BacktestEngine, HyperparameterTuner
except ImportError:

    class BacktestEngine:
        def run_backtest(self, *args, **kwargs):
            return {"error": "BacktestEngine not available"}

    class HyperparameterTuner:
        def grid_search(self, *args, **kwargs):
            return {"error": "HyperparameterTuner not available"}


class SimpleModelTester:
    """Simplified model testing class with working functionality"""

    def __init__(self, data_path: str = None, output_dir: str = "model_experiments"):
        self.logger = LoggerFactory.get_logger("SimpleModelTester")
        self.data_path = data_path or "data/BTCUSDT_1d.csv"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories
        (self.output_dir / "results").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True, parents=True)

        self.facade = ModelFacade()
        self.viz_utils = VisualizationUtils()
        self.backtest_engine = BacktestEngine()

        self.results = {}

    async def run_simple_experiment(self):
        """Run a simple experiment with both models"""
        self.logger.info("=== Starting Simple Model Experiment ===")

        # 1. Load data
        data = await self._load_data()
        if data is None:
            self.logger.error("Failed to load data")
            return None

        # 2. Test models
        models_to_test = [
            ModelType.PATCHTST,
            ModelType.PATCHTSMIXER,
            ModelType.PYTORCH_TRANSFORMER,
        ]

        for model_type in models_to_test:
            try:
                self.logger.info(f"=== Testing {model_type.value} ===")
                model_results = await self._test_single_model(model_type, data)
                self.results[model_type.value] = model_results

            except Exception as e:
                self.logger.error(f"Model {model_type.value} failed: {e}")
                self.results[model_type.value] = {"error": str(e), "success": False}

        # 3. Save results
        await self._save_results()

        return self.results

    async def _load_data(self) -> pd.DataFrame:
        """Load and prepare data"""
        try:
            self.logger.info("Loading data...")

            # Check if data file exists
            data_file = Path(self.data_path)
            if not data_file.exists():
                # Try alternative paths
                alt_paths = [
                    Path("data/BTCUSDT_1d.csv"),
                    Path("../data/BTCUSDT_1d.csv"),
                ]

                for alt_path in alt_paths:
                    if alt_path.exists():
                        data_file = alt_path
                        break
                else:
                    self.logger.error(f"Data file not found: {self.data_path}")
                    return None

            # Load with CSVDataLoader
            loader = CSVDataLoader()
            data = loader.load_data("BTCUSDT", TimeFrame.DAY_1, data_file)

            self.logger.info(f"Loaded {len(data)} rows of data")
            self.logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

            return data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None

    async def _test_single_model(
        self, model_type: ModelType, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Test a single model with simplified configuration"""

        # Use smaller parameters for testing
        config = {
            "target_column": "close",
            "context_length": 64,  # Smaller for faster training
            "prediction_length": 1,
            "num_epochs": 2,  # Fewer epochs for testing
            "batch_size": 16,  # Smaller batch size
            "learning_rate": 1e-4,
            "use_technical_indicators": True,
            "normalize_features": True,
        }

        results = {}

        # 1. Train model
        self.logger.info(f"Training {model_type.value}...")
        training_result = self.facade.create_and_train_model(
            model_type=model_type, data=data, **config
        )
        results["training"] = training_result

        # 2. Make some predictions if training was successful
        if training_result.get("success", False):
            self.logger.info(f"Making forecasts with {model_type.value}...")
            try:
                forecast_result = self.facade.forecast(
                    model_type=model_type,
                    data=data.tail(100),  # Use last 100 rows for forecasting
                    n_steps=5,  # Request 5 forecasts for testing
                )
                results["forecast"] = forecast_result

            except Exception as e:
                self.logger.error(f"Prediction failed: {e}")
                results["prediction"] = {"error": str(e)}

        # 3. Basic evaluation
        if training_result.get("success", False):
            self.logger.info(f"Evaluating {model_type.value}...")
            try:
                eval_result = self.facade.evaluate_model(
                    model_type=model_type,
                    test_data=data.tail(
                        100
                    ),  # Use last 100 rows for evaluation (at least 3x context_length)
                    detailed_metrics=True,
                )
                results["evaluation"] = eval_result

            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                results["evaluation"] = {"error": str(e)}

        # 4. Generate visualizations
        try:
            viz_dir = (
                self.output_dir / "visualizations" / model_type.value.replace("/", "-")
            )
            viz_dir.mkdir(parents=True, exist_ok=True)

            train_result = results.get("training")
            if train_result and isinstance(train_result, dict):
                self.viz_utils.plot_training_metrics(
                    train_result, save_path=viz_dir / "training_metrics.png"
                )

            pred_result = results.get("prediction")
            if pred_result is not None and (
                isinstance(pred_result, dict)
                or (hasattr(pred_result, "__len__") and len(pred_result) > 0)
            ):
                self.viz_utils.plot_prediction_analysis(
                    pred_result, save_path=viz_dir / "prediction_analysis.png"
                )

        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")

        results["success"] = training_result.get("success", False)
        return results

    async def _save_results(self):
        """Save experiment results"""
        try:
            results_file = (
                self.output_dir / "results" / "simple_experiment_results.json"
            )

            # Prepare results for JSON serialization
            serializable_results = self._make_json_serializable(self.results)

            # Add metadata
            experiment_summary = {
                "timestamp": datetime.now().isoformat(),
                "experiment_type": "simple_model_test",
                "models_tested": list(self.results.keys()),
                "results": serializable_results,
            }

            with open(results_file, "w") as f:
                json.dump(experiment_summary, f, indent=2, default=str)

            self.logger.info(f"Results saved to {results_file}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj


async def run_simple_experiment():
    """Run the simple experiment"""
    tester = SimpleModelTester()
    results = await tester.run_simple_experiment()

    print("\n=== Experiment Summary ===")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        if model_results.get("success"):
            print(f"  [OK] Training: Success")
            if "prediction" in model_results:
                pred_result = model_results["prediction"]
                if isinstance(pred_result, dict):
                    success_val = pred_result.get("success", "Unknown")
                else:
                    # If it's a numpy array or other type, consider it successful
                    success_val = True if pred_result is not None else False
                print(f"  [OK] Prediction: {success_val}")
            if "evaluation" in model_results:
                eval_result = model_results["evaluation"]
                if isinstance(eval_result, dict):
                    success_val = eval_result.get("success", "Unknown")
                else:
                    success_val = True if eval_result is not None else False
                print(f"  [OK] Evaluation: {success_val}")
            if "forecast" in model_results:
                forecast_result = model_results["forecast"]
                if isinstance(forecast_result, dict):
                    success_val = forecast_result.get("success", "Unknown")
                else:
                    success_val = True if forecast_result is not None else False
                print(f"  [OK] Forecasting: {success_val}")
        else:
            print(f"  [FAILED] Failed: {model_results.get('error', 'Unknown error')}")

    return results


def run_hyperparameter_experiment():
    """Run a hyperparameter matrix experiment"""
    logger = LoggerFactory.get_logger("HyperparameterExperiment")
    logger.info("=== Running Hyperparameter Matrix Experiment ===")

    # Define hyperparameter combinations to test
    param_combinations = [
        {
            "context_length": 32,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "add_datetime_features": False,
        },
        {
            "context_length": 64,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "add_datetime_features": False,
        },
        {
            "context_length": 32,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "add_datetime_features": True,
        },
        {
            "context_length": 64,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "add_datetime_features": True,
        },
    ]

    results = []

    for i, params in enumerate(param_combinations):
        logger.info(f"Testing configuration {i+1}/{len(param_combinations)}: {params}")

        try:
            # This would normally run training with these parameters
            # For now, we'll simulate results
            dummy_result = {
                "config": params,
                "training_loss": np.random.uniform(0.1, 0.5),
                "validation_loss": np.random.uniform(0.15, 0.6),
                "accuracy": np.random.uniform(0.6, 0.9),
                "training_time": np.random.uniform(60, 300),  # seconds
            }

            results.append(dummy_result)
            logger.info(
                f"  Result: val_loss={dummy_result['validation_loss']:.4f}, accuracy={dummy_result['accuracy']:.4f}"
            )

        except Exception as e:
            logger.error(f"Configuration {i+1} failed: {e}")
            results.append({"config": params, "error": str(e)})

    # Find best configuration
    successful_results = [r for r in results if "error" not in r]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x["validation_loss"])
        logger.info(f"Best configuration: {best_result['config']}")
        logger.info(f"Best validation loss: {best_result['validation_loss']:.4f}")

    return results


async def main():
    """Main execution function"""
    logger = LoggerFactory.get_logger("ModelExampleMain")

    try:
        # Run simple experiment
        logger.info("Starting model example usage demonstration")
        results = await run_simple_experiment()

        # Run hyperparameter experiment
        # hp_results = run_hyperparameter_experiment()

        logger.info("All experiments completed successfully!")
        return {
            "simple_experiment": results,
            # "hyperparameter_experiment": hp_results,
        }

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())
