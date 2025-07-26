# models/model_example_usage.py

"""
Example usage of the FinSight Model Builder system

This module demonstrates how to use the various components of the system
for training time series models, making predictions, and performing backtests.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from pathlib import Path

from .model_facade import ModelFacade
from ..data.data_loader import CSVDataLoader
from ..data.feature_engineering import BasicFeatureEngineering
from ..schemas.enums import ModelType, TimeFrame
from common.logger.logger_factory import LoggerFactory

from ..utils.visualization_utils import VisualizationUtils
from ..utils.backtest_strategy_utils import BacktestEngine, HyperparameterTuner


# models/model_example_usage.py

"""
Comprehensive Model Example Usage with JSON Output and Visualizations

This module demonstrates how to use the FinSight Model Builder system for:
- Training time series models (PatchTST, PatchTSMixer)
- Making predictions with comprehensive analysis
- Performing backtests with visualization
- Running hyperparameter experiments
- Generating comparative reports and visualizations
"""

import asyncio
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from .model_facade import ModelFacade
from ..data.data_loader import CSVDataLoader
from ..data.feature_engineering import BasicFeatureEngineering
from ..schemas.enums import ModelType, TimeFrame
from common.logger.logger_factory import LoggerFactory

# Import utilities - these will be created if they don't exist
try:
    from ..utils.visualization_utils import VisualizationUtils
except ImportError:

    class VisualizationUtils:
        @staticmethod
        def plot_training_metrics(*args, **kwargs):
            print("VisualizationUtils not available")
            return None

        @staticmethod
        def plot_prediction_analysis(*args, **kwargs):
            print("VisualizationUtils not available")
            return None

        @staticmethod
        def plot_backtest_results(*args, **kwargs):
            print("VisualizationUtils not available")
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


class ModelExperimentRunner:
    """
    Comprehensive model experiment runner with JSON output and visualizations
    """

    def __init__(self, data_path: str = None, output_dir: str = "model_experiments"):
        self.logger = LoggerFactory.get_logger("ModelExperimentRunner")
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create output subdirectories
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)

        self.facade = ModelFacade()
        self.viz_utils = VisualizationUtils()
        self.backtest_engine = BacktestEngine()

        self.experiment_results = {}

    async def run_comprehensive_experiment(
        self,
        models_to_test: List[ModelType] = [ModelType.PATCHTST, ModelType.PATCHTSMIXER],
        hyperparameter_experiments: bool = True,
        generate_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """
        Run comprehensive model experiments with JSON output and visualizations
        """
        self.logger.info("=== Starting Comprehensive Model Experiments ===")

        # 1. Load and prepare data
        data = await self._load_and_prepare_data()
        if data is None:
            return {"success": False, "error": "Failed to load data"}

        # 2. Run experiments for each model
        for model_type in models_to_test:
            await self._run_single_model_experiment(
                model_type, data, hyperparameter_experiments, generate_visualizations
            )

        # 3. Generate comparative analysis
        await self._generate_comparative_analysis()

        # 4. Save comprehensive results
        await self._save_experiment_results()

        self.logger.info("=== Comprehensive Experiments Completed ===")
        return {
            "success": True,
            "output_directory": str(self.output_dir),
            "models_tested": [model.value for model in models_to_test],
            "experiment_summary": self.experiment_results,
        }

    async def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare data for experiments"""
        try:
            self.logger.info("Loading and preparing data...")

            if self.data_path:
                # Load from specified path
                data = pd.read_csv(self.data_path)
                data["timestamp"] = pd.to_datetime(data["timestamp"])
            else:
                # Load sample data
                data_loader = CSVDataLoader()
                data = data_loader.load_data("BTCUSDT", TimeFrame.DAY_1)

            # Basic validation
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Data must contain columns: {required_cols}")

            self.logger.info(f"Loaded {len(data)} rows of data")
            self.logger.info(
                f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}"
            )

            return data

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return None

    async def _run_single_model_experiment(
        self,
        model_type: ModelType,
        data: pd.DataFrame,
        hyperparameter_experiments: bool,
        generate_visualizations: bool,
    ):
        """Run comprehensive experiment for a single model"""
        self.logger.info(f"=== Running {model_type.value} Experiment ===")

        model_results = {
            "model_type": model_type.value,
            "timestamp": datetime.now().isoformat(),
            "training_results": {},
            "evaluation_results": {},
            "prediction_results": {},
            "backtest_results": {},
            "hyperparameter_results": {},
            "visualizations": [],
        }

        try:
            # 1. Basic model training
            training_result = await self._train_model(model_type, data)
            model_results["training_results"] = training_result

            if not training_result.get("success"):
                model_results["error"] = "Training failed"
                self.experiment_results[model_type.value] = model_results
                return

            # 2. Model evaluation
            evaluation_result = await self._evaluate_model(model_type, data)
            model_results["evaluation_results"] = evaluation_result

            # 3. Generate predictions with analysis
            prediction_result = await self._generate_predictions_with_analysis(
                model_type, data
            )
            model_results["prediction_results"] = prediction_result

            # 4. Comprehensive backtesting
            backtest_result = await self._run_comprehensive_backtest(model_type, data)
            model_results["backtest_results"] = backtest_result

            # 5. Hyperparameter experiments (if enabled)
            if hyperparameter_experiments:
                hp_result = await self._run_hyperparameter_experiments(model_type, data)
                model_results["hyperparameter_results"] = hp_result

            # 6. Generate visualizations (if enabled)
            if generate_visualizations:
                viz_files = await self._generate_model_visualizations(
                    model_type, model_results
                )
                model_results["visualizations"] = viz_files

            self.experiment_results[model_type.value] = model_results
            self.logger.info(f"{model_type.value} experiment completed successfully")

        except Exception as e:
            self.logger.error(f"{model_type.value} experiment failed: {e}")
            model_results["error"] = str(e)
            self.experiment_results[model_type.value] = model_results

    async def _train_model(
        self, model_type: ModelType, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train a model with comprehensive logging"""
        self.logger.info(f"Training {model_type.value} model...")

        # Split data for training
        train_size = int(len(data) * 0.8)
        val_size = int(len(data) * 0.1)

        train_data = data[:train_size]
        val_data = data[train_size : train_size + val_size]

        # Configure model parameters
        config = {
            "target_column": "close",
            "context_length": 64,
            "prediction_length": 1,
            "num_epochs": 10,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "use_technical_indicators": True,
            "normalize_features": True,
        }

        # Train the model
        result = self.facade.create_and_train_model(
            model_type=model_type, data=data, **config
        )

        return result

    async def _evaluate_model(
        self, model_type: ModelType, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate model performance with detailed metrics"""
        self.logger.info(f"Evaluating {model_type.value} model...")

        # Use test split for evaluation
        test_start = int(len(data) * 0.9)
        test_data = data[test_start:]

        # Run evaluation through facade
        eval_result = self.facade.evaluate_model(
            model_type=model_type, test_data=test_data, detailed_metrics=True
        )

        return eval_result

    async def _generate_predictions_with_analysis(
        self, model_type: ModelType, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate predictions with comprehensive analysis"""
        self.logger.info(f"Generating predictions for {model_type.value}...")

        # Use recent data for predictions
        recent_data = data.tail(100)  # Last 100 data points

        predictions = []
        actuals = []

        # Generate rolling predictions
        for i in range(10):  # Generate 10 predictions
            pred_result = self.facade.predict(
                model_type=model_type,
                data=recent_data.iloc[: -(10 - i)] if i > 0 else recent_data,
                n_steps=1,
            )

            if pred_result.get("success"):
                predictions.append(pred_result["predictions"][0])
                # Get actual value if available
                if i < len(recent_data) - 1:
                    actuals.append(recent_data.iloc[-(10 - i)]["close"])

        # Calculate prediction accuracy metrics
        if actuals and predictions:
            accuracy_metrics = self._calculate_prediction_accuracy(
                np.array(predictions[: len(actuals)]), np.array(actuals)
            )
        else:
            accuracy_metrics = {}

        return {
            "predictions": predictions,
            "actuals": actuals[: len(predictions)],
            "accuracy_metrics": accuracy_metrics,
            "current_price": float(data["close"].iloc[-1]),
            "prediction_confidence": (
                pred_result.get("confidence", 0.0)
                if pred_result.get("success")
                else 0.0
            ),
        }

    async def _run_comprehensive_backtest(
        self, model_type: ModelType, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run comprehensive backtesting with detailed analysis"""
        self.logger.info(f"Running comprehensive backtest for {model_type.value}...")

        # Use BacktestEngine for comprehensive analysis
        backtest_result = self.backtest_engine.run_backtest(
            model_type=model_type,
            data=data,
            lookback_periods=[7, 14, 30],  # Different lookback periods
            metrics_to_calculate=[
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "profit_factor",
                "total_return",
            ],
        )

        return backtest_result

    async def _run_hyperparameter_experiments(
        self, model_type: ModelType, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run hyperparameter experiments with grid search"""
        self.logger.info(
            f"Running hyperparameter experiments for {model_type.value}..."
        )

        # Define hyperparameter grid
        param_grids = {
            ModelType.PATCHTST: {
                "context_length": [32, 64, 96],
                "learning_rate": [1e-4, 1e-3, 1e-2],
                "batch_size": [16, 32, 64],
            },
            ModelType.PATCHTSMIXER: {
                "context_length": [32, 64, 96],
                "learning_rate": [1e-4, 1e-3, 1e-2],
                "num_layers": [2, 3, 4],
            },
        }

        tuner = HyperparameterTuner()
        hp_results = tuner.grid_search(
            model_type=model_type,
            data=data,
            param_grid=param_grids.get(model_type, {}),
            cv_folds=3,
            metric="rmse",
        )

        return hp_results

    async def _generate_model_visualizations(
        self, model_type: ModelType, model_results: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive visualizations for model results"""
        self.logger.info(f"Generating visualizations for {model_type.value}...")

        viz_files = []
        viz_dir = self.output_dir / "visualizations" / model_type.value.lower()
        viz_dir.mkdir(exist_ok=True)

        # 1. Training metrics visualization
        if model_results.get("training_results"):
            training_viz = self.viz_utils.plot_training_metrics(
                model_results["training_results"],
                save_path=viz_dir / "training_metrics.png",
            )
            viz_files.append(str(training_viz))

        # 2. Prediction analysis
        if model_results.get("prediction_results"):
            pred_viz = self.viz_utils.plot_prediction_analysis(
                model_results["prediction_results"],
                save_path=viz_dir / "prediction_analysis.png",
            )
            viz_files.append(str(pred_viz))

        # 3. Backtest results
        if model_results.get("backtest_results"):
            backtest_viz = self.viz_utils.plot_backtest_results(
                model_results["backtest_results"],
                save_path=viz_dir / "backtest_analysis.png",
            )
            viz_files.append(str(backtest_viz))

        # 4. Hyperparameter results (if available)
        if model_results.get("hyperparameter_results"):
            hp_viz = self.viz_utils.plot_hyperparameter_results(
                model_results["hyperparameter_results"],
                save_path=viz_dir / "hyperparameter_analysis.png",
            )
            viz_files.append(str(hp_viz))

        return viz_files

    async def _generate_comparative_analysis(self):
        """Generate comparative analysis across all models"""
        self.logger.info("Generating comparative analysis...")

        if len(self.experiment_results) < 2:
            self.logger.warning("Need at least 2 models for comparative analysis")
            return

        # Compare model performance
        comparison_metrics = {}
        for model_name, results in self.experiment_results.items():
            if results.get("evaluation_results"):
                comparison_metrics[model_name] = results["evaluation_results"]

        # Generate comparative visualizations
        if comparison_metrics:
            comp_viz = self.viz_utils.plot_model_comparison(
                comparison_metrics,
                save_path=self.output_dir / "visualizations" / "model_comparison.png",
            )

            # Add comparison to results
            self.experiment_results["comparative_analysis"] = {
                "metrics_comparison": comparison_metrics,
                "visualization": str(comp_viz),
            }

    async def _save_experiment_results(self):
        """Save comprehensive experiment results to JSON"""
        self.logger.info("Saving experiment results...")

        # Ensure all required directories exist
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        visualizations_dir = self.output_dir / "visualizations"
        visualizations_dir.mkdir(parents=True, exist_ok=True)

        # Create comprehensive results summary
        summary = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "models_tested": list(self.experiment_results.keys()),
                "output_directory": str(self.output_dir),
            },
            "model_results": self.experiment_results,
            "summary_statistics": self._calculate_summary_statistics(),
        }

        # Save main results file
        results_file = results_dir / "comprehensive_experiment_results.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save individual model results
        for model_name, model_results in self.experiment_results.items():
            if model_name != "comparative_analysis":  # Skip the comparison section
                # Create model-specific directory with safe filename
                safe_model_name = model_name.replace("/", "-").replace("\\", "-")
                model_dir = results_dir / safe_model_name
                model_dir.mkdir(parents=True, exist_ok=True)

                model_file = model_dir / f"{safe_model_name}_results.json"
                with open(model_file, "w") as f:
                    json.dump(model_results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {results_file}")

        # Generate and save summary report
        await self._generate_summary_report(summary)

    async def _generate_summary_report(self, summary: Dict[str, Any]):
        """Generate a comprehensive HTML summary report"""
        # This would create an HTML report with visualizations
        pass

    def _calculate_prediction_accuracy(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calculate detailed prediction accuracy metrics"""
        try:
            # Direction accuracy (up/down prediction)
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
            direction_accuracy = (
                np.mean(pred_direction == actual_direction) * 100
                if len(pred_direction) > 0
                else 0
            )

            # Price accuracy within tolerance
            tolerance_1pct = (
                np.mean(np.abs(predictions - actuals) / actuals < 0.01) * 100
            )
            tolerance_5pct = (
                np.mean(np.abs(predictions - actuals) / actuals < 0.05) * 100
            )

            # Statistical metrics
            correlation = (
                np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
            )

            return {
                "direction_accuracy_pct": float(direction_accuracy),
                "within_1pct_tolerance": float(tolerance_1pct),
                "within_5pct_tolerance": float(tolerance_5pct),
                "correlation": float(correlation),
                "mean_prediction": float(np.mean(predictions)),
                "mean_actual": float(np.mean(actuals)),
                "prediction_std": float(np.std(predictions)),
                "actual_std": float(np.std(actuals)),
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate accuracy metrics: {e}")
            return {}

    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across all experiments"""
        summary = {
            "total_models_tested": len(
                [
                    k
                    for k in self.experiment_results.keys()
                    if k != "comparative_analysis"
                ]
            ),
            "successful_experiments": len(
                [
                    k
                    for k, v in self.experiment_results.items()
                    if k != "comparative_analysis"
                    and v.get("training_results", {}).get("success")
                ]
            ),
            "best_performing_model": None,
            "experiment_duration": None,  # Could add timing if needed
        }

        # Find best performing model based on validation loss or similar metric
        best_model = None
        best_metric = float("inf")

        for model_name, results in self.experiment_results.items():
            if model_name == "comparative_analysis":
                continue

            eval_results = results.get("evaluation_results", {})
            if eval_results.get("rmse") and eval_results["rmse"] < best_metric:
                best_metric = eval_results["rmse"]
                best_model = model_name

        summary["best_performing_model"] = best_model
        return summary


# Example usage functions
async def run_quick_experiment():
    """Quick experiment for demonstration"""
    runner = ModelExperimentRunner()
    results = await runner.run_comprehensive_experiment(
        models_to_test=[ModelType.PATCHTST],
        hyperparameter_experiments=False,
        generate_visualizations=True,
    )
    return results


async def run_full_experiment():
    """Full comprehensive experiment with all features"""
    runner = ModelExperimentRunner()
    results = await runner.run_comprehensive_experiment(
        models_to_test=[ModelType.PATCHTST, ModelType.PATCHTSMIXER],
        hyperparameter_experiments=True,
        generate_visualizations=True,
    )
    return results


async def example_complete_workflow():
    """
    Legacy complete example workflow - now redirects to new comprehensive system
    """
    logger = LoggerFactory.get_logger("ModelExampleUsage")
    logger.info("Running comprehensive model experiment workflow...")

    # Use the new comprehensive experiment runner
    runner = ModelExperimentRunner()
    results = await runner.run_comprehensive_experiment(
        models_to_test=[ModelType.PATCHTST, ModelType.PATCHTSMIXER],
        hyperparameter_experiments=True,
        generate_visualizations=True,
    )

    return results


# Main execution functions
async def main():
    """Main execution function"""
    logger = LoggerFactory.get_logger("ModelExampleMain")

    try:
        # Run the comprehensive experiment
        results = await run_full_experiment()

        if results.get("success"):
            logger.info("🎉 Comprehensive experiment completed successfully!")
            logger.info(f"📁 Results saved to: {results['output_directory']}")
            logger.info(f"🤖 Models tested: {', '.join(results['models_tested'])}")

            # Print summary
            print("\n" + "=" * 60)
            print("EXPERIMENT SUMMARY")
            print("=" * 60)
            for model in results["models_tested"]:
                if model in results["experiment_summary"]:
                    model_result = results["experiment_summary"][model]
                    print(f"\n{model.upper()}:")

                    # Training status
                    training_success = model_result.get("training_results", {}).get(
                        "success", False
                    )
                    print(
                        f"  ✅ Training: {'Success' if training_success else 'Failed'}"
                    )

                    # Evaluation metrics
                    eval_results = model_result.get("evaluation_results", {})
                    if eval_results.get("rmse"):
                        print(f"  📊 RMSE: {eval_results['rmse']:.4f}")

                    # Prediction accuracy
                    pred_results = model_result.get("prediction_results", {})
                    accuracy = pred_results.get("accuracy_metrics", {})
                    if accuracy.get("direction_accuracy_pct"):
                        print(
                            f"  🎯 Direction Accuracy: {accuracy['direction_accuracy_pct']:.1f}%"
                        )

                    # Visualizations
                    viz_count = len(model_result.get("visualizations", []))
                    print(f"  📈 Visualizations: {viz_count} files generated")

            print("\n" + "=" * 60)

        else:
            logger.error(f"❌ Experiment failed: {results}")

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())


def example_api_style_usage():
    """
    Example showing API-style usage similar to what the FastAPI endpoints do
    """
    logger = LoggerFactory.get_logger("APIStyleExample")

    # This simulates what happens when you call the training endpoint
    from ..services.training_service import TrainingService
    from ..schemas.training_schemas import TrainingRequest

    logger.info("=== API Style Training Example ===")

    # Create training request (like what comes from POST /train)
    training_request = TrainingRequest(
        symbol="BTCUSDT",
        timeframe=TimeFrame.DAY_1,
        model_type=ModelType.PATCHTST,
        context_length=32,
        prediction_length=1,
        num_epochs=2,
        batch_size=16,
        target_column="close",
        use_technical_indicators=True,
        normalize_features=True,
    )

    # Call training service
    training_service = TrainingService()
    result = training_service.start_training(training_request)

    logger.info(f"Training result: {result}")

    if result.success:
        logger.info("=== API Style Prediction Example ===")

        # Now simulate prediction (like POST /predict)
        from ..services.prediction_service import PredictionService
        from ..schemas.prediction_schemas import PredictionRequest

        prediction_request = PredictionRequest(
            symbol="BTCUSDT",
            timeframe=TimeFrame.DAY_1,
            n_steps=1,
            model_type=ModelType.PATCHTST,
        )

        prediction_service = PredictionService()
        prediction_result = prediction_service.predict(prediction_request)

        logger.info(f"Prediction result: {prediction_result}")


def example_hyperparameter_matrix():
    """
    Example showing hyperparameter matrix experiments with visualization
    """
    logger = LoggerFactory.get_logger("HyperparameterMatrix")

    logger.info("=== Hyperparameter Matrix Experiments ===")

    # Load data
    data_loader = CSVDataLoader()
    try:
        data = data_loader.load_data("BTCUSDT", TimeFrame.DAY_1)
        logger.info(f"Loaded {len(data)} rows for hyperparameter experiments")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Create output directory for results
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)

    # Define hyperparameter grid
    param_grid = {
        "context_length": [16, 32, 64],
        "prediction_length": [1, 3, 5],
        "num_epochs": [2, 5],  # Reduced for demo
        "batch_size": [16, 32],
        "learning_rate": [1e-4, 1e-3],
    }

    # Run experiments for PatchTST
    logger.info("Running PatchTST hyperparameter matrix...")

    facade = ModelFacade()
    results = []

    # Generate parameter combinations (limit for demo)
    from itertools import product

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Limit combinations for demo (take first 6)
    combinations = list(product(*param_values))[:6]

    for i, param_combo in enumerate(combinations):
        param_dict = dict(zip(param_names, param_combo))
        logger.info(f"Experiment {i+1}/{len(combinations)}: {param_dict}")

        try:
            # Train model with these parameters
            result = facade.create_and_train_model(
                model_type=ModelType.PATCHTST,
                data=data,
                target_column="close",
                use_technical_indicators=True,
                add_datetime_features=True,
                normalize_features=True,
                **param_dict,
            )

            if result.get("success"):
                # Extract metrics
                result_row = param_dict.copy()
                if "validation_metrics" in result:
                    result_row.update(result["validation_metrics"])
                if "training_metrics" in result:
                    for key, value in result["training_metrics"].items():
                        result_row[f"train_{key}"] = value

                results.append(result_row)
                logger.info(f"Experiment {i+1} completed successfully")
            else:
                logger.error(f"Experiment {i+1} failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Experiment {i+1} error: {e}")
            continue

    if not results:
        logger.error("No successful experiments!")
        return

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(output_dir / "hyperparameter_results.csv", index=False)
    logger.info(f"Saved results to {output_dir / 'hyperparameter_results.csv'}")

    # Visualizations
    logger.info("Creating visualizations...")
    vis_utils = VisualizationUtils()

    # 1. Model comparison on eval_loss
    if "eval_loss" in results_df.columns:
        model_results = {}
        for i, row in results_df.iterrows():
            model_name = f"Config_{i+1}"
            model_results[model_name] = {"rmse": row.get("eval_loss", 0)}

        vis_utils.plot_model_comparison(
            model_results,
            metric="rmse",
            title="Hyperparameter Configuration Comparison",
            save_path=output_dir / "model_comparison.png",
        )

    # 2. Hyperparameter heatmap (if we have enough data)
    if len(results_df) >= 4:
        try:
            # Create simplified heatmap with context_length vs batch_size
            if (
                "context_length" in results_df.columns
                and "batch_size" in results_df.columns
            ):
                vis_utils.plot_hyperparameter_heatmap(
                    results_df,
                    param1="context_length",
                    param2="batch_size",
                    metric="eval_loss",
                    title="Context Length vs Batch Size",
                    save_path=output_dir / "hyperparameter_heatmap.png",
                )
        except Exception as e:
            logger.warning(f"Could not create heatmap: {e}")

    # 3. Feature correlation (using sample data)
    try:
        feature_eng = BasicFeatureEngineering(
            add_technical_indicators=True,
            add_datetime_features=True,
            normalize_features=False,
        )
        sample_data = data.tail(1000)  # Use recent data
        processed_sample = feature_eng.fit_transform(sample_data)

        # Select numeric features only
        numeric_features = processed_sample.select_dtypes(include=[np.number]).columns[
            :15
        ]  # Limit for readability

        vis_utils.plot_correlation_matrix(
            processed_sample,
            features=list(numeric_features),
            title="Feature Correlation Matrix",
            save_path=output_dir / "feature_correlation.png",
        )
    except Exception as e:
        logger.warning(f"Could not create correlation matrix: {e}")

    # Print summary
    logger.info("=== Hyperparameter Experiment Summary ===")
    logger.info(f"Total experiments: {len(results_df)}")

    if "eval_loss" in results_df.columns:
        best_idx = results_df["eval_loss"].idxmin()
        best_config = results_df.loc[best_idx]
        logger.info(f"Best configuration (lowest eval_loss): {best_config.to_dict()}")

    logger.info(f"Results saved to: {output_dir}")


def example_backtesting_with_visualization():
    """
    Example showing backtesting with visualization
    """
    logger = LoggerFactory.get_logger("BacktestExample")

    logger.info("=== Backtesting with Visualization ===")

    # Load data
    data_loader = CSVDataLoader()
    try:
        data = data_loader.load_data("BTCUSDT", TimeFrame.DAY_1)
        test_data = data.tail(500)  # Use recent 500 points for testing
        logger.info(f"Using {len(test_data)} points for backtesting")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Create output directory
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)

    # Generate sample predictions (using simple moving average strategy)
    prices = test_data["close"].values
    predictions = []

    for i in range(len(prices)):
        if i < 10:
            predictions.append(prices[i])
        else:
            # Simple prediction: slight trend continuation
            trend = (prices[i] - prices[i - 5]) / prices[i - 5]
            predicted_price = prices[i] * (1 + trend * 0.5)  # Damped trend
            predictions.append(predicted_price)

    logger.info(f"Generated {len(predictions)} predictions")

    # Run backtest
    backtest_engine = BacktestEngine(
        initial_capital=10000.0, commission=0.001, slippage=0.0001
    )

    # Test different strategies
    strategies = {
        "simple_threshold": {"buy_threshold": 0.02, "sell_threshold": -0.02},
        "momentum": {"lookback": 5, "threshold": 0.01},
        "mean_reversion": {"window": 20, "std_threshold": 1.5},
    }

    backtest_results = {}

    for strategy_name, strategy_params in strategies.items():
        try:
            logger.info(f"Running backtest with {strategy_name} strategy...")

            result = backtest_engine.run_backtest(
                data=test_data,
                predictions=predictions,
                strategy=strategy_name,
                **strategy_params,
            )

            backtest_results[strategy_name] = result
            logger.info(
                f"{strategy_name} completed: Final portfolio value = ${result['final_capital']:.2f}"
            )

        except Exception as e:
            logger.error(f"Backtest failed for {strategy_name}: {e}")
            continue

    # Visualizations
    logger.info("Creating backtest visualizations...")
    vis_utils = VisualizationUtils()

    # 1. Plot predictions vs actual
    vis_utils.plot_price_predictions(
        data=test_data.tail(100),  # Last 100 points for clarity
        predictions=predictions[-100:],
        title="Price Predictions vs Actual (Last 100 Points)",
        save_path=output_dir / "price_predictions.png",
    )

    # 2. Plot backtest results for each strategy
    for strategy_name, result in backtest_results.items():
        if result:
            vis_utils.plot_backtest_results(
                backtest_results=result,
                title=f"Backtest Results - {strategy_name.title()} Strategy",
                save_path=output_dir / f"backtest_{strategy_name}.png",
            )

    # 3. Strategy comparison
    if backtest_results:
        strategy_comparison = {}
        for strategy_name, result in backtest_results.items():
            if result and "metrics" in result:
                strategy_comparison[strategy_name] = result["metrics"]

        if strategy_comparison:
            vis_utils.plot_model_comparison(
                model_results=strategy_comparison,
                metric="total_return",
                title="Strategy Comparison - Total Return",
                save_path=output_dir / "strategy_comparison.png",
            )

    # Print summary
    logger.info("=== Backtesting Summary ===")
    for strategy_name, result in backtest_results.items():
        if result and "metrics" in result:
            metrics = result["metrics"]
            logger.info(f"{strategy_name.title()} Strategy:")
            logger.info(f"  Total Return: {metrics.get('total_return', 0):.4f}")
            logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
            logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.4f}")

    logger.info(f"Visualizations saved to: {output_dir}")


def example_advanced_features():
    """
    Example showing advanced features and customization
    """
    logger = LoggerFactory.get_logger("AdvancedExample")

    logger.info("=== Advanced Features Example ===")

    # Custom feature engineering strategy with datetime features
    custom_feature_eng = BasicFeatureEngineering(
        feature_columns=["open", "high", "low", "close", "volume"],
        add_technical_indicators=True,
        add_datetime_features=True,  # Enable datetime features
        normalize_features=True,
    )

    # Load and process data
    data_loader = CSVDataLoader()
    try:
        data = data_loader.load_data("BTCUSDT", TimeFrame.DAY_1)

        # Advanced preprocessing
        processed_data = custom_feature_eng.fit_transform(data)

        logger.info(
            f"Generated {len(custom_feature_eng.get_feature_names())} features:"
        )
        for feature in custom_feature_eng.get_feature_names()[:20]:  # Show first 20
            logger.info(f"  - {feature}")

        # Create sequences for manual model training
        sequences, targets = custom_feature_eng.create_sequences(
            data=processed_data,
            context_length=64,
            prediction_length=5,  # Multi-step prediction
            target_column="close",
        )

        logger.info(f"Generated sequences shape: {sequences.shape}")
        logger.info(f"Generated targets shape: {targets.shape}")

        # Show data split capabilities
        train_data, val_data, test_data = data_loader.split_data(
            processed_data, train_ratio=0.7, val_ratio=0.15
        )

        logger.info(
            f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )

        # Create feature importance visualization (mock data)
        feature_names = custom_feature_eng.get_feature_names()[:15]
        importance_scores = np.random.normal(
            0, 1, len(feature_names)
        )  # Mock importance

        output_dir = Path("demo_results")
        output_dir.mkdir(exist_ok=True)

        vis_utils = VisualizationUtils()
        vis_utils.plot_feature_importance(
            feature_names=feature_names,
            importance_scores=importance_scores,
            title="Mock Feature Importance",
            save_path=output_dir / "feature_importance.png",
        )

    except Exception as e:
        logger.error(f"Advanced features demo failed: {e}")


if __name__ == "__main__":
    """
    Run examples when script is executed directly
    """
    print("Running FinSight Model Builder Examples...")

    # Run basic workflow
    asyncio.run(example_complete_workflow())

    # Run API style example
    example_api_style_usage()

    # Run hyperparameter matrix experiments
    # example_hyperparameter_matrix()

    # # Run backtesting with visualization
    # example_backtesting_with_visualization()

    # # Run advanced features example
    # example_advanced_features()

    print("All examples completed!")
