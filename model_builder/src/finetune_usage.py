"""
Comprehensive usage examples for the FineTune module.
Demonstrates training, evaluation, prediction, and visualization capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel

from .finetune.main import FineTuneFacade, create_default_facade
from .finetune.config import FineTuneConfig, ModelType, TaskType
from .finetune.predictor import FineTunePredictor
from .common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel


class TrainingRequest(BaseModel):
    """Request schema for training service"""

    data_path: str
    model_name: str = ModelType.PATCH_TSMIXER
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    sequence_length: int = 60
    prediction_horizon: int = 1
    features: List[str] = ["open", "high", "low", "close", "volume"]
    target_column: str = "close"
    use_peft: bool = False
    output_dir: Optional[str] = None


class TrainingResponse(BaseModel):
    """Response schema for training service"""

    success: bool
    model_path: Optional[str] = None
    training_loss: Optional[float] = None
    validation_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    training_duration: Optional[float] = None


class PredictionRequest(BaseModel):
    """Request schema for prediction service"""

    model_path: str
    data_path: Optional[str] = None
    data: Optional[List[Dict[str, float]]] = None
    prediction_timeframe: str = "1d"  # 1h, 4h, 12h, 1d, 1w
    n_steps: int = 1


class PredictionResponse(BaseModel):
    """Response schema for prediction service"""

    success: bool
    predictions: Optional[List[float]] = None
    prediction_dates: Optional[List[str]] = None
    current_price: Optional[float] = None
    predicted_change_pct: Optional[float] = None
    confidence: Optional[float] = None
    error_message: Optional[str] = None


class FineTuneUsageDemo:
    """
    Comprehensive demonstration of FineTune capabilities.
    Shows training, evaluation, prediction, and visualization.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(
            name="FineTuneDemo",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
        )
        self.facade = None

    def basic_training_example(self, data_path: str = "data/1d.csv") -> Dict[str, Any]:
        """
        Basic example of training a model

        Args:
            data_path: Path to the dataset

        Returns:
            Training results
        """
        self.logger.info("🚀 Starting basic training example...")

        # Create facade with default configuration
        self.facade = create_default_facade()

        # Run training
        results = self.facade.finetune(data_path)

        if results["status"] == "success":
            self.logger.info("✅ Training completed successfully!")
            self.logger.info(f"Model saved to: {results['model_path']}")
        else:
            self.logger.error(
                f"❌ Training failed: {results.get('error', 'Unknown error')}"
            )

        return results

    def advanced_training_example(
        self, data_path: str = "data/1d.csv"
    ) -> Dict[str, Any]:
        """
        Advanced training with custom configuration

        Args:
            data_path: Path to the dataset

        Returns:
            Training results
        """
        self.logger.info("🚀 Starting advanced training example...")

        # Create custom configuration
        config = FineTuneConfig()
        config.model_name = ModelType.PATCH_TSMIXER
        config.task_type = TaskType.FORECASTING
        config.num_epochs = 5
        config.batch_size = 8
        config.learning_rate = 3e-5
        config.sequence_length = 120
        config.features = ["open", "high", "low", "close", "volume"]
        config.use_peft = True
        config.use_fp16 = True

        # Create facade with custom config
        self.facade = FineTuneFacade(config)

        # Run training
        results = self.facade.finetune(data_path)

        self.logger.info("📊 Training Results:")
        if results["status"] == "success":
            training = results.get("training", {})
            evaluation = results.get("evaluation", {})

            self.logger.info(f"Training Loss: {training.get('training_loss', 'N/A')}")
            self.logger.info(
                f"Validation R²: {evaluation.get('basic_metrics', {}).get('r2_score', 'N/A')}"
            )
            self.logger.info(
                f"Direction Accuracy: {evaluation.get('financial_metrics', {}).get('direction_accuracy', 'N/A')}"
            )

        return results

    def load_and_predict_example(
        self, model_path: str, data_path: str = "data/1d.csv"
    ) -> Dict[str, Any]:
        """
        Example of loading a trained model and making predictions

        Args:
            model_path: Path to the trained model
            data_path: Path to test data

        Returns:
            Prediction results
        """
        self.logger.info("🔮 Starting prediction example...")

        try:
            # Load data
            df = pd.read_csv(data_path)

            # Create predictor with default config
            config = FineTuneConfig()
            predictor = FineTunePredictor(config)

            # Load the trained model
            predictor.load_model(model_path)

            # Make single prediction
            single_result = predictor.predict_single(df)
            self.logger.info(f"Single Prediction: {single_result['prediction']:.4f}")
            self.logger.info(f"Current Price: {single_result['current_price']:.4f}")
            self.logger.info(
                f"Predicted Change: {single_result['predicted_change_pct']:.2f}%"
            )

            # Make multi-step predictions
            sequence_result = predictor.predict_sequence(df, n_steps=5)
            self.logger.info(f"5-step predictions: {sequence_result['predictions']}")

            return {
                "single_prediction": single_result,
                "sequence_prediction": sequence_result,
            }

        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {str(e)}")
            return {"error": str(e)}

    def backtest_and_visualize(
        self, model_path: str, data_path: str = "data/1d.csv"
    ) -> Dict[str, Any]:
        """
        Backtest model on historical data and create visualizations

        Args:
            model_path: Path to the trained model
            data_path: Path to historical data

        Returns:
            Backtest results with visualization paths
        """
        self.logger.info("📈 Starting backtest and visualization...")

        try:
            # Load data
            df = pd.read_csv(data_path)

            # Ensure timestamp column
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            elif "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
            else:
                df["timestamp"] = pd.date_range(
                    start="2020-01-01", periods=len(df), freq="D"
                )

            # Create predictor
            config = FineTuneConfig()
            predictor = FineTunePredictor(config)
            predictor.load_model(model_path)

            # Backtest parameters
            sequence_length = config.sequence_length
            test_start = len(df) - 100  # Last 100 data points for testing

            predictions = []
            actual_values = []
            dates = []

            # Rolling predictions
            for i in range(test_start, len(df) - sequence_length):
                # Get historical data up to current point
                historical_data = df.iloc[: i + sequence_length]

                # Make prediction
                try:
                    result = predictor.predict_single(historical_data)
                    predictions.append(result["prediction"])
                    actual_values.append(df.iloc[i + sequence_length]["close"])
                    dates.append(df.iloc[i + sequence_length]["timestamp"])
                except Exception as e:
                    self.logger.warning(f"Prediction failed at index {i}: {e}")
                    continue

            # Create visualization
            viz_path = self._create_backtest_visualization(
                predictions, actual_values, dates, model_path
            )

            # Calculate metrics
            if predictions and actual_values:
                mse = np.mean((np.array(predictions) - np.array(actual_values)) ** 2)
                mae = np.mean(np.abs(np.array(predictions) - np.array(actual_values)))
                direction_accuracy = np.mean(
                    np.sign(np.diff(predictions)) == np.sign(np.diff(actual_values))
                )

                results = {
                    "n_predictions": len(predictions),
                    "mse": mse,
                    "mae": mae,
                    "rmse": np.sqrt(mse),
                    "direction_accuracy": direction_accuracy,
                    "visualization_path": viz_path,
                    "predictions": predictions,
                    "actual_values": actual_values,
                    "dates": [
                        d.isoformat() if hasattr(d, "isoformat") else str(d)
                        for d in dates
                    ],
                }

                self.logger.info(f"Backtest Results:")
                self.logger.info(f"  Predictions: {len(predictions)}")
                self.logger.info(f"  RMSE: {np.sqrt(mse):.4f}")
                self.logger.info(f"  Direction Accuracy: {direction_accuracy:.4f}")
                self.logger.info(f"  Visualization: {viz_path}")

                return results
            else:
                return {"error": "No valid predictions generated"}

        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {str(e)}")
            return {"error": str(e)}

    def _create_backtest_visualization(
        self,
        predictions: List[float],
        actual_values: List[float],
        dates: List,
        model_path: str,
    ) -> str:
        """Create backtest visualization"""

        # Setup the plot
        plt.style.use(
            "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
        )
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # Plot 1: Predictions vs Actual
        ax1.plot(
            dates, actual_values, label="Actual Close Price", color="blue", alpha=0.8
        )
        ax1.plot(
            dates, predictions, label="Predicted Close Price", color="red", alpha=0.8
        )
        ax1.set_title(
            "Model Predictions vs Actual Close Prices", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Prediction Error
        errors = np.array(predictions) - np.array(actual_values)
        ax2.plot(dates, errors, color="green", alpha=0.7)
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.8)
        ax2.fill_between(dates, errors, alpha=0.3, color="green")
        ax2.set_title("Prediction Errors", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Error (Predicted - Actual)")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Error Distribution
        ax3.hist(errors, bins=30, alpha=0.7, color="purple", edgecolor="black")
        ax3.axvline(
            x=np.mean(errors),
            color="red",
            linestyle="--",
            label=f"Mean Error: {np.mean(errors):.4f}",
        )
        ax3.axvline(x=0, color="green", linestyle="-", alpha=0.8, label="Zero Error")
        ax3.set_title(
            "Distribution of Prediction Errors", fontsize=14, fontweight="bold"
        )
        ax3.set_xlabel("Error")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add metrics text
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        metrics_text = f"MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {np.sqrt(mse):.6f}"
        fig.text(
            0.02,
            0.02,
            metrics_text,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )

        plt.tight_layout()

        # Save the plot
        output_dir = Path("./finetune_outputs/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_path = (
            output_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(viz_path)

    # Service Functions for API Endpoints

    def training_service(self, request: TrainingRequest) -> TrainingResponse:
        """
        Training service for API endpoint

        Args:
            request: Training request parameters

        Returns:
            Training response with results
        """
        start_time = datetime.now()

        try:
            self.logger.info(
                f"Starting training service with model: {request.model_name}"
            )

            # Validate model name
            valid_models = [model.value for model in ModelType]
            if request.model_name not in valid_models:
                raise ValueError(f"Invalid model name. Must be one of: {valid_models}")

            # Create custom configuration
            config = FineTuneConfig()
            config.model_name = request.model_name
            config.task_type = TaskType.FORECASTING
            config.num_epochs = request.num_epochs
            config.batch_size = request.batch_size
            config.learning_rate = request.learning_rate
            config.sequence_length = request.sequence_length
            config.prediction_horizon = request.prediction_horizon
            config.features = request.features
            config.target_column = request.target_column
            config.use_peft = request.use_peft

            if request.output_dir:
                config.output_dir = Path(request.output_dir)

            # Create facade and run training
            facade = FineTuneFacade(config)
            results = facade.finetune(request.data_path)

            training_duration = (datetime.now() - start_time).total_seconds()

            if results["status"] == "success":
                return TrainingResponse(
                    success=True,
                    model_path=results["model_path"],
                    training_loss=results.get("training", {}).get("training_loss"),
                    validation_metrics=results.get("evaluation", {}).get(
                        "basic_metrics"
                    ),
                    training_duration=training_duration,
                )
            else:
                return TrainingResponse(
                    success=False,
                    error_message=results.get("error", "Unknown training error"),
                    training_duration=training_duration,
                )

        except Exception as e:
            training_duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Training service failed: {str(e)}")
            return TrainingResponse(
                success=False,
                error_message=str(e),
                training_duration=training_duration,
            )

    def prediction_service(self, request: PredictionRequest) -> PredictionResponse:
        """
        Prediction service for API endpoint

        Args:
            request: Prediction request parameters

        Returns:
            Prediction response with results
        """
        try:
            self.logger.info(
                f"Starting prediction service for timeframe: {request.prediction_timeframe}"
            )

            # Create predictor
            config = FineTuneConfig()
            predictor = FineTunePredictor(config)
            predictor.load_model(request.model_path)

            # Prepare data
            if request.data_path:
                df = pd.read_csv(request.data_path)
            elif request.data:
                df = pd.DataFrame(request.data)
            else:
                raise ValueError("Either data_path or data must be provided")

            # Convert timeframe to steps
            timeframe_mapping = {
                "1h": 1,
                "4h": 4,
                "12h": 12,
                "1d": 24,
                "1w": 168,
            }
            n_steps = timeframe_mapping.get(
                request.prediction_timeframe, request.n_steps
            )

            # Make predictions
            if n_steps == 1:
                result = predictor.predict_single(df)
                return PredictionResponse(
                    success=True,
                    predictions=[result["prediction"]],
                    prediction_dates=[datetime.now().isoformat()],
                    current_price=result["current_price"],
                    predicted_change_pct=result["predicted_change_pct"],
                    confidence=result["confidence"],
                )
            else:
                result = predictor.predict_sequence(df, n_steps=n_steps)
                return PredictionResponse(
                    success=True,
                    predictions=result["predictions"],
                    prediction_dates=result["dates"],
                    current_price=result["base_price"],
                    predicted_change_pct=(
                        (result["predictions"][-1] - result["base_price"])
                        / result["base_price"]
                        * 100
                        if result["predictions"]
                        else 0.0
                    ),
                    confidence=0.8,  # Placeholder confidence for multi-step
                )

        except Exception as e:
            self.logger.error(f"Prediction service failed: {str(e)}")
            return PredictionResponse(
                success=False,
                error_message=str(e),
            )


def main():
    """
    Main function demonstrating all capabilities
    """
    demo = FineTuneUsageDemo()

    print("🚀 FineTune Usage Demo")
    print("=" * 50)

    # 1. Basic Training
    print("\n1. Basic Training Example")
    print("-" * 30)
    training_results = demo.basic_training_example("data/1d.csv")

    if training_results["status"] == "success":
        model_path = training_results["model_path"]

        # 2. Prediction Example (skip for now due to loading issues)
        print("\n2. Prediction Example (Skipped - will be fixed in next iteration)")
        print("-" * 30)

        # 3. Service Examples
        print("\n3. Service API Examples")
        print("-" * 30)

        # Training service
        training_request = TrainingRequest(
            data_path="data/1d.csv",
            model_name=ModelType.PATCH_TSMIXER,
            num_epochs=2,
            batch_size=4,
        )
        training_response = demo.training_service(training_request)
        print(f"Training Service Success: {training_response.success}")
        if training_response.success:
            print(f"Model Path: {training_response.model_path}")
            print(f"Training Loss: {training_response.training_loss}")

    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
    #     print(f"Prediction Service Success: {prediction_response.success}")
    #     if prediction_response.success:
    #         print(f"Predicted Price: {prediction_response.predictions[0]:.4f}")
    #         print(
    #             f"Expected Change: {prediction_response.predicted_change_pct:.2f}%"
    #         )

    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
