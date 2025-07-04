# finetune/evaluator.py

"""
Comprehensive evaluator for fine-tuned financial models.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import wandb

from ..common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel
from .config import FineTuneConfig, TaskType
from .data_processor import FinancialTimeSeriesDataset


class FineTuneEvaluator:
    """Comprehensive evaluator for fine-tuned models"""

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger(
            name="FineTuneEvaluator",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
        )

    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_dataset: FinancialTimeSeriesDataset,
        tokenizer=None,
        log_to_wandb: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation

        Args:
            model: Trained model to evaluate
            test_dataset: Test dataset
            tokenizer: Tokenizer if needed
            log_to_wandb: Whether to log results to W&B

        Returns:
            Comprehensive evaluation results
        """
        self.logger.info("🔍 Starting comprehensive model evaluation...")

        # Get predictions
        predictions, targets = self._get_predictions(model, test_dataset)

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)

        # Financial-specific analysis
        financial_metrics = self._calculate_financial_metrics(predictions, targets)

        # Trend analysis
        trend_analysis = self._analyze_trends(predictions, targets)

        # Create visualizations
        viz_paths = self._create_visualizations(predictions, targets)

        results = {
            "basic_metrics": metrics,
            "financial_metrics": financial_metrics,
            "trend_analysis": trend_analysis,
            "visualizations": viz_paths,
            "summary": self._create_summary(metrics, financial_metrics),
        }

        # Log to W&B if enabled
        if log_to_wandb and self.config.wandb.enabled:
            self._log_to_wandb(predictions, targets, metrics)

        self.logger.info("✅ Evaluation completed!")
        self._log_results(results)

        return results

    def _get_predictions(
        self, model: torch.nn.Module, test_dataset: FinancialTimeSeriesDataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions on test data"""
        model.eval()
        predictions = []
        targets = []

        device = next(model.parameters()).device

        with torch.no_grad():
            for i in range(len(test_dataset)):
                batch = test_dataset[i]

                # Move to device
                if isinstance(batch["input_ids"], torch.Tensor):
                    input_ids = batch["input_ids"].unsqueeze(0).to(device)
                else:
                    input_ids = torch.FloatTensor([batch["input_ids"]]).to(device)

                # Get prediction
                if (
                    hasattr(model, "generate")
                    and self.config.task_type != TaskType.REGRESSION
                ):
                    # For generative models
                    outputs = model.generate(input_ids, max_length=32, do_sample=False)
                    pred = outputs[0][-1].cpu().item()  # Simple extraction
                else:
                    # For regression/classification
                    outputs = model(input_ids)
                    if hasattr(outputs, "logits"):
                        pred = outputs.logits.cpu().numpy().flatten()[0]
                    else:
                        pred = outputs.cpu().numpy().flatten()[0]

                predictions.append(pred)
                targets.append(
                    batch["targets"].item()
                    if isinstance(batch["targets"], torch.Tensor)
                    else batch["targets"]
                )

        return np.array(predictions), np.array(targets)

    def _calculate_metrics(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic evaluation metrics"""
        # Remove any invalid values
        mask = ~(
            np.isnan(predictions)
            | np.isnan(targets)
            | np.isinf(predictions)
            | np.isinf(targets)
        )
        pred_clean = predictions[mask]
        target_clean = targets[mask]

        if len(pred_clean) == 0:
            return {"error": "No valid predictions"}

        metrics = {
            "mse": mean_squared_error(target_clean, pred_clean),
            "mae": mean_absolute_error(target_clean, pred_clean),
            "rmse": np.sqrt(mean_squared_error(target_clean, pred_clean)),
        }

        # R² score
        try:
            metrics["r2_score"] = r2_score(target_clean, pred_clean)
        except:
            metrics["r2_score"] = 0.0

        # MAPE (Mean Absolute Percentage Error)
        try:
            mape = np.mean(np.abs((target_clean - pred_clean) / target_clean)) * 100
            metrics["mape"] = mape if np.isfinite(mape) else 0.0
        except:
            metrics["mape"] = 0.0

        return metrics

    def _calculate_financial_metrics(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate financial-specific metrics"""
        # Direction accuracy (trend prediction)
        pred_changes = np.diff(predictions)
        target_changes = np.diff(targets)

        direction_accuracy = np.mean(np.sign(pred_changes) == np.sign(target_changes))

        # Volatility comparison
        pred_volatility = np.std(predictions)
        target_volatility = np.std(targets)
        volatility_ratio = (
            pred_volatility / target_volatility if target_volatility > 0 else 1.0
        )

        # Maximum drawdown (for price predictions)
        def max_drawdown(prices):
            peak = np.maximum.accumulate(prices)
            drawdown = (peak - prices) / peak
            return np.max(drawdown)

        pred_max_dd = max_drawdown(predictions)
        target_max_dd = max_drawdown(targets)

        return {
            "direction_accuracy": direction_accuracy,
            "volatility_ratio": volatility_ratio,
            "predicted_volatility": pred_volatility,
            "actual_volatility": target_volatility,
            "predicted_max_drawdown": pred_max_dd,
            "actual_max_drawdown": target_max_dd,
            "correlation": (
                np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0
            ),
        }

    def _analyze_trends(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction trends and patterns"""
        # Trend strength
        pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
        target_trend = np.polyfit(range(len(targets)), targets, 1)[0]

        # Bias analysis
        residuals = predictions - targets
        bias = np.mean(residuals)
        bias_trend = np.polyfit(range(len(residuals)), residuals, 1)[0]

        return {
            "predicted_trend_slope": pred_trend,
            "actual_trend_slope": target_trend,
            "trend_agreement": (
                1.0 if np.sign(pred_trend) == np.sign(target_trend) else 0.0
            ),
            "prediction_bias": bias,
            "bias_trend": bias_trend,
            "residual_std": np.std(residuals),
        }

    def _create_visualizations(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, str]:
        """Create evaluation visualizations"""
        output_dir = self.config.get_model_output_dir() / "evaluation"
        output_dir.mkdir(exist_ok=True)

        viz_paths = {}

        try:
            # 1. Predictions vs Actual
            plt.figure(figsize=(12, 6))
            plt.plot(targets, label="Actual", alpha=0.8)
            plt.plot(predictions, label="Predicted", alpha=0.8)
            plt.legend()
            plt.title("Predictions vs Actual Values")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.grid(True)
            path = output_dir / "predictions_vs_actual.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            viz_paths["predictions_vs_actual"] = str(path)

            # 2. Scatter plot
            plt.figure(figsize=(8, 8))
            plt.scatter(targets, predictions, alpha=0.6)
            plt.plot([min(targets), max(targets)], [min(targets), max(targets)], "r--")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Predictions vs Actual (Scatter)")
            plt.grid(True)
            path = output_dir / "scatter_plot.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            viz_paths["scatter_plot"] = str(path)

            # 3. Residuals
            residuals = predictions - targets
            plt.figure(figsize=(12, 4))
            plt.plot(residuals)
            plt.axhline(y=0, color="r", linestyle="--")
            plt.title("Prediction Residuals")
            plt.xlabel("Time Step")
            plt.ylabel("Residual")
            plt.grid(True)
            path = output_dir / "residuals.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            viz_paths["residuals"] = str(path)

        except Exception as e:
            self.logger.warning(f"Failed to create some visualizations: {e}")

        return viz_paths

    def _create_summary(
        self, metrics: Dict[str, float], financial_metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """Create evaluation summary"""
        summary = {}

        # Performance grade
        r2 = metrics.get("r2_score", 0)
        if r2 > 0.8:
            grade = "Excellent"
        elif r2 > 0.6:
            grade = "Good"
        elif r2 > 0.4:
            grade = "Fair"
        else:
            grade = "Poor"

        summary["performance_grade"] = grade
        summary["key_strength"] = f"R² Score: {r2:.3f}"

        # Direction accuracy assessment
        dir_acc = financial_metrics.get("direction_accuracy", 0)
        if dir_acc > 0.6:
            summary["trend_prediction"] = "Strong"
        elif dir_acc > 0.5:
            summary["trend_prediction"] = "Moderate"
        else:
            summary["trend_prediction"] = "Weak"

        return summary

    def _log_results(self, results: Dict[str, Any]) -> None:
        """Log evaluation results"""
        metrics = results["basic_metrics"]
        financial = results["financial_metrics"]

        self.logger.info("📊 Evaluation Results:")
        self.logger.info(f"  R² Score: {metrics.get('r2_score', 0):.4f}")
        self.logger.info(f"  RMSE: {metrics.get('rmse', 0):.4f}")
        self.logger.info(
            f"  Direction Accuracy: {financial.get('direction_accuracy', 0):.4f}"
        )
        self.logger.info(f"  Correlation: {financial.get('correlation', 0):.4f}")
        self.logger.info(
            f"  Performance Grade: {results['summary'].get('performance_grade', 'N/A')}"
        )

    def _log_to_wandb(
        self, predictions: np.ndarray, targets: np.ndarray, metrics: Dict[str, float]
    ) -> None:
        """Log evaluation results to W&B"""
        try:
            # Log metrics
            wandb.log(metrics)

            # Create and log visualizations
            self._create_evaluation_plots(predictions, targets)

            self.logger.info("✓ Results logged to W&B")

        except Exception as e:
            self.logger.warning(f"Failed to log to W&B: {str(e)}")

    def _create_evaluation_plots(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> None:
        """Create evaluation plots for W&B"""

        # Filter valid values for plotting
        valid_mask = np.isfinite(predictions) & np.isfinite(targets)
        pred_clean = predictions[valid_mask]
        target_clean = targets[valid_mask]

        if len(pred_clean) == 0:
            return

        # 1. Scatter plot of predictions vs targets
        plt.figure(figsize=(10, 8))
        plt.scatter(target_clean, pred_clean, alpha=0.6)
        plt.plot(
            [target_clean.min(), target_clean.max()],
            [target_clean.min(), target_clean.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Predictions vs True Values")
        wandb.log({"predictions_vs_targets": wandb.Image(plt)})
        plt.close()

        # 2. Residual plot
        residuals = target_clean - pred_clean
        plt.figure(figsize=(10, 6))
        plt.scatter(pred_clean, residuals, alpha=0.6)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predictions")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        wandb.log({"residual_plot": wandb.Image(plt)})
        plt.close()

        # 3. Distribution plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.hist(target_clean, bins=50, alpha=0.7, label="True Values", color="blue")
        ax1.hist(pred_clean, bins=50, alpha=0.7, label="Predictions", color="red")
        ax1.set_xlabel("Values")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution Comparison")
        ax1.legend()

        ax2.hist(residuals, bins=50, alpha=0.7, color="green")
        ax2.set_xlabel("Residuals")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Residuals Distribution")

        wandb.log({"distributions": wandb.Image(fig)})
        plt.close()
