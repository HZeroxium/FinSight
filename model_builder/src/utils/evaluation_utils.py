# utils/evaluation_utils.py

from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import DataLoader
import numpy

from ..common.logger.logger_factory import LoggerFactory
from ..core.config import Config
from ..training.trainer import ModelTrainer


class EvaluationUtils:
    """Utility class for model evaluation and performance analysis"""

    _logger = LoggerFactory.get_logger(__name__)

    @staticmethod
    def evaluate_single_model(
        model: torch.nn.Module,
        test_loader: DataLoader,
        config: Config,
        device: torch.device,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively

        Args:
            model: PyTorch model to evaluate
            test_loader: Test data loader
            config: Configuration object
            device: PyTorch device
            model_name: Name of the model for logging

        Returns:
            Dict containing evaluation results
        """
        EvaluationUtils._logger.info(f"📊 Evaluating {model_name} model...")

        # Create trainer for evaluation
        trainer = ModelTrainer(config, device)

        # Basic evaluation
        eval_result = trainer.evaluate(model, test_loader)

        # Get predictions for additional analysis
        predictions, targets = model.predict_batch(test_loader, device)

        # Calculate additional metrics using local function
        additional_metrics = EvaluationUtils._calculate_additional_metrics(
            predictions.numpy(), targets.numpy()
        )

        result = {
            "evaluation_result": eval_result,
            "additional_metrics": additional_metrics,
            "model_name": model_name,
            "predictions_sample": {
                "predictions": predictions[:10].tolist(),
                "targets": targets[:10].tolist(),
            },
        }

        EvaluationUtils._logger.info(
            f"✓ {model_name.capitalize()} evaluation completed:"
        )
        for metric, value in eval_result["test_metrics"].items():
            if isinstance(value, (int, float)):
                EvaluationUtils._logger.info(f"  {metric.upper()}: {value:.4f}")

        return result

    @staticmethod
    def evaluate_multiple_models(
        models: Dict[str, torch.nn.Module],
        test_loader: DataLoader,
        config: Config,
        device: torch.device,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate multiple models and compare results

        Args:
            models: Dictionary of models to evaluate
            test_loader: Test data loader
            config: Configuration object
            device: PyTorch device
            model_names: Optional list of model names to evaluate

        Returns:
            Dict containing evaluation results for all models
        """
        if model_names is None:
            model_names = list(models.keys())

        evaluation_results = {}

        for model_name in model_names:
            if model_name not in models:
                EvaluationUtils._logger.warning(
                    f"Model {model_name} not found, skipping..."
                )
                continue

            model = models[model_name]
            result = EvaluationUtils.evaluate_single_model(
                model, test_loader, config, device, model_name
            )
            evaluation_results[model_name] = result

        # Add model comparison
        if len(evaluation_results) > 1:
            comparison = EvaluationUtils._compare_models(evaluation_results)
            evaluation_results["model_comparison"] = comparison

        EvaluationUtils._logger.info(
            f"✓ Completed evaluation for {len(evaluation_results)} models"
        )
        return evaluation_results

    @staticmethod
    def _calculate_additional_metrics(
        predictions: "numpy.ndarray", targets: "numpy.ndarray"
    ) -> Dict[str, float]:
        """Calculate additional evaluation metrics beyond basic ones"""
        import numpy as np

        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Remove invalid values
        valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
        if not valid_mask.any():
            return {
                key: 0.0
                for key in [
                    "prediction_bias",
                    "prediction_variance",
                    "correlation",
                    "max_error",
                    "quantile_loss_50",
                    "hit_rate_5pct",
                ]
            }

        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        try:
            correlation = (
                np.corrcoef(pred_valid, target_valid)[0, 1]
                if len(pred_valid) > 1
                else 0.0
            )
            correlation = correlation if np.isfinite(correlation) else 0.0
        except:
            correlation = 0.0

        try:
            # Safe hit rate calculation
            relative_errors = np.abs(pred_valid - target_valid)
            threshold_values = 0.05 * np.abs(target_valid)
            safe_threshold = np.where(threshold_values == 0, 1e-8, threshold_values)
            hit_rate = np.mean(relative_errors < safe_threshold)
        except:
            hit_rate = 0.0

        return {
            "prediction_bias": float(np.mean(pred_valid - target_valid)),
            "prediction_variance": float(np.var(pred_valid - target_valid)),
            "correlation": float(correlation),
            "max_error": float(np.max(np.abs(pred_valid - target_valid))),
            "quantile_loss_50": float(
                np.mean(
                    np.maximum(
                        0.5 * (pred_valid - target_valid),
                        (0.5 - 1) * (pred_valid - target_valid),
                    )
                )
            ),
            "hit_rate_5pct": float(hit_rate),
        }

    @staticmethod
    def _compare_models(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare models based on evaluation metrics"""
        comparison = {}
        metrics_to_compare = ["rmse", "mae", "mape", "directional_accuracy", "r2"]

        for metric in metrics_to_compare:
            metric_values = {}
            for model_name, results in evaluation_results.items():
                if model_name != "model_comparison":
                    test_metrics = results["evaluation_result"]["test_metrics"]
                    if metric in test_metrics:
                        metric_values[model_name] = test_metrics[metric]

            if metric_values:
                best_model = min(metric_values.items(), key=lambda x: x[1])
                worst_model = max(metric_values.items(), key=lambda x: x[1])

                comparison[metric] = {
                    "values": metric_values,
                    "best_model": best_model[0],
                    "best_value": float(best_model[1]),
                    "worst_model": worst_model[0],
                    "worst_value": float(worst_model[1]),
                    "improvement": float(
                        (worst_model[1] - best_model[1]) / worst_model[1] * 100
                    ),
                }

        return comparison

    @staticmethod
    def create_evaluation_report(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive evaluation report

        Args:
            evaluation_results: Evaluation results from multiple models

        Returns:
            Dict containing evaluation report
        """
        report = {
            "model_count": len(
                [k for k in evaluation_results.keys() if k != "model_comparison"]
            ),
            "detailed_results": evaluation_results,
            "performance_ranking": EvaluationUtils._rank_models(evaluation_results),
            "recommendations": EvaluationUtils._generate_evaluation_recommendations(
                evaluation_results
            ),
        }

        return report

    @staticmethod
    def _rank_models(evaluation_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Rank models by different metrics"""
        rankings = {}
        metrics_to_rank = ["rmse", "mae", "mape", "directional_accuracy", "r2"]

        for metric in metrics_to_rank:
            model_scores = {}
            for model_name, results in evaluation_results.items():
                if model_name != "model_comparison":
                    test_metrics = results["evaluation_result"]["test_metrics"]
                    if metric in test_metrics:
                        model_scores[model_name] = test_metrics[metric]

            if model_scores:
                # For most metrics, lower is better. For directional_accuracy and r2, higher is better
                reverse = metric in ["directional_accuracy", "r2"]
                sorted_models = sorted(
                    model_scores.items(), key=lambda x: x[1], reverse=reverse
                )
                rankings[metric] = [model[0] for model in sorted_models]

        return rankings

    @staticmethod
    def _generate_evaluation_recommendations(
        evaluation_results: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []

        if not evaluation_results or len(evaluation_results) == 0:
            return ["No evaluation results available for analysis"]

        # Get model comparison if available
        model_comparison = evaluation_results.get("model_comparison", {})

        if model_comparison:
            # Find consistently best performing model
            rankings = EvaluationUtils._rank_models(evaluation_results)
            model_rank_scores = {}

            for model_name in evaluation_results.keys():
                if model_name != "model_comparison":
                    total_rank = 0
                    rank_count = 0

                    for metric, ranking in rankings.items():
                        if model_name in ranking:
                            total_rank += ranking.index(model_name) + 1
                            rank_count += 1

                    if rank_count > 0:
                        model_rank_scores[model_name] = total_rank / rank_count

            if model_rank_scores:
                best_overall = min(model_rank_scores.items(), key=lambda x: x[1])
                recommendations.append(
                    f"Best overall model: {best_overall[0]} (avg rank: {best_overall[1]:.1f})"
                )

        # Analyze specific metrics
        for model_name, results in evaluation_results.items():
            if model_name != "model_comparison":
                test_metrics = results["evaluation_result"]["test_metrics"]

                # Check for poor performance indicators
                if "r2" in test_metrics and test_metrics["r2"] < 0:
                    recommendations.append(
                        f"{model_name}: Poor R² score ({test_metrics['r2']:.3f}) - consider model architecture changes"
                    )

                if (
                    "directional_accuracy" in test_metrics
                    and test_metrics["directional_accuracy"] < 55
                ):
                    recommendations.append(
                        f"{model_name}: Low directional accuracy ({test_metrics['directional_accuracy']:.1f}%) - may need feature engineering"
                    )

                if "mape" in test_metrics and test_metrics["mape"] > 100:
                    recommendations.append(
                        f"{model_name}: High MAPE ({test_metrics['mape']:.1f}%) - check for outliers or scaling issues"
                    )

        return recommendations

    @staticmethod
    def get_best_model(
        evaluation_results: Dict[str, Any], metric: str = "rmse"
    ) -> Optional[str]:
        """
        Get the best performing model based on a specific metric

        Args:
            evaluation_results: Evaluation results
            metric: Metric to use for comparison

        Returns:
            Name of best model or None
        """
        model_scores = {}

        for model_name, results in evaluation_results.items():
            if model_name != "model_comparison":
                test_metrics = results["evaluation_result"]["test_metrics"]
                if metric in test_metrics:
                    model_scores[model_name] = test_metrics[metric]

        if not model_scores:
            return None

        # For most metrics, lower is better. For directional_accuracy and r2, higher is better
        reverse = metric in ["directional_accuracy", "r2"]
        best_model = (
            max(model_scores.items(), key=lambda x: x[1])
            if reverse
            else min(model_scores.items(), key=lambda x: x[1])
        )

        return best_model[0]

    @staticmethod
    def print_evaluation_summary(evaluation_results: Dict[str, Any]) -> None:
        """Print a formatted summary of evaluation results"""
        EvaluationUtils._logger.info("\n" + "=" * 60)
        EvaluationUtils._logger.info("📈 EVALUATION SUMMARY")
        EvaluationUtils._logger.info("=" * 60)

        # Print individual model results
        for model_name, results in evaluation_results.items():
            if model_name != "model_comparison":
                EvaluationUtils._logger.info(f"\n{model_name.upper()} MODEL:")
                test_metrics = results["evaluation_result"]["test_metrics"]

                for metric, value in test_metrics.items():
                    if isinstance(value, (int, float)):
                        EvaluationUtils._logger.info(f"  {metric}: {value:.4f}")

        # Print model comparison if available
        if "model_comparison" in evaluation_results:
            EvaluationUtils._logger.info("\nMODEL COMPARISON:")
            comparison = evaluation_results["model_comparison"]

            for metric, comp_data in comparison.items():
                best_model = comp_data["best_model"]
                best_value = comp_data["best_value"]
                EvaluationUtils._logger.info(
                    f"  Best {metric.upper()}: {best_model} ({best_value:.4f})"
                )
