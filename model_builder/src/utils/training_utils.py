# utils/training_utils.py

import time
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import DataLoader

from ..common.logger.logger_factory import LoggerFactory
from ..core.config import Config
from ..training.trainer import ModelTrainer
from . import CommonUtils


class TrainingUtils:
    """Utility class for training operations and monitoring"""

    _logger = LoggerFactory.get_logger(__name__)

    @staticmethod
    def train_single_model(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device,
        model_name: str,
        save_best: bool = True,
    ) -> Dict[str, Any]:
        """
        Train a single model with comprehensive monitoring

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: PyTorch device
            model_name: Name of the model for logging
            save_best: Whether to save the best model

        Returns:
            Dict containing training results and metadata
        """
        TrainingUtils._logger.info(f"🚂 Training {model_name} model...")

        # Create trainer
        trainer = ModelTrainer(config, device)

        # Train model
        start_time = time.time()
        training_result = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_best=save_best,
        )
        training_time = time.time() - start_time

        # Get training summary
        training_summary = trainer.get_training_summary()

        result = {
            "training_result": training_result,
            "training_summary": training_summary,
            "training_time": training_time,
            "model_name": model_name,
            "trainer_config": {
                "optimizer": config.model.optimizer_type.value,
                "scheduler": config.model.scheduler_type.value,
                "learning_rate": config.model.learning_rate,
                "epochs": config.model.epochs,
            },
        }

        TrainingUtils._logger.info(f"✓ {model_name.capitalize()} training completed:")
        TrainingUtils._logger.info(
            f"  Best val loss: {training_result['best_val_loss']:.6f}"
        )
        TrainingUtils._logger.info(
            f"  Training time: {CommonUtils.format_duration(training_time)}"
        )
        TrainingUtils._logger.info(f"  Epochs: {training_result['epochs_completed']}")

        return result

    @staticmethod
    def train_multiple_models(
        models: Dict[str, torch.nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Train multiple models and return consolidated results

        Args:
            models: Dictionary of models to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: PyTorch device
            model_names: Optional list of model names to train (None for all)

        Returns:
            Dict containing training results for all models
        """
        if model_names is None:
            model_names = list(models.keys())

        training_results = {}

        for model_name in model_names:
            if model_name not in models:
                TrainingUtils._logger.warning(
                    f"Model {model_name} not found, skipping..."
                )
                continue

            model = models[model_name]
            result = TrainingUtils.train_single_model(
                model, train_loader, val_loader, config, device, model_name
            )
            training_results[model_name] = result

        TrainingUtils._logger.info(
            f"✓ Completed training for {len(training_results)} models"
        )
        return training_results

    @staticmethod
    def get_training_summary(training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of training results across all models

        Args:
            training_results: Training results from multiple models

        Returns:
            Dict containing training summary
        """
        summary = {
            "total_models": len(training_results),
            "total_training_time": sum(
                result["training_time"] for result in training_results.values()
            ),
            "best_models": {},
            "training_metrics": {},
        }

        # Find best performing models by validation loss
        if training_results:
            best_val_loss = float("inf")
            best_model = None

            for model_name, result in training_results.items():
                val_loss = result["training_result"]["best_val_loss"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model_name

            summary["best_models"]["by_val_loss"] = {
                "model": best_model,
                "val_loss": best_val_loss,
            }

        # Aggregate training metrics
        for model_name, result in training_results.items():
            training_result = result["training_result"]
            summary["training_metrics"][model_name] = {
                "final_train_loss": training_result["training_history"]["train_loss"][
                    -1
                ],
                "final_val_loss": training_result["training_history"]["val_loss"][-1],
                "best_val_loss": training_result["best_val_loss"],
                "epochs_completed": training_result["epochs_completed"],
                "training_time": result["training_time"],
            }

        return summary

    @staticmethod
    def monitor_training_progress(training_results: Dict[str, Any]) -> None:
        """
        Monitor and log training progress across models

        Args:
            training_results: Training results from multiple models
        """
        TrainingUtils._logger.info("\n" + "=" * 60)
        TrainingUtils._logger.info("📊 TRAINING PROGRESS SUMMARY")
        TrainingUtils._logger.info("=" * 60)

        summary = TrainingUtils.get_training_summary(training_results)

        TrainingUtils._logger.info(f"Total models trained: {summary['total_models']}")
        TrainingUtils._logger.info(
            f"Total training time: {CommonUtils.format_duration(summary['total_training_time'])}"
        )

        if "by_val_loss" in summary["best_models"]:
            best_info = summary["best_models"]["by_val_loss"]
            TrainingUtils._logger.info(
                f"Best model: {best_info['model']} (val_loss: {best_info['val_loss']:.6f})"
            )

        TrainingUtils._logger.info("\nIndividual model performance:")
        for model_name, metrics in summary["training_metrics"].items():
            TrainingUtils._logger.info(f"  {model_name}:")
            TrainingUtils._logger.info(
                f"    Best val loss: {metrics['best_val_loss']:.6f}"
            )
            TrainingUtils._logger.info(f"    Epochs: {metrics['epochs_completed']}")
            TrainingUtils._logger.info(
                f"    Time: {CommonUtils.format_duration(metrics['training_time'])}"
            )

    @staticmethod
    def create_training_report(training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive training report

        Args:
            training_results: Training results from multiple models

        Returns:
            Dict containing comprehensive training report
        """
        summary = TrainingUtils.get_training_summary(training_results)

        report = {
            "summary": summary,
            "detailed_results": training_results,
            "recommendations": TrainingUtils._generate_training_recommendations(
                training_results
            ),
            "timestamp": CommonUtils.get_readable_timestamp(),
        }

        return report

    @staticmethod
    def _generate_training_recommendations(
        training_results: Dict[str, Any],
    ) -> List[str]:
        """Generate training recommendations based on results"""
        recommendations = []

        if not training_results:
            return ["No training results available for analysis"]

        # Analyze convergence
        for model_name, result in training_results.items():
            history = result["training_result"]["training_history"]
            train_losses = history["train_loss"]
            val_losses = history["val_loss"]

            # Check for overfitting
            if len(val_losses) > 5:
                recent_val_trend = val_losses[-3:]
                if (
                    len(recent_val_trend) == 3
                    and recent_val_trend[-1] > recent_val_trend[0]
                ):
                    recommendations.append(
                        f"{model_name}: Consider early stopping - validation loss increasing"
                    )

            # Check for underfitting
            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]
            if final_val_loss < final_train_loss * 1.5:
                recommendations.append(
                    f"{model_name}: May benefit from more training epochs"
                )

        # Compare models
        if len(training_results) > 1:
            val_losses = {
                name: result["training_result"]["best_val_loss"]
                for name, result in training_results.items()
            }
            best_model = min(val_losses.items(), key=lambda x: x[1])
            recommendations.append(
                f"Best performing model: {best_model[0]} (val_loss: {best_model[1]:.6f})"
            )

        return recommendations
