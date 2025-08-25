# models/trainer.py

"""Model training and evaluation for sentiment analysis."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import os

import mlflow
import numpy as np
import torch
from datasets import DatasetDict
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ..core.config import TrainingConfig
from ..data.dataset import DatasetPreparator
from ..schemas.training_schemas import (
    TrainingMetrics,
    EvaluationResult,
    ClassificationReport,
)
from ..utils.file_utils import ensure_directory


class SentimentTrainer:
    """Handles training and evaluation of sentiment analysis models."""

    def __init__(self, config: TrainingConfig, dataset_preparator: DatasetPreparator):
        """Initialize the trainer.

        Args:
            config: Training configuration
            dataset_preparator: Dataset preparation utility
        """
        self.config = config
        self.dataset_preparator = dataset_preparator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")

        # Set random seeds for reproducibility
        self._set_random_seeds()

    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed_all(self.config.random_seed)

    def _setup_mlflow_environment(self, registry_config: Any) -> None:
        """Setup MLflow environment variables.

        Args:
            registry_config: Registry configuration object
        """
        if hasattr(registry_config, "tracking_uri"):
            mlflow.set_tracking_uri(registry_config.tracking_uri)
            logger.info(f"MLflow tracking URI set to: {registry_config.tracking_uri}")

        # Set environment variables for S3/MinIO integration
        if (
            hasattr(registry_config, "aws_access_key_id")
            and registry_config.aws_access_key_id
        ):
            os.environ["AWS_ACCESS_KEY_ID"] = registry_config.aws_access_key_id
            logger.info("AWS_ACCESS_KEY_ID environment variable set")

        if (
            hasattr(registry_config, "aws_secret_access_key")
            and registry_config.aws_secret_access_key
        ):
            os.environ["AWS_SECRET_ACCESS_KEY"] = registry_config.aws_secret_access_key
            logger.info("AWS_SECRET_ACCESS_KEY environment variable set")

        if hasattr(registry_config, "aws_region") and registry_config.aws_region:
            os.environ["AWS_DEFAULT_REGION"] = registry_config.aws_region
            logger.info(f"AWS_DEFAULT_REGION set to: {registry_config.aws_region}")

        if (
            hasattr(registry_config, "s3_endpoint_url")
            and registry_config.s3_endpoint_url
        ):
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = registry_config.s3_endpoint_url
            logger.info(
                f"MLFLOW_S3_ENDPOINT_URL set to: {registry_config.s3_endpoint_url}"
            )

        # Also set artifact root if available
        if (
            hasattr(registry_config, "artifact_location")
            and registry_config.artifact_location
        ):
            os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = (
                registry_config.artifact_location
            )
            logger.info(
                f"MLFLOW_DEFAULT_ARTIFACT_ROOT set to: {registry_config.artifact_location}"
            )

        logger.info("MLflow environment configured for training")

    def train(
        self,
        datasets: DatasetDict,
        output_dir: Path,
        experiment_name: str = "sentiment-analysis",
        registry_config: Optional[Any] = None,
    ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, TrainingMetrics]:
        """Train the sentiment analysis model.

        Args:
            datasets: DatasetDict containing train/validation/test data
            output_dir: Directory to save training outputs
            experiment_name: Name for MLflow experiment
            registry_config: MLflow registry configuration

        Returns:
            Tuple of (trained_model, tokenizer, training_metrics)
        """
        logger.info(f"Starting training with {self.config.backbone}")

        # Setup MLflow environment variables if registry config is provided
        if registry_config:
            self._setup_mlflow_environment(registry_config)
        else:
            logger.warning(
                "No registry config provided - MLflow artifact logging may not work properly"
            )

        # Setup MLflow experiment
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set to: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set MLflow experiment: {e}")
            raise

        with mlflow.start_run() as run:
            # Log configuration
            self._log_config_to_mlflow()

            # Load tokenizer and model
            tokenizer, model = self._load_tokenizer_and_model()

            # Tokenize datasets first
            tokenized_datasets = self._tokenize_datasets(datasets, tokenizer)

            # Prepare training arguments
            training_args = self._prepare_training_arguments(output_dir)

            # Create trainer with tokenized datasets
            trainer = self._create_trainer(
                model, tokenizer, tokenized_datasets, training_args
            )

            # Train the model
            logger.info("Starting model training")
            train_result = trainer.train()

            # Evaluate the model using tokenized datasets
            logger.info("Evaluating model performance")
            eval_results, classification_report_result = self._evaluate_model(
                trainer, tokenized_datasets
            )

            # Log metrics to MLflow
            self._log_metrics_to_mlflow(train_result, eval_results)

            # Save model and tokenizer
            model_path = output_dir / "model"
            self._save_model_and_tokenizer(trainer, tokenizer, model_path)

            # Log artifacts to MLflow with error handling
            try:
                logger.info("Logging model artifacts to MLflow...")
                mlflow.log_artifact(str(model_path), "model")
                logger.info("Model artifacts logged successfully")
            except Exception as e:
                logger.error(f"Failed to log model artifacts: {e}")
                logger.warning("Continuing without artifact logging...")

            # Save preprocessing config and label mapping
            self._save_training_artifacts(output_dir)

            # Log additional artifacts to MLflow with error handling
            try:
                logger.info("Logging preprocessing artifacts to MLflow...")
                mlflow.log_artifact(
                    str(output_dir / "preprocessing_config.json"),
                    "preprocessing_config",
                )
                mlflow.log_artifact(str(output_dir / "id2label.json"), "label_mapping")
                logger.info("Preprocessing artifacts logged successfully")
            except Exception as e:
                logger.error(f"Failed to log preprocessing artifacts: {e}")
                logger.warning("Continuing without preprocessing artifact logging...")

            # Get training metrics
            training_metrics = TrainingMetrics(
                train_loss=train_result.training_loss,
                eval_results=eval_results,
                classification_report=classification_report_result,
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
            )

            logger.info("Training completed successfully")
            return model, tokenizer, training_metrics

    def _load_tokenizer_and_model(
        self,
    ) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """Load tokenizer and model from the specified backbone.

        Returns:
            Tuple of (tokenizer, model)
        """
        backbone = self.config.backbone.value
        num_labels = len(self.dataset_preparator.label_mapping)

        logger.info(f"Loading tokenizer and model from {backbone}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(backbone)

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            backbone,
            num_labels=num_labels,
            id2label=self.dataset_preparator.reverse_label_mapping,
            label2id=self.dataset_preparator.label_mapping,
        )

        # Move model to device
        model = model.to(self.device)

        logger.info(f"Model loaded with {num_labels} labels")
        return tokenizer, model

    def _prepare_training_arguments(self, output_dir: Path) -> TrainingArguments:
        """Prepare training arguments for the Hugging Face Trainer.

        Args:
            output_dir: Directory to save training outputs

        Returns:
            TrainingArguments configuration
        """
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.gradient_clip_val,  # Fixed parameter name
            # Evaluation and logging
            eval_strategy=self.config.evaluation_strategy.value,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            # Early stopping
            load_best_model_at_end=True,
            metric_for_best_model=self.config.primary_metric.value,
            greater_is_better=True,
            # Reproducibility
            seed=self.config.random_seed,
            dataloader_pin_memory=False,
            # Reporting
            report_to="none",  # Disable wandb, use MLflow instead
            run_name="sentiment-analysis-training",
            # Save strategy
            save_strategy=self.config.save_strategy.value,
            # Remove unused columns
            remove_unused_columns=False,
            fp16=True,
            tf32=True,
            group_by_length=True,
            gradient_checkpointing=True,
        )

    def _create_trainer(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        tokenized_datasets: DatasetDict,
        training_args: TrainingArguments,
    ) -> Trainer:
        """Create the Hugging Face Trainer.

        Args:
            model: The model to train
            tokenizer: The tokenizer for text processing
            tokenized_datasets: DatasetDict containing tokenized data
            training_args: Training arguments

        Returns:
            Configured Trainer instance
        """

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            processing_class=tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            ],
        )

        return trainer

    def _tokenize_datasets(
        self, datasets: DatasetDict, tokenizer: AutoTokenizer
    ) -> DatasetDict:
        """Tokenize the datasets using the provided tokenizer.

        Args:
            datasets: DatasetDict containing the data
            tokenizer: Tokenizer to use

        Returns:
            Tokenized DatasetDict
        """
        tokenizer_config = self.dataset_preparator.get_tokenizer_config()

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=tokenizer_config.truncation,
                padding=tokenizer_config.padding,
                max_length=tokenizer_config.max_length,
                return_tensors=None,  # Return lists, not tensors
            )

        # Tokenize all datasets
        tokenized_datasets = {}
        for split_name, dataset in datasets.items():
            # Tokenize the dataset
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                # num_proc=os.cpu_count(),
                remove_columns=[
                    col for col in dataset.column_names if col not in ["label"]
                ],
            )
            # Rename 'label' column to 'labels' for Hugging Face compatibility
            tokenized_datasets[split_name] = tokenized_dataset.rename_column(
                "label", "labels"
            )

        return DatasetDict(tokenized_datasets)

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            eval_pred: Tuple of (predictions, labels) from the trainer

        Returns:
            Dictionary of metric names and values
        """
        predictions, labels = eval_pred

        # Convert logits to predictions
        predictions = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average="macro")
        f1_weighted = f1_score(labels, predictions, average="weighted")

        # Per-class F1 scores
        f1_per_class = f1_score(labels, predictions, average=None)

        # Create metrics dictionary
        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

        # Add per-class F1 scores
        for i, f1_score_val in enumerate(f1_per_class):
            label_name = self.dataset_preparator.reverse_label_mapping.get(
                i, f"class_{i}"
            )
            metrics[f"f1_{label_name.lower()}"] = f1_score_val

        return metrics

    def _evaluate_model(
        self, trainer: Trainer, datasets: DatasetDict
    ) -> Tuple[Dict[str, EvaluationResult], Optional[ClassificationReport]]:
        """Evaluate the trained model on all datasets.

        Args:
            trainer: Trained trainer instance
            datasets: DatasetDict containing the data

        Returns:
            Tuple of (evaluation results, classification report)
        """
        results = {}
        classification_report_result = None

        # Evaluate on validation set
        if "validation" in datasets:
            logger.info("Evaluating on validation set")
            val_results = trainer.evaluate()
            results["validation"] = EvaluationResult(
                split_name="validation",
                eval_loss=val_results.get("eval_loss", 0.0),
                eval_accuracy=val_results.get("eval_accuracy", 0.0),
                eval_f1_macro=val_results.get("eval_f1_macro", 0.0),
                eval_f1_weighted=val_results.get("eval_f1_weighted", 0.0),
                eval_runtime=val_results.get("eval_runtime", 0.0),
                eval_samples_per_second=val_results.get("eval_samples_per_second", 0.0),
                eval_steps_per_second=val_results.get("eval_steps_per_second", 0.0),
            )

        # Evaluate on test set
        if "test" in datasets:
            logger.info("Evaluating on test set")
            test_results = trainer.evaluate(eval_dataset=datasets["test"])
            results["test"] = EvaluationResult(
                split_name="test",
                eval_loss=test_results.get("eval_loss", 0.0),
                eval_accuracy=test_results.get("eval_accuracy", 0.0),
                eval_f1_macro=test_results.get("eval_f1_macro", 0.0),
                eval_f1_weighted=test_results.get("eval_f1_weighted", 0.0),
                eval_runtime=test_results.get("eval_runtime", 0.0),
                eval_samples_per_second=test_results.get(
                    "eval_samples_per_second", 0.0
                ),
                eval_steps_per_second=test_results.get("eval_steps_per_second", 0.0),
            )

            # Generate detailed classification report for test set
            logger.info("Generating detailed classification report")
            test_predictions = trainer.predict(datasets["test"])
            predictions = np.argmax(test_predictions.predictions, axis=1)
            labels = test_predictions.label_ids

            # Generate classification report
            report = classification_report(
                labels,
                predictions,
                target_names=list(
                    self.dataset_preparator.reverse_label_mapping.values()
                ),
                output_dict=True,
            )

            # Convert to our schema
            classification_report_result = ClassificationReport(
                accuracy=report["accuracy"],
                macro_avg=report["macro avg"],
                weighted_avg=report["weighted avg"],
                per_class_metrics={
                    k: v
                    for k, v in report.items()
                    if k not in ["accuracy", "macro avg", "weighted avg"]
                },
            )

            # Log detailed metrics
            logger.info("Test Set Performance:")
            logger.info(f"  Accuracy: {report['accuracy']:.4f}")
            logger.info(f"  Macro F1: {report['macro avg']['f1-score']:.4f}")
            logger.info(f"  Weighted F1: {report['weighted avg']['f1-score']:.4f}")

            for label_name, metrics in report.items():
                if label_name not in ["accuracy", "macro avg", "weighted avg"]:
                    logger.info(
                        f"  {label_name}: F1={metrics['f1-score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}"
                    )

        return results, classification_report_result

    def _save_model_and_tokenizer(
        self, trainer: Trainer, tokenizer: AutoTokenizer, model_path: Path
    ) -> None:
        """Save the trained model and tokenizer.

        Args:
            trainer: Trained trainer instance
            tokenizer: Tokenizer to save
            model_path: Path to save the model
        """
        ensure_directory(model_path)

        # Save the best model
        trainer.save_model(str(model_path))
        tokenizer.save_pretrained(str(model_path))

        logger.info(f"Model and tokenizer saved to {model_path}")

    def _save_training_artifacts(self, output_dir: Path) -> None:
        """Save training artifacts (preprocessing config, label mapping).

        Args:
            output_dir: Directory to save artifacts
        """
        # Save preprocessing configuration
        preprocessing_path = output_dir / "preprocessing_config.json"
        self.dataset_preparator.save_preprocessing_config(preprocessing_path)

        # Save label mapping
        label_mapping_path = output_dir / "id2label.json"
        self.dataset_preparator.save_label_mapping(label_mapping_path)

        logger.info("Training artifacts saved")

    def _log_config_to_mlflow(self) -> None:
        """Log training configuration to MLflow."""
        config_dict = self.config.model_dump()
        mlflow.log_params(config_dict)

        # Log model backbone separately
        mlflow.log_param("model_backbone", self.config.backbone.value)

        # Log label mapping
        mlflow.log_dict(self.dataset_preparator.label_mapping, "label_mapping.json")

    def _log_metrics_to_mlflow(
        self, train_result: Any, eval_results: Dict[str, EvaluationResult]
    ) -> None:
        """Log training and evaluation metrics to MLflow.

        Args:
            train_result: Training results from trainer.train()
            eval_results: Evaluation results from _evaluate_model()
        """
        # Log training loss
        mlflow.log_metric("train_loss", train_result.training_loss)

        # Log evaluation metrics
        for split_name, results in eval_results.items():
            if split_name == "classification_report":
                continue

            # Log metrics from EvaluationResult
            mlflow.log_metric(f"{split_name}_eval_loss", results.eval_loss)
            mlflow.log_metric(f"{split_name}_eval_accuracy", results.eval_accuracy)
            mlflow.log_metric(f"{split_name}_eval_f1_macro", results.eval_f1_macro)
            mlflow.log_metric(
                f"{split_name}_eval_f1_weighted", results.eval_f1_weighted
            )

        # Log detailed test metrics if available
        if "classification_report" in eval_results:
            report = eval_results["classification_report"]

            # Log overall metrics
            mlflow.log_metric("test_accuracy", report.accuracy)
            mlflow.log_metric("test_f1_macro", report.macro_avg["f1-score"])
            mlflow.log_metric("test_f1_weighted", report.weighted_avg["f1-score"])

            # Log per-class metrics
            for label_name, metrics in report.per_class_metrics.items():
                mlflow.log_metric(f"test_f1_{label_name.lower()}", metrics["f1-score"])
                mlflow.log_metric(
                    f"test_precision_{label_name.lower()}", metrics["precision"]
                )
                mlflow.log_metric(
                    f"test_recall_{label_name.lower()}", metrics["recall"]
                )

    def get_best_checkpoint_path(self, output_dir: Path) -> Optional[Path]:
        """Get the path to the best checkpoint.

        Args:
            output_dir: Training output directory

        Returns:
            Path to the best checkpoint, or None if not found
        """
        checkpoint_dirs = [
            d for d in output_dir.iterdir() if d.name.startswith(PREFIX_CHECKPOINT_DIR)
        ]

        if not checkpoint_dirs:
            return None

        # Sort by checkpoint number and return the highest
        checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[-1]))
        return checkpoint_dirs[-1]
