# utils/model_evaluator.py

"""Model evaluation utilities for sentiment analysis models."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..core.config import PreprocessingConfig, TrainingConfig
from ..core.enums import ModelBackbone, SentimentLabel
from ..data.data_loader import DataLoader, NewsArticle
from ..data.dataset import DatasetPreparator
from ..schemas.data_schemas import TrainingExample
from ..schemas.training_schemas import ClassificationReport, EvaluationResult
from ..utils.file_utils import ensure_directory, load_json, save_json

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"


class ModelEvaluationMetrics:
    """Container for model evaluation metrics."""

    def __init__(
        self,
        accuracy: float,
        f1_macro: float,
        f1_weighted: float,
        precision_macro: float,
        recall_macro: float,
        per_class_metrics: Dict[str, Dict[str, float]],
        confusion_matrix: np.ndarray,
        runtime_seconds: float,
        samples_per_second: float,
    ):
        """Initialize evaluation metrics.

        Args:
            accuracy: Overall accuracy
            f1_macro: Macro-averaged F1 score
            f1_weighted: Weighted-averaged F1 score
            precision_macro: Macro-averaged precision
            recall_macro: Macro-averaged recall
            per_class_metrics: Per-class metrics dictionary
            confusion_matrix: Confusion matrix
            runtime_seconds: Evaluation runtime in seconds
            samples_per_second: Processing speed
        """
        self.accuracy = accuracy
        self.f1_macro = f1_macro
        self.f1_weighted = f1_weighted
        self.precision_macro = precision_macro
        self.recall_macro = recall_macro
        self.per_class_metrics = per_class_metrics
        self.confusion_matrix = confusion_matrix
        self.runtime_seconds = runtime_seconds
        self.samples_per_second = samples_per_second

    def to_dict(self) -> Dict[str, any]:
        """Convert metrics to dictionary format."""
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "runtime_seconds": self.runtime_seconds,
            "samples_per_second": self.samples_per_second,
        }


class ModelEvaluator:
    """Evaluates sentiment analysis models on given datasets."""

    def __init__(
        self,
        preprocessing_config: PreprocessingConfig,
        training_config: Optional[TrainingConfig] = None,
    ):
        """Initialize the model evaluator.

        Args:
            preprocessing_config: Configuration for text preprocessing
            training_config: Optional training configuration (used for tokenization params)
        """
        self.preprocessing_config = preprocessing_config
        self.training_config = training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_mapping = preprocessing_config.label_mapping
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

        logger.info(f"ModelEvaluator initialized on device: {self.device}")

    def evaluate_pretrained_model(
        self,
        model_name_or_path: str,
        articles: List[NewsArticle],
        output_dir: Optional[Path] = None,
    ) -> ModelEvaluationMetrics:
        """Evaluate a pretrained model on the given dataset.

        Args:
            model_name_or_path: Hugging Face model name or local path to pretrained model
            articles: List of news articles for evaluation
            output_dir: Optional output directory to save results

        Returns:
            ModelEvaluationMetrics object containing evaluation results

        Raises:
            ValueError: If model loading or evaluation fails
        """
        logger.info(f"Evaluating pretrained model: {model_name_or_path}")

        try:
            # Load model and tokenizer
            model, tokenizer = self._load_pretrained_model(model_name_or_path)

            # Prepare dataset for evaluation
            dataset = self._prepare_evaluation_dataset(articles, tokenizer)

            # Run evaluation
            metrics = self._evaluate_dataset(model, tokenizer, dataset, articles)

            # Save results if output directory provided
            if output_dir:
                self._save_evaluation_results(
                    metrics, output_dir, model_name_or_path, is_pretrained=True
                )

            logger.info("Pretrained model evaluation completed successfully")
            return metrics

        except Exception as e:
            logger.error(f"Failed to evaluate pretrained model: {e}")
            raise ValueError(f"Model evaluation failed: {e}")

    def evaluate_finetuned_model(
        self,
        model_path: Path,
        articles: List[NewsArticle],
        output_dir: Optional[Path] = None,
    ) -> ModelEvaluationMetrics:
        """Evaluate a fine-tuned model on the given dataset.

        Args:
            model_path: Path to the fine-tuned model directory
            articles: List of news articles for evaluation
            output_dir: Optional output directory to save results

        Returns:
            ModelEvaluationMetrics object containing evaluation results

        Raises:
            ValueError: If model loading or evaluation fails
        """
        logger.info(f"Evaluating fine-tuned model: {model_path}")

        try:
            # Validate model path
            if not model_path.exists():
                raise ValueError(f"Model path does not exist: {model_path}")

            if not (model_path / "config.json").exists():
                raise ValueError(f"Invalid model directory: {model_path}")

            # Load model and tokenizer
            model, tokenizer = self._load_finetuned_model(model_path)

            # Load preprocessing config if available
            self._load_model_preprocessing_config(model_path)

            # Prepare dataset for evaluation
            dataset = self._prepare_evaluation_dataset(articles, tokenizer)

            # Run evaluation
            metrics = self._evaluate_dataset(model, tokenizer, dataset, articles)

            # Save results if output directory provided
            if output_dir:
                self._save_evaluation_results(
                    metrics, output_dir, str(model_path), is_pretrained=False
                )

            logger.info("Fine-tuned model evaluation completed successfully")
            return metrics

        except Exception as e:
            logger.error(f"Failed to evaluate fine-tuned model: {e}")
            raise ValueError(f"Model evaluation failed: {e}")

    def _load_pretrained_model(
        self, model_name_or_path: str
    ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Load a pretrained model from Hugging Face Hub or local path.

        Args:
            model_name_or_path: Model name or path

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading pretrained model: {model_name_or_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Load model with proper number of labels
        num_labels = len(self.label_mapping)

        # For pretrained models, we need to handle potential label mismatch
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, num_labels=num_labels
            )
        except Exception as e:
            # If loading fails due to label mismatch, try loading with ignore_mismatched_sizes
            logger.warning(
                f"Model loading failed, trying with ignore_mismatched_sizes: {e}"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, num_labels=num_labels, ignore_mismatched_sizes=True
            )

        model.eval()
        model = model.to(self.device)

        logger.info(f"Model loaded successfully with {num_labels} labels")
        return model, tokenizer

    def _load_finetuned_model(
        self, model_path: Path
    ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Load a fine-tuned model from local directory.

        Args:
            model_path: Path to model directory

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading fine-tuned model from: {model_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model = model.to(self.device)

        logger.info("Fine-tuned model loaded successfully")
        return model, tokenizer

    def _load_model_preprocessing_config(self, model_path: Path) -> None:
        """Load preprocessing configuration from model directory if available.

        Args:
            model_path: Path to model directory
        """
        config_path = model_path / "preprocessing_config.json"
        if config_path.exists():
            try:
                config_dict = load_json(config_path)
                # Update label mapping if available
                if "label_mapping" in config_dict:
                    self.label_mapping = config_dict["label_mapping"]
                    self.reverse_label_mapping = {
                        v: k for k, v in self.label_mapping.items()
                    }
                    logger.info("Loaded preprocessing configuration from model")
            except Exception as e:
                logger.warning(f"Failed to load preprocessing config: {e}")

    def _prepare_evaluation_dataset(
        self, articles: List[NewsArticle], tokenizer: AutoTokenizer
    ) -> Dataset:
        """Prepare dataset for evaluation.

        Args:
            articles: List of news articles
            tokenizer: Model tokenizer

        Returns:
            Hugging Face Dataset ready for evaluation
        """
        logger.info(f"Preparing evaluation dataset with {len(articles)} articles")

        # Convert articles to training examples
        examples = []
        for article in articles:
            if article.label is None:
                logger.warning(f"Article {article.id} has no sentiment label, skipping")
                continue

            # Map sentiment to integer label
            label = self.label_mapping.get(article.label.value)
            if label is None:
                logger.warning(
                    f"Unknown sentiment label: {article.label.value}, skipping"
                )
                continue

            examples.append(
                TrainingExample(
                    text=article.text,
                    label=label,
                    label_text=article.label.value,
                    id=article.id or "",
                    title=article.title or "",
                    source=article.source or "",
                    published_at=(
                        str(article.published_at) if article.published_at else ""
                    ),
                    tickers=article.tickers or [],
                )
            )

        logger.info(f"Created {len(examples)} training examples")

        # Convert to dataset
        dataset_dict = [example.model_dump() for example in examples]
        dataset = Dataset.from_list(dataset_dict)

        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.preprocessing_config.max_length,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[
                "text",
                "id",
                "title",
                "source",
                "published_at",
                "tickers",
                "label_text",
            ],
        )

        # Rename label column for Hugging Face compatibility
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

        logger.info("Dataset tokenization completed")
        return tokenized_dataset

    def _evaluate_dataset(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        original_articles: List[NewsArticle],
    ) -> ModelEvaluationMetrics:
        """Evaluate model on the prepared dataset.

        Args:
            model: Model to evaluate
            tokenizer: Model tokenizer
            dataset: Tokenized dataset
            original_articles: Original articles for reference

        Returns:
            ModelEvaluationMetrics object
        """
        logger.info("Starting model evaluation")

        # Create minimal training arguments for evaluation
        training_args = TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=(
                self.training_config.eval_batch_size if self.training_config else 32
            ),
            dataloader_drop_last=False,
            eval_accumulation_steps=1,
            report_to="none",  # Disable logging to wandb, tensorboard, etc.
            logging_strategy="no",  # Disable logging
        )

        # Create trainer for evaluation
        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
        )

        # Start timing
        start_time = time.time()

        # Run evaluation
        logger.info("Running model prediction...")
        predictions = trainer.predict(dataset)

        # Calculate runtime
        runtime_seconds = time.time() - start_time
        samples_per_second = len(dataset) / runtime_seconds

        # Extract predictions and labels
        predicted_logits = predictions.predictions
        true_labels = predictions.label_ids

        # Convert logits to predicted labels
        predicted_labels = np.argmax(predicted_logits, axis=1)

        # Calculate metrics
        metrics = self._calculate_metrics(
            true_labels, predicted_labels, runtime_seconds, samples_per_second
        )

        logger.info(f"Evaluation completed in {runtime_seconds:.2f} seconds")
        logger.info(f"Processing speed: {samples_per_second:.2f} samples/second")

        return metrics

    def _calculate_metrics(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        runtime_seconds: float,
        samples_per_second: float,
    ) -> ModelEvaluationMetrics:
        """Calculate comprehensive evaluation metrics.

        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted labels
            runtime_seconds: Evaluation runtime
            samples_per_second: Processing speed

        Returns:
            ModelEvaluationMetrics object
        """
        # Basic metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1_macro = f1_score(true_labels, predicted_labels, average="macro")
        f1_weighted = f1_score(true_labels, predicted_labels, average="weighted")
        precision_macro = precision_score(
            true_labels, predicted_labels, average="macro"
        )
        recall_macro = recall_score(true_labels, predicted_labels, average="macro")

        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Per-class metrics
        label_names = list(self.reverse_label_mapping.values())
        class_report = classification_report(
            true_labels,
            predicted_labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        # Extract per-class metrics
        per_class_metrics = {}
        for label_name in label_names:
            if label_name in class_report:
                per_class_metrics[label_name] = {
                    "precision": class_report[label_name]["precision"],
                    "recall": class_report[label_name]["recall"],
                    "f1-score": class_report[label_name]["f1-score"],
                    "support": class_report[label_name]["support"],
                }

        # Log key metrics
        logger.info("Evaluation Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Macro: {f1_macro:.4f}")
        logger.info(f"  F1 Weighted: {f1_weighted:.4f}")
        logger.info(f"  Precision Macro: {precision_macro:.4f}")
        logger.info(f"  Recall Macro: {recall_macro:.4f}")

        return ModelEvaluationMetrics(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            per_class_metrics=per_class_metrics,
            confusion_matrix=cm,
            runtime_seconds=runtime_seconds,
            samples_per_second=samples_per_second,
        )

    def _save_evaluation_results(
        self,
        metrics: ModelEvaluationMetrics,
        output_dir: Path,
        model_name: str,
        is_pretrained: bool,
    ) -> None:
        """Save evaluation results to output directory.

        Args:
            metrics: Evaluation metrics
            output_dir: Output directory
            model_name: Model name or path
            is_pretrained: Whether the model is pretrained or fine-tuned
        """
        ensure_directory(output_dir)

        # Create results dictionary
        results = {
            "model_info": {
                "name": model_name,
                "type": "pretrained" if is_pretrained else "fine-tuned",
                "device": str(self.device),
                "label_mapping": self.label_mapping,
            },
            "metrics": metrics.to_dict(),
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save detailed results
        results_path = output_dir / "evaluation_results.json"
        save_json(results, results_path)

        # Save confusion matrix separately
        cm_path = output_dir / "confusion_matrix.json"
        save_json(
            {
                "confusion_matrix": metrics.confusion_matrix.tolist(),
                "labels": list(self.reverse_label_mapping.values()),
            },
            cm_path,
        )

        # Save summary metrics
        summary_path = output_dir / "evaluation_summary.json"
        summary = {
            "accuracy": metrics.accuracy,
            "f1_macro": metrics.f1_macro,
            "f1_weighted": metrics.f1_weighted,
            "precision_macro": metrics.precision_macro,
            "recall_macro": metrics.recall_macro,
            "runtime_seconds": metrics.runtime_seconds,
            "samples_per_second": metrics.samples_per_second,
        }
        save_json(summary, summary_path)

        logger.info(f"Evaluation results saved to: {output_dir}")
        logger.info(f"  Detailed results: {results_path}")
        logger.info(f"  Confusion matrix: {cm_path}")
        logger.info(f"  Summary: {summary_path}")
