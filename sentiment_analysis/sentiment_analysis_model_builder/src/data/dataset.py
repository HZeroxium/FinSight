"""Dataset preparation for Hugging Face training."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .data_loader import NewsArticle
from ..core.config import PreprocessingConfig, TrainingConfig
from ..core.enums import DataSplit
from ..schemas.data_schemas import TrainingExample, DatasetStats, TokenizerConfig
from ..utils.file_utils import save_json, ensure_directory


class DatasetPreparator:
    """Prepares news articles for Hugging Face training."""

    def __init__(
        self, preprocessing_config: PreprocessingConfig, training_config: TrainingConfig
    ):
        """Initialize the dataset preparator.

        Args:
            preprocessing_config: Configuration for text preprocessing
            training_config: Configuration for training and data splitting
        """
        self.preprocessing_config = preprocessing_config
        self.training_config = training_config
        self.label_mapping = preprocessing_config.label_mapping
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

        # Set random seed for reproducibility
        np.random.seed(training_config.random_seed)

    def prepare_datasets(self, articles: List[NewsArticle]) -> DatasetDict:
        """Prepare Hugging Face datasets from news articles.

        Args:
            articles: List of processed news articles

        Returns:
            DatasetDict containing train, validation, and test datasets

        Raises:
            ValueError: If data preparation fails
        """
        logger.info("Preparing datasets for training")

        # Convert articles to training format
        training_data = self._convert_articles_to_training_format(articles)

        # Split data into train/val/test
        train_data, val_data, test_data = self._split_data(training_data)

        # Create Hugging Face datasets
        datasets = DatasetDict(
            {
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data),
                "test": Dataset.from_list(test_data),
            }
        )

        # Log dataset statistics
        self._log_dataset_stats(datasets)

        return datasets

    def _convert_articles_to_training_format(
        self, articles: List[NewsArticle]
    ) -> List[TrainingExample]:
        """Convert news articles to the format expected by Hugging Face training.

        Args:
            articles: List of processed news articles

        Returns:
            List of training examples
        """
        training_data = []

        for article in articles:
            try:
                # Convert label string to integer
                if article.label.value not in self.label_mapping:
                    logger.warning(
                        f"Unknown label '{article.label.value}' for article {article.id}"
                    )
                    continue

                label_id = self.label_mapping[article.label.value]

                # Create training example
                example = TrainingExample(
                    text=article.text,
                    label=label_id,
                    label_text=article.label.value,
                    id=article.id or "",
                    title=article.title or "",
                    source=article.source or "",
                    published_at=(
                        str(article.published_at) if article.published_at else ""
                    ),
                    tickers=article.tickers or [],
                )

                training_data.append(example)

            except Exception as e:
                logger.warning(f"Failed to convert article {article.id}: {e}")
                continue

        logger.info(f"Converted {len(training_data)} articles to training format")
        return training_data

    def _split_data(
        self, training_data: List[TrainingExample]
    ) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
        """Split data into train/validation/test sets.

        Args:
            training_data: List of training examples

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Check if we have predefined splits
        if self._has_predefined_splits(training_data):
            logger.info("Using predefined data splits")
            return self._extract_predefined_splits(training_data)

        # Perform stratified split
        logger.info("Performing stratified train/validation/test split")

        # First split: separate test set
        train_val_data, test_data = train_test_split(
            training_data,
            test_size=self.training_config.test_split,
            random_state=self.training_config.random_seed,
            stratify=[item.label for item in training_data],
        )

        # Second split: separate validation set from training
        val_ratio = self.training_config.val_split / (
            self.training_config.train_split + self.training_config.val_split
        )
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio,
            random_state=self.training_config.random_seed,
            stratify=[item.label for item in train_val_data],
        )

        return train_data, val_data, test_data

    def _has_predefined_splits(self, training_data: List[TrainingExample]) -> bool:
        """Check if the data has predefined split assignments.

        Args:
            training_data: List of training examples

        Returns:
            True if predefined splits are available
        """
        # Check if any examples have a 'split' field
        return any(hasattr(item, "split") and item.split for item in training_data)

    def _extract_predefined_splits(
        self, training_data: List[TrainingExample]
    ) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
        """Extract predefined data splits.

        Args:
            training_data: List of training examples with split assignments

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        train_data = []
        val_data = []
        test_data = []

        for item in training_data:
            split = getattr(item, "split", "").lower() if hasattr(item, "split") else ""

            if split in ["train", "training"]:
                train_data.append(item)
            elif split in ["val", "validation", "dev", "development"]:
                val_data.append(item)
            elif split in ["test", "testing"]:
                test_data.append(item)
            else:
                # Default to training if split is unclear
                logger.warning(
                    f"Unknown split '{split}' for item {getattr(item, 'id', 'unknown')}, defaulting to train"
                )
                train_data.append(item)

        # Log split distribution
        logger.info(
            f"Predefined splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )

        return train_data, val_data, test_data

    def _log_dataset_stats(self, datasets: DatasetDict) -> None:
        """Log statistics about the prepared datasets.

        Args:
            datasets: DatasetDict containing the prepared datasets
        """
        logger.info("Dataset statistics:")

        for split_name, dataset in datasets.items():
            # Count samples per class
            labels = dataset["label"]
            unique_labels, counts = np.unique(labels, return_counts=True)

            logger.info(f"  {split_name}:")
            logger.info(f"    Total samples: {len(dataset)}")

            for label_id, count in zip(unique_labels, counts):
                label_text = self.reverse_label_mapping.get(
                    label_id, f"Unknown({label_id})"
                )
                logger.info(f"    {label_text}: {count}")

            # Text length statistics
            text_lengths = [len(text) for text in dataset["text"]]
            avg_length = np.mean(text_lengths)
            std_length = np.std(text_lengths)
            logger.info(
                f"    Avg text length: {avg_length:.1f} ± {std_length:.1f} characters"
            )

    def compute_class_weights(self, train_dataset: Dataset) -> np.ndarray:
        """Compute class weights for handling class imbalance.

        Args:
            train_dataset: Training dataset

        Returns:
            Array of class weights
        """
        labels = train_dataset["label"]

        # Compute class weights using sklearn
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )

        # Create mapping from label ID to weight
        label_to_weight = {
            label_id: weight
            for label_id, weight in zip(np.unique(labels), class_weights)
        }

        # Convert to array in the same order as label_mapping
        weights_array = np.array(
            [
                label_to_weight[label_id]
                for label_id in sorted(self.label_mapping.values())
            ]
        )

        logger.info("Class weights computed:")
        for label_id, weight in enumerate(weights_array):
            label_text = self.reverse_label_mapping.get(
                label_id, f"Unknown({label_id})"
            )
            logger.info(f"  {label_text}: {weight:.3f}")

        return weights_array

    def save_preprocessing_config(self, output_path: Path) -> None:
        """Save preprocessing configuration for model serving.

        Args:
            output_path: Path to save the configuration file
        """
        config = {
            "max_length": self.preprocessing_config.max_length,
            "min_length": self.preprocessing_config.min_length,
            "remove_html": self.preprocessing_config.remove_html,
            "normalize_unicode": self.preprocessing_config.normalize_unicode,
            "lowercase": self.preprocessing_config.lowercase,
            "remove_urls": self.preprocessing_config.remove_urls,
            "remove_emails": self.preprocessing_config.remove_emails,
            "label_mapping": self.label_mapping,
            "reverse_label_mapping": self.reverse_label_mapping,
        }

        save_json(config, output_path)

    def save_label_mapping(self, output_path: Path) -> None:
        """Save label mapping for model serving.

        Args:
            output_path: Path to save the label mapping file
        """
        save_json(self.reverse_label_mapping, output_path)

    def get_tokenizer_config(self) -> TokenizerConfig:
        """Get configuration for tokenizer setup.

        Returns:
            TokenizerConfig object
        """
        return TokenizerConfig(
            max_length=self.preprocessing_config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
