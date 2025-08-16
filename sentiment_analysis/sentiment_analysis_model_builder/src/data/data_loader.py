"""Data loading and preprocessing for sentiment analysis training."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from ..core.config import DataConfig, PreprocessingConfig
from ..core.enums import DataFormat, SentimentLabel
from ..schemas.data_schemas import NewsArticle
from ..utils.text_utils import clean_text, normalize_text, validate_text_length


class DataLoader:
    """Handles loading and preprocessing of news data for sentiment analysis."""

    def __init__(
        self, data_config: DataConfig, preprocessing_config: PreprocessingConfig
    ):
        """Initialize the data loader.

        Args:
            data_config: Configuration for data loading
            preprocessing_config: Configuration for text preprocessing
        """
        self.data_config = data_config
        self.preprocessing_config = preprocessing_config
        self.label_mapping = preprocessing_config.label_mapping

        # Validate label mapping
        self._validate_label_mapping()

    def _validate_label_mapping(self) -> None:
        """Validate that the label mapping is properly configured."""
        if not self.label_mapping:
            raise ValueError("Label mapping cannot be empty")

        # Check for duplicate values
        if len(self.label_mapping.values()) != len(set(self.label_mapping.values())):
            raise ValueError("Label mapping contains duplicate values")

        # Check for non-negative integer values
        for label, value in self.label_mapping.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Label value must be non-negative integer: {label}={value}"
                )

    def load_data(self) -> List[NewsArticle]:
        """Load data from the configured input file.

        Returns:
            List of processed news articles

        Raises:
            ValueError: If the input format is not supported or data is invalid
        """
        input_path = self.data_config.input_path
        input_format = self.data_config.input_format

        if input_path is None:
            raise ValueError("Input path is not configured")

        logger.info(f"Loading data from {input_path} (format: {input_format})")

        if input_format == DataFormat.JSON:
            articles = self._load_json_data(input_path)
        elif input_format == DataFormat.JSONL:
            articles = self._load_jsonl_data(input_path)
        elif input_format == DataFormat.CSV:
            articles = self._load_csv_data(input_path)
        elif input_format == DataFormat.PARQUET:
            articles = self._load_parquet_data(input_path)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        # Apply preprocessing
        articles = self._preprocess_articles(articles)

        # Limit samples if configured
        if (
            self.data_config.max_samples
            and len(articles) > self.data_config.max_samples
        ):
            logger.info(f"Limiting to {self.data_config.max_samples} samples")
            articles = articles[: self.data_config.max_samples]

        logger.info(f"Loaded {len(articles)} articles")
        return articles

    def _load_json_data(self, file_path: Path) -> List[NewsArticle]:
        """Load data from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of news articles
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                articles = []
                for item in data:
                    try:
                        article = self._create_article_from_dict(item)
                        articles.append(article)
                    except Exception as e:
                        logger.warning(f"Failed to parse article: {e}")
                        continue
                return articles
            else:
                raise ValueError("JSON file must contain a list of articles")

        except Exception as e:
            logger.error(f"Failed to load JSON data from {file_path}: {e}")
            raise

    def _load_jsonl_data(self, file_path: Path) -> List[NewsArticle]:
        """Load data from JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of news articles
        """
        articles = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                        article = self._create_article_from_dict(item)
                        articles.append(article)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Failed to create article from line {line_num}: {e}"
                        )
                        continue

            return articles

        except Exception as e:
            logger.error(f"Failed to load JSONL data from {file_path}: {e}")
            raise

    def _load_csv_data(self, file_path: Path) -> List[NewsArticle]:
        """Load data from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            List of news articles
        """
        try:
            df = pd.read_csv(file_path)
            articles = []

            for _, row in df.iterrows():
                try:
                    article = self._create_article_from_dict(row.to_dict())
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse CSV row: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Failed to load CSV data from {file_path}: {e}")
            raise

    def _load_parquet_data(self, file_path: Path) -> List[NewsArticle]:
        """Load data from Parquet file.

        Args:
            file_path: Path to Parquet file

        Returns:
            List of news articles
        """
        try:
            df = pd.read_parquet(file_path)
            articles = []

            for _, row in df.iterrows():
                try:
                    article = self._create_article_from_dict(row.to_dict())
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse Parquet row: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Failed to load Parquet data from {file_path}: {e}")
            raise

    def _create_article_from_dict(self, item: Dict[str, Any]) -> NewsArticle:
        """Create NewsArticle from dictionary.

        Args:
            item: Dictionary containing article data

        Returns:
            NewsArticle object
        """
        # Extract text and label (required fields)
        text = item.get(self.data_config.text_column, "")
        label_str = item.get(self.data_config.label_column, "")

        if not text or not label_str:
            raise ValueError("Text and label are required fields")

        # Convert label string to enum
        try:
            label = SentimentLabel(label_str.upper())
        except ValueError:
            raise ValueError(f"Invalid label: {label_str}")

        # Extract optional fields
        article_id = (
            item.get(self.data_config.id_column) if self.data_config.id_column else None
        )
        title = (
            item.get(self.data_config.title_column)
            if self.data_config.title_column
            else None
        )
        published_at = (
            item.get(self.data_config.published_at_column)
            if self.data_config.published_at_column
            else None
        )
        tickers = (
            item.get(self.data_config.tickers_column)
            if self.data_config.tickers_column
            else None
        )
        split = (
            item.get(self.data_config.split_column)
            if self.data_config.split_column
            else None
        )
        source = item.get("source", None)
        url = item.get("url", None)

        # Parse published_at if it's a string
        if published_at and isinstance(published_at, str):
            try:
                from datetime import datetime

                published_at = datetime.fromisoformat(
                    published_at.replace("Z", "+00:00")
                )
            except ValueError:
                logger.warning(f"Could not parse published_at: {published_at}")
                published_at = None

        # Parse tickers if it's a string
        if tickers and isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]

        # Parse split if it's a string
        if split and isinstance(split, str):
            try:
                from ..core.enums import DataSplit

                split = DataSplit(split.lower())
            except ValueError:
                logger.warning(f"Invalid split value: {split}")
                split = None

        return NewsArticle(
            id=article_id,
            text=text,
            label=label,
            title=title,
            published_at=published_at,
            tickers=tickers,
            split=split,
            source=source,
            url=url,
        )

    def _preprocess_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Apply text preprocessing to articles.

        Args:
            articles: List of articles to preprocess

        Returns:
            List of preprocessed articles
        """
        preprocessed_articles = []

        for article in articles:
            try:
                # Clean text
                cleaned_text = clean_text(
                    article.text,
                    remove_html=self.preprocessing_config.remove_html,
                    remove_urls=self.preprocessing_config.remove_urls,
                    remove_emails=self.preprocessing_config.remove_emails,
                )

                # Normalize text
                normalized_text = normalize_text(
                    cleaned_text,
                    lowercase=self.preprocessing_config.lowercase,
                    normalize_unicode=self.preprocessing_config.normalize_unicode,
                )

                # Validate text length
                if not validate_text_length(
                    normalized_text,
                    min_length=self.preprocessing_config.min_length,
                    max_length=self.preprocessing_config.max_length,
                ):
                    logger.warning(
                        f"Article {article.id} text length out of range, skipping"
                    )
                    continue

                # Create new article with preprocessed text
                preprocessed_article = NewsArticle(
                    id=article.id,
                    text=normalized_text,
                    label=article.label,
                    title=article.title,
                    published_at=article.published_at,
                    tickers=article.tickers,
                    split=article.split,
                    source=article.source,
                    url=article.url,
                )

                preprocessed_articles.append(preprocessed_article)

            except Exception as e:
                logger.warning(f"Failed to preprocess article {article.id}: {e}")
                continue

        logger.info(f"Preprocessed {len(preprocessed_articles)} articles")
        return preprocessed_articles

    def validate_data(self, articles: List[NewsArticle]) -> bool:
        """Validate the loaded data.

        Args:
            articles: List of articles to validate

        Returns:
            True if data is valid
        """
        if not articles:
            logger.error("No articles loaded")
            return False

        # Check for required fields
        for article in articles:
            if not article.text or not article.label:
                logger.error(f"Article {article.id} missing required fields")
                return False

        # Check label distribution
        label_counts = {}
        for article in articles:
            label = article.label.value
            label_counts[label] = label_counts.get(label, 0) + 1

        logger.info(f"Label distribution: {label_counts}")

        # Check for class imbalance
        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        if max_count > min_count * 10:
            logger.warning("Severe class imbalance detected")

        return True

    def get_data_statistics(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Get statistics about the loaded data.

        Args:
            articles: List of articles

        Returns:
            Dictionary with data statistics
        """
        if not articles:
            return {}

        # Text length statistics
        text_lengths = [len(article.text) for article in articles]

        # Label distribution
        label_counts = {}
        for article in articles:
            label = article.label.value
            label_counts[label] = label_counts.get(label, 0) + 1

        # Source distribution
        source_counts = {}
        for article in articles:
            source = article.source or "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "total_articles": len(articles),
            "text_length": {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "mean": sum(text_lengths) / len(text_lengths),
                "median": sorted(text_lengths)[len(text_lengths) // 2],
            },
            "label_distribution": label_counts,
            "source_distribution": source_counts,
        }
