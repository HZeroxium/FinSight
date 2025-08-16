"""Crypto News Sentiment Analysis Model Builder.

A production-ready Python project for training and packaging sentiment analysis models
for cryptocurrency news using Hugging Face Transformers and MLflow.
"""

__version__ = "0.1.0"
__author__ = "FinSight Team"
__email__ = "team@finsight.ai"

from .core.config import Config
from .data.data_loader import DataLoader, NewsArticle
from .data.dataset import DatasetPreparator
from .models.trainer import SentimentTrainer
from .models.exporter import ModelExporter
from .registry.mlflow_registry import MLflowRegistry

__all__ = [
    "Config",
    "DataLoader",
    "NewsArticle",
    "DatasetPreparator",
    "SentimentTrainer",
    "ModelExporter",
    "MLflowRegistry",
]
