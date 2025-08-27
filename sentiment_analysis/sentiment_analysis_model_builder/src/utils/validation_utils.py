# utils/validation_utils.py

"""Validation utility functions for sentiment analysis."""

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from ..core.enums import DataFormat, ExportFormat, ModelBackbone
from ..schemas.data_schemas import NewsArticle


def validate_file_path(
    file_path: Path,
    required: bool = True,
    allowed_extensions: Optional[list[str]] = None,
) -> bool:
    """Validate file path.

    Args:
        file_path: Path to validate
        required: Whether file is required to exist
        allowed_extensions: List of allowed file extensions

    Returns:
        True if path is valid
    """
    if not file_path:
        if required:
            logger.error("File path is required")
            return False
        return True

    # Check if file exists (if required)
    if required and not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False

    # Check file extension if specified
    if allowed_extensions and file_path.suffix not in allowed_extensions:
        logger.error(
            f"File extension {file_path.suffix} not allowed. Allowed: {allowed_extensions}"
        )
        return False

    return True


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        True if configuration is valid
    """
    if not config:
        logger.error("Configuration cannot be empty")
        return False

    required_keys = ["data", "preprocessing", "training", "export", "registry"]

    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration key: {key}")
            return False

    return True


def validate_model_path(model_path: Path) -> bool:
    """Validate model directory path.

    Args:
        model_path: Path to model directory

    Returns:
        True if model path is valid
    """
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return False

    if not model_path.is_dir():
        logger.error(f"Model path is not a directory: {model_path}")
        return False

    # Check for required model files
    required_files = ["config.json", "pytorch_model.bin"]
    for file_name in required_files:
        file_path = model_path / file_name
        if not file_path.exists():
            logger.error(f"Required model file not found: {file_path}")
            return False

    return True


def validate_data_format(data_format: str) -> bool:
    """Validate data format string.

    Args:
        data_format: Data format to validate

    Returns:
        True if format is valid
    """
    try:
        DataFormat(data_format)
        return True
    except ValueError:
        logger.error(
            f"Invalid data format: {data_format}. Allowed: {[f.value for f in DataFormat]}"
        )
        return False


def validate_model_backbone(backbone: str) -> bool:
    """Validate model backbone string.

    Args:
        backbone: Model backbone to validate

    Returns:
        True if backbone is valid
    """
    try:
        ModelBackbone(backbone)
        return True
    except ValueError:
        logger.error(
            f"Invalid model backbone: {backbone}. Allowed: {[f.value for f in ModelBackbone]}"
        )
        return False


def validate_export_format(export_format: str) -> bool:
    """Validate export format string.

    Args:
        export_format: Export format to validate

    Returns:
        True if format is valid
    """
    try:
        ExportFormat(export_format)
        return True
    except ValueError:
        logger.error(
            f"Invalid export format: {export_format}. Allowed: {[f.value for f in ExportFormat]}"
        )
        return False


def validate_news_article(article: NewsArticle) -> bool:
    """Validate news article.

    Args:
        article: News article to validate

    Returns:
        True if article is valid
    """
    try:
        # Validate text content
        if not article.text or len(article.text.strip()) < 10:
            logger.error(f"Article text must be at least 10 characters: {article.id}")
            return False

        # Validate label
        if not article.label:
            logger.error(f"Article must have a label: {article.id}")
            return False

        # Validate tickers if present
        if article.tickers:
            for ticker in article.tickers:
                if not ticker or len(ticker.strip()) < 2:
                    logger.error(f"Invalid ticker in article {article.id}: {ticker}")
                    return False

        return True

    except Exception as e:
        logger.error(f"Failed to validate article {article.id}: {e}")
        return False


def validate_label_mapping(label_mapping: Dict[str, int]) -> bool:
    """Validate label mapping.

    Args:
        label_mapping: Label mapping to validate

    Returns:
        True if mapping is valid
    """
    if not label_mapping:
        logger.error("Label mapping cannot be empty")
        return False

    # Check for duplicate values
    values = list(label_mapping.values())
    if len(values) != len(set(values)):
        logger.error("Label mapping contains duplicate values")
        return False

    # Check for non-negative integer values
    for label, value in label_mapping.items():
        if not isinstance(value, int) or value < 0:
            logger.error(f"Label value must be non-negative integer: {label}={value}")
            return False

    return True


def validate_training_config(training_config: Dict[str, Any]) -> bool:
    """Validate training configuration.

    Args:
        training_config: Training configuration to validate

    Returns:
        True if configuration is valid
    """
    required_keys = ["backbone", "batch_size", "learning_rate", "num_epochs"]

    for key in required_keys:
        if key not in training_config:
            logger.error(f"Missing required training configuration key: {key}")
            return False

    # Validate specific values
    if training_config.get("batch_size", 0) < 1:
        logger.error("Batch size must be at least 1")
        return False

    if training_config.get("learning_rate", 0) <= 0:
        logger.error("Learning rate must be positive")
        return False

    if training_config.get("num_epochs", 0) < 1:
        logger.error("Number of epochs must be at least 1")
        return False

    return True


def validate_export_config(export_config: Dict[str, Any]) -> bool:
    """Validate export configuration.

    Args:
        export_config: Export configuration to validate

    Returns:
        True if configuration is valid
    """
    if "format" not in export_config:
        logger.error("Export format is required")
        return False

    if not validate_export_format(export_config["format"]):
        return False

    # Validate ONNX opset version if exporting to ONNX
    if export_config["format"] in ["onnx", "both"]:
        opset_version = export_config.get("onnx_opset_version", 17)
        if not (11 <= opset_version <= 18):
            logger.error(
                f"ONNX opset version must be between 11 and 18, got {opset_version}"
            )
            return False

    return True
