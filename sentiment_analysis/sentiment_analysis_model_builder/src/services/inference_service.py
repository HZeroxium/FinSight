# services/inference_service.py

"""Inference service for sentiment analysis using trained FinBERT model."""

import time
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
import numpy as np
from loguru import logger

from ..core.config import APIConfig
from ..core.enums import SentimentLabel
from ..schemas.api_schemas import SentimentResult, SentimentScore
from ..utils.text_utils import clean_text, normalize_text, validate_text_length
from ..utils.file_utils import load_json


class InferenceError(Exception):
    """Custom exception for inference errors."""

    pass


class SentimentInferenceService:
    """Service for sentiment analysis inference using trained FinBERT model."""

    def __init__(self, config: APIConfig):
        """Initialize the inference service.

        Args:
            config: API configuration containing model paths and settings
        """
        self.config = config
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.pipeline = None
        self.device: str = "cpu"
        self.label_mapping: Dict[int, str] = {}
        self.preprocessing_config: Dict[str, Any] = {}
        self.model_info: Dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)

        logger.info("SentimentInferenceService initialized")

    async def initialize(self) -> None:
        """Initialize the model and tokenizer asynchronously."""
        try:
            logger.info("Loading sentiment analysis model...")

            # Load configurations
            await self._load_configurations()

            # Set device
            self.device = self._get_device()
            logger.info(f"Using device: {self.device}")

            # Load model and tokenizer in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._load_model_sync)

            # Create pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True,
                truncation=True,
                max_length=self.config.max_text_length,
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize inference service: {e}")
            raise InferenceError(f"Model initialization failed: {e}")

    async def _load_configurations(self) -> None:
        """Load preprocessing configuration and label mapping."""
        try:
            # Load preprocessing config
            if self.config.preprocessing_config_path.exists():
                self.preprocessing_config = load_json(
                    self.config.preprocessing_config_path
                )
                logger.info("Preprocessing configuration loaded")
            else:
                logger.warning("Preprocessing config not found, using defaults")
                self.preprocessing_config = {
                    "remove_html": True,
                    "normalize_unicode": True,
                    "lowercase": True,
                    "remove_urls": True,
                    "remove_emails": True,
                    "max_length": 512,
                    "min_length": 10,
                }

            # Load label mapping
            if self.config.label_mapping_path.exists():
                id2label = load_json(self.config.label_mapping_path)
                self.label_mapping = {int(k): v for k, v in id2label.items()}
                logger.info(f"Label mapping loaded: {self.label_mapping}")
            else:
                # Default mapping
                self.label_mapping = {
                    0: SentimentLabel.NEGATIVE.value,
                    1: SentimentLabel.NEUTRAL.value,
                    2: SentimentLabel.POSITIVE.value,
                }
                logger.warning("Label mapping not found, using defaults")

        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise InferenceError(f"Configuration loading failed: {e}")

    def _get_device(self) -> str:
        """Determine the device to use for inference."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return self.config.device

    def _load_model_sync(self) -> None:
        """Load model and tokenizer synchronously (runs in thread pool)."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_path
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Store model info
            self.model_info = {
                "model_name": "FinBERT-Sentiment",
                "model_path": str(self.config.model_path),
                "num_labels": len(self.label_mapping),
                "max_sequence_length": self.config.max_text_length,
                "device": self.device,
                "vocab_size": len(self.tokenizer),
            }

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text according to training configuration.

        Args:
            text: Raw input text

        Returns:
            Preprocessed text
        """
        try:
            # Clean text
            cleaned_text = clean_text(
                text,
                remove_html=self.preprocessing_config.get("remove_html", True),
                remove_urls=self.preprocessing_config.get("remove_urls", True),
                remove_emails=self.preprocessing_config.get("remove_emails", True),
            )

            # Normalize text
            normalized_text = normalize_text(
                cleaned_text,
                lowercase=self.preprocessing_config.get("lowercase", True),
                normalize_unicode=self.preprocessing_config.get(
                    "normalize_unicode", True
                ),
            )

            # Validate length
            min_length = self.preprocessing_config.get("min_length", 10)
            max_length = self.preprocessing_config.get("max_character_length", 2048)

            if not validate_text_length(normalized_text, min_length, max_length):
                logger.warning(f"Text length validation failed: {len(normalized_text)}")

            return normalized_text

        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return text  # Return original text if preprocessing fails

    async def predict_sentiment(self, text: str) -> SentimentResult:
        """Predict sentiment for a single text.

        Args:
            text: Input text for sentiment analysis

        Returns:
            SentimentResult with prediction and scores

        Raises:
            InferenceError: If prediction fails
        """
        if not self.is_ready():
            raise InferenceError("Model not loaded. Call initialize() first.")

        start_time = time.time()

        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                self._executor, self._predict_sync, processed_text
            )

            # Process results
            result = self._process_predictions(predictions[0])

            # Add timing
            processing_time_ms = (time.time() - start_time) * 1000
            result.processing_time_ms = round(processing_time_ms, 2)

            return result

        except Exception as e:
            logger.error(f"Sentiment prediction failed: {e}")
            raise InferenceError(f"Prediction failed: {e}")

    async def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Predict sentiment for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of SentimentResult objects

        Raises:
            InferenceError: If batch prediction fails
        """
        if not self.is_ready():
            raise InferenceError("Model not loaded. Call initialize() first.")

        if len(texts) > self.config.max_batch_size:
            raise InferenceError(
                f"Batch size {len(texts)} exceeds maximum {self.config.max_batch_size}"
            )

        start_time = time.time()

        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]

            # Run batch inference in thread pool
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                self._executor, self._predict_batch_sync, processed_texts
            )

            # Process results
            results = []
            processing_time_ms = (time.time() - start_time) * 1000

            for prediction in predictions:
                result = self._process_predictions(prediction)
                result.processing_time_ms = round(processing_time_ms / len(texts), 2)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Batch sentiment prediction failed: {e}")
            raise InferenceError(f"Batch prediction failed: {e}")

    def _predict_sync(self, text: str) -> List[Dict[str, Any]]:
        """Synchronous prediction for single text."""
        return self.pipeline(text)

    def _predict_batch_sync(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Synchronous prediction for batch of texts."""
        return self.pipeline(texts)

    def _process_predictions(
        self, predictions: List[Dict[str, Any]]
    ) -> SentimentResult:
        """Process raw model predictions into SentimentResult.

        Args:
            predictions: Raw predictions from the model

        Returns:
            Processed SentimentResult
        """
        try:
            # Convert predictions to scores
            scores = {}
            for pred in predictions:
                label = pred["label"]
                score = pred["score"]

                # Map label to sentiment
                if label == "LABEL_0" or label == 0:
                    scores["negative"] = score
                elif label == "LABEL_1" or label == 1:
                    scores["neutral"] = score
                elif label == "LABEL_2" or label == 2:
                    scores["positive"] = score
                else:
                    # Try to map using our label mapping
                    for label_id, sentiment in self.label_mapping.items():
                        if label == f"LABEL_{label_id}":
                            scores[sentiment.lower()] = score

            # Ensure all scores are present
            scores.setdefault("positive", 0.0)
            scores.setdefault("negative", 0.0)
            scores.setdefault("neutral", 0.0)

            # Find the predicted label
            max_score = max(scores.values())
            predicted_label = max(scores, key=scores.get)

            # Convert to enum
            if predicted_label == "positive":
                sentiment_label = SentimentLabel.POSITIVE
            elif predicted_label == "negative":
                sentiment_label = SentimentLabel.NEGATIVE
            else:
                sentiment_label = SentimentLabel.NEUTRAL

            return SentimentResult(
                label=sentiment_label,
                confidence=round(max_score, 4),
                scores=SentimentScore(
                    positive=round(scores["positive"], 4),
                    negative=round(scores["negative"], 4),
                    neutral=round(scores["neutral"], 4),
                ),
            )

        except Exception as e:
            logger.error(f"Failed to process predictions: {e}")
            # Return neutral fallback
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                confidence=0.33,
                scores=SentimentScore(positive=0.33, negative=0.33, neutral=0.34),
            )

    def is_ready(self) -> bool:
        """Check if the service is ready for inference."""
        return (
            self.model is not None
            and self.tokenizer is not None
            and self.pipeline is not None
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_ready():
            return {"status": "not_loaded"}

        return {
            **self.model_info,
            "labels": list(self.label_mapping.values()),
            "ready": True,
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None

            self._executor.shutdown(wait=True)

            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Inference service cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
