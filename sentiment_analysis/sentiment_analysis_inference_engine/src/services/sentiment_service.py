# services/sentiment_service.py

"""Sentiment analysis service with Triton client integration."""

import time
from typing import List, Dict, Any
import numpy as np
from transformers import AutoTokenizer
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

from ..core.config import SentimentConfig
from ..core.enums import SentimentLabel, SentimentScore
from ..models.schemas import SentimentResult, SentimentScore as SentimentScoreModel
from common.logger.logger_factory import LoggerFactory, LoggerType

logger = LoggerFactory.get_logger(
    name="sentiment_service",
    logger_type=LoggerType.STANDARD,
    log_file="logs/sentiment_service.log",
)


class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis operations."""

    pass


class SentimentAnalysisService:
    """Service for sentiment analysis using Triton Inference Server."""

    def __init__(
        self,
        config: SentimentConfig,
        triton_host: str = "localhost",
        triton_port: int = 8000,
    ):
        """Initialize sentiment analysis service.

        Args:
            config: Sentiment analysis configuration
            triton_host: Triton server host
            triton_port: Triton server HTTP port
        """
        self.config = config
        self.triton_host = triton_host
        self.triton_port = triton_port
        self.tokenizer = None
        self.triton_client = None

        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_processing_time = 0.0

    async def initialize(self) -> None:
        """Initialize tokenizer and Triton client."""
        try:
            logger.info("Initializing sentiment analysis service...")

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            logger.info(f"Tokenizer loaded: {self.config.tokenizer_name}")

            # Initialize Triton client
            self.triton_client = httpclient.InferenceServerClient(
                url=f"{self.triton_host}:{self.triton_port}", verbose=False
            )

            # Test connection
            if not self.triton_client.is_server_ready():
                raise SentimentAnalysisError("Triton server is not ready")

            # Check model availability
            if not self.triton_client.is_model_ready(self.config.model_name):
                raise SentimentAnalysisError(
                    f"Model '{self.config.model_name}' is not ready"
                )

            logger.info("Sentiment analysis service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize sentiment analysis service: {e}")
            raise SentimentAnalysisError(f"Initialization failed: {e}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.triton_client:
                self.triton_client.close()
                self.triton_client = None
            logger.info("Sentiment analysis service cleaned up")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment for a single text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment analysis result
        """
        start_time = time.time()

        try:
            self._total_requests += 1

            # Preprocess text
            processed_text = self._preprocess_text(text)

            # Tokenize
            inputs = self._tokenize_text(processed_text)

            # Run inference
            outputs = await self._run_inference([inputs])

            # Post-process results
            result = self._postprocess_single_result(text, outputs[0], start_time)

            self._successful_requests += 1
            processing_time = (time.time() - start_time) * 1000
            self._total_processing_time += processing_time

            return result

        except Exception as e:
            self._failed_requests += 1
            logger.error(f"Sentiment analysis failed for text: {e}")
            raise SentimentAnalysisError(f"Analysis failed: {e}")

    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment analysis results
        """
        if not texts:
            return []

        start_time = time.time()

        try:
            self._total_requests += len(texts)

            # Process in batches to respect max batch size
            batch_size = min(len(texts), self.config.max_batch_size)
            results = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_results = await self._process_batch(batch_texts, start_time)
                results.extend(batch_results)

            self._successful_requests += len(results)
            processing_time = (time.time() - start_time) * 1000
            self._total_processing_time += processing_time

            return results

        except Exception as e:
            self._failed_requests += len(texts)
            logger.error(f"Batch sentiment analysis failed: {e}")
            raise SentimentAnalysisError(f"Batch analysis failed: {e}")

    async def _process_batch(
        self, texts: List[str], start_time: float
    ) -> List[SentimentResult]:
        """Process a single batch of texts.

        Args:
            texts: Batch of texts to process
            start_time: Processing start time

        Returns:
            List of sentiment results
        """
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]

        # Tokenize batch
        batch_inputs = [self._tokenize_text(text) for text in processed_texts]

        # Run batch inference
        batch_outputs = await self._run_inference(batch_inputs)

        # Post-process results
        results = []
        for i, (original_text, output) in enumerate(zip(texts, batch_outputs)):
            result = self._postprocess_single_result(original_text, output, start_time)
            results.append(result)

        return results

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis.

        Args:
            text: Raw input text

        Returns:
            Preprocessed text
        """
        # Basic text cleaning
        processed = text.strip()

        # Remove excessive whitespace
        processed = " ".join(processed.split())

        # Truncate if too long (will be handled by tokenizer, but good to be safe)
        if len(processed) > 5000:  # Conservative limit
            processed = processed[:5000]
            logger.warning("Text truncated due to length")

        return processed

    def _tokenize_text(self, text: str) -> Dict[str, np.ndarray]:
        """Tokenize text for model input.

        Args:
            text: Text to tokenize

        Returns:
            Tokenized inputs as numpy arrays
        """
        try:
            # Tokenize with padding and truncation
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="np",
            )

            # Convert to format expected by Triton
            inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }

            return inputs

        except Exception as e:
            raise SentimentAnalysisError(f"Tokenization failed: {e}")

    async def _run_inference(
        self, batch_inputs: List[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        """Run inference on Triton server.

        Args:
            batch_inputs: List of tokenized inputs

        Returns:
            List of model outputs
        """
        try:
            if not batch_inputs:
                return []

            # Prepare batch tensors
            batch_size = len(batch_inputs)

            # Stack inputs into batch tensors
            input_ids_batch = np.vstack(
                [inputs["input_ids"] for inputs in batch_inputs]
            )
            attention_mask_batch = np.vstack(
                [inputs["attention_mask"] for inputs in batch_inputs]
            )

            # Create Triton inference inputs
            triton_inputs = [
                httpclient.InferInput(
                    "input_ids",
                    input_ids_batch.shape,
                    np_to_triton_dtype(input_ids_batch.dtype),
                ),
                httpclient.InferInput(
                    "attention_mask",
                    attention_mask_batch.shape,
                    np_to_triton_dtype(attention_mask_batch.dtype),
                ),
            ]

            # Set input data
            triton_inputs[0].set_data_from_numpy(input_ids_batch)
            triton_inputs[1].set_data_from_numpy(attention_mask_batch)

            # Create output requests
            triton_outputs = [httpclient.InferRequestedOutput("logits")]

            # Run inference
            response = self.triton_client.infer(
                model_name=self.config.model_name,
                inputs=triton_inputs,
                outputs=triton_outputs,
            )

            # Get logits output
            logits = response.as_numpy("logits")  # Shape: (batch_size, num_classes)

            # Convert logits to probabilities using softmax
            probabilities = self._softmax(logits)

            # Split batch results
            return [probabilities[i] for i in range(batch_size)]

        except Exception as e:
            logger.error(f"Triton inference failed: {e}")
            raise SentimentAnalysisError(f"Inference failed: {e}")

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits.

        Args:
            logits: Raw model logits

        Returns:
            Probability distributions
        """
        # Subtract max for numerical stability
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _postprocess_single_result(
        self, original_text: str, probabilities: np.ndarray, start_time: float
    ) -> SentimentResult:
        """Post-process single inference result.

        Args:
            original_text: Original input text
            probabilities: Model probability outputs
            start_time: Processing start time

        Returns:
            Sentiment analysis result
        """
        # Get predicted label
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = SentimentLabel(self.config.label_mapping[predicted_idx])
        confidence = float(probabilities[predicted_idx])

        # Create detailed scores
        scores = SentimentScoreModel(
            negative=float(probabilities[SentimentScore.NEGATIVE]),
            neutral=float(probabilities[SentimentScore.NEUTRAL]),
            positive=float(probabilities[SentimentScore.POSITIVE]),
        )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        return SentimentResult(
            text=original_text,
            label=predicted_label,
            confidence=confidence,
            scores=scores,
            processing_time_ms=processing_time_ms,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics.

        Returns:
            Service statistics
        """
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "average_processing_time_ms": (
                self._total_processing_time / max(self._successful_requests, 1)
            ),
            "success_rate": (self._successful_requests / max(self._total_requests, 1)),
        }

    def reset_statistics(self) -> None:
        """Reset service statistics."""
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_processing_time = 0.0
        logger.info("Service statistics reset")
