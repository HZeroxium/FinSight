# finetune/predictor.py

"""
Modern predictor for fine-tuned financial models.
"""

from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path

from ..common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel
from .config import FineTuneConfig
from .data_processor import FinancialDataProcessor


class FineTunePredictor:
    """Modern predictor for fine-tuned financial models"""

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger(
            name="FineTunePredictor",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
        )
        self.model = None
        self.tokenizer = None
        self.data_processor = None

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load fine-tuned model for inference

        Args:
            model_path: Path to the fine-tuned model
        """
        from .model_factory import ModelFactory

        self.logger.info(f"Loading model from {model_path}")

        try:
            # Initialize components
            model_factory = ModelFactory(self.config)
            self.data_processor = FinancialDataProcessor(self.config)

            # Load model and tokenizer
            self.model, self.tokenizer = model_factory.create_model_and_tokenizer()

            # Load fine-tuned weights
            model_path = Path(model_path)
            if model_path.is_dir():
                # Load from directory (HuggingFace format)
                from transformers import AutoModel

                self.model = AutoModel.from_pretrained(model_path)
            else:
                # Load state dict
                state_dict = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state_dict)

            self.model.eval()

            self.logger.info("✅ Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict_single(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction on single sequence

        Args:
            data: DataFrame with financial data

        Returns:
            Prediction result with metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Process data
            processed_data = self.data_processor._preprocess_data(data)

            # Create sequence
            if len(processed_data) < self.config.sequence_length:
                raise ValueError(
                    f"Need at least {self.config.sequence_length} data points"
                )

            feature_data = processed_data[self.config.features].values
            sequence = feature_data[-self.config.sequence_length :]

            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                if hasattr(self.model, "generate"):
                    # Generative model
                    if self.tokenizer:
                        # Convert to text and tokenize
                        text_input = self._format_sequence_as_text(sequence)
                        inputs = self.tokenizer(
                            text_input,
                            return_tensors="pt",
                            max_length=self.config.sequence_length,
                            truncation=True,
                            padding=True,
                        )
                        outputs = self.model.generate(
                            inputs["input_ids"], max_length=32, do_sample=False
                        )
                        prediction = self._extract_prediction_from_generated(outputs[0])
                    else:
                        prediction = self.model(sequence_tensor).logits.item()
                else:
                    # Discriminative model
                    outputs = self.model(sequence_tensor)
                    if hasattr(outputs, "logits"):
                        prediction = outputs.logits.squeeze().item()
                    else:
                        prediction = outputs.squeeze().item()

            # Get current price for context
            current_price = processed_data[self.config.target_column].iloc[-1]

            # Calculate change
            price_change = prediction - current_price
            price_change_pct = (price_change / current_price) * 100

            result = {
                "prediction": float(prediction),
                "current_price": float(current_price),
                "predicted_change": float(price_change),
                "predicted_change_pct": float(price_change_pct),
                "direction": "up" if price_change > 0 else "down",
                "confidence": self._calculate_confidence(sequence, prediction),
                "timestamp": datetime.now().isoformat(),
                "sequence_length": self.config.sequence_length,
                "features_used": self.config.features,
            }

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(self, data_list: List[pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple sequences

        Args:
            data_list: List of DataFrames with financial data

        Returns:
            List of prediction results
        """
        results = []

        for i, data in enumerate(data_list):
            try:
                result = self.predict_single(data)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to predict batch item {i}: {e}")
                results.append(
                    {
                        "error": str(e),
                        "batch_index": i,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return results

    def predict_sequence(self, data: pd.DataFrame, n_steps: int = 5) -> Dict[str, Any]:
        """
        Make multi-step ahead predictions

        Args:
            data: Historical data
            n_steps: Number of steps to predict ahead

        Returns:
            Multi-step prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        predictions = []
        current_data = data.copy()

        for step in range(n_steps):
            # Predict next value
            result = self.predict_single(current_data)
            prediction = result["prediction"]
            predictions.append(prediction)

            # Update data with prediction for next step
            # Create new row with predicted value
            last_row = current_data.iloc[-1].copy()
            last_row[self.config.target_column] = prediction

            # Simple feature updates (can be enhanced)
            for feature in self.config.features:
                if feature != self.config.target_column:
                    if feature in ["Open", "High", "Low"]:
                        last_row[feature] = prediction * (
                            0.99 + 0.02 * np.random.random()
                        )
                    elif feature == "Volume":
                        last_row[feature] = current_data["Volume"].iloc[-5:].mean()

            # Add to data
            current_data = pd.concat(
                [current_data, last_row.to_frame().T], ignore_index=True
            )

        # Generate future dates
        if "Date" in data.columns:
            last_date = pd.to_datetime(data["Date"].iloc[-1])
            future_dates = [
                (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                for i in range(n_steps)
            ]
        else:
            future_dates = [f"Step_{i+1}" for i in range(n_steps)]

        return {
            "predictions": predictions,
            "dates": future_dates,
            "n_steps": n_steps,
            "base_price": float(data[self.config.target_column].iloc[-1]),
            "total_change": (
                float(predictions[-1] - predictions[0]) if predictions else 0.0
            ),
            "timestamp": datetime.now().isoformat(),
        }

    def _format_sequence_as_text(self, sequence: np.ndarray) -> str:
        """Format sequence as text for language models"""
        # Simple formatting - can be enhanced
        values = sequence.flatten()[-10:]  # Last 10 values
        formatted = [f"{val:.4f}" for val in values]
        return f"Financial sequence: {', '.join(formatted)}. Predict next:"

    def _extract_prediction_from_generated(self, generated_ids: torch.Tensor) -> float:
        """Extract numerical prediction from generated tokens"""
        if self.tokenizer:
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Extract number from text (simple approach)
            import re

            numbers = re.findall(r"-?\d+\.?\d*", text)
            if numbers:
                return float(numbers[-1])
        return 0.0

    def _calculate_confidence(self, sequence: np.ndarray, prediction: float) -> float:
        """Calculate prediction confidence based on sequence stability"""
        # Simple confidence measure based on sequence volatility
        volatility = np.std(sequence[:, -1])  # Target column volatility
        mean_val = np.mean(sequence[:, -1])

        # Normalize volatility
        normalized_vol = volatility / mean_val if mean_val > 0 else 1.0

        # Confidence inversely related to volatility
        confidence = 1.0 / (1.0 + normalized_vol)

        return min(max(confidence, 0.0), 1.0)
