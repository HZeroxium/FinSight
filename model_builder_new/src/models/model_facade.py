# models/model_facade.py

"""
Model Facade - High-level interface for model operations

This facade provides a simplified interface for common model operations,
abstracting the complexity of different model types and implementations.
"""

from typing import Dict, Any, Optional, List
import pandas as pd

from ..interfaces.model_interface import ITimeSeriesModel
from ..models.model_factory import ModelFactory
from ..data.feature_engineering import BasicFeatureEngineering
from ..schemas.enums import ModelType
from ..logger.logger_factory import LoggerFactory


class ModelFacade:
    """
    Facade for simplifying model operations

    Provides high-level methods for training, prediction, and model management
    while hiding the complexity of different adapters and implementations.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger("ModelFacade")
        self._models: Dict[str, ITimeSeriesModel] = {}

    def create_and_train_model(
        self,
        model_type: ModelType,
        data: pd.DataFrame,
        target_column: str = "close",
        context_length: int = 64,
        prediction_length: int = 1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        **training_kwargs,
    ) -> Dict[str, Any]:
        """
        Create and train a model with minimal configuration

        Args:
            model_type: Type of model to create
            data: Training data
            target_column: Target column to predict
            context_length: Input sequence length
            prediction_length: Prediction horizon
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            **training_kwargs: Additional training parameters

        Returns:
            Training results
        """
        try:
            self.logger.info(f"Creating and training {model_type} model")

            # Create feature engineering
            feature_engineering = BasicFeatureEngineering(
                add_technical_indicators=training_kwargs.get(
                    "use_technical_indicators", True
                ),
                add_datetime_features=training_kwargs.get(
                    "add_datetime_features", False
                ),
                normalize_features=training_kwargs.get("normalize_features", True),
            )

            # Process data
            processed_data = feature_engineering.fit_transform(data)

            # Split data
            n = len(processed_data)
            train_size = int(n * train_ratio)
            val_size = int(n * val_ratio)

            train_data = processed_data[:train_size]
            val_data = processed_data[train_size : train_size + val_size]
            test_data = processed_data[train_size + val_size :]

            # Create model config
            config = {
                "context_length": context_length,
                "prediction_length": prediction_length,
                "num_input_channels": len(feature_engineering.get_feature_names()),
                "target_column": target_column,
                "num_epochs": training_kwargs.get("num_epochs", 10),
                "batch_size": training_kwargs.get("batch_size", 32),
                "learning_rate": training_kwargs.get("learning_rate", 1e-3),
            }

            # Create and train model
            model = ModelFactory.create_model(model_type, config)

            training_result = model.train(
                train_data=train_data,
                val_data=val_data,
                feature_engineering=feature_engineering,
            )

            # Store model for future use
            model_key = f"{model_type.value}_{target_column}_{context_length}_{prediction_length}"
            self._models[model_key] = model

            # Evaluate on test data if available
            if len(test_data) > 0:
                eval_metrics = model.evaluate(test_data)
                training_result["test_metrics"] = eval_metrics

            return training_result

        except Exception as e:
            self.logger.error(f"Failed to create and train model: {e}")
            return {"success": False, "error": str(e)}

    def quick_predict(
        self, model_key: str, data: pd.DataFrame, n_steps: int = 1
    ) -> Dict[str, Any]:
        """
        Make predictions using a cached model

        Args:
            model_key: Key of the cached model
            data: Input data for prediction
            n_steps: Number of prediction steps

        Returns:
            Prediction results
        """
        if model_key not in self._models:
            return {
                "success": False,
                "error": f"Model '{model_key}' not found in cache",
            }

        try:
            model = self._models[model_key]
            return model.predict(data, n_steps)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {"success": False, "error": str(e)}

    def predict(
        self, model_type: ModelType, data: pd.DataFrame, n_steps: int = 1
    ) -> Dict[str, Any]:
        """Make predictions using a model type"""

        # Look for an existing model of this type
        for model_key, model in self._models.items():
            if model_type.value in model_key:
                return model.predict(data, n_steps)

        # If no model found, return error
        return {
            "success": False,
            "error": f"No trained model found for {model_type.value}",
        }

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a cached model"""
        if model_key not in self._models:
            return None

        return {
            "model_key": model_key,
            "is_loaded": True,
            "model_type": type(self._models[model_key]).__name__,
        }

    def list_cached_models(self) -> List[str]:
        """List all cached model keys"""
        return list(self._models.keys())

    def clear_cache(self) -> None:
        """Clear all cached models"""
        self._models.clear()
        self.logger.info("Model cache cleared")

    def evaluate_model(
        self,
        model_type: ModelType,
        test_data: pd.DataFrame,
        detailed_metrics: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a model on test data"""

        # Look for an existing model of this type
        for model_key, model in self._models.items():
            if model_type.value in model_key:
                return model.evaluate(test_data)

        # If no model found, return error
        return {
            "success": False,
            "error": f"No trained model found for {model_type.value}",
        }
