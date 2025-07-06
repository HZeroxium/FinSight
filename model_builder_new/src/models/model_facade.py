# models/model_facade.py

"""
Model Facade - High-level interface for model operations
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from ..interfaces.model_interface import ITimeSeriesModel
from ..models.model_factory import ModelFactory
from ..schemas.enums import ModelType, TimeFrame
from ..schemas.model_schemas import ModelConfig, ModelInfo
from ..utils.model_utils import ModelUtils
from ..logger.logger_factory import LoggerFactory


class ModelFacade:
    """
    Facade for simplifying model operations
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger("ModelFacade")
        self.model_utils = ModelUtils()
        self._cached_models: Dict[str, ITimeSeriesModel] = {}

    def train_model(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        feature_engineering: Any,
        config: ModelConfig,
    ) -> Dict[str, Any]:
        """
        Train a model and save it

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type to train
            train_data: Training data
            val_data: Validation data
            feature_engineering: Feature engineering instance
            config: Model configuration

        Returns:
            Training results
        """
        try:
            self.logger.info(
                f"Training {model_type.value} model for {symbol} {timeframe}"
            )

            # Create model configuration for adapter
            adapter_config = {
                "context_length": config.context_length,
                "prediction_length": config.prediction_length,
                "target_column": config.target_column,
                "feature_columns": config.feature_columns,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            }

            # Create model using factory
            model = ModelFactory.create_model(model_type, adapter_config)

            # Train model
            training_result = model.train(
                train_data=train_data,
                val_data=val_data,
                feature_engineering=feature_engineering,
                **adapter_config,
            )

            if training_result.get("success", False):
                # Save model
                model_path = self.model_utils.ensure_model_directory(
                    symbol, timeframe, model_type
                )
                model.save_model(model_path)

                # Save metadata
                self._save_model_metadata(
                    symbol, timeframe, model_type, config, training_result
                )

                # Cache model
                cache_key = self.model_utils.generate_model_identifier(
                    symbol, timeframe, model_type
                )
                self._cached_models[cache_key] = model

                training_result.update(
                    {
                        "model_path": str(model_path),
                        "model_identifier": cache_key,
                    }
                )

            return training_result

        except Exception as e:
            self.logger.error(f"Failed to train model: {e}")
            return {"success": False, "error": str(e)}

    def predict(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        recent_data: pd.DataFrame,
        n_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained model

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            recent_data: Recent data for prediction
            n_steps: Number of prediction steps

        Returns:
            Prediction results
        """
        try:
            self.logger.info(
                f"Making {n_steps}-step prediction for {symbol} {timeframe} using {model_type.value}"
            )

            # Load model
            model = self._load_model(symbol, timeframe, model_type)

            # Make forecast
            prediction_result = model.forecast(recent_data, n_steps=n_steps)

            # Add model info
            prediction_result["model_info"] = {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "model_type": model_type.value,
                "n_steps": n_steps,
            }

            return prediction_result

        except Exception as e:
            self.logger.error(f"Failed to make prediction: {e}")
            return {"success": False, "error": str(e)}

    def forecast(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        recent_data: pd.DataFrame,
        n_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Make forecasts using a trained model

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            recent_data: Recent data for prediction
            n_steps: Number of forecast steps

        Returns:
            Forecast results
        """

        # Alias for predict method
        return self.predict(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            recent_data=recent_data,
            n_steps=n_steps,
        )

    def evaluate_model(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        test_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            test_data: Test data

        Returns:
            Evaluation results
        """
        try:
            self.logger.info(
                f"Evaluating {model_type.value} model for {symbol} {timeframe}"
            )

            # Load model
            model = self._load_model(symbol, timeframe, model_type)

            # Evaluate
            evaluation_result = model.evaluate(test_data)

            return evaluation_result

        except Exception as e:
            self.logger.error(f"Failed to evaluate model: {e}")
            return {"success": False, "error": str(e)}

    def list_available_models(self) -> List[ModelInfo]:
        """
        List all available trained models

        Returns:
            List of model information
        """
        try:
            models = []

            if not self.model_utils.settings.models_dir.exists():
                return models

            for model_dir in self.model_utils.settings.models_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                metadata_path = (
                    self.model_utils.settings.models_dir
                    / model_dir.name
                    / self.model_utils.settings.metadata_filename
                )
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    # Get file size
                    checkpoint_path = (
                        self.model_utils.settings.models_dir
                        / model_dir.name
                        / self.model_utils.settings.checkpoint_filename
                    )
                    file_size_mb = None
                    if checkpoint_path.exists():
                        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

                    model_info = ModelInfo(
                        symbol=metadata["symbol"],
                        timeframe=TimeFrame(metadata["timeframe"]),
                        model_type=ModelType(metadata["model_type"]),
                        model_path=str(model_dir),
                        created_at=datetime.fromisoformat(metadata["created_at"]),
                        config=ModelConfig(**metadata["config"]),
                        file_size_mb=file_size_mb,
                        is_available=checkpoint_path.exists(),
                    )

                    models.append(model_info)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse model metadata from {metadata_path}: {e}"
                    )
                    continue

            return models

        except Exception as e:
            self.logger.error(f"Failed to list available models: {e}")
            return []

    def model_exists(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> bool:
        """Check if a model exists"""
        checkpoint_path = self.model_utils.get_checkpoint_path(
            symbol, timeframe, model_type
        )
        return checkpoint_path.exists()

    def _load_model(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> ITimeSeriesModel:
        """Load a model from cache or disk"""
        cache_key = self.model_utils.generate_model_identifier(
            symbol, timeframe, model_type
        )

        # Check cache first
        if cache_key in self._cached_models:
            return self._cached_models[cache_key]

        # Load from disk
        model_path = self.model_utils.generate_model_path(symbol, timeframe, model_type)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load config to create model
        config_path = self.model_utils.get_config_path(symbol, timeframe, model_type)
        with open(config_path, "r") as f:
            saved_config = json.load(f)

        # Create model and load state
        model = ModelFactory.create_model(model_type, saved_config)
        model.load_model(model_path)

        # Cache model
        self._cached_models[cache_key] = model

        return model

    def _save_model_metadata(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        config: ModelConfig,
        training_result: Dict[str, Any],
    ) -> None:
        """Save model metadata"""
        metadata_path = self.model_utils.get_metadata_path(
            symbol, timeframe, model_type
        )

        metadata = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "model_type": model_type.value,
            "created_at": datetime.now().isoformat(),
            "config": config.model_dump(),
            "training_result": training_result,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Also save config separately for model loading
        config_path = self.model_utils.get_config_path(symbol, timeframe, model_type)
        adapter_config = {
            "context_length": config.context_length,
            "prediction_length": config.prediction_length,
            "target_column": config.target_column,
            "feature_columns": config.feature_columns,
        }

        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)

    def clear_cache(self):
        """Clear model cache"""
        self._cached_models.clear()
        self.logger.info("Model cache cleared")
