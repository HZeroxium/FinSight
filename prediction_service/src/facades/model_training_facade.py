# facades/model_training_facade.py

"""
Model Training Facade - Specialized facade for model training operations

This facade handles all training-related operations including:
- Model training with various configurations
- Training progress monitoring
- Model validation and evaluation
- Training metadata management
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from common.logger.logger_factory import LoggerFactory, LogLevel

from ..core.config import get_settings
from ..core.constants import FacadeConstants
from ..interfaces.model_interface import ITimeSeriesModel
from ..models.model_factory import ModelFactory
from ..schemas.enums import ModelType, TimeFrame
from ..schemas.model_schemas import ModelConfig
from ..utils.model_utils import ModelUtils


class ModelTrainingFacade:
    """
    Specialized facade for model training operations.

    This facade provides:
    - Model training with comprehensive configuration
    - Training progress tracking and monitoring
    - Model validation and evaluation
    - Training metadata management
    - Model lifecycle management for training
    """

    def __init__(self):
        """Initialize the model training facade."""
        self.logger = LoggerFactory.get_logger(
            "ModelTrainingFacade",
            level=LogLevel.INFO,
            console_level=LogLevel.INFO,
            log_file="logs/model_training_facade.log",
        )

        self.settings = get_settings()
        self.model_utils = ModelUtils()

        # Model cache for training operations
        self._cached_models: Dict[str, ITimeSeriesModel] = {}
        self._training_sessions: Dict[str, Dict[str, Any]] = {}

        self.logger.info("Model training facade initialized")

    async def train_model(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        feature_engineering: Any,
        config: ModelConfig,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train a time series model with the specified configuration.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            train_data: Training data
            val_data: Validation data
            feature_engineering: Feature engineering instance
            config: Model configuration
            run_id: Optional experiment run ID

        Returns:
            Training result dictionary
        """
        try:
            # Validate inputs
            self._validate_training_inputs(train_data, val_data, config)

            # Determine feature configuration
            actual_feature_columns = self._determine_feature_configuration(
                config, feature_engineering
            )

            # Create adapter configuration
            adapter_config = self._create_adapter_config(config, actual_feature_columns)

            # Create model instance
            model = self._create_model(model_type, adapter_config)

            # Set run ID for experiment tracking
            if run_id:
                model.set_run_id(run_id)

            # Execute training
            training_result = await self._execute_training(
                model=model,
                train_data=train_data,
                val_data=val_data,
                feature_engineering=feature_engineering,
                adapter_config=adapter_config,
                training_id=f"{symbol}_{timeframe.value}_{model_type.value}_{int(time.time())}",
                run_id=run_id,
            )

            # Save model if training was successful
            if training_result.get("success", False):
                # Save model and metadata
                model_path = await self._save_trained_model(
                    model, symbol, timeframe, model_type, config, training_result
                )

                # Load the trained model
                model = await self._load_model(symbol, timeframe, model_type)
                if model is None:
                    return {"success": False, "error": "Model not found"}

                # Add model path to result
                training_result["model_path"] = model_path
                training_result["model_loaded"] = True

            return training_result

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e)}

    def get_training_progress(self, training_id: str) -> Dict[str, Any]:
        """
        Get training progress for a specific training session.

        Args:
            training_id: Training session identifier

        Returns:
            Dict containing training progress information
        """
        if training_id not in self._training_sessions:
            return {"error": "Training session not found"}

        session = self._training_sessions[training_id]
        return {
            "training_id": training_id,
            "status": session["status"],
            "progress": session["progress"],
            "symbol": session["symbol"],
            "timeframe": session["timeframe"],
            "model_type": session["model_type"],
            "start_time": session["start_time"].isoformat(),
            "elapsed_time": (datetime.now() - session["start_time"]).total_seconds(),
        }

    def list_active_training_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active training sessions.

        Returns:
            List of active training session information
        """
        active_sessions = []
        for training_id, session in self._training_sessions.items():
            if session["status"] in ["initializing", "training"]:
                active_sessions.append(self.get_training_progress(training_id))

        return active_sessions

    def cancel_training(self, training_id: str) -> Dict[str, Any]:
        """
        Cancel an active training session.

        Args:
            training_id: Training session identifier

        Returns:
            Dict containing cancellation result
        """
        if training_id not in self._training_sessions:
            return {"success": False, "error": "Training session not found"}

        session = self._training_sessions[training_id]
        if session["status"] not in ["initializing", "training"]:
            return {"success": False, "error": "Training session is not active"}

        # Mark as cancelled
        session["status"] = "cancelled"
        self.logger.info(f"Training session cancelled: {training_id}")

        return {"success": True, "message": "Training session cancelled"}

    async def evaluate_model(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        test_data: pd.DataFrame,
        config: Optional[ModelConfig] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            test_data: Test dataset
            config: Optional model configuration

        Returns:
            Dict containing evaluation metrics
        """
        try:
            self.logger.info(
                f"Evaluating model: {symbol} {timeframe.value} {model_type.value}"
            )

            # Load the trained model
            model = await self._load_model(symbol, timeframe, model_type)
            if model is None:
                return {"success": False, "error": "Model not found"}

            # Perform evaluation
            evaluation_result = model.evaluate(test_data)

            self.logger.info(f"Model evaluation completed: {symbol} {timeframe.value}")
            return {
                "success": True,
                "metrics": evaluation_result,
                "model_info": {
                    "symbol": symbol,
                    "timeframe": timeframe.value,
                    "model_type": model_type.value,
                },
            }

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {"success": False, "error": str(e)}

    def get_training_history(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get training history with optional filtering by symbol.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of training session information
        """
        history = []
        for training_id, session in self._training_sessions.items():
            if symbol is None or session["symbol"] == symbol:
                history.append(self.get_training_progress(training_id))

        # Sort by start time (most recent first)
        history.sort(key=lambda x: x["start_time"], reverse=True)
        return history

    def cleanup_training_sessions(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Clean up old training sessions.

        Args:
            max_age_hours: Maximum age of sessions to keep

        Returns:
            Dict containing cleanup results
        """
        cutoff_time = datetime.now() - pd.Timedelta(hours=max_age_hours)
        sessions_to_remove = []

        for training_id, session in self._training_sessions.items():
            if session["start_time"] < cutoff_time and session["status"] in [
                "completed",
                "failed",
                "cancelled",
            ]:
                sessions_to_remove.append(training_id)

        # Remove old sessions
        for training_id in sessions_to_remove:
            del self._training_sessions[training_id]

        self.logger.info(f"Cleaned up {len(sessions_to_remove)} old training sessions")

        return {
            "success": True,
            "cleaned_sessions": len(sessions_to_remove),
            "remaining_sessions": len(self._training_sessions),
        }

    # Private helper methods

    def _validate_training_inputs(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, config: ModelConfig
    ) -> None:
        """Validate training inputs."""
        if train_data is None or train_data.empty:
            raise ValueError("Training data cannot be None or empty")

        if val_data is None or val_data.empty:
            raise ValueError("Validation data cannot be None or empty")

        if config is None:
            raise ValueError("Model configuration cannot be None")

        # Additional validations can be added here
        if len(train_data) < 100:
            raise ValueError("Training data must have at least 100 rows")

        if len(val_data) < 20:
            raise ValueError("Validation data must have at least 20 rows")

    def _determine_feature_configuration(
        self, config: ModelConfig, feature_engineering: Any
    ) -> List[str]:
        """Determine actual feature configuration after feature engineering."""
        actual_feature_columns = config.feature_columns

        if feature_engineering is not None:
            try:
                actual_feature_columns = feature_engineering.get_feature_names()
                self.logger.info(
                    f"Feature engineering produced {len(actual_feature_columns)} features"
                )
            except Exception as fe_error:
                self.logger.warning(
                    f"Could not get feature names from feature engineering: {fe_error}"
                )

        return actual_feature_columns

    def _create_adapter_config(
        self, config: ModelConfig, actual_feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Create adapter configuration for model training."""
        adapter_config = {
            "context_length": config.context_length,
            "prediction_length": config.prediction_length,
            "target_column": config.target_column,
            "feature_columns": actual_feature_columns,
            "input_dim": len(actual_feature_columns) if actual_feature_columns else 5,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        }

        # Add model-specific parameters if any
        if hasattr(config, "model_specific_params") and config.model_specific_params:
            adapter_config.update(config.model_specific_params)

        return adapter_config

    def _create_model(
        self, model_type: ModelType, adapter_config: Dict[str, Any]
    ) -> ITimeSeriesModel:
        """Create model instance using factory."""
        model = ModelFactory.create_model(model_type, adapter_config)
        if model is None:
            raise RuntimeError(f"Failed to create model of type {model_type}")
        return model

    async def _execute_training(
        self,
        model: ITimeSeriesModel,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        feature_engineering: Any,
        adapter_config: Dict[str, Any],
        training_id: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the actual model training with experiment tracking."""
        try:
            # Train model with progress tracking and experiment run ID
            training_result = model.train(
                train_data=train_data,
                val_data=val_data,
                feature_engineering=feature_engineering,
                run_id=run_id,
                **adapter_config,
            )

            return training_result

        except Exception as e:
            self.logger.error(f"Training execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _save_trained_model(
        self,
        model: ITimeSeriesModel,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        config: ModelConfig,
        training_result: Dict[str, Any],
    ) -> str:
        """Save trained model and metadata in multiple formats for all adapters."""
        from ..core.constants import FacadeConstants
        from ..utils.model_format_converter import ModelFormatConverter

        # Save model in simple format first (default behavior)
        simple_model_path = self.model_utils.ensure_model_directory(
            symbol, timeframe, model_type, FacadeConstants.ADAPTER_SIMPLE
        )

        # Save model in simple format
        model.save_model(simple_model_path)

        # Save metadata for simple format
        self._save_model_metadata(
            symbol, timeframe, model_type, config, training_result, simple_model_path
        )

        # Convert and save model for all other supported adapters
        try:
            # Check if multi-format saving is enabled
            settings = self.model_utils.settings
            if getattr(settings, "save_multiple_formats", True):
                converter = ModelFormatConverter()
                enabled_adapters = getattr(
                    settings, "enabled_adapters", FacadeConstants.SUPPORTED_ADAPTERS
                )
                conversion_results = converter.convert_model_for_all_adapters(
                    model=model,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    source_path=simple_model_path,
                    target_adapters=enabled_adapters,
                )
            else:
                # Only save simple format
                conversion_results = {FacadeConstants.ADAPTER_SIMPLE: True}

            # Log conversion results
            successful_adapters = [
                adapter for adapter, success in conversion_results.items() if success
            ]
            failed_adapters = [
                adapter
                for adapter, success in conversion_results.items()
                if not success
            ]

            if successful_adapters:
                self.logger.info(
                    f"Model successfully converted for adapters: {successful_adapters}"
                )

            if failed_adapters:
                self.logger.warning(
                    f"Model conversion failed for adapters: {failed_adapters}"
                )

            # Save conversion results in metadata
            training_result["adapter_conversions"] = conversion_results

        except Exception as e:
            self.logger.error(f"Failed to convert model for multiple adapters: {e}")
            # Continue execution - at least we have the simple format
            training_result["adapter_conversions"] = {
                FacadeConstants.ADAPTER_SIMPLE: True
            }

        # Sync to cloud storage if enabled
        try:
            if settings.enable_cloud_storage and settings.enable_model_cloud_sync:
                self.logger.info(
                    f"Syncing model to cloud storage: {symbol}_{timeframe.value}_{model_type.value}"
                )

                # Get run_id from training result if available
                run_id = training_result.get("run_id")

                # Sync simple format to cloud
                cloud_sync_result = await self.model_utils.sync_model_to_cloud(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    adapter_type=FacadeConstants.ADAPTER_SIMPLE,
                    run_id=run_id,
                    force_upload=True,  # Force upload since this is a new model
                    enable_upsert=True,  # Enable upsert for training uploads
                )

                if cloud_sync_result["success"]:
                    self.logger.info(
                        f"Successfully synced model to cloud: {cloud_sync_result.get('action', 'uploaded')}"
                    )
                    training_result["cloud_sync"] = {
                        "simple_adapter": cloud_sync_result,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    self.logger.warning(
                        f"Failed to sync model to cloud: {cloud_sync_result.get('error')}"
                    )
                    training_result["cloud_sync"] = {
                        "simple_adapter": cloud_sync_result,
                        "timestamp": datetime.now().isoformat(),
                    }

                # Sync only specified adapter formats to cloud based on configuration
                configured_cloud_adapters = set(settings.cloud_sync_adapters)
                self.logger.info(
                    f"📤 Configured cloud sync adapters: {configured_cloud_adapters}"
                )

                adapters_to_sync = []
                for adapter_type, success in conversion_results.items():
                    if success and adapter_type in configured_cloud_adapters:
                        adapters_to_sync.append(adapter_type)
                    elif success and adapter_type not in configured_cloud_adapters:
                        self.logger.info(
                            f"⏭️ Skipping cloud sync for {adapter_type} (not in cloud_sync_adapters configuration)"
                        )

                self.logger.info(
                    f"📦 Syncing {len(adapters_to_sync)} adapters to cloud: {adapters_to_sync}"
                )

                for adapter_type in adapters_to_sync:
                    try:
                        self.logger.info(
                            f"🚀 Syncing {adapter_type} adapter to cloud..."
                        )
                        adapter_sync_result = await self.model_utils.sync_model_to_cloud(
                            symbol=symbol,
                            timeframe=timeframe,
                            model_type=model_type,
                            adapter_type=adapter_type,
                            run_id=run_id,
                            force_upload=True,
                            enable_upsert=True,  # Enable upsert for training uploads
                        )

                        if adapter_sync_result["success"]:
                            self.logger.info(
                                f"✅ Successfully synced {adapter_type} adapter to cloud"
                            )
                            training_result["cloud_sync"][
                                adapter_type
                            ] = adapter_sync_result
                        else:
                            self.logger.warning(
                                f"❌ Failed to sync {adapter_type} adapter to cloud: {adapter_sync_result.get('error')}"
                            )
                            training_result["cloud_sync"][
                                adapter_type
                            ] = adapter_sync_result

                    except Exception as e:
                        self.logger.error(
                            f"💥 Error syncing {adapter_type} adapter to cloud: {e}"
                        )
                        training_result["cloud_sync"][adapter_type] = {
                            "success": False,
                            "error": str(e),
                        }

        except Exception as e:
            self.logger.error(f"Failed to sync model to cloud storage: {e}")
            training_result["cloud_sync"] = {"success": False, "error": str(e)}

        return simple_model_path

    def _save_model_metadata(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        config: ModelConfig,
        training_result: Dict[str, Any],
        model_path: str,
    ) -> None:
        """Save model metadata and ensure adapter compatibility."""
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "model_type": model_type.value,
            "training_config": (
                config.model_dump() if hasattr(config, "model_dump") else config
            ),
            "training_result": training_result,
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_path),
        }

        # Save metadata to local storage
        metadata_path = Path(model_path) / self.model_utils.settings.metadata_filename
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Saved model metadata to {metadata_path}")

    async def _load_model(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> Optional[ITimeSeriesModel]:
        """Load a trained model with cloud-first strategy."""
        cache_key = f"{symbol}_{timeframe.value}_{model_type.value}"

        # Check cache first
        if cache_key in self._cached_models:
            return self._cached_models[cache_key]

        try:
            # Use cloud-first loading strategy
            load_result = await self.model_utils.load_model_with_cloud_fallback(
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_type,
                adapter_type="simple",  # Default to simple adapter for loading
            )

            if not load_result["success"]:
                self.logger.error(f"Failed to load model: {load_result.get('error')}")
                return None

            # Load from the determined path
            model_path = Path(load_result["path"])
            if not model_path.exists():
                self.logger.error(f"Model path does not exist: {model_path}")
                return None

            # Create model instance and load
            model = ModelFactory.create_model(model_type, {})
            model.load_model(str(model_path))

            # Cache the model
            self._cached_models[cache_key] = model

            self.logger.info(f"Loaded model from {load_result['source']}: {model_path}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._cached_models.clear()
        self.logger.info("Model cache cleared")
