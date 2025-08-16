# utils/model_utils.py

"""
Model Utils Orchestrator - Coordinated interface for model operations

This module provides a unified interface that orchestrates different model utility
components including path management, metadata operations, local operations,
and cloud storage operations. Maintains backward compatibility while delegating
to specialized components.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from ..schemas.enums import (
    ModelType,
    TimeFrame,
    PredictionTrend,
    PercentageCalculationMethod,
    PredictionConfidenceLevel,
)
from ..core.config import get_settings
from .model.path_manager import ModelPathManager
from .model.metadata_manager import ModelMetadataManager
from .model.local_operations import LocalModelOperations
from .model.cloud_operations import CloudModelOperations
from common.logger.logger_factory import LoggerFactory


class ModelUtils:
    """
    Model utilities orchestrator providing unified interface for all model operations.

    This class coordinates between different specialized components:
    - PathManager: Model path generation and management
    - MetadataManager: Model metadata operations
    - LocalOperations: Local model operations
    - CloudOperations: Cloud storage operations
    """

    def __init__(self, storage_client: Optional[Any] = None):
        self.logger = LoggerFactory.get_logger("ModelUtils")
        self.settings = get_settings()

        # Initialize specialized components
        self.path_manager = ModelPathManager()
        self.metadata_manager = ModelMetadataManager(self.path_manager)
        self.local_ops = LocalModelOperations(self.path_manager, self.metadata_manager)
        self.cloud_ops = CloudModelOperations(
            storage_client, self.path_manager, self.metadata_manager, self.local_ops
        )

        # Backward compatibility properties
        self._storage_client = storage_client
        self._experiment_tracker = None

    @property
    def storage_client(self):
        """Storage client instance - delegated to cloud operations"""
        return self.cloud_ops.storage_client

    @property
    def experiment_tracker(self):
        """Lazy-loaded experiment tracker instance"""
        if self._experiment_tracker is None:
            # Avoid circular import by using string import
            try:
                from .dependencies import get_experiment_tracker

                self._experiment_tracker = get_experiment_tracker()
            except ImportError:
                self.logger.warning(
                    "Experiment tracker not available due to import issues"
                )
                self._experiment_tracker = None
        return self._experiment_tracker

    # ===== Path Management Operations (delegated to PathManager) =====

    def generate_model_identifier(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> str:
        """Generate standardized model identifier - delegated to path manager"""
        return self.path_manager.generate_model_identifier(
            symbol, timeframe, model_type
        )

    def generate_cloud_object_key(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        file_name: Optional[str] = None,
    ) -> str:
        """Generate cloud storage object key - delegated to path manager"""
        return self.path_manager.generate_cloud_object_key(
            symbol, timeframe, model_type, adapter_type, file_name
        )

    def generate_cloud_serving_key(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        file_name: Optional[str] = None,
    ) -> str:
        """Generate cloud serving key - delegated to path manager"""
        return self.path_manager.generate_cloud_serving_key(
            symbol, timeframe, model_type, adapter_type, file_name
        )

    def generate_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Generate model path - delegated to path manager"""
        return self.path_manager.generate_model_path(
            symbol, timeframe, model_type, adapter_type
        )

    def get_checkpoint_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get checkpoint path - delegated to path manager"""
        return self.path_manager.get_checkpoint_path(
            symbol, timeframe, model_type, adapter_type
        )

    def get_metadata_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get metadata path - delegated to path manager"""
        return self.path_manager.get_metadata_path(
            symbol, timeframe, model_type, adapter_type
        )

    def get_config_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get config path - delegated to path manager"""
        return self.path_manager.get_config_path(
            symbol, timeframe, model_type, adapter_type
        )

    def ensure_model_directory(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Ensure model directory - delegated to path manager"""
        return self.path_manager.ensure_model_directory(
            symbol, timeframe, model_type, adapter_type
        )

    def get_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Path:
        """Get model path - delegated to path manager"""
        return self.path_manager.get_model_path(
            symbol, timeframe, model_type, adapter_type
        )

    def get_simple_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get simple model path - delegated to path manager"""
        return self.path_manager.get_simple_model_path(symbol, timeframe, model_type)

    def get_torchscript_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get torchscript model path - delegated to path manager"""
        return self.path_manager.get_torchscript_model_path(
            symbol, timeframe, model_type
        )

    def get_torchserve_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get torchserve model path - delegated to path manager"""
        return self.path_manager.get_torchserve_model_path(
            symbol, timeframe, model_type
        )

    def get_triton_model_path(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
    ) -> Path:
        """Get triton model path - delegated to path manager"""
        return self.path_manager.get_triton_model_path(symbol, timeframe, model_type)

    # ===== Metadata Operations (delegated to MetadataManager) =====

    def save_json(self, data: Dict[str, Any], file_path: str) -> None:
        """Save JSON data - delegated to metadata manager"""
        return self.metadata_manager.save_json(data, file_path)

    def load_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load JSON data - delegated to metadata manager"""
        return self.metadata_manager.load_json(file_path)

    # ===== Local Operations (delegated to LocalOperations) =====

    def model_exists(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """Check if model exists - delegated to local operations"""
        return self.local_ops.model_exists(symbol, timeframe, model_type, adapter_type)

    def copy_model_for_adapter(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        source_adapter: str = "simple",
        target_adapter: str = "torchscript",
    ) -> bool:
        """Copy model for adapter - delegated to local operations"""
        return self.local_ops.copy_model_for_adapter(
            symbol, timeframe, model_type, source_adapter, target_adapter
        )

    def ensure_adapter_compatibility(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        target_adapters: List[str] = None,
    ) -> None:
        """Ensure adapter compatibility - delegated to local operations"""
        return self.local_ops.ensure_adapter_compatibility(
            symbol, timeframe, model_type, target_adapters
        )

    # ===== Cloud Operations (delegated to CloudOperations) =====

    async def sync_model_to_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        run_id: Optional[str] = None,
        force_upload: bool = False,
        enable_upsert: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Sync model to cloud - delegated to cloud operations"""
        return await self.cloud_ops.sync_model_to_cloud(
            symbol,
            timeframe,
            model_type,
            adapter_type,
            run_id,
            force_upload,
            enable_upsert,
        )

    async def sync_model_from_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """Sync model from cloud - delegated to cloud operations"""
        return await self.cloud_ops.sync_model_from_cloud(
            symbol, timeframe, model_type, adapter_type, force_download
        )

    async def upload_model_to_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload model to cloud - delegated to cloud operations"""
        return await self.cloud_ops.upload_model_to_cloud(
            symbol, timeframe, model_type, adapter_type, run_id
        )

    async def download_model_from_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """Download model from cloud - delegated to cloud operations"""
        return await self.cloud_ops.download_model_from_cloud(
            symbol, timeframe, model_type, adapter_type, force_download
        )

    async def model_exists_in_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """Check if model exists in cloud - delegated to cloud operations"""
        return await self.cloud_ops.model_exists_in_cloud(
            symbol, timeframe, model_type, adapter_type
        )

    async def save_model_metadata_to_cloud(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        metadata: Dict[str, Any],
        adapter_type: str = "simple",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save model metadata to cloud - delegated to cloud operations"""
        return await self.cloud_ops.save_model_metadata_to_cloud(
            symbol, timeframe, model_type, metadata, adapter_type, run_id
        )

    async def load_model_with_cloud_fallback(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_cloud_download: bool = False,
    ) -> Dict[str, Any]:
        """Load model with cloud fallback - delegated to cloud operations"""
        return await self.cloud_ops.load_model_with_cloud_fallback(
            symbol, timeframe, model_type, adapter_type, force_cloud_download
        )

    async def health_check_cloud_storage(self) -> Dict[str, Any]:
        """Health check cloud storage - delegated to cloud operations"""
        return await self.cloud_ops.health_check_cloud_storage()

    async def load_model_for_serving(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        force_cloud_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Load model for serving with cloud-first strategy - delegated to cloud operations.

        This method implements cloud-first loading for serving operations:
        1. Try to load from cloud storage first
        2. Fall back to local storage if cloud fails
        3. Cache the model locally for future use
        """
        return await self.cloud_ops.load_model_with_cloud_fallback(
            symbol, timeframe, model_type, adapter_type, force_cloud_download
        )

    # ===== Utility Methods for Prediction Processing =====

    def calculate_prediction_percentages(
        self,
        raw_predictions: List[float],
        current_price: Optional[float] = None,
        base_price: Optional[float] = None,
        method: PercentageCalculationMethod = PercentageCalculationMethod.CURRENT_PRICE_BASED,
    ) -> List[float]:
        """
        Calculate percentage changes from raw prediction values.

        Args:
            raw_predictions: List of raw prediction values
            current_price: Current market price (if None, uses first prediction as base)
            base_price: Base price for percentage calculation (if None, uses current_price)
            method: Method for calculating percentage changes

        Returns:
            List of percentage changes where positive values indicate increase, negative indicate decrease
        """
        if not raw_predictions:
            return []

        try:
            # Determine base price for percentage calculation based on method
            if (
                method == PercentageCalculationMethod.CUSTOM_BASE_PRICE
                and base_price is not None
            ):
                reference_price = base_price
            elif (
                method == PercentageCalculationMethod.CURRENT_PRICE_BASED
                and current_price is not None
            ):
                reference_price = current_price
            elif method == PercentageCalculationMethod.FIRST_PREDICTION_BASED:
                reference_price = raw_predictions[0]
                self.logger.info(
                    f"Using first prediction as base for percentage calculation: {reference_price}"
                )
            elif method == PercentageCalculationMethod.ROLLING_BASE:
                # For rolling base, we'll calculate relative to previous prediction
                return self._calculate_rolling_percentages(raw_predictions)
            else:
                # Fallback to current price or first prediction
                reference_price = (
                    current_price if current_price is not None else raw_predictions[0]
                )
                self.logger.warning(
                    f"Using fallback reference price: {reference_price} for method: {method.value}"
                )

            if reference_price <= 0:
                self.logger.error(
                    f"Invalid reference price for percentage calculation: {reference_price}"
                )
                return []

            # Calculate percentage changes
            percentages = []
            for prediction in raw_predictions:
                if prediction <= 0:
                    self.logger.warning(
                        f"Invalid prediction value for percentage calculation: {prediction}"
                    )
                    percentages.append(0.0)
                else:
                    percentage_change = (
                        (prediction - reference_price) / reference_price
                    ) * 100
                    percentages.append(round(percentage_change, 4))

            self.logger.debug(
                f"Calculated {len(percentages)} percentage changes using method: {method.value}"
            )
            return percentages

        except Exception as e:
            self.logger.error(f"Error calculating prediction percentages: {e}")
            return []

    def _calculate_rolling_percentages(
        self, raw_predictions: List[float]
    ) -> List[float]:
        """Calculate percentage changes using rolling base (each prediction relative to previous)."""
        if len(raw_predictions) < 2:
            return []

        percentages = [0.0]  # First prediction has no change
        for i in range(1, len(raw_predictions)):
            prev_price = raw_predictions[i - 1]
            curr_price = raw_predictions[i]

            if prev_price <= 0:
                percentages.append(0.0)
            else:
                percentage_change = ((curr_price - prev_price) / prev_price) * 100
                percentages.append(round(percentage_change, 4))

        return percentages

    def calculate_prediction_metadata(
        self,
        raw_predictions: List[float],
        percentages: List[float],
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive prediction metadata including statistics.

        Args:
            raw_predictions: List of raw prediction values
            percentages: List of percentage changes
            current_price: Current market price

        Returns:
            Dictionary containing prediction metadata and statistics
        """
        if not raw_predictions or not percentages:
            return {}

        try:
            # Calculate trend analysis
            overall_trend = self._determine_trend(percentages)
            trend_strength = (
                abs(sum(percentages)) / len(percentages) if percentages else 0.0
            )
            volatility = max(percentages) - min(percentages) if percentages else 0.0

            # Determine confidence level based on trend strength and volatility
            confidence_level = self._determine_confidence_level(
                trend_strength, volatility
            )

            metadata = {
                "prediction_count": len(raw_predictions),
                "raw_predictions_summary": {
                    "min": min(raw_predictions),
                    "max": max(raw_predictions),
                    "mean": sum(raw_predictions) / len(raw_predictions),
                },
                "percentage_summary": {
                    "min": min(percentages),
                    "max": max(percentages),
                    "mean": sum(percentages) / len(percentages),
                },
                "trend_analysis": {
                    "overall_trend": overall_trend.value,
                    "trend_strength": round(trend_strength, 4),
                    "volatility": round(volatility, 4),
                    "confidence_level": confidence_level.value,
                },
            }

            # Add current price context if available
            if current_price is not None:
                metadata["price_context"] = {
                    "current_price": current_price,
                    "predicted_prices": raw_predictions,
                    "price_changes": [
                        round(p - current_price, 4) for p in raw_predictions
                    ],
                }

            self.logger.debug(
                f"Generated prediction metadata for {len(raw_predictions)} predictions"
            )
            return metadata

        except Exception as e:
            self.logger.error(f"Error calculating prediction metadata: {e}")
            return {}

    def _determine_trend(self, percentages: List[float]) -> PredictionTrend:
        """Determine the overall trend based on percentage changes."""
        if not percentages:
            return PredictionTrend.NEUTRAL

        avg_percentage = sum(percentages) / len(percentages)
        volatility = max(percentages) - min(percentages)

        # High volatility threshold (can be made configurable)
        HIGH_VOLATILITY_THRESHOLD = 10.0

        if volatility > HIGH_VOLATILITY_THRESHOLD:
            return PredictionTrend.VOLATILE
        elif avg_percentage > 1.0:  # More than 1% average increase
            return PredictionTrend.BULLISH
        elif avg_percentage < -1.0:  # More than 1% average decrease
            return PredictionTrend.BEARISH
        else:
            return PredictionTrend.NEUTRAL

    def _determine_confidence_level(
        self, trend_strength: float, volatility: float
    ) -> PredictionConfidenceLevel:
        """Determine confidence level based on trend strength and volatility."""
        # Normalize trend strength (0-100 scale)
        normalized_strength = min(trend_strength * 10, 100)  # Scale factor of 10

        # Volatility penalty (higher volatility reduces confidence)
        volatility_penalty = min(volatility * 2, 50)  # Scale factor of 2

        # Calculate final confidence score
        confidence_score = max(0, normalized_strength - volatility_penalty)

        # Map to confidence levels
        if confidence_score >= 90:
            return PredictionConfidenceLevel.VERY_HIGH
        elif confidence_score >= 75:
            return PredictionConfidenceLevel.HIGH
        elif confidence_score >= 50:
            return PredictionConfidenceLevel.MEDIUM
        elif confidence_score >= 25:
            return PredictionConfidenceLevel.LOW
        else:
            return PredictionConfidenceLevel.VERY_LOW

    # ===== Static Methods for Backward Compatibility =====

    @staticmethod
    def save_model_metadata(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        metadata: Dict[str, Any],
        adapter_type: str = "simple",
    ) -> Path:
        """Save model metadata - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.metadata_manager.save_model_metadata(
            symbol, timeframe, model_type, metadata, adapter_type
        )

    @staticmethod
    def load_model_metadata(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Optional[Dict[str, Any]]:
        """Load model metadata - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.metadata_manager.load_model_metadata(
            symbol, timeframe, model_type, adapter_type
        )

    @staticmethod
    def model_exists(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """Check if model exists - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.local_ops.model_exists(symbol, timeframe, model_type, adapter_type)

    @staticmethod
    def list_available_models(base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """List available models - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.local_ops.list_available_models(base_dir)

    @staticmethod
    def delete_model(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> bool:
        """Delete model - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.local_ops.delete_model(symbol, timeframe, model_type, adapter_type)

    @staticmethod
    def get_model_size(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
    ) -> Optional[int]:
        """Get model size - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.local_ops.get_model_size(
            symbol, timeframe, model_type, adapter_type
        )

    @staticmethod
    def backup_model(
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        adapter_type: str = "simple",
        backup_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Backup model - static method for backward compatibility"""
        utils = ModelUtils()
        return utils.local_ops.backup_model(
            symbol, timeframe, model_type, adapter_type, backup_dir
        )
