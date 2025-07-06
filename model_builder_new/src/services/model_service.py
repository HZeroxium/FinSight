# services/model_service.py

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from ..models.model_facade import ModelFacade
from ..schemas.enums import ModelType, TimeFrame
from ..schemas.base_schemas import ModelInfoResponse
from ..logger.logger_factory import LoggerFactory
from ..core.config import get_settings


class ModelService:
    """Service for managing models and providing model information"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("ModelService")
        self.settings = get_settings()
        self.model_facade = ModelFacade()

    def get_model_info(self) -> ModelInfoResponse:
        """Get comprehensive information about models"""

        try:
            # Get available model types
            available_models = [
                "patchtst",
                "patchtsmixer",
                "pytorch_lightning_transformer",
            ]

            # Get trained models information
            trained_models = self._scan_trained_models()

            # Get supported timeframes
            supported_timeframes = [tf.value for tf in TimeFrame]

            # Get available symbols from datasets
            supported_symbols = self._get_available_symbols()

            return ModelInfoResponse(
                available_models=available_models,
                trained_models=trained_models,
                supported_timeframes=supported_timeframes,
                supported_symbols=supported_symbols,
            )

        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return ModelInfoResponse(
                available_models=[],
                trained_models={},
                supported_timeframes=[],
                supported_symbols=[],
            )

    def _scan_trained_models(self) -> Dict[str, Dict[str, Any]]:
        """Scan and catalog all trained models"""

        trained_models = {}

        try:
            if not self.settings.models_dir.exists():
                return trained_models

            for model_dir in self.settings.models_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                metadata_file = model_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    # Check if model file exists
                    model_files = list(model_dir.glob("model_*.pkl"))
                    if not model_files:
                        continue

                    model_key = f"{metadata['symbol']}_{metadata['timeframe']}_{metadata['model_type']}"

                    trained_models[model_key] = {
                        "symbol": metadata["symbol"],
                        "timeframe": metadata["timeframe"],
                        "model_type": metadata["model_type"],
                        "training_id": metadata["training_id"],
                        "created_at": metadata["created_at"],
                        "config": metadata["config"],
                        "model_path": str(model_files[0]),
                        "status": "available",
                    }

                except Exception as e:
                    self.logger.warning(
                        f"Failed to read model metadata from {metadata_file}: {e}"
                    )
                    continue

        except Exception as e:
            self.logger.error(f"Failed to scan trained models: {e}")

        return trained_models

    def _get_available_symbols(self) -> List[str]:
        """Get list of available symbols from dataset files"""

        symbols = set()

        try:
            # Scan data directory for CSV files
            if hasattr(self.settings, "data_dir") and self.settings.data_dir.exists():
                for data_file in self.settings.data_dir.glob("*.csv"):
                    # Extract symbol from filename (assuming format: SYMBOL_TIMEFRAME.csv)
                    filename = data_file.stem
                    parts = filename.split("_")
                    if len(parts) >= 2:
                        symbol = parts[0]
                        symbols.add(symbol)

        except Exception as e:
            self.logger.error(f"Failed to scan for available symbols: {e}")

        return sorted(list(symbols))

    def check_model_exists(
        self, symbol: str, timeframe: TimeFrame, model_type: Optional[ModelType] = None
    ) -> bool:
        """Check if a trained model exists for given parameters"""

        try:
            if model_type:
                model_pattern = f"{symbol}_{timeframe}_{model_type}"
                model_dirs = list(self.settings.models_dir.glob(model_pattern))
            else:
                model_pattern = f"{symbol}_{timeframe}_*"
                model_dirs = list(self.settings.models_dir.glob(model_pattern))

            for model_dir in model_dirs:
                if model_dir.is_dir():
                    # Check if model file exists
                    model_files = list(model_dir.glob("model_*.pkl"))
                    if model_files and (model_dir / "metadata.json").exists():
                        return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to check model existence: {e}")
            return False

    def get_model_details(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""

        try:
            model_dir = self.settings.models_dir / f"{symbol}_{timeframe}_{model_type}"
            metadata_file = model_dir / "metadata.json"

            if not metadata_file.exists():
                return None

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Check if model file exists
            model_files = list(model_dir.glob("model_*.pkl"))
            if not model_files:
                return None

            return {
                **metadata,
                "model_path": str(model_files[0]),
                "model_size": model_files[0].stat().st_size,
                "status": "available",
            }

        except Exception as e:
            self.logger.error(f"Failed to get model details: {e}")
            return None

    def delete_model(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> bool:
        """Delete a trained model"""

        try:
            model_dir = self.settings.models_dir / f"{symbol}_{timeframe}_{model_type}"

            if not model_dir.exists():
                return False

            # Remove all files in the model directory
            for file_path in model_dir.iterdir():
                file_path.unlink()

            # Remove the directory
            model_dir.rmdir()

            self.logger.info(f"Deleted model: {symbol}_{timeframe}_{model_type}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete model: {e}")
            return False

    def get_model_performance(
        self, symbol: str, timeframe: TimeFrame, model_type: ModelType
    ) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific model"""

        try:
            model_dir = self.settings.models_dir / f"{symbol}_{timeframe}_{model_type}"
            performance_file = model_dir / "performance.json"

            if not performance_file.exists():
                return None

            with open(performance_file, "r") as f:
                performance_data = json.load(f)

            return performance_data

        except Exception as e:
            self.logger.error(f"Failed to get model performance: {e}")
            return None
