# services/model_service.py

import json
from typing import Dict, Any, List, Optional

from ..facades import get_unified_facade
from ..schemas.enums import ModelType, TimeFrame
from ..schemas.base_schemas import ModelInfoResponse
from common.logger.logger_factory import LoggerFactory
from ..core.config import get_settings


class ModelService:
    """Service for managing models and providing model information"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("ModelService")
        self.settings = get_settings()
        self.model_facade = get_unified_facade()

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

                    # Check if model files exist (check for different possible model file patterns)
                    model_files = []
                    for pattern in ["model_state_dict.pt", "model.pt", "*.pt"]:
                        model_files.extend(list(model_dir.glob(pattern)))

                    if not model_files:
                        self.logger.debug(f"No model files found in {model_dir}")
                        continue

                    # Use consistent key generation
                    from ..utils.model_utils import ModelUtils
                    from ..schemas.enums import TimeFrame, ModelType

                    try:
                        symbol = metadata["symbol"]
                        timeframe = TimeFrame(metadata["timeframe"])
                        model_type = ModelType(metadata["model_type"])

                        utils = ModelUtils()
                        model_key = utils.generate_model_identifier(
                            symbol, timeframe, model_type
                        )

                        trained_models[model_key] = {
                            "symbol": symbol,
                            "timeframe": metadata["timeframe"],
                            "model_type": metadata["model_type"],
                            "created_at": metadata.get("created_at"),
                            "config": metadata.get("config", {}),
                            "model_path": str(model_files[0]),
                            "model_dir": str(model_dir),
                            "status": "available",
                        }

                        self.logger.debug(f"Found model: {model_key}")

                    except (ValueError, KeyError) as e:
                        self.logger.warning(f"Invalid metadata in {metadata_file}: {e}")
                        continue

                except Exception as e:
                    self.logger.warning(
                        f"Failed to read model metadata from {metadata_file}: {e}"
                    )
                    continue

        except Exception as e:
            self.logger.error(f"Failed to scan trained models: {e}")

        self.logger.info(f"Scanned {len(trained_models)} trained models")
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
            # Ensure symbol is string
            if hasattr(symbol, "value"):
                symbol = symbol.value

            from ..utils.model_utils import ModelUtils

            utils = ModelUtils()

            if model_type:
                return utils.model_exists(symbol, timeframe, model_type)
            else:
                # Check for any model type
                for mt in ModelType:
                    if utils.model_exists(symbol, timeframe, mt):
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
            # Ensure symbol is string
            if hasattr(symbol, "value"):
                symbol = symbol.value

            from ..utils.model_utils import ModelUtils

            utils = ModelUtils()

            model_dir = utils.generate_model_path(symbol, timeframe, model_type)
            metadata_file = model_dir / "metadata.json"

            if not metadata_file.exists():
                return None

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Check if model files exist
            model_files = list(model_dir.glob("model_state_dict.pt"))
            if not model_files:
                model_files = list(model_dir.glob("model.pt"))
            if not model_files:
                model_files = list(model_dir.glob("*.pt"))

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
            # Ensure symbol is string
            if hasattr(symbol, "value"):
                symbol = symbol.value

            from ..utils.model_utils import ModelUtils

            utils = ModelUtils()

            model_dir = utils.generate_model_path(symbol, timeframe, model_type)

            if not model_dir.exists():
                return False

            # Remove all files in the model directory
            import shutil

            shutil.rmtree(model_dir)

            self.logger.info(
                f"Deleted model: {utils.generate_model_identifier(symbol, timeframe, model_type)}"
            )
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
