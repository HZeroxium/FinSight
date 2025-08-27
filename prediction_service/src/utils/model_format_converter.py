# utils/model_format_converter.py

"""
Model format converter for multiple serving adapters.

This module handles conversion of trained models from the base format
to specific formats required by different serving adapters.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from common.logger.logger_factory import LoggerFactory

from ..core.config import get_settings
from ..core.constants import FacadeConstants
from ..interfaces.model_interface import ITimeSeriesModel
from ..schemas.enums import ModelType, TimeFrame


class ModelFormatConverter:
    """Converter for model formats across different serving adapters."""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("ModelFormatConverter")
        self.settings = get_settings()

    def convert_model_for_all_adapters(
        self,
        model: ITimeSeriesModel,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        source_path: Path,
        target_adapters: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """
        Convert trained model to all supported adapter formats.

        Args:
            model: Trained model instance
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            source_path: Path to source model (simple format)
            target_adapters: List of target adapters (defaults to all)

        Returns:
            Dict mapping adapter type to conversion success status
        """
        if target_adapters is None:
            target_adapters = FacadeConstants.SUPPORTED_ADAPTERS

        results = {}

        for adapter_type in target_adapters:
            if adapter_type == FacadeConstants.ADAPTER_SIMPLE:
                # Simple format is already saved during training
                results[adapter_type] = True
                continue

            try:
                success = self._convert_to_adapter_format(
                    model=model,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    source_path=source_path,
                    adapter_type=adapter_type,
                )
                results[adapter_type] = success

                if success:
                    self.logger.info(
                        f"Successfully converted model to {adapter_type} format: "
                        f"{symbol}_{timeframe}_{model_type}"
                    )
                else:
                    self.logger.warning(
                        f"Failed to convert model to {adapter_type} format: "
                        f"{symbol}_{timeframe}_{model_type}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error converting model to {adapter_type} format: {e}"
                )
                results[adapter_type] = False

        return results

    def _convert_to_adapter_format(
        self,
        model: ITimeSeriesModel,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        source_path: Path,
        adapter_type: str,
    ) -> bool:
        """Convert model to specific adapter format."""

        if adapter_type == FacadeConstants.ADAPTER_TORCHSCRIPT:
            return self._convert_to_torchscript(
                model, symbol, timeframe, model_type, source_path
            )
        elif adapter_type == FacadeConstants.ADAPTER_TORCHSERVE:
            return self._convert_to_torchserve(
                model, symbol, timeframe, model_type, source_path
            )
        elif adapter_type == FacadeConstants.ADAPTER_TRITON:
            return self._convert_to_triton(
                model, symbol, timeframe, model_type, source_path
            )
        else:
            self.logger.warning(f"Unsupported adapter type: {adapter_type}")
            return False

    def _convert_to_torchscript(
        self,
        model: ITimeSeriesModel,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        source_path: Path,
    ) -> bool:
        """Convert model to TorchScript format."""
        try:
            from .model_utils import ModelUtils

            model_utils = ModelUtils()
            target_dir = model_utils.generate_model_path(
                symbol, timeframe, model_type, FacadeConstants.ADAPTER_TORCHSCRIPT
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy all files from source path first
            self._copy_model_files(source_path, target_dir)

            # Convert PyTorch model to TorchScript if possible
            if hasattr(model, "model") and model.model is not None:
                try:
                    # Ensure model is on CPU for tracing (avoid device mismatch)
                    device_before = next(model.model.parameters()).device
                    model.model.cpu()
                    model.model.eval()

                    # Create example input for tracing
                    example_input = self._create_example_input(model)

                    if example_input is not None:
                        # Ensure input is also on CPU
                        if isinstance(example_input, torch.Tensor):
                            example_input = example_input.cpu()
                        elif isinstance(example_input, (list, tuple)):
                            example_input = [
                                x.cpu() if isinstance(x, torch.Tensor) else x
                                for x in example_input
                            ]

                        # Trace the model with strict=False to handle dict outputs
                        with torch.no_grad():
                            scripted_model = torch.jit.trace(
                                model.model,
                                example_input,
                                strict=False,  # Allow dict outputs from models like PatchTSMixer
                            )

                        # Save TorchScript model with multiple naming conventions for compatibility
                        scripted_model.save(
                            target_dir / "model_torchscript.pt"
                        )  # Standard TorchScript name
                        scripted_model.save(
                            target_dir / "scripted_model.pt"
                        )  # Alternative name
                        scripted_model.save(
                            target_dir / "model.pt"
                        )  # Adapter expected name

                        # Restore original device
                        model.model.to(device_before)

                        self.logger.info(
                            f"Successfully created TorchScript model: {target_dir}"
                        )
                    else:
                        self.logger.warning(
                            "Could not create example input for TorchScript tracing. "
                            "Using model state dict only."
                        )

                except Exception as trace_error:
                    self.logger.warning(
                        f"TorchScript tracing failed: {trace_error}. "
                        "Model will be available as state dict only."
                    )

                    # Restore model to original device even if tracing failed
                    try:
                        if "device_before" in locals():
                            model.model.to(device_before)
                    except:
                        pass

            # Ensure compatibility files exist for TorchScript adapter
            self._ensure_torchscript_compatibility(target_dir)

            return True

        except Exception as e:
            self.logger.error(f"Failed to convert to TorchScript: {e}")
            return False

    def _convert_to_torchserve(
        self,
        model: ITimeSeriesModel,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        source_path: Path,
    ) -> bool:
        """Convert model to TorchServe format."""
        try:
            from .model_utils import ModelUtils

            model_utils = ModelUtils()
            target_dir = model_utils.generate_model_path(
                symbol, timeframe, model_type, FacadeConstants.ADAPTER_TORCHSERVE
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy all files from source path
            self._copy_model_files(source_path, target_dir)

            # Create TorchServe specific structure and files
            self._create_torchserve_structure(
                model, symbol, timeframe, model_type, target_dir
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to convert to TorchServe: {e}")
            return False

    def _convert_to_triton(
        self,
        model: ITimeSeriesModel,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        source_path: Path,
    ) -> bool:
        """Convert model to Triton format."""
        try:
            from .model_utils import ModelUtils

            model_utils = ModelUtils()
            target_dir = model_utils.generate_model_path(
                symbol, timeframe, model_type, FacadeConstants.ADAPTER_TRITON
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy all files from source path
            self._copy_model_files(source_path, target_dir)

            # Create Triton specific structure and config
            self._create_triton_structure(
                model, symbol, timeframe, model_type, target_dir
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to convert to Triton: {e}")
            return False

    def _copy_model_files(self, source_path: Path, target_path: Path) -> None:
        """Copy model files from source to target directory."""
        try:
            if not source_path.exists():
                raise FileNotFoundError(f"Source path does not exist: {source_path}")

            # Copy all files from source to target
            for item in source_path.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(source_path)
                    target_file = target_path / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target_file)

        except Exception as e:
            self.logger.error(f"Failed to copy model files: {e}")
            raise

    def _ensure_torchscript_compatibility(self, target_dir: Path) -> None:
        """Ensure TorchScript adapter can find expected files."""
        try:
            # Create alternative file names that TorchScript adapter expects
            state_dict_path = target_dir / "model_state_dict.pt"

            if state_dict_path.exists():
                # Create additional copies with expected names
                expected_names = ["pytorch_model.bin"]

                for expected_name in expected_names:
                    expected_path = target_dir / expected_name
                    if not expected_path.exists():
                        shutil.copy2(state_dict_path, expected_path)
                        self.logger.debug(
                            f"Created compatibility file: {expected_path}"
                        )

        except Exception as e:
            self.logger.warning(
                f"Failed to create TorchScript compatibility files: {e}"
            )

    def _create_example_input(self, model: ITimeSeriesModel) -> Optional[torch.Tensor]:
        """Create example input tensor for TorchScript tracing."""
        try:
            # Create example input based on actual feature engineering output
            context_length = getattr(model, "context_length", 64)

            # Use actual feature engineering to determine correct number of features
            if (
                hasattr(model, "feature_engineering")
                and model.feature_engineering is not None
            ):
                try:
                    # Get the actual feature names from fitted feature engineering
                    feature_names = model.feature_engineering.get_feature_names()
                    num_features = len(feature_names)
                    self.logger.info(
                        f"Using feature engineering output: {num_features} features"
                    )
                except Exception as fe_error:
                    self.logger.warning(
                        f"Could not get feature names from feature engineering: {fe_error}"
                    )
                    # Fallback to basic feature columns
                    num_features = len(getattr(model, "feature_columns", ["close"]))
            else:
                # Fallback to basic feature columns
                num_features = len(getattr(model, "feature_columns", ["close"]))
                self.logger.warning(
                    f"No feature engineering found, using basic features: {num_features}"
                )

            # Create example tensor with proper shape
            example_input = torch.randn(1, context_length, num_features)

            self.logger.info(f"Created example input with shape: {example_input.shape}")
            return example_input

        except Exception as e:
            self.logger.warning(f"Failed to create example input: {e}")
            return None

    def _create_torchserve_structure(
        self,
        model: ITimeSeriesModel,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        target_dir: Path,
    ) -> None:
        """Create TorchServe specific structure and files."""
        try:
            # Create model handler file for TorchServe
            handler_content = self._generate_torchserve_handler(model)
            with open(target_dir / "handler.py", "w") as f:
                f.write(handler_content)

            # Create model archive configuration
            model_name = f"{symbol}_{timeframe.value}_{model_type.value}"
            archive_config = {
                "model_name": model_name,
                "version": "1.0",
                "serialized_file": "model_state_dict.pt",
                "handler": "handler.py",
                "requirements_file": "requirements.txt",
            }

            with open(target_dir / "archive_config.json", "w") as f:
                json.dump(archive_config, f, indent=2)

            # Create requirements file
            requirements = [
                "torch>=1.9.0",
                "numpy",
                "pandas",
                "transformers",
                "scikit-learn",
            ]

            with open(target_dir / "requirements.txt", "w") as f:
                f.write("\n".join(requirements))

        except Exception as e:
            self.logger.error(f"Failed to create TorchServe structure: {e}")
            raise

    def _create_triton_structure(
        self,
        model: ITimeSeriesModel,
        symbol: str,
        timeframe: TimeFrame,
        model_type: ModelType,
        target_dir: Path,
    ) -> None:
        """Create Triton Inference Server specific structure."""
        try:
            # Create model repository structure
            # Clean model type name for filesystem compatibility
            clean_model_type = model_type.value.replace("/", "_").replace("-", "_")
            model_name = f"{symbol}_{timeframe.value}_{clean_model_type}"

            # Create model repository directory inside target_dir
            model_repo_dir = target_dir / model_name
            model_repo_dir.mkdir(parents=True, exist_ok=True)

            # Create version directory
            version_dir = model_repo_dir / "1"
            version_dir.mkdir(parents=True, exist_ok=True)

            # Copy model files to version directory (but not nested directories)
            for file_path in target_dir.iterdir():
                if file_path.is_file() and file_path.suffix in [
                    ".pt",
                    ".pkl",
                    ".json",
                    ".bin",
                ]:
                    shutil.copy2(file_path, version_dir)

            # Create Triton model configuration
            config_content = self._generate_triton_config(model, model_name)
            with open(model_repo_dir / "config.pbtxt", "w") as f:
                f.write(config_content)

            self.logger.info(f"Created Triton model repository at: {model_repo_dir}")

        except Exception as e:
            self.logger.error(f"Failed to create Triton structure: {e}")
            raise

    def _generate_torchserve_handler(self, model: ITimeSeriesModel) -> str:
        """Generate TorchServe handler code."""
        return '''
import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from ts.torch_handler.base_handler import BaseHandler

class TimeSeriesHandler(BaseHandler):
    """Custom handler for time series prediction models."""
    
    def __init__(self):
        super().__init__()
        self.feature_scaler = None
        self.target_scaler = None
        
    def initialize(self, context):
        """Initialize the handler."""
        super().initialize(context)
        
        # Load scalers if available
        import pickle
        from pathlib import Path
        
        model_dir = Path(context.system_properties.get("model_dir"))
        
        feature_scaler_path = model_dir / "feature_scaler.pkl"
        if feature_scaler_path.exists():
            with open(feature_scaler_path, "rb") as f:
                self.feature_scaler = pickle.load(f)
                
        target_scaler_path = model_dir / "target_scaler.pkl"
        if target_scaler_path.exists():
            with open(target_scaler_path, "rb") as f:
                self.target_scaler = pickle.load(f)
    
    def preprocess(self, data):
        """Preprocess input data."""
        # Implement preprocessing logic here
        return data
    
    def inference(self, data):
        """Run inference on the model."""
        # Implement inference logic here
        with torch.no_grad():
            predictions = self.model(data)
        return predictions
    
    def postprocess(self, data):
        """Postprocess model outputs."""
        # Implement postprocessing logic here
        return data
'''

    def _generate_triton_config(self, model: ITimeSeriesModel, model_name: str) -> str:
        """Generate Triton model configuration."""
        context_length = getattr(model, "context_length", 64)
        num_features = len(getattr(model, "feature_columns", ["close"]))
        prediction_length = getattr(model, "prediction_length", 1)

        return f"""
name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {{
    name: "input_data"
    data_type: TYPE_FP32
    dims: [ {context_length}, {num_features} ]
  }}
]
output [
  {{
    name: "predictions"
    data_type: TYPE_FP32
    dims: [ {prediction_length} ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
dynamic_batching {{
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}}
"""
