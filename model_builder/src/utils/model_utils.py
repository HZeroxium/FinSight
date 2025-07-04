# utils/model_utils.py

"""
Model utilities for loading, saving, and managing PyTorch models
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import time
from datetime import datetime
import numpy as np

from ..core.config import Config
from ..models import create_model
from ..common.logger.logger_factory import LoggerFactory


class ModelUtils:
    """Comprehensive utilities for model operations including saving, loading, and analysis"""

    @staticmethod
    def _safe_torch_load(
        filepath: Union[str, Path], map_location=None
    ) -> Dict[str, Any]:
        """
        Safely load PyTorch checkpoint with fallback for different PyTorch versions

        Args:
            filepath: Path to checkpoint file
            map_location: Device mapping for loading

        Returns:
            Loaded checkpoint dictionary
        """
        try:
            # First try with weights_only=True (secure, PyTorch 2.6+ default)
            return torch.load(filepath, map_location=map_location, weights_only=True)
        except Exception as secure_error:
            try:
                # Fallback to weights_only=False for backward compatibility
                # This is safe for trusted checkpoints from our own system
                return torch.load(
                    filepath, map_location=map_location, weights_only=False
                )
            except Exception as fallback_error:
                # If both fail, try with safe globals for our custom classes
                try:
                    # Import and register safe globals for our custom classes
                    from ..core.config import ModelType

                    # Use safe globals context manager
                    with torch.serialization.safe_globals([ModelType]):
                        return torch.load(
                            filepath, map_location=map_location, weights_only=True
                        )
                except Exception as safe_globals_error:
                    raise RuntimeError(
                        f"Failed to load checkpoint with all methods:\n"
                        f"1. Secure load: {str(secure_error)}\n"
                        f"2. Fallback load: {str(fallback_error)}\n"
                        f"3. Safe globals load: {str(safe_globals_error)}"
                    )

    @staticmethod
    def save_model_checkpoint(
        model: nn.Module,
        filepath: Union[str, Path],
        config: Optional[Config] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save comprehensive model checkpoint with metadata

        Args:
            model: PyTorch model to save
            filepath: Path to save the checkpoint
            config: Model configuration
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            epoch: Current epoch number
            metrics: Training metrics
            additional_info: Additional metadata to save
        """
        logger = LoggerFactory.get_logger("model_utils")

        try:
            # Ensure directory exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Prepare checkpoint data with serializable format
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_class": model.__class__.__name__,
                "epoch": epoch if epoch is not None else 0,
                "metrics": metrics or {},
                "save_timestamp": datetime.now().isoformat(),
                "model_info": {
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "model_size_mb": sum(
                        p.numel() * p.element_size() for p in model.parameters()
                    )
                    / 1024
                    / 1024,
                },
            }

            # Add model-specific information
            if hasattr(model, "get_model_info"):
                try:
                    model_info = model.get_model_info()
                    # Filter out non-serializable values
                    serializable_info = {}
                    for key, value in model_info.items():
                        if isinstance(
                            value, (str, int, float, bool, list, dict, type(None))
                        ):
                            serializable_info[key] = value
                        elif hasattr(value, "__str__"):
                            serializable_info[key] = str(value)
                    checkpoint["model_info"].update(serializable_info)
                except Exception as e:
                    logger.warning(f"Could not get model info: {str(e)}")

            # Add optimizer state
            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                checkpoint["optimizer_class"] = optimizer.__class__.__name__

            # Add scheduler state
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
                checkpoint["scheduler_class"] = scheduler.__class__.__name__

            # Add configuration in serializable format
            if config is not None:
                try:
                    if hasattr(config, "to_dict"):
                        config_dict = config.to_dict()
                    else:
                        # Extract basic config info with enum handling
                        config_dict = {
                            "model_type": getattr(config.model, "model_type", None),
                            "input_dim": getattr(config.model, "input_dim", None),
                            "output_dim": getattr(config.model, "output_dim", None),
                            "sequence_length": getattr(
                                config.model, "sequence_length", None
                            ),
                            "d_model": getattr(config.model, "d_model", None),
                            "n_layers": getattr(config.model, "n_layers", None),
                            "n_heads": getattr(config.model, "n_heads", None),
                        }

                        # Convert enum to string if present
                        if hasattr(config_dict["model_type"], "value"):
                            config_dict["model_type"] = config_dict["model_type"].value
                        elif hasattr(config_dict["model_type"], "name"):
                            config_dict["model_type"] = config_dict["model_type"].name

                    checkpoint["config"] = config_dict
                except Exception as e:
                    logger.warning(f"Could not save config: {str(e)}")
                    checkpoint["config"] = {}

            # Add additional information
            if additional_info:
                checkpoint["additional_info"] = additional_info

            # Add training metadata
            checkpoint["training_metadata"] = {
                "device": str(next(model.parameters()).device),
                "pytorch_version": torch.__version__,
                "model_name": getattr(
                    model, "get_model_name", lambda: model.__class__.__name__
                )(),
            }

            # Save checkpoint - use weights_only=False for our own checkpoints
            torch.save(checkpoint, filepath)

            # Verify save was successful
            if filepath.exists():
                file_size = filepath.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"Checkpoint saved successfully to {filepath}")
                logger.info(f"File size: {file_size:.2f} MB")
            else:
                raise RuntimeError("Checkpoint file was not created")

        except Exception as e:
            logger.error(f"Failed to save model checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint save failed: {str(e)}")

    @staticmethod
    def load_model_checkpoint(
        filepath: Union[str, Path],
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load model checkpoint with comprehensive error handling

        Args:
            filepath: Path to checkpoint file
            model: Optional existing model to load state into
            device: Device to load model on
            strict: Whether to strictly enforce state dict matching

        Returns:
            Tuple of (loaded_model, checkpoint_metadata)
        """
        logger = LoggerFactory.get_logger("model_utils")

        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

            # Load checkpoint using safe loading method
            logger.info(f"Loading model checkpoint from {filepath}")
            checkpoint = ModelUtils._safe_torch_load(
                filepath, map_location=device or "cpu"
            )

            # Extract metadata with safe defaults
            metadata = {
                "epoch": checkpoint.get("epoch", 0),
                "metrics": checkpoint.get("metrics", {}),
                "save_timestamp": checkpoint.get("save_timestamp", "unknown"),
                "model_class": checkpoint.get("model_class", "unknown"),
                "model_info": checkpoint.get("model_info", {}),
                "config": checkpoint.get("config", {}),
                "training_metadata": checkpoint.get("training_metadata", {}),
            }

            # Load model state
            if model is not None:
                # Load into existing model
                model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
                logger.info("Model state dict loaded successfully")
            else:
                # Need to create model from checkpoint
                logger.warning("No model provided - returning state dict only")
                return None, metadata

            return model, metadata

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {filepath}: {str(e)}")
            raise RuntimeError(f"Checkpoint load failed: {str(e)}")

    @staticmethod
    def _infer_config_from_checkpoint(checkpoint: Dict[str, Any]) -> Config:
        """Infer configuration from checkpoint data with enhanced detection"""
        from ..core.config import Config, ModelConfig, DataConfig, ModelType

        # Start with default config
        config = Config()

        # Extract configuration from checkpoint
        saved_config = checkpoint.get("config", {})
        model_info = checkpoint.get("model_info", {})
        state_dict = checkpoint.get("model_state_dict", {})

        # Create new model config to avoid modifying defaults
        model_config = ModelConfig()

        # 1. Determine model type from class name or saved config
        model_class = checkpoint.get("model_class", "FinancialTransformer")
        if "model_type" in saved_config:
            model_type_value = saved_config["model_type"]
            if isinstance(model_type_value, str):
                # Convert string to enum
                if hasattr(ModelType, model_type_value.upper()):
                    model_config.model_type = getattr(
                        ModelType, model_type_value.upper()
                    )
                elif "lightweight" in model_type_value.lower():
                    model_config.model_type = ModelType.LIGHTWEIGHT_TRANSFORMER
                elif "hybrid" in model_type_value.lower():
                    model_config.model_type = ModelType.HYBRID_TRANSFORMER
                else:
                    model_config.model_type = ModelType.TRANSFORMER
        else:
            # Infer from class name
            if "Lightweight" in model_class:
                model_config.model_type = ModelType.LIGHTWEIGHT_TRANSFORMER
            elif "Hybrid" in model_class:
                model_config.model_type = ModelType.HYBRID_TRANSFORMER
            else:
                model_config.model_type = ModelType.TRANSFORMER

        # 2. Extract dimensions from saved config first, then model_info, then infer from state_dict
        dimension_params = [
            "input_dim",
            "d_model",
            "n_layers",
            "n_heads",
            "output_dim",
            "sequence_length",
            "d_ff",
        ]

        for param in dimension_params:
            if param in saved_config and saved_config[param] is not None:
                setattr(model_config, param, saved_config[param])
            elif param in model_info and model_info[param] is not None:
                setattr(model_config, param, model_info[param])

        # 3. Infer missing parameters from state_dict
        if state_dict:
            # Infer d_model from embedding or projection layers
            if model_config.d_model is None:
                for key, tensor in state_dict.items():
                    if "input_proj.weight" in key and len(tensor.shape) >= 2:
                        model_config.d_model = tensor.shape[0]  # Output dimension
                        break
                    elif "embedding" in key.lower() and len(tensor.shape) >= 2:
                        model_config.d_model = tensor.shape[-1]
                        break
                    elif (
                        "transformer_encoder.layers.0" in key
                        and "self_attn.out_proj.weight" in key
                    ):
                        model_config.d_model = tensor.shape[0]
                        break

            # Infer input_dim from input projection layer
            if model_config.input_dim is None:
                for key, tensor in state_dict.items():
                    if "input_proj.weight" in key and len(tensor.shape) >= 2:
                        model_config.input_dim = tensor.shape[1]  # Input dimension
                        break

            # Infer number of transformer layers
            if model_config.n_layers is None:
                max_layer_idx = -1
                for key in state_dict.keys():
                    if "transformer_encoder.layers." in key:
                        # Extract layer index
                        parts = key.split(".")
                        for i, part in enumerate(parts):
                            if part == "layers" and i + 1 < len(parts):
                                try:
                                    layer_idx = int(parts[i + 1])
                                    max_layer_idx = max(max_layer_idx, layer_idx)
                                except (ValueError, IndexError):
                                    continue
                if max_layer_idx >= 0:
                    model_config.n_layers = max_layer_idx + 1

            # Infer number of heads from attention layer
            if model_config.n_heads is None and model_config.d_model is not None:
                for key, tensor in state_dict.items():
                    if "self_attn.in_proj_weight" in key and len(tensor.shape) >= 2:
                        # in_proj_weight shape is [3*d_model, d_model] for multi-head attention
                        if tensor.shape[0] == 3 * model_config.d_model:
                            # Default to 8 heads, but try to infer if possible
                            if model_config.d_model % 8 == 0:
                                model_config.n_heads = 8
                            elif model_config.d_model % 4 == 0:
                                model_config.n_heads = 4
                            else:
                                model_config.n_heads = 1
                        break

            # Infer d_ff from feed-forward layers
            if model_config.d_ff is None:
                for key, tensor in state_dict.items():
                    if "linear1.weight" in key and len(tensor.shape) >= 2:
                        model_config.d_ff = tensor.shape[
                            0
                        ]  # Output of first linear layer
                        break

        # 4. Set reasonable defaults for missing parameters
        if model_config.d_model is None:
            model_config.d_model = 256  # Default from training
        if model_config.n_layers is None:
            model_config.n_layers = 3  # Default from training
        if model_config.n_heads is None:
            model_config.n_heads = 8
        if model_config.d_ff is None:
            model_config.d_ff = 1024
        if model_config.input_dim is None:
            model_config.input_dim = 5
        if model_config.output_dim is None:
            model_config.output_dim = 1
        if model_config.sequence_length is None:
            model_config.sequence_length = 60

        # 5. Update the main config
        config.model = model_config

        # 6. Copy other training parameters if available
        training_params = ["dropout", "batch_size", "learning_rate"]
        for param in training_params:
            if param in saved_config:
                try:
                    setattr(config.model, param, saved_config[param])
                except:
                    pass  # Skip if attribute doesn't exist or can't be set

        return config

    @staticmethod
    def load_model_for_inference(
        checkpoint_path: Union[str, Path],
        config: Optional[Config] = None,
        device: Optional[torch.device] = None,
        force_cpu: bool = False,
    ) -> nn.Module:
        """
        Load model specifically for inference with automatic configuration detection

        Args:
            checkpoint_path: Path to the checkpoint file
            config: Optional configuration (will try to infer if not provided)
            device: Target device
            force_cpu: Force model to CPU regardless of availability

        Returns:
            Loaded model ready for inference
        """
        logger = LoggerFactory.get_logger("model_utils")

        try:
            checkpoint_path = Path(checkpoint_path)

            # Load checkpoint to inspect configuration using safe loading
            checkpoint = ModelUtils._safe_torch_load(
                checkpoint_path, map_location="cpu"
            )

            # Extract or infer configuration
            if config is None:
                config = ModelUtils._infer_config_from_checkpoint(checkpoint)
                logger.info(
                    f"Inferred config - d_model: {config.model.d_model}, "
                    f"n_layers: {config.model.n_layers}, "
                    f"n_heads: {config.model.n_heads}, "
                    f"input_dim: {config.model.input_dim}"
                )

            # Determine device
            if force_cpu:
                device = torch.device("cpu")
            elif device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create model using inferred configuration
            model_class = checkpoint.get("model_class", "FinancialTransformer")

            # Map model class to model type string
            if "Lightweight" in model_class:
                model_type = "lightweight_transformer"
            elif "Hybrid" in model_class:
                model_type = "hybrid_transformer"
            else:
                model_type = "transformer"

            # Create model
            model = create_model(model_type, config)

            # Load state dict with strict=False to handle potential mismatches gracefully
            try:
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                logger.info("Model state dict loaded successfully (strict=True)")
            except RuntimeError as e:
                logger.warning(
                    f"Strict loading failed, trying with strict=False: {str(e)}"
                )
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
                if missing_keys:
                    logger.warning(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {unexpected_keys}")
                logger.info("Model state dict loaded successfully (strict=False)")

            # Move to device and set to eval mode
            model.to(device)
            model.eval()

            logger.info(f"Model loaded: {model_class}")
            logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"Device: {device}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model for inference: {str(e)}")
            raise

    @staticmethod
    def list_available_checkpoints(
        checkpoint_dir: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """
        List all available model checkpoints with metadata

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            List of checkpoint information dictionaries
        """
        logger = LoggerFactory.get_logger("model_utils")

        checkpoint_dir = Path(checkpoint_dir)
        checkpoints = []

        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return checkpoints

        # Find all .pt and .pth files
        for pattern in ["*.pt", "*.pth"]:
            for checkpoint_path in checkpoint_dir.glob(pattern):
                try:
                    # Load checkpoint metadata only using safe loading
                    checkpoint = ModelUtils._safe_torch_load(
                        checkpoint_path, map_location="cpu"
                    )

                    info = {
                        "path": str(checkpoint_path),
                        "filename": checkpoint_path.name,
                        "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(
                            checkpoint_path.stat().st_mtime
                        ).isoformat(),
                        "epoch": checkpoint.get("epoch", "unknown"),
                        "model_class": checkpoint.get("model_class", "unknown"),
                        "metrics": checkpoint.get("metrics", {}),
                        "save_timestamp": checkpoint.get("save_timestamp", "unknown"),
                    }

                    # Extract key metrics if available
                    metrics = checkpoint.get("metrics", {})
                    if metrics:
                        info["best_val_loss"] = metrics.get("val_loss", None)
                        info["best_val_rmse"] = metrics.get("val_rmse", None)

                    checkpoints.append(info)

                except Exception as e:
                    logger.warning(
                        f"Could not read checkpoint {checkpoint_path.name}: {str(e)}"
                    )
                    # Add basic info even if we can't read the checkpoint
                    checkpoints.append(
                        {
                            "path": str(checkpoint_path),
                            "filename": checkpoint_path.name,
                            "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
                            "modified": datetime.fromtimestamp(
                                checkpoint_path.stat().st_mtime
                            ).isoformat(),
                            "error": str(e),
                            "epoch": "unknown",
                            "model_class": "unknown",
                            "metrics": {},
                            "save_timestamp": "unknown",
                        }
                    )

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x.get("modified", ""), reverse=True)

        logger.info(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}")
        return checkpoints

    @staticmethod
    def compare_model_architectures(models: Dict[str, nn.Module]) -> Dict[str, Any]:
        """
        Compare multiple model architectures

        Args:
            models: Dictionary of model name to model instance

        Returns:
            Comparison results
        """
        comparison = {}

        for name, model in models.items():
            info = {
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "layers": len(list(model.named_modules())),
                "size_mb": sum(p.numel() * p.element_size() for p in model.parameters())
                / (1024 * 1024),
            }

            # Add model-specific info if available
            if hasattr(model, "get_model_info"):
                try:
                    model_info = model.get_model_info()
                    info.update(model_info)
                except:
                    pass

            comparison[name] = info

        return comparison

    @staticmethod
    def estimate_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """
        Estimate model FLOPs (Floating Point Operations)

        Args:
            model: PyTorch model
            input_shape: Input tensor shape (batch_size, seq_len, features)

        Returns:
            Estimated FLOPs
        """
        # This is a simplified FLOP estimation
        # For more accurate measurements, consider using libraries like thop or ptflops

        total_params = sum(p.numel() for p in model.parameters())

        # Rough estimation: 2 * params * input_elements
        # This is very approximate and model-dependent
        input_elements = np.prod(input_shape)
        estimated_flops = 2 * total_params * input_elements

        return estimated_flops

    @staticmethod
    def profile_model_inference(
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Profile model inference time

        Args:
            model: Model to profile
            input_tensor: Sample input tensor
            num_runs: Number of inference runs for timing
            warmup_runs: Number of warmup runs

        Returns:
            Timing statistics
        """
        logger = LoggerFactory.get_logger("model_utils")

        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)

        # Synchronize if using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time inference
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms

        times = np.array(times)

        stats = {
            "mean_time_ms": float(np.mean(times)),
            "std_time_ms": float(np.std(times)),
            "min_time_ms": float(np.min(times)),
            "max_time_ms": float(np.max(times)),
            "median_time_ms": float(np.median(times)),
            "throughput_samples_per_sec": float(
                input_tensor.shape[0] / (np.mean(times) / 1000)
            ),
        }

        logger.info(f"Inference profiling results:")
        logger.info(f"  Mean time: {stats['mean_time_ms']:.3f} ms")
        logger.info(
            f"  Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec"
        )

        return stats

    @staticmethod
    def export_model_to_onnx(
        model: nn.Module,
        input_tensor: torch.Tensor,
        output_path: Union[str, Path],
        opset_version: int = 11,
    ) -> bool:
        """
        Export PyTorch model to ONNX format

        Args:
            model: PyTorch model to export
            input_tensor: Sample input tensor
            output_path: Path to save ONNX model
            opset_version: ONNX opset version

        Returns:
            Success status
        """
        logger = LoggerFactory.get_logger("model_utils")

        try:
            import torch.onnx

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            model.eval()
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    input_tensor,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )

            logger.info(f"Model exported to ONNX: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {str(e)}")
            return False

    @staticmethod
    def validate_checkpoint_integrity(
        checkpoint_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Validate checkpoint file integrity and completeness

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Validation results
        """
        logger = LoggerFactory.get_logger("model_utils")

        validation_results = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "info": {},
        }

        try:
            checkpoint_path = Path(checkpoint_path)

            # Check file exists
            if not checkpoint_path.exists():
                validation_results["errors"].append("Checkpoint file does not exist")
                return validation_results

            # Check file size
            file_size = checkpoint_path.stat().st_size
            if file_size == 0:
                validation_results["errors"].append("Checkpoint file is empty")
                return validation_results

            validation_results["info"]["file_size_mb"] = file_size / (1024 * 1024)

            # Try to load checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            except Exception as e:
                validation_results["errors"].append(f"Cannot load checkpoint: {str(e)}")
                return validation_results

            # Check required fields
            required_fields = ["model_state_dict"]
            for field in required_fields:
                if field not in checkpoint:
                    validation_results["errors"].append(
                        f"Missing required field: {field}"
                    )

            # Check optional but important fields
            important_fields = ["epoch", "metrics", "model_class", "config"]
            for field in important_fields:
                if field not in checkpoint:
                    validation_results["warnings"].append(
                        f"Missing optional field: {field}"
                    )

            # Validate state dict
            state_dict = checkpoint.get("model_state_dict", {})
            if not state_dict:
                validation_results["errors"].append("Empty model state dict")
            else:
                validation_results["info"]["state_dict_keys"] = len(state_dict)
                validation_results["info"]["total_parameters"] = sum(
                    tensor.numel() for tensor in state_dict.values()
                )

            # Check for metadata
            validation_results["info"]["has_metrics"] = "metrics" in checkpoint
            validation_results["info"]["has_config"] = "config" in checkpoint
            validation_results["info"]["model_class"] = checkpoint.get(
                "model_class", "unknown"
            )
            validation_results["info"]["epoch"] = checkpoint.get("epoch", "unknown")

            # If we get here without errors, checkpoint is valid
            if not validation_results["errors"]:
                validation_results["is_valid"] = True

            logger.info(f"Checkpoint validation completed for {checkpoint_path.name}")
            logger.info(f"Valid: {validation_results['is_valid']}")
            if validation_results["errors"]:
                logger.warning(f"Errors: {validation_results['errors']}")

        except Exception as e:
            validation_results["errors"].append(f"Validation failed: {str(e)}")
            logger.error(f"Checkpoint validation failed: {str(e)}")

        return validation_results
