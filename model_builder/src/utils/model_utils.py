# utils/model_utils.py

"""
Model utilities for loading, saving, and managing PyTorch models
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import json
import time
import pickle
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

            # Determine device
            if force_cpu:
                device = torch.device("cpu")
            elif device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create model
            model_class = checkpoint.get("model_class", "FinancialTransformer")

            # Try to determine model type from class name
            if "Lightweight" in model_class:
                model_type = "lightweight_transformer"
            elif "Hybrid" in model_class:
                model_type = "hybrid_transformer"
            else:
                model_type = "transformer"

            # Update config with inferred model type
            if hasattr(config, "model") and hasattr(config.model, "model_type"):
                from ..core.config import ModelType

                if hasattr(ModelType, model_type.upper()):
                    config.model.model_type = getattr(ModelType, model_type.upper())

            # Create model
            model = create_model(model_type, config)

            # Load state dict
            model.load_state_dict(checkpoint["model_state_dict"])

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
    def _infer_config_from_checkpoint(checkpoint: Dict[str, Any]) -> Config:
        """Infer configuration from checkpoint data"""
        from ..core.config import Config, create_development_config

        # Start with default config
        config = create_development_config()

        # Extract configuration from checkpoint
        saved_config = checkpoint.get("config", {})
        model_info = checkpoint.get("model_info", {})

        # Update model dimensions
        if "input_dim" in saved_config:
            config.model.input_dim = saved_config["input_dim"]
        elif "input_dim" in model_info:
            config.model.input_dim = model_info["input_dim"]
        else:
            # Try to infer from state dict
            state_dict = checkpoint.get("model_state_dict", {})
            for key, tensor in state_dict.items():
                if "embedding" in key.lower() and len(tensor.shape) >= 2:
                    config.model.input_dim = tensor.shape[-1]
                    break

        # Update other model parameters
        for param in [
            "d_model",
            "n_layers",
            "n_heads",
            "output_dim",
            "sequence_length",
        ]:
            if param in saved_config:
                setattr(config.model, param, saved_config[param])
            elif param in model_info:
                setattr(config.model, param, model_info[param])

        # Infer some parameters from state dict if needed
        state_dict = checkpoint.get("model_state_dict", {})

        # Infer d_model from transformer layers
        for key, tensor in state_dict.items():
            if "transformer" in key and "weight" in key and len(tensor.shape) >= 2:
                if config.model.d_model is None:
                    config.model.d_model = tensor.shape[-1]
                break

        # Infer number of layers
        if config.model.n_layers is None:
            layer_count = 0
            for key in state_dict.keys():
                if "transformer_layers" in key:
                    layer_nums = [int(s) for s in key.split(".") if s.isdigit()]
                    if layer_nums:
                        layer_count = max(layer_count, max(layer_nums) + 1)
            if layer_count > 0:
                config.model.n_layers = layer_count

        return config

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
