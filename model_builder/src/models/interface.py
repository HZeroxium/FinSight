# models/interface.py

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List, Union
from torch.utils.data import DataLoader
from pathlib import Path

from ..core.config import Config
from common.logger.logger_factory import LoggerFactory


class ModelInterface(ABC, nn.Module):
    """
    Abstract base class for all prediction models with enhanced functionality

    This interface defines the contract that all models must implement,
    ensuring consistency across different model architectures.
    """

    def __init__(self, config: Config):
        """
        Initialize model with configuration

        Args:
            config: Model configuration containing hyperparameters and settings
        """
        super().__init__()
        self.config = config
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._training_step = 0
        self._validation_step = 0

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            **kwargs: Additional keyword arguments for specific models

        Returns:
            torch.Tensor: Model predictions
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get model name for logging and saving

        Returns:
            str: Unique model identifier
        """
        pass

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters

        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information dictionary

        Returns:
            dict: Model metadata including architecture details and parameter counts
        """
        return {
            "model_name": self.get_model_name(),
            "num_parameters": self.get_num_parameters(),
            "input_dim": getattr(self.config.model, "input_dim", None),
            "output_dim": getattr(self.config.model, "output_dim", None),
            "sequence_length": getattr(self.config.model, "sequence_length", None),
            "prediction_horizon": getattr(
                self.config.model, "prediction_horizon", None
            ),
            "model_size_mb": self.get_model_size(),
            "device": next(self.parameters()).device.type,
        }

    def get_model_size(self) -> float:
        """
        Calculate model size in megabytes

        Returns:
            float: Model size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Make predictions in inference mode

        Args:
            x: Input tensor
            **kwargs: Additional arguments for specific models

        Returns:
            torch.Tensor: Model predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, **kwargs)

    def predict_batch(
        self,
        data_loader: DataLoader,
        device: torch.device,
        return_attention: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List]
    ]:
        """
        Make predictions on a batch of data with optional attention weights

        Args:
            data_loader: DataLoader containing the data
            device: Device to run predictions on
            return_attention: Whether to return attention weights (if supported)

        Returns:
            tuple: (predictions, targets) or (predictions, targets, attention_weights)
        """
        self.eval()
        all_predictions = []
        all_targets = []
        all_attention = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                if return_attention and hasattr(self, "get_attention_weights"):
                    predictions = self.forward(batch_x)
                    attention_weights = self.get_attention_weights(batch_x)
                    all_attention.append(attention_weights)
                else:
                    predictions = self.forward(batch_x)

                all_predictions.append(predictions.cpu())
                all_targets.append(batch_y.cpu())

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        if return_attention and all_attention:
            return predictions, targets, all_attention
        return predictions, targets

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters by category with detailed breakdown

        Returns:
            dict: Detailed parameter counts
        """
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        layer_params = {}

        for name, param in self.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name not in layer_params:
                layer_params[layer_name] = 0
            layer_params[layer_name] += param.numel()

            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params,
            "non_trainable": total_params - trainable_params,
            "by_layer": layer_params,
        }

    def freeze_layers(self, layer_names: List[str]) -> None:
        """
        Freeze specific layers to prevent training

        Args:
            layer_names: List of layer names to freeze
        """
        frozen_count = 0
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                frozen_count += 1

        self.logger.info(f"Frozen {frozen_count} parameters in layers: {layer_names}")

    def unfreeze_layers(self, layer_names: List[str]) -> None:
        """
        Unfreeze specific layers to allow training

        Args:
            layer_names: List of layer names to unfreeze
        """
        unfrozen_count = 0
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                unfrozen_count += 1

        self.logger.info(
            f"Unfrozen {unfrozen_count} parameters in layers: {layer_names}"
        )

    def get_learning_rate_schedule(self) -> Optional[Dict[str, Any]]:
        """
        Get learning rate schedule information (to be overridden by specific models)

        Returns:
            dict: Learning rate schedule configuration or None
        """
        return None

    def get_regularization_info(self) -> Dict[str, Any]:
        """
        Get regularization information

        Returns:
            dict: Regularization settings and current values
        """
        dropout_layers = []
        batch_norm_layers = []

        for name, module in self.named_modules():
            if isinstance(
                module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)
            ):
                dropout_layers.append({"name": name, "p": module.p})
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                batch_norm_layers.append({"name": name, "momentum": module.momentum})

        return {
            "dropout_layers": dropout_layers,
            "batch_norm_layers": batch_norm_layers,
            "weight_decay": getattr(self.config.training, "weight_decay", 0.0),
        }

    def summary(self) -> str:
        """
        Generate a detailed model summary

        Returns:
            str: Formatted model summary
        """
        info = self.get_model_info()
        param_info = self.count_parameters()
        reg_info = self.get_regularization_info()

        summary_lines = [
            f"Model: {info['model_name']}",
            f"Total Parameters: {param_info['total']:,}",
            f"Trainable Parameters: {param_info['trainable']:,}",
            f"Model Size: {info['model_size_mb']:.2f} MB",
            f"Device: {info['device']}",
        ]

        if info.get("input_dim"):
            summary_lines.append(f"Input Dimension: {info['input_dim']}")
        if info.get("output_dim"):
            summary_lines.append(f"Output Dimension: {info['output_dim']}")
        if info.get("sequence_length"):
            summary_lines.append(f"Sequence Length: {info['sequence_length']}")

        if reg_info["dropout_layers"]:
            summary_lines.append(f"Dropout Layers: {len(reg_info['dropout_layers'])}")

        return "\n".join(summary_lines)

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Save model checkpoint with additional metadata

        Args:
            filepath: Path to save checkpoint
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            epoch: Current epoch number
            metrics: Training metrics to save
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": self.get_model_info(),
            "model_name": self.get_model_name(),
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint with metadata

        Args:
            filepath: Path to checkpoint file
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load model on

        Returns:
            dict: Checkpoint metadata
        """
        checkpoint = torch.load(filepath, map_location=device)

        self.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(f"Checkpoint loaded from {filepath}")

        return {
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics"),
            "model_config": checkpoint.get("model_config"),
        }
