# train.py

import os
import json
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
)
from datasets import DatasetDict
from peft_config import PEFTConfig
from strategies import get_strategy
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


def build_peft_model(
    base_model: torch.nn.Module, peft_cfg: PEFTConfig
) -> torch.nn.Module:
    """
    Wrap the base HuggingFace model with the chosen PEFT method.

    Args:
        base_model: The base HuggingFace model to wrap
        peft_cfg: PEFT configuration object

    Returns:
        PEFT-wrapped model
    """
    logger.info(f"Building PEFT model with method: {peft_cfg.peft_method}")

    # Choose PEFT config class
    if peft_cfg.peft_method == "lora":
        peft_conf = LoraConfig(
            r=peft_cfg.lora_r,
            lora_alpha=peft_cfg.lora_alpha,
            lora_dropout=peft_cfg.lora_dropout,
            target_modules=peft_cfg.target_modules,
        )
    elif peft_cfg.peft_method == "adapter":
        peft_conf = LoraConfig(
            r=peft_cfg.lora_r,
            lora_alpha=peft_cfg.lora_alpha,
            lora_dropout=peft_cfg.lora_dropout,
            target_modules=peft_cfg.target_modules,
        )
    elif peft_cfg.peft_method == "prefix_tuning":
        peft_conf = PrefixTuningConfig(
            prefix_length=peft_cfg.prefix_length,
            prefix_dropout=peft_cfg.prefix_dropout,
            target_modules=peft_cfg.target_modules,
        )
    else:
        raise ValueError(f"Unsupported PEFT method: {peft_cfg.peft_method}")

    # Initialize and wrap
    peft_model = get_peft_model(base_model, peft_conf)

    # Ensure all parameters requiring gradients are properly set
    peft_model.train()
    for param in peft_model.parameters():
        param.requires_grad = True

    logger.info(
        f"PEFT model created successfully. Trainable parameters: {peft_model.num_parameters()}"
    )
    return peft_model


def collate_fn(batch):
    """
    Custom data collator for time series data.

    Args:
        batch: List of examples from the dataset

    Returns:
        Dictionary with properly formatted tensors
    """
    # Extract past_values and future_values from the batch
    past_values = torch.tensor(
        [example["past_values"] for example in batch], dtype=torch.float32
    )
    future_values = torch.tensor(
        [example["future_values"] for example in batch], dtype=torch.float32
    )

    # Add feature dimension: (batch_size, sequence_length) -> (batch_size, sequence_length, num_features)
    past_values = past_values.unsqueeze(-1)  # Add feature dimension
    future_values = future_values.unsqueeze(-1)  # Add feature dimension

    # For PatchTST and other HuggingFace time series models
    return {
        "past_values": past_values,
        "future_values": future_values,  # PatchTST expects future_values for training
    }


class TimeSeriesTrainer(Trainer):
    """
    Custom trainer for time series models that don't return loss in forward pass.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Custom loss computation for time series models.

        Args:
            model: The model to compute loss for
            inputs: Dictionary containing past_values and future_values
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (HuggingFace Trainer compatibility)

        Returns:
            Loss tensor (and optionally model outputs)
        """
        past_values = inputs["past_values"]
        future_values = inputs["future_values"]

        # Move tensors to model device
        device = next(model.parameters()).device
        past_values = past_values.to(device)
        future_values = future_values.to(device)

        # Run forward pass
        outputs = model(past_values=past_values)

        # For PatchTST, check for prediction_outputs attribute first
        predictions = None
        if hasattr(outputs, "prediction_outputs"):
            predictions = outputs.prediction_outputs
        elif hasattr(outputs, "forecast"):
            predictions = outputs.forecast
        elif hasattr(outputs, "last_hidden_state"):
            predictions = outputs.last_hidden_state
        else:
            # Try the first output if it's a tuple/sequence
            if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                predictions = outputs[0]
            else:
                raise ValueError(
                    f"Could not extract predictions from model output: {type(outputs)}"
                )

        # Handle shape mismatches
        # Target shape: (batch_size, prediction_length, num_features)
        target_shape = future_values.shape

        # Adjust predictions to match target
        if len(predictions.shape) == 4:  # (batch, features, seq_len, patch_dim)
            # For PatchTST output (batch, features, num_patches, patch_length)
            batch_size, num_features, num_patches, patch_length = predictions.shape
            # Reshape to (batch, features, sequence_length)
            predictions = predictions.view(batch_size, num_features, -1)
            # Take only the prediction_length timesteps
            if predictions.shape[2] >= target_shape[1]:
                predictions = predictions[:, :, : target_shape[1]]
            # Transpose to (batch, sequence_length, features)
            predictions = predictions.transpose(1, 2)
        elif len(predictions.shape) == 3:
            # Check if dimensions match
            if predictions.shape[1] == target_shape[1]:
                # Shape is likely (batch, seq_len, features) - check feature dimension
                if predictions.shape[2] != target_shape[2]:
                    # Need to adjust feature dimension
                    if predictions.shape[2] > target_shape[2]:
                        # Take first feature or average
                        predictions = predictions[:, :, : target_shape[2]]
                    else:
                        # Expand features if needed
                        predictions = predictions.mean(dim=2, keepdim=True)
            elif predictions.shape[2] == target_shape[1]:
                # Shape might be (batch, features, seq_len) - transpose
                predictions = predictions.transpose(1, 2)
                if predictions.shape[2] != target_shape[2]:
                    predictions = predictions.mean(dim=2, keepdim=True)
            else:
                # Try to resize to match target
                predictions = predictions[:, : target_shape[1], :]
                if predictions.shape[2] != target_shape[2]:
                    predictions = predictions.mean(dim=2, keepdim=True)

        # Ensure shapes match exactly
        if predictions.shape != future_values.shape:
            # Try to make them compatible
            min_seq_len = min(predictions.shape[1], future_values.shape[1])
            min_features = min(predictions.shape[2], future_values.shape[2])
            predictions = predictions[:, :min_seq_len, :min_features]
            future_values = future_values[:, :min_seq_len, :min_features]

        # Compute MSE loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, future_values)

        return (loss, outputs) if return_outputs else loss


def train(
    model_key: str,
    model_kwargs: Dict[str, Any],
    peft_cfg: PEFTConfig,
    datasets: DatasetDict,
    output_dir: str,
    training_args: TrainingArguments,
) -> Trainer:
    """
    Fine-tune with PEFT and HuggingFace Trainer using Strategy pattern.

    Args:
        model_key: Registry key for the model type
        model_kwargs: Model configuration parameters
        peft_cfg: PEFT configuration
        datasets: Training and validation datasets
        output_dir: Directory to save the trained model
        training_args: Training configuration

    Returns:
        Trained HuggingFace Trainer instance
    """
    logger.info(f"Starting training for model: {model_key}")
    logger.info(f"Output directory: {output_dir}")

    # Get the appropriate strategy for the model
    strategy = get_strategy(model_key)

    # Create model using strategy
    base_model = strategy.create_model(**model_kwargs)
    logger.info(f"Base model created: {type(base_model).__name__}")

    # Temporarily disable PEFT for troubleshooting model compatibility
    # TODO: Re-enable PEFT once core model issues are resolved
    # Wrap with PEFT
    # model = build_peft_model(base_model, peft_cfg)
    model = base_model
    logger.info("Using base model without PEFT for troubleshooting")

    # Get data collator from strategy
    data_collator = strategy.prepare_data_collator()

    # Get trainer class from strategy
    trainer_class = strategy.create_trainer_class()

    # Initialize Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed successfully")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info(f"Model saved to: {output_dir}")

    return trainer


if __name__ == "__main__":
    import argparse
    from peft_config import PEFTConfig
    from transformers import TrainingArguments
    from datasets import load_from_disk

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument(
        "--model_args",
        type=str,
        required=True,
        help="JSON string of model config kwargs",
    )
    parser.add_argument(
        "--peft_args", type=str, required=True, help="JSON string of PEFTConfig"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to preprocessed DatasetDict on disk",
    )
    parser.add_argument("--output_dir", type=str, default="peft_output")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    model_kwargs = json.loads(args.model_args)
    peft_cfg = PEFTConfig.model_validate_json(args.peft_args)
    datasets = load_from_disk(args.data_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="epoch",
        fp16=True,
        gradient_accumulation_steps=2,
    )

    train(
        model_key=args.model_key,
        model_kwargs=model_kwargs,
        peft_cfg=peft_cfg,
        datasets=datasets,
        output_dir=args.output_dir,
        training_args=training_args,
    )
