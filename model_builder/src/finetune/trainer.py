# finetune/trainer.py

"""
Modern trainer using HuggingFace Transformers Trainer with PEFT support.
"""

from typing import Dict, Any, Optional, Callable
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from ..common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel
from .config import FineTuneConfig, TaskType
from .model_factory import ModelFactory
from .data_processor import FinancialDataProcessor
from .evaluator import FineTuneEvaluator


class SimpleDataCollator:
    """Simple data collator for time series data"""

    def __call__(self, features):
        batch = {}

        # Handle different input formats
        if "past_values" in features[0]:
            # Time series format
            batch["past_values"] = torch.stack([f["past_values"] for f in features])
            batch["future_values"] = torch.stack([f["future_values"] for f in features])
            if "attention_mask" in features[0]:
                batch["attention_mask"] = torch.stack(
                    [f["attention_mask"] for f in features]
                )
        else:
            # Language model format
            batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
            batch["attention_mask"] = torch.stack(
                [f["attention_mask"] for f in features]
            )
            if "labels" in features[0]:
                batch["labels"] = torch.stack([f["labels"] for f in features])

        return batch


class FineTuneTrainer:
    """Trainer for fine-tuning models"""

    def __init__(
        self,
        config: FineTuneConfig,
        model_factory: ModelFactory,
        data_processor: FinancialDataProcessor,
        evaluator: FineTuneEvaluator,
    ):
        self.config = config
        self.model_factory = model_factory
        self.data_processor = data_processor
        self.evaluator = evaluator

        self.logger = LoggerFactory.get_logger(
            name="finetune_trainer",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
        )

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self) -> None:
        """Setup model and tokenizer"""
        self.logger.info("Setting up model and tokenizer...")

        # Load model and tokenizer
        self.model, self.tokenizer = self.model_factory.create_model_and_tokenizer()

        self.logger.info("✓ Model setup completed")

    def train(
        self,
        train_dataset,
        val_dataset=None,
        compute_metrics: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Fine-tune the model with improved error handling

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            compute_metrics: Custom metrics function

        Returns:
            Training results
        """
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")

        self.logger.info("Starting fine-tuning...")

        # Create output directory
        self.config.output_dir.mkdir(exist_ok=True, parents=True)

        try:
            # Check if we can use HuggingFace Trainer
            if hasattr(self.model, "config") and hasattr(self.model, "save_pretrained"):
                return self._train_with_hf_trainer(
                    train_dataset, val_dataset, compute_metrics
                )
            else:
                return self._train_simple_model(train_dataset, val_dataset)

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            # Fallback to simple training
            self.logger.info("Falling back to simple training loop...")
            return self._train_simple_model(train_dataset, val_dataset)

    def _train_with_hf_trainer(self, train_dataset, val_dataset, compute_metrics):
        """Train using HuggingFace Trainer"""
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps if val_dataset else None,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            fp16=self.config.use_fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=False,
            report_to=None,  # Disable W&B for now
            save_total_limit=2,  # Keep only 2 checkpoints
        )

        # Setup data collator
        data_collator = SimpleDataCollator()

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=None,  # No tokenizer for time series
            data_collator=data_collator,
            compute_metrics=compute_metrics or self._default_compute_metrics,
        )

        # Start training
        train_result = self.trainer.train()

        self.logger.info("✓ Training completed")
        self.logger.info(f"Training loss: {train_result.training_loss:.4f}")

        return {"training_loss": train_result.training_loss}

    def _train_simple_model(self, train_dataset, val_dataset=None):
        """Custom training loop for simple models with better error handling"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=torch.cuda.is_available(),
        )

        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            try:
                for batch_idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()

                    # Handle different batch formats
                    if "past_values" in batch:
                        inputs = batch["past_values"].to(device).float()
                        targets = batch["future_values"].to(device).float()
                    else:
                        inputs = batch["input_ids"].to(device).float()
                        targets = batch["labels"].to(device).float()

                    # Ensure inputs have correct shape for model
                    if inputs.dim() == 2:
                        inputs = inputs.unsqueeze(-1)  # Add feature dimension

                    # Forward pass
                    outputs = self.model(inputs)

                    # Ensure output and target shapes match
                    if outputs.dim() > targets.dim():
                        outputs = outputs.squeeze()
                    if targets.dim() > 1:
                        targets = targets.squeeze()

                    loss = criterion(outputs, targets)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_batches += 1

                    if batch_idx % self.config.logging_steps == 0:
                        self.logger.info(
                            f"Epoch {epoch+1}/{self.config.num_epochs}, "
                            f"Batch {batch_idx}/{len(train_loader)}, "
                            f"Loss: {loss.item():.4f}"
                        )

                avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
                total_loss += avg_epoch_loss
                total_batches += 1

                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} completed, "
                    f"Average Loss: {avg_epoch_loss:.4f}"
                )

            except Exception as e:
                self.logger.error(f"Error in epoch {epoch+1}: {e}")
                continue

        avg_training_loss = total_loss / max(total_batches, 1)
        return {"training_loss": avg_training_loss}

    def _default_compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Default metrics computation"""
        predictions, labels = eval_pred

        # Handle different shapes
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = predictions.flatten()
        labels = labels.flatten()

        # Calculate basic metrics
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))

        return {
            "mse": mse,
            "mae": mae,
            "rmse": np.sqrt(mse),
        }

    def evaluate_model(self, eval_dataset) -> Dict[str, Any]:
        """Evaluate the fine-tuned model"""
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")

        self.logger.info("Evaluating model...")

        if self.trainer:
            eval_results = self.trainer.evaluate(eval_dataset)
        else:
            # Simple evaluation for custom models
            eval_results = self._evaluate_simple_model(eval_dataset)

        self.logger.info("✓ Evaluation completed")
        for metric, value in eval_results.items():
            self.logger.info(f"{metric}: {value:.4f}")

        return eval_results

    def _evaluate_simple_model(self, eval_dataset):
        """Custom evaluation for simple models"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        predictions = []
        targets = []

        with torch.no_grad():
            for batch in eval_loader:
                if "past_values" in batch:
                    inputs = batch["past_values"].to(device)
                    batch_targets = batch["future_values"].to(device)
                else:
                    inputs = batch["input_ids"].to(device)
                    batch_targets = batch["labels"].to(device)

                    if inputs.dim() == 2:
                        inputs = inputs.unsqueeze(-1)

                outputs = self.model(inputs)

                predictions.extend(outputs.cpu().numpy().flatten())
                targets.extend(batch_targets.cpu().numpy().flatten())

        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        return {
            "eval_mse": mse,
            "eval_mae": mae,
            "eval_rmse": np.sqrt(mse),
        }

    def save_model(self, save_path: Optional[str] = None) -> str:
        """Save the fine-tuned model"""
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")

        if save_path is None:
            save_path = str(self.config.output_dir / "final_model")

        Path(save_path).mkdir(exist_ok=True, parents=True)

        self.logger.info(f"Saving model to {save_path}")

        # Save using HuggingFace format
        if self.trainer:
            self.trainer.save_model(save_path)
        else:
            # Manual save for models without trainer
            self.model.save_pretrained(save_path)

        # Save config
        config_path = Path(save_path) / "training_config.json"
        with open(config_path, "w") as f:
            f.write(self.config.model_dump_json(indent=2))

        self.logger.info(f"✓ Model saved to {save_path}")
        return save_path

    def finish(self) -> None:
        """Clean up and finish training"""
        self.logger.info("✓ Training session finished")
