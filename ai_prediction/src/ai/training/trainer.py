import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
)
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm

from ..config import Config, OptimizerType, SchedulerType
from ..models.interface import ModelInterface
from ..utils import FileUtils, MetricUtils, CommonUtils
from common.logger.logger_factory import LoggerFactory


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop

        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from

        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class ModelTrainer:
    """
    Comprehensive trainer for financial prediction models with advanced features:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Comprehensive monitoring
    - Automatic checkpointing
    """

    def __init__(self, config: Config, device: torch.device):
        """
        Initialize model trainer

        Args:
            config: Training configuration
            device: Device to train on (CPU/GPU)
        """
        self.config = config
        self.device = device
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_time": [],
        }

        # Components (initialized during training)
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler: Optional[GradScaler] = None
        self.early_stopping: Optional[EarlyStopping] = None

        # Paths
        self.checkpoint_dir = Path(config.model.checkpoint_dir)
        self.log_dir = Path(config.model.log_dir)
        FileUtils.ensure_dir(self.checkpoint_dir)
        FileUtils.ensure_dir(self.log_dir)

        # Mixed precision training
        self.use_amp = config.model.mixed_precision and device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")

        # Gradient clipping
        self.gradient_clip_value = config.model.gradient_clip_value

        self.logger.info(f"Trainer initialized for device: {device}")

    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Create optimizer based on configuration

        Args:
            model: Model to optimize

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        params = model.parameters()
        lr = self.config.model.learning_rate
        weight_decay = self.config.model.weight_decay

        if self.config.model.optimizer_type == OptimizerType.ADAM:
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif self.config.model.optimizer_type == OptimizerType.ADAMW:
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif self.config.model.optimizer_type == OptimizerType.SGD:
            optimizer = optim.SGD(
                params, lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        elif self.config.model.optimizer_type == OptimizerType.RMSPROP:
            optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.model.optimizer_type}")

        self.logger.info(f"Created {self.config.model.optimizer_type} optimizer")
        return optimizer

    def _create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """
        Create learning rate scheduler

        Args:
            optimizer: Optimizer to schedule

        Returns:
            Learning rate scheduler or None
        """
        scheduler_type = self.config.model.scheduler_type

        if scheduler_type == SchedulerType.STEP_LR:
            scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        elif scheduler_type == SchedulerType.COSINE:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.model.epochs,
                eta_min=self.config.model.min_lr,
            )
        elif scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5, verbose=True
            )
        elif scheduler_type == SchedulerType.WARM_RESTART:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, eta_min=self.config.model.min_lr
            )
        else:
            return None

        self.logger.info(f"Created {scheduler_type} scheduler")
        return scheduler

    def _train_epoch(
        self, model: ModelInterface, train_loader: DataLoader, criterion: nn.Module
    ) -> float:
        """
        Train model for one epoch

        Args:
            model: Model to train
            train_loader: Training data loader
            criterion: Loss function

        Returns:
            float: Average training loss for epoch
        """
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.model.epochs}",
            leave=False,
        )

        for batch_idx, (batch_x, batch_y) in enumerate(progress_bar):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.gradient_clip_value
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()

                # Gradient clipping
                if self.gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.gradient_clip_value
                    )

                self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.6f}",
                    "Avg Loss": f"{total_loss / (batch_idx + 1):.6f}",
                }
            )

        return total_loss / num_batches

    def _validate_epoch(
        self, model: ModelInterface, val_loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate model for one epoch

        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                if self.use_amp:
                    with autocast():
                        predictions = model(batch_x)
                        loss = criterion(predictions, batch_y)
                else:
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)

                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_y.cpu())

        # Calculate comprehensive metrics
        predictions_tensor = torch.cat(all_predictions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)

        metrics = MetricUtils.calculate_all_metrics(
            targets_tensor.numpy(), predictions_tensor.numpy()
        )

        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics

    def _save_checkpoint(
        self, model: ModelInterface, epoch: int, val_loss: float, is_best: bool = False
    ) -> str:
        """
        Save model checkpoint

        Args:
            model: Model to save
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far

        Returns:
            str: Path to saved checkpoint
        """
        checkpoint_name = CommonUtils.create_checkpoint_name(
            model.get_model_name(), epoch, val_loss, "val_loss"
        )
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save checkpoint with comprehensive metadata
        model.save_checkpoint(
            checkpoint_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metrics={"val_loss": val_loss, "is_best": is_best},
        )

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / f"best_{model.get_model_name()}.pt"
            model.save_checkpoint(best_path)
            self.logger.info(f"New best model saved: {best_path}")

        return str(checkpoint_path)

    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, float],
        epoch_time: float,
    ) -> None:
        """
        Log training metrics

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_metrics: Validation metrics
            epoch_time: Time taken for epoch
        """
        lr = self.optimizer.param_groups[0]["lr"]

        # Update training history
        self.training_history["train_loss"].append(train_loss)
        self.training_history["val_loss"].append(val_loss)
        self.training_history["learning_rate"].append(lr)
        self.training_history["epoch_time"].append(epoch_time)

        # Log to console
        self.logger.info(
            f"Epoch {epoch + 1:3d}/{self.config.model.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {lr:.2e} | "
            f"Time: {CommonUtils.format_duration(epoch_time)}"
        )

        # Log key metrics
        for metric_name, value in val_metrics.items():
            if metric_name in ["rmse", "mae", "mape", "directional_accuracy"]:
                self.logger.info(f"  {metric_name.upper()}: {value:.4f}")

    def train(
        self,
        model: ModelInterface,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        save_best: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model with comprehensive monitoring and optimization

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function (MSE if None)
            save_best: Whether to save best model

        Returns:
            Dict containing training history and final metrics
        """
        self.logger.info("Starting model training...")
        start_time = time.time()

        # Setup training components
        if criterion is None:
            criterion = nn.MSELoss()

        self.optimizer = self._create_optimizer(model)
        self.scheduler = self._create_scheduler(self.optimizer)
        self.early_stopping = EarlyStopping(
            patience=self.config.model.patience,
            min_delta=1e-6,
            restore_best_weights=True,
        )

        # Log training setup
        self.logger.info(f"Training setup:")
        self.logger.info(f"  Model: {model.get_model_name()}")
        self.logger.info(f"  Parameters: {model.get_num_parameters():,}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Epochs: {self.config.model.epochs}")
        self.logger.info(f"  Batch size: {self.config.model.batch_size}")
        self.logger.info(f"  Learning rate: {self.config.model.learning_rate}")
        self.logger.info(f"  Mixed precision: {self.use_amp}")

        # Training loop
        try:
            for epoch in range(self.config.model.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                # Training phase
                train_loss = self._train_epoch(model, train_loader, criterion)

                # Validation phase
                val_loss, val_metrics = self._validate_epoch(
                    model, val_loader, criterion
                )

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time

                # Log metrics
                self._log_metrics(epoch, train_loss, val_loss, val_metrics, epoch_time)

                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                if save_best and (is_best or (epoch + 1) % 10 == 0):
                    self._save_checkpoint(model, epoch, val_loss, is_best)

                # Early stopping check
                if self.early_stopping(val_loss, model):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

                # Memory cleanup
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

        # Training completed
        total_time = time.time() - start_time
        self.logger.info(
            f"Training completed in {CommonUtils.format_duration(total_time)}"
        )
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")

        # Final evaluation
        final_val_loss, final_metrics = self._validate_epoch(
            model, val_loader, criterion
        )

        # Prepare training results
        training_results = {
            "training_history": self.training_history,
            "final_metrics": final_metrics,
            "best_val_loss": self.best_val_loss,
            "total_training_time": total_time,
            "epochs_completed": self.current_epoch + 1,
            "model_info": model.get_model_info(),
            "training_config": {
                "optimizer": str(self.config.model.optimizer_type),
                "scheduler": str(self.config.model.scheduler_type),
                "learning_rate": self.config.model.learning_rate,
                "batch_size": self.config.model.batch_size,
                "mixed_precision": self.use_amp,
                "gradient_clipping": self.gradient_clip_value,
            },
        }

        return training_results

    def evaluate(
        self,
        model: ModelInterface,
        test_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data

        Args:
            model: Model to evaluate
            test_loader: Test data loader
            criterion: Loss function

        Returns:
            Dict containing comprehensive evaluation metrics
        """
        self.logger.info("Evaluating model on test data...")

        if criterion is None:
            criterion = nn.MSELoss()

        # Evaluation
        test_loss, test_metrics = self._validate_epoch(model, test_loader, criterion)

        # Additional analysis
        predictions, targets = model.predict_batch(test_loader, self.device)

        # Calculate prediction intervals and confidence metrics
        prediction_std = predictions.std().item()
        target_std = targets.std().item()

        evaluation_results = {
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "prediction_statistics": {
                "prediction_mean": predictions.mean().item(),
                "prediction_std": prediction_std,
                "target_mean": targets.mean().item(),
                "target_std": target_std,
                "prediction_range": (
                    predictions.min().item(),
                    predictions.max().item(),
                ),
                "target_range": (targets.min().item(), targets.max().item()),
            },
            "model_performance": {
                "explained_variance": test_metrics.get("r2", 0.0),
                "prediction_accuracy": 100 - test_metrics.get("mape", 100),
                "directional_accuracy": test_metrics.get("directional_accuracy", 0.0),
            },
        }

        # Log results
        self.logger.info("Test Results:")
        self.logger.info(f"  Test Loss: {test_loss:.6f}")
        for metric_name, value in test_metrics.items():
            self.logger.info(f"  {metric_name.upper()}: {value:.4f}")

        return evaluation_results

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary

        Returns:
            Dict containing training summary
        """
        if not self.training_history["train_loss"]:
            return {"status": "No training completed"}

        return {
            "status": "Training completed",
            "epochs_completed": len(self.training_history["train_loss"]),
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.training_history["train_loss"][-1],
            "final_val_loss": self.training_history["val_loss"][-1],
            "total_training_time": sum(self.training_history["epoch_time"]),
            "average_epoch_time": np.mean(self.training_history["epoch_time"]),
            "learning_rate_range": (
                min(self.training_history["learning_rate"]),
                max(self.training_history["learning_rate"]),
            ),
            "training_history": self.training_history,
        }
