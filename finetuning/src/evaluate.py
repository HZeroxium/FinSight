# evaluate.py

from typing import Any, Dict
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import Trainer
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType
from strategies import get_strategy

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


class EvalConfig(BaseModel):
    """
    Configuration for evaluation metrics.
    """

    model_key: str = Field(..., description="Registry key of the model to evaluate")
    model_path: str = Field(..., description="Path to pretrained or fine-tuned model")
    context_length: int = Field(..., description="Number of past steps used")
    prediction_length: int = Field(..., description="Number of future steps to predict")


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute fraction of correct direction forecasts.
    """
    return np.mean(
        np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])
    )


def sharpe_ratio(returns: np.ndarray) -> float:
    """
    Compute annualized Sharpe ratio, assuming daily returns.
    """
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    return (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0


def max_drawdown(series: np.ndarray) -> float:
    """
    Compute maximum drawdown of a return series.
    """
    cum = np.cumprod(1 + series) - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def evaluate(ds: Dataset, cfg: EvalConfig) -> Dict[str, float]:
    """
    Run model on ds and compute MAPE, RMSE, directional accuracy,
    Sharpe ratio and max drawdown.
    """
    logger.info(f"Starting evaluation for model: {cfg.model_key}")

    # Get strategy for the model
    strategy = get_strategy(cfg.model_key)

    # Create model config
    model_config = strategy.get_default_config(
        context_length=cfg.context_length, prediction_length=cfg.prediction_length
    )

    # Load model for inference
    model = strategy.load_model_for_inference(cfg.model_path, **model_config)
    model.eval()

    # Create trainer for evaluation
    trainer_class = strategy.create_trainer_class()
    collator = strategy.prepare_data_collator()
    trainer = trainer_class(
        model=model,
        data_collator=collator,
    )

    logger.info("Running model predictions on test dataset")

    # Get predictions
    eval_results = trainer.predict(ds)
    preds = eval_results.predictions

    # Extract true values
    trues = np.array([x["future_values"] for x in ds])

    # Handle predictions shape - extract using strategy if needed
    if hasattr(strategy, "extract_predictions") and not isinstance(preds, np.ndarray):
        try:
            # Convert predictions if they're model outputs
            preds_tensor = strategy.extract_predictions(preds)
            if isinstance(preds_tensor, torch.Tensor):
                preds = preds_tensor.cpu().numpy()
            else:
                preds = np.array(preds_tensor)
        except Exception as e:
            logger.warning(f"Failed to extract predictions using strategy: {e}")
            # Fallback to direct conversion
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
            elif not isinstance(preds, np.ndarray):
                preds = np.array(preds)

    # Ensure predictions are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)

    logger.info(f"Predictions shape: {preds.shape}, Trues shape: {trues.shape}")

    # Handle PatchTST-specific output shape (batch, features, num_patches, patch_length)
    if len(preds.shape) == 4:
        batch_size, num_features, num_patches, patch_length = preds.shape
        # Reshape to (batch, features, sequence_length)
        preds = preds.reshape(batch_size, num_features, -1)
        # Take only the prediction_length timesteps to match target
        if preds.shape[2] >= trues.shape[1]:
            preds = preds[:, :, : trues.shape[1]]
        # If multiple features, take mean or first feature
        if num_features > 1:
            preds = preds.mean(axis=1)  # Average across features
        else:
            preds = preds.squeeze(1)  # Remove feature dimension

    # Ensure predictions match target shape
    if len(preds.shape) == 3 and preds.shape[2] == 1:
        preds = preds.squeeze(-1)  # Remove last dimension if it's 1

    # Match batch sizes first - take minimum
    min_batch_size = min(preds.shape[0], trues.shape[0])
    preds = preds[:min_batch_size]
    trues = trues[:min_batch_size]

    # Match sequence length if different
    if (
        len(preds.shape) > 1
        and len(trues.shape) > 1
        and preds.shape[1] != trues.shape[1]
    ):
        min_seq_len = min(preds.shape[1], trues.shape[1])
        preds = preds[:, :min_seq_len]
        trues = trues[:, :min_seq_len]

    logger.info(f"Final shapes - Predictions: {preds.shape}, Trues: {trues.shape}")

    # Flatten for metric computation
    y_true = trues.flatten()
    y_pred = preds.flatten()

    # Final safety check for shape mismatch
    if y_true.shape != y_pred.shape:
        logger.error(
            f"Shape mismatch after processing: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
        # Take minimum length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        logger.warning(f"Truncated to length {min_len}")

    # Compute metrics
    y_true = trues.flatten()
    y_pred = preds.flatten()

    # Compute metrics
    logger.info("Computing evaluation metrics")

    # MAPE
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))) * 100

    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Directional accuracy
    da = directional_accuracy(y_true, y_pred)

    # Daily returns from prediction errors
    returns = (y_pred[1:] - y_pred[:-1]) / (y_pred[:-1] + 1e-8)
    sr = sharpe_ratio(returns)
    dd = max_drawdown(returns)

    metrics = {
        "MAPE (%)": mape,
        "RMSE": rmse,
        "Directional Accuracy": da,
        "Sharpe Ratio": sr,
        "Max Drawdown": dd,
    }

    logger.info(f"Evaluation completed. Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    import argparse
    from datasets import load_from_disk

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--prediction_length", type=int, default=24)
    args = parser.parse_args()

    from evaluate import EvalConfig, evaluate

    ds = load_from_disk(args.data_dir)["test"]
    cfg = EvalConfig(
        model_key=args.model_key,
        model_path=args.model_path,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )
    metrics = evaluate(ds, cfg)
    print(pd.Series(metrics))
