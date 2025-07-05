# predict.py

from typing import List
from pydantic import BaseModel, Field
import torch
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


class PredictConfig(BaseModel):
    """
    Configuration for single-shot multi-step forecasting.
    """

    model_key: str = Field(..., description="Registry key of model")
    model_path: str = Field(..., description="Path to pretrained/fine-tuned model")
    context: List[float] = Field(..., description="Most recent past values")
    n_steps: int = Field(..., description="Number of future steps to predict")


def predict_next(cfg: PredictConfig) -> List[float]:
    """
    Load model and predict next n_steps given past context.
    """
    logger.info(f"Starting prediction for model: {cfg.model_key}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Get strategy for the model
    strategy = get_strategy(cfg.model_key)

    # Create model config - we need context and prediction lengths
    model_config = strategy.get_default_config(
        context_length=len(cfg.context), prediction_length=cfg.n_steps
    )

    # Load model for inference
    model = strategy.load_model_for_inference(cfg.model_path, **model_config)
    model = model.to(device)
    model.eval()

    # Build input tensor with proper shape
    past = torch.tensor(cfg.context, dtype=torch.float32, device=device)

    # Add batch and feature dimensions: (seq_len,) -> (1, seq_len, 1)
    if len(past.shape) == 1:
        past = past.unsqueeze(0).unsqueeze(-1)  # (seq_len,) -> (1, seq_len, 1)
    elif len(past.shape) == 2:
        if past.shape[0] == 1:
            past = past.unsqueeze(-1)  # (1, seq_len) -> (1, seq_len, 1)
        else:
            past = past.unsqueeze(0)  # (seq_len, features) -> (1, seq_len, features)

    logger.info(f"Input shape: {past.shape}")

    # Create input dict (some models may need additional inputs)
    inputs = {"past_values": past}

    # Add any additional inputs that the model might need
    if hasattr(model.config, "num_time_features"):
        # Create dummy time features if needed
        time_features = torch.zeros(
            1, len(cfg.context), model.config.num_time_features, device=device
        )
        inputs["past_time_features"] = time_features

    with torch.no_grad():
        logger.info("Running model inference")
        out = model(**inputs)

        # Extract predictions using strategy
        try:
            preds_tensor = strategy.extract_predictions(out)
            if isinstance(preds_tensor, torch.Tensor):
                preds = preds_tensor.cpu().numpy().flatten().tolist()
            else:
                preds = (
                    list(preds_tensor.flatten())
                    if hasattr(preds_tensor, "flatten")
                    else list(preds_tensor)
                )
        except Exception as e:
            logger.warning(
                f"Strategy extraction failed: {e}, falling back to manual extraction"
            )
            # Fallback extraction methods
            if hasattr(out, "predictions"):
                preds = out.predictions.cpu().numpy().flatten().tolist()
            elif hasattr(out, "forecast"):
                preds = out.forecast.cpu().numpy().flatten().tolist()
            elif hasattr(out, "last_hidden_state"):
                preds = out.last_hidden_state.cpu().numpy().flatten().tolist()
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                pred_tensor = out[0]
                if isinstance(pred_tensor, torch.Tensor):
                    preds = pred_tensor.cpu().numpy().flatten().tolist()
                else:
                    preds = (
                        list(pred_tensor.flatten())
                        if hasattr(pred_tensor, "flatten")
                        else list(pred_tensor)
                    )
            else:
                logger.error(
                    f"Could not extract predictions from output type: {type(out)}"
                )
                raise ValueError(f"Could not extract predictions from model output")

        logger.info(f"Generated {len(preds)} predictions")
        return preds


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--context", type=str, required=True, help="JSON list of past values"
    )
    parser.add_argument("--n_steps", type=int, required=True)
    args = parser.parse_args()

    cfg = PredictConfig(
        model_key=args.model_key,
        model_path=args.model_path,
        context=json.loads(args.context),
        n_steps=args.n_steps,
    )
    preds = predict_next(cfg)
    print(preds)
