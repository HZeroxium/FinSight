# example_usage.py

import os
import sys

# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Add common path for logger
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

from transformers import TrainingArguments
from datasets import DatasetDict

from data_loader import DataLoader, DataLoaderConfig
from features import FeatureEngineer, FeatureConfig
from preprocessing import Preprocessor, PreprocessorConfig
from peft_config import PEFTConfig
from train import train
from evaluate import evaluate, EvalConfig
from predict import PredictConfig, predict_next
from strategies import get_strategy

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)


def run_model_example(
    model_key: str, context_length: int = 128, prediction_length: int = 24
):
    """
    Run a complete example for a single model: train, evaluate, and predict.

    Args:
        model_key: The model to use (patchtst, autoformer, informer, etc.)
        context_length: Number of past timesteps to use
        prediction_length: Number of future timesteps to predict
    """
    logger.info(f"Starting example for model: {model_key}")

    try:
        # 1. Load & preprocess data once
        logger.info("Loading and preprocessing data")
        import os

        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "BTCUSDT_1d.csv"
        )
        dl_cfg = DataLoaderConfig(csv_path=data_path, symbol="BTCUSDT", timeframe="1d")
        raw_ds: DatasetDict = DataLoader(dl_cfg).load()

        feat_cfg = FeatureConfig()  # defaults
        fe = FeatureEngineer(feat_cfg)
        ds_feat = {split: fe.transform(raw_ds[split]) for split in raw_ds}

        prep_cfg = PreprocessorConfig(
            context_length=context_length, prediction_length=prediction_length, stride=1
        )
        prep = Preprocessor(prep_cfg)
        ds_windows = DatasetDict(
            {split: prep.transform(ds_feat[split]) for split in ds_feat}
        )

        logger.info(
            f"Dataset prepared - Train: {len(ds_windows['train'])}, Test: {len(ds_windows['test'])}"
        )

        # 2. Get strategy for model-specific configuration
        strategy = get_strategy(model_key)
        peft_cfg = strategy.get_default_peft_config()

        logger.info(f"Using PEFT config: {peft_cfg}")

        # 3. Fine-tune with PEFT
        output_dir = f"models/{model_key}_lora"
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,  # Smaller batch size for stability
            per_device_eval_batch_size=8,
            num_train_epochs=2,  # Fewer epochs for demo
            learning_rate=3e-4,
            fp16=False,  # Disable FP16 to avoid inf checks error
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to=None,  # Disable wandb
            logging_steps=50,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        logger.info(f"Starting training for {model_key}")
        trainer = train(
            model_key=model_key,
            model_kwargs={
                "context_length": prep_cfg.context_length,
                "prediction_length": prep_cfg.prediction_length,
            },
            peft_cfg=peft_cfg,
            datasets=ds_windows,
            output_dir=output_dir,
            training_args=training_args,
        )
        logger.info(f"Training completed for {model_key}")

        # 4. Evaluate
        logger.info(f"Starting evaluation for {model_key}")
        eval_cfg = EvalConfig(
            model_key=model_key,
            model_path=output_dir,
            context_length=prep_cfg.context_length,
            prediction_length=prep_cfg.prediction_length,
        )
        metrics = evaluate(ds_windows["test"], eval_cfg)
        logger.info(f"Test metrics for {model_key}: {metrics}")

        # 5. Predict next steps
        logger.info(f"Starting prediction for {model_key}")
        # Pull last window of test set
        last_window = ds_windows["test"][-1]["past_values"]
        pred_cfg = PredictConfig(
            model_key=model_key,
            model_path=output_dir,
            context=(
                last_window.tolist()
                if hasattr(last_window, "tolist")
                else list(last_window)
            ),
            n_steps=prep_cfg.prediction_length,
        )
        forecast = predict_next(pred_cfg)
        logger.info(
            f"Next {prep_cfg.prediction_length} predictions for {model_key}: {forecast[:5]}..."
        )  # Show first 5

        return {
            "model_key": model_key,
            "status": "success",
            "metrics": metrics,
            "forecast_sample": forecast[:5],
        }

    except Exception as e:
        logger.error(f"Error running example for {model_key}: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        # Instead of returning error dict, raise the exception to stop execution
        raise RuntimeError(f"Model {model_key} failed: {e}") from e


def run_all_models():
    """
    Run examples for all supported models.
    """
    # All supported models
    models = [
        "patchtst",
        "autoformer",
        "informer",
        "patchtsmixer",
        "ts_transformer",
        # "timesfm",  # Include TimesFM
    ]

    results = []

    for model_key in models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running example for {model_key.upper()}")
        logger.info(f"{'='*50}")

        try:
            result = run_model_example(model_key)
            results.append(result)

            # Print summary
            logger.info(f"✓ {model_key} completed successfully")
            logger.info(f"  MAPE: {result['metrics']['MAPE (%)']:.2f}%")
            logger.info(f"  RMSE: {result['metrics']['RMSE']:.4f}")
        except Exception as e:
            logger.error(f"✗ {model_key} failed: {e}")
            results.append(
                {"model_key": model_key, "status": "failed", "error": str(e)}
            )

    # Final summary
    logger.info(f"\n{'='*50}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*50}")

    successful_models = [r for r in results if r["status"] == "success"]
    failed_models = [r for r in results if r["status"] == "failed"]

    logger.info(f"Successful models: {len(successful_models)}/{len(results)}")
    for result in successful_models:
        logger.info(
            f"  ✓ {result['model_key']}: MAPE={result['metrics']['MAPE (%)']:.2f}%"
        )

    if failed_models:
        logger.info(f"Failed models: {len(failed_models)}")
        for result in failed_models:
            logger.info(f"  ✗ {result['model_key']}: {result['error']}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run fine-tuning examples")
    parser.add_argument(
        "--model", type=str, help="Specific model to run (patchtst, autoformer, etc.)"
    )
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument(
        "--context_length", type=int, default=128, help="Context length"
    )
    parser.add_argument(
        "--prediction_length", type=int, default=24, help="Prediction length"
    )

    args = parser.parse_args()

    if args.all:
        run_all_models()
    elif args.model:
        run_model_example(args.model, args.context_length, args.prediction_length)
    else:
        # Default: run PatchTST example
        logger.info("Running default PatchTST example")
        run_model_example("patchtst", args.context_length, args.prediction_length)
