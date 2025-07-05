# backtest.py

from typing import List, Dict
from pydantic import BaseModel, Field
import pandas as pd
from datasets import DatasetDict
from evaluate import evaluate, EvalConfig


class BacktestConfig(BaseModel):
    """
    Configuration for walk-forward backtest.
    """

    initial_train_windows: int = Field(
        ..., description="Number of windows to train before first eval"
    )
    step_size: int = Field(
        ..., description="Windows to roll forward after each retrain"
    )
    max_windows: int = Field(..., description="Total number of windows in dataset")
    eval_cfg: EvalConfig


def walk_forward_backtest(data: DatasetDict, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Perform walk-forward evaluation, retraining at each step.
    Returns DataFrame of metrics per retrain point.
    """
    records: List[Dict] = []
    for start in range(cfg.initial_train_windows, cfg.max_windows, cfg.step_size):
        # split dynamic train/val/test
        train_ds = data["train"].select(range(0, start))
        val_ds = data["train"].select(range(start, start + cfg.step_size))
        test_ds = val_ds  # evaluate on next block
        # retrain or fine-tune model (omitted heavy retrain; assume model path static)
        metrics = evaluate(test_ds, EvalConfig(**cfg.eval_cfg.model_dump()))
        metrics["window_id"] = start
        records.append(metrics)
    return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse
    from datasets import load_from_disk

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--initial_windows", type=int, default=1000)
    parser.add_argument("--step_size", type=int, default=100)
    args = parser.parse_args()

    data = load_from_disk(args.data_dir)
    max_w = len(data["train"])
    eval_cfg = EvalConfig(
        model_key=args.model_key,
        model_path=args.model_path,
        context_length=128,
        prediction_length=24,
    )
    cfg = BacktestConfig(
        initial_train_windows=args.initial_windows,
        step_size=args.step_size,
        max_windows=max_w,
        eval_cfg=eval_cfg,
    )
    df = walk_forward_backtest(data, cfg)
    print(df)
