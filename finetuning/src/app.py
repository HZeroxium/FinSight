# app.py

from typing import List, Dict
import os
import uuid
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_loader import DataLoader, DataLoaderConfig
from features import FeatureEngineer, FeatureConfig
from preprocessing import Preprocessor, PreprocessorConfig
from peft_config import PEFTConfig
from transformers import TrainingArguments
from train import train
from evaluate import evaluate, EvalConfig
from predict import predict_next, PredictConfig
from strategies import get_strategy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common"))
from logger import LoggerFactory, LogLevel, LoggerType

logger = LoggerFactory.get_logger(
    __name__,
    level=LogLevel.INFO,
    logger_type=LoggerType.STANDARD,
)

app = FastAPI(title="TS-Forecast-API", version="1.0.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TrainRequest(BaseModel):
    csv_path: str = Field(..., description="Path to input CSV")
    symbol: str = Field(..., description="Asset symbol, e.g. BTCUSDT")
    timeframe: str = Field(..., description="Time granularity, e.g. 1h, 1d")
    model_key: str = Field(..., description="One of supported model keys")
    context_length: int = Field(128)
    prediction_length: int = Field(24)
    peft_args: PEFTConfig
    epochs: int = Field(5)
    batch_size: int = Field(16)
    learning_rate: float = Field(3e-4)


class TrainResponse(BaseModel):
    model_path: str = Field(..., description="Directory of saved fine-tuned model")
    metrics: Dict[str, float]


@app.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest):
    try:
        # 1. Load & filter
        dl_cfg = DataLoaderConfig(
            csv_path=req.csv_path, symbol=req.symbol, timeframe=req.timeframe
        )
        raw_ds = DataLoader(dl_cfg).load()
        # 2. Features + windows
        fe = FeatureEngineer(FeatureConfig())
        ds_feat = {s: fe.transform(raw_ds[s]) for s in raw_ds}
        prep_cfg = PreprocessorConfig(
            context_length=req.context_length, prediction_length=req.prediction_length
        )
        prep = Preprocessor(prep_cfg)
        ds_win = {s: prep.transform(ds_feat[s]) for s in ds_feat}
        # 3. Train
        train_args = TrainingArguments(
            output_dir=f"models/{uuid.uuid4()}",
            per_device_train_batch_size=req.batch_size,
            per_device_eval_batch_size=req.batch_size,
            num_train_epochs=req.epochs,
            learning_rate=req.learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            report_to=None,  # Disable wandb
        )
        trainer = train(
            model_key=req.model_key,
            model_kwargs={
                "context_length": req.context_length,
                "prediction_length": req.prediction_length,
            },
            peft_cfg=req.peft_args,
            datasets=ds_win,
            output_dir=train_args.output_dir,
            training_args=train_args,
        )
        # 4. Eval
        eval_cfg = EvalConfig(
            model_key=req.model_key,
            model_path=train_args.output_dir,
            context_length=req.context_length,
            prediction_length=req.prediction_length,
        )
        metrics = evaluate(ds_win["test"], eval_cfg)
        return TrainResponse(model_path=train_args.output_dir, metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictRequest(BaseModel):
    csv_path: str = Field(..., description="Path to input CSV")
    symbol: str = Field(..., description="Asset symbol, e.g. BTCUSDT")
    timeframe: str = Field(..., description="Time granularity, e.g. 1h, 1d")
    model_key: str = Field(..., description="Which model to use")
    model_path: str = Field(..., description="Directory of trained model")
    n_steps: int = Field(..., description="How many future steps to predict")


class PredictResponse(BaseModel):
    predictions: List[float]


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    try:
        # 1. Load & filter CSV, extract last window
        dl_cfg = DataLoaderConfig(
            csv_path=req.csv_path, symbol=req.symbol, timeframe=req.timeframe
        )
        raw_ds = DataLoader(dl_cfg).load()
        fe = FeatureEngineer(FeatureConfig())
        ds_feat = fe.transform(raw_ds["test"])
        prep_cfg = PreprocessorConfig(
            context_length=128,  # Use a default value instead of accessing model config
            prediction_length=req.n_steps,
        )
        ds_win = Preprocessor(prep_cfg).transform(ds_feat)
        last_ctx = ds_win[-1]["past_values"]
        # 2. Predict
        cfg = PredictConfig(
            model_key=req.model_key,
            model_path=req.model_path,
            context=last_ctx,
            n_steps=req.n_steps,
        )
        preds = predict_next(cfg)
        return PredictResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "device": DEVICE}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        workers=int(os.getenv("WORKERS", "2")),
    )
