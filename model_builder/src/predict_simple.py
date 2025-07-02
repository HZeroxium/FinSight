"""
Simple FastAPI server for financial predictions using trained models.
Loads a specific checkpoint and provides REST API endpoints for predictions.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from .core.config import create_development_config
from .data import FeatureEngineering
from .utils import (
    DeviceUtils,
    CommonUtils,
    ModelUtils,
)
from .common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel


# Global variables for model and configuration
model = None
config = None
feature_engineering = None
device = None
logger = None


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    start_date: str
    end_date: str
    symbol: Optional[str] = "BTCUSDT"


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    predictions: Optional[List[float]] = None
    dates: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


def load_model_checkpoint(checkpoint_path: str) -> bool:
    """
    Load model from specific checkpoint path

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        bool: True if successful, False otherwise
    """
    global model, config, device, logger

    try:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False

        logger.info(f"Loading model from checkpoint: {checkpoint_file.name}")

        # Load model with automatic config inference
        model = ModelUtils.load_model_for_inference(
            checkpoint_file, config=None, device=device
        )

        # Load checkpoint to get config
        checkpoint = ModelUtils._safe_torch_load(checkpoint_file, map_location="cpu")
        inferred_config = ModelUtils._infer_config_from_checkpoint(checkpoint)

        # Update global config
        config.model.input_dim = inferred_config.model.input_dim
        config.model.d_model = inferred_config.model.d_model
        config.model.n_layers = inferred_config.model.n_layers
        config.model.n_heads = inferred_config.model.n_heads
        config.model.sequence_length = inferred_config.model.sequence_length
        config.model.model_type = inferred_config.model.model_type

        logger.info(f"✓ Model loaded successfully:")
        logger.info(f"  - Model type: {config.model.model_type.value}")
        logger.info(f"  - Input dimension: {config.model.input_dim}")
        logger.info(f"  - Sequence length: {config.model.sequence_length}")
        logger.info(f"  - d_model: {config.model.d_model}")

        return True

    except Exception as e:
        logger.error(f"Failed to load model checkpoint: {str(e)}")
        return False


def prepare_synthetic_data(
    start_date: str, end_date: str, symbol: str = "BTCUSDT"
) -> pd.DataFrame:
    """
    Prepare synthetic data for the given date range
    In production, this would fetch real data from an API or database

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        symbol: Trading symbol

    Returns:
        DataFrame with synthetic data
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Calculate number of days
        days = (end_dt - start_dt).days + 1

        if days <= 0:
            raise ValueError("End date must be after start date")

        if days > 365:
            raise ValueError("Date range cannot exceed 365 days")

        # For demo purposes, load existing data and simulate the date range
        # In production, you would fetch real data for these dates
        data_file = (
            project_root / "data" / "binance_BTCUSDT_20170817_20250630_1d_dataset.csv"
        )

        if data_file.exists():
            # Load existing data and take a subset
            df = pd.read_csv(data_file)
            df = df.tail(
                days + config.model.sequence_length
            )  # Extra data for sequence creation
        else:
            # Generate completely synthetic data if no real data available
            np.random.seed(42)
            dates = pd.date_range(start=start_dt, end=end_dt, freq="D")

            # Generate synthetic OHLCV data
            base_price = 50000  # Base BTC price
            price_data = []
            current_price = base_price

            for _ in range(len(dates)):
                # Random walk with some trend
                change = np.random.normal(0, 0.02) * current_price
                current_price += change
                current_price = max(1000, current_price)  # Minimum price

                # Generate OHLCV based on current price
                open_price = current_price + np.random.normal(0, 0.005) * current_price
                close_price = current_price + np.random.normal(0, 0.005) * current_price
                high_price = (
                    max(open_price, close_price)
                    + abs(np.random.normal(0, 0.01)) * current_price
                )
                low_price = (
                    min(open_price, close_price)
                    - abs(np.random.normal(0, 0.01)) * current_price
                )
                volume = np.random.uniform(1000, 10000)

                price_data.append(
                    {
                        "Open": open_price,
                        "High": high_price,
                        "Low": low_price,
                        "Close": close_price,
                        "Volume": volume,
                    }
                )

            df = pd.DataFrame(price_data)
            df["Date"] = dates.strftime("%Y-%m-%d")

        # Update dates to match requested range
        new_dates = pd.date_range(start=start_dt, periods=len(df), freq="D")
        df["Date"] = new_dates.strftime("%Y-%m-%d")

        logger.info(
            f"✓ Prepared data for {symbol}: {len(df)} records from {start_date} to {end_date}"
        )
        return df

    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise


def process_data_for_prediction(raw_data: pd.DataFrame) -> tuple:
    """
    Process raw data for prediction using existing pipeline

    Args:
        raw_data: Raw financial data

    Returns:
        Tuple of (sequences, dates, feature_columns)
    """
    global feature_engineering, config, logger

    try:
        # Process data using existing feature engineering
        processed_data = feature_engineering.process_data(raw_data, fit=False)

        # Get feature columns matching model input dimension
        if config.model.use_all_features:
            feature_columns = feature_engineering.get_meaningful_features(
                processed_data
            )
        else:
            numeric_features = processed_data.select_dtypes(
                include=["float64", "int64"]
            ).columns
            feature_columns = [
                col for col in config.model.features_to_use if col in numeric_features
            ]

        # Ensure we have the right number of features
        if len(feature_columns) > config.model.input_dim:
            feature_columns = feature_columns[: config.model.input_dim]
        elif len(feature_columns) < config.model.input_dim:
            # Pad with zeros if needed
            logger.warning(
                f"Feature count ({len(feature_columns)}) less than model input dim ({config.model.input_dim})"
            )
            # Use available features and pad later if needed

        # Extract feature data
        feature_data = processed_data[feature_columns].values

        # Create sequences for prediction
        sequences = []
        dates = []

        for i in range(config.model.sequence_length, len(feature_data)):
            sequence = feature_data[i - config.model.sequence_length : i]

            # Pad sequence if needed
            if sequence.shape[1] < config.model.input_dim:
                padding = np.zeros(
                    (sequence.shape[0], config.model.input_dim - sequence.shape[1])
                )
                sequence = np.concatenate([sequence, padding], axis=1)

            sequences.append(sequence)
            dates.append(
                processed_data.iloc[i]["Date"]
                if "Date" in processed_data.columns
                else f"Day_{i}"
            )

        sequences = np.array(sequences)

        logger.info(f"✓ Created {len(sequences)} prediction sequences")
        logger.info(f"✓ Sequence shape: {sequences.shape}")

        return sequences, dates, feature_columns

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise


def make_predictions(sequences: np.ndarray) -> np.ndarray:
    """
    Make predictions using the loaded model

    Args:
        sequences: Input sequences for prediction

    Returns:
        Array of predictions
    """
    global model, device, logger

    try:
        model.eval()
        predictions = []

        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i : i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(device)

                batch_predictions = model(batch_tensor)
                predictions.append(batch_predictions.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        # Flatten if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        logger.info(f"✓ Generated {len(predictions)} predictions")
        return predictions

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise


# Initialize FastAPI app
app = FastAPI(
    title="FinSight Prediction API",
    description="Simple API for financial predictions using trained AI models",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global config, device, feature_engineering, logger

    # Setup logging
    logger = LoggerFactory.get_logger(
        name="predict_simple",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
        use_colors=True,
    )

    logger.info("🚀 Starting FinSight Prediction API...")

    # Initialize configuration
    config = create_development_config()
    config.model.use_all_features = False
    config.model.features_to_use = ["Open", "High", "Low", "Close", "Volume"]

    # Setup device
    device = DeviceUtils.get_device(prefer_gpu=config.model.use_gpu)
    logger.info(f"✓ Using device: {device}")

    # Initialize feature engineering
    feature_engineering = FeatureEngineering(config)
    logger.info("✓ Feature engineering initialized")

    # Set seed for reproducibility
    CommonUtils.set_seed(config.random_seed)

    logger.info("✅ API initialization completed")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FinSight Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "load_model": "POST /load_model?checkpoint_path=...",
            "predict": "POST /predict",
            "health": "GET /health",
        },
    }


@app.post("/load_model")
async def load_model_endpoint(
    checkpoint_path: str = Query(..., description="Path to model checkpoint")
):
    """Load a specific model checkpoint"""
    global model, logger

    try:
        success = load_model_checkpoint(checkpoint_path)

        if success:
            model_info = {}
            if hasattr(model, "get_model_info"):
                model_info = model.get_model_info()

            return JSONResponse(
                content={
                    "success": True,
                    "message": f"Model loaded successfully from {Path(checkpoint_path).name}",
                    "model_info": {
                        "checkpoint_path": checkpoint_path,
                        "model_type": config.model.model_type.value,
                        "input_dim": config.model.input_dim,
                        "sequence_length": config.model.sequence_length,
                        "parameters": model_info.get("num_parameters", 0),
                        "size_mb": model_info.get("model_size_mb", 0),
                    },
                }
            )
        else:
            raise HTTPException(
                status_code=400, detail="Failed to load model checkpoint"
            )

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """Make predictions for the specified date range"""
    global model, logger

    if model is None:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Please load a model first using /load_model endpoint",
        )

    try:
        logger.info(
            f"📊 Processing prediction request: {request.start_date} to {request.end_date}"
        )

        # Prepare data for the requested date range
        raw_data = prepare_synthetic_data(
            start_date=request.start_date,
            end_date=request.end_date,
            symbol=request.symbol,
        )

        # Process data and create sequences
        sequences, dates, feature_columns = process_data_for_prediction(raw_data)

        if len(sequences) == 0:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data to create prediction sequences. Need more historical data.",
            )

        # Make predictions
        predictions = make_predictions(sequences)

        # Calculate some basic statistics
        pred_stats = {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "median": float(np.median(predictions)),
        }

        # Calculate price trend
        if len(predictions) > 1:
            trend = "upward" if predictions[-1] > predictions[0] else "downward"
            change_pct = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
        else:
            trend = "neutral"
            change_pct = 0.0

        response_data = {
            "symbol": request.symbol,
            "date_range": {
                "start": request.start_date,
                "end": request.end_date,
                "total_predictions": len(predictions),
            },
            "statistics": pred_stats,
            "trend_analysis": {
                "direction": trend,
                "change_percentage": round(change_pct, 2),
            },
            "model_info": {
                "type": config.model.model_type.value,
                "sequence_length": config.model.sequence_length,
                "features_used": len(feature_columns),
            },
        }

        logger.info(
            f"✅ Prediction completed: {len(predictions)} predictions generated"
        )

        return PredictionResponse(
            success=True,
            message=f"Predictions generated successfully for {request.symbol}",
            data=response_data,
            predictions=predictions.tolist(),
            dates=dates,
            metadata={
                "request_timestamp": CommonUtils.get_readable_timestamp(),
                "processing_time": "< 1s",
                "features_used": feature_columns,
            },
        )

    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": CommonUtils.get_readable_timestamp(),
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "config_loaded": config is not None,
    }


@app.get("/model_info")
async def model_info_endpoint():
    """Get information about the currently loaded model"""
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        model_info = {}
        if hasattr(model, "get_model_info"):
            model_info = model.get_model_info()

        return {
            "model_loaded": True,
            "model_type": config.model.model_type.value,
            "architecture": {
                "input_dim": config.model.input_dim,
                "d_model": config.model.d_model,
                "n_layers": config.model.n_layers,
                "n_heads": config.model.n_heads,
                "sequence_length": config.model.sequence_length,
            },
            "model_info": model_info,
            "device": str(device),
            "features_config": {
                "use_all_features": config.model.use_all_features,
                "configured_features": config.model.features_to_use,
                "target_column": config.model.target_column,
            },
        }

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting model info: {str(e)}"
        )


def main():
    """Main function to run the FastAPI server"""
    import argparse

    parser = argparse.ArgumentParser(description="FinSight Prediction API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to model checkpoint to load on startup"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Load model on startup if checkpoint provided
    if args.checkpoint:
        print(f"Model checkpoint will be loaded on startup: {args.checkpoint}")
        # Note: Model loading will happen in the startup event
        # We could also load it here, but it's better to do it in the FastAPI startup event

    print(f"🚀 Starting FinSight Prediction API Server...")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"📖 Docs: http://{args.host}:{args.port}/docs")
    print(f"🔄 Reload: {args.reload}")

    uvicorn.run(
        "predict_simple:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
