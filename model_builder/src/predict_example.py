# predict_example.py

"""
Simple example demonstrating how to use a trained model for prediction.
This example creates synthetic data to demonstrate the prediction process.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from .core.config import Config, create_development_config
from .models import create_model
from .data import FeatureEngineering
from .utils import DeviceUtils, CommonUtils, FileUtils, ModelUtils
from .common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel


def create_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic financial data for demonstration

    Args:
        n_samples: Number of data points to generate

    Returns:
        DataFrame with synthetic OHLCV data
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate synthetic price data with some trend and noise
    base_price = 100.0
    trend = np.linspace(0, 20, n_samples)  # Upward trend
    noise = np.random.normal(0, 2, n_samples)  # Random noise

    # Generate Close prices
    close_prices = base_price + trend + noise.cumsum() * 0.1

    # Generate other OHLC data based on Close
    volatility = np.random.uniform(0.5, 2.0, n_samples)

    high_prices = close_prices + np.random.uniform(0, volatility)
    low_prices = close_prices - np.random.uniform(0, volatility)

    # Generate Open prices (previous close + small gap)
    open_prices = np.roll(close_prices, 1) + np.random.normal(0, 0.5, n_samples)
    open_prices[0] = close_prices[0]

    # Generate Volume data
    volume = np.random.randint(1000, 10000, n_samples)

    # Create DataFrame
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volume,
        }
    )

    return data


def prepare_data_for_prediction(data: pd.DataFrame, config: Config) -> tuple:
    """
    Prepare data for prediction using the same pipeline as training

    Args:
        data: Raw financial data
        config: Configuration object

    Returns:
        Tuple of (processed_data, feature_columns, scaler_info)
    """
    logger = LoggerFactory.get_logger("data_preparation")

    # Initialize feature engineering
    feature_engineering = FeatureEngineering(config)

    # Process data (fit=False for inference)
    processed_data = feature_engineering.process_data(
        data, fit=True
    )  # fit=True for demo

    # Determine features to use
    if config.model.use_all_features:
        feature_columns = feature_engineering.get_meaningful_features(processed_data)
        logger.info(f"Using all meaningful features: {len(feature_columns)} features")
    else:
        numeric_features = processed_data.select_dtypes(
            include=["float64", "int64"]
        ).columns
        feature_columns = [
            col for col in config.model.features_to_use if col in numeric_features
        ]
        logger.info(f"Using configured features: {len(feature_columns)} features")

    return processed_data, feature_columns, feature_engineering


def create_prediction_sequences(
    data: pd.DataFrame, feature_columns: list, target_column: str, sequence_length: int
) -> tuple:
    """
    Create sequences for prediction

    Args:
        data: Processed data
        feature_columns: List of feature column names
        target_column: Target column name
        sequence_length: Length of input sequences

    Returns:
        Tuple of (sequences, targets, feature_data, target_data)
    """
    # Extract feature and target data
    feature_data = data[feature_columns].values
    target_data = data[target_column].values

    # Create sequences
    sequences = []
    targets = []

    for i in range(sequence_length, len(data)):
        sequence = feature_data[i - sequence_length : i]
        target = target_data[i]

        sequences.append(sequence)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    return sequences, targets, feature_data, target_data


def load_or_create_model(config: Config, device: torch.device) -> torch.nn.Module:
    """
    Load a trained model or create a new one for demonstration

    Args:
        config: Configuration object (will be updated with inferred config)
        device: PyTorch device

    Returns:
        Model instance
    """
    logger = LoggerFactory.get_logger("model_loading")

    # Look for available checkpoints
    checkpoint_dir = Path("checkpoints")

    # Find any available checkpoint, sorted by modification time (newest first)
    available_checkpoints = []
    if checkpoint_dir.exists():
        available_checkpoints = ModelUtils.list_available_checkpoints(checkpoint_dir)

    model_loaded = False
    loaded_model = None

    # Try to load any available checkpoint with proper config inference
    if available_checkpoints:
        for checkpoint_info in available_checkpoints:
            # Skip checkpoints with errors
            if "error" in checkpoint_info:
                continue

            try:
                checkpoint_path = Path(checkpoint_info["path"])
                logger.info(f"Attempting to load checkpoint: {checkpoint_path.name}")

                # Load model with automatic config inference
                loaded_model = ModelUtils.load_model_for_inference(
                    checkpoint_path, config=None, device=device  # Let it infer config
                )

                # If successful, also update our config for data processing consistency
                # Load checkpoint again to get the inferred config
                checkpoint = ModelUtils._safe_torch_load(
                    checkpoint_path, map_location="cpu"
                )
                inferred_config = ModelUtils._infer_config_from_checkpoint(checkpoint)

                # Update the passed config with inferred model parameters for data processing
                config.model.input_dim = inferred_config.model.input_dim
                config.model.d_model = inferred_config.model.d_model
                config.model.n_layers = inferred_config.model.n_layers
                config.model.n_heads = inferred_config.model.n_heads
                config.model.sequence_length = inferred_config.model.sequence_length
                config.model.model_type = inferred_config.model.model_type

                model_loaded = True
                logger.info(f"Successfully loaded model from {checkpoint_path.name}")
                logger.info(
                    f"Model config - d_model: {config.model.d_model}, "
                    f"n_layers: {config.model.n_layers}, "
                    f"input_dim: {config.model.input_dim}"
                )
                break

            except Exception as e:
                logger.warning(
                    f"Failed to load checkpoint {checkpoint_path.name}: {str(e)}"
                )
                continue

    # Create new model if no checkpoint could be loaded
    if not model_loaded:
        logger.info("No existing model found or all loading attempts failed.")
        logger.warning("Creating new model for demonstration.")
        logger.warning(
            "Note: This model is not trained and predictions will be random!"
        )

        loaded_model = create_model(config.model.model_type.value, config)
        loaded_model.to(device)
        loaded_model.eval()

        # Log available checkpoints for debugging with safe key access
        if available_checkpoints:
            logger.info("Available checkpoints found but couldn't load:")
            for cp in available_checkpoints[:3]:  # Show first 3
                filename = cp.get("filename", "unknown")
                model_class = cp.get("model_class", "unknown")
                epoch = cp.get("epoch", "unknown")
                error = cp.get("error", None)

                if error:
                    logger.info(f"  - {filename} (ERROR: {error})")
                else:
                    logger.info(f"  - {filename} ({model_class}, epoch {epoch})")

    return loaded_model


def make_predictions(
    model: torch.nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Make predictions using the model

    Args:
        model: Trained model
        sequences: Input sequences
        device: PyTorch device
        batch_size: Batch size for prediction

    Returns:
        Predictions array
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)

            batch_predictions = model(batch_tensor)
            predictions.append(batch_predictions.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def analyze_predictions(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Analyze prediction quality

    Args:
        predictions: Model predictions
        targets: True targets

    Returns:
        Dictionary with analysis results
    """
    # Flatten arrays if needed
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if targets.ndim > 1:
        targets = targets.flatten()

    # Calculate metrics
    errors = predictions - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / (targets + 1e-8))) * 100

    # Directional accuracy
    pred_direction = np.diff(predictions) > 0
    target_direction = np.diff(targets) > 0
    directional_accuracy = np.mean(pred_direction == target_direction) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "mean_prediction": np.mean(predictions),
        "mean_target": np.mean(targets),
        "prediction_std": np.std(predictions),
        "target_std": np.std(targets),
    }


def predict_future(
    model: torch.nn.Module,
    last_sequence: np.ndarray,
    device: torch.device,
    n_steps: int = 5,
) -> np.ndarray:
    """
    Predict future values using the model

    Args:
        model: Trained model
        last_sequence: Last sequence from the data
        device: PyTorch device
        n_steps: Number of future steps to predict

    Returns:
        Future predictions
    """
    model.eval()
    future_predictions = []
    current_sequence = last_sequence.copy()

    with torch.no_grad():
        for _ in range(n_steps):
            # Prepare input
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)

            # Make prediction
            prediction = model(input_tensor)
            future_predictions.append(prediction.cpu().numpy().flatten()[0])

            # Update sequence for next prediction (simple approach)
            # In practice, you might want to update all features, not just the target
            # This is a simplified example
            new_row = current_sequence[-1].copy()
            new_row[0] = (
                prediction.cpu().numpy().flatten()[0]
            )  # Assuming first feature is the target

            # Shift sequence and add new prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row

    return np.array(future_predictions)


def main():
    """Main function demonstrating model prediction"""

    # Setup logging
    logger = LoggerFactory.get_logger(
        name="prediction_example",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
        use_colors=True,
    )

    logger.info("=" * 60)
    logger.info("🔮 FINANCIAL PREDICTION EXAMPLE")
    logger.info("=" * 60)

    try:
        # 1. Create configuration
        logger.info("📋 Setting up configuration...")
        config = create_development_config()

        # Use configured features initially (will be updated after model loading)
        config.model.use_all_features = False
        config.model.features_to_use = ["Open", "High", "Low", "Close", "Volume"]

        # Setup device
        device = DeviceUtils.get_device(prefer_gpu=config.model.use_gpu)
        CommonUtils.set_seed(config.random_seed)

        logger.info(f"✓ Using device: {device}")

        # 2. Create synthetic data
        logger.info("\n📊 Creating synthetic financial data...")
        raw_data = create_synthetic_data(n_samples=500)
        logger.info(f"✓ Created {len(raw_data)} data points")
        logger.info(
            f"✓ Date range: {raw_data['Date'].min()} to {raw_data['Date'].max()}"
        )
        logger.info(
            f"✓ Price range: ${raw_data['Close'].min():.2f} - ${raw_data['Close'].max():.2f}"
        )

        # 3. Load model first (this will update config with proper dimensions)
        logger.info("\n🤖 Loading model...")
        model = load_or_create_model(config, device)

        logger.info(f"✓ Final config after model loading:")
        logger.info(f"  - d_model: {config.model.d_model}")
        logger.info(f"  - n_layers: {config.model.n_layers}")
        logger.info(f"  - input_dim: {config.model.input_dim}")
        logger.info(f"  - sequence_length: {config.model.sequence_length}")

        # 4. Prepare data for prediction using updated config
        logger.info("\n🔧 Preparing data for prediction...")
        processed_data, feature_columns, feature_engineering = (
            prepare_data_for_prediction(raw_data, config)
        )
        logger.info(f"✓ Processed data shape: {processed_data.shape}")
        logger.info(
            f"✓ Selected features ({len(feature_columns)}): {feature_columns[:5]}..."
        )

        # Update config with actual feature dimensions if needed
        if len(feature_columns) != config.model.input_dim:
            logger.warning(
                f"Feature count mismatch: {len(feature_columns)} vs {config.model.input_dim}"
            )
            logger.info(
                "Using first {} features to match model input dimension".format(
                    config.model.input_dim
                )
            )
            feature_columns = feature_columns[: config.model.input_dim]

        # 5. Create sequences
        logger.info("\n📈 Creating prediction sequences...")
        sequences, targets, feature_data, target_data = create_prediction_sequences(
            processed_data,
            feature_columns,
            config.model.target_column,
            config.model.sequence_length,
        )
        logger.info(f"✓ Created {len(sequences)} sequences")
        logger.info(f"✓ Sequence shape: {sequences.shape}")
        logger.info(f"✓ Target shape: {targets.shape}")

        # Get model info for logging
        if hasattr(model, "get_model_info"):
            model_info = model.get_model_info()
            logger.info(f"✓ Model: {model_info.get('model_name', 'Unknown')}")
            logger.info(f"✓ Parameters: {model_info.get('num_parameters', 0):,}")
            logger.info(f"✓ Model size: {model_info.get('model_size_mb', 0):.2f} MB")

        # 6. Make predictions on recent data
        logger.info("\n🔮 Making predictions...")
        # Use last 100 sequences for prediction
        test_sequences = sequences[-100:]
        test_targets = targets[-100:]

        predictions = make_predictions(model, test_sequences, device)
        logger.info(f"✓ Generated {len(predictions)} predictions")

        # 7. Analyze predictions
        logger.info("\n📊 Analyzing prediction quality...")
        analysis = analyze_predictions(predictions, test_targets)

        logger.info(f"✓ Mean Absolute Error (MAE): {analysis['mae']:.4f}")
        logger.info(f"✓ Root Mean Square Error (RMSE): {analysis['rmse']:.4f}")
        logger.info(f"✓ Mean Absolute Percentage Error (MAPE): {analysis['mape']:.2f}%")
        logger.info(f"✓ Directional Accuracy: {analysis['directional_accuracy']:.2f}%")

        # 8. Predict future values
        logger.info("\n🔮 Predicting future values...")
        last_sequence = sequences[-1]
        future_predictions = predict_future(model, last_sequence, device, n_steps=5)

        logger.info("✓ Future predictions (next 5 steps):")
        current_price = targets[-1]
        for i, pred in enumerate(future_predictions):
            change = ((pred - current_price) / current_price) * 100
            logger.info(f"   Step {i+1}: ${pred:.2f} ({change:+.2f}%)")
            current_price = pred

        # 9. Save results (optional)
        logger.info("\n💾 Saving results...")
        results = {
            "config": config.to_dict(),
            "data_info": {
                "total_samples": len(raw_data),
                "feature_count": len(feature_columns),
                "sequence_count": len(sequences),
            },
            "predictions": {
                "test_predictions": predictions.tolist(),
                "test_targets": test_targets.tolist(),
                "future_predictions": future_predictions.tolist(),
            },
            "analysis": analysis,
            "model_info": model_info if hasattr(model, "get_model_info") else {},
        }

        FileUtils.ensure_dir("prediction_results")
        results_path = (
            f"prediction_results/prediction_example_{CommonUtils.get_timestamp()}.json"
        )
        FileUtils.save_json(results, results_path)
        logger.info(f"✓ Results saved to: {results_path}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ PREDICTION EXAMPLE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"❌ Prediction example failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
