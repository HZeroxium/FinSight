# predict_example.py

"""
Enhanced example demonstrating how to use a trained model for prediction with comprehensive visualization.
This example creates synthetic data to demonstrate the prediction process and generates detailed visualizations.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from .core.config import Config, create_development_config
from .models import create_model
from .data import FeatureEngineering
from .utils import (
    DeviceUtils,
    CommonUtils,
    FileUtils,
    ModelUtils,
    PredictionUtils,
    VisualizationUtils,
)
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

    df = pd.read_csv("data/binance_BTCUSDT_20170817_20250630_1d_dataset.csv")

    # Last 1000 samples for demonstration
    if n_samples > len(df):
        raise ValueError(
            f"Requested n_samples ({n_samples}) exceeds available data ({len(df)} samples)."
        )
    df = df[-n_samples:]

    return df


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


def create_data_loader_from_sequences(
    sequences: np.ndarray, targets: np.ndarray, batch_size: int = 32
) -> DataLoader:
    """
    Create a DataLoader from sequences for compatibility with PredictionUtils

    Args:
        sequences: Input sequences
        targets: Target values
        batch_size: Batch size for DataLoader

    Returns:
        DataLoader instance
    """
    # Convert to tensors
    sequences_tensor = torch.FloatTensor(sequences)
    targets_tensor = torch.FloatTensor(targets)

    # Create dataset and dataloader
    dataset = TensorDataset(sequences_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def create_comprehensive_predictions(
    model: torch.nn.Module,
    sequences: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    model_name: str = "loaded_model",
) -> dict:
    """
    Create comprehensive predictions using PredictionUtils

    Args:
        model: Trained model
        sequences: Input sequences
        targets: Target values
        device: PyTorch device
        model_name: Name of the model

    Returns:
        Dictionary with comprehensive prediction results
    """
    # Create DataLoader for compatibility with PredictionUtils
    test_loader = create_data_loader_from_sequences(sequences, targets)

    # Generate comprehensive predictions using existing utilities
    prediction_results = PredictionUtils.generate_predictions(
        model=model, test_loader=test_loader, device=device, model_name=model_name
    )

    return prediction_results


def create_prediction_visualizations(
    prediction_results: dict,
    feature_names: list,
    raw_data: pd.DataFrame,
    processed_data: pd.DataFrame,
    output_dir: Path = None,
) -> dict:
    """
    Create comprehensive visualizations for prediction results

    Args:
        prediction_results: Results from comprehensive predictions
        feature_names: List of feature names used
        raw_data: Original raw data
        processed_data: Processed data
        output_dir: Output directory for visualizations

    Returns:
        Dictionary mapping visualization types to file paths
    """
    if output_dir is None:
        output_dir = Path("prediction_visualizations")

    FileUtils.ensure_dir(output_dir)

    logger = LoggerFactory.get_logger("visualization")
    logger.info(f"📊 Creating comprehensive visualizations in {output_dir}...")

    visualization_paths = {}

    try:
        # 1. Detailed prediction analysis using existing visualizer
        viz_path = VisualizationUtils.plot_predictions(
            predictions=prediction_results["predictions"],
            targets=prediction_results["targets"],
            save_path=output_dir / "detailed_prediction_analysis.png",
            model_name=prediction_results["model_used"],
        )
        visualization_paths["detailed_prediction_analysis"] = viz_path
        logger.info(f"✓ Created detailed prediction analysis: {Path(viz_path).name}")

        # 2. Trading simulation based on predictions
        viz_path = VisualizationUtils.plot_trading_simulation(
            predictions=prediction_results["predictions"],
            targets=prediction_results["targets"],
            save_path=output_dir / "trading_simulation.png",
        )
        visualization_paths["trading_simulation"] = viz_path
        logger.info(f"✓ Created trading simulation: {Path(viz_path).name}")

        # 3. Feature importance if available
        if "feature_importance" in prediction_results:
            viz_path = VisualizationUtils.plot_feature_importance(
                feature_importance=prediction_results["feature_importance"],
                feature_names=feature_names,
                title="Feature Importance Analysis",
                save_path=output_dir / "feature_importance.png",
            )
            visualization_paths["feature_importance"] = viz_path
            logger.info(f"✓ Created feature importance plot: {Path(viz_path).name}")

        # 4. Attention weights visualization if available
        if "attention_weights" in prediction_results:
            viz_path = VisualizationUtils.plot_attention_weights(
                attention_weights=prediction_results["attention_weights"],
                sequence_length=30,  # Default sequence length
                save_path=output_dir / "attention_weights.png",
            )
            visualization_paths["attention_weights"] = viz_path
            logger.info(
                f"✓ Created attention weights visualization: {Path(viz_path).name}"
            )

        # 5. Feature analysis
        viz_path = VisualizationUtils.plot_feature_analysis(
            feature_names=feature_names, save_path=output_dir / "feature_analysis.png"
        )
        visualization_paths["feature_analysis"] = viz_path
        logger.info(f"✓ Created feature analysis: {Path(viz_path).name}")

        # 6. Correlation matrix from processed data
        if processed_data is not None and not processed_data.empty:
            viz_path = VisualizationUtils.plot_correlation_matrix(
                data=processed_data,
                features=feature_names,
                save_path=output_dir / "correlation_matrix.png",
            )
            visualization_paths["correlation_matrix"] = viz_path
            logger.info(f"✓ Created correlation matrix: {Path(viz_path).name}")

            # 7. Feature distributions
            viz_path = VisualizationUtils.plot_feature_distributions(
                data=processed_data,
                features=feature_names,
                save_path=output_dir / "feature_distributions.png",
            )
            visualization_paths["feature_distributions"] = viz_path
            logger.info(f"✓ Created feature distributions: {Path(viz_path).name}")

        # 8. Price series from raw data
        if raw_data is not None and "Close" in raw_data.columns:
            viz_path = VisualizationUtils.plot_price_series(
                data=raw_data,
                price_cols=["Close"],
                title="Price Series Analysis",
                save_path=output_dir / "price_series.png",
            )
            visualization_paths["price_series"] = viz_path
            logger.info(f"✓ Created price series plot: {Path(viz_path).name}")

            # 9. Candlestick chart with prediction overlay
            if all(col in raw_data.columns for col in ["Open", "High", "Low", "Close"]):
                # Use last 50 data points for better visualization
                recent_data = raw_data.tail(50)
                recent_predictions = prediction_results["predictions"][
                    -len(recent_data) :
                ]

                viz_path = VisualizationUtils.plot_candlestick_chart(
                    data=recent_data,
                    overlay_predictions=recent_predictions,
                    save_path=output_dir / "candlestick_with_predictions.png",
                )
                visualization_paths["candlestick_with_predictions"] = viz_path
                logger.info(
                    f"✓ Created candlestick chart with predictions: {Path(viz_path).name}"
                )

        logger.info(
            f"✅ Successfully created {len(visualization_paths)} visualizations"
        )

    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}", exc_info=True)

    return visualization_paths


def create_prediction_report(
    prediction_results: dict,
    feature_names: list,
    model_info: dict,
    config: Config,
    output_dir: Path = None,
) -> str:
    """
    Create a comprehensive prediction report

    Args:
        prediction_results: Results from comprehensive predictions
        feature_names: List of feature names
        model_info: Model information
        config: Configuration object
        output_dir: Output directory

    Returns:
        Path to the generated report
    """
    if output_dir is None:
        output_dir = Path("prediction_reports")

    FileUtils.ensure_dir(output_dir)

    # Create detailed report
    report = {
        "prediction_summary": {
            "total_predictions": len(prediction_results["predictions"]),
            "model_used": prediction_results["model_used"],
            "timestamp": CommonUtils.get_readable_timestamp(),
        },
        "model_info": model_info,
        "prediction_analysis": prediction_results["analysis"],
        "future_predictions": prediction_results.get("future_predictions", {}),
        "configuration": {
            "model_config": {
                "model_type": config.model.model_type.value,
                "d_model": config.model.d_model,
                "n_layers": config.model.n_layers,
                "sequence_length": config.model.sequence_length,
                "input_dim": config.model.input_dim,
            },
            "features_used": feature_names,
            "target_column": config.model.target_column,
        },
        "performance_metrics": {
            "mae": prediction_results["analysis"]["mae"],
            "rmse": prediction_results["analysis"]["rmse"],
            "directional_accuracy": prediction_results["analysis"][
                "directional_accuracy"
            ],
            "accuracy_percentage": prediction_results["analysis"]["accuracy"],
        },
    }

    # Save report
    timestamp = CommonUtils.get_timestamp()
    report_path = output_dir / f"prediction_report_{timestamp}.json"
    FileUtils.save_json(report, str(report_path))

    return str(report_path)


def main():
    """Enhanced main function with comprehensive prediction analysis and visualization"""

    # Setup logging
    logger = LoggerFactory.get_logger(
        name="prediction_example",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
        use_colors=True,
    )

    logger.info("=" * 60)
    logger.info("🔮 ENHANCED FINANCIAL PREDICTION EXAMPLE")
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
        model_info = {}
        if hasattr(model, "get_model_info"):
            model_info = model.get_model_info()
            logger.info(f"✓ Model: {model_info.get('model_name', 'Unknown')}")
            logger.info(f"✓ Parameters: {model_info.get('num_parameters', 0):,}")
            logger.info(f"✓ Model size: {model_info.get('model_size_mb', 0):.2f} MB")

        # 6. Enhanced predictions using PredictionUtils
        logger.info("\n🔮 Generating comprehensive predictions...")
        # Use last 100 sequences for prediction
        test_sequences = sequences[-100:]
        test_targets = targets[-100:]

        # Generate comprehensive predictions using existing utilities
        prediction_results = create_comprehensive_predictions(
            model=model,
            sequences=test_sequences,
            targets=test_targets,
            device=device,
            model_name=model_info.get("model_name", "loaded_model"),
        )

        logger.info(f"✓ Generated comprehensive predictions:")
        logger.info(f"  - Test samples: {len(prediction_results['predictions'])}")
        logger.info(
            f"  - Model accuracy: {prediction_results['analysis']['accuracy']:.2f}%"
        )
        logger.info(f"  - MAE: {prediction_results['analysis']['mae']:.4f}")
        logger.info(f"  - RMSE: {prediction_results['analysis']['rmse']:.4f}")
        logger.info(
            f"  - Directional accuracy: {prediction_results['analysis']['directional_accuracy']:.2f}%"
        )

        # 7. Feature importance analysis
        logger.info("\n🔍 Analyzing feature importance...")
        try:
            # Create DataLoader for feature importance analysis
            importance_loader = create_data_loader_from_sequences(
                test_sequences, test_targets, batch_size=10
            )

            feature_importance = PredictionUtils.estimate_feature_importance(
                model=model,
                test_loader=importance_loader,
                feature_names=feature_columns,
                device=device,
                n_samples=20,
            )
            prediction_results["feature_importance"] = feature_importance

            # Log top features
            importance_pairs = list(zip(feature_columns, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)

            logger.info("✓ Top 5 most important features:")
            for i, (feature, importance) in enumerate(importance_pairs[:5]):
                logger.info(f"   {i+1}. {feature}: {importance:.4f}")

        except Exception as e:
            logger.warning(f"Could not estimate feature importance: {str(e)}")

        # 8. Extract attention weights if model supports it
        if hasattr(model, "get_attention_weights"):
            logger.info("\n🎯 Extracting attention weights...")
            try:
                sample_input = torch.FloatTensor(test_sequences[:1]).to(device)
                attention_weights = model.get_attention_weights(sample_input)
                prediction_results["attention_weights"] = attention_weights
                logger.info("✓ Attention weights extracted successfully")
            except Exception as e:
                logger.warning(f"Could not extract attention weights: {str(e)}")

        # 9. Future predictions using existing results
        logger.info("\n🔮 Future predictions:")
        if "future_predictions" in prediction_results:
            future_preds = prediction_results["future_predictions"]
            if isinstance(future_preds, dict) and "prediction" in future_preds:
                logger.info("✓ Future predictions (next steps):")
                for i, pred in enumerate(future_preds["prediction"][:5]):
                    logger.info(f"   Step {i+1}: {pred}")
                    # pass
            else:
                logger.info("✓ Future predictions generated (format varies)")
        else:
            logger.info("⚠ No future predictions available")

        # 10. Create comprehensive visualizations
        logger.info("\n📊 Creating comprehensive visualizations...")
        visualization_dir = Path("prediction_visualizations")
        visualization_paths = create_prediction_visualizations(
            prediction_results=prediction_results,
            feature_names=feature_columns,
            raw_data=raw_data,
            processed_data=processed_data,
            output_dir=visualization_dir,
        )

        logger.info(f"✅ Created {len(visualization_paths)} visualizations:")
        for viz_type, path in visualization_paths.items():
            logger.info(f"   - {viz_type}: {Path(path).name}")

        # 11. Generate comprehensive report
        logger.info("\n📄 Generating prediction report...")
        report_path = create_prediction_report(
            prediction_results=prediction_results,
            feature_names=feature_columns,
            model_info=model_info,
            config=config,
        )
        logger.info(f"✓ Report saved to: {report_path}")

        # 12. Save enhanced results
        logger.info("\n💾 Saving enhanced results...")
        enhanced_results = {
            "config": config.to_dict(),
            "data_info": {
                "total_samples": len(raw_data),
                "feature_count": len(feature_columns),
                "sequence_count": len(sequences),
                "test_samples": len(test_sequences),
            },
            "model_info": model_info,
            "prediction_results": {
                "predictions": prediction_results["predictions"].tolist(),
                "targets": prediction_results["targets"].tolist(),
                "analysis": prediction_results["analysis"],
                "future_predictions": prediction_results.get("future_predictions", {}),
                "feature_importance": (
                    prediction_results.get("feature_importance", []).tolist()
                    if "feature_importance" in prediction_results
                    else []
                ),
            },
            "visualizations": visualization_paths,
            "report_path": report_path,
            "execution_info": {
                "timestamp": CommonUtils.get_readable_timestamp(),
                "device_used": str(device),
                "features_used": feature_columns,
            },
        }

        FileUtils.ensure_dir("prediction_results")
        results_path = f"prediction_results/enhanced_prediction_example_{CommonUtils.get_timestamp()}.json"
        FileUtils.save_json(enhanced_results, results_path)
        logger.info(f"✓ Enhanced results saved to: {results_path}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ ENHANCED PREDICTION EXAMPLE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 {len(visualization_paths)} visualizations created")
        logger.info(f"📄 Report generated: {Path(report_path).name}")
        logger.info(f"💾 Results saved: {Path(results_path).name}")
        logger.info("=" * 60)

        return enhanced_results

    except Exception as e:
        logger.error(f"❌ Enhanced prediction example failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
