# utils/demo_utils.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import torch

from ..common.logger.logger_factory import LoggerFactory
from ..core.config import Config, ModelConfig, ModelType
from ..data import FinancialDataLoader, FeatureEngineering
from ..models import create_model
from .device_utils import DeviceUtils
from .common_utils import CommonUtils
from .file_utils import FileUtils
from torch.utils.data import DataLoader


class DemoUtils:
    """Utility class for demo operations and analysis"""

    _logger = LoggerFactory.get_logger(__name__)

    @staticmethod
    def setup_demo_environment(config: Config, verbose: bool = True) -> Dict[str, Any]:
        """
        Setup demo environment and return setup info

        Args:
            config: Configuration object
            verbose: Whether to enable verbose logging

        Returns:
            Dict containing setup information
        """
        # Set reproducibility
        CommonUtils.ensure_reproducibility(config.random_seed)

        # Setup device
        device = DeviceUtils.get_device(
            prefer_gpu=config.model.use_gpu, gpu_id=config.model.gpu_id
        )

        # Get environment info
        env_info = CommonUtils.get_environment_info()

        setup_info = {
            "device": str(device),
            "environment": env_info,
            "config": config.to_dict(),
            "timestamp": CommonUtils.get_readable_timestamp(),
            "random_seed": config.random_seed,
        }

        DemoUtils._logger.info(f"✓ Demo setup completed")
        DemoUtils._logger.info(f"  Device: {device}")
        DemoUtils._logger.info(f"  Random seed: {config.random_seed}")
        DemoUtils._logger.info(f"  Configuration: {config.environment}")

        return setup_info

    @staticmethod
    def analyze_raw_data(data: pd.DataFrame, config: Config) -> Dict[str, Any]:
        """Analyze raw data characteristics"""
        # Get only numeric data for statistics
        numeric_data = data.select_dtypes(include=[np.number])

        return {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": {str(k): str(v) for k, v in data.dtypes.items()},
            "missing_values": data.isnull().sum().to_dict(),
            "date_range": (
                (
                    str(data[config.data.date_column].min()),
                    str(data[config.data.date_column].max()),
                )
                if config.data.date_column in data.columns
                else None
            ),
            "numeric_summary": (
                numeric_data.describe().to_dict() if not numeric_data.empty else {}
            ),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
            "numeric_columns": numeric_data.columns.tolist(),
            "non_numeric_columns": data.select_dtypes(
                exclude=[np.number]
            ).columns.tolist(),
        }

    @staticmethod
    def analyze_processed_data(data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze processed data characteristics"""
        numeric_data = data.select_dtypes(include=[np.number])

        # Only calculate correlations if we have numeric data
        high_corr = []
        correlation_matrix_size = (0, 0)
        feature_variance = {}

        if not numeric_data.empty:
            high_corr = DemoUtils._find_high_correlations(numeric_data)
            correlation_matrix_size = numeric_data.corr().shape
            feature_variance = numeric_data.var().describe().to_dict()

        return {
            "shape": data.shape,
            "feature_count": len(numeric_data.columns),
            "missing_values": int(data.isnull().sum().sum()),
            "correlation_matrix_size": correlation_matrix_size,
            "high_correlation_pairs": high_corr,
            "feature_variance": feature_variance,
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
            "numeric_columns": numeric_data.columns.tolist(),
            "non_numeric_columns": data.select_dtypes(
                exclude=[np.number]
            ).columns.tolist(),
        }

    @staticmethod
    def _find_high_correlations(
        data: pd.DataFrame, threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """Find highly correlated feature pairs"""
        try:
            # Ensure we only work with numeric data
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty or numeric_data.shape[1] < 2:
                return []

            corr_matrix = numeric_data.corr().abs()
            high_corr = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if pd.notna(corr_value) and corr_value > threshold:
                        high_corr.append(
                            (
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                float(corr_value),
                            )
                        )
            return high_corr[:10]  # Top 10 pairs
        except Exception as e:
            DemoUtils._logger.warning(f"Could not calculate correlations: {str(e)}")
            return []

    @staticmethod
    def validate_data_quality(data: pd.DataFrame, config: Config) -> Dict[str, Any]:
        """Validate data quality and return quality metrics"""
        # Completeness
        completeness = (1 - data.isnull().sum().sum() / data.size) * 100

        # Duplicates
        duplicates = int(data.duplicated().sum())

        # Infinite values (numeric columns only)
        numeric_data = data.select_dtypes(include=[np.number])
        infinite_values = 0
        zero_variance_features = 0

        if not numeric_data.empty:
            infinite_values = int(np.isinf(numeric_data).sum().sum())
            # Zero-variance features
            variances = numeric_data.var()
            zero_variance_features = int((variances == 0).sum())

        # Temporal consistency
        temporal_consistency = True if config.data.date_column in data.columns else None

        return {
            "completeness": float(completeness),
            "duplicates": duplicates,
            "infinite_values": infinite_values,
            "zero_variance_features": zero_variance_features,
            "data_types_consistency": True,
            "temporal_consistency": temporal_consistency,
            "numeric_feature_count": (
                len(numeric_data.columns) if not numeric_data.empty else 0
            ),
            "total_feature_count": len(data.columns),
        }

    @staticmethod
    def prepare_data_pipeline(
        config: Config, data_path: Optional[str] = None
    ) -> Tuple[Any, Any, Any]:
        """
        Execute complete data preparation pipeline

        Args:
            config: Configuration object
            data_path: Optional data file path

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        DemoUtils._logger.info("Starting data processing pipeline...")

        # Initialize components
        data_loader = FinancialDataLoader(config)
        feature_engineering = FeatureEngineering(config)

        # Load and process data
        if data_path:
            config.data.data_file = data_path

        raw_data = data_loader.load_data()
        DemoUtils._logger.info(f"✓ Raw data loaded: {raw_data.shape}")

        # Feature engineering
        processed_data = feature_engineering.process_data(raw_data, fit=True)
        DemoUtils._logger.info(f"✓ Data processed: {processed_data.shape}")

        # Update config with actual feature dimensions
        numeric_features = processed_data.select_dtypes(
            include=["float64", "int64"]
        ).columns
        available_features = [
            col for col in config.model.features_to_use if col in numeric_features
        ]

        if len(available_features) != len(config.model.features_to_use):
            DemoUtils._logger.warning(
                f"Some configured features not found. Using: {available_features}"
            )
            config.model.features_to_use = available_features

        config.model.input_dim = len(available_features)
        DemoUtils._logger.info(
            f"✓ Updated model input_dim to: {config.model.input_dim}"
        )

        # Create data loaders
        train_loader, val_loader, test_loader = data_loader.prepare_data(
            processed_data,
            feature_columns=config.model.features_to_use,
            target_column=config.model.target_column,
        )

        return train_loader, val_loader, test_loader

    @staticmethod
    def prepare_data_pipeline_with_data(
        config: Config, data_path: Optional[str] = None
    ) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], pd.DataFrame, pd.DataFrame]:
        """
        Execute complete data preparation pipeline and return loaders and data

        Args:
            config: Configuration object
            data_path: Optional data file path

        Returns:
            Tuple of ((train_loader, val_loader, test_loader), raw_data, processed_data)
        """
        DemoUtils._logger.info("Starting data processing pipeline...")

        # Initialize components
        data_loader = FinancialDataLoader(config)
        feature_engineering = FeatureEngineering(config)

        # Load and process data
        if data_path:
            config.data.data_file = data_path

        raw_data = data_loader.load_data()
        DemoUtils._logger.info(f"- Raw data loaded: {raw_data.shape}")
        # Save raw data for later use
        FileUtils.save_csv(raw_data, "demo_results/raw_data.csv")

        # Feature engineering
        processed_data = feature_engineering.process_data(raw_data, fit=True)
        DemoUtils._logger.info(f"- Data processed: {processed_data.shape}")

        # Update config with actual feature dimensions
        numeric_features = processed_data.select_dtypes(
            include=["float64", "int64"]
        ).columns
        available_features = [
            col for col in config.model.features_to_use if col in numeric_features
        ]

        if len(available_features) != len(config.model.features_to_use):
            DemoUtils._logger.warning(
                f"Some configured features not found. Using: {available_features}"
            )
            config.model.features_to_use = available_features

        config.model.input_dim = len(available_features)
        DemoUtils._logger.info(
            f"✓ Updated model input_dim to: {config.model.input_dim}"
        )

        # Create data loaders
        train_loader, val_loader, test_loader = data_loader.prepare_data(
            processed_data,
            feature_columns=config.model.features_to_use,
            target_column=config.model.target_column,
        )

        data_loaders = (train_loader, val_loader, test_loader)
        return data_loaders, raw_data, processed_data

    @staticmethod
    def create_model_variants(
        config: Config, device: torch.device
    ) -> Dict[str, torch.nn.Module]:
        """
        Create different model variants for comparison

        Args:
            config: Base configuration
            device: PyTorch device

        Returns:
            Dict mapping model names to model instances
        """
        from ..core.config import create_lightweight_config

        models = {}

        # Create model configurations
        model_configs = {
            "transformer": config,
            "lightweight": create_lightweight_config(),
            "hybrid": Config(
                model=ModelConfig(model_type=ModelType.HYBRID_TRANSFORMER)
            ),
        }

        for model_name, model_config in model_configs.items():
            DemoUtils._logger.info(f"🔧 Creating {model_name} model...")

            # Update config with current data dimensions
            model_config.model.input_dim = config.model.input_dim
            model_config.model.features_to_use = config.model.features_to_use
            model_config.model.sequence_length = config.model.sequence_length
            model_config.model.output_dim = config.model.output_dim
            model_config.model.prediction_horizon = config.model.prediction_horizon

            # Create model
            model = create_model(model_config.model.model_type.value, model_config)
            model.to(device)

            # Log model info
            model_info = model.get_model_info()
            param_count = model.count_parameters()

            DemoUtils._logger.info(f"✓ {model_name.capitalize()} model created:")
            DemoUtils._logger.info(f"  Parameters: {param_count['total']:,}")
            DemoUtils._logger.info(f"  Size: {model_info['model_size_mb']:.2f} MB")

            models[model_name] = model

        return models

    @staticmethod
    def calculate_additional_metrics(
        predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate additional evaluation metrics beyond basic ones"""
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Remove invalid values
        valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
        if not valid_mask.any():
            return {
                key: 0.0
                for key in [
                    "prediction_bias",
                    "prediction_variance",
                    "correlation",
                    "max_error",
                    "quantile_loss_50",
                    "hit_rate_5pct",
                ]
            }

        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        try:
            correlation = (
                np.corrcoef(pred_valid, target_valid)[0, 1]
                if len(pred_valid) > 1
                else 0.0
            )
            correlation = correlation if np.isfinite(correlation) else 0.0
        except:
            correlation = 0.0

        try:
            # Safe hit rate calculation
            relative_errors = np.abs(pred_valid - target_valid)
            threshold_values = 0.05 * np.abs(target_valid)
            safe_threshold = np.where(threshold_values == 0, 1e-8, threshold_values)
            hit_rate = np.mean(relative_errors < safe_threshold)
        except:
            hit_rate = 0.0

        return {
            "prediction_bias": float(np.mean(pred_valid - target_valid)),
            "prediction_variance": float(np.var(pred_valid - target_valid)),
            "correlation": float(correlation),
            "max_error": float(np.max(np.abs(pred_valid - target_valid))),
            "quantile_loss_50": float(
                np.mean(
                    np.maximum(
                        0.5 * (pred_valid - target_valid),
                        (0.5 - 1) * (pred_valid - target_valid),
                    )
                )
            ),
            "hit_rate_5pct": float(hit_rate),
        }

    @staticmethod
    def compare_models(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare models based on evaluation metrics"""
        comparison = {}
        metrics_to_compare = ["rmse", "mae", "mape", "directional_accuracy", "r2"]

        for metric in metrics_to_compare:
            metric_values = {}
            for model_name, results in evaluation_results.items():
                if model_name != "model_comparison":
                    test_metrics = results["evaluation_result"]["test_metrics"]
                    if metric in test_metrics:
                        metric_values[model_name] = test_metrics[metric]

            if metric_values:
                best_model = min(metric_values.items(), key=lambda x: x[1])
                worst_model = max(metric_values.items(), key=lambda x: x[1])

                comparison[metric] = {
                    "values": metric_values,
                    "best_model": best_model[0],
                    "best_value": float(best_model[1]),
                    "worst_model": worst_model[0],
                    "worst_value": float(worst_model[1]),
                    "improvement": float(
                        (worst_model[1] - best_model[1]) / worst_model[1] * 100
                    ),
                }

        return comparison

    @staticmethod
    def analyze_predictions(
        predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction quality"""
        errors = predictions - targets

        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "accuracy": float(100 - np.mean(np.abs(errors / (targets + 1e-8))) * 100),
            "directional_accuracy": float(
                np.mean(
                    (predictions[1:] - predictions[:-1]) * (targets[1:] - targets[:-1])
                    > 0
                )
                * 100
            ),
            "error_distribution": {
                "min": float(np.min(errors)),
                "max": float(np.max(errors)),
                "q25": float(np.percentile(errors, 25)),
                "q50": float(np.percentile(errors, 50)),
                "q75": float(np.percentile(errors, 75)),
            },
        }

    @staticmethod
    def prepare_results_for_serialization(results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization by converting numpy arrays"""
        serializable_results = {}

        for key, value in results.items():
            if key == "predictions" and isinstance(value, dict):
                # Handle the new consistent prediction structure
                if "predictions" in value and "targets" in value:
                    # This is the test_predictions structure
                    serializable_results[key] = {
                        "predictions": (
                            value["predictions"].tolist()
                            if hasattr(value["predictions"], "tolist")
                            else value["predictions"]
                        ),
                        "targets": (
                            value["targets"].tolist()
                            if hasattr(value["targets"], "tolist")
                            else value["targets"]
                        ),
                        "analysis": value.get("analysis", {}),
                        "future_predictions": value.get("future_predictions", {}),
                        "model_used": value.get("model_used", ""),
                    }
                elif "test_predictions" in value:
                    # Handle the old structure for backward compatibility
                    test_preds = value["test_predictions"]
                    serializable_results[key] = {
                        "test_predictions": {
                            "predictions": (
                                test_preds["predictions"].tolist()
                                if hasattr(test_preds["predictions"], "tolist")
                                else test_preds["predictions"]
                            ),
                            "targets": (
                                test_preds["targets"].tolist()
                                if hasattr(test_preds["targets"], "tolist")
                                else test_preds["targets"]
                            ),
                            "analysis": test_preds.get("analysis", {}),
                        },
                        "future_predictions": value.get("future_predictions", {}),
                        "model_used": value.get("model_used", ""),
                    }

                    # Handle all_model_predictions if present
                    if "all_model_predictions" in value:
                        all_preds = {}
                        for model_name, preds in value["all_model_predictions"].items():
                            if hasattr(preds, "tolist"):
                                all_preds[model_name] = preds.tolist()
                            else:
                                all_preds[model_name] = preds
                        serializable_results[key]["all_model_predictions"] = all_preds
                else:
                    # Handle other prediction formats
                    serializable_results[key] = value
            else:
                serializable_results[key] = value

        return serializable_results

    @staticmethod
    def save_demo_results(
        results: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """
        Save demo results to file

        Args:
            results: Results dictionary
            output_path: Optional output path

        Returns:
            str: Path to saved file
        """
        if output_path is None:
            timestamp = CommonUtils.get_timestamp()
            output_path = f"demo_results/{timestamp}.json"

        # Prepare results for serialization
        serializable_results = DemoUtils.prepare_results_for_serialization(results)

        # Add metadata
        serializable_results["demo_metadata"] = {
            "completion_status": "success",
            "timestamp": CommonUtils.get_readable_timestamp(),
            "total_time": results.get("total_time", 0),
        }

        # Save results
        FileUtils.save_json(serializable_results, output_path)
        DemoUtils._logger.info(f"✓ Demo results saved to: {output_path}")

        return output_path

    @staticmethod
    def load_latest_demo_results() -> Optional[Dict[str, Any]]:
        """
        Load the latest demo results from file system

        Returns:
            Dict containing the latest demo results or None if not found
        """
        import glob
        import os

        # Find all demo results files
        result_files = glob.glob("demo_results_*.json")
        if not result_files:
            return None

        # Get the most recent file
        latest_file = max(result_files, key=os.path.getmtime)
        try:
            return FileUtils.load_json(latest_file)
        except Exception as e:
            DemoUtils._logger.error(f"Error loading demo results: {str(e)}")
            return None
