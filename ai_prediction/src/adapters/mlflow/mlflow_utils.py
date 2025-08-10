"""
MLflow-specific utilities and helpers for experiment tracking
"""

from typing import Dict, Any, Set, Optional
from datetime import datetime
import asyncio
from pathlib import Path

from common.logger.logger_factory import LoggerFactory


class MLflowRunState:
    """Manages state for an MLflow run to prevent duplicate parameter logging"""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.logged_params: Set[str] = set()
        self.logged_tags: Set[str] = set()
        self.created_at = datetime.now()
        self.logger = LoggerFactory.get_logger(f"MLflowRunState-{run_id[:8]}")

    def can_log_param(self, key: str) -> bool:
        """Check if parameter can be logged (not already logged)"""
        return key not in self.logged_params

    def mark_param_logged(self, key: str) -> None:
        """Mark parameter as logged"""
        self.logged_params.add(key)

    def can_log_tag(self, key: str) -> bool:
        """Check if tag can be logged (not already logged)"""
        return key not in self.logged_tags

    def mark_tag_logged(self, key: str) -> None:
        """Mark tag as logged"""
        self.logged_tags.add(key)


class MLflowRunManager:
    """Manages MLflow run states to prevent duplicate parameter/tag logging"""

    def __init__(self):
        self.run_states: Dict[str, MLflowRunState] = {}
        self.logger = LoggerFactory.get_logger("MLflowRunManager")

    def create_run_state(self, run_id: str) -> MLflowRunState:
        """Create and register a new run state"""
        if run_id in self.run_states:
            self.logger.warning(f"Run state already exists for {run_id}, reusing")
            return self.run_states[run_id]

        run_state = MLflowRunState(run_id)
        self.run_states[run_id] = run_state
        self.logger.debug(f"Created run state for {run_id}")
        return run_state

    def get_run_state(self, run_id: str) -> Optional[MLflowRunState]:
        """Get run state if it exists"""
        return self.run_states.get(run_id)

    def remove_run_state(self, run_id: str) -> None:
        """Remove run state (when run ends)"""
        if run_id in self.run_states:
            del self.run_states[run_id]
            self.logger.debug(f"Removed run state for {run_id}")

    def cleanup_old_states(self, max_age_hours: int = 24) -> None:
        """Clean up old run states"""
        current_time = datetime.now()
        to_remove = []

        for run_id, state in self.run_states.items():
            age_hours = (current_time - state.created_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                to_remove.append(run_id)

        for run_id in to_remove:
            self.remove_run_state(run_id)
            self.logger.info(f"Cleaned up old run state for {run_id}")


# Global run manager instance
_run_manager = MLflowRunManager()


def get_run_manager() -> MLflowRunManager:
    """Get the global run manager instance"""
    return _run_manager


class MLflowParameterDeduplicator:
    """Utility to deduplicate parameters before logging to MLflow"""

    @staticmethod
    def deduplicate_config_params(
        config_dict: Dict[str, Any], base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Remove duplicate parameters from config_dict that are already in base_params

        Args:
            config_dict: Configuration parameters to log
            base_params: Parameters that have already been logged

        Returns:
            Dict with duplicates removed
        """
        deduplicated = {}

        for key, value in config_dict.items():
            if key not in base_params:
                deduplicated[key] = value
            else:
                # Only add if values are different (in case we want to log a change)
                if base_params[key] != value:
                    # Create a new key to avoid collision
                    new_key = f"{key}_updated"
                    deduplicated[new_key] = value

        return deduplicated

    @staticmethod
    def create_training_params(
        symbol: str, timeframe: str, model_type: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create standardized training parameters for MLflow logging

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            model_type: Model type
            config: Model configuration

        Returns:
            Dict of parameters ready for logging
        """
        # Base parameters
        base_params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_type": model_type,
        }

        # Add config parameters with prefix to avoid conflicts
        config_params = {}
        for key, value in config.items():
            # Convert complex types to strings
            if isinstance(value, (list, dict)):
                config_params[f"config_{key}"] = str(value)
            else:
                config_params[f"config_{key}"] = value

        return {**base_params, **config_params}


class MLflowArtifactHelper:
    """Helper for managing MLflow artifacts"""

    @staticmethod
    def prepare_model_artifacts(model_path: str | Path) -> Dict[str, Any]:
        """
        Prepare model artifacts for MLflow logging

        Args:
            model_path: Path to the model file or directory

        Returns:
            Dict with artifact information
        """
        model_file = Path(model_path)

        if not model_file.exists():
            return {"error": f"Model path does not exist: {model_path}"}

        artifacts_info = {
            "model_path": str(model_file),
            "is_file": model_file.is_file(),
            "is_directory": model_file.is_dir(),
        }

        if model_file.is_file():
            artifacts_info["file_size"] = model_file.stat().st_size
            artifacts_info["artifact_type"] = "file"
        elif model_file.is_dir():
            # Count files in directory
            files = list(model_file.rglob("*"))
            artifacts_info["total_files"] = len([f for f in files if f.is_file()])
            artifacts_info["total_size"] = sum(
                f.stat().st_size for f in files if f.is_file()
            )
            artifacts_info["artifact_type"] = "directory"

        return artifacts_info

    @staticmethod
    def create_model_signature(
        input_shape: tuple, output_shape: tuple
    ) -> Dict[str, Any]:
        """
        Create model signature information for MLflow

        Args:
            input_shape: Input tensor shape
            output_shape: Output tensor shape

        Returns:
            Dict with signature information
        """
        return {
            "input_shape": str(input_shape),
            "output_shape": str(output_shape),
            "input_features": input_shape[-1] if len(input_shape) > 1 else 1,
            "sequence_length": input_shape[-2] if len(input_shape) > 2 else 1,
        }


class MLflowMetricsValidator:
    """Validator for MLflow metrics to ensure proper data types"""

    @staticmethod
    def validate_and_convert_metric(key: str, value: Any) -> tuple[str, float | None]:
        """
        Validate and convert metric value to float

        Args:
            key: Metric name
            value: Metric value

        Returns:
            Tuple of (validated_key, converted_value or None if invalid)
        """
        logger = LoggerFactory.get_logger("MLflowMetricsValidator")

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                logger.warning(f"Empty list/tuple for metric {key}")
                return key, None

            # Convert list to average or first value
            numeric_values = [v for v in value if isinstance(v, (int, float))]
            if not numeric_values:
                logger.warning(f"No numeric values in list for metric {key}: {value}")
                return key, None

            converted_value = sum(numeric_values) / len(numeric_values)
            return key, float(converted_value)

        elif isinstance(value, (int, float)):
            return key, float(value)

        elif isinstance(value, str):
            try:
                return key, float(value)
            except ValueError:
                logger.warning(
                    f"Cannot convert string to float for metric {key}: {value}"
                )
                return key, None

        else:
            logger.warning(f"Unsupported metric type for {key}: {type(value)} {value}")
            return key, None

    @staticmethod
    def validate_metrics_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate and convert all metrics in a dictionary

        Args:
            metrics: Dictionary of metrics

        Returns:
            Dictionary of validated metrics (invalid ones removed)
        """
        validated_metrics = {}

        for key, value in metrics.items():
            validated_key, validated_value = (
                MLflowMetricsValidator.validate_and_convert_metric(key, value)
            )

            if validated_value is not None:
                validated_metrics[validated_key] = validated_value

        return validated_metrics
