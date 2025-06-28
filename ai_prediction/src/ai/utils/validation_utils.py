import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import warnings

from common.logger.logger_factory import LoggerFactory


class ValidationUtils:
    """Utility class for data validation and type checking"""

    _logger = LoggerFactory.get_logger(__name__)

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
    ) -> bool:
        """
        Validate DataFrame structure and content

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required

        Returns:
            bool: True if valid

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if len(df) < min_rows:
            raise ValueError(
                f"DataFrame must have at least {min_rows} rows, got {len(df)}"
            )

        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for any completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            ValidationUtils._logger.warning(
                f"Found completely empty columns: {empty_columns}"
            )

        ValidationUtils._logger.debug(f"DataFrame validation passed: {df.shape}")
        return True

    @staticmethod
    def validate_numeric_data(
        data: Union[np.ndarray, pd.Series, torch.Tensor],
        allow_nan: bool = False,
        allow_inf: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> bool:
        """
        Validate numeric data array

        Args:
            data: Numeric data to validate
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            bool: True if valid

        Raises:
            ValueError: If validation fails
        """
        # Convert to numpy for uniform handling
        if isinstance(data, torch.Tensor):
            np_data = data.detach().cpu().numpy()
        elif isinstance(data, pd.Series):
            np_data = data.values
        else:
            np_data = np.asarray(data)

        # Check for NaN values
        if not allow_nan and np.isnan(np_data).any():
            nan_count = np.isnan(np_data).sum()
            raise ValueError(f"Found {nan_count} NaN values in data")

        # Check for infinite values
        if not allow_inf and np.isinf(np_data).any():
            inf_count = np.isinf(np_data).sum()
            raise ValueError(f"Found {inf_count} infinite values in data")

        # Check value ranges (only for finite values)
        finite_mask = np.isfinite(np_data)
        finite_data = np_data[finite_mask]

        if len(finite_data) > 0:
            if min_value is not None and np.any(finite_data < min_value):
                raise ValueError(f"Found values below minimum {min_value}")

            if max_value is not None and np.any(finite_data > max_value):
                raise ValueError(f"Found values above maximum {max_value}")

        ValidationUtils._logger.debug(
            f"Numeric data validation passed: shape {np_data.shape}"
        )
        return True

    @staticmethod
    def validate_tensor_shape(
        tensor: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        min_dims: Optional[int] = None,
        max_dims: Optional[int] = None,
    ) -> bool:
        """
        Validate tensor shape

        Args:
            tensor: Tensor to validate
            expected_shape: Expected exact shape (None for any dimension)
            min_dims: Minimum number of dimensions
            max_dims: Maximum number of dimensions

        Returns:
            bool: True if valid

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")

        actual_shape = tensor.shape
        ndims = len(actual_shape)

        # Check number of dimensions
        if min_dims is not None and ndims < min_dims:
            raise ValueError(
                f"Tensor must have at least {min_dims} dimensions, got {ndims}"
            )

        if max_dims is not None and ndims > max_dims:
            raise ValueError(
                f"Tensor must have at most {max_dims} dimensions, got {ndims}"
            )

        # Check exact shape if provided
        if expected_shape is not None:
            if len(expected_shape) != ndims:
                raise ValueError(
                    f"Shape dimension mismatch: expected {len(expected_shape)}, got {ndims}"
                )

            for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                if expected is not None and expected != actual:
                    raise ValueError(
                        f"Shape mismatch at dimension {i}: expected {expected}, got {actual}"
                    )

        ValidationUtils._logger.debug(f"Tensor shape validation passed: {actual_shape}")
        return True

    @staticmethod
    def validate_file_path(
        path: Union[str, Path],
        must_exist: bool = True,
        allowed_extensions: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate file path

        Args:
            path: File path to validate
            must_exist: Whether file must exist
            allowed_extensions: List of allowed file extensions

        Returns:
            bool: True if valid

        Raises:
            ValueError: If validation fails
        """
        path = Path(path)

        if must_exist and not path.exists():
            raise ValueError(f"File does not exist: {path}")

        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                raise ValueError(
                    f"File extension {path.suffix} not in allowed extensions: {allowed_extensions}"
                )

        ValidationUtils._logger.debug(f"File path validation passed: {path}")
        return True

    @staticmethod
    def validate_config_dict(
        config: Dict[str, Any],
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate configuration dictionary

        Args:
            config: Configuration dictionary
            required_keys: List of required keys
            optional_keys: List of optional keys (if provided, only these + required keys are allowed)

        Returns:
            bool: True if valid

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        # Check required keys
        missing_keys = set(required_keys) - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        # Check for unknown keys if optional_keys is specified
        if optional_keys is not None:
            allowed_keys = set(required_keys) | set(optional_keys)
            unknown_keys = set(config.keys()) - allowed_keys
            if unknown_keys:
                ValidationUtils._logger.warning(f"Unknown config keys: {unknown_keys}")

        ValidationUtils._logger.debug("Configuration validation passed")
        return True

    @staticmethod
    def check_data_leakage(
        train_data: Union[pd.DataFrame, np.ndarray],
        val_data: Union[pd.DataFrame, np.ndarray],
        test_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        index_col: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Check for data leakage between train/validation/test sets

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data (optional)
            index_col: Index column name for DataFrames

        Returns:
            dict: Results of leakage checks
        """
        results = {}

        def _get_index(data):
            if isinstance(data, pd.DataFrame):
                return data[index_col].values if index_col else data.index.values
            else:
                return np.arange(len(data))

        try:
            train_idx = set(_get_index(train_data))
            val_idx = set(_get_index(val_data))

            # Check train-validation overlap
            train_val_overlap = len(train_idx & val_idx)
            results["train_val_leakage"] = train_val_overlap > 0

            if train_val_overlap > 0:
                ValidationUtils._logger.warning(
                    f"Found {train_val_overlap} overlapping samples between train and validation"
                )

            # Check with test data if provided
            if test_data is not None:
                test_idx = set(_get_index(test_data))

                train_test_overlap = len(train_idx & test_idx)
                val_test_overlap = len(val_idx & test_idx)

                results["train_test_leakage"] = train_test_overlap > 0
                results["val_test_leakage"] = val_test_overlap > 0

                if train_test_overlap > 0:
                    ValidationUtils._logger.warning(
                        f"Found {train_test_overlap} overlapping samples between train and test"
                    )

                if val_test_overlap > 0:
                    ValidationUtils._logger.warning(
                        f"Found {val_test_overlap} overlapping samples between validation and test"
                    )

        except Exception as e:
            ValidationUtils._logger.error(f"Error checking data leakage: {str(e)}")
            results["error"] = str(e)

        return results

    @staticmethod
    def validate_model_inputs(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        expected_output_shape: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        """
        Validate model can process input and produce expected output

        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            expected_output_shape: Expected output shape

        Returns:
            bool: True if valid

        Raises:
            ValueError: If validation fails
        """
        try:
            model.eval()
            with torch.no_grad():
                output = model(sample_input)

            if expected_output_shape is not None:
                if output.shape != expected_output_shape:
                    raise ValueError(
                        f"Model output shape {output.shape} != expected {expected_output_shape}"
                    )

            ValidationUtils._logger.debug(
                f"Model validation passed: input {sample_input.shape} -> output {output.shape}"
            )
            return True

        except Exception as e:
            ValidationUtils._logger.error(f"Model validation failed: {str(e)}")
            raise ValueError(f"Model validation failed: {str(e)}")
