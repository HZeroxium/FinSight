# data/data_loader.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict, Any, Union
from pathlib import Path
import warnings

from ..common.logger.logger_factory import LoggerFactory
from ..core.config import Config
from ..utils import ValidationUtils, FileUtils


class FinancialDataset(Dataset):
    """PyTorch Dataset for financial time series data with enhanced functionality"""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize dataset

        Args:
            sequences: Input sequences of shape (n_samples, seq_len, n_features)
            targets: Target values of shape (n_samples, prediction_horizon)
            metadata: Optional metadata about the dataset
        """
        ValidationUtils.validate_numeric_data(sequences)
        ValidationUtils.validate_numeric_data(targets)

        if len(sequences) != len(targets):
            raise ValueError(
                f"Sequences and targets length mismatch: {len(sequences)} vs {len(targets)}"
            )

        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.metadata = metadata or {}

    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index

        Args:
            idx: Sample index

        Returns:
            Tuple of (sequence, target)
        """
        return self.sequences[idx], self.targets[idx]

    def get_feature_dim(self) -> int:
        """Get number of features"""
        return self.sequences.shape[-1]

    def get_sequence_length(self) -> int:
        """Get sequence length"""
        return self.sequences.shape[1]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics

        Returns:
            Dictionary with dataset statistics
        """
        return {
            "n_samples": len(self),
            "sequence_length": self.get_sequence_length(),
            "n_features": self.get_feature_dim(),
            "target_shape": self.targets.shape,
            "sequences_mean": self.sequences.mean().item(),
            "sequences_std": self.sequences.std().item(),
            "targets_mean": self.targets.mean().item(),
            "targets_std": self.targets.std().item(),
        }


class FinancialDataLoader:
    """Enhanced data loader for financial time series data"""

    def __init__(self, config: Config):
        """
        Initialize data loader

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.data_stats: Dict[str, Any] = {}

    def load_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load data from CSV file with enhanced validation

        Args:
            file_path: Path to CSV file (optional, uses config if not provided)

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        if file_path is None:
            file_path = self.config.data.data_file

        file_path = Path(file_path)
        ValidationUtils.validate_file_path(
            file_path, must_exist=True, allowed_extensions=[".csv"]
        )

        self.logger.info(f"Loading data from {file_path}")

        try:
            # Load CSV data
            data = pd.read_csv(file_path)

            # Validate basic structure
            if data.empty:
                raise ValueError("Loaded data is empty")

            # Parse and validate date column
            if self.config.data.date_column in data.columns:
                data[self.config.data.date_column] = pd.to_datetime(
                    data[self.config.data.date_column], errors="coerce"
                )

                # Check for invalid dates
                invalid_dates = data[self.config.data.date_column].isna().sum()
                if invalid_dates > 0:
                    self.logger.warning(
                        f"Found {invalid_dates} invalid dates, dropping rows"
                    )
                    data = data.dropna(subset=[self.config.data.date_column])

                data = data.sort_values(self.config.data.date_column).reset_index(
                    drop=True
                )

                # Log date range
                self.logger.info(
                    f"Date range: {data[self.config.data.date_column].min()} to "
                    f"{data[self.config.data.date_column].max()}"
                )

            # Validate required columns for financial data
            required_financial_columns = ["Open", "High", "Low", "Close"]
            missing_columns = [
                col for col in required_financial_columns if col not in data.columns
            ]
            if missing_columns:
                self.logger.warning(
                    f"Missing typical financial columns: {missing_columns}"
                )

            # Store data statistics
            self.data_stats = {
                "total_records": len(data),
                "columns": len(data.columns),
                "date_range_days": (
                    (
                        data[self.config.data.date_column].max()
                        - data[self.config.data.date_column].min()
                    ).days
                    if self.config.data.date_column in data.columns
                    else None
                ),
                "missing_values": data.isnull().sum().sum(),
                "duplicate_rows": data.duplicated().sum(),
            }

            self.logger.info(
                f"Loaded {len(data)} records with {len(data.columns)} columns"
            )
            self.logger.info(f"Data statistics: {self.data_stats}")

            # Remove duplicates if any
            if self.data_stats["duplicate_rows"] > 0:
                self.logger.warning(
                    f"Removing {self.data_stats['duplicate_rows']} duplicate rows"
                )
                data = data.drop_duplicates().reset_index(drop=True)

            self.data = data

        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise

        return data

    def create_sequences(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        sequence_length: Optional[int] = None,
        prediction_horizon: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction with enhanced validation

        Args:
            data: Input dataframe
            feature_columns: List of feature column names
            target_column: Target column name
            sequence_length: Override sequence length from config
            prediction_horizon: Override prediction horizon from config

        Returns:
            tuple: (sequences, targets) arrays

        Raises:
            ValueError: If validation fails
        """
        sequence_length = sequence_length or self.config.model.sequence_length
        prediction_horizon = prediction_horizon or self.config.model.prediction_horizon

        # Validate inputs
        ValidationUtils.validate_dataframe(
            data, min_rows=sequence_length + prediction_horizon
        )

        missing_features = set(feature_columns) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        try:
            # Extract features and targets
            features = data[feature_columns].values
            targets = data[target_column].values

            # Validate numeric data
            ValidationUtils.validate_numeric_data(features, allow_nan=False)
            ValidationUtils.validate_numeric_data(targets, allow_nan=False)

            sequences = []
            target_values = []

            for i in range(len(data) - sequence_length - prediction_horizon + 1):
                # Input sequence
                seq = features[i : i + sequence_length]
                # Target value(s)
                target = targets[
                    i + sequence_length : i + sequence_length + prediction_horizon
                ]

                sequences.append(seq)
                target_values.append(target)

            sequences = np.array(sequences)
            target_values = np.array(target_values)

            # Validate output shapes
            expected_seq_shape = (len(sequences), sequence_length, len(feature_columns))
            expected_target_shape = (len(target_values), prediction_horizon)

            if sequences.shape != expected_seq_shape:
                raise ValueError(
                    f"Unexpected sequence shape: {sequences.shape} vs {expected_seq_shape}"
                )

            if target_values.shape != expected_target_shape:
                raise ValueError(
                    f"Unexpected target shape: {target_values.shape} vs {expected_target_shape}"
                )

            self.logger.info(f"Created {len(sequences)} sequences")
            self.logger.info(f"Sequence shape: {sequences.shape}")
            self.logger.info(f"Target shape: {target_values.shape}")

            # Store feature columns for later use
            self.feature_columns = feature_columns

        except Exception as e:
            self.logger.error(f"Error creating sequences: {str(e)}")
            raise

        return sequences, target_values

    def split_data(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],  # train
        Tuple[np.ndarray, np.ndarray],  # val
        Tuple[np.ndarray, np.ndarray],  # test
    ]:
        """
        Split data into train, validation, and test sets with validation

        Args:
            sequences: Input sequences
            targets: Target values
            train_ratio: Training data ratio (overrides config)
            val_ratio: Validation data ratio (overrides config)
            test_ratio: Test data ratio (overrides config)

        Returns:
            tuple: ((train_X, train_y), (val_X, val_y), (test_X, test_y))

        Raises:
            ValueError: If ratios don't sum to 1 or data is too small
        """
        # Use config values if not provided
        train_ratio = train_ratio or self.config.model.train_ratio
        val_ratio = val_ratio or self.config.model.val_ratio
        test_ratio = test_ratio or (1.0 - train_ratio - val_ratio)

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")

        if any(ratio <= 0 for ratio in [train_ratio, val_ratio, test_ratio]):
            raise ValueError("All split ratios must be positive")

        n_samples = len(sequences)

        if n_samples < 100:  # Minimum samples for meaningful split
            self.logger.warning(f"Very small dataset: {n_samples} samples")

        try:
            # Calculate split indices
            train_size = int(n_samples * train_ratio)
            val_size = int(n_samples * val_ratio)

            # Ensure minimum sizes
            train_size = max(train_size, 10)
            val_size = max(val_size, 5)
            test_size = n_samples - train_size - val_size

            if test_size < 5:
                raise ValueError(f"Test set too small: {test_size} samples")

            # Time series split (no shuffling to maintain temporal order)
            train_sequences = sequences[:train_size]
            train_targets = targets[:train_size]

            val_sequences = sequences[train_size : train_size + val_size]
            val_targets = targets[train_size : train_size + val_size]

            test_sequences = sequences[train_size + val_size :]
            test_targets = targets[train_size + val_size :]

            self.logger.info(
                f"Data split - Train: {len(train_sequences)} ({len(train_sequences)/n_samples:.1%}), "
                f"Val: {len(val_sequences)} ({len(val_sequences)/n_samples:.1%}), "
                f"Test: {len(test_sequences)} ({len(test_sequences)/n_samples:.1%})"
            )

            # Validate splits
            splits = [
                (train_sequences, train_targets, "train"),
                (val_sequences, val_targets, "validation"),
                (test_sequences, test_targets, "test"),
            ]

            for seq, tgt, name in splits:
                if len(seq) == 0:
                    raise ValueError(f"{name} set is empty")
                if seq.shape[0] != tgt.shape[0]:
                    raise ValueError(f"{name} set sequence/target length mismatch")

        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

        return (
            (train_sequences, train_targets),
            (val_sequences, val_targets),
            (test_sequences, test_targets),
        )

    def create_data_loaders(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders with enhanced configuration

        Args:
            train_data: Training data tuple (X, y)
            val_data: Validation data tuple (X, y)
            test_data: Test data tuple (X, y)
            batch_size: Batch size (overrides config)
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        batch_size = batch_size or self.config.model.batch_size

        try:
            # Create datasets with metadata
            train_dataset = FinancialDataset(
                *train_data,
                metadata={"split": "train", "feature_columns": self.feature_columns},
            )
            val_dataset = FinancialDataset(
                *val_data,
                metadata={
                    "split": "validation",
                    "feature_columns": self.feature_columns,
                },
            )
            test_dataset = FinancialDataset(
                *test_data,
                metadata={"split": "test", "feature_columns": self.feature_columns},
            )

            # Log dataset statistics
            for dataset, name in [
                (train_dataset, "train"),
                (val_dataset, "val"),
                (test_dataset, "test"),
            ]:
                stats = dataset.get_stats()
                self.logger.info(f"{name.capitalize()} dataset stats: {stats}")

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=True,  # Drop last incomplete batch for training
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=False,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=False,
            )

            self.logger.info(f"Created data loaders with batch size {batch_size}")

        except Exception as e:
            self.logger.error(f"Error creating data loaders: {str(e)}")
            raise

        return train_loader, val_loader, test_loader

    def prepare_data(
        self,
        processed_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare complete data pipeline with validation

        Args:
            processed_data: Preprocessed dataframe
            feature_columns: Override feature columns from config
            target_column: Override target column from config

        Returns:
            tuple: (train_loader, val_loader, test_loader)

        Raises:
            ValueError: If required columns are missing
        """
        feature_columns = feature_columns or self.config.model.features_to_use
        target_column = target_column or self.config.model.target_column

        # Validate columns exist
        missing_cols = set(feature_columns + [target_column]) - set(
            processed_data.columns
        )
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        try:
            # Create sequences
            sequences, targets = self.create_sequences(
                processed_data, feature_columns, target_column
            )

            # Split data
            train_data, val_data, test_data = self.split_data(sequences, targets)

            # Create data loaders
            return self.create_data_loaders(train_data, val_data, test_data)

        except Exception as e:
            self.logger.error(f"Error in data preparation pipeline: {str(e)}")
            raise

    def get_feature_dim(self) -> int:
        """
        Get the number of features

        Returns:
            int: Number of features
        """
        return len(self.feature_columns) if self.feature_columns else 0

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive data information

        Returns:
            dict: Data information including statistics and metadata
        """
        info = {
            "data_stats": self.data_stats,
            "feature_columns": self.feature_columns,
            "n_features": self.get_feature_dim(),
            "config": {
                "sequence_length": self.config.model.sequence_length,
                "prediction_horizon": self.config.model.prediction_horizon,
                "batch_size": self.config.model.batch_size,
                "data_splits": {
                    "train_ratio": self.config.model.train_ratio,
                    "val_ratio": self.config.model.val_ratio,
                },
            },
        }

        if self.data is not None:
            info["data_shape"] = self.data.shape
            info["data_columns"] = self.data.columns.tolist()

        return info

    def save_preprocessed_data(self, filepath: Union[str, Path]) -> None:
        """
        Save preprocessed data to file

        Args:
            filepath: Path to save the data
        """
        if self.processed_data is None:
            raise ValueError("No processed data to save")

        try:
            filepath = Path(filepath)
            FileUtils.ensure_dir(filepath.parent)

            if filepath.suffix.lower() == ".csv":
                self.processed_data.to_csv(filepath, index=False)
            elif filepath.suffix.lower() in [".pkl", ".pickle"]:
                FileUtils.save_object(self.processed_data, filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")

            self.logger.info(f"Preprocessed data saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving preprocessed data: {str(e)}")
            raise

    def load_preprocessed_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load preprocessed data from file

        Args:
            filepath: Path to load the data from

        Returns:
            pd.DataFrame: Loaded preprocessed data
        """
        try:
            filepath = Path(filepath)
            ValidationUtils.validate_file_path(filepath, must_exist=True)

            if filepath.suffix.lower() == ".csv":
                data = pd.read_csv(filepath)
            elif filepath.suffix.lower() in [".pkl", ".pickle"]:
                data = FileUtils.load_object(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")

            self.processed_data = data
            self.logger.info(f"Preprocessed data loaded from {filepath}")

            return data

        except Exception as e:
            self.logger.error(f"Error loading preprocessed data: {str(e)}")
            raise
