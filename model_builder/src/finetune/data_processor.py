# finetune/data_processor.py

"""
Modern data processor for financial time series using HuggingFace datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from transformers import PreTrainedTokenizer
from sklearn.preprocessing import RobustScaler

from ..common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel
from .config import FineTuneConfig, TaskType


class FinancialTimeSeriesDataset(Dataset):
    """PyTorch Dataset for financial time series data"""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: FineTuneConfig = None,
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        target = self.targets[idx]

        # For time series models, return numerical data directly
        if self.tokenizer is None or self.config.task_type == TaskType.FORECASTING:
            return {
                "input_ids": sequence.flatten(),  # Use input_ids for consistency
                "targets": target,
                "attention_mask": torch.ones(sequence.shape[0]),
                "labels": target,  # Add labels for trainer compatibility
            }

        # For language models, convert to text format
        text_input = self._format_sequence_as_text(sequence)

        inputs = self.tokenizer(
            text_input,
            truncation=True,
            padding="max_length",
            max_length=self.config.sequence_length,
            return_tensors="pt",
        )

        target_text = (
            f"{target.item():.4f}" if target.dim() == 0 else f"{target[0].item():.4f}"
        )
        labels = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=32,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
            "targets": target,
        }

    def _format_sequence_as_text(self, sequence: torch.Tensor) -> str:
        """Convert numerical sequence to text format for language models"""
        values = sequence.numpy().flatten()[-10:]  # Use last 10 values
        formatted_values = [f"{val:.4f}" for val in values]
        return f"Financial sequence: {', '.join(formatted_values)}. Predict next value:"


class FinancialDataProcessor:
    """Modern data processor for financial time series fine-tuning"""

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger(
            name="FinancialDataProcessor",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
        )
        self.scaler = RobustScaler()
        self.fitted = False

    def load_and_prepare_data(self, data_path: Path) -> DatasetDict:
        """
        Load and prepare financial data for fine-tuning

        Args:
            data_path: Path to the CSV data file

        Returns:
            DatasetDict with train/validation/test splits
        """
        self.logger.info(f"Loading data from {data_path}")

        # Load raw data
        df = pd.read_csv(data_path)

        # Handle column name variations
        if "timestamp" not in df.columns:
            df.columns = df.columns.str.lower()

        # Basic preprocessing
        df = self._preprocess_data(df)

        # Create sequences
        sequences, targets = self._create_sequences(df)

        # Split data
        train_data, val_data, test_data = self._split_data(sequences, targets)

        # Create HuggingFace datasets
        dataset_dict = DatasetDict(
            {
                "train": HFDataset.from_dict(
                    {"sequences": train_data[0], "targets": train_data[1]}
                ),
                "validation": HFDataset.from_dict(
                    {"sequences": val_data[0], "targets": val_data[1]}
                ),
                "test": HFDataset.from_dict(
                    {"sequences": test_data[0], "targets": test_data[1]}
                ),
            }
        )

        self.logger.info(
            f"✓ Data prepared: {len(dataset_dict['train'])} train, "
            f"{len(dataset_dict['validation'])} val, {len(dataset_dict['test'])} test"
        )

        return dataset_dict

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw financial data"""
        # Ensure required columns exist (handle case variations)
        available_cols = df.columns.tolist()

        # Map config features to available columns
        feature_mapping = {}
        for feature in self.config.features:
            if feature in available_cols:
                feature_mapping[feature] = feature
            else:
                # Try case variations
                for col in available_cols:
                    if col.lower() == feature.lower():
                        feature_mapping[feature] = col
                        break

        # Update config features with actual column names
        mapped_features = [feature_mapping.get(f, f) for f in self.config.features]
        target_col = feature_mapping.get(
            self.config.target_column, self.config.target_column
        )

        # Check for missing columns
        required_cols = mapped_features + [target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Try to find alternatives
            self.logger.warning(
                f"Missing columns: {missing_cols}, using available columns"
            )
            available_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(available_numeric) >= len(self.config.features):
                mapped_features = available_numeric[: len(self.config.features)]
                target_col = (
                    available_numeric[0] if target_col not in df.columns else target_col
                )
                self.logger.info(
                    f"Using alternative features: {mapped_features}, target: {target_col}"
                )
            else:
                raise ValueError(
                    f"Not enough numeric columns available: {available_numeric}"
                )

        # Update features for this session
        self.config.features = mapped_features
        self.config.target_column = target_col

        # Handle missing values
        df = df.dropna(subset=required_cols)

        # Sort by timestamp if available
        timestamp_cols = [
            col for col in df.columns if "time" in col.lower() or "date" in col.lower()
        ]
        if timestamp_cols:
            df[timestamp_cols[0]] = pd.to_datetime(
                df[timestamp_cols[0]], errors="coerce"
            )
            df = df.sort_values(timestamp_cols[0]).reset_index(drop=True)

        # Normalize features
        feature_data = df[mapped_features].values
        if not self.fitted:
            df[mapped_features] = self.scaler.fit_transform(feature_data)
            self.fitted = True
        else:
            df[mapped_features] = self.scaler.transform(feature_data)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators"""
        if "close" in df.columns:
            # Simple moving averages
            df["sma_5"] = df["close"].rolling(window=5).mean()
            df["sma_20"] = df["close"].rolling(window=20).mean()

            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            # Volatility
            df["volatility"] = df["close"].rolling(window=20).std()

        # Fill NaN values created by indicators
        df = df.bfill().ffill()

        return df

    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        feature_data = df[self.config.features].values
        target_data = df[self.config.target_column].values

        sequences = []
        targets = []

        for i in range(
            self.config.sequence_length, len(df) - self.config.prediction_horizon + 1
        ):
            # Input sequence
            seq = feature_data[i - self.config.sequence_length : i]
            sequences.append(seq)

            # Target (next value(s))
            if self.config.prediction_horizon == 1:
                target = target_data[i]
            else:
                target = target_data[i : i + self.config.prediction_horizon]
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def _split_data(
        self, sequences: np.ndarray, targets: np.ndarray
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """Split data into train/validation/test sets"""
        n_samples = len(sequences)

        train_end = int(n_samples * self.config.train_split)
        val_end = int(n_samples * (self.config.train_split + self.config.val_split))

        train_data = (sequences[:train_end], targets[:train_end])
        val_data = (sequences[train_end:val_end], targets[train_end:val_end])
        test_data = (sequences[val_end:], targets[val_end:])

        return train_data, val_data, test_data

    def create_torch_datasets(
        self, dataset_dict: DatasetDict, tokenizer: Optional[PreTrainedTokenizer] = None
    ) -> Dict[str, FinancialTimeSeriesDataset]:
        """Create PyTorch datasets from HuggingFace DatasetDict"""
        torch_datasets = {}

        for split_name, dataset in dataset_dict.items():
            sequences = np.array(dataset["sequences"])
            targets = np.array(dataset["targets"])

            torch_datasets[split_name] = FinancialTimeSeriesDataset(
                sequences=sequences,
                targets=targets,
                tokenizer=tokenizer,
                config=self.config,
            )

        return torch_datasets
