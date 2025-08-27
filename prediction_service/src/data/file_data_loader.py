from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from common.logger.logger_factory import LoggerFactory, LoggerType

from ..core.config import get_settings
from ..interfaces.data_loader_interface import IDataLoader
from ..schemas.enums import TimeFrame


class FileDataLoader(IDataLoader):
    """File data loader implementation"""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            settings = get_settings()
            self.data_dir = settings.data_dir
        else:
            self.data_dir = Path(data_dir)

        self.logger = LoggerFactory.get_logger(
            name="file-data-loader",
            logger_type=LoggerType.STANDARD,
            log_file="logs/file_data_loader.log",
        )

    async def load_data(
        self, symbol: str, timeframe: TimeFrame, data_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Load data from file"""

        if data_path is None:
            # Try different naming patterns
            possible_names = [
                f"{symbol}_{timeframe.value}.csv",
                f"{symbol}_{timeframe}.csv",
                f"{symbol.upper()}_{timeframe.value}.csv",
                f"{symbol.lower()}_{timeframe.value}.csv",
            ]

            data_path = None
            for name in possible_names:
                potential_path = self.data_dir / name
                if potential_path.exists():
                    data_path = potential_path
                    break

            if data_path is None:
                raise FileNotFoundError(
                    f"No data file found for {symbol} {timeframe}. "
                    f"Tried: {possible_names}"
                )

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.logger.info(f"Loading data from {data_path}")

        try:
            df = pd.read_csv(data_path)

            # Ensure timestamp column exists and is properly formatted
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
            else:
                self.logger.warning("No timestamp column found in data")

            # Validate required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                # Try to find case variations
                df_lower = df.columns.str.lower()
                mapped_columns = {}
                for req_col in missing_columns:
                    matches = [
                        col for col in df.columns if col.lower() == req_col.lower()
                    ]
                    if matches:
                        mapped_columns[req_col] = matches[0]

                # Rename columns to standard format
                if mapped_columns:
                    df = df.rename(columns=mapped_columns)
                    self.logger.info(f"Mapped columns: {mapped_columns}")

                # Check again for missing columns
                still_missing = [
                    col for col in required_columns if col not in df.columns
                ]
                if still_missing:
                    raise ValueError(f"Missing required columns: {still_missing}")

            self.logger.info(f"Loaded {len(df)} rows of data")
            return df

        except Exception as e:
            self.logger.error(f"Failed to load data from {data_path}: {e}")
            raise

    async def check_data_exists(self, symbol: str, timeframe: TimeFrame) -> bool:
        """Check if data file exists"""

        # Try different naming patterns
        possible_names = [
            f"{symbol}_{timeframe.value}.csv",
            f"{symbol}_{timeframe}.csv",
            f"{symbol.upper()}_{timeframe.value}.csv",
            f"{symbol.lower()}_{timeframe.value}.csv",
        ]

        for name in possible_names:
            data_path = self.data_dir / name
            if data_path.exists():
                self.logger.info(f"Data exists for {symbol}_{timeframe.value}: True")
                return True

        self.logger.info(f"Data exists for {symbol}_{timeframe.value}: False")
        return False

    def split_data(
        self, data: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically into train/validation/test sets"""

        total_samples = len(data)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)

        train_data = data.iloc[:train_size].copy()
        val_data = data.iloc[train_size : train_size + val_size].copy()
        test_data = data.iloc[train_size + val_size :].copy()

        self.logger.info(
            f"Data split - Train: {len(train_data)}, "
            f"Val: {len(val_data)}, Test: {len(test_data)}"
        )

        return train_data, val_data, test_data
