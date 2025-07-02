# utils/data_storage.py

"""
Data storage utilities for saving market data to various formats.
Supports JSON, CSV, and Parquet formats for dataset creation.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from ..common.logger import LoggerFactory, LoggerType, LogLevel


class MarketDataStorage:
    """Utility class for saving market data in various formats"""

    def __init__(self, base_dir: str = "data", logger_name: str = "data_storage"):
        """
        Initialize DataStorage with base directory and logger

        Args:
            base_dir: Base directory for saving data files
            logger_name: Name for the logger instance
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.logger = LoggerFactory.get_logger(
            name=logger_name,
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            use_colors=True,
        )

    def save_json(
        self,
        data: Union[Dict, List],
        filename: str,
        subfolder: Optional[str] = None,
        timestamp_suffix: bool = True,
    ) -> Path:
        """
        Save data as JSON file

        Args:
            data: Data to save (dict or list)
            filename: Base filename (without extension)
            subfolder: Optional subfolder within base_dir
            timestamp_suffix: Whether to add timestamp to filename

        Returns:
            Path to saved file
        """
        try:
            # Create subfolder if specified
            save_dir = self.base_dir
            if subfolder:
                save_dir = self.base_dir / subfolder
                save_dir.mkdir(parents=True, exist_ok=True)

            # Add timestamp if requested
            if timestamp_suffix:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename}_{timestamp}"

            file_path = save_dir / f"{filename}.json"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Saved JSON data to {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Failed to save JSON data: {e}")
            raise

    def save_csv(
        self,
        data: Union[List[Dict], pd.DataFrame],
        filename: str,
        subfolder: Optional[str] = None,
        timestamp_suffix: bool = True,
        index: bool = False,
    ) -> Path:
        """
        Save data as CSV file

        Args:
            data: Data to save (list of dicts or DataFrame)
            filename: Base filename (without extension)
            subfolder: Optional subfolder within base_dir
            timestamp_suffix: Whether to add timestamp to filename
            index: Whether to include index in CSV

        Returns:
            Path to saved file
        """
        try:
            # Create subfolder if specified
            save_dir = self.base_dir
            if subfolder:
                save_dir = self.base_dir / subfolder
                save_dir.mkdir(parents=True, exist_ok=True)

            # Add timestamp if requested
            if timestamp_suffix:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename}_{timestamp}"

            file_path = save_dir / f"{filename}.csv"

            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            df.to_csv(file_path, index=index, encoding="utf-8")

            self.logger.info(f"Saved CSV data to {file_path} ({len(df)} rows)")
            return file_path

        except Exception as e:
            self.logger.error(f"Failed to save CSV data: {e}")
            raise

    def save_parquet(
        self,
        data: Union[List[Dict], pd.DataFrame],
        filename: str,
        subfolder: Optional[str] = None,
        timestamp_suffix: bool = True,
    ) -> Path:
        """
        Save data as Parquet file (efficient for large datasets)

        Args:
            data: Data to save (list of dicts or DataFrame)
            filename: Base filename (without extension)
            subfolder: Optional subfolder within base_dir
            timestamp_suffix: Whether to add timestamp to filename

        Returns:
            Path to saved file
        """
        try:
            # Create subfolder if specified
            save_dir = self.base_dir
            if subfolder:
                save_dir = self.base_dir / subfolder
                save_dir.mkdir(parents=True, exist_ok=True)

            # Add timestamp if requested
            if timestamp_suffix:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename}_{timestamp}"

            file_path = save_dir / f"{filename}.parquet"

            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            df.to_parquet(file_path, engine="pyarrow", index=False)

            self.logger.info(f"Saved Parquet data to {file_path} ({len(df)} rows)")
            return file_path

        except Exception as e:
            self.logger.error(f"Failed to save Parquet data: {e}")
            raise

    def create_dataset_structure(self, exchange_name: str) -> Dict[str, Path]:
        """
        Create standard dataset directory structure for an exchange

        Args:
            exchange_name: Name of the exchange

        Returns:
            Dictionary mapping data type to directory path
        """
        base_exchange_dir = self.base_dir / exchange_name

        structure = {
            "ohlcv": base_exchange_dir / "ohlcv",
            "tickers": base_exchange_dir / "tickers",
            "trades": base_exchange_dir / "trades",
            "orderbook": base_exchange_dir / "orderbook",
            "markets": base_exchange_dir / "markets",
            "symbols": base_exchange_dir / "symbols",
            "klines": base_exchange_dir / "klines",
            "depth": base_exchange_dir / "depth",
            "raw": base_exchange_dir / "raw",
        }

        # Create all directories
        for data_type, path in structure.items():
            path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Created dataset structure for {exchange_name}")
        return structure
