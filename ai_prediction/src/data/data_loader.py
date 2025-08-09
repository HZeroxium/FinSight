# data/data_loader.py

import os
import tempfile
import asyncio
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import pandas as pd
from pathlib import Path
from ..interfaces.data_loader_interface import IDataLoader
from ..schemas.enums import TimeFrame
from common.logger.logger_factory import LoggerFactory, LoggerType, LogLevel
from ..core.config import get_settings
from ..utils.storage_client import StorageClient


class CloudDataLoader(IDataLoader):
    """
    Cloud-first data loader with local fallback.

    This loader attempts to load data from cloud storage first, then falls back
    to local files if cloud data is not available. It also implements intelligent
    caching to improve performance and updates local storage for fallback scenarios.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        settings = get_settings()

        if data_dir is None:
            self.data_dir = settings.data_dir
        else:
            self.data_dir = Path(data_dir)

        self.settings = settings
        self.cache_dir = settings.cloud_data_cache_dir
        self.cache_ttl_hours = settings.cloud_data_cache_ttl_hours

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name="cloud-data-loader",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/cloud_data_loader.log",
        )

        # Initialize storage client
        self.storage_client = None
        self.cloud_enabled = False

        try:
            storage_config = settings.get_storage_config()
            self.storage_client = StorageClient(**storage_config)
            self.cloud_enabled = True
            self.logger.info(
                f"Cloud storage initialized: {storage_config['bucket_name']}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize cloud storage: {e}")
            self.cloud_enabled = False

    async def load_data(
        self, symbol: str, timeframe: TimeFrame, data_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load data with cloud-first approach.

        Priority order:
        1. Valid cached cloud data (if within TTL)
        2. Fresh cloud data (download and cache, update local)
        3. Local data files (fallback)
        """
        self.logger.info(f"Loading data for {symbol}_{timeframe.value}")

        # If specific path provided, use traditional file loading
        if data_path is not None:
            return await self._load_from_file(data_path)

        # Try cloud-first approach
        if self.cloud_enabled:
            try:
                # Check cache first
                cached_data = await self._load_from_cache(symbol, timeframe)
                if cached_data is not None:
                    self.logger.info(
                        f"Loaded data from cache for {symbol}_{timeframe.value}"
                    )
                    return cached_data

                # Try to download from cloud
                cloud_data = await self._load_from_cloud(symbol, timeframe)
                if cloud_data is not None:
                    # Cache the downloaded data
                    await self._cache_data(symbol, timeframe, cloud_data)

                    # Update local storage for fallback scenarios
                    await self._update_local_storage(symbol, timeframe, cloud_data)

                    self.logger.info(
                        f"Loaded data from cloud for {symbol}_{timeframe.value}"
                    )
                    return cloud_data

            except Exception as e:
                self.logger.warning(f"Cloud data loading failed: {e}")

        # Fallback to local data
        self.logger.info(f"Falling back to local data for {symbol}_{timeframe.value}")
        return await self._load_from_local(symbol, timeframe)

    async def check_data_exists(self, symbol: str, timeframe: TimeFrame) -> bool:
        """Check if data exists in cloud, cache, or local storage."""

        # Check cache first
        if await self._check_cache_exists(symbol, timeframe):
            return True

        # Check cloud storage
        if self.cloud_enabled and await self._check_cloud_exists(symbol, timeframe):
            return True

        # Check local storage
        return await self._check_local_exists(symbol, timeframe)

    def split_data(
        self, data: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically into train/validation/test sets."""
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

    # ===== Cloud Storage Methods =====

    async def _load_from_cloud(
        self, symbol: str, timeframe: TimeFrame
    ) -> Optional[pd.DataFrame]:
        """Load data from cloud storage using storage client."""
        if not self.cloud_enabled or not self.storage_client:
            return None

        try:
            # Generate cloud object keys following backtesting service pattern
            object_keys = self._generate_cloud_object_keys(symbol, timeframe)

            for object_key in object_keys:
                try:
                    # Check if object exists
                    if not await self.storage_client.object_exists(object_key):
                        continue

                    # Download to temporary file
                    with tempfile.NamedTemporaryFile(
                        suffix=".csv", delete=False
                    ) as tmp_file:
                        success = await self.storage_client.download_file(
                            object_key=object_key,
                            local_file_path=tmp_file.name,
                        )

                        if success:
                            # Load the data
                            data = await self._load_from_file(Path(tmp_file.name))

                            # Clean up temp file
                            os.unlink(tmp_file.name)

                            self.logger.info(
                                f"Successfully loaded from cloud: {object_key}"
                            )
                            return data

                except Exception as e:
                    self.logger.warning(f"Error downloading {object_key}: {e}")
                    continue

            self.logger.debug(f"No cloud data found for {symbol}_{timeframe.value}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to load from cloud: {e}")
            return None

    async def _check_cloud_exists(self, symbol: str, timeframe: TimeFrame) -> bool:
        """Check if data exists in cloud storage."""
        if not self.cloud_enabled or not self.storage_client:
            return False

        try:
            object_keys = self._generate_cloud_object_keys(symbol, timeframe)

            for object_key in object_keys:
                try:
                    if await self.storage_client.object_exists(object_key):
                        return True
                except Exception as e:
                    self.logger.warning(f"Error checking {object_key}: {e}")
                    continue

            return False

        except Exception as e:
            self.logger.error(f"Failed to check cloud existence: {e}")
            return False

    def _generate_cloud_object_keys(
        self, symbol: str, timeframe: TimeFrame
    ) -> list[str]:
        """
        Generate possible cloud object keys for the given symbol and timeframe.

        Following the pattern from backtesting service:
        datasets/{exchange}/{symbol}/{timeframe}/{format}/{date}/{filename}
        """
        # Try different date patterns (recent dates first)
        dates = []
        current_date = datetime.now()
        for i in range(30):  # Check last 30 days
            date_str = (current_date - timedelta(days=i)).strftime("%Y%m%d")
            dates.append(date_str)

        # Try different exchange and format combinations
        exchanges = ["binance", "coinbase", "kraken"]  # Common exchanges
        formats = ["csv", "parquet"]

        object_keys = []

        for exchange in exchanges:
            for format_type in formats:
                for date_str in dates:
                    # Try different filename patterns
                    filenames = [
                        f"{symbol}_{timeframe.value}.{format_type}",
                        f"{symbol.upper()}_{timeframe.value}.{format_type}",
                        f"{symbol.lower()}_{timeframe.value}.{format_type}",
                        f"{exchange}_{symbol}_{timeframe.value}_{date_str}.{format_type}",
                        f"data.{format_type}",  # Generic filename
                    ]

                    for filename in filenames:
                        object_key = (
                            self.settings.build_dataset_path(
                                exchange=exchange,
                                symbol=symbol,
                                timeframe=timeframe.value,
                                format_type=format_type,
                                date=date_str,
                            )
                            + f"/{filename}"
                        )
                        object_keys.append(object_key)

        return object_keys

    # ===== Cache Methods =====

    async def _load_from_cache(
        self, symbol: str, timeframe: TimeFrame
    ) -> Optional[pd.DataFrame]:
        """Load data from local cache if valid."""
        cache_file = self._get_cache_file_path(symbol, timeframe)

        if not cache_file.exists():
            return None

        # Check if cache is still valid
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age.total_seconds() > (self.cache_ttl_hours * 3600):
            self.logger.debug(f"Cache expired for {symbol}_{timeframe.value}")
            # Remove expired cache
            try:
                cache_file.unlink()
            except Exception:
                pass
            return None

        try:
            return await self._load_from_file(cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return None

    async def _cache_data(
        self, symbol: str, timeframe: TimeFrame, data: pd.DataFrame
    ) -> None:
        """Cache data to local storage."""
        try:
            cache_file = self._get_cache_file_path(symbol, timeframe)
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Save to cache
            data.to_csv(cache_file, index=False)
            self.logger.debug(f"Cached data for {symbol}_{timeframe.value}")

        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")

    async def _check_cache_exists(self, symbol: str, timeframe: TimeFrame) -> bool:
        """Check if valid cached data exists."""
        cache_file = self._get_cache_file_path(symbol, timeframe)

        if not cache_file.exists():
            return False

        # Check if cache is still valid
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age.total_seconds() <= (self.cache_ttl_hours * 3600)

    def _get_cache_file_path(self, symbol: str, timeframe: TimeFrame) -> Path:
        """Get cache file path for symbol and timeframe."""
        return self.cache_dir / f"{symbol}_{timeframe.value}_cached.csv"

    # ===== Local Storage Update Methods =====

    async def _update_local_storage(
        self, symbol: str, timeframe: TimeFrame, data: pd.DataFrame
    ) -> None:
        """Update local storage with cloud data for fallback scenarios."""
        try:
            # Create local file path
            local_file = self.data_dir / f"{symbol}_{timeframe.value}.csv"
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Save data to local storage
            data.to_csv(local_file, index=False)
            self.logger.info(f"Updated local storage: {local_file}")

        except Exception as e:
            self.logger.warning(f"Failed to update local storage: {e}")

    # ===== Local File Methods =====

    async def _load_from_local(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
        """Load data from local files."""
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

        return await self._load_from_file(data_path)

    async def _check_local_exists(self, symbol: str, timeframe: TimeFrame) -> bool:
        """Check if local data files exist."""
        possible_names = [
            f"{symbol}_{timeframe.value}.csv",
            f"{symbol}_{timeframe}.csv",
            f"{symbol.upper()}_{timeframe.value}.csv",
            f"{symbol.lower()}_{timeframe.value}.csv",
        ]

        for name in possible_names:
            data_path = self.data_dir / name
            if data_path.exists():
                return True

        return False

    async def _load_from_file(self, data_path: Path) -> pd.DataFrame:
        """Load data from a specific file path."""
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.logger.debug(f"Loading data from {data_path}")

        try:
            # Determine file format and load accordingly
            if data_path.suffix.lower() == ".csv":
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in [".parquet", ".pq"]:
                df = pd.read_parquet(data_path)
            else:
                # Default to CSV
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
                    self.logger.debug(f"Mapped columns: {mapped_columns}")

                # Check again for missing columns
                still_missing = [
                    col for col in required_columns if col not in df.columns
                ]
                if still_missing:
                    raise ValueError(f"Missing required columns: {still_missing}")

            self.logger.debug(f"Loaded {len(df)} rows of data")
            return df

        except Exception as e:
            self.logger.error(f"Failed to load data from {data_path}: {e}")
            raise


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
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
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


# ===== Data Loader Factory =====


class DataLoaderFactory:
    """Factory for creating data loaders based on configuration."""

    @staticmethod
    def create_data_loader(data_dir: Optional[Path] = None) -> IDataLoader:
        """
        Create a data loader based on configuration.

        Args:
            data_dir: Optional data directory override

        Returns:
            Configured data loader instance
        """
        settings = get_settings()
        loader_type = settings.data_loader_type.lower()

        if loader_type == "cloud":
            return CloudDataLoader(data_dir)
        elif loader_type == "local":
            return FileDataLoader(data_dir)
        elif loader_type == "hybrid":
            # Hybrid mode uses CloudDataLoader which has built-in fallback
            return CloudDataLoader(data_dir)
        else:
            # Default to hybrid mode
            return CloudDataLoader(data_dir)

    @staticmethod
    def create_cloud_loader(data_dir: Optional[Path] = None) -> CloudDataLoader:
        """Create a cloud data loader."""
        return CloudDataLoader(data_dir)

    @staticmethod
    def create_file_loader(data_dir: Optional[Path] = None) -> FileDataLoader:
        """Create a file data loader."""
        return FileDataLoader(data_dir)
