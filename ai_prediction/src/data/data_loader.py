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
from common.logger.logger_factory import LoggerFactory
from ..core.config import get_settings

# Try to import cloud storage dependencies
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None


class CloudDataLoader(IDataLoader):
    """
    Cloud-first data loader with local fallback.

    This loader attempts to load data from cloud storage first, then falls back
    to local files if cloud data is not available. It also implements intelligent
    caching to improve performance.
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
        self.cloud_enabled = settings.enable_cloud_storage and AWS_AVAILABLE

        self.logger = LoggerFactory.get_logger("CloudDataLoader")

        # Initialize cloud storage client if available
        self.s3_client = None
        if self.cloud_enabled:
            try:
                self.s3_client = boto3.client(
                    "s3",
                    region_name=settings.cloud_storage_region,
                    endpoint_url=settings.cloud_storage_endpoint,
                    aws_access_key_id=settings.cloud_storage_access_key,
                    aws_secret_access_key=settings.cloud_storage_secret_key,
                )
                self.bucket = settings.cloud_storage_bucket
                self.logger.info(f"Cloud storage initialized: {self.bucket}")
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
        2. Fresh cloud data (download and cache)
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
        """Load data from cloud storage."""
        if not self.cloud_enabled:
            return None

        try:
            # Generate cloud object key following backtesting service pattern
            # Format: datasets/{exchange}/{symbol}/{timeframe}/{format}/{date}/{filename}
            object_keys = self._generate_cloud_object_keys(symbol, timeframe)

            for object_key in object_keys:
                try:
                    # Download to temporary file
                    with tempfile.NamedTemporaryFile(
                        suffix=".csv", delete=False
                    ) as tmp_file:
                        self.s3_client.download_file(
                            self.bucket, object_key, tmp_file.name
                        )

                        # Load the data
                        data = await self._load_from_file(Path(tmp_file.name))

                        # Clean up temp file
                        os.unlink(tmp_file.name)

                        return data

                except ClientError as e:
                    error_code = e.response["Error"]["Code"]
                    if error_code != "NoSuchKey":
                        self.logger.warning(f"Error downloading {object_key}: {e}")
                    continue

            self.logger.debug(f"No cloud data found for {symbol}_{timeframe.value}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to load from cloud: {e}")
            return None

    async def _check_cloud_exists(self, symbol: str, timeframe: TimeFrame) -> bool:
        """Check if data exists in cloud storage."""
        if not self.cloud_enabled:
            return False

        try:
            object_keys = self._generate_cloud_object_keys(symbol, timeframe)

            for object_key in object_keys:
                try:
                    self.s3_client.head_object(Bucket=self.bucket, Key=object_key)
                    return True
                except ClientError as e:
                    error_code = e.response["Error"]["Code"]
                    if error_code != "404":
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
                    ]

                    for filename in filenames:
                        object_key = f"datasets/{exchange}/{symbol}/{timeframe.value}/{format_type}/{date_str}/{filename}"
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

        self.logger = LoggerFactory.get_logger("FileDataLoader")

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
