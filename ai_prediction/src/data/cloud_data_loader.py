# data/data_loader.py

import tempfile
import zipfile
import tarfile
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any
import pandas as pd
from pathlib import Path
from ..interfaces.data_loader_interface import IDataLoader
from ..schemas.enums import TimeFrame
from common.logger.logger_factory import LoggerFactory, LoggerType
from ..core.config import get_settings
from ..utils.storage_client import StorageClient
from .file_data_loader import FileDataLoader


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
        """Load data from cloud storage using smart discovery and archive extraction."""
        if not self.cloud_enabled or not self.storage_client:
            return None

        try:
            self.logger.info(f"Searching for cloud data: {symbol}_{timeframe.value}")

            # Find available datasets using smart listing
            datasets = await self._discover_cloud_datasets(symbol, timeframe)

            if not datasets:
                self.logger.debug(
                    f"No cloud datasets found for {symbol}_{timeframe.value}"
                )
                return None

            # Sort datasets by preference (parquet first, then most recent)
            sorted_datasets = self._sort_datasets_by_preference(datasets)

            # Try to load from the best available dataset
            for dataset in sorted_datasets:
                try:
                    self.logger.info(
                        f"Attempting to load dataset: {dataset['object_key']}"
                    )
                    data = await self._download_and_extract_dataset(dataset)

                    if data is not None:
                        self.logger.info(
                            f"Successfully loaded {len(data)} records from cloud"
                        )
                        return data

                except Exception as e:
                    self.logger.warning(
                        f"Failed to load dataset {dataset['object_key']}: {e}"
                    )
                    continue

            self.logger.warning(
                f"Failed to load any cloud dataset for {symbol}_{timeframe.value}"
            )
            return None

        except Exception as e:
            self.logger.error(f"Failed to load from cloud: {e}")
            return None

    async def _check_cloud_exists(self, symbol: str, timeframe: TimeFrame) -> bool:
        """Check if data exists in cloud storage using smart discovery."""
        if not self.cloud_enabled or not self.storage_client:
            return False

        try:
            datasets = await self._discover_cloud_datasets(symbol, timeframe)
            return len(datasets) > 0

        except Exception as e:
            self.logger.error(f"Failed to check cloud existence: {e}")
            return False

    async def _discover_cloud_datasets(
        self, symbol: str, timeframe: TimeFrame
    ) -> List[Dict[str, Any]]:
        """
        Discover available datasets in cloud storage using prefix-based listing.

        Returns a list of dataset metadata dictionaries with object_key, format, date, etc.
        """
        try:
            datasets = []

            # Search in common exchanges - prioritize binance as default
            exchanges = ["binance"]  # Start with binance as it's most common

            for exchange in exchanges:
                try:
                    # Build search prefix following the backtesting pattern
                    # {storage_prefix}/{exchange}/{symbol}/{timeframe}/
                    prefix = self.settings.build_dataset_path(
                        exchange=exchange, symbol=symbol, timeframe=timeframe.value
                    )

                    self.logger.debug(f"Searching with prefix: {prefix}")

                    # List objects with this prefix
                    objects = await self.storage_client.list_objects(
                        prefix=prefix, max_keys=100  # Limit to avoid too many results
                    )

                    for obj in objects:
                        dataset_info = self._parse_cloud_object_info(obj)
                        if dataset_info:
                            datasets.append(dataset_info)

                except Exception as e:
                    self.logger.warning(f"Error searching in exchange {exchange}: {e}")
                    continue

            self.logger.info(
                f"Found {len(datasets)} datasets for {symbol}_{timeframe.value}"
            )
            return datasets

        except Exception as e:
            self.logger.error(f"Failed to discover cloud datasets: {e}")
            return []

    def _parse_cloud_object_info(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse cloud object information to extract dataset metadata.

        Expected key layout (after the configured storage prefix):
        {exchange}/{symbol}/{timeframe}/{format}/{date}/{filename}

        This function handles multi-level storage prefixes, e.g.:
        storage_prefix = "finsight/market_data/datasets"
        object_key     = "finsight/market_data/datasets/binance/BTCUSDT/1h/parquet/20250809/file.zip"

        It also supports a legacy single-level prefix "datasets/...".
        """
        try:
            object_key = obj.get("key", "")
            if not object_key:
                return None

            # --- Normalize and split both the key and the configured prefix ---
            # Strip leading/trailing slashes to avoid empty parts on split.
            key_parts = object_key.strip("/").split("/")

            # The configured prefix may contain multiple folders (subfolders).
            prefix_str = self.settings.get_storage_prefix()
            prefix_parts = prefix_str.strip("/").split("/") if prefix_str else []

            # --- Determine the offset after the prefix ---
            # Case 1: Key starts with the multi-level configured prefix.
            if prefix_parts and key_parts[: len(prefix_parts)] == prefix_parts:
                offset = len(prefix_parts)
            # Case 2: Legacy layout starting with a single-level "datasets".
            elif key_parts and key_parts[0] == "datasets":
                offset = 1
            else:
                # The object key is not under our expected prefix; ignore it.
                return None

            # --- Ensure we have enough parts after the prefix ---
            # We expect: exchange, symbol, timeframe, format, date, filename  => 6 parts
            if len(key_parts) < offset + 6:
                return None

            exchange = key_parts[offset + 0]
            symbol = key_parts[offset + 1]
            timeframe_str = key_parts[offset + 2]
            format_type = key_parts[offset + 3]
            date_str = key_parts[offset + 4]
            filename = key_parts[-1]  # allow for additional nesting before filename

            # --- Validate timeframe using the TimeFrame enum values ---
            try:
                valid_timeframes = {tf.value for tf in TimeFrame}
                if timeframe_str not in valid_timeframes:
                    # Not a valid timeframe; skip this object.
                    self.logger.warning(
                        f"Invalid timeframe '{timeframe_str}' in object key: {object_key}"
                    )
                    return None
            except Exception as e:
                self.logger.warning(
                    f"Failed to validate timeframe '{timeframe_str}': {e}"
                )
                return None

            return {
                "object_key": object_key,
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe_str,
                "format": format_type,
                "date": date_str,
                "filename": filename,
                "size": obj.get("size", 0),
                "last_modified": obj.get("last_modified", ""),
                "is_archive": self._is_archive_file(filename),
            }

        except Exception as e:
            # Keep the parser resilient: on any unexpected shape, skip and continue.
            self.logger.warning(f"Error parsing object info: {e}")
            return None

    def _sort_datasets_by_preference(
        self, datasets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort datasets by preference: parquet first, then most recent."""

        def sort_key(dataset):
            # Prefer parquet over csv
            format_priority = 0 if dataset.get("format") == "parquet" else 1
            # Prefer more recent dates (negative for reverse sort)
            date_priority = -(
                int(dataset.get("date", "0"))
                if dataset.get("date", "").isdigit()
                else 0
            )
            return (format_priority, date_priority)

        return sorted(datasets, key=sort_key)

    async def _download_and_extract_dataset(
        self, dataset: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Download and extract a dataset from cloud storage."""
        try:
            object_key = dataset["object_key"]
            is_archive = dataset.get("is_archive", False)

            # Create temporary directory for downloads
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Download the file
                download_file = temp_path / dataset["filename"]
                success = await self.storage_client.download_file(
                    object_key=object_key, local_file_path=download_file
                )

                if not success:
                    self.logger.warning(f"Failed to download {object_key}")
                    return None

                # Extract if it's an archive
                if is_archive:
                    extract_dir = temp_path / "extracted"
                    extract_dir.mkdir(exist_ok=True)

                    extracted_files = await self._extract_archive(
                        download_file, extract_dir
                    )
                    if not extracted_files:
                        self.logger.warning(
                            f"Failed to extract archive {download_file}"
                        )
                        return None

                    # Find the main data file
                    data_file = self._find_main_data_file(
                        extracted_files, dataset["format"]
                    )
                    if not data_file:
                        self.logger.warning(f"No data file found in extracted archive")
                        return None
                else:
                    data_file = download_file

                # Load the data
                data = await self._load_from_file(data_file)

                if data is not None:
                    # Cache the data locally for future use
                    await self._cache_data_from_cloud(dataset, data)

                return data

        except Exception as e:
            self.logger.error(f"Error downloading and extracting dataset: {e}")
            return None

    def _is_archive_file(self, filename: str) -> bool:
        """Check if filename indicates an archive file."""
        archive_extensions = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2"}
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in archive_extensions)

    async def _extract_archive(
        self, archive_path: Path, extract_dir: Path
    ) -> List[Path]:
        """Extract archive and return list of extracted files."""
        extracted_files = []

        try:
            if archive_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zipf:
                    zipf.extractall(extract_dir)
                    extracted_files = [
                        extract_dir / name
                        for name in zipf.namelist()
                        if not name.endswith("/")
                    ]

            elif archive_path.suffix.lower() in {
                ".tar",
                ".tar.gz",
                ".tgz",
                ".tar.bz2",
            } or str(archive_path).endswith(".tar.gz"):
                with tarfile.open(archive_path, "r:*") as tarf:
                    tarf.extractall(extract_dir)
                    extracted_files = [
                        extract_dir / name
                        for name in tarf.getnames()
                        if not name.endswith("/")
                    ]

            self.logger.debug(
                f"Extracted {len(extracted_files)} files from {archive_path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to extract archive {archive_path}: {e}")

        return [f for f in extracted_files if f.is_file()]

    def _find_main_data_file(
        self, extracted_files: List[Path], format_type: str
    ) -> Optional[Path]:
        """Find the main data file from extracted files."""
        # Look for files with the expected format
        candidates = [
            f
            for f in extracted_files
            if f.suffix.lower() in [f".{format_type}", ".csv", ".parquet"]
        ]

        if not candidates:
            return None

        # Prefer files that look like main data files
        for candidate in candidates:
            filename_lower = candidate.name.lower()
            if any(
                keyword in filename_lower for keyword in ["data", "ohlcv", "candles"]
            ):
                return candidate

        # Fall back to the largest file (likely the data file)
        return max(candidates, key=lambda f: f.stat().st_size if f.exists() else 0)

    async def _cache_data_from_cloud(
        self, dataset: Dict[str, Any], data: pd.DataFrame
    ) -> None:
        """Cache cloud data locally and update local storage."""
        try:
            symbol = dataset["symbol"]
            timeframe_str = dataset["timeframe"]

            # Validate timeframe string before conversion
            try:
                timeframe = TimeFrame(timeframe_str)
            except ValueError as e:
                self.logger.error(
                    f"Invalid timeframe '{timeframe_str}' for caching: {e}"
                )
                return

            # Cache in the standard cache location
            await self._cache_data(symbol, timeframe, data)

            # Also update local storage for fallback
            await self._update_local_storage(symbol, timeframe, data)

            self.logger.info(
                f"Cached {len(data)} records for {symbol}_{timeframe.value}"
            )

        except Exception as e:
            self.logger.error(f"Failed to cache cloud data: {e}")

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


# ===== Factory functions moved to dependencies.py =====
# DataLoaderFactory has been moved to the dependency injection container
# in src/utils/dependencies.py for centralized configuration management.
