# services/dataset_management_service.py

"""
Dataset Management Service for AI Prediction Module

Handles comprehensive dataset management operations including:
- Dataset discovery and listing from cloud storage
- Dataset availability checking across all sources
- Dataset download and intelligent caching
- Cache management and invalidation
- Dataset statistics and health monitoring
"""


from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import uuid
from pathlib import Path

from ..data.cloud_data_loader import CloudDataLoader
from ..data.file_data_loader import FileDataLoader
from ..utils.storage_client import StorageClient
from ..core.config import get_settings
from ..schemas.dataset_schemas import (
    DatasetInfo,
    DatasetAvailabilityResponse,
    DatasetDownloadResponse,
    CacheInfo,
    DatasetStatistics,
    DatasetHealthCheck,
    BulkOperationResponse,
)
from ..schemas.enums import TimeFrame
from ..utils.dataset_utils import (
    validate_timeframe_string,
    parse_datetime_string,
    calculate_file_age_hours,
    calculate_cache_expiry_hours,
    merge_dataset_lists,
    calculate_dataset_statistics,
)
from common.logger.logger_factory import LoggerFactory, LoggerType


class DatasetManagementService:
    """
    Service for managing datasets across cloud storage, local cache, and local files.

    Provides unified interface for dataset operations with intelligent fallback
    and caching strategies.
    """

    def __init__(self):
        """Initialize the dataset management service."""
        self.settings = get_settings()

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name="dataset-management-service",
            logger_type=LoggerType.STANDARD,
            log_file="logs/dataset_management.log",
        )

        # Initialize components
        self.cloud_loader = CloudDataLoader()
        self.file_loader = FileDataLoader()
        self.storage_client = StorageClient(**self.settings.get_storage_config())

        # Cache directory
        self.cache_dir = self.settings.cloud_data_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self._download_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0

        self.logger.info("Dataset Management Service initialized")

    async def list_datasets(
        self,
        exchange_filter: Optional[str] = None,
        symbol_filter: Optional[str] = None,
        timeframe_filter: Optional[str] = None,
        format_filter: Optional[str] = None,
        prefix: Optional[str] = None,
        include_cached: bool = True,
        include_cloud: bool = True,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "last_modified",
        sort_order: str = "desc",
    ) -> List[DatasetInfo]:
        """
        List available datasets with filtering and pagination.

        Args:
            exchange_filter: Filter by exchange
            symbol_filter: Filter by symbol
            timeframe_filter: Filter by timeframe
            format_filter: Filter by format
            prefix: Object key prefix
            include_cached: Include cached datasets
            include_cloud: Include cloud datasets
            limit: Maximum results
            offset: Result offset
            sort_by: Sort field
            sort_order: Sort order

        Returns:
            List of dataset information
        """
        try:
            self.logger.info(
                f"Listing datasets with filters: {exchange_filter}, {symbol_filter}, {timeframe_filter}"
            )

            # Get cloud datasets
            cloud_datasets_raw = []
            if include_cloud and self.cloud_loader.cloud_enabled:
                cloud_datasets_raw = await self._discover_cloud_datasets(
                    exchange_filter,
                    symbol_filter,
                    timeframe_filter,
                    format_filter,
                    prefix,
                )

            # Get cached datasets
            cached_datasets = []
            if include_cached:
                cached_datasets = await self._list_cached_datasets(
                    symbol_filter, timeframe_filter
                )

            # Get local datasets
            local_datasets = []
            if symbol_filter and timeframe_filter:
                try:
                    timeframe = TimeFrame(timeframe_filter)
                    local_exists = await self.cloud_loader._check_local_exists(
                        symbol_filter, timeframe
                    )
                    if local_exists:
                        local_datasets = [
                            self._create_local_dataset_info(symbol_filter, timeframe)
                        ]
                except Exception as e:
                    self.logger.debug(f"Failed to check local datasets: {e}")

            # Use utility function to merge and deduplicate datasets
            all_datasets = merge_dataset_lists(
                cloud_datasets_raw, cached_datasets, local_datasets
            )

            # Apply format filter if specified
            if format_filter:
                all_datasets = [
                    ds for ds in all_datasets if ds.format_type == format_filter
                ]

            # Apply sorting
            all_datasets = self._sort_datasets(all_datasets, sort_by, sort_order)

            # Apply pagination
            start_idx = offset
            end_idx = start_idx + limit
            paginated_datasets = all_datasets[start_idx:end_idx]

            self.logger.info(
                f"Found {len(all_datasets)} total datasets, returning {len(paginated_datasets)}"
            )
            return paginated_datasets

        except Exception as e:
            self.logger.error(f"Failed to list datasets: {e}")
            return []

    async def check_dataset_availability(
        self,
        symbol: str,
        timeframe: TimeFrame,
        exchange: str = "binance",
        check_cloud: bool = True,
        check_cache: bool = True,
        check_local: bool = True,
    ) -> DatasetAvailabilityResponse:
        """
        Check dataset availability across all sources.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            exchange: Exchange name
            check_cloud: Check cloud storage
            check_cache: Check local cache
            check_local: Check local files

        Returns:
            Dataset availability information
        """
        try:
            self.logger.info(f"Checking availability for {symbol}_{timeframe.value}")

            available_sources = []
            cloud_available = False
            cache_available = False
            local_available = False
            dataset_info = None
            cache_age_hours = None
            cache_expires_in_hours = None

            # Check cloud storage
            if check_cloud and self.cloud_loader.cloud_enabled:
                cloud_exists = await self.cloud_loader._check_cloud_exists(
                    symbol, timeframe
                )
                if cloud_exists:
                    cloud_available = True
                    available_sources.append("cloud")

                    # Get cloud dataset info
                    cloud_datasets = await self.cloud_loader._discover_cloud_datasets(
                        symbol, timeframe
                    )
                    if cloud_datasets:
                        dataset_info = self._convert_to_dataset_info(cloud_datasets[0])

            # Check local cache
            if check_cache:
                cache_exists = await self.cloud_loader._check_cache_exists(
                    symbol, timeframe
                )
                if cache_exists:
                    cache_available = True
                    available_sources.append("cache")

                    # Get cache info
                    cache_info = await self._get_cache_info(symbol, timeframe)
                    if cache_info:
                        # Calculate cache age and expiry using utility functions
                        cache_age_hours = calculate_file_age_hours(
                            Path(cache_info.file_path)
                        )
                        cache_expires_in_hours = calculate_cache_expiry_hours(
                            Path(cache_info.file_path), cache_info.ttl_hours
                        )

                        if not dataset_info:
                            dataset_info = self._convert_cache_to_dataset_info(
                                cache_info
                            )

            # Check local files
            if check_local:
                local_exists = await self.cloud_loader._check_local_exists(
                    symbol, timeframe
                )
                if local_exists:
                    local_available = True
                    available_sources.append("local")

                    if not dataset_info:
                        dataset_info = self._create_local_dataset_info(
                            symbol, timeframe
                        )

            # Determine overall availability
            exists = len(available_sources) > 0

            # Generate recommendation
            recommended_action = self._generate_availability_recommendation(
                cloud_available, cache_available, local_available, cache_age_hours
            )

            response = DatasetAvailabilityResponse(
                success=True,
                message=f"Dataset availability checked for {symbol}_{timeframe.value}",
                exists=exists,
                available_sources=available_sources,
                cloud_available=cloud_available,
                cache_available=cache_available,
                local_available=local_available,
                dataset_info=dataset_info,
                cache_age_hours=cache_age_hours,
                cache_expires_in_hours=cache_expires_in_hours,
                recommended_action=recommended_action,
            )

            self.logger.info(
                f"Availability check completed: {exists}, sources: {available_sources}"
            )
            return response

        except Exception as e:
            self.logger.error(f"Failed to check dataset availability: {e}")
            return DatasetAvailabilityResponse(
                success=False,
                message=f"Failed to check availability: {str(e)}",
                exists=False,
                available_sources=[],
                cloud_available=False,
                cache_available=False,
                local_available=False,
            )

    async def download_dataset(
        self,
        symbol: str,
        timeframe: TimeFrame,
        exchange: str = "binance",
        target_format: Optional[str] = None,
        force_download: bool = False,
        update_cache: bool = True,
        update_local: bool = False,
        object_key: Optional[str] = None,
        cache_ttl_hours: Optional[int] = None,
    ) -> DatasetDownloadResponse:
        """
        Download dataset with intelligent caching and fallback.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            exchange: Exchange name
            target_format: Target format
            force_download: Force download even if cached
            update_cache: Update local cache
            update_local: Update local storage
            object_key: Specific object key
            cache_ttl_hours: Custom cache TTL

        Returns:
            Download response with results
        """
        try:
            download_id = str(uuid.uuid4())
            start_time = datetime.now()

            self.logger.info(
                f"Starting dataset download: {download_id} for {symbol}_{timeframe.value}"
            )

            # Check if we can use cached data
            if not force_download and update_cache:
                cache_exists = await self.cloud_loader._check_cache_exists(
                    symbol, timeframe
                )
                if cache_exists:
                    self._cache_hits += 1
                    self.logger.info(
                        f"Using cached data for {symbol}_{timeframe.value}"
                    )

                    return DatasetDownloadResponse(
                        success=True,
                        message=f"Dataset retrieved from cache: {symbol}_{timeframe.value}",
                        download_id=download_id,
                        status="cached",
                        cached_path=str(
                            self.cloud_loader._get_cache_file_path(symbol, timeframe)
                        ),
                    )

            # Download from cloud
            if self.cloud_loader.cloud_enabled:
                try:
                    # Load data using cloud loader (includes caching)
                    data = await self.cloud_loader.load_data(symbol, timeframe)

                    if data is not None and not data.empty:
                        self._cache_misses += 1

                        # Calculate performance metrics
                        end_time = datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        self._download_times.append(duration)

                        # Get cache info
                        cache_path = self.cloud_loader._get_cache_file_path(
                            symbol, timeframe
                        )
                        cache_ttl = cache_ttl_hours or self.cloud_loader.cache_ttl_hours

                        # Create dataset info
                        dataset_info = DatasetInfo(
                            exchange=exchange,
                            symbol=symbol,
                            timeframe=timeframe.value,
                            format_type=target_format or "csv",
                            is_cached=True,
                            cache_age_hours=0.0,
                            record_count=len(data),
                        )

                        response = DatasetDownloadResponse(
                            success=True,
                            message=f"Dataset downloaded successfully: {symbol}_{timeframe.value}",
                            download_id=download_id,
                            dataset_info=dataset_info,
                            cached_path=str(cache_path),
                            cache_ttl_hours=cache_ttl,
                            download_duration_seconds=duration,
                            status="downloaded",
                        )

                        self.logger.info(
                            f"Download completed: {symbol}_{timeframe.value} in {duration:.2f}s"
                        )
                        return response

                except Exception as e:
                    self.logger.warning(f"Cloud download failed: {e}")

            # Fallback to local data
            try:
                data = await self.cloud_loader._load_from_local(symbol, timeframe)

                if data is not None and not data.empty:
                    response = DatasetDownloadResponse(
                        success=True,
                        message=f"Dataset loaded from local storage: {symbol}_{timeframe.value}",
                        download_id=download_id,
                        status="local",
                    )

                    self.logger.info(
                        f"Local fallback successful: {symbol}_{timeframe.value}"
                    )
                    return response

            except Exception as e:
                self.logger.error(f"Local fallback failed: {e}")

            # All attempts failed
            return DatasetDownloadResponse(
                success=False,
                message=f"Failed to download dataset: {symbol}_{timeframe.value}",
                download_id=download_id,
                status="failed",
                error_message="All download methods failed",
            )

        except Exception as e:
            self.logger.error(f"Download operation failed: {e}")
            return DatasetDownloadResponse(
                success=False,
                message=f"Download operation failed: {str(e)}",
                download_id=download_id,
                status="error",
                error_message=str(e),
            )

    async def list_cached_datasets(
        self,
        symbol_filter: Optional[str] = None,
        timeframe_filter: Optional[str] = None,
        expired_only: bool = False,
        valid_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CacheInfo]:
        """
        List cached datasets with filtering.

        Args:
            symbol_filter: Filter by symbol
            timeframe_filter: Filter by timeframe
            expired_only: Show only expired caches
            valid_only: Show only valid caches
            limit: Maximum results
            offset: Result offset

        Returns:
            List of cache information
        """
        try:
            self.logger.info("Listing cached datasets")

            cache_files = list(self.cache_dir.glob("*_cached.csv"))
            cache_infos: List[CacheInfo] = []

            for cache_file in cache_files:
                try:
                    # Parse cache filename: {symbol}_{timeframe}_cached.csv
                    parts = cache_file.stem.split("_")
                    if len(parts) >= 2:
                        symbol = parts[0]
                        timeframe = "_".join(
                            parts[1:-1]
                        )  # Handle timeframes like "1h", "4h", "1d"

                        # Apply filters
                        if symbol_filter and symbol != symbol_filter:
                            continue
                        if timeframe_filter and timeframe != timeframe_filter:
                            continue

                        # Get cache info
                        cache_info = await self._get_cache_info(
                            symbol, TimeFrame(timeframe)
                        )
                        if cache_info:
                            # Apply expired/valid filters
                            if expired_only and not cache_info.is_expired:
                                continue
                            if valid_only and cache_info.is_expired:
                                continue

                            cache_infos.append(cache_info)

                except Exception as e:
                    self.logger.warning(f"Failed to parse cache file {cache_file}: {e}")
                    continue

            # Apply pagination
            start_idx = offset
            end_idx = start_idx + limit
            return cache_infos[start_idx:end_idx]

        except Exception as e:
            self.logger.error(f"Failed to list cached datasets: {e}")
            return []

    async def invalidate_cache(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[TimeFrame] = None,
        expired_only: bool = False,
        all_caches: bool = False,
        confirm_invalidation: bool = False,
    ) -> Dict[str, Any]:
        """
        Invalidate caches based on criteria.

        Args:
            symbol: Specific symbol to invalidate
            timeframe: Specific timeframe to invalidate
            expired_only: Invalidate only expired caches
            all_caches: Invalidate all caches
            confirm_invalidation: Confirmation flag

        Returns:
            Invalidation results
        """
        try:
            if all_caches and not confirm_invalidation:
                return {
                    "success": False,
                    "message": "Bulk cache invalidation requires confirmation",
                    "invalidated_count": 0,
                    "freed_space_bytes": 0,
                }

            self.logger.info(
                f"Invalidating caches: symbol={symbol}, timeframe={timeframe}, expired_only={expired_only}"
            )

            cache_files = list(self.cache_dir.glob("*_cached.csv"))
            invalidated_count = 0
            freed_space_bytes = 0
            invalidated_caches = []
            failed_invalidations = []

            for cache_file in cache_files:
                try:
                    # Parse filename
                    parts = cache_file.stem.split("_")
                    if len(parts) >= 2:
                        file_symbol = parts[0]
                        file_timeframe = "_".join(parts[1:-1])

                        # Check if this cache should be invalidated
                        should_invalidate = False

                        if all_caches:
                            should_invalidate = True
                        elif symbol and file_symbol == symbol:
                            if timeframe is None or file_timeframe == timeframe.value:
                                should_invalidate = True
                        elif expired_only:
                            # Check if cache is expired
                            cache_info = await self._get_cache_info(
                                file_symbol, TimeFrame(file_timeframe)
                            )
                            if cache_info and cache_info.is_expired:
                                should_invalidate = True

                        if should_invalidate:
                            # Get file size before deletion
                            file_size = cache_file.stat().st_size

                            # Delete cache file
                            cache_file.unlink()

                            invalidated_count += 1
                            freed_space_bytes += file_size
                            invalidated_caches.append(f"{file_symbol}_{file_timeframe}")

                            self.logger.debug(
                                f"Invalidated cache: {file_symbol}_{file_timeframe}"
                            )

                except Exception as e:
                    failed_invalidations.append(str(cache_file))
                    self.logger.warning(f"Failed to invalidate cache {cache_file}: {e}")

            result = {
                "success": True,
                "message": f"Cache invalidation completed: {invalidated_count} caches invalidated",
                "invalidated_count": invalidated_count,
                "freed_space_bytes": freed_space_bytes,
                "invalidated_caches": invalidated_caches,
                "failed_invalidations": failed_invalidations,
                "status": "completed",
            }

            self.logger.info(
                f"Cache invalidation completed: {invalidated_count} caches, {freed_space_bytes} bytes freed"
            )
            return result

        except Exception as e:
            self.logger.error(f"Cache invalidation failed: {e}")
            return {
                "success": False,
                "message": f"Cache invalidation failed: {str(e)}",
                "invalidated_count": 0,
                "freed_space_bytes": 0,
                "invalidated_caches": [],
                "failed_invalidations": [],
                "status": "failed",
            }

    async def get_dataset_statistics(self) -> DatasetStatistics:
        """Get comprehensive dataset statistics."""
        try:
            self.logger.info("Generating dataset statistics")

            # Get all datasets
            all_datasets = await self.list_datasets(
                include_cached=True,
                include_cloud=True,
                limit=10000,  # Get all datasets for statistics
            )

            # Use utility function to calculate statistics
            stats = calculate_dataset_statistics(all_datasets)

            return DatasetStatistics(
                success=True,
                message="Dataset statistics generated successfully",
                total_datasets=stats["total_datasets"],
                total_size_bytes=stats["total_size_bytes"],
                exchange_statistics=stats["exchange_statistics"],
                symbol_statistics=stats["symbol_statistics"],
                timeframe_statistics=stats["timeframe_statistics"],
                cache_statistics={
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                    "cache_hit_rate": (
                        self._cache_hits / (self._cache_hits + self._cache_misses)
                        if (self._cache_hits + self._cache_misses) > 0
                        else 0
                    ),
                },
                last_updated=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Failed to get dataset statistics: {e}")
            return DatasetStatistics(
                success=False,
                message=f"Failed to generate statistics: {str(e)}",
                total_datasets=0,
                total_size_bytes=0,
                exchange_statistics=[],
                symbol_statistics=[],
                timeframe_statistics=[],
                cache_statistics={},
                last_updated=datetime.now(),
            )

    async def get_health_check(self) -> DatasetHealthCheck:
        """Get dataset management service health status."""
        try:
            self.logger.info("Performing health check")

            # Check cloud storage connectivity
            cloud_healthy = False
            cloud_error = None
            try:
                if self.cloud_loader.cloud_enabled:
                    # Try to list a small number of objects to test connectivity
                    test_objects = await self.storage_client.list_objects(
                        prefix="", max_keys=1
                    )
                    cloud_healthy = True
                else:
                    cloud_healthy = True  # Not enabled is considered healthy
            except Exception as e:
                cloud_healthy = False
                cloud_error = str(e)

            # Check cache directory
            cache_healthy = True
            cache_error = None
            try:
                if not self.cache_dir.exists():
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                # Test write access
                test_file = self.cache_dir / ".health_check"
                test_file.write_text("health_check")
                test_file.unlink()
            except Exception as e:
                cache_healthy = False
                cache_error = str(e)

            # Check local data directory
            local_healthy = True
            local_error = None
            try:
                local_dir = self.settings.data_dir
                if not local_dir.exists():
                    local_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                local_healthy = False
                local_error = str(e)

            # Overall health
            overall_healthy = cloud_healthy and cache_healthy and local_healthy

            return DatasetHealthCheck(
                success=True,
                message="Health check completed",
                overall_status="healthy" if overall_healthy else "unhealthy",
                timestamp=datetime.now(),
                components={
                    "cloud_storage": {
                        "status": "healthy" if cloud_healthy else "unhealthy",
                        "error": cloud_error,
                    },
                    "cache_directory": {
                        "status": "healthy" if cache_healthy else "unhealthy",
                        "error": cache_error,
                    },
                    "local_storage": {
                        "status": "healthy" if local_healthy else "unhealthy",
                        "error": local_error,
                    },
                },
                recommendations=[
                    (
                        "Ensure cloud storage credentials are valid"
                        if not cloud_healthy
                        else None
                    ),
                    "Check cache directory permissions" if not cache_healthy else None,
                    (
                        "Verify local storage directory access"
                        if not local_healthy
                        else None
                    ),
                ],
            )

        except Exception as e:
            self.logger.error(f"Failed to perform health check: {e}")
            return DatasetHealthCheck(
                success=False,
                message=f"Health check failed: {str(e)}",
                overall_status="unknown",
                timestamp=datetime.now(),
                components={},
                recommendations=[
                    "Service health check failed - check logs for details"
                ],
            )

    async def bulk_dataset_operation(
        self, operation: str, datasets: List[Dict[str, Any]], **kwargs
    ) -> BulkOperationResponse:
        """Perform bulk operations on multiple datasets."""
        try:
            self.logger.info(
                f"Starting bulk operation: {operation} on {len(datasets)} datasets"
            )

            results = {
                "successful": [],
                "failed": [],
                "skipped": [],
                "total_processed": 0,
            }

            for dataset in datasets:
                try:
                    symbol = dataset.get("symbol")
                    timeframe_str = dataset.get("timeframe")

                    if not symbol or not timeframe_str:
                        results["skipped"].append(
                            {
                                "dataset": dataset,
                                "reason": "Missing symbol or timeframe",
                            }
                        )
                        continue

                    # Convert timeframe string to TimeFrame enum
                    try:
                        timeframe = TimeFrame(timeframe_str)
                    except ValueError:
                        results["failed"].append(
                            {
                                "dataset": dataset,
                                "reason": f"Invalid timeframe: {timeframe_str}",
                            }
                        )
                        continue

                    # Perform the specific operation
                    if operation == "download":
                        result = await self.download_dataset(
                            symbol=symbol,
                            timeframe=timeframe,
                            exchange=dataset.get("exchange", "binance"),
                            **kwargs,
                        )
                        if result.success:
                            results["successful"].append(
                                {"dataset": dataset, "result": result}
                            )
                        else:
                            results["failed"].append(
                                {"dataset": dataset, "reason": result.message}
                            )

                    elif operation == "cache_invalidate":
                        result = await self.invalidate_cache(
                            symbol=symbol, timeframe=timeframe
                        )
                        if result.success:
                            results["successful"].append(
                                {"dataset": dataset, "result": result}
                            )
                        else:
                            results["failed"].append(
                                {"dataset": dataset, "reason": result.message}
                            )

                    else:
                        results["failed"].append(
                            {
                                "dataset": dataset,
                                "reason": f"Unsupported operation: {operation}",
                            }
                        )

                    results["total_processed"] += 1

                except Exception as e:
                    self.logger.error(f"Failed to process dataset {dataset}: {e}")
                    results["failed"].append({"dataset": dataset, "reason": str(e)})
                    results["total_processed"] += 1

            # Generate summary
            success_count = len(results["successful"])
            failed_count = len(results["failed"])
            skipped_count = len(results["skipped"])

            overall_success = failed_count == 0

            return BulkOperationResponse(
                success=overall_success,
                message=f"Bulk operation completed: {success_count} successful, {failed_count} failed, {skipped_count} skipped",
                operation=operation,
                total_datasets=len(datasets),
                results=results,
                summary={
                    "successful": success_count,
                    "failed": failed_count,
                    "skipped": skipped_count,
                    "total_processed": results["total_processed"],
                    "success_rate": success_count / len(datasets) if datasets else 0,
                },
            )

        except Exception as e:
            self.logger.error(f"Bulk operation failed: {e}")
            return BulkOperationResponse(
                success=False,
                message=f"Bulk operation failed: {str(e)}",
                operation=operation,
                total_datasets=len(datasets),
                results={
                    "successful": [],
                    "failed": [],
                    "skipped": [],
                    "total_processed": 0,
                },
                summary={
                    "successful": 0,
                    "failed": len(datasets),
                    "skipped": 0,
                    "total_processed": 0,
                    "success_rate": 0,
                },
            )

    # ===== Private Helper Methods =====

    async def _discover_cloud_datasets(
        self,
        exchange_filter: Optional[str] = None,
        symbol_filter: Optional[str] = None,
        timeframe_filter: Optional[str] = None,
        format_filter: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Discover datasets in cloud storage."""
        try:
            if not self.cloud_loader.cloud_enabled:
                return []

            # Use the cloud loader's discovery method
            all_datasets = []

            # Get default symbols and timeframes from settings
            symbols = (
                [symbol_filter] if symbol_filter else self.settings.default_symbols
            )
            timeframes = (
                [timeframe_filter]
                if timeframe_filter
                else [tf.value for tf in self.settings.default_timeframes]
            )
            exchanges = [exchange_filter] if exchange_filter else ["binance"]

            for exchange in exchanges:
                for symbol in symbols:
                    for timeframe in timeframes:
                        try:
                            datasets = await self.cloud_loader._discover_cloud_datasets(
                                symbol, TimeFrame(timeframe)
                            )
                            all_datasets.extend(datasets)
                        except Exception as e:
                            self.logger.debug(
                                f"Failed to discover datasets for {exchange}/{symbol}/{timeframe}: {e}"
                            )
                            continue

            # Apply format filter if specified
            if format_filter:
                all_datasets = [
                    ds for ds in all_datasets if ds.get("format") == format_filter
                ]

            return all_datasets

        except Exception as e:
            self.logger.error(f"Failed to discover cloud datasets: {e}")
            return []

    async def _list_cached_datasets(
        self,
        symbol_filter: Optional[str] = None,
        timeframe_filter: Optional[str] = None,
    ) -> List[DatasetInfo]:
        """List cached datasets as DatasetInfo objects."""
        try:
            cached_datasets = await self.list_cached_datasets(
                symbol_filter, timeframe_filter
            )
            return [
                self._convert_cache_to_dataset_info(cache) for cache in cached_datasets
            ]
        except Exception as e:
            self.logger.error(f"Failed to list cached datasets: {e}")
            return []

    def _sort_datasets(
        self,
        datasets: List[DatasetInfo],
        sort_by: str,
        sort_order: str,
    ) -> List[DatasetInfo]:
        """Sort datasets by specified field and order."""
        try:
            reverse = sort_order.lower() == "desc"

            if sort_by == "last_modified":
                return sorted(
                    datasets,
                    key=lambda x: x.last_modified or datetime.min,
                    reverse=reverse,
                )
            elif sort_by == "size":
                return sorted(
                    datasets, key=lambda x: x.size_bytes or 0, reverse=reverse
                )
            elif sort_by == "symbol":
                return sorted(datasets, key=lambda x: x.symbol, reverse=reverse)
            elif sort_by == "timeframe":
                return sorted(datasets, key=lambda x: x.timeframe, reverse=reverse)
            else:
                # Default sorting by symbol
                return sorted(datasets, key=lambda x: x.symbol)

        except Exception as e:
            self.logger.warning(f"Failed to sort datasets: {e}")
            return datasets

    async def _get_cache_info(
        self, symbol: str, timeframe: TimeFrame
    ) -> Optional[CacheInfo]:
        """Get detailed cache information for a dataset."""
        try:
            cache_file = self.cloud_loader._get_cache_file_path(symbol, timeframe)

            if not cache_file.exists():
                return None

            stat = cache_file.stat()
            created_at = datetime.fromtimestamp(stat.st_ctime)
            last_accessed = datetime.fromtimestamp(stat.st_atime)

            # Calculate TTL and expiration using utility functions
            ttl_hours = self.cloud_loader.cache_ttl_hours
            expires_at = created_at + timedelta(hours=ttl_hours)
            is_expired = datetime.now() > expires_at

            # Calculate cache age using utility function
            cache_age_hours = calculate_file_age_hours(cache_file)

            return CacheInfo(
                cache_key=f"{symbol}_{timeframe.value}",
                symbol=symbol,
                timeframe=timeframe.value,
                file_path=str(cache_file),
                file_size_bytes=stat.st_size,
                created_at=created_at,
                last_accessed=last_accessed,
                access_count=0,  # Would need to track actual access counts
                ttl_hours=ttl_hours,
                expires_at=expires_at,
                is_expired=is_expired,
                source_object_key=None,  # Would need to track source information
                source_exchange=None,
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to get cache info for {symbol}_{timeframe.value}: {e}"
            )
            return None

    def _convert_to_dataset_info(self, cloud_dataset: Dict[str, Any]) -> DatasetInfo:
        """Convert cloud dataset dict to DatasetInfo (alias for _convert_cloud_to_dataset_info)."""
        return self._convert_cloud_to_dataset_info(cloud_dataset)

    def _convert_cache_to_dataset_info(self, cache_info: CacheInfo) -> DatasetInfo:
        """Convert CacheInfo to DatasetInfo object."""
        try:
            return DatasetInfo(
                exchange=cache_info.source_exchange or "unknown",
                symbol=cache_info.symbol,
                timeframe=cache_info.timeframe,
                object_key=cache_info.source_object_key,
                format_type=None,  # Not available from cache info
                date=None,  # Not available from cache info
                filename=(
                    cache_info.file_path.split("/")[-1]
                    if cache_info.file_path
                    else None
                ),
                size_bytes=cache_info.file_size_bytes,
                last_modified=cache_info.last_accessed,
                etag=None,  # Not available from cache info
                is_archived=False,  # Cache files are not archived
                is_cached=True,
                cache_age_hours=calculate_file_age_hours(Path(cache_info.file_path)),
                record_count=None,  # Not available from cache info
                date_range_start=None,  # Not available from cache info
                date_range_end=None,  # Not available from cache info
            )
        except Exception as e:
            self.logger.error(f"Failed to convert cache info to DatasetInfo: {e}")
            # Return a minimal valid DatasetInfo
            return DatasetInfo(
                exchange=cache_info.source_exchange or "unknown",
                symbol=cache_info.symbol,
                timeframe=cache_info.timeframe,
            )

    def _create_local_dataset_info(
        self, symbol: str, timeframe: TimeFrame
    ) -> DatasetInfo:
        """Create DatasetInfo for local dataset."""
        try:
            local_file = self.settings.data_dir / f"{symbol}_{timeframe.value}.parquet"
            if local_file.exists():
                stat = local_file.stat()
                return DatasetInfo(
                    exchange="local",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    object_key=None,
                    format_type="parquet",
                    date=None,
                    filename=local_file.name,
                    size_bytes=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime),
                    etag=None,
                    is_archived=False,
                    is_cached=False,
                    cache_age_hours=None,
                    record_count=None,
                    date_range_start=None,
                    date_range_end=None,
                )
            else:
                return DatasetInfo(
                    exchange="local",
                    symbol=symbol,
                    timeframe=timeframe.value,
                )
        except Exception as e:
            self.logger.error(f"Failed to create local dataset info: {e}")
            return DatasetInfo(
                exchange="local",
                symbol=symbol,
                timeframe=timeframe.value,
            )

    def _generate_availability_recommendation(
        self,
        cloud_available: bool,
        cache_available: bool,
        local_available: bool,
        cache_age_hours: Optional[float],
    ) -> str:
        """Generate recommendation based on availability status."""
        if local_available:
            return "Use local data - fastest access"
        elif cache_available and cache_age_hours is not None:
            if cache_age_hours < 1:
                return "Use cached data - very recent"
            elif cache_age_hours < 24:
                return "Use cached data - reasonably fresh"
            else:
                return "Refresh cache - data may be stale"
        elif cloud_available:
            return "Download from cloud - not cached locally"
        else:
            return "No data available - consider data collection"

    def _convert_cloud_to_dataset_info(
        self, cloud_dataset: Dict[str, Any]
    ) -> DatasetInfo:
        """Convert cloud dataset dictionary to DatasetInfo object."""
        try:
            # Parse last_modified using utility function
            last_modified = None
            if cloud_dataset.get("last_modified"):
                last_modified = parse_datetime_string(
                    str(cloud_dataset["last_modified"])
                )

            return DatasetInfo(
                exchange=cloud_dataset.get("exchange", "unknown"),
                symbol=cloud_dataset.get("symbol", "unknown"),
                timeframe=cloud_dataset.get("timeframe", "unknown"),
                object_key=cloud_dataset.get("object_key"),
                format_type=cloud_dataset.get("format"),
                date=cloud_dataset.get("date"),
                filename=cloud_dataset.get("filename"),
                size_bytes=cloud_dataset.get("size", 0),
                last_modified=last_modified,
                etag=None,  # Cloud datasets don't have ETags
                is_archived=cloud_dataset.get("is_archive", False),
                is_cached=False,  # Cloud datasets are not cached by default
                cache_age_hours=None,
                record_count=None,  # Not available from cloud metadata
                date_range_start=None,  # Not available from cloud metadata
                date_range_end=None,  # Not available from cloud metadata
            )
        except Exception as e:
            self.logger.error(f"Failed to convert cloud dataset to DatasetInfo: {e}")
            # Return a minimal valid DatasetInfo
            return DatasetInfo(
                exchange=cloud_dataset.get("exchange", "unknown"),
                symbol=cloud_dataset.get("symbol", "unknown"),
                timeframe=cloud_dataset.get("timeframe", "unknown"),
            )
