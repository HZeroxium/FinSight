# utils/dataset_utils.py

"""
Dataset Utility Functions for AI Prediction Module

Common utility functions for dataset management operations including:
- Data validation and conversion
- File operations and path management
- Statistics calculation
- Error handling utilities
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd

from ..schemas.dataset_schemas import DatasetInfo, CacheInfo
from ..schemas.enums import TimeFrame
from common.logger.logger_factory import LoggerFactory, LoggerType


# Initialize logger
logger = LoggerFactory.get_logger(
    name="dataset-utils",
    logger_type=LoggerType.STANDARD,
    log_file="logs/dataset_utils.log",
)


def validate_timeframe_string(timeframe_str: str) -> bool:
    """
    Validate if a timeframe string is a valid TimeFrame enum value.

    Args:
        timeframe_str: Timeframe string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        valid_timeframes = [tf.value for tf in TimeFrame]
        return timeframe_str in valid_timeframes
    except Exception:
        return False


def parse_datetime_string(datetime_str: str) -> Optional[datetime]:
    """
    Parse datetime string to datetime object with error handling.

    Args:
        datetime_str: Datetime string to parse

    Returns:
        Parsed datetime object or None if parsing fails
    """
    try:
        # Handle common datetime formats
        if datetime_str.endswith("Z"):
            datetime_str = datetime_str.replace("Z", "+00:00")

        # Try ISO format first
        try:
            return datetime.fromisoformat(datetime_str)
        except ValueError:
            pass

        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse datetime string: {datetime_str}")
        return None

    except Exception as e:
        logger.error(f"Failed to parse datetime string '{datetime_str}': {e}")
        return None


def calculate_file_age_hours(file_path: Path) -> Optional[float]:
    """
    Calculate file age in hours.

    Args:
        file_path: Path to the file

    Returns:
        Age in hours or None if calculation fails
    """
    try:
        if not file_path.exists():
            return None

        stat = file_path.stat()
        created_time = datetime.fromtimestamp(stat.st_ctime)
        age = datetime.now() - created_time
        return age.total_seconds() / 3600

    except Exception as e:
        logger.error(f"Failed to calculate file age for {file_path}: {e}")
        return None


def calculate_cache_expiry_hours(file_path: Path, ttl_hours: int) -> Optional[float]:
    """
    Calculate cache expiry time in hours from now.

    Args:
        file_path: Path to the cache file
        ttl_hours: Time-to-live in hours

    Returns:
        Hours until expiry or None if calculation fails
    """
    try:
        if not file_path.exists():
            return None

        stat = file_path.stat()
        created_time = datetime.fromtimestamp(stat.st_ctime)
        expiry_time = created_time + timedelta(hours=ttl_hours)
        hours_until_expiry = (expiry_time - datetime.now()).total_seconds() / 3600

        return max(0, hours_until_expiry)  # Return 0 if already expired

    except Exception as e:
        logger.error(f"Failed to calculate cache expiry for {file_path}: {e}")
        return None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    try:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    except Exception:
        return "Unknown size"


def validate_dataset_path(path: Path) -> bool:
    """
    Validate if a dataset path exists and is accessible.

    Args:
        path: Path to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception as e:
        logger.error(f"Failed to validate dataset path {path}: {e}")
        return False


def get_dataset_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from a dataset file.

    Args:
        file_path: Path to the dataset file

    Returns:
        Dictionary containing metadata
    """
    try:
        if not file_path.exists():
            return {}

        stat = file_path.stat()
        metadata = {
            "file_size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
            "accessed_at": datetime.fromtimestamp(stat.st_atime),
            "file_path": str(file_path),
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
        }

        # Try to get additional metadata based on file type
        if file_path.suffix.lower() == ".parquet":
            try:
                # Read parquet metadata without loading full data
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(file_path)
                metadata.update(
                    {
                        "record_count": parquet_file.metadata.num_rows,
                        "columns": list(parquet_file.schema.names),
                        "compression": parquet_file.metadata.row_group(0)
                        .column(0)
                        .compression,
                    }
                )
            except Exception as e:
                logger.debug(f"Could not read parquet metadata: {e}")

        elif file_path.suffix.lower() == ".csv":
            try:
                # Read CSV header to get column count
                with open(file_path, "r") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        metadata["columns"] = first_line.split(",")
                        metadata["column_count"] = len(metadata["columns"])
            except Exception as e:
                logger.debug(f"Could not read CSV metadata: {e}")

        return metadata

    except Exception as e:
        logger.error(f"Failed to get dataset metadata for {file_path}: {e}")
        return {}


def merge_dataset_lists(
    cloud_datasets: List[Dict[str, Any]],
    cached_datasets: List[DatasetInfo],
    local_datasets: List[DatasetInfo],
) -> List[DatasetInfo]:
    """
    Merge multiple dataset lists and remove duplicates.

    Args:
        cloud_datasets: List of cloud dataset dictionaries
        cached_datasets: List of cached DatasetInfo objects
        local_datasets: List of local DatasetInfo objects

    Returns:
        Merged list of unique DatasetInfo objects
    """
    try:
        # Convert cloud datasets to DatasetInfo objects
        cloud_dataset_infos = []
        for cloud_ds in cloud_datasets:
            try:
                dataset_info = DatasetInfo(
                    exchange=cloud_ds.get("exchange", "unknown"),
                    symbol=cloud_ds.get("symbol", "unknown"),
                    timeframe=cloud_ds.get("timeframe", "unknown"),
                    object_key=cloud_ds.get("object_key"),
                    format_type=cloud_ds.get("format"),
                    date=cloud_ds.get("date"),
                    filename=cloud_ds.get("filename"),
                    size_bytes=cloud_ds.get("size", 0),
                    last_modified=parse_datetime_string(
                        str(cloud_ds.get("last_modified", ""))
                    ),
                    etag=None,
                    is_archived=cloud_ds.get("is_archive", False),
                    is_cached=False,
                    cache_age_hours=None,
                    record_count=None,
                    date_range_start=None,
                    date_range_end=None,
                )
                cloud_dataset_infos.append(dataset_info)
            except Exception as e:
                logger.warning(f"Failed to convert cloud dataset: {e}")
                continue

        # Combine all datasets
        all_datasets = cloud_dataset_infos + cached_datasets + local_datasets

        # Remove duplicates based on exchange, symbol, and timeframe
        seen = set()
        unique_datasets = []

        for dataset in all_datasets:
            key = (dataset.exchange, dataset.symbol, dataset.timeframe)
            if key not in seen:
                seen.add(key)
                unique_datasets.append(dataset)

        logger.info(
            f"Merged {len(all_datasets)} datasets into {len(unique_datasets)} unique datasets"
        )
        return unique_datasets

    except Exception as e:
        logger.error(f"Failed to merge dataset lists: {e}")
        return []


def calculate_dataset_statistics(datasets: List[DatasetInfo]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics from a list of datasets.

    Args:
        datasets: List of DatasetInfo objects

    Returns:
        Dictionary containing calculated statistics
    """
    try:
        if not datasets:
            return {
                "total_datasets": 0,
                "total_size_bytes": 0,
                "exchange_count": 0,
                "symbol_count": 0,
                "timeframe_count": 0,
                "format_count": 0,
            }

        # Basic counts
        total_datasets = len(datasets)
        total_size_bytes = sum(ds.size_bytes or 0 for ds in datasets)

        # Group by various attributes
        exchanges = {}
        symbols = {}
        timeframes = {}
        formats = {}

        for dataset in datasets:
            # Exchange statistics
            if dataset.exchange not in exchanges:
                exchanges[dataset.exchange] = {"count": 0, "size": 0}
            exchanges[dataset.exchange]["count"] += 1
            exchanges[dataset.exchange]["size"] += dataset.size_bytes or 0

            # Symbol statistics
            if dataset.symbol not in symbols:
                symbols[dataset.symbol] = {"count": 0, "size": 0}
            symbols[dataset.symbol]["count"] += 1
            symbols[dataset.symbol]["size"] += dataset.size_bytes or 0

            # Timeframe statistics
            if dataset.timeframe not in timeframes:
                timeframes[dataset.timeframe] = {"count": 0, "size": 0}
            timeframes[dataset.timeframe]["count"] += 1
            timeframes[dataset.timeframe]["size"] += dataset.size_bytes or 0

            # Format statistics
            if dataset.format_type:
                if dataset.format_type not in formats:
                    formats[dataset.format_type] = {"count": 0, "size": 0}
                formats[dataset.format_type]["count"] += 1
                formats[dataset.format_type]["size"] += dataset.size_bytes or 0

        # Format statistics
        exchange_stats = [
            {
                "exchange": ex,
                "dataset_count": info["count"],
                "total_size_bytes": info["size"],
            }
            for ex, info in exchanges.items()
        ]
        symbol_stats = [
            {
                "symbol": sym,
                "dataset_count": info["count"],
                "total_size_bytes": info["size"],
            }
            for sym, info in symbols.items()
        ]
        timeframe_stats = [
            {
                "timeframe": tf,
                "dataset_count": info["count"],
                "total_size_bytes": info["size"],
            }
            for tf, info in timeframes.items()
        ]
        format_stats = [
            {
                "format": fmt,
                "dataset_count": info["count"],
                "total_size_bytes": info["size"],
            }
            for fmt, info in formats.items()
        ]

        return {
            "total_datasets": total_datasets,
            "total_size_bytes": total_size_bytes,
            "exchange_count": len(exchanges),
            "symbol_count": len(symbols),
            "timeframe_count": len(timeframes),
            "format_count": len(formats),
            "exchange_statistics": exchange_stats,
            "symbol_statistics": symbol_stats,
            "timeframe_statistics": timeframe_stats,
            "format_statistics": format_stats,
            "average_dataset_size_bytes": (
                total_size_bytes / total_datasets if total_datasets > 0 else 0
            ),
        }

    except Exception as e:
        logger.error(f"Failed to calculate dataset statistics: {e}")
        return {
            "total_datasets": 0,
            "total_size_bytes": 0,
            "exchange_count": 0,
            "symbol_count": 0,
            "timeframe_count": 0,
            "format_count": 0,
            "error": str(e),
        }
