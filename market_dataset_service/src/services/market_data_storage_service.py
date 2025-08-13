# services/market_data_storage_service.py

"""
Market Data Storage Service

Handles upload, download, and management of market data datasets
in object storage (S3-compatible) with format conversion support.
"""

import asyncio
import json
import zipfile
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..core.config import settings
from ..utils.storage_client import StorageClient
from ..adapters.csv_market_data_repository import CSVMarketDataRepository
from ..adapters.parquet_market_data_repository import ParquetMarketDataRepository
from ..interfaces.market_data_repository import MarketDataRepository
from ..services.market_data_service import MarketDataService
from ..schemas.ohlcv_schemas import OHLCVQuerySchema, OHLCVSchema
from common.logger import LoggerFactory, LoggerType, LogLevel
from ..utils.datetime_utils import DateTimeUtils


class RepositoryError(Exception):
    """Exception raised for repository-related errors"""

    pass


class MarketDataStorageService:
    """
    Service for managing market data storage operations.

    Handles upload, download, format conversion, and dataset management
    in object storage with support for multiple formats and timeframes.
    """

    def __init__(
        self,
        storage_client: StorageClient,
        csv_repository: Optional[CSVMarketDataRepository] = None,
        parquet_repository: Optional[ParquetMarketDataRepository] = None,
        temp_dir: str = "temp/storage",
        max_workers: int = 4,
        chunk_size: int = 1000,
        compression_level: int = 6,
    ):
        """
        Initialize the storage service.

        Args:
            storage_client: Storage client for object storage operations
            csv_repository: CSV repository for data access
            parquet_repository: Parquet repository for data access
            temp_dir: Temporary directory for file operations
            max_workers: Maximum number of concurrent workers
            chunk_size: Size of data chunks for processing
            compression_level: Compression level for archives
        """
        self.storage_client = storage_client
        self.csv_repository = csv_repository
        self.parquet_repository = parquet_repository
        self.temp_dir = Path(temp_dir)
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.compression_level = compression_level

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name="storage-service",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/storage_service.log",
        )

        # Initialize services for the repositories if available
        self.csv_service = None
        if self.csv_repository:
            self.csv_service = MarketDataService(self.csv_repository)
            self.logger.info("CSV repository service initialized")

        self.parquet_service = None
        if self.parquet_repository:
            self.parquet_service = MarketDataService(self.parquet_repository)
            self.logger.info("Parquet repository service initialized")

        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Market Data Storage Service initialized")

    async def upload_dataset(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source_format: str = "csv",
        target_format: str = "parquet",
        compress: bool = True,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Upload a dataset to object storage.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date in ISO format
            end_date: End date in ISO format
            source_format: Source format for upload ('csv' or 'parquet')
            target_format: Target format for upload ('csv' or 'parquet')
            compress: Whether to compress the dataset
            include_metadata: Whether to include metadata files

        Returns:
            Upload result dictionary
        """
        try:
            self.logger.info(
                f"Starting dataset upload: {exchange}/{symbol}/{timeframe} "
                f"({start_date} to {end_date})"
            )

            # Get source repository and service
            source_repo, source_service = self._get_repository_service(source_format)
            if not source_service:
                raise RepositoryError(
                    f"Source repository not available for format: {source_format}"
                )

            # Retrieve data from source
            query = OHLCVQuerySchema(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=DateTimeUtils.parse_iso_string(start_date),
                end_date=DateTimeUtils.parse_iso_string(end_date),
            )

            data = await source_service.get_ohlcv_data(
                exchange, symbol, timeframe, start_date, end_date
            )

            if not data.data:
                return {
                    "success": False,
                    "message": "No data found for the specified criteria",
                    "records_processed": 0,
                }

            # Create temporary working directory
            work_dir = (
                self.temp_dir
                / f"upload_{exchange}_{symbol}_{timeframe}_{int(datetime.now().timestamp())}"
            )
            work_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Convert to target format if needed
                if source_format != target_format:
                    converted_data = await self._convert_data_format(
                        data.data, source_format, target_format, work_dir
                    )
                else:
                    converted_data = data.data

                # Save data to temporary files
                file_path = await self._save_temp_data(
                    converted_data, exchange, symbol, timeframe, target_format, work_dir
                )

                # Create metadata if requested
                metadata_files = []
                if include_metadata:
                    metadata_files = await self._create_metadata_files(
                        exchange, symbol, timeframe, data, work_dir
                    )

                # Create archive if compression is requested
                if compress:
                    archive_path = await self._create_archive(
                        work_dir,
                        [file_path] + metadata_files,
                        f"{exchange}_{symbol}_{timeframe}",
                    )
                    upload_files = [archive_path]
                else:
                    upload_files = [file_path] + metadata_files

                # Upload to object storage
                uploaded_objects = []
                for file_to_upload in upload_files:
                    object_key = self._generate_object_key(
                        exchange, symbol, timeframe, file_to_upload, target_format
                    )

                    success = await self.storage_client.upload_file(
                        local_file_path=file_to_upload,
                        object_key=object_key,
                        metadata={
                            "exchange": exchange,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "start_date": start_date,
                            "end_date": end_date,
                            "format": target_format,
                            "records": str(len(data.data)),
                        },
                    )

                    if success:
                        uploaded_objects.append(
                            {
                                "object_key": object_key,
                                "size": file_to_upload.stat().st_size,
                                "file_name": file_to_upload.name,
                            }
                        )
                    else:
                        raise RepositoryError(
                            f"Failed to upload file: {file_to_upload}"
                        )

                # Calculate total size
                total_size = sum(obj["size"] for obj in uploaded_objects)

                result = {
                    "success": True,
                    "message": f"Upload completed: {len(uploaded_objects)} files uploaded",
                    "records_processed": len(data.data),
                    "uploaded_objects": uploaded_objects,
                    "total_size_bytes": total_size,
                    "format": target_format,
                    "compressed": compress,
                }

                self.logger.info(f"Dataset upload completed: {result['message']}")
                return result

            finally:
                # Clean up temporary files
                try:
                    import shutil

                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp directory: {e}")

        except Exception as e:
            self.logger.error(f"Error uploading dataset: {e}")
            return {
                "success": False,
                "message": f"Upload failed: {str(e)}",
                "records_processed": 0,
            }

    async def download_dataset(
        self,
        object_key: str,
        target_format: str = "csv",
        local_path: Optional[str] = None,
        extract_archives: bool = True,
    ) -> Dict[str, Any]:
        """
        Download a dataset from object storage.

        Args:
            object_key: Object key in storage
            target_format: Target format for download ('csv' or 'parquet')
            local_path: Local destination path (optional)
            extract_archives: Whether to extract compressed archives

        Returns:
            Download result dictionary
        """
        try:
            self.logger.info(f"Starting dataset download: {object_key}")

            # Generate local path if not provided
            if not local_path:
                timestamp = int(datetime.now().timestamp())
                local_path = self.temp_dir / f"download_{timestamp}"
            else:
                local_path = Path(local_path)

            local_path.mkdir(parents=True, exist_ok=True)

            # Download from object storage
            download_file = local_path / Path(object_key).name
            success = await self.storage_client.download_file(
                object_key=object_key,
                local_file_path=download_file,
            )

            if not success:
                return {
                    "success": False,
                    "message": f"Failed to download object: {object_key}",
                }

            # Extract archive if it's compressed and extraction is requested
            extracted_files = []
            if extract_archives and self._is_archive(download_file):
                extracted_files = await self._extract_archive(download_file, local_path)

            # Get object metadata
            object_info = await self.storage_client.get_object_info(object_key)

            result = {
                "success": True,
                "message": "Download completed successfully",
                "local_path": str(local_path),
                "downloaded_file": str(download_file),
                "extracted_files": [str(f) for f in extracted_files],
                "object_info": object_info,
                "size_bytes": download_file.stat().st_size,
            }

            self.logger.info(f"Dataset download completed: {object_key}")
            return result

        except Exception as e:
            self.logger.error(f"Error downloading dataset: {e}")
            return {
                "success": False,
                "message": f"Download failed: {str(e)}",
            }

    async def convert_dataset_format(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source_format: str,
        target_format: str,
        upload_result: bool = True,
        target_timeframes: Optional[List[str]] = None,
        overwrite_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert dataset between different formats.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            start_date: Start date in ISO format
            end_date: End date in ISO format
            source_format: Source format ('csv' or 'parquet')
            target_format: Target format ('csv' or 'parquet')
            upload_result: Whether to upload converted dataset
            target_timeframes: List of target timeframes for conversion (optional)
            overwrite_existing: Whether to overwrite existing data

        Returns:
            Conversion result dictionary
        """
        try:
            if source_format == target_format:
                return {
                    "success": False,
                    "message": "Source and target formats are the same",
                }

            self.logger.info(
                f"Starting format conversion: {source_format} → {target_format} "
                f"for {exchange}/{symbol}/{timeframe}"
            )

            # Get repositories
            source_repo, source_service = self._get_repository_service(source_format)
            target_repo, target_service = self._get_repository_service(target_format)

            if not source_service or not target_service:
                return {
                    "success": False,
                    "message": "Required repositories not available",
                }

            # Convert data by reading from source and saving to target
            # Get data from source service
            source_data = await source_service.get_ohlcv_data(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            if not source_data.data:
                return {
                    "success": False,
                    "message": "No data found in source repository",
                    "total_records_processed": 0,
                }

            # Save data to target service
            save_success = await target_service.save_ohlcv_data(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                data=source_data.data,
                validate=True,
            )

            result = {
                "success": save_success,
                "total_records_processed": len(source_data.data) if save_success else 0,
                "source_format": source_format,
                "target_format": target_format,
            }

            conversion_result = {
                "success": result.get("success", False),
                "message": f"Conversion completed: {source_format} → {target_format}",
                "source_format": source_format,
                "target_format": target_format,
                "records_processed": result.get("total_records_processed", 0),
                "conversion_details": result,
            }

            # Upload converted dataset if requested
            if upload_result and conversion_result["success"]:
                upload_result_data = await self.upload_dataset(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    source_format=target_format,
                    target_format=target_format,
                    compress=True,
                )
                conversion_result["upload_result"] = upload_result_data

            self.logger.info(
                f"Format conversion completed: {conversion_result['message']}"
            )
            return conversion_result

        except Exception as e:
            self.logger.error(f"Error converting dataset format: {e}")
            return {
                "success": False,
                "message": f"Conversion failed: {str(e)}",
            }

    async def bulk_upload_datasets(
        self,
        datasets: List[Dict[str, Any]],
        source_format: str = "csv",
        target_format: str = "parquet",
        max_concurrent: int = 3,
    ) -> Dict[str, Any]:
        """
        Upload multiple datasets concurrently.

        Args:
            datasets: List of dataset specifications
            source_format: Source data format
            target_format: Target format for upload
            max_concurrent: Maximum concurrent uploads

        Returns:
            Bulk upload result dictionary
        """
        try:
            self.logger.info(f"Starting bulk upload of {len(datasets)} datasets")

            # Validate datasets
            for i, dataset in enumerate(datasets):
                required_fields = [
                    "exchange",
                    "symbol",
                    "timeframe",
                    "start_date",
                    "end_date",
                ]
                missing_fields = [
                    field for field in required_fields if field not in dataset
                ]
                if missing_fields:
                    raise ValueError(
                        f"Dataset {i}: missing required fields: {missing_fields}"
                    )

            # Use semaphore to limit concurrent uploads
            semaphore = asyncio.Semaphore(max_concurrent)

            async def upload_single_dataset(dataset):
                async with semaphore:
                    return await self.upload_dataset(
                        exchange=dataset["exchange"],
                        symbol=dataset["symbol"],
                        timeframe=dataset["timeframe"],
                        start_date=dataset["start_date"],
                        end_date=dataset["end_date"],
                        source_format=source_format,
                        target_format=target_format,
                        compress=dataset.get("compress", True),
                        include_metadata=dataset.get("include_metadata", True),
                    )

            # Execute uploads concurrently
            upload_tasks = [upload_single_dataset(dataset) for dataset in datasets]
            results = await asyncio.gather(*upload_tasks, return_exceptions=True)

            # Process results
            successful_uploads = 0
            failed_uploads = 0
            total_records = 0
            total_size = 0
            detailed_results = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_uploads += 1
                    detailed_results.append(
                        {
                            "dataset_index": i,
                            "dataset": datasets[i],
                            "success": False,
                            "error": str(result),
                        }
                    )
                elif result.get("success", False):
                    successful_uploads += 1
                    total_records += result.get("records_processed", 0)
                    total_size += result.get("total_size_bytes", 0)
                    detailed_results.append(
                        {
                            "dataset_index": i,
                            "dataset": datasets[i],
                            "success": True,
                            "result": result,
                        }
                    )
                else:
                    failed_uploads += 1
                    detailed_results.append(
                        {
                            "dataset_index": i,
                            "dataset": datasets[i],
                            "success": False,
                            "error": result.get("message", "Unknown error"),
                        }
                    )

            bulk_result = {
                "success": successful_uploads > 0,
                "message": f"Bulk upload completed: {successful_uploads} successful, {failed_uploads} failed",
                "total_datasets": len(datasets),
                "successful_uploads": successful_uploads,
                "failed_uploads": failed_uploads,
                "total_records_processed": total_records,
                "total_size_bytes": total_size,
                "detailed_results": detailed_results,
            }

            self.logger.info(f"Bulk upload completed: {bulk_result['message']}")
            return bulk_result

        except Exception as e:
            self.logger.error(f"Error in bulk upload: {e}")
            return {
                "success": False,
                "message": f"Bulk upload failed: {str(e)}",
                "total_datasets": len(datasets),
                "successful_uploads": 0,
                "failed_uploads": len(datasets),
            }

    async def list_datasets(
        self,
        exchange_filter: Optional[str] = None,
        symbol_filter: Optional[str] = None,
        timeframe_filter: Optional[str] = None,
        prefix: str = None,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """
        List available datasets in object storage.

        Args:
            exchange_filter: Filter by exchange
            symbol_filter: Filter by symbol
            timeframe_filter: Filter by timeframe
            prefix: Object key prefix to search (if None, uses configured default)
            limit: Maximum number of objects to return

        Returns:
            Dataset listing result
        """
        try:
            self.logger.info("Listing available datasets")

            # Use configured prefix if none provided
            if prefix is None:
                prefix = settings.get_storage_prefix() + "/"
            elif not prefix.startswith(settings.get_storage_prefix() + "/"):
                # Ensure prefix includes the configured storage prefix
                prefix = settings.build_storage_path(prefix)

            self.logger.info(f"Using prefix for listing: {prefix}")

            # List objects from storage
            objects = await self.storage_client.list_objects(
                prefix=prefix, max_keys=limit
            )

            self.logger.info(f"Found {len(objects)} objects in storage")

            # Parse and filter datasets
            datasets = []
            for obj in objects:
                try:
                    # Parse object key to extract metadata
                    dataset_info = self._parse_object_key(obj["key"])
                    if dataset_info:
                        # Apply filters
                        if (
                            exchange_filter
                            and dataset_info.get("exchange") != exchange_filter
                        ):
                            continue
                        if (
                            symbol_filter
                            and dataset_info.get("symbol") != symbol_filter
                        ):
                            continue
                        if (
                            timeframe_filter
                            and dataset_info.get("timeframe") != timeframe_filter
                        ):
                            continue

                        # Add object information
                        dataset_info.update(
                            {
                                "object_key": obj["key"],
                                "size": obj["size"],
                                "last_modified": obj["last_modified"],
                                "etag": obj["etag"],
                            }
                        )
                        datasets.append(dataset_info)

                except Exception as e:
                    self.logger.warning(f"Failed to parse object key {obj['key']}: {e}")
                    continue

            # Sort datasets by exchange, symbol, timeframe
            datasets.sort(
                key=lambda x: (
                    x.get("exchange", ""),
                    x.get("symbol", ""),
                    x.get("timeframe", ""),
                    x.get("last_modified", ""),
                )
            )

            # Calculate summary statistics
            total_size = sum(ds["size"] for ds in datasets)
            exchanges = set(ds.get("exchange") for ds in datasets if ds.get("exchange"))
            symbols = set(ds.get("symbol") for ds in datasets if ds.get("symbol"))
            timeframes = set(
                ds.get("timeframe") for ds in datasets if ds.get("timeframe")
            )

            result = {
                "success": True,
                "message": f"Found {len(datasets)} datasets",
                "datasets": datasets,
                "total_datasets": len(datasets),
                "total_size_bytes": total_size,
                "unique_exchanges": sorted(list(exchanges)),
                "unique_symbols": sorted(list(symbols)),
                "unique_timeframes": sorted(list(timeframes)),
            }

            self.logger.info(
                f"Dataset listing completed: {len(datasets)} datasets found"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error listing datasets: {e}")
            return {
                "success": False,
                "message": f"Failed to list datasets: {str(e)}",
                "datasets": [],
                "total_datasets": 0,
                "total_size_bytes": 0,
                "unique_exchanges": [],
                "unique_symbols": [],
                "unique_timeframes": [],
            }

    async def delete_dataset(
        self,
        object_key: str,
        confirm_deletion: bool = False,
    ) -> Dict[str, Any]:
        """
        Delete a dataset from object storage.

        Args:
            object_key: Object key to delete
            confirm_deletion: Confirmation flag for deletion

        Returns:
            Deletion result dictionary
        """
        try:
            if not confirm_deletion:
                return {
                    "success": False,
                    "message": "Deletion not confirmed. Set confirm_deletion=True to proceed.",
                }

            self.logger.info(f"Deleting dataset: {object_key}")

            # Get object info before deletion
            object_info = await self.storage_client.get_object_info(object_key)
            if not object_info:
                return {
                    "success": False,
                    "message": f"Object not found: {object_key}",
                }

            # Delete object
            success = await self.storage_client.delete_object(object_key)

            if success:
                result = {
                    "success": True,
                    "message": f"Dataset deleted successfully: {object_key}",
                    "deleted_object": object_info,
                }
                self.logger.info(f"Dataset deleted: {object_key}")
            else:
                result = {
                    "success": False,
                    "message": f"Failed to delete dataset: {object_key}",
                }

            return result

        except Exception as e:
            self.logger.error(f"Error deleting dataset: {e}")
            return {
                "success": False,
                "message": f"Deletion failed: {str(e)}",
            }

    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        try:
            self.logger.info("Collecting storage statistics")

            # Get basic storage info
            storage_info = await self.storage_client.get_storage_info()

            # Get repository statistics if available
            csv_stats = {}
            parquet_stats = {}

            if self.csv_repository:
                try:
                    csv_storage_info = await self.csv_repository.get_storage_info()
                    csv_stats = {
                        "available": True,
                        "storage_info": csv_storage_info,
                    }
                except Exception as e:
                    csv_stats = {"available": False, "error": str(e)}

            if self.parquet_repository:
                try:
                    parquet_storage_info = (
                        await self.parquet_repository.get_storage_info()
                    )
                    parquet_stats = {
                        "available": True,
                        "storage_info": parquet_storage_info,
                    }
                except Exception as e:
                    parquet_stats = {"available": False, "error": str(e)}

            # Get dataset listing for additional stats
            datasets_info = await self.list_datasets()

            result = {
                "timestamp": datetime.now().isoformat(),
                "object_storage": storage_info,
                "csv_repository": csv_stats,
                "parquet_repository": parquet_stats,
                "datasets_summary": {
                    "total_datasets": datasets_info.get("total_datasets", 0),
                    "total_size_bytes": datasets_info.get("total_size_bytes", 0),
                    "unique_exchanges": datasets_info.get("unique_exchanges", []),
                    "unique_symbols": datasets_info.get("unique_symbols", []),
                    "unique_timeframes": datasets_info.get("unique_timeframes", []),
                },
                "temp_directory": {
                    "path": str(self.temp_dir),
                    "exists": self.temp_dir.exists(),
                    "size_bytes": (
                        self._get_directory_size(self.temp_dir)
                        if self.temp_dir.exists()
                        else 0
                    ),
                },
            }

            self.logger.info("Storage statistics collected successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error collecting storage statistics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # Helper methods

    def _get_repository_service(
        self, format_type: str
    ) -> Tuple[Optional[MarketDataRepository], Optional[MarketDataService]]:
        """Get repository and service for the specified format."""
        if format_type.lower() == "csv":
            return self.csv_repository, self.csv_service
        elif format_type.lower() == "parquet":
            return self.parquet_repository, self.parquet_service
        else:
            return None, None

    async def _convert_data_format(
        self,
        data: List[OHLCVSchema],
        source_format: str,
        target_format: str,
        work_dir: Path,
    ) -> List[OHLCVSchema]:
        """Convert data between formats (placeholder for now)."""
        # For now, just return the data as-is since the schemas are the same
        # In a more complex implementation, this could handle format-specific conversions
        return data

    async def _save_temp_data(
        self,
        data: List[OHLCVSchema],
        exchange: str,
        symbol: str,
        timeframe: str,
        format_type: str,
        work_dir: Path,
    ) -> Path:
        """Save data to temporary files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{exchange}_{symbol}_{timeframe}_{timestamp}.{format_type}"
        file_path = work_dir / filename

        # Get appropriate repository for saving
        _, service = self._get_repository_service(format_type)
        if service:
            # Create a temporary batch and save it
            # Assuming OHLCVBatchSchema is no longer needed or replaced
            # For now, we'll create a simple file
            if format_type == "csv":
                # Save as CSV (simplified)
                import csv

                with open(file_path, "w", newline="") as csvfile:
                    fieldnames = ["timestamp", "open", "high", "low", "close", "volume"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for record in data:
                        writer.writerow(
                            {
                                "timestamp": record.timestamp.isoformat(),
                                "open": record.open,
                                "high": record.high,
                                "low": record.low,
                                "close": record.close,
                                "volume": record.volume,
                            }
                        )
            else:
                # Save as Parquet (simplified)
                import pandas as pd

                df_data = []
                for record in data:
                    df_data.append(
                        {
                            "timestamp": record.timestamp,
                            "open": record.open,
                            "high": record.high,
                            "low": record.low,
                            "close": record.close,
                            "volume": record.volume,
                        }
                    )
                df = pd.DataFrame(df_data)
                df.to_parquet(file_path, index=False)

        return file_path

    async def _create_metadata_files(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data_response,
        work_dir: Path,
    ) -> List[Path]:
        """Create metadata files for the dataset."""
        metadata_files = []

        # Create dataset info file
        info_file = work_dir / "dataset_info.json"
        info_data = {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": data_response.start_date.isoformat(),
            "end_date": data_response.end_date.isoformat(),
            "record_count": len(data_response.data),
            "generated_at": datetime.now().isoformat(),
            "has_more": data_response.has_more,
        }

        with open(info_file, "w") as f:
            json.dump(info_data, f, indent=2)
        metadata_files.append(info_file)

        # Create README file
        readme_file = work_dir / "README.md"
        readme_content = f"""# Market Data Dataset

## Dataset Information
- **Exchange**: {exchange}
- **Symbol**: {symbol}
- **Timeframe**: {timeframe}
- **Date Range**: {data_response.start_date.isoformat()} to {data_response.end_date.isoformat()}
- **Records**: {len(data_response.data)}
- **Generated**: {datetime.now().isoformat()}

## Data Format
This dataset contains OHLCV (Open, High, Low, Close, Volume) data for the specified symbol and timeframe.

## Files
- Main data file: Contains the market data in the specified format
- dataset_info.json: Metadata about the dataset
- README.md: This documentation file
"""

        with open(readme_file, "w") as f:
            f.write(readme_content)
        metadata_files.append(readme_file)

        return metadata_files

    async def _create_archive(
        self,
        work_dir: Path,
        files: List[Path],
        archive_name: str,
        format_type: str = "zip",
    ) -> Path:
        """Create compressed archive of files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_type == "zip":
            archive_path = work_dir / f"{archive_name}_{timestamp}.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files:
                    if file_path.exists():
                        zipf.write(file_path, file_path.name)
        else:
            # Default to tar.gz
            archive_path = work_dir / f"{archive_name}_{timestamp}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tarf:
                for file_path in files:
                    if file_path.exists():
                        tarf.add(file_path, file_path.name)

        return archive_path

    def _generate_object_key(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        file_path: Path,
        format_type: str,
    ) -> str:
        """Generate object key for storage using configured prefix."""
        timestamp = datetime.now().strftime("%Y%m%d")

        # Use settings to build the path consistently
        return (
            settings.build_dataset_path(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                format_type=format_type,
                date=timestamp,
            )
            + f"/{file_path.name}"
        )

    def _is_archive(self, file_path: Path) -> bool:
        """Check if file is an archive."""
        archive_extensions = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2"}
        return file_path.suffix.lower() in archive_extensions or str(
            file_path
        ).endswith(".tar.gz")

    async def _extract_archive(
        self, archive_path: Path, extract_dir: Path
    ) -> List[Path]:
        """Extract archive and return list of extracted files."""
        extracted_files = []

        try:
            if archive_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zipf:
                    zipf.extractall(extract_dir)
                    extracted_files = [extract_dir / name for name in zipf.namelist()]
            elif archive_path.suffix.lower() in {
                ".tar",
                ".tar.gz",
                ".tgz",
                ".tar.bz2",
            } or str(archive_path).endswith(".tar.gz"):
                with tarfile.open(archive_path, "r:*") as tarf:
                    tarf.extractall(extract_dir)
                    extracted_files = [extract_dir / name for name in tarf.getnames()]

        except Exception as e:
            self.logger.warning(f"Failed to extract archive {archive_path}: {e}")

        return extracted_files

    def _parse_object_key(self, object_key: str) -> Optional[Dict[str, Any]]:
        """Parse object key to extract dataset metadata."""
        try:
            # Expected format: {storage_prefix}/{exchange}/{symbol}/{timeframe}/{format}/{date}/{filename}
            # Example: finsight/market_data/datasets/binance/BTCUSDT/1h/parquet/20250809/filename.zip
            parts = object_key.split("/")

            # Check if the object key starts with the configured storage prefix
            storage_prefix_parts = settings.get_storage_prefix().split("/")
            if len(parts) >= len(storage_prefix_parts) + 6:
                # Check if the beginning matches the storage prefix
                if all(
                    parts[i] == storage_prefix_parts[i]
                    for i in range(len(storage_prefix_parts))
                ):
                    offset = len(storage_prefix_parts)
                    return {
                        "exchange": parts[offset],
                        "symbol": parts[offset + 1],
                        "timeframe": parts[offset + 2],
                        "format": parts[offset + 3],
                        "date": parts[offset + 4],
                        "filename": parts[-1],
                    }

            # Fallback for old format or different prefix
            elif len(parts) >= 6 and parts[0] == "datasets":
                return {
                    "exchange": parts[1],
                    "symbol": parts[2],
                    "timeframe": parts[3],
                    "format": parts[4],
                    "date": parts[5],
                    "filename": parts[-1],
                }
        except Exception:
            pass
        return None

    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        try:
            total_size = 0
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0
