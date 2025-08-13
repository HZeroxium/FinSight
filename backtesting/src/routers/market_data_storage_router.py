# routers/market_data_storage_router.py

"""
Market Data Storage Router

RESTful endpoints for managing market data storage operations including:
- Object storage management (upload, download, list, delete)
- Dataset format conversion (CSV ↔ Parquet)
- Bulk data operations and archiving
- Storage statistics and monitoring
- User-friendly shortcut endpoints for common operations

Based on the storage service layer and cross-repository pipeline logic.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime

from ..services.market_data_storage_service import MarketDataStorageService
from ..misc.timeframe_load_convert_save import CrossRepositoryTimeFramePipeline
from ..schemas.enums import Exchange, TimeFrame, CryptoSymbol, RepositoryType
from ..schemas.storage_schema import (
    DatasetUploadRequest,
    DatasetDownloadRequest,
    FormatConversionRequest,
    BulkOperationRequest,
    StorageStatsResponse,
)
from ..utils.dependencies import (
    require_admin_access,
    get_storage_service,
    get_cross_repository_pipeline,
)
from common.logger import LoggerFactory, LoggerType, LogLevel

# Initialize logger
logger = LoggerFactory.get_logger(
    name="storage-router",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/storage_router.log",
)

# Create router
router = APIRouter(prefix="/storage", tags=["market-data-storage"])


@router.get("/query/{symbol}")
async def query_dataset_by_symbol(
    symbol: str,
    timeframe: Optional[str] = Query(
        TimeFrame.HOUR_1.value, description="Timeframe to query (defaults to 1h)"
    ),
    exchange: Optional[str] = Query(
        Exchange.BINANCE.value, description="Exchange name (defaults to binance)"
    ),
    format_type: Optional[str] = Query(
        RepositoryType.CSV.value,
        description="Data format to query (csv or parquet, defaults to csv)",
    ),
    storage_service: MarketDataStorageService = Depends(get_storage_service),
) -> Dict[str, Any]:
    """
    Query dataset information by symbol and timeframe.

    User-friendly endpoint that defaults exchange to "binance" and provides
    smart defaults for common parameters.
    """
    try:
        logger.info(
            f"Querying dataset: {exchange}/{symbol}/{timeframe} ({format_type})"
        )

        # Generate object key prefix for the dataset using settings
        from ..core.config import settings

        prefix = settings.build_dataset_path(
            exchange=exchange, symbol=symbol, timeframe=timeframe
        )

        result = await storage_service.list_datasets(
            prefix=prefix,
            limit=10,  # Limit to most recent datasets
        )

        datasets = result.get("datasets", [])
        filtered_datasets = [
            ds for ds in datasets if ds.get("format", "").lower() == format_type.lower()
        ]

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange": exchange,
            "format": format_type,
            "datasets": filtered_datasets,
            "total_count": len(filtered_datasets),
            "message": f"Found {len(filtered_datasets)} datasets for {symbol}/{timeframe}",
        }

    except Exception as e:
        logger.error(f"Error querying dataset for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset query failed: {str(e)}")


@router.get("/download/{symbol}")
async def download_dataset_by_symbol(
    symbol: str,
    timeframe: Optional[str] = Query(
        TimeFrame.HOUR_1.value, description="Timeframe to download (defaults to 1h)"
    ),
    exchange: Optional[str] = Query(
        Exchange.BINANCE.value, description="Exchange name (defaults to binance)"
    ),
    format_type: Optional[str] = Query(
        RepositoryType.CSV.value,
        description="Data format to download (csv or parquet, defaults to csv)",
    ),
    extract_archive: Optional[bool] = Query(
        False, description="Whether to extract archive files (defaults to False)"
    ),
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Download dataset by symbol and timeframe.

    User-friendly endpoint that automatically generates the object key
    based on symbol, timeframe, and exchange parameters.
    """
    try:
        logger.info(
            f"Downloading dataset: {exchange}/{symbol}/{timeframe} ({format_type})"
        )

        # Generate object key for the dataset using settings
        from ..core.config import settings

        object_key = (
            settings.build_dataset_path(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                format_type=format_type,
            )
            + f"/data.{format_type}"
        )

        result = await storage_service.download_dataset(
            object_key=object_key,
            extract_archives=extract_archive,
            # target_directory=target_directory,
        )

        if result["success"]:
            logger.info(f"Dataset download completed: {result['message']}")
        else:
            logger.warning(f"Dataset download failed: {result['message']}")

        return {
            **result,
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange": exchange,
            "format": format_type,
            "object_key": object_key,
        }

    except Exception as e:
        logger.error(f"Error downloading dataset for {symbol}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Dataset download failed: {str(e)}"
        )


@router.post("/upload/{symbol}")
async def upload_dataset_by_symbol(
    symbol: str,
    timeframe: Optional[str] = Query(
        TimeFrame.HOUR_1.value, description="Timeframe to upload (defaults to 1h)"
    ),
    exchange: Optional[str] = Query(
        Exchange.BINANCE.value, description="Exchange name (defaults to binance)"
    ),
    source_format: Optional[str] = Query(
        RepositoryType.CSV.value,
        description="Source data format (csv or parquet, defaults to csv)",
    ),
    target_format: Optional[str] = Query(
        RepositoryType.PARQUET.value,
        description="Target format for upload (csv or parquet, defaults to parquet)",
    ),
    start_date: Optional[str] = Query(
        default=None,
        description="Start date in ISO format (defaults to dataset's first date)",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="End date in ISO format (defaults to dataset's last date)",
    ),
    compress: Optional[bool] = Query(
        default=True, description="Whether to compress the dataset (defaults to True)"
    ),
    include_metadata: Optional[bool] = Query(
        default=True, description="Whether to include metadata files (defaults to True)"
    ),
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Upload dataset by symbol and timeframe.

    User-friendly endpoint that provides smart defaults for all parameters
    and automatically handles date ranges if not specified.

    If start_date and end_date are not provided, the endpoint will automatically
    determine the full date range of the available dataset.
    """
    try:
        logger.info(f"Uploading dataset: {exchange}/{symbol}/{timeframe}")

        # If dates are not provided, try to get them from the dataset
        if not start_date or not end_date:
            logger.info("No date range specified, attempting to get full dataset range")

            # Get source repository to determine data range
            source_repo, source_service = storage_service._get_repository_service(
                source_format
            )
            if not source_service:
                raise HTTPException(
                    status_code=500,
                    detail=f"Source repository not available for format: {source_format}",
                )

            # Get the full data range from the repository
            data_range = await source_service.repository.get_data_range(
                exchange=exchange,
                symbol=symbol,
                data_type="ohlcv",
                timeframe=timeframe,
            )

            if data_range:
                actual_start_date = start_date or data_range["start_date"]
                actual_end_date = end_date or data_range["end_date"]
                logger.info(
                    f"Auto-detected dataset range: {actual_start_date} to {actual_end_date}"
                )
            else:
                # Fallback to current date if no data exists
                current_date = datetime.now().strftime("%Y-%m-%d")
                actual_start_date = start_date or current_date
                actual_end_date = end_date or current_date
                logger.warning(
                    f"No data found, using fallback date range: {actual_start_date} to {actual_end_date}"
                )
        else:
            actual_start_date = start_date
            actual_end_date = end_date

        result = await storage_service.upload_dataset(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=actual_start_date,
            end_date=actual_end_date,
            source_format=source_format,
            target_format=target_format,
            compress=compress,
            include_metadata=include_metadata,
        )

        if result["success"]:
            logger.info(f"Dataset upload completed: {result['message']}")
        else:
            logger.warning(f"Dataset upload failed: {result['message']}")

        return {
            **result,
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange": exchange,
            "source_format": source_format,
            "target_format": target_format,
            "start_date": actual_start_date,
            "end_date": actual_end_date,
            "auto_detected_range": not (start_date and end_date),
        }

    except Exception as e:
        logger.error(f"Error uploading dataset for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset upload failed: {str(e)}")


@router.get("/list/simple")
async def list_datasets_simple(
    symbol: Optional[str] = Query(
        default=CryptoSymbol.BTCUSDT.value, description="Filter by trading symbol"
    ),
    timeframe: Optional[str] = Query(
        default=TimeFrame.HOUR_1.value, description="Filter by timeframe"
    ),
    exchange: Optional[str] = Query(
        Exchange.BINANCE.value, description="Filter by exchange (defaults to binance)"
    ),
    format_type: Optional[str] = Query(
        default=RepositoryType.CSV.value,
        description="Filter by data format (csv or parquet)",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="Maximum number of objects to return (defaults to 50)",
    ),
    storage_service: MarketDataStorageService = Depends(get_storage_service),
) -> Dict[str, Any]:
    """
    List datasets with simple filtering options.

    User-friendly endpoint that allows filtering by symbol, timeframe, exchange,
    and format with smart defaults.
    """
    try:
        # Import settings for prefix building
        from ..core.config import settings

        # Build prefix based on provided filters using settings
        if symbol and timeframe:
            prefix = settings.build_dataset_path(
                exchange=exchange, symbol=symbol, timeframe=timeframe
            )
        elif symbol:
            prefix = settings.build_storage_path(exchange, symbol)
        elif timeframe:
            prefix = settings.build_storage_path(exchange, timeframe)
        else:
            prefix = settings.build_storage_path(exchange)

        logger.info(
            f"Listing datasets with filters: symbol={symbol}, timeframe={timeframe}, exchange={exchange}, format={format_type}"
        )

        result = await storage_service.list_datasets(
            prefix=prefix,
            limit=limit,
        )

        datasets = result.get("datasets", [])

        # Apply format filter if specified
        if format_type:
            datasets = [
                ds
                for ds in datasets
                if ds.get("format", "").lower() == format_type.lower()
            ]

        return {
            "success": True,
            "filters": {
                "symbol": symbol,
                "timeframe": timeframe,
                "exchange": exchange,
                "format": format_type,
            },
            "datasets": datasets,
            "total_count": len(datasets),
            "limit": limit,
            "message": f"Found {len(datasets)} datasets matching criteria",
        }

    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset listing failed: {str(e)}")


# Storage management endpoints


@router.post("/upload")
async def upload_dataset(
    request: DatasetUploadRequest,
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Upload a dataset to object storage.

    Supports format conversion and compression during upload.

    If start_date and end_date are not provided, the endpoint will automatically
    determine the full date range of the available dataset.
    """
    try:
        # Apply smart defaults
        exchange = request.exchange or Exchange.BINANCE.value
        symbol = request.symbol or CryptoSymbol.BTCUSDT.value
        timeframe = request.timeframe or TimeFrame.HOUR_1.value
        source_format = request.source_format or RepositoryType.CSV.value
        target_format = request.target_format or RepositoryType.PARQUET.value
        compress = request.compress if request.compress is not None else True
        include_metadata = (
            request.include_metadata if request.include_metadata is not None else True
        )

        # Handle date defaults with auto-detection
        if not request.start_date or not request.end_date:
            logger.info("No date range specified, attempting to get full dataset range")

            # Get source repository to determine data range
            source_repo, source_service = storage_service._get_repository_service(
                source_format
            )
            if not source_service:
                raise HTTPException(
                    status_code=500,
                    detail=f"Source repository not available for format: {source_format}",
                )

            # Get the full data range from the repository
            data_range = await source_service.repository.get_data_range(
                exchange=exchange,
                symbol=symbol,
                data_type="ohlcv",
                timeframe=timeframe,
            )

            if data_range:
                start_date = request.start_date or data_range["start_date"]
                end_date = request.end_date or data_range["end_date"]
                logger.info(f"Auto-detected dataset range: {start_date} to {end_date}")
            else:
                # Fallback to current date if no data exists
                current_date = datetime.now().strftime("%Y-%m-%d")
                start_date = request.start_date or current_date
                end_date = request.end_date or current_date
                logger.warning(
                    f"No data found, using fallback date range: {start_date} to {end_date}"
                )
        else:
            start_date = request.start_date
            end_date = request.end_date

        logger.info(f"Starting dataset upload: {exchange}/{symbol}/{timeframe}")

        result = await storage_service.upload_dataset(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source_format=source_format,
            target_format=target_format,
            compress=compress,
            include_metadata=include_metadata,
        )

        if result["success"]:
            logger.info(f"Dataset upload completed: {result['message']}")
        else:
            logger.warning(f"Dataset upload failed: {result['message']}")

        # Add auto-detection info to response
        result["auto_detected_range"] = not (request.start_date and request.end_date)
        result["actual_start_date"] = start_date
        result["actual_end_date"] = end_date

        return result

    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset upload failed: {str(e)}")


@router.post("/download")
async def download_dataset(
    request: DatasetDownloadRequest,
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Download a dataset from object storage.

    Supports automatic archive extraction and object key auto-generation.
    """
    try:
        # Auto-generate object key if not provided
        if not request.object_key:
            exchange = request.exchange or Exchange.BINANCE.value
            symbol = request.symbol or CryptoSymbol.BTCUSDT.value
            timeframe = request.timeframe or TimeFrame.HOUR_1.value
            from ..core.config import settings

            object_key = (
                settings.build_dataset_path(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    format_type="csv",
                )
                + "/data.csv"
            )
        else:
            object_key = request.object_key

        logger.info(f"Starting dataset download: {object_key}")

        result = await storage_service.download_dataset(
            object_key=object_key,
            extract_archives=request.extract_archive or False,
        )

        if result["success"]:
            logger.info(f"Dataset download completed: {result['message']}")
        else:
            logger.warning(f"Dataset download failed: {result['message']}")

        return result

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise HTTPException(
            status_code=500, detail=f"Dataset download failed: {str(e)}"
        )


@router.get("/list")
async def list_datasets(
    prefix: Optional[str] = Query(
        default=None, description="Object key prefix for filtering"
    ),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of objects to return"
    ),
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    List available datasets in object storage.
    """
    try:
        logger.info(f"Listing datasets with prefix: {prefix}")

        result = await storage_service.list_datasets(
            prefix=prefix,
            limit=limit,
        )

        logger.info(f"Found {len(result.get('datasets', []))} datasets")
        return result

    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset listing failed: {str(e)}")


@router.delete("/dataset/{object_key:path}")
async def delete_dataset(
    object_key: str,
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Delete a dataset from object storage.
    """
    try:
        logger.info(f"Deleting dataset: {object_key}")

        result = await storage_service.delete_dataset(object_key)

        if result["success"]:
            logger.info(f"Dataset deleted successfully: {object_key}")
        else:
            logger.warning(f"Dataset deletion failed: {result['message']}")

        return result

    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(
            status_code=500, detail=f"Dataset deletion failed: {str(e)}"
        )


# Data conversion endpoints


@router.post("/convert")
async def convert_dataset_format(
    request: FormatConversionRequest,
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Convert dataset between formats (CSV ↔ Parquet).

    Based on timeframe_load_convert_save.py logic.

    If start_date and end_date are not provided, the endpoint will automatically
    determine the full date range of the available dataset.
    """
    try:
        # Apply smart defaults
        exchange = request.exchange or Exchange.BINANCE.value
        symbol = request.symbol or CryptoSymbol.BTCUSDT.value
        timeframe = request.timeframe or TimeFrame.HOUR_1.value
        source_format = request.source_format or RepositoryType.CSV.value
        target_format = request.target_format or RepositoryType.PARQUET.value
        overwrite_existing = request.overwrite_existing or False

        # Handle date defaults with auto-detection
        if not request.start_date or not request.end_date:
            logger.info("No date range specified, attempting to get full dataset range")

            # Get source repository to determine data range
            source_repo, source_service = storage_service._get_repository_service(
                source_format
            )
            if not source_service:
                raise HTTPException(
                    status_code=500,
                    detail=f"Source repository not available for format: {source_format}",
                )

            # Get the full data range from the repository
            data_range = await source_service.repository.get_data_range(
                exchange=exchange,
                symbol=symbol,
                data_type="ohlcv",
                timeframe=timeframe,
            )

            if data_range:
                start_date = request.start_date or data_range["start_date"]
                end_date = request.end_date or data_range["end_date"]
                logger.info(f"Auto-detected dataset range: {start_date} to {end_date}")
            else:
                # Fallback to current date if no data exists
                current_date = datetime.now().strftime("%Y-%m-%d")
                start_date = request.start_date or current_date
                end_date = request.end_date or current_date
                logger.warning(
                    f"No data found, using fallback date range: {start_date} to {end_date}"
                )
        else:
            start_date = request.start_date
            end_date = request.end_date

        logger.info(
            f"Starting format conversion: {source_format} -> {target_format} "
            f"for {exchange}/{symbol}/{timeframe}"
        )

        result = await storage_service.convert_dataset_format(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source_format=source_format,
            target_format=target_format,
            target_timeframes=request.target_timeframes,
            overwrite_existing=overwrite_existing,
        )

        if result["success"]:
            logger.info(f"Format conversion completed: {result['message']}")
        else:
            logger.warning(f"Format conversion failed: {result['message']}")

        # Add auto-detection info to response
        result["auto_detected_range"] = not (request.start_date and request.end_date)
        result["actual_start_date"] = start_date
        result["actual_end_date"] = end_date

        return result

    except Exception as e:
        logger.error(f"Error converting dataset format: {e}")
        raise HTTPException(
            status_code=500, detail=f"Format conversion failed: {str(e)}"
        )


@router.post("/convert/timeframes")
async def convert_timeframes(
    exchange: str = Query(Exchange.BINANCE.value, description="Exchange name"),
    symbol: str = Query(CryptoSymbol.BTCUSDT.value, description="Trading symbol"),
    source_timeframe: str = Query(
        TimeFrame.HOUR_1.value, description="Source timeframe"
    ),
    target_timeframes: List[str] = Query(
        default=[TimeFrame.HOUR_4.value, TimeFrame.DAY_1.value],
        description="Target timeframes for conversion",
    ),
    start_date: Optional[str] = Query(
        default=None,
        description="Start date in ISO format (defaults to dataset's first date)",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="End date in ISO format (defaults to dataset's last date)",
    ),
    source_format: str = Query(
        default=RepositoryType.CSV.value, description="Source repository format"
    ),
    target_format: str = Query(
        default=RepositoryType.PARQUET.value, description="Target repository format"
    ),
    overwrite_existing: bool = Query(False, description="Overwrite existing data"),
    pipeline: CrossRepositoryTimeFramePipeline = Depends(get_cross_repository_pipeline),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Convert timeframes using the cross-repository pipeline.

    References timeframe_load_convert_save.py logic for data conversion.

    If start_date and end_date are not provided, the endpoint will automatically
    determine the full date range of the available dataset.
    """
    try:
        logger.info(
            f"Starting timeframe conversion: {source_timeframe} -> {target_timeframes} "
            f"for {exchange}/{symbol}"
        )

        # Validate format parameters
        if source_format not in [
            RepositoryType.CSV.value,
            RepositoryType.PARQUET.value,
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source format: {source_format}. Supported formats: csv, parquet",
            )

        if target_format not in [
            RepositoryType.CSV.value,
            RepositoryType.PARQUET.value,
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported target format: {target_format}. Supported formats: csv, parquet",
            )

        # Handle date defaults with auto-detection
        actual_start_date = start_date
        actual_end_date = end_date

        if not start_date or not end_date:
            logger.info("No date range specified, attempting to get full dataset range")

            # Get source repository to determine data range
            from ..adapters.csv_market_data_repository import CSVMarketDataRepository
            from ..adapters.parquet_market_data_repository import (
                ParquetMarketDataRepository,
            )

            if source_format == RepositoryType.CSV.value:
                source_repo = CSVMarketDataRepository()
            elif source_format == RepositoryType.PARQUET.value:
                source_repo = ParquetMarketDataRepository()
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unsupported source format: {source_format}",
                )

            # Get the full data range from the repository
            data_range = await source_repo.get_data_range(
                exchange=exchange,
                symbol=symbol,
                data_type="ohlcv",
                timeframe=source_timeframe,
            )

            if data_range:
                actual_start_date = start_date or data_range["start_date"]
                actual_end_date = end_date or data_range["end_date"]
                logger.info(
                    f"Auto-detected dataset range: {actual_start_date} to {actual_end_date}"
                )
            else:
                # Fallback to current date if no data exists
                current_date = datetime.now().strftime("%Y-%m-%d")
                actual_start_date = start_date or current_date
                actual_end_date = end_date or current_date
                logger.warning(
                    f"No data found, using fallback date range: {actual_start_date} to {actual_end_date}"
                )

        # Configure pipeline repositories based on formats
        logger.info(
            f"Configuring repositories for {source_format} to {target_format} conversion"
        )

        if source_format == target_format:
            # Same format conversion (e.g., CSV to CSV, Parquet to Parquet)
            if source_format == RepositoryType.CSV.value:
                from ..adapters.csv_market_data_repository import (
                    CSVMarketDataRepository,
                )

                pipeline.source_repository = CSVMarketDataRepository()
                pipeline.target_repository = CSVMarketDataRepository()
                logger.info("Using CSV repository for both source and target")
            elif source_format == RepositoryType.PARQUET.value:
                from ..adapters.parquet_market_data_repository import (
                    ParquetMarketDataRepository,
                )

                pipeline.source_repository = ParquetMarketDataRepository()
                pipeline.target_repository = ParquetMarketDataRepository()
                logger.info("Using Parquet repository for both source and target")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported format for same-format conversion: {source_format}",
                )
        else:
            # Cross-repository conversion
            if (
                source_format == RepositoryType.CSV.value
                and target_format == RepositoryType.PARQUET.value
            ):
                from ..adapters.csv_market_data_repository import (
                    CSVMarketDataRepository,
                )
                from ..adapters.parquet_market_data_repository import (
                    ParquetMarketDataRepository,
                )

                pipeline.source_repository = CSVMarketDataRepository()
                pipeline.target_repository = ParquetMarketDataRepository()
                logger.info(
                    "Using CSV repository as source, Parquet repository as target"
                )
            elif (
                source_format == RepositoryType.PARQUET.value
                and target_format == RepositoryType.CSV.value
            ):
                from ..adapters.parquet_market_data_repository import (
                    ParquetMarketDataRepository,
                )
                from ..adapters.csv_market_data_repository import (
                    CSVMarketDataRepository,
                )

                pipeline.source_repository = ParquetMarketDataRepository()
                pipeline.target_repository = CSVMarketDataRepository()
                logger.info(
                    "Using Parquet repository as source, CSV repository as target"
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported format combination: {source_format} to {target_format}",
                )

        # Configure target timeframes
        pipeline.target_timeframes = target_timeframes

        # Validate pipeline configuration
        if not pipeline.source_repository or not pipeline.target_repository:
            raise HTTPException(
                status_code=500,
                detail="Pipeline repositories not properly configured",
            )

        logger.info(
            f"Pipeline configured with source: {type(pipeline.source_repository).__name__}"
        )
        logger.info(
            f"Pipeline configured with target: {type(pipeline.target_repository).__name__}"
        )

        # Run the conversion pipeline
        result = await pipeline.run_cross_repository_pipeline(
            symbols=[symbol],
            exchange=exchange,
            start_date=actual_start_date,
            end_date=actual_end_date,
            overwrite_existing=overwrite_existing,
        )

        success = result.get("errors", []) == []
        if success:
            logger.info(f"Timeframe conversion completed for {symbol}")
        else:
            logger.warning(
                f"Timeframe conversion failed for {symbol}: {result.get('message', '')}"
            )

        return {
            "success": success,
            "message": result.get("message", "Timeframe conversion completed"),
            "symbol": symbol,
            "source_timeframe": source_timeframe,
            "target_timeframes": target_timeframes,
            "statistics": result.get("timeframe_statistics", {}),
            "converted_records": result.get("total_conversions", 0),
            "auto_detected_range": not (start_date and end_date),
            "actual_start_date": actual_start_date,
            "actual_end_date": actual_end_date,
        }

    except Exception as e:
        logger.error(f"Error converting timeframes: {e}")
        raise HTTPException(
            status_code=500, detail=f"Timeframe conversion failed: {str(e)}"
        )


# Bulk operations


@router.post("/bulk")
async def bulk_operations(
    request: BulkOperationRequest,
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Perform bulk storage operations.

    Supports concurrent execution with error handling.
    """
    try:
        logger.info(f"Starting bulk operations: {len(request.operations)} operations")

        result = await storage_service.bulk_upload_datasets(
            datasets=request.operations,
            source_format=request.source_format or "csv",
            target_format=request.target_format or "parquet",
            max_concurrent=request.max_concurrent,
        )

        logger.info(f"Bulk operations completed: {result.get('message', '')}")
        return result

    except Exception as e:
        logger.error(f"Error in bulk operations: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk operations failed: {str(e)}")


# Storage statistics and monitoring


@router.get("/stats", response_model=StorageStatsResponse)
async def get_storage_statistics(
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> StorageStatsResponse:
    """
    Get storage statistics and health information.
    """
    try:
        logger.info("Retrieving storage statistics")

        stats = await storage_service.get_storage_statistics()

        return StorageStatsResponse(
            total_objects=stats.get("total_objects", 0),
            total_size_bytes=stats.get("total_size_bytes", 0),
            datasets_by_format=stats.get("datasets_by_format", {}),
            datasets_by_exchange=stats.get("datasets_by_exchange", {}),
            storage_health=stats.get("storage_health", {}),
            last_updated=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error retrieving storage statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Storage statistics retrieval failed: {str(e)}"
        )


@router.get("/health")
async def storage_health_check(
    storage_service: MarketDataStorageService = Depends(get_storage_service),
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Check storage service health and connectivity.
    """
    try:
        # Test storage client connectivity
        storage_info = await storage_service.storage_client.get_storage_info()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "storage_client": "connected",
            "storage_info": storage_info,
            "service": "market-data-storage",
        }

    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "storage_client": "disconnected",
            "error": str(e),
            "service": "market-data-storage",
        }


# Service information


@router.get("/")
async def storage_service_info() -> Dict[str, Any]:
    """
    Get storage service information and available endpoints.
    """
    return {
        "service": "Market Data Storage Service",
        "version": "1.0.0",
        "description": "Object storage management and data conversion for market data",
        "endpoints": {
            # User-friendly shortcut endpoints
            "query_by_symbol": "GET /storage/query/{symbol} - Query dataset by symbol and timeframe",
            "download_by_symbol": "GET /storage/download/{symbol} - Download dataset by symbol and timeframe",
            "upload_by_symbol": "POST /storage/upload/{symbol} - Upload dataset by symbol and timeframe",
            "list_simple": "GET /storage/list/simple - List datasets with simple filtering",
            # Standard endpoints
            "upload": "POST /storage/upload - Upload dataset to object storage",
            "download": "POST /storage/download - Download dataset from object storage",
            "list": "GET /storage/list - List available datasets",
            "delete": "DELETE /storage/dataset/{object_key} - Delete dataset",
            "convert": "POST /storage/convert - Convert dataset format",
            "convert_timeframes": "POST /storage/convert/timeframes - Convert timeframes",
            "bulk": "POST /storage/bulk - Bulk operations",
            "stats": "GET /storage/stats - Storage statistics",
            "health": "GET /storage/health - Health check",
        },
        "supported_formats": [
            RepositoryType.CSV.value,
            RepositoryType.PARQUET.value,
        ],
        "supported_operations": [
            "upload",
            "download",
            "format_conversion",
            "timeframe_conversion",
            "bulk_operations",
            "compression",
        ],
        "features": [
            "S3-compatible object storage",
            "Cross-repository format conversion",
            "Timeframe aggregation",
            "Compression and archiving",
            "Bulk operations",
            "Storage monitoring",
            "User-friendly shortcut endpoints",
            "Smart defaults for common parameters",
        ],
        "smart_defaults": {
            "exchange": Exchange.BINANCE.value,
            "symbol": CryptoSymbol.BTCUSDT.value,
            "timeframe": TimeFrame.HOUR_1.value,
            "source_format": RepositoryType.CSV.value,
            "target_format": RepositoryType.PARQUET.value,
            "compress": True,
            "include_metadata": True,
        },
    }
