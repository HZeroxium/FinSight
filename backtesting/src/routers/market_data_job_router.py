# routers/market_data_job_router.py

"""
Market Data Job Management Router

REST API endpoints for managing market data collection jobs.
Provides job control, configuration management, and monitoring capabilities.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any

from ..schemas.job_schemas import (
    JobStatusResponse,
    JobStartRequest,
    JobStopRequest,
    ManualJobRequest,
    ManualJobResponse,
    MarketDataJobConfigModel,
    JobConfigUpdateRequest,
    JobOperationResponse,
    JobStatsModel,
    DataCollectionJobRequest,
    DataCollectionJobResponse,
    HealthCheckResponse,
)
from ..services.market_data_job_service import MarketDataJobManagementService
from ..utils.dependencies import get_market_data_job_service, require_admin_access
from common.logger import LoggerFactory, LoggerType, LogLevel

# Initialize router
router = APIRouter(
    prefix="/market-data-jobs",
    tags=["market-data-job-management"],
    responses={
        401: {"description": "Unauthorized - Missing or invalid API key"},
        403: {"description": "Forbidden - Invalid API key"},
        500: {"description": "Internal server error"},
    },
)

# Initialize logger
logger = LoggerFactory.get_logger(
    name="md-job-router",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/market_data_job_router.log",
)


@router.get("/status", response_model=JobStatusResponse)
async def get_job_status(
    job_service: MarketDataJobManagementService = Depends(get_market_data_job_service),
    _: bool = Depends(require_admin_access),
) -> JobStatusResponse:
    """
    Get current market data job service status and statistics.

    **Requires admin authentication via API key.**

    Returns detailed information about the job service including:
    - Current running status
    - Process information
    - Scheduler status
    - Next scheduled run time
    - Job execution statistics

    Returns:
        JobStatusResponse: Current job status and statistics

    Raises:
        HTTPException: 401 if no API key provided, 403 if invalid API key
    """
    try:
        logger.info("Getting market data job status")
        result = await job_service.get_status()

        if result.status:
            logger.info(f"Job status retrieved successfully: {result.status}")
        else:
            logger.warning("Job status retrieval returned no status")

        return result

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get job status: {str(e)}"
        )


@router.post("/start", response_model=JobOperationResponse)
async def start_job(
    request: JobStartRequest = JobStartRequest(),
    job_service: MarketDataJobManagementService = Depends(get_market_data_job_service),
    _: bool = Depends(require_admin_access),
) -> JobOperationResponse:
    """
    Start the market data job service.

    **Requires admin authentication via API key.**

    Starts the background job service with optional configuration.
    If the service is already running, use force_restart=true to restart it.

    Args:
        request: Job start request with optional configuration

    Returns:
        JobOperationResponse: Operation result with success status

    Raises:
        HTTPException: 401 if no API key provided, 403 if invalid API key, 500 on service errors
    """
    try:
        logger.info(
            f"Starting market data job service (force_restart={request.force_restart})"
        )
        result = await job_service.start_job(request)

        if result.success:
            logger.info("Market data job service started successfully")
        else:
            logger.warning(f"Market data job service start failed: {result.message}")

        return result

    except Exception as e:
        logger.error(f"Failed to start job service: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start job service: {str(e)}"
        )


@router.post("/stop", response_model=JobOperationResponse)
async def stop_job(
    request: JobStopRequest = JobStopRequest(),
    job_service: MarketDataJobManagementService = Depends(get_market_data_job_service),
    _: bool = Depends(require_admin_access),
) -> JobOperationResponse:
    """
    Stop the market data job service.

    **Requires admin authentication via API key.**

    Gracefully stops the background job service.
    Use graceful=false for immediate termination.

    Args:
        request: Job stop request with shutdown options

    Returns:
        JobOperationResponse: Operation result with success status

    Raises:
        HTTPException: 401 if no API key provided, 403 if invalid API key, 500 on service errors
    """
    try:
        logger.info(f"Stopping market data job service (graceful={request.graceful})")
        result = await job_service.stop_job(request)

        if result.success:
            logger.info("Market data job service stopped successfully")
        else:
            logger.warning(f"Market data job service stop failed: {result.message}")

        return result

    except Exception as e:
        logger.error(f"Failed to stop job service: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to stop job service: {str(e)}"
        )


@router.post("/run", response_model=ManualJobResponse)
async def run_manual_job(
    request: ManualJobRequest = ManualJobRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    job_service: MarketDataJobManagementService = Depends(get_market_data_job_service),
    _: bool = Depends(require_admin_access),
) -> ManualJobResponse:
    """
    Run a manual market data collection job.

    **Requires admin authentication via API key.**

    Executes a one-time data collection job with specified parameters.
    Can run asynchronously in the background or synchronously.

    Args:
        request: Manual job request with collection parameters
        background_tasks: FastAPI background tasks for async execution

    Returns:
        ManualJobResponse: Job execution result and details

    Raises:
        HTTPException: 401 if no API key provided, 403 if invalid API key, 500 on service errors
    """
    try:
        symbols_str = (
            f"{len(request.symbols or [])} symbols"
            if request.symbols
            else "default symbols"
        )
        timeframes_str = (
            f"{len(request.timeframes or [])} timeframes"
            if request.timeframes
            else "default timeframes"
        )
        logger.info(f"Running manual job with {symbols_str} and {timeframes_str}")

        if request.async_execution:
            # Execute in background
            background_tasks.add_task(job_service.run_manual_job, request)
            # Return immediate response for async execution
            from datetime import datetime
            import uuid

            return ManualJobResponse(
                status="started",
                job_id=str(uuid.uuid4()),
                symbols=request.symbols or [],
                timeframes=request.timeframes or [],
                exchange=request.exchange or "binance",
                max_lookback_days=request.max_lookback_days or 30,
                start_time=datetime.now(),
                async_execution=True,
                results={"message": "Job started asynchronously"},
            )
        else:
            # Execute synchronously
            result = await job_service.run_manual_job(request)

            if result.status in ["completed", "success"]:
                logger.info(
                    f"Manual job completed: {result.records_collected} records collected"
                )
            else:
                logger.warning(f"Manual job failed or had issues: {result.status}")

            return result

    except Exception as e:
        logger.error(f"Failed to run manual job: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to run manual job: {str(e)}"
        )


@router.get("/config", response_model=MarketDataJobConfigModel)
async def get_job_config(
    job_service: MarketDataJobManagementService = Depends(get_market_data_job_service),
    _: bool = Depends(require_admin_access),
) -> MarketDataJobConfigModel:
    """
    Get current market data job configuration.

    **Requires admin authentication via API key.**

    Returns the current job configuration including scheduling,
    collection parameters, and notification settings.

    Returns:
        MarketDataJobConfigModel: Current job configuration

    Raises:
        HTTPException: 401 if no API key provided, 403 if invalid API key, 500 on service errors
    """
    try:
        logger.info("Getting market data job configuration")
        config = await job_service.get_config()

        logger.info(
            f"Job config retrieved: exchange={config.exchange}, schedule={config.cron_schedule}"
        )
        return config

    except Exception as e:
        logger.error(f"Failed to get job config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get job configuration: {str(e)}"
        )


@router.put("/config", response_model=JobOperationResponse)
async def update_job_config(
    request: JobConfigUpdateRequest,
    job_service: MarketDataJobManagementService = Depends(get_market_data_job_service),
    _: bool = Depends(require_admin_access),
) -> JobOperationResponse:
    """
    Update market data job configuration.

    **Requires admin authentication via API key.**

    Updates the job configuration with the provided parameters.
    Only specified fields will be updated, others remain unchanged.

    Args:
        request: Configuration update request with new values

    Returns:
        JobOperationResponse: Operation result with success status

    Raises:
        HTTPException: 401 if no API key provided, 403 if invalid API key, 400 on validation errors, 500 on service errors
    """
    try:
        logger.info("Updating market data job configuration")
        result = await job_service.update_config(request)

        if result.success:
            logger.info("Job configuration updated successfully")
        else:
            logger.warning(f"Job config update failed: {result.message}")

        return result

    except ValueError as e:
        logger.error(f"Invalid configuration values: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to update job config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update job configuration: {str(e)}"
        )


@router.get("/stats", response_model=JobStatsModel)
async def get_job_stats(
    job_service: MarketDataJobManagementService = Depends(get_market_data_job_service),
    _: bool = Depends(require_admin_access),
) -> JobStatsModel:
    """
    Get market data job execution statistics.

    **Requires admin authentication via API key.**

    Returns statistics about job execution including success/failure counts,
    last run times, and performance metrics.

    Returns:
        JobStatsModel: Job execution statistics

    Raises:
        HTTPException: 401 if no API key provided, 403 if invalid API key, 500 on service errors
    """
    try:
        logger.info("Getting market data job statistics")
        stats = await job_service.get_stats()

        logger.info(
            f"Job stats retrieved: {stats.total_jobs} total jobs, {stats.successful_jobs} successful"
        )
        return stats

    except Exception as e:
        logger.error(f"Failed to get job stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get job statistics: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def job_health_check(
    job_service: MarketDataJobManagementService = Depends(get_market_data_job_service),
    _: bool = Depends(require_admin_access),
) -> HealthCheckResponse:
    """
    Perform comprehensive health check of the job service.

    **Requires admin authentication via API key.**

    Checks the health of all job service components including
    the job service itself, scheduler, configuration files, and dependencies.

    Returns:
        HealthCheckResponse: Detailed health information

    Raises:
        HTTPException: 401 if no API key provided, 403 if invalid API key, 500 on service errors
    """
    try:
        logger.info("Performing market data job health check")
        health = await job_service.health_check()

        logger.info(f"Health check completed: overall status = {health.status}")
        return health

    except Exception as e:
        logger.error(f"Failed to perform health check: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/collect", response_model=DataCollectionJobResponse)
async def run_data_collection_job(
    request: DataCollectionJobRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    job_service: MarketDataJobManagementService = Depends(get_market_data_job_service),
    _: bool = Depends(require_admin_access),
) -> DataCollectionJobResponse:
    """
    Run a specific data collection job.

    **Requires admin authentication via API key.**

    Executes a data collection job for specified symbols and timeframes
    with custom parameters.

    Args:
        request: Data collection job request
        background_tasks: FastAPI background tasks for async execution

    Returns:
        DataCollectionJobResponse: Job execution result

    Raises:
        HTTPException: 401 if no API key provided, 403 if invalid API key, 400 on validation errors, 500 on service errors
    """
    try:
        logger.info(
            f"Running data collection job for {len(request.symbols)} symbols and {len(request.timeframes)} timeframes"
        )

        # Execute the collection job
        result = await job_service.run_data_collection_job(request)

        if result.status in ["completed", "success"]:
            logger.info(
                f"Data collection job completed: {result.total_records_collected} records collected"
            )
        else:
            logger.warning(f"Data collection job failed or had issues: {result.status}")

        return result

    except ValueError as e:
        logger.error(f"Invalid collection job parameters: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid job parameters: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to run data collection job: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to run data collection job: {str(e)}"
        )


@router.get("/")
async def job_service_info(
    _: bool = Depends(require_admin_access),
) -> Dict[str, Any]:
    """
    Get market data job service information and available endpoints.

    **Requires admin authentication via API key.**

    Returns information about the job management service and
    descriptions of all available endpoints.

    Returns:
        dict: Service information and endpoint descriptions

    Raises:
        HTTPException: 401 if no API key provided, 403 if invalid API key
    """
    try:
        logger.info("Getting market data job service information")

        return {
            "service": "market-data-job-management",
            "version": "1.0.0",
            "description": "Market data collection job management service",
            "endpoints": {
                "GET /market-data-jobs/": "Get service information",
                "GET /market-data-jobs/status": "Get job service status",
                "POST /market-data-jobs/start": "Start job service",
                "POST /market-data-jobs/stop": "Stop job service",
                "POST /market-data-jobs/run": "Run manual job",
                "POST /market-data-jobs/collect": "Run data collection job",
                "GET /market-data-jobs/config": "Get job configuration",
                "PUT /market-data-jobs/config": "Update job configuration",
                "GET /market-data-jobs/stats": "Get job statistics",
                "GET /market-data-jobs/health": "Perform health check",
            },
            "authentication": "Required: Bearer token with API_KEY",
            "features": [
                "Cron job scheduling for automated data collection",
                "Manual job execution with custom parameters",
                "Configuration management via REST API",
                "Comprehensive monitoring and statistics",
                "Health checking and diagnostics",
                "Multi-exchange support",
                "Multiple repository backends",
                "Error handling and retry logic",
                "Notification support",
            ],
            "supported_exchanges": ["binance"],
            "supported_repositories": ["mongodb", "csv", "parquet", "influxdb"],
            "default_timeframes": ["1h", "4h", "1d"],
            "max_concurrent_symbols": 20,
        }

    except Exception as e:
        logger.error(f"Failed to get service info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get service information: {str(e)}"
        )
