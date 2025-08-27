# routers/job_router.py

"""
REST API router for cron job management operations.
Provides endpoints to manage the news crawler background job service.
"""

from common.logger import LoggerFactory, LoggerType, LogLevel
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..schemas.job_schemas import (JobConfigModel, JobConfigUpdateRequest,
                                   JobOperationResponse, JobStartRequest,
                                   JobStatsModel, JobStatusResponse,
                                   JobStopRequest, ManualJobRequest,
                                   ManualJobResponse)
from ..services.job_management_service import JobManagementService
from ..utils.dependencies import (get_job_management_service,
                                  require_admin_access)

# Initialize router
router = APIRouter(prefix="/jobs", tags=["job-management"])

# Setup logging
logger = LoggerFactory.get_logger(
    name="job-router",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/job_router.log",
)


@router.get("/status", response_model=JobStatusResponse)
async def get_job_status(
    job_service: JobManagementService = Depends(get_job_management_service),
    _: bool = Depends(require_admin_access),
) -> JobStatusResponse:
    """
    Get current job service status and statistics.

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
        logger.info("Getting job service status")
        status = await job_service.get_status()
        logger.info(f"Job status retrieved: {status.status}")
        return status

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve job status: {str(e)}"
        )


@router.post("/start", response_model=JobOperationResponse)
async def start_job(
    request: JobStartRequest = JobStartRequest(),
    job_service: JobManagementService = Depends(get_job_management_service),
    _: bool = Depends(require_admin_access),
) -> JobOperationResponse:
    """
    Start the cron job service.

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
        logger.info(f"Starting job service (force_restart={request.force_restart})")
        result = await job_service.start_job(request)

        if result.success:
            logger.info("Job service started successfully")
        else:
            logger.warning(f"Job service start failed: {result.message}")

        return result

    except Exception as e:
        logger.error(f"Failed to start job service: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start job service: {str(e)}"
        )


@router.post("/stop", response_model=JobOperationResponse)
async def stop_job(
    request: JobStopRequest = JobStopRequest(),
    job_service: JobManagementService = Depends(get_job_management_service),
    _: bool = Depends(require_admin_access),
) -> JobOperationResponse:
    """
    Stop the cron job service.

    Stops the background job service gracefully or forcefully.

    Args:
        request: Job stop request with graceful shutdown option

    Returns:
        JobOperationResponse: Operation result with success status
    """
    try:
        logger.info(f"Stopping job service (graceful={request.graceful})")
        result = await job_service.stop_job(request)

        if result.success:
            logger.info("Job service stopped successfully")
        else:
            logger.warning(f"Job service stop failed: {result.message}")

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
    job_service: JobManagementService = Depends(get_job_management_service),
    _: bool = Depends(require_admin_access),
) -> ManualJobResponse:
    """
    Run a manual news collection job immediately.

    Executes a one-time news collection job with specified parameters.
    This is independent of the scheduled cron job and runs immediately.

    Args:
        request: Manual job request with sources and limits
        background_tasks: FastAPI background tasks for async execution

    Returns:
        ManualJobResponse: Job execution result with timing and results
    """
    try:
        logger.info(f"Running manual job with sources: {request.sources}")
        result = await job_service.run_manual_job(request)

        if result.status == "completed":
            logger.info(f"Manual job {result.job_id} completed successfully")
        else:
            logger.warning(f"Manual job {result.job_id} failed")

        return result

    except Exception as e:
        logger.error(f"Failed to run manual job: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to run manual job: {str(e)}"
        )


@router.get("/config", response_model=JobConfigModel)
async def get_job_config(
    job_service: JobManagementService = Depends(get_job_management_service),
    _: bool = Depends(require_admin_access),
) -> JobConfigModel:
    """
    Get current job configuration.

    Returns the current configuration used by the job service including:
    - News sources to crawl
    - Collection preferences and limits
    - Cron schedule
    - Notification settings

    Returns:
        JobConfigModel: Current job configuration
    """
    try:
        logger.info("Getting job configuration")
        config = await job_service.get_config()
        logger.info("Job configuration retrieved successfully")
        return config

    except Exception as e:
        logger.error(f"Failed to get job configuration: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve job configuration: {str(e)}"
        )


@router.put("/config", response_model=JobOperationResponse)
async def update_job_config(
    request: JobConfigUpdateRequest,
    job_service: JobManagementService = Depends(get_job_management_service),
    _: bool = Depends(require_admin_access),
) -> JobOperationResponse:
    """
    Update job configuration.

    Updates the job configuration with the provided values.
    Only non-null fields in the request will be updated.
    If the job is running and the schedule is changed, it will be rescheduled.

    Args:
        request: Configuration update request with optional fields

    Returns:
        JobOperationResponse: Operation result with update status
    """
    try:
        logger.info("Updating job configuration")
        result = await job_service.update_config(request)

        if result.success:
            logger.info("Job configuration updated successfully")
        else:
            logger.warning(f"Job configuration update failed: {result.message}")

        return result

    except Exception as e:
        logger.error(f"Failed to update job configuration: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update job configuration: {str(e)}"
        )


@router.get("/stats", response_model=JobStatsModel)
async def get_job_stats(
    job_service: JobManagementService = Depends(get_job_management_service),
    _: bool = Depends(require_admin_access),
) -> JobStatsModel:
    """
    Get job execution statistics.

    Returns statistics about job executions including:
    - Total number of jobs executed
    - Success and failure counts
    - Last execution timestamps
    - Error information

    Returns:
        JobStatsModel: Job execution statistics
    """
    try:
        logger.info("Getting job statistics")
        stats = await job_service.get_stats()
        logger.info("Job statistics retrieved successfully")
        return stats

    except Exception as e:
        logger.error(f"Failed to get job statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve job statistics: {str(e)}"
        )


@router.get("/health")
async def job_health_check(
    job_service: JobManagementService = Depends(get_job_management_service),
    _: bool = Depends(require_admin_access),
) -> dict:
    """
    Perform health check for job management service.

    Returns the health status of the job management service.
    This is useful for monitoring and alerting systems.

    Returns:
        dict: Health check result with status and timestamp
    """
    try:
        logger.debug("Performing job service health check")
        is_healthy = await job_service.health_check()

        return {
            "service": "job-management",
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": None,
            "component": "job-service",
        }

    except Exception as e:
        logger.error(f"Job health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "job-management",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": None,
                "component": "job-service",
            },
        )


@router.get("/")
async def job_service_info(
    _: bool = Depends(require_admin_access),
) -> dict:
    """
    Get job service information and available endpoints.

    Returns basic information about the job management service
    and its capabilities.

    Returns:
        dict: Service information and features
    """
    return {
        "service": "job-management",
        "version": "1.0.0",
        "description": "News crawler cron job management service",
        "features": {
            "start_stop_control": True,
            "manual_execution": True,
            "configuration_management": True,
            "statistics_monitoring": True,
            "health_checking": True,
        },
        "endpoints": {
            "status": "GET /jobs/status - Get current job status",
            "start": "POST /jobs/start - Start the job service",
            "stop": "POST /jobs/stop - Stop the job service",
            "run": "POST /jobs/run - Run manual job",
            "config": "GET /jobs/config - Get configuration",
            "update_config": "PUT /jobs/config - Update configuration",
            "stats": "GET /jobs/stats - Get execution statistics",
            "health": "GET /jobs/health - Health check",
        },
        "supported_sources": ["coindesk", "cointelegraph"],
        "cron_schedule_format": "minute hour day month day_of_week (5 fields)",
    }
