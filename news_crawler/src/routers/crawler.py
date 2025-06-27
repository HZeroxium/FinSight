# routers/crawler.py

"""
REST API routes for crawler operations.
"""

from datetime import datetime
from typing import Dict, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks

from ..schemas.crawler_schemas import (
    CrawlerConfigSchema,
    CrawlJobSchema,
    CrawlResultSchema,
    CrawlStatsSchema,
)
from ..schemas.common_schemas import HealthCheckSchema
from ..services.crawler_service import CrawlerService
from ..models.crawler import CrawlerConfig
from ..utils.dependencies import get_crawler_service
from ..common.logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="crawler-router", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)

router = APIRouter(prefix="/api/v1/crawler", tags=["crawler"])


@router.post("/crawl", response_model=CrawlResultSchema)
async def start_crawl_job(
    job: CrawlJobSchema,
    background_tasks: BackgroundTasks,
    crawler_service: CrawlerService = Depends(get_crawler_service),
) -> CrawlResultSchema:
    """
    Start a crawl job for a specific source.

    Args:
        job: Crawl job configuration
        background_tasks: FastAPI background tasks
        crawler_service: Injected crawler service

    Returns:
        CrawlResultSchema: Job status and initial results
    """
    try:
        logger.info(f"Starting crawl job for source: {job.source_name}")

        # Start crawl in background and get result
        async def background_crawl():
            return await crawler_service.crawl_source(job.source_name)

        background_tasks.add_task(background_crawl)

        return CrawlResultSchema(
            job_id=f"crawl_{job.source_name}_{int(datetime.utcnow().timestamp())}",
            source_name=job.source_name,
            articles_found=0,
            articles_crawled=0,
            articles_saved=0,
            duration=0.0,
            status="started",
            started_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Failed to start crawl job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crawl/all", response_model=Dict[str, CrawlResultSchema])
async def start_crawl_all_sources(
    background_tasks: BackgroundTasks,
    crawler_service: CrawlerService = Depends(get_crawler_service),
) -> Dict[str, CrawlResultSchema]:
    """
    Start crawl jobs for all configured sources.

    Args:
        background_tasks: FastAPI background tasks
        crawler_service: Injected crawler service

    Returns:
        Dict[str, CrawlResultSchema]: Results keyed by source name
    """
    try:
        logger.info("Starting crawl for all sources")

        # Start crawl in background
        async def background_crawl_all():
            return await crawler_service.crawl_all_sources()

        background_tasks.add_task(background_crawl_all)

        # Return initial status for all sources
        stats = crawler_service.get_crawler_stats()
        results = {}

        for source_name in stats["enabled_crawler_names"]:
            results[source_name] = CrawlResultSchema(
                job_id=f"crawl_all_{source_name}_{int(datetime.utcnow().timestamp())}",
                source_name=source_name,
                articles_found=0,
                articles_crawled=0,
                articles_saved=0,
                duration=0.0,
                status="started",
                started_at=datetime.utcnow(),
            )

        return results

    except Exception as e:
        logger.error(f"Failed to start crawl all job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=CrawlStatsSchema)
async def get_crawler_stats(
    crawler_service: CrawlerService = Depends(get_crawler_service),
) -> CrawlStatsSchema:
    """
    Get crawler statistics.

    Returns:
        CrawlStatsSchema: Current crawler statistics
    """
    try:
        stats = crawler_service.get_crawler_stats()
        detailed_stats = crawler_service.get_detailed_crawler_stats()

        # Aggregate detailed stats
        total_crawled = sum(s.total_articles_crawled for s in detailed_stats)
        total_saved = sum(s.total_articles_saved for s in detailed_stats)
        avg_success_rate = (
            sum(s.success_rate for s in detailed_stats) / len(detailed_stats)
            if detailed_stats
            else 0.0
        )
        avg_crawl_time = (
            sum(s.average_crawl_time for s in detailed_stats) / len(detailed_stats)
            if detailed_stats
            else 0.0
        )

        return CrawlStatsSchema(
            total_crawlers=stats["total_crawlers"],
            active_crawlers=stats["enabled_crawlers"],
            total_articles_crawled=total_crawled,
            total_articles_saved=total_saved,
            success_rate=avg_success_rate,
            average_crawl_time=avg_crawl_time,
            sources=stats["enabled_crawler_names"],
        )

    except Exception as e:
        logger.error(f"Failed to get crawler stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/detailed", response_model=List[Dict])
async def get_detailed_crawler_stats(
    crawler_service: CrawlerService = Depends(get_crawler_service),
) -> List[Dict]:
    """
    Get detailed statistics for each crawler.

    Returns:
        List[Dict]: Detailed statistics per crawler
    """
    try:
        detailed_stats = crawler_service.get_detailed_crawler_stats()
        return [stat.dict() for stat in detailed_stats]

    except Exception as e:
        logger.error(f"Failed to get detailed crawler stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sources", response_model=Dict[str, str])
async def add_crawler_source(
    config: CrawlerConfigSchema,
    crawler_service: CrawlerService = Depends(get_crawler_service),
) -> Dict[str, str]:
    """
    Add a new crawler source configuration.

    Args:
        config: Crawler configuration
        crawler_service: Injected crawler service

    Returns:
        Dict[str, str]: Success message
    """
    try:
        logger.info(f"Adding new crawler source: {config.name}")

        # Convert schema to domain model
        crawler_config = CrawlerConfig(
            name=config.name,
            base_url=config.base_url,
            listing_url=config.listing_url,
            listing_selector=config.listing_selector,
            title_selector=config.title_selector,
            content_selector=config.content_selector,
            date_selector=config.date_selector,
            author_selector=config.author_selector,
            date_format=config.date_format,
            category=config.category,
            credibility_score=config.credibility_score,
            enabled=config.enabled,
        )

        crawler_service.add_crawler(crawler_config)

        return {"message": f"Crawler source '{config.name}' added successfully"}

    except Exception as e:
        logger.error(f"Failed to add crawler source: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthCheckSchema)
async def health_check(
    crawler_service: CrawlerService = Depends(get_crawler_service),
) -> HealthCheckSchema:
    """
    Health check endpoint for crawler service.

    Returns:
        HealthCheckSchema: Health status
    """
    try:
        is_healthy = await crawler_service.health_check()
        stats = crawler_service.get_crawler_stats()

        return HealthCheckSchema(
            status="healthy" if is_healthy else "unhealthy",
            service="crawler-service",
            dependencies={
                "total_crawlers": str(stats["total_crawlers"]),
                "enabled_crawlers": str(stats["enabled_crawlers"]),
                "message_broker": "healthy" if is_healthy else "unhealthy",
                "article_repository": "healthy" if is_healthy else "unhealthy",
            },
        )

    except Exception as e:
        logger.error(f"Crawler health check failed: {str(e)}")
        return HealthCheckSchema(
            status="unhealthy",
            service="crawler-service",
            dependencies={"error": str(e)},
        )
