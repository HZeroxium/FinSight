# utils/dependencies.py

"""
Dependency Injection Container for Backtesting Module

This module sets up dependency injection using the dependency_injector library
to provide clean separation of concerns and enable easy testing and configuration.
"""

from common.logger import LoggerFactory
from dependency_injector import containers, providers
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..adapters.binance_market_data_collector import BinanceMarketDataCollector
from ..adapters.csv_market_data_repository import CSVMarketDataRepository
from ..adapters.mongodb_market_data_repository import \
    MongoDBMarketDataRepository
from ..adapters.parquet_market_data_repository import \
    ParquetMarketDataRepository
from ..converters.ohlcv_converter import OHLCVConverter
from ..converters.timeframe_converter import TimeFrameConverter
from ..core.config import settings
from ..interfaces.market_data_collector import MarketDataCollector
from ..interfaces.market_data_repository import MarketDataRepository
from ..misc.timeframe_load_convert_save import CrossRepositoryTimeFramePipeline
from ..schemas.enums import RepositoryType
from ..services.admin_service import AdminService
from ..services.eureka_client_service import EurekaClientService
from ..services.market_data_collector_service import MarketDataCollectorService
from ..services.market_data_job_management_service import MarketDataJobManagementService
from ..services.market_data_service import MarketDataService
from ..services.market_data_storage_service import MarketDataStorageService
from ..utils.storage_client import StorageClient
from .datetime_utils import DateTimeUtils
from .timeframe_utils import TimeFrameUtils


def _create_repository(
    repository_type: str,
    csv_repo: CSVMarketDataRepository,
    mongodb_repo: MongoDBMarketDataRepository,
    parquet_repo: ParquetMarketDataRepository,
) -> MarketDataRepository:
    """Factory function to create repository based on type"""

    logger = LoggerFactory.get_logger("repository_factory_dependency")
    logger.info(f"Creating repository with type: {repository_type}")

    if repository_type.lower() == RepositoryType.CSV.value:
        logger.info("Selected CSV repository")
        return csv_repo
    elif repository_type.lower() == RepositoryType.MONGODB.value:
        logger.info("Selected MongoDB repository")
        return mongodb_repo
    elif repository_type.lower() == RepositoryType.PARQUET.value:
        logger.info("Selected Parquet repository")
        return parquet_repo
    else:
        # Default to CSV repository
        logger.warning(
            f"Unknown repository type '{repository_type}', defaulting to CSV"
        )
        return csv_repo


def _create_collector(
    exchange: str,
    binance_collector: BinanceMarketDataCollector,
) -> MarketDataCollector:
    """Factory function to create collector based on exchange"""

    if exchange.lower() == "binance":
        return binance_collector
    else:
        # Default to Binance collector
        return binance_collector


class Container(containers.DeclarativeContainer):
    """Main dependency injection container"""

    # Configuration - using the centralized settings
    config = providers.Object(settings)

    # Core utilities
    logger_factory = providers.Singleton(LoggerFactory)
    datetime_utils = providers.Singleton(DateTimeUtils)
    timeframe_utils = providers.Singleton(TimeFrameUtils)

    # Converters - TimeFrameConverter now gets TimeFrameUtils injected
    ohlcv_converter = providers.Singleton(OHLCVConverter)
    timeframe_converter = providers.Singleton(
        TimeFrameConverter, timeframe_utils=timeframe_utils
    )

    # Repository providers (configured from settings)
    csv_repository = providers.Singleton(
        CSVMarketDataRepository,
        base_directory=settings.storage_base_directory,
    )

    mongodb_repository = providers.Singleton(
        MongoDBMarketDataRepository,
        connection_string=settings.mongodb_url,
        database_name=settings.mongodb_database,
        ohlcv_collection="ohlcv",
    )

    parquet_repository = providers.Singleton(
        ParquetMarketDataRepository,
        base_path=settings.storage_base_directory,
        use_object_storage=False,  # Can be configured later
    )

    # Storage client for object storage - using configuration from settings
    storage_client = providers.Singleton(
        StorageClient,
        **settings.get_storage_config(),
    )

    # Repository factory
    repository = providers.Factory(
        _create_repository,
        repository_type=settings.repository_type,  # Use general repository type
        csv_repo=csv_repository,
        mongodb_repo=mongodb_repository,
        parquet_repo=parquet_repository,
    )

    # Market data collectors
    binance_collector = providers.Singleton(
        BinanceMarketDataCollector,
        api_key=settings.binance_api_key,
        api_secret=settings.binance_secret_key,
        testnet=False,
    )

    # Collector factory
    collector = providers.Factory(
        _create_collector,
        exchange="binance",
        binance_collector=binance_collector,
    )

    # Services - Using Singleton to prevent repeated initialization
    market_data_service = providers.Singleton(
        MarketDataService,
        repository=repository,
    )

    market_data_collector_service = providers.Singleton(
        MarketDataCollectorService,
        collector=collector,
        data_service=market_data_service,
        collection_interval_seconds=3600,  # Default 1 hour
    )

    # Storage service for object storage operations
    market_data_storage_service = providers.Singleton(
        MarketDataStorageService,
        storage_client=storage_client,
        csv_repository=csv_repository,
        parquet_repository=parquet_repository,
    )

    # Cross-repository timeframe conversion pipeline
    # Note: This is a default configuration that will be overridden by the router
    # based on the actual source_format and target_format parameters
    cross_repository_pipeline = providers.Singleton(
        CrossRepositoryTimeFramePipeline,
        source_repository=csv_repository,  # Default source, will be overridden
        target_repository=csv_repository,  # Default target, will be overridden
    )

    # Job management service for market data collection jobs
    market_data_job_service = providers.Singleton(
        MarketDataJobManagementService,
        config_file="market_data_job_config.json",
        pid_file="market_data_job.pid",
        log_file="logs/market_data_job.log",
    )

    # Eureka Client Service
    eureka_client_service = providers.Singleton(
        EurekaClientService,
    )

    admin_service = providers.Singleton(
        AdminService,
        market_data_service=market_data_service,
        collector_service=market_data_collector_service,
        repository=repository,
        storage_service=market_data_storage_service,
        market_data_job_service=market_data_job_service,
        cross_repository_pipeline=cross_repository_pipeline,
    )


class DependencyManager:
    """
    Dependency Manager for easy access to container services.

    Provides a high-level interface for accessing dependencies
    and managing container configuration.
    """

    def __init__(self):
        self.container = Container()
        # Initialize wire
        self.container.wire(modules=["__main__"])

    def get_market_data_service(self) -> MarketDataService:
        """Get configured market data service"""
        return self.container.market_data_service()

    def get_market_data_collector_service(self) -> MarketDataCollectorService:
        """Get configured market data collector service"""
        return self.container.market_data_collector_service()

    def get_timeframe_converter(self) -> TimeFrameConverter:
        """Get timeframe converter"""
        return self.container.timeframe_converter()

    def get_timeframe_utils(self) -> TimeFrameUtils:
        """Get timeframe utilities"""
        return self.container.timeframe_utils()

    def get_repository(self) -> MarketDataRepository:
        """Get configured repository"""
        return self.container.repository()

    def get_collector(self) -> MarketDataCollector:
        """Get configured collector"""
        return self.container.collector()

    def get_storage_service(self) -> MarketDataStorageService:
        """Get configured storage service"""
        return self.container.market_data_storage_service()

    def get_job_management_service(self) -> MarketDataJobManagementService:
        """Get configured job management service"""
        return self.container.market_data_job_service()

    def get_storage_client(self) -> StorageClient:
        """Get configured storage client"""
        return self.container.storage_client()

    def get_parquet_repository(self) -> ParquetMarketDataRepository:
        """Get configured parquet repository"""
        return self.container.parquet_repository()

    def get_cross_repository_pipeline(self) -> CrossRepositoryTimeFramePipeline:
        """Get configured cross-repository timeframe conversion pipeline"""
        return self.container.cross_repository_pipeline()

    def get_eureka_client_service(self) -> EurekaClientService:
        """Get Eureka client service"""
        return self.container.eureka_client_service()

    def configure_csv_storage(self, base_directory: str = None) -> None:
        """Configure CSV storage"""
        if base_directory:
            # This would require updating the settings instance,
            # but for now we'll use the default from settings
            pass

    def configure_mongodb_storage(
        self,
        connection_string: str = None,
        database_name: str = None,
        collection_prefix: str = "ohlcv",
    ) -> None:
        """Configure MongoDB storage"""
        # This would require updating the settings instance,
        # but for now we'll use the default from settings
        pass

    def configure_binance_exchange(
        self,
    ) -> None:
        """Configure Binance exchange"""
        # This would require updating the settings instance,
        # but for now we'll use the default from settings
        pass

    def configure_collection_settings(self, interval_seconds: int = 3600) -> None:
        """Configure data collection settings"""
        # This would require updating the settings instance,
        # but for now we'll use the default from settings
        pass

    def reset_configuration(self) -> None:
        """Reset container configuration to defaults"""
        self.container.reset_last_provided()

    def shutdown(self) -> None:
        """Shutdown container and clean up resources"""
        self.container.shutdown_resources()


# Global dependency manager instance
dependency_manager = DependencyManager()


def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager instance"""
    return dependency_manager


def get_market_data_service() -> MarketDataService:
    """Get configured market data service (convenience function)"""
    return dependency_manager.get_market_data_service()


def get_market_data_collector_service() -> MarketDataCollectorService:
    """Get configured market data collector service (convenience function)"""
    return dependency_manager.get_market_data_collector_service()


def get_timeframe_converter() -> TimeFrameConverter:
    """Get timeframe converter (convenience function)"""
    return dependency_manager.get_timeframe_converter()


def get_timeframe_utils() -> TimeFrameUtils:
    """Get timeframe utilities (convenience function)"""
    return dependency_manager.get_timeframe_utils()


def get_storage_service() -> MarketDataStorageService:
    """Get configured storage service (convenience function)"""
    return dependency_manager.get_storage_service()


def get_market_data_job_service() -> MarketDataJobManagementService:
    """Get configured job management service (convenience function)"""
    return dependency_manager.get_job_management_service()


def get_storage_client() -> StorageClient:
    """Get configured storage client (convenience function)"""
    return dependency_manager.get_storage_client()


def get_parquet_repository() -> ParquetMarketDataRepository:
    """Get configured parquet repository (convenience function)"""
    return dependency_manager.get_parquet_repository()


def get_cross_repository_pipeline() -> CrossRepositoryTimeFramePipeline:
    """Get configured cross-repository timeframe conversion pipeline (convenience function)"""
    return dependency_manager.get_cross_repository_pipeline()


def get_eureka_client_service() -> EurekaClientService:
    """Get Eureka client service instance"""
    try:
        return dependency_manager.container.eureka_client_service()
    except Exception as e:
        print(f"Failed to get Eureka client service: {e}")
        raise


# Security dependencies
security = HTTPBearer()


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """
    Verify API key for admin endpoints.

    Args:
        credentials: HTTP Bearer credentials from request header

    Returns:
        bool: True if API key is valid

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not settings.api_key:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: API_KEY not configured",
        )

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Please provide Authorization: Bearer <api_key>",
        )

    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key. Check API_KEY environment variable.",
        )

    return True


def require_admin_access(api_key_verified: bool = Depends(verify_api_key)) -> bool:
    """
    Dependency that requires admin access via API key.

    Args:
        api_key_verified: Result from API key verification

    Returns:
        bool: True if admin access is granted
    """
    return True
