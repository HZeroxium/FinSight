# utils/dependencies.py

"""
Dependency Injection Container for Backtesting Module

This module sets up dependency injection using the dependency_injector library
to provide clean separation of concerns and enable easy testing and configuration.
"""

from dependency_injector import containers, providers
from typing import Optional, Dict, Any

from ..core.config import settings
from ..interfaces.market_data_repository import MarketDataRepository
from ..interfaces.market_data_collector import MarketDataCollector
from ..services.market_data_service import MarketDataService
from ..services.market_data_collector_service import MarketDataCollectorService
from ..adapters.csv_market_data_repository import CSVMarketDataRepository
from ..adapters.mongodb_market_data_repository import MongoDBMarketDataRepository
from ..adapters.binance_market_data_collector import BinanceMarketDataCollector
from ..converters.ohlcv_converter import OHLCVConverter
from ..converters.timeframe_converter import TimeFrameConverter
from .timeframe_utils import TimeFrameUtils
from .datetime_utils import DateTimeUtils
from common.logger import LoggerFactory
from ..schemas.enums import RepositoryType


def _create_repository(
    repository_type: str,
    csv_repo: CSVMarketDataRepository,
    mongodb_repo: MongoDBMarketDataRepository,
) -> MarketDataRepository:
    """Factory function to create repository based on type"""

    if repository_type.lower() == RepositoryType.CSV.value:
        return csv_repo
    elif repository_type.lower() == RepositoryType.MONGODB.value:
        return mongodb_repo
    else:
        # Default to CSV repository
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
        collection_prefix="ohlcv",
    )

    # Repository factory
    repository = providers.Factory(
        _create_repository,
        repository_type="csv",  # Default to CSV, can be overridden
        csv_repo=csv_repository,
        mongodb_repo=mongodb_repository,
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

    # Services
    market_data_service = providers.Factory(
        MarketDataService,
        repository=repository,
    )

    market_data_collector_service = providers.Factory(
        MarketDataCollectorService,
        collector=collector,
        data_service=market_data_service,
        collection_interval_seconds=3600,  # Default 1 hour
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
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
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
