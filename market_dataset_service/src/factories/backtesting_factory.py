# factories/backtesting_factory.py

"""
Factory for creating backtesting components.
Implements Factory Pattern for consistent component creation.
"""

from typing import Dict, Any, Optional
from enum import Enum

from ..interfaces.backtesting_engine import BacktestingEngine
from ..interfaces.backtesting_repository import BacktestingRepository
from ..adapters.backtesting.backtrader_adapter import BacktraderAdapter
from ..adapters.file_backtesting_repository import FileBacktestingRepository
from ..adapters.mongodb_backtesting_repository import MongoDBBacktestingRepository
from ..services.backtesting_service import BacktestingService
from ..services.backtesting_data_service import BacktestingDataService
from ..services.market_data_service import MarketDataService
from common.logger import LoggerFactory
from ..core.config import Settings


class BacktestingEngineType(str, Enum):
    """Supported backtesting engine types."""

    BACKTRADER = "backtrader"
    # VECTORBT = "vectorbt"  # For future implementation


class BacktestingFactory:
    """
    Factory for creating backtesting components.

    Centralizes the creation of backtesting engines and services
    with proper dependency injection.
    """

    @staticmethod
    def create_backtesting_engine(
        engine_type: BacktestingEngineType, config: Optional[Dict[str, Any]] = None
    ) -> BacktestingEngine:
        """
        Create a backtesting engine instance.

        Args:
            engine_type: Type of engine to create
            config: Optional configuration parameters

        Returns:
            BacktestingEngine instance

        Raises:
            ValueError: If engine type is not supported
        """
        logger = LoggerFactory.get_logger(name="backtesting_factory")
        config = config or {}

        if engine_type == BacktestingEngineType.BACKTRADER:
            logger.info("Creating Backtrader engine")
            return BacktraderAdapter()

        # Future implementations
        # elif engine_type == BacktestingEngineType.VECTORBT:
        #     logger.info("Creating VectorBT engine")
        #     return VectorBTAdapter(config)

        else:
            available_engines = [e.value for e in BacktestingEngineType]
            raise ValueError(
                f"Unsupported engine type: {engine_type}. "
                f"Available engines: {available_engines}"
            )

    @staticmethod
    def create_backtesting_service(
        market_data_service: MarketDataService,
        engine_type: BacktestingEngineType = BacktestingEngineType.BACKTRADER,
        engine_config: Optional[Dict[str, Any]] = None,
    ) -> BacktestingService:
        """
        Create a complete backtesting service.

        Args:
            market_data_service: Market data service instance
            engine_type: Type of backtesting engine to use
            engine_config: Optional engine configuration

        Returns:
            BacktestingService instance
        """
        logger = LoggerFactory.get_logger(name="backtesting_factory")

        # Create backtesting engine
        engine = BacktestingFactory.create_backtesting_engine(
            engine_type=engine_type, config=engine_config
        )

        # Create backtesting service
        service = BacktestingService(
            market_data_service=market_data_service, backtesting_engine=engine
        )

        logger.info(f"Created backtesting service with {engine_type.value} engine")
        return service

    @staticmethod
    def create_backtesting_repository(
        repository_type: str = "file", config: Optional[Dict[str, Any]] = None
    ) -> BacktestingRepository:
        """
        Create a backtesting repository instance.

        Args:
            repository_type: Type of repository ("file" or "mongodb")
            config: Optional repository configuration

        Returns:
            BacktestingRepository instance

        Raises:
            ValueError: If repository type is not supported
        """
        logger = LoggerFactory.get_logger(name="backtesting_factory")
        config = config or {}

        if repository_type.lower() == "file":
            base_directory = config.get("base_directory", "data/backtests")
            logger.info(f"Creating file-based backtesting repository: {base_directory}")
            return FileBacktestingRepository(base_directory=base_directory)

        elif repository_type.lower() == "mongodb":
            connection_string = config.get(
                "connection_string", "mongodb://localhost:27017/"
            )
            database_name = config.get("database_name", "finsight_backtesting")
            results_collection = config.get("results_collection", "backtest_results")
            history_collection = config.get("history_collection", "backtest_history")

            logger.info(f"Creating MongoDB backtesting repository: {database_name}")
            return MongoDBBacktestingRepository(
                connection_string=connection_string,
                database_name=database_name,
                results_collection=results_collection,
                history_collection=history_collection,
            )

        else:
            available_types = ["file", "mongodb"]
            raise ValueError(
                f"Unsupported repository type: {repository_type}. "
                f"Available types: {available_types}"
            )

    @staticmethod
    def create_backtesting_data_service(
        repository_type: str = "file",
        repository_config: Optional[Dict[str, Any]] = None,
    ) -> BacktestingDataService:
        """
        Create a backtesting data service instance.

        Args:
            repository_type: Type of repository to use
            repository_config: Optional repository configuration

        Returns:
            BacktestingDataService instance
        """
        logger = LoggerFactory.get_logger(name="backtesting_factory")

        # Create repository
        repository = BacktestingFactory.create_backtesting_repository(
            repository_type=repository_type, config=repository_config
        )

        # Create data service
        service = BacktestingDataService(repository=repository)

        logger.info(
            f"Created backtesting data service with {repository_type} repository"
        )
        return service

    @staticmethod
    def get_available_engines() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available backtesting engines.

        Returns:
            Dictionary mapping engine names to their information
        """
        engines = {}

        for engine_type in BacktestingEngineType:
            try:
                engine = BacktestingFactory.create_backtesting_engine(engine_type)
                engines[engine_type.value] = engine.get_engine_info()
            except Exception as e:
                engines[engine_type.value] = {
                    "name": engine_type.value,
                    "error": f"Failed to create engine: {str(e)}",
                    "available": False,
                }

        return engines

    @staticmethod
    def validate_engine_requirements(
        engine_type: BacktestingEngineType,
    ) -> Dict[str, Any]:
        """
        Validate that requirements for an engine are met.

        Args:
            engine_type: Engine type to validate

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "engine_type": engine_type.value,
            "available": False,
            "missing_dependencies": [],
            "errors": [],
        }

        try:
            if engine_type == BacktestingEngineType.BACKTRADER:
                # Check backtrader dependencies
                try:
                    import backtrader
                    import pandas

                    validation_result["available"] = True
                    validation_result["versions"] = {
                        "backtrader": backtrader.__version__,
                        "pandas": pandas.__version__,
                    }
                except ImportError as e:
                    validation_result["missing_dependencies"].append(str(e))

            # Add validation for other engines here

        except Exception as e:
            validation_result["errors"].append(str(e))

        return validation_result


def get_backtesting_service():
    """
    Dependency injection function for FastAPI.

    Creates and returns a BacktestingService instance with default configuration.
    This function is used as a FastAPI dependency to provide the service to endpoints.

    Returns:
        BacktestingService instance configured with default components
    """
    from ..services.backtesting_service import BacktestingService

    # Get dependencies
    market_data_service = get_market_data_service()
    backtesting_engine = BacktestingFactory.create_backtesting_engine(
        BacktestingEngineType.BACKTRADER
    )
    backtesting_data_service = get_backtesting_data_service()

    # Create backtesting service
    backtesting_service = BacktestingService(
        market_data_service=market_data_service,
        backtesting_engine=backtesting_engine,
        backtesting_data_service=backtesting_data_service,
    )

    return backtesting_service


def get_market_data_service():
    """
    Create MarketDataService instance for backtesting factory.

    Returns:
        MarketDataService instance
    """
    from .market_data_repository_factory import get_market_data_service as get_service

    return get_service()


def get_backtesting_data_service() -> BacktestingDataService:
    """
    Create BacktestingDataService instance for dependency injection.

    Returns:
        BacktestingDataService instance configured with default repository
    """
    settings = Settings()

    # Determine repository type and configuration from settings
    repository_type = getattr(settings, "BACKTESTING_REPOSITORY_TYPE", "file")

    if repository_type.lower() == "file":
        repository_config = {
            "base_directory": getattr(
                settings, "BACKTESTING_DATA_DIRECTORY", "data/backtests"
            )
        }
    elif repository_type.lower() == "mongodb":
        repository_config = {
            "connection_string": getattr(
                settings, "MONGODB_URL", "mongodb://localhost:27017/"
            ),
            "database_name": getattr(
                settings, "BACKTESTING_DATABASE_NAME", "finsight_backtesting"
            ),
            "results_collection": getattr(
                settings, "BACKTESTING_RESULTS_COLLECTION", "backtest_results"
            ),
            "history_collection": getattr(
                settings, "BACKTESTING_HISTORY_COLLECTION", "backtest_history"
            ),
        }
    else:
        repository_config = {}

    return BacktestingFactory.create_backtesting_data_service(
        repository_type=repository_type, repository_config=repository_config
    )


def get_backtesting_repository() -> BacktestingRepository:
    """
    Create BacktestingRepository instance for dependency injection.

    Returns:
        BacktestingRepository instance configured with default settings
    """
    settings = Settings()

    # Determine repository type and configuration from settings
    repository_type = getattr(settings, "BACKTESTING_REPOSITORY_TYPE", "file")

    if repository_type.lower() == "file":
        repository_config = {
            "base_directory": getattr(
                settings, "BACKTESTING_DATA_DIRECTORY", "data/backtests"
            )
        }
    elif repository_type.lower() == "mongodb":
        repository_config = {
            "connection_string": getattr(
                settings, "MONGODB_URL", "mongodb://localhost:27017/"
            ),
            "database_name": getattr(
                settings, "BACKTESTING_DATABASE_NAME", "finsight_backtesting"
            ),
            "results_collection": getattr(
                settings, "BACKTESTING_RESULTS_COLLECTION", "backtest_results"
            ),
            "history_collection": getattr(
                settings, "BACKTESTING_HISTORY_COLLECTION", "backtest_history"
            ),
        }
    else:
        repository_config = {}

    return BacktestingFactory.create_backtesting_repository(
        repository_type=repository_type, config=repository_config
    )


def get_market_data_collector_service():
    """
    Create MarketDataCollectorService instance.

    Returns:
        MarketDataCollectorService instance
    """
    from ..services.market_data_collector_service import MarketDataCollectorService
    from ..adapters.binance_market_data_collector import BinanceMarketDataCollector

    # Create collector
    collector = BinanceMarketDataCollector()

    # Create service
    data_service = get_market_data_service()
    collector_service = MarketDataCollectorService(
        collector=collector,
        data_service=data_service,
    )

    return collector_service
