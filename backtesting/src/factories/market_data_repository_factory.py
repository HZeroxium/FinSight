"""
Market Data Repository Factory

Factory pattern implementation for creating different types of market data repositories.
Enables easy switching between storage backends (CSV, MongoDB, InfluxDB, etc.).
"""

from typing import Dict, Any, Optional, Type

from ..interfaces.market_data_repository import MarketDataRepository
from ..interfaces.errors import RepositoryError
from common.logger import LoggerFactory
from ..schemas.enums import RepositoryType


class MarketDataRepositoryFactory:
    """Factory for creating market data repository instances"""

    def __init__(self):
        """Initialize the repository factory"""
        self.logger = LoggerFactory.get_logger(name="repository_factory")
        self._registry: Dict[RepositoryType, Type[MarketDataRepository]] = {}
        self._register_default_repositories()

    def create_repository(
        self,
        repository_type: RepositoryType,
        config: Optional[Dict[str, Any]] = None,
    ) -> MarketDataRepository:
        """
        Create a market data repository instance.

        Args:
            repository_type: Type of repository to create
            config: Configuration dictionary for the repository

        Returns:
            MarketDataRepository instance

        Raises:
            RepositoryError: If repository type is not supported or creation fails
        """
        if config is None:
            config = {}

        try:
            if repository_type not in self._registry:
                raise RepositoryError(
                    f"Repository type {repository_type} not registered"
                )

            repository_class = self._registry[repository_type]

            # Create repository with config
            repository = self._create_with_config(repository_class, config)

            self.logger.info(f"Created {repository_type.value} repository")
            return repository

        except Exception as e:
            raise RepositoryError(
                f"Failed to create {repository_type.value} repository: {str(e)}"
            )

    def create_csv_repository(
        self, base_directory: str = "data"
    ) -> MarketDataRepository:
        """
        Create a CSV repository with specific configuration.

        Args:
            base_directory: Base directory for CSV files

        Returns:
            CSV MarketDataRepository instance
        """
        config = {"base_directory": base_directory}
        return self.create_repository(RepositoryType.CSV, config)

    def create_mongodb_repository(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "finsight_market_data",
        collection_prefix: str = "ohlcv",
    ) -> MarketDataRepository:
        """
        Create a MongoDB repository with specific configuration.

        Args:
            connection_string: MongoDB connection string
            database_name: Database name
            collection_prefix: Prefix for collection names

        Returns:
            MongoDB MarketDataRepository instance
        """
        config = {
            "connection_string": connection_string,
            "database_name": database_name,
            "collection_prefix": collection_prefix,
        }
        return self.create_repository(RepositoryType.MONGODB, config)

    def create_influxdb_repository(
        self,
        url: str = "http://localhost:8086",
        token: str = "",
        org: str = "finsight",
        bucket: str = "market_data",
    ) -> MarketDataRepository:
        """
        Create an InfluxDB repository with specific configuration.

        Args:
            url: InfluxDB server URL
            token: Authentication token
            org: Organization name
            bucket: Bucket name

        Returns:
            InfluxDB MarketDataRepository instance
        """
        config = {
            "url": url,
            "token": token,
            "org": org,
            "bucket": bucket,
        }
        return self.create_repository(RepositoryType.INFLUXDB, config)

    def create_from_config(self, config: Dict[str, Any]) -> MarketDataRepository:
        """
        Create repository from configuration dictionary.

        Args:
            config: Configuration dictionary with 'type' key and repository-specific config

        Returns:
            MarketDataRepository instance

        Example:
            config = {
                "type": "mongodb",
                "connection_string": "mongodb://localhost:27017/",
                "database_name": "market_data"
            }
        """
        if "type" not in config:
            raise RepositoryError("Repository type not specified in config")

        repo_type_str = config.pop("type")
        try:
            repo_type = RepositoryType(repo_type_str)
        except ValueError:
            raise RepositoryError(f"Invalid repository type: {repo_type_str}")

        return self.create_repository(repo_type, config)

    def register_repository(
        self,
        repository_type: RepositoryType,
        repository_class: Type[MarketDataRepository],
    ) -> None:
        """
        Register a custom repository implementation.

        Args:
            repository_type: Type identifier for the repository
            repository_class: Repository class to register
        """
        self._registry[repository_type] = repository_class
        self.logger.info(f"Registered custom repository: {repository_type.value}")

    def get_supported_types(self) -> list[RepositoryType]:
        """Get list of supported repository types"""
        return list(self._registry.keys())

    def _register_default_repositories(self) -> None:
        """Register default repository implementations"""
        try:
            # CSV Repository
            from ..adapters.csv_market_data_repository import CSVMarketDataRepository

            self._registry[RepositoryType.CSV] = CSVMarketDataRepository

            # MongoDB Repository
            try:
                from ..adapters.mongodb_market_data_repository import (
                    MongoDBMarketDataRepository,
                )

                self._registry[RepositoryType.MONGODB] = MongoDBMarketDataRepository
            except ImportError:
                self.logger.warning(
                    "MongoDB repository not available (PyMongo not installed)"
                )

            # InfluxDB Repository
            try:
                from ..adapters.influx_market_data_repository import (
                    InfluxMarketDataRepository,
                )

                self._registry[RepositoryType.INFLUXDB] = InfluxMarketDataRepository
            except ImportError:
                self.logger.warning(
                    "InfluxDB repository not available (influxdb-client not installed)"
                )

            # TimescaleDB Repository would be added here when implemented
            # self._registry[RepositoryType.TIMESCALEDB] = TimescaleDBMarketDataRepository

        except Exception as e:
            self.logger.error(f"Error registering default repositories: {e}")

    def _create_with_config(
        self,
        repository_class: Type[MarketDataRepository],
        config: Dict[str, Any],
    ) -> MarketDataRepository:
        """
        Create repository instance with configuration.

        Args:
            repository_class: Repository class to instantiate
            config: Configuration parameters

        Returns:
            Repository instance
        """
        try:
            # Filter config to only include parameters the class accepts
            import inspect

            sig = inspect.signature(repository_class.__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}
            filtered_config = {k: v for k, v in config.items() if k in valid_params}

            return repository_class(**filtered_config)

        except Exception as e:
            raise RepositoryError(f"Failed to create repository instance: {str(e)}")


# Global factory instance for convenience
repository_factory = MarketDataRepositoryFactory()


def create_repository(
    repository_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> MarketDataRepository:
    """
    Convenience function to create repository using global factory.

    Args:
        repository_type: Repository type as string
        config: Optional configuration dictionary

    Returns:
        MarketDataRepository instance
    """
    try:
        repo_type = RepositoryType(repository_type)
        return repository_factory.create_repository(repo_type, config)
    except ValueError:
        raise RepositoryError(f"Invalid repository type: {repository_type}")


def create_repository_from_config(config: Dict[str, Any]) -> MarketDataRepository:
    """
    Convenience function to create repository from config using global factory.

    Args:
        config: Configuration dictionary

    Returns:
        MarketDataRepository instance
    """
    return repository_factory.create_from_config(config)


def get_market_data_service():
    """
    Dependency injection function for FastAPI.

    Creates and returns a MarketDataService instance with the default repository.
    This function is used as a FastAPI dependency to provide the service to endpoints.

    Returns:
        MarketDataService instance configured with default repository
    """
    from ..services.market_data_service import MarketDataService
    from ..core.config import settings

    # Create repository with default configuration from settings
    repository = repository_factory.create_from_config(
        {
            "type": "mongodb",  # Default to MongoDB
            "mongodb": {
                "connection_string": settings.mongodb_url,
                "database_name": settings.mongodb_database,
            },
        }
    )

    # Create and return service
    return MarketDataService(repository=repository)
