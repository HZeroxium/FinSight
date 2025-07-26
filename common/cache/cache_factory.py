# common/cache/cache_factory.py

from typing import Optional, Dict, Any
from enum import Enum

from .cache_interface import CacheInterface
from .in_memory_cache import InMemoryCache
from .redis_cache import RedisCache
from .file_cache import FileCache
from common.logger import LoggerFactory, LoggerType, LogLevel

# Create logger for cache factory
logger = LoggerFactory.get_logger(
    name="cache-factory", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class CacheType(Enum):
    """Available cache types"""

    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"


class CacheFactory:
    """Factory for creating cache instances"""

    _instances: Dict[str, CacheInterface] = {}

    @classmethod
    def get_cache(
        cls,
        name: str = "default",
        cache_type: CacheType = CacheType.MEMORY,
        **kwargs: Any,
    ) -> CacheInterface:
        """
        Get or create a cache instance (cached)

        Args:
            name: Cache instance name
            cache_type: Type of cache to create
            **kwargs: Additional arguments for cache construction

        Returns:
            Cache instance
        """
        cache_key = f"{name}_{cache_type.value}"
        logger.debug(f"Requesting cache instance: {cache_key}")

        if cache_key not in cls._instances:
            logger.debug(f"Creating new cache instance: {cache_key}")
            try:
                cls._instances[cache_key] = cls.create_cache(
                    name=name, cache_type=cache_type, **kwargs
                )
                logger.info(f"Created cache instance: {cache_key}")
            except Exception as e:
                logger.error(f"Failed to create cache instance {cache_key}: {e}")
                # Fallback to memory cache if requested cache type fails
                if cache_type != CacheType.MEMORY:
                    logger.warning(f"Falling back to memory cache for {cache_key}")
                    fallback_key = f"{name}_memory_fallback"
                    try:
                        cls._instances[cache_key] = cls.create_cache(
                            name=fallback_key, cache_type=CacheType.MEMORY
                        )
                        logger.info(f"Created fallback memory cache for {cache_key}")
                    except Exception as fallback_error:
                        logger.error(
                            f"Even fallback cache creation failed: {fallback_error}"
                        )
                        raise
                else:
                    raise
        else:
            logger.debug(f"Returning existing cache instance: {cache_key}")

        return cls._instances[cache_key]

    @classmethod
    def create_cache(
        cls,
        name: str = "default",
        cache_type: CacheType = CacheType.MEMORY,
        **kwargs: Any,
    ) -> CacheInterface:
        """
        Create a new cache instance (not cached)

        Args:
            name: Cache instance name
            cache_type: Type of cache to create
            **kwargs: Additional arguments for cache construction

        Returns:
            New cache instance
        """
        logger.debug(f"Creating {cache_type.value} cache: {name}")

        try:
            if cache_type == CacheType.MEMORY:
                cache_instance = cls._create_memory_cache(**kwargs)
            elif cache_type == CacheType.REDIS:
                cache_instance = cls._create_redis_cache(**kwargs)
            elif cache_type == CacheType.FILE:
                cache_instance = cls._create_file_cache(**kwargs)
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")

            logger.info(f"Successfully created {cache_type.value} cache: {name}")
            return cache_instance

        except Exception as e:
            logger.error(f"Failed to create {cache_type.value} cache {name}: {e}")
            raise

    @classmethod
    def _create_memory_cache(cls, **kwargs) -> InMemoryCache:
        """Create in-memory cache with default parameters"""
        defaults = {
            "max_size": None,
            "default_ttl": None,
            "cleanup_interval": 60,
            "enable_lru": True,
        }
        defaults.update(kwargs)
        logger.debug(f"Creating memory cache with params: {defaults}")
        return InMemoryCache(**defaults)

    @classmethod
    def _create_redis_cache(cls, **kwargs) -> RedisCache:
        """Create Redis cache with default parameters"""
        # Filter allowed parameters for Redis
        allowed = {
            "host",
            "port",
            "db",
            "password",
            "serialization",
            "key_prefix",
            "socket_timeout",
            "socket_connect_timeout",
            "socket_keepalive",
            "socket_keepalive_options",
            "connection_pool",
            "unix_socket_path",
            "encoding",
            "encoding_errors",
            "decode_responses",
            "retry_on_timeout",
            "ssl",
            "ssl_keyfile",
            "ssl_certfile",
            "ssl_cert_reqs",
            "ssl_ca_certs",
            "ssl_check_hostname",
            "max_connections",
        }
        params = {k: v for k, v in kwargs.items() if k in allowed}

        defaults = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": None,
            "serialization": "json",
            "key_prefix": "",
        }
        defaults.update(params)

        logger.debug(f"Creating Redis cache with params: {defaults}")
        return RedisCache(**defaults)

    @classmethod
    def _create_file_cache(cls, **kwargs) -> FileCache:
        """Create file cache with default parameters"""
        defaults = {
            "cache_dir": ".cache",
            "serialization": "json",
            "file_extension": None,
            "max_files": None,
            "cleanup_interval": 300,
            "create_subdirs": True,
            "safe_filenames": True,
        }
        defaults.update(kwargs)
        logger.debug(f"Creating file cache with params: {defaults}")
        return FileCache(**defaults)

    @classmethod
    def create_memory_cache(
        cls,
        max_size: Optional[int] = None,
        default_ttl: Optional[int] = None,
        cleanup_interval: int = 60,
        enable_lru: bool = True,
    ) -> InMemoryCache:
        """
        Create in-memory cache with specific parameters

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            cleanup_interval: Cleanup interval in seconds
            enable_lru: Enable LRU eviction

        Returns:
            InMemoryCache instance
        """
        return InMemoryCache(
            max_size=max_size,
            default_ttl=default_ttl,
            cleanup_interval=cleanup_interval,
            enable_lru=enable_lru,
        )

    @classmethod
    def create_redis_cache(
        cls,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        serialization: str = "json",
        key_prefix: str = "",
        **redis_kwargs,
    ) -> RedisCache:
        """
        Create Redis cache with specific parameters

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password
            serialization: Serialization method ("json" or "pickle")
            key_prefix: Prefix for cache keys
            **redis_kwargs: Additional Redis connection parameters

        Returns:
            RedisCache instance
        """
        return RedisCache(
            host=host,
            port=port,
            db=db,
            password=password,
            serialization=serialization,
            key_prefix=key_prefix,
            **redis_kwargs,
        )

    @classmethod
    def create_file_cache(
        cls,
        cache_dir: str = ".cache",
        serialization: str = "json",
        file_extension: Optional[str] = None,
        max_files: Optional[int] = None,
        cleanup_interval: int = 300,
        create_subdirs: bool = True,
        safe_filenames: bool = True,
    ) -> FileCache:
        """
        Create file cache with specific parameters

        Args:
            cache_dir: Directory to store cache files
            serialization: Serialization method ("json" or "pickle")
            file_extension: File extension
            max_files: Maximum number of cache files
            cleanup_interval: Cleanup interval in seconds
            create_subdirs: Create subdirectories
            safe_filenames: Use safe filenames

        Returns:
            FileCache instance
        """
        return FileCache(
            cache_dir=cache_dir,
            serialization=serialization,
            file_extension=file_extension,
            max_files=max_files,
            cleanup_interval=cleanup_interval,
            create_subdirs=create_subdirs,
            safe_filenames=safe_filenames,
        )

    @classmethod
    def clear_cache_instances(cls) -> None:
        """Clear cached instances"""
        logger.info(f"Clearing {len(cls._instances)} cached instances")
        cls._instances.clear()

    @classmethod
    def get_cache_instance_info(cls) -> Dict[str, str]:
        """Get information about cached instances"""
        info = {
            key: type(instance).__name__ for key, instance in cls._instances.items()
        }
        logger.debug(f"Cache instance info: {info}")
        return info

    @classmethod
    def health_check_all_caches(cls) -> Dict[str, bool]:
        """Perform health check on all cached instances"""
        results = {}
        for cache_key, cache_instance in cls._instances.items():
            try:
                # Basic health check - try to set and get a test value
                test_key = "health_check_test"
                test_value = {"test": True}

                set_success = cache_instance.set(test_key, test_value, ttl=30)
                get_result = cache_instance.get(test_key)
                delete_success = cache_instance.delete(test_key)

                is_healthy = set_success and get_result == test_value and delete_success

                results[cache_key] = is_healthy
                logger.debug(f"Health check for {cache_key}: {is_healthy}")

            except Exception as e:
                logger.error(f"Health check failed for {cache_key}: {e}")
                results[cache_key] = False

        return results
