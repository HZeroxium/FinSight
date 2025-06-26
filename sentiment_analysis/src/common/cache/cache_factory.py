# common/cache/cache_factory.py

from typing import Optional, Dict, Any
from enum import Enum

from .cache_interface import CacheInterface
from .in_memory_cache import InMemoryCache
from .redis_cache import RedisCache
from .file_cache import FileCache


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

        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls.create_cache(
                name=name, cache_type=cache_type, **kwargs
            )

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
        if cache_type == CacheType.MEMORY:
            return cls._create_memory_cache(**kwargs)
        elif cache_type == CacheType.REDIS:
            return cls._create_redis_cache(**kwargs)
        elif cache_type == CacheType.FILE:
            return cls._create_file_cache(**kwargs)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

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
        return InMemoryCache(**defaults)

    @classmethod
    def _create_redis_cache(cls, **kwargs) -> RedisCache:
        """Create Redis cache with default parameters"""
        defaults = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": None,
            "serialization": "json",
            "key_prefix": "",
        }
        defaults.update(kwargs)
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
        cls._instances.clear()

    @classmethod
    def get_cache_instance_info(cls) -> Dict[str, str]:
        """Get information about cached instances"""
        return {
            key: type(instance).__name__ for key, instance in cls._instances.items()
        }
