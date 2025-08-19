# utils/cache_utils.py

"""
Cache utilities for news service.
Provides cache management, key generation, and invalidation mechanisms.
"""

from typing import Optional, Dict, Any, Callable
from datetime import datetime, timezone
from enum import Enum

from common.cache import CacheFactory, CacheType
from common.cache.cache_interface import CacheInterface
from common.logger import LoggerFactory, LoggerType, LogLevel
from ..core.config import settings


class CacheEndpoint(Enum):
    """Cache endpoint types for different TTL configurations"""

    SEARCH_NEWS = "search_news"
    RECENT_NEWS = "recent_news"
    NEWS_BY_SOURCE = "news_by_source"
    NEWS_BY_KEYWORDS = "news_by_keywords"
    NEWS_BY_TAGS = "news_by_tags"
    AVAILABLE_TAGS = "available_tags"
    REPOSITORY_STATS = "repository_stats"
    NEWS_ITEM = "news_item"
    HEALTH_CHECK = "health_check"


class CacheManager:
    """Centralized cache manager for news service"""

    def __init__(self):
        """Initialize cache manager with Redis configuration"""
        self.logger = LoggerFactory.get_logger(
            name="cache-manager",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file=f"{settings.log_file_path}cache_manager.log",
        )

        self._cache_instance: Optional[CacheInterface] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize cache instance with Redis configuration"""
        if self._initialized:
            return

        try:
            if not settings.enable_caching:
                self.logger.info("Caching is disabled, using memory cache fallback")
                self._cache_instance = CacheFactory.get_cache(
                    name="news-service-memory",
                    cache_type=CacheType.MEMORY,
                    # key_prefix=settings.redis_key_prefix,
                )
            else:
                self.logger.info("Initializing Redis cache for news service")
                self._cache_instance = CacheFactory.get_cache(
                    name="news-service-redis",
                    cache_type=CacheType.REDIS,
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    password=settings.redis_password,
                    key_prefix=settings.redis_key_prefix,
                    connection_timeout=settings.redis_connection_timeout,
                    socket_timeout=settings.redis_socket_timeout,
                    socket_connect_timeout=settings.redis_socket_connect_timeout,
                    socket_keepalive=settings.redis_socket_keepalive,
                    retry_on_timeout=settings.redis_retry_on_timeout,
                    max_connections=settings.redis_max_connections,
                )

            self._initialized = True
            self.logger.info("Cache manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {e}")
            # Fallback to memory cache
            self._cache_instance = CacheFactory.get_cache(
                name="news-service-memory-fallback",
                cache_type=CacheType.MEMORY,
                key_prefix=settings.redis_key_prefix,
            )
            self._initialized = True
            self.logger.warning("Using memory cache fallback")

    @property
    def cache(self) -> CacheInterface:
        """Get cache instance"""
        if not self._initialized:
            raise RuntimeError(
                "Cache manager not initialized. Call initialize() first."
            )
        return self._cache_instance

    def get_ttl_for_endpoint(self, endpoint: CacheEndpoint) -> int:
        """Get TTL for specific endpoint"""
        ttl_mapping = {
            CacheEndpoint.SEARCH_NEWS: settings.cache_ttl_search_news,
            CacheEndpoint.RECENT_NEWS: settings.cache_ttl_recent_news,
            CacheEndpoint.NEWS_BY_SOURCE: settings.cache_ttl_news_by_source,
            CacheEndpoint.NEWS_BY_KEYWORDS: settings.cache_ttl_news_by_keywords,
            CacheEndpoint.NEWS_BY_TAGS: settings.cache_ttl_news_by_tags,
            CacheEndpoint.AVAILABLE_TAGS: settings.cache_ttl_available_tags,
            CacheEndpoint.REPOSITORY_STATS: settings.cache_ttl_repository_stats,
            CacheEndpoint.NEWS_ITEM: settings.cache_ttl_news_item,
            CacheEndpoint.HEALTH_CHECK: 300,  # 5 minutes for health checks
        }
        return ttl_mapping.get(endpoint, settings.cache_ttl_seconds)

    def generate_cache_key(self, endpoint: CacheEndpoint, *args, **kwargs) -> str:
        """Generate cache key for endpoint with arguments"""
        # Convert args and kwargs to string representation
        args_str = "_".join(str(arg) for arg in args if arg is not None)
        kwargs_str = "_".join(
            f"{k}_{v}" for k, v in sorted(kwargs.items()) if v is not None
        )

        # Combine all parts
        key_parts = [endpoint.value]
        if args_str:
            key_parts.append(args_str)
        if kwargs_str:
            key_parts.append(kwargs_str)

        cache_key = "_".join(key_parts)

        # Clean up the key (remove special characters, limit length)
        cache_key = cache_key.replace(" ", "_").replace(":", "_").replace("/", "_")
        cache_key = cache_key[:200]  # Limit key length

        self.logger.debug(f"Generated cache key: {cache_key}")
        return cache_key

    async def get_cached_data(
        self, endpoint: CacheEndpoint, *args, **kwargs
    ) -> Optional[Any]:
        """Get cached data for endpoint"""
        try:
            cache_key = self.generate_cache_key(endpoint, *args, **kwargs)
            cached_data = self.cache.get(cache_key)

            if cached_data is not None:
                self.logger.debug(f"Cache hit for {endpoint.value}: {cache_key}")
                return cached_data
            else:
                self.logger.debug(f"Cache miss for {endpoint.value}: {cache_key}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting cached data for {endpoint.value}: {e}")
            return None

    async def set_cached_data(
        self, endpoint: CacheEndpoint, data: Any, *args, **kwargs
    ) -> bool:
        """Set cached data for endpoint"""
        try:
            cache_key = self.generate_cache_key(endpoint, *args, **kwargs)
            ttl = self.get_ttl_for_endpoint(endpoint)

            success = self.cache.set(cache_key, data, ttl)

            if success:
                self.logger.debug(
                    f"Cached data for {endpoint.value}: {cache_key} (TTL: {ttl}s)"
                )
            else:
                self.logger.warning(
                    f"Failed to cache data for {endpoint.value}: {cache_key}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error setting cached data for {endpoint.value}: {e}")
            return False

    async def invalidate_endpoint_cache(
        self, endpoint: CacheEndpoint, *args, **kwargs
    ) -> bool:
        """Invalidate cache for specific endpoint and arguments"""
        try:
            cache_key = self.generate_cache_key(endpoint, *args, **kwargs)
            success = self.cache.delete(cache_key)

            if success:
                self.logger.info(f"Invalidated cache for {endpoint.value}: {cache_key}")
            else:
                self.logger.debug(
                    f"No cache entry found for {endpoint.value}: {cache_key}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error invalidating cache for {endpoint.value}: {e}")
            return False

    async def invalidate_all_cache(self) -> bool:
        """Invalidate all cache entries for this service prefix.

        Delegates to cache.clear() which is prefix-aware in Redis implementation.
        """
        try:
            if not settings.cache_invalidation_enabled:
                self.logger.info("Cache invalidation is disabled")
                return True

            result = self.cache.clear()
            self.logger.info(
                "All cache entries invalidated"
                if result
                else "No cache entries invalidated"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error invalidating all cache: {e}")
            return False

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = self.cache.get_stats()
            # Rely on implementation to apply prefix automatically
            keys = self.cache.keys()

            return {
                "cache_type": type(self.cache).__name__,
                "total_keys": len(keys),
                "hit_rate": stats.get_hit_rate(),
                "uptime": stats.get_uptime(),
                "memory_usage": self.cache.get_memory_usage(),
                "cache_keys_sample": keys[:10],  # Show first 10 keys
            }

        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """Check cache health"""
        try:
            # Try to set and get a test value (no manual prefix)
            test_key = "health_check"
            test_value = {"timestamp": datetime.now(timezone.utc).isoformat()}

            # Set test value
            set_success = self.cache.set(test_key, test_value, 60)
            if not set_success:
                return False

            # Get test value
            retrieved_value = self.cache.get(test_key)
            if retrieved_value != test_value:
                return False

            # Clean up
            self.cache.delete(test_key)

            return True

        except Exception as e:
            self.logger.error(f"Cache health check failed: {e}")
            return False


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()

    return _cache_manager


def create_cache_decorator(endpoint: CacheEndpoint):
    """Create a cache decorator for specific endpoint"""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            cache_manager = await get_cache_manager()

            # Try to get from cache
            cached_result = await cache_manager.get_cached_data(
                endpoint, *args, **kwargs
            )
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set_cached_data(endpoint, result, *args, **kwargs)

            return result

        # Add cache invalidation method to function
        async def invalidate_cache(*args, **kwargs):
            cache_manager = await get_cache_manager()
            return await cache_manager.invalidate_endpoint_cache(
                endpoint, *args, **kwargs
            )

        wrapper.invalidate_cache = invalidate_cache
        return wrapper

    return decorator


async def invalidate_all_news_cache() -> bool:
    """Invalidate all news-related cache entries"""
    cache_manager = await get_cache_manager()
    return await cache_manager.invalidate_all_cache()


async def get_cache_statistics() -> Dict[str, Any]:
    """Get cache statistics"""
    cache_manager = await get_cache_manager()
    return await cache_manager.get_cache_stats()


async def check_cache_health() -> bool:
    """Check cache health"""
    cache_manager = await get_cache_manager()
    return await cache_manager.health_check()
