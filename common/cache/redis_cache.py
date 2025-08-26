# common/cache/redis_cache.py

import json
import pickle
import time
from typing import Any, Dict, List, Optional

from common.logger import LoggerFactory, LoggerType, LogLevel

from .cache_interface import CacheInterface, CacheStats

# Create logger for Redis cache
logger = LoggerFactory.get_logger(
    name="redis-cache",
    logger_type=LoggerType.STANDARD,
    level=LogLevel.INFO,
    console_level=LogLevel.INFO,
    file_level=LogLevel.DEBUG,
    log_file="logs/redis_cache.log",
)

try:
    import redis

    REDIS_AVAILABLE = True
    logger.info("Redis library is available")
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis library is not available. Install with: pip install redis")


class RedisCache(CacheInterface):
    """Redis-based cache implementation with comprehensive logging"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: Optional[float] = None,
        socket_connect_timeout: Optional[float] = None,
        socket_keepalive: bool = False,
        socket_keepalive_options: Optional[Dict] = None,
        connection_pool: Optional[Any] = None,
        unix_socket_path: Optional[str] = None,
        encoding: str = "utf-8",
        encoding_errors: str = "strict",
        decode_responses: bool = False,
        retry_on_timeout: bool = False,
        ssl: bool = False,
        ssl_keyfile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        ssl_cert_reqs: str = "required",
        ssl_ca_certs: Optional[str] = None,
        ssl_check_hostname: bool = False,
        max_connections: Optional[int] = None,
        serialization: str = "json",  # "json" or "pickle"
        key_prefix: str = "",
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password
            serialization: Serialization method ("json" or "pickle")
            key_prefix: Prefix for all cache keys
            **kwargs: Additional Redis connection parameters
        """
        if not REDIS_AVAILABLE:
            logger.error("Redis is not available. Install with: pip install redis")
            raise ImportError("Redis is not available. Install with: pip install redis")

        self.key_prefix = key_prefix
        self.serialization = serialization
        self._stats = CacheStats()
        self._connection_info = {"host": host, "port": port, "db": db}

        logger.info(f"Initializing Redis cache: {host}:{port}/{db}")

        # Redis connection parameters
        redis_params = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "socket_timeout": socket_timeout,
            "socket_connect_timeout": socket_connect_timeout,
            "socket_keepalive": socket_keepalive,
            "socket_keepalive_options": socket_keepalive_options,
            "connection_pool": connection_pool,
            "unix_socket_path": unix_socket_path,
            "encoding": encoding,
            "encoding_errors": encoding_errors,
            "decode_responses": decode_responses,
            "retry_on_timeout": retry_on_timeout,
            "ssl": ssl,
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile,
            "ssl_cert_reqs": ssl_cert_reqs,
            "ssl_ca_certs": ssl_ca_certs,
            "ssl_check_hostname": ssl_check_hostname,
            "max_connections": max_connections,
        }

        # Remove None values
        redis_params = {k: v for k, v in redis_params.items() if v is not None}

        try:
            logger.debug(f"Creating Redis connection with params: {redis_params}")
            self.redis_client = redis.Redis(**redis_params)

            # Test connection
            ping_result = self.redis_client.ping()
            logger.info(f"Redis connection successful: ping={ping_result}")

            # Log Redis server info
            info = self.redis_client.info()
            logger.info(f"Redis server version: {info.get('redis_version', 'unknown')}")
            logger.info(
                f"Redis memory usage: {info.get('used_memory_human', 'unknown')}"
            )

        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.error(f"Connection details: {self._connection_info}")
            raise ConnectionError(f"Failed to connect to Redis: {e}")
        except redis.AuthenticationError as e:
            logger.error(f"Redis authentication failed: {e}")
            raise ConnectionError(f"Redis authentication failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        prefixed_key = f"{self.key_prefix}{key}" if self.key_prefix else key
        logger.debug(f"Generated cache key: {prefixed_key}")
        return prefixed_key

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            if self.serialization == "json":
                # Handle Pydantic models
                if hasattr(value, "model_dump"):
                    serialized_data = value.model_dump()
                elif hasattr(value, "__dict__"):
                    serialized_data = value.__dict__
                else:
                    serialized_data = value

                serialized = json.dumps(serialized_data, default=str).encode("utf-8")
                logger.debug(f"JSON serialized value of size: {len(serialized)} bytes")
                return serialized
            elif self.serialization == "pickle":
                serialized = pickle.dumps(value)
                logger.debug(
                    f"Pickle serialized value of size: {len(serialized)} bytes"
                )
                return serialized
            else:
                raise ValueError(f"Unsupported serialization: {self.serialization}")
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            if self.serialization == "json":
                result = json.loads(data.decode("utf-8"))
                logger.debug(f"JSON deserialized value from {len(data)} bytes")
                return result
            elif self.serialization == "pickle":
                result = pickle.loads(data)
                logger.debug(f"Pickle deserialized value from {len(data)} bytes")
                return result
            else:
                raise ValueError(f"Unsupported serialization: {self.serialization}")
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve value from cache"""
        start_time = time.time()
        try:
            redis_key = self._make_key(key)
            logger.debug(f"Getting value for key: {redis_key}")

            data = self.redis_client.get(redis_key)

            if data is None:
                logger.debug(f"Cache miss for key: {redis_key}")
                self._stats.record_miss()
                return default

            value = self._deserialize(data)
            duration = time.time() - start_time
            logger.debug(f"Cache hit for key: {redis_key} (took {duration:.3f}s)")
            self._stats.record_hit()
            return value

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during get: {e}")
            self._stats.record_miss()
            return default
        except Exception as e:
            logger.error(f"Error getting cache value for key {key}: {e}")
            self._stats.record_miss()
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache"""
        start_time = time.time()
        try:
            redis_key = self._make_key(key)
            serialized_value = self._serialize(value)

            logger.debug(f"Setting value for key: {redis_key} (TTL: {ttl})")

            if ttl is not None:
                result = self.redis_client.setex(redis_key, ttl, serialized_value)
            else:
                result = self.redis_client.set(redis_key, serialized_value)

            duration = time.time() - start_time
            if result:
                logger.debug(
                    f"Successfully set cache value for key: {redis_key} (took {duration:.3f}s)"
                )
                self._stats.record_set()
                return True
            else:
                logger.warning(f"Failed to set cache value for key: {redis_key}")
                return False

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during set: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting cache value for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            redis_key = self._make_key(key)
            logger.debug(f"Deleting key: {redis_key}")

            result = self.redis_client.delete(redis_key)

            if result > 0:
                logger.debug(f"Successfully deleted key: {redis_key}")
                self._stats.record_delete()
                return True
            else:
                logger.debug(f"Key not found for deletion: {redis_key}")
                return False

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during delete: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            logger.info("Clearing cache entries")

            if self.key_prefix:
                # Delete only keys with our prefix
                pattern = f"{self.key_prefix}*"
                logger.debug(f"Clearing keys matching pattern: {pattern}")
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted_count = self.redis_client.delete(*keys)
                    logger.info(
                        f"Deleted {deleted_count} keys with prefix '{self.key_prefix}'"
                    )
                else:
                    logger.info(f"No keys found with prefix '{self.key_prefix}'")
            else:
                # Clear entire database
                logger.warning("Clearing entire Redis database")
                self.redis_client.flushdb()

            self._stats.record_clear()
            return True

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during clear: {e}")
            return False
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_key = self._make_key(key)
            result = bool(self.redis_client.exists(redis_key))
            logger.debug(f"Key existence check for {redis_key}: {result}")
            return result
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during exists check: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking key existence for {key}: {e}")
            return False

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get list of cache keys"""
        try:
            if pattern:
                search_pattern = self._make_key(pattern)
            else:
                search_pattern = f"{self.key_prefix}*" if self.key_prefix else "*"

            logger.debug(f"Searching for keys with pattern: {search_pattern}")
            redis_keys = self.redis_client.keys(search_pattern)

            # Remove prefix from keys
            if self.key_prefix:
                prefix_len = len(self.key_prefix)
                result_keys = [key.decode("utf-8")[prefix_len:] for key in redis_keys]
            else:
                result_keys = [key.decode("utf-8") for key in redis_keys]

            logger.debug(f"Found {len(result_keys)} keys matching pattern")
            return result_keys

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during keys search: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting cache keys: {e}")
            return []

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._stats

    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key"""
        try:
            redis_key = self._make_key(key)
            ttl = self.redis_client.ttl(redis_key)

            if ttl == -1:  # Key exists but has no TTL
                logger.debug(f"Key {redis_key} exists but has no TTL")
                return None
            elif ttl == -2:  # Key doesn't exist
                logger.debug(f"Key {redis_key} doesn't exist")
                return None
            else:
                logger.debug(f"Key {redis_key} TTL: {ttl} seconds")
                return ttl

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during TTL check: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return None

    def set_ttl(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        try:
            redis_key = self._make_key(key)
            result = bool(self.redis_client.expire(redis_key, ttl))
            logger.debug(
                f"Set TTL for key {redis_key}: {ttl} seconds, success: {result}"
            )
            return result
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during TTL set: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting TTL for key {key}: {e}")
            return False

    def get_size(self) -> int:
        """Get current cache size"""
        try:
            if self.key_prefix:
                pattern = f"{self.key_prefix}*"
                size = len(self.redis_client.keys(pattern))
            else:
                size = self.redis_client.dbsize()

            logger.debug(f"Cache size: {size} keys")
            return size

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during size check: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            return 0

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            info = self.redis_client.info("memory")
            memory_info = {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "used_memory_rss": info.get("used_memory_rss", 0),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
                "total_system_memory": info.get("total_system_memory", 0),
                "maxmemory": info.get("maxmemory", 0),
                "entry_count": self.get_size(),
            }
            logger.debug(f"Memory usage: {memory_info['used_memory_human']}")
            return memory_info

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during memory usage check: {e}")
            return {"error": f"Redis connection error: {e}"}
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": f"Unable to get memory usage: {e}"}

    def ping(self) -> bool:
        """Test Redis connection"""
        try:
            result = self.redis_client.ping()
            logger.debug(f"Redis ping result: {result}")
            return result
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during ping: {e}")
            return False
        except Exception as e:
            logger.error(f"Error pinging Redis: {e}")
            return False

    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        try:
            info = dict(self.redis_client.info())
            logger.debug("Retrieved Redis server information")
            return info
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during info retrieval: {e}")
            return {"error": f"Redis connection error: {e}"}
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {"error": f"Unable to get Redis info: {e}"}
