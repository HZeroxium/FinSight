# common/cache/redis_cache.py

import json
import pickle
import time
from typing import Any, Optional, Dict, List

from .cache_interface import CacheInterface, CacheStats

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCache(CacheInterface):
    """Redis-based cache implementation"""

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
            raise ImportError("Redis is not available. Install with: pip install redis")

        self.key_prefix = key_prefix
        self.serialization = serialization
        self._stats = CacheStats()

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
            self.redis_client = redis.Redis(**redis_params)
            # Test connection
            self.redis_client.ping()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self.key_prefix}{key}" if self.key_prefix else key

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if self.serialization == "json":
            return json.dumps(value, default=str).encode("utf-8")
        elif self.serialization == "pickle":
            return pickle.dumps(value)
        else:
            raise ValueError(f"Unsupported serialization: {self.serialization}")

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if self.serialization == "json":
            return json.loads(data.decode("utf-8"))
        elif self.serialization == "pickle":
            return pickle.loads(data)
        else:
            raise ValueError(f"Unsupported serialization: {self.serialization}")

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve value from cache"""
        try:
            redis_key = self._make_key(key)
            data = self.redis_client.get(redis_key)

            if data is None:
                self._stats.record_miss()
                return default

            value = self._deserialize(data)
            self._stats.record_hit()
            return value
        except Exception:
            self._stats.record_miss()
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache"""
        try:
            redis_key = self._make_key(key)
            serialized_value = self._serialize(value)

            if ttl is not None:
                result = self.redis_client.setex(redis_key, ttl, serialized_value)
            else:
                result = self.redis_client.set(redis_key, serialized_value)

            if result:
                self._stats.record_set()
                return True
            return False
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            redis_key = self._make_key(key)
            result = self.redis_client.delete(redis_key)

            if result > 0:
                self._stats.record_delete()
                return True
            return False
        except Exception:
            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            if self.key_prefix:
                # Delete only keys with our prefix
                pattern = f"{self.key_prefix}*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                # Clear entire database
                self.redis_client.flushdb()

            self._stats.record_clear()
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_key = self._make_key(key)
            return bool(self.redis_client.exists(redis_key))
        except Exception:
            return False

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get list of cache keys"""
        try:
            if pattern:
                search_pattern = self._make_key(pattern)
            else:
                search_pattern = f"{self.key_prefix}*" if self.key_prefix else "*"

            redis_keys = self.redis_client.keys(search_pattern)

            # Remove prefix from keys
            if self.key_prefix:
                prefix_len = len(self.key_prefix)
                return [key.decode("utf-8")[prefix_len:] for key in redis_keys]
            else:
                return [key.decode("utf-8") for key in redis_keys]
        except Exception:
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
                return None
            elif ttl == -2:  # Key doesn't exist
                return None
            else:
                return ttl
        except Exception:
            return None

    def set_ttl(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        try:
            redis_key = self._make_key(key)
            return bool(self.redis_client.expire(redis_key, ttl))
        except Exception:
            return False

    def get_size(self) -> int:
        """Get current cache size"""
        try:
            if self.key_prefix:
                pattern = f"{self.key_prefix}*"
                return len(self.redis_client.keys(pattern))
            else:
                return self.redis_client.dbsize()
        except Exception:
            return 0

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            info = self.redis_client.info("memory")
            return {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "used_memory_rss": info.get("used_memory_rss", 0),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
                "total_system_memory": info.get("total_system_memory", 0),
                "maxmemory": info.get("maxmemory", 0),
                "entry_count": self.get_size(),
            }
        except Exception:
            return {"error": "Unable to get memory usage information"}

    def ping(self) -> bool:
        """Test Redis connection"""
        try:
            return self.redis_client.ping()
        except Exception:
            return False

    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        try:
            return dict(self.redis_client.info())
        except Exception:
            return {}
