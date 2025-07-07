# common/cache/cache_interface.py

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from enum import Enum
import time


class CacheStatus(Enum):
    """Cache operation status"""

    HIT = "hit"
    MISS = "miss"
    SET = "set"
    DELETE = "delete"
    CLEAR = "clear"
    EXPIRED = "expired"


class CacheStats:
    """Cache statistics tracking"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.clears = 0
        self.expired = 0
        self.start_time = time.time()

    def record_hit(self):
        self.hits += 1

    def record_miss(self):
        self.misses += 1

    def record_set(self):
        self.sets += 1

    def record_delete(self):
        self.deletes += 1

    def record_clear(self):
        self.clears += 1

    def record_expired(self):
        self.expired += 1

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_uptime(self) -> float:
        return time.time() - self.start_time


class CacheInterface(ABC):
    """Abstract base interface for all cache implementations"""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve value from cache

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store value in cache

        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds (None for no expiration)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete key from cache

        Args:
            key: Cache key to delete

        Returns:
            True if key existed and was deleted, False otherwise
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all cache entries

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache

        Args:
            key: Cache key to check

        Returns:
            True if key exists, False otherwise
        """
        pass

    @abstractmethod
    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get list of cache keys

        Args:
            pattern: Optional pattern to match keys (implementation specific)

        Returns:
            List of cache keys
        """
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """
        Get cache statistics

        Returns:
            CacheStats object with usage statistics
        """
        pass

    @abstractmethod
    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, None if no TTL or key doesn't exist
        """
        pass

    @abstractmethod
    def set_ttl(self, key: str, ttl: int) -> bool:
        """
        Set TTL for existing key

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            True if successful, False if key doesn't exist
        """
        pass

    @abstractmethod
    def get_size(self) -> int:
        """
        Get current cache size (number of keys)

        Returns:
            Number of keys in cache
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information

        Returns:
            Dictionary with memory usage details
        """
        pass
