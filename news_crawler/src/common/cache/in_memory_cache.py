# common/cache/in_memory_cache.py

import time
import threading
import sys
import re
from typing import Any, Optional, Dict, List
from collections import OrderedDict

from .cache_interface import CacheInterface, CacheStats


class CacheEntry:
    """Internal cache entry with TTL support"""

    def __init__(self, value: Any, ttl: Optional[int] = None):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.expires_at = self.created_at + ttl if ttl else None

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def get_remaining_ttl(self) -> Optional[int]:
        """Get remaining TTL in seconds"""
        if self.expires_at is None:
            return None
        remaining = int(self.expires_at - time.time())
        return max(0, remaining)


class InMemoryCache(CacheInterface):
    """Thread-safe in-memory cache implementation with TTL support"""

    def __init__(
        self,
        max_size: Optional[int] = None,
        default_ttl: Optional[int] = None,
        cleanup_interval: int = 60,
        enable_lru: bool = True,
    ):
        """
        Initialize in-memory cache

        Args:
            max_size: Maximum number of entries (None for unlimited)
            default_ttl: Default TTL in seconds (None for no expiration)
            cleanup_interval: Interval for cleanup expired entries in seconds
            enable_lru: Enable LRU eviction when max_size is reached
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.enable_lru = enable_lru

        # Use OrderedDict for LRU support
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired, daemon=True
        )
        self._cleanup_thread.start()

    def _cleanup_expired(self):
        """Background thread to clean up expired entries"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._remove_expired_entries()
            except Exception:
                # Continue running even if cleanup fails
                pass

    def _remove_expired_entries(self):
        """Remove expired entries from cache"""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self._stats.record_expired()

    def _evict_lru(self):
        """Evict least recently used entry"""
        if self._cache:
            self._cache.popitem(last=False)  # Remove first (oldest) item

    def _ensure_capacity(self):
        """Ensure cache doesn't exceed max_size"""
        if self.max_size is None:
            return

        while len(self._cache) >= self.max_size:
            if self.enable_lru:
                self._evict_lru()
            else:
                # Remove oldest entry
                if self._cache:
                    self._cache.popitem(last=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve value from cache"""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.record_miss()
                return default

            if entry.is_expired():
                del self._cache[key]
                self._stats.record_expired()
                self._stats.record_miss()
                return default

            # Move to end for LRU
            if self.enable_lru:
                self._cache.move_to_end(key)

            self._stats.record_hit()
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache"""
        try:
            with self._lock:
                # Use provided TTL or default TTL
                effective_ttl = ttl if ttl is not None else self.default_ttl

                # Ensure capacity before adding new entry
                if key not in self._cache:
                    self._ensure_capacity()

                # Create cache entry
                entry = CacheEntry(value, effective_ttl)
                self._cache[key] = entry

                # Move to end for LRU
                if self.enable_lru:
                    self._cache.move_to_end(key)

                self._stats.record_set()
                return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.record_delete()
                return True
            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            with self._lock:
                self._cache.clear()
                self._stats.record_clear()
                return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False

            if entry.is_expired():
                del self._cache[key]
                self._stats.record_expired()
                return False

            return True

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get list of cache keys"""
        with self._lock:
            # Remove expired entries first
            self._remove_expired_entries()

            all_keys = list(self._cache.keys())

            if pattern is None:
                return all_keys

            # Simple pattern matching with wildcards
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            compiled_pattern = re.compile(regex_pattern)

            return [key for key in all_keys if compiled_pattern.match(key)]

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._stats

    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats.record_expired()
                return None

            return entry.get_remaining_ttl()

    def set_ttl(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False

            if entry.is_expired():
                del self._cache[key]
                self._stats.record_expired()
                return False

            # Create new entry with same value but new TTL
            new_entry = CacheEntry(entry.value, ttl)
            self._cache[key] = new_entry

            if self.enable_lru:
                self._cache.move_to_end(key)

            return True

    def get_size(self) -> int:
        """Get current cache size"""
        with self._lock:
            self._remove_expired_entries()
            return len(self._cache)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        with self._lock:
            total_size = 0
            for key, entry in self._cache.items():
                # Rough estimation of memory usage
                key_size = sys.getsizeof(key)
                value_size = sys.getsizeof(entry.value)
                entry_overhead = sys.getsizeof(entry)
                total_size += key_size + value_size + entry_overhead

            return {
                "total_bytes": total_size,
                "total_mb": total_size / (1024 * 1024),
                "entry_count": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size if self.max_size else 0,
            }
