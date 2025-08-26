# common/cache/file_cache.py

import glob
import hashlib
import json
import os
import pickle
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .cache_interface import CacheInterface, CacheStats


class FileCacheEntry:
    """File cache entry with metadata"""

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileCacheEntry":
        """Create from dictionary"""
        entry = cls.__new__(cls)
        entry.value = data["value"]
        entry.created_at = data["created_at"]
        entry.ttl = data["ttl"]
        entry.expires_at = data["expires_at"]
        return entry


class FileCache(CacheInterface):
    """File-based cache implementation with TTL support"""

    def __init__(
        self,
        cache_dir: str = ".cache",
        serialization: str = "json",  # "json" or "pickle"
        file_extension: Optional[str] = None,
        max_files: Optional[int] = None,
        cleanup_interval: int = 300,  # 5 minutes
        create_subdirs: bool = True,
        safe_filenames: bool = True,
    ):
        """
        Initialize file cache

        Args:
            cache_dir: Directory to store cache files
            serialization: Serialization method ("json" or "pickle")
            file_extension: File extension (auto-determined if None)
            max_files: Maximum number of cache files (None for unlimited)
            cleanup_interval: Interval for cleanup expired files in seconds
            create_subdirs: Create subdirectories based on key hash
            safe_filenames: Use safe filenames (hash-based)
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.serialization = serialization
        self.max_files = max_files
        self.cleanup_interval = cleanup_interval
        self.create_subdirs = create_subdirs
        self.safe_filenames = safe_filenames

        # Determine file extension
        if file_extension is None:
            self.file_extension = ".json" if serialization == "json" else ".pkl"
        else:
            self.file_extension = (
                file_extension
                if file_extension.startswith(".")
                else f".{file_extension}"
            )

        self._lock = threading.RLock()
        self._stats = CacheStats()

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired, daemon=True
        )
        self._cleanup_thread.start()

    def _cleanup_expired(self):
        """Background thread to clean up expired files"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._remove_expired_files()
            except Exception:
                # Continue running even if cleanup fails
                pass

    def _remove_expired_files(self):
        """Remove expired cache files"""
        with self._lock:
            pattern = (
                f"**/*{self.file_extension}"
                if self.create_subdirs
                else f"*{self.file_extension}"
            )
            cache_files = glob.glob(str(self.cache_dir / pattern), recursive=True)

            for file_path in cache_files:
                try:
                    entry = self._load_entry(file_path)
                    if entry and entry.is_expired():
                        os.remove(file_path)
                        self._stats.record_expired()
                except Exception:
                    continue

    def _make_filename(self, key: str) -> str:
        """Generate filename from cache key"""
        if self.safe_filenames:
            # Use hash for safe filename
            key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()

            if self.create_subdirs:
                # Create subdirectory based on first 2 characters of hash
                subdir = key_hash[:2]
                filename = f"{key_hash[2:]}{self.file_extension}"
                return str(self.cache_dir / subdir / filename)
            else:
                return str(self.cache_dir / f"{key_hash}{self.file_extension}")
        else:
            # Use key as filename (may need escaping)
            safe_key = re.sub(r"[^\w\-_.]", "_", key)
            return str(self.cache_dir / f"{safe_key}{self.file_extension}")

    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        return Path(self._make_filename(key))

    def _serialize_entry(self, entry: FileCacheEntry) -> bytes:
        """Serialize cache entry"""
        data = entry.to_dict()

        if self.serialization == "json":
            return json.dumps(data, default=str, indent=2).encode("utf-8")
        elif self.serialization == "pickle":
            return pickle.dumps(data)
        else:
            raise ValueError(f"Unsupported serialization: {self.serialization}")

    def _deserialize_entry(self, data: bytes) -> FileCacheEntry:
        """Deserialize cache entry"""
        if self.serialization == "json":
            entry_data = json.loads(data.decode("utf-8"))
        elif self.serialization == "pickle":
            entry_data = pickle.loads(data)
        else:
            raise ValueError(f"Unsupported serialization: {self.serialization}")

        return FileCacheEntry.from_dict(entry_data)

    def _load_entry(self, file_path: Union[str, Path]) -> Optional[FileCacheEntry]:
        """Load cache entry from file"""
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            return self._deserialize_entry(data)
        except Exception:
            return None

    def _save_entry(self, file_path: Path, entry: FileCacheEntry) -> bool:
        """Save cache entry to file"""
        try:
            # Create parent directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            serialized_data = self._serialize_entry(entry)

            # Atomic write using temporary file
            temp_path = file_path.with_suffix(f"{self.file_extension}.tmp")
            with open(temp_path, "wb") as f:
                f.write(serialized_data)

            # Atomic rename
            temp_path.rename(file_path)
            return True
        except Exception:
            return False

    def _ensure_capacity(self):
        """Ensure cache doesn't exceed max_files"""
        if self.max_files is None:
            return

        pattern = (
            f"**/*{self.file_extension}"
            if self.create_subdirs
            else f"*{self.file_extension}"
        )
        cache_files = glob.glob(str(self.cache_dir / pattern), recursive=True)

        if len(cache_files) >= self.max_files:
            # Remove oldest files
            cache_files.sort(key=lambda x: os.path.getmtime(x))
            files_to_remove = cache_files[: len(cache_files) - self.max_files + 1]

            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                except Exception:
                    continue

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve value from cache"""
        with self._lock:
            file_path = self._get_file_path(key)

            if not file_path.exists():
                self._stats.record_miss()
                return default

            entry = self._load_entry(file_path)
            if entry is None:
                self._stats.record_miss()
                return default

            if entry.is_expired():
                try:
                    file_path.unlink()
                    self._stats.record_expired()
                except Exception:
                    pass
                self._stats.record_miss()
                return default

            self._stats.record_hit()
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache"""
        with self._lock:
            try:
                self._ensure_capacity()

                file_path = self._get_file_path(key)
                entry = FileCacheEntry(value, ttl)

                if self._save_entry(file_path, entry):
                    self._stats.record_set()
                    return True
                return False
            except Exception:
                return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            file_path = self._get_file_path(key)

            if file_path.exists():
                try:
                    file_path.unlink()
                    self._stats.record_delete()
                    return True
                except Exception:
                    return False
            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        with self._lock:
            try:
                pattern = (
                    f"**/*{self.file_extension}"
                    if self.create_subdirs
                    else f"*{self.file_extension}"
                )
                cache_files = glob.glob(str(self.cache_dir / pattern), recursive=True)

                for file_path in cache_files:
                    try:
                        os.remove(file_path)
                    except Exception:
                        continue

                self._stats.record_clear()
                return True
            except Exception:
                return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            file_path = self._get_file_path(key)

            if not file_path.exists():
                return False

            entry = self._load_entry(file_path)
            if entry is None:
                return False

            if entry.is_expired():
                try:
                    file_path.unlink()
                    self._stats.record_expired()
                except Exception:
                    pass
                return False

            return True

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get list of cache keys"""
        with self._lock:
            # Remove expired files first
            self._remove_expired_files()

            cache_files = []
            file_pattern = (
                f"**/*{self.file_extension}"
                if self.create_subdirs
                else f"*{self.file_extension}"
            )
            cache_files = glob.glob(str(self.cache_dir / file_pattern), recursive=True)

            # For safe filenames, we can't reverse the hash, so return file paths
            if self.safe_filenames:
                keys = []
                for file_path in cache_files:
                    # Try to load entry to verify it's valid
                    entry = self._load_entry(file_path)
                    if entry and not entry.is_expired():
                        # Use relative path as key for hashed filenames
                        rel_path = Path(file_path).relative_to(self.cache_dir)
                        key = str(rel_path.with_suffix(""))
                        keys.append(key)
                return keys
            else:
                # Extract keys from filenames
                keys = []
                for file_path in cache_files:
                    filename = Path(file_path).stem
                    keys.append(filename)

                if pattern is None:
                    return keys

                # Simple pattern matching
                regex_pattern = pattern.replace("*", ".*").replace("?", ".")
                compiled_pattern = re.compile(regex_pattern)

                return [key for key in keys if compiled_pattern.match(key)]

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._stats

    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key"""
        with self._lock:
            file_path = self._get_file_path(key)

            if not file_path.exists():
                return None

            entry = self._load_entry(file_path)
            if entry is None:
                return None

            if entry.is_expired():
                try:
                    file_path.unlink()
                    self._stats.record_expired()
                except Exception:
                    pass
                return None

            return entry.get_remaining_ttl()

    def set_ttl(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        with self._lock:
            file_path = self._get_file_path(key)

            if not file_path.exists():
                return False

            entry = self._load_entry(file_path)
            if entry is None:
                return False

            if entry.is_expired():
                try:
                    file_path.unlink()
                    self._stats.record_expired()
                except Exception:
                    pass
                return False

            # Create new entry with same value but new TTL
            new_entry = FileCacheEntry(entry.value, ttl)
            return self._save_entry(file_path, new_entry)

    def get_size(self) -> int:
        """Get current cache size"""
        with self._lock:
            pattern = (
                f"**/*{self.file_extension}"
                if self.create_subdirs
                else f"*{self.file_extension}"
            )
            cache_files = glob.glob(str(self.cache_dir / pattern), recursive=True)

            # Filter out expired files
            valid_files = 0
            for file_path in cache_files:
                entry = self._load_entry(file_path)
                if entry and not entry.is_expired():
                    valid_files += 1

            return valid_files

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get disk usage information"""
        with self._lock:
            total_size = 0
            file_count = 0

            pattern = (
                f"**/*{self.file_extension}"
                if self.create_subdirs
                else f"*{self.file_extension}"
            )
            cache_files = glob.glob(str(self.cache_dir / pattern), recursive=True)

            for file_path in cache_files:
                try:
                    file_stat = os.stat(file_path)
                    total_size += file_stat.st_size
                    file_count += 1
                except Exception:
                    continue

            return {
                "total_bytes": total_size,
                "total_mb": total_size / (1024 * 1024),
                "file_count": file_count,
                "cache_dir": str(self.cache_dir),
                "max_files": self.max_files,
                "utilization": file_count / self.max_files if self.max_files else 0,
            }
