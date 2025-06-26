# common/cache/decorators.py

import functools
import hashlib
import json
from typing import Any, Optional, Callable

from .cache_interface import CacheInterface
from .cache_factory import CacheFactory, CacheType


def cache_result(
    cache: Optional[CacheInterface] = None,
    ttl: Optional[int] = None,
    key_prefix: str = "",
    include_args: bool = True,
    include_kwargs: bool = True,
    exclude_args: Optional[list] = None,
    exclude_kwargs: Optional[list] = None,
    cache_type: CacheType = CacheType.MEMORY,
    cache_name: str = "decorator_cache",
    **cache_kwargs,
):
    """
    Decorator to cache function results with TTL support

    Args:
        cache: Cache instance to use (if None, creates one)
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        include_args: Include positional arguments in cache key
        include_kwargs: Include keyword arguments in cache key
        exclude_args: List of positional argument indices to exclude
        exclude_kwargs: List of keyword argument names to exclude
        cache_type: Type of cache to create if cache is None
        cache_name: Name for the cache instance
        **cache_kwargs: Additional arguments for cache creation

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        # Get or create cache instance
        if cache is not None:
            cache_instance = cache
        else:
            cache_instance = CacheFactory.get_cache(
                name=cache_name, cache_type=cache_type, **cache_kwargs
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _generate_cache_key(
                func=func,
                args=args,
                kwargs=kwargs,
                key_prefix=key_prefix,
                include_args=include_args,
                include_kwargs=include_kwargs,
                exclude_args=exclude_args or [],
                exclude_kwargs=exclude_kwargs or [],
            )

            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)

            return result

        # Add cache management methods to wrapper
        wrapper._cache_instance = cache_instance
        wrapper._cache_key_prefix = key_prefix
        wrapper._original_func = func

        def clear_cache():
            """Clear all cached results for this function"""
            prefix = f"{key_prefix}{func.__module__}.{func.__name__}"
            keys = cache_instance.keys(f"{prefix}*")
            for key in keys:
                cache_instance.delete(key)

        def get_cache_info():
            """Get cache information for this function"""
            prefix = f"{key_prefix}{func.__module__}.{func.__name__}"
            keys = cache_instance.keys(f"{prefix}*")
            return {
                "cached_entries": len(keys),
                "cache_keys": keys,
                "cache_stats": cache_instance.get_stats(),
            }

        def invalidate_cache(*args, **kwargs):
            """Invalidate cache for specific arguments"""
            cache_key = _generate_cache_key(
                func=func,
                args=args,
                kwargs=kwargs,
                key_prefix=key_prefix,
                include_args=include_args,
                include_kwargs=include_kwargs,
                exclude_args=exclude_args or [],
                exclude_kwargs=exclude_kwargs or [],
            )
            return cache_instance.delete(cache_key)

        wrapper.clear_cache = clear_cache
        wrapper.get_cache_info = get_cache_info
        wrapper.invalidate_cache = invalidate_cache

        return wrapper

    return decorator


def _generate_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    key_prefix: str,
    include_args: bool,
    include_kwargs: bool,
    exclude_args: list,
    exclude_kwargs: list,
) -> str:
    """Generate cache key from function and arguments"""

    key_parts = [key_prefix, func.__module__, func.__name__]

    # Include positional arguments
    if include_args and args:
        filtered_args = [arg for i, arg in enumerate(args) if i not in exclude_args]
        if filtered_args:
            args_str = _serialize_for_key(filtered_args)
            key_parts.append(f"args:{args_str}")

    # Include keyword arguments
    if include_kwargs and kwargs:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_kwargs}
        if filtered_kwargs:
            # Sort kwargs for consistent key generation
            sorted_kwargs = dict(sorted(filtered_kwargs.items()))
            kwargs_str = _serialize_for_key(sorted_kwargs)
            key_parts.append(f"kwargs:{kwargs_str}")

    # Create final key
    key = "|".join(str(part) for part in key_parts if part)

    # Hash the key if it's too long
    if len(key) > 200:
        key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
        return f"{key_prefix}hash:{key_hash}"

    return key


def _serialize_for_key(obj: Any) -> str:
    """Serialize object for use in cache key"""
    try:
        # Try JSON serialization first (fastest and most readable)
        return json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Fallback to string representation
        return str(obj)


def cache_property(
    cache: Optional[CacheInterface] = None,
    ttl: Optional[int] = None,
    cache_type: CacheType = CacheType.MEMORY,
    cache_name: str = "property_cache",
    **cache_kwargs,
):
    """
    Decorator to cache property results

    Args:
        cache: Cache instance to use
        ttl: Time to live in seconds
        cache_type: Type of cache to create if cache is None
        cache_name: Name for the cache instance
        **cache_kwargs: Additional arguments for cache creation

    Returns:
        Decorated property
    """

    def decorator(func: Callable) -> property:
        # Get or create cache instance
        if cache is not None:
            cache_instance = cache
        else:
            cache_instance = CacheFactory.get_cache(
                name=cache_name, cache_type=cache_type, **cache_kwargs
            )

        @functools.wraps(func)
        def wrapper(self):
            # Generate cache key based on object id and method name
            cache_key = f"property:{id(self)}:{func.__name__}"

            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute property and cache result
            result = func(self)
            cache_instance.set(cache_key, result, ttl)

            return result

        return property(wrapper)

    return decorator


class cached_method:
    """
    Descriptor for caching instance method results
    """

    def __init__(
        self,
        cache: Optional[CacheInterface] = None,
        ttl: Optional[int] = None,
        cache_type: CacheType = CacheType.MEMORY,
        cache_name: str = "method_cache",
        include_self: bool = False,
        **cache_kwargs,
    ):
        self.cache = cache
        self.ttl = ttl
        self.cache_type = cache_type
        self.cache_name = cache_name
        self.include_self = include_self
        self.cache_kwargs = cache_kwargs
        self.func = None
        self.cache_instance = None

    def __call__(self, func: Callable) -> "cached_method":
        self.func = func
        # Get or create cache instance
        if self.cache is not None:
            self.cache_instance = self.cache
        else:
            self.cache_instance = CacheFactory.get_cache(
                name=self.cache_name, cache_type=self.cache_type, **self.cache_kwargs
            )
        return self

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        @functools.wraps(self.func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if self.include_self:
                cache_args = (obj,) + args
            else:
                cache_args = args

            cache_key = _generate_cache_key(
                func=self.func,
                args=cache_args,
                kwargs=kwargs,
                key_prefix=f"method:{id(obj)}:",
                include_args=True,
                include_kwargs=True,
                exclude_args=[],
                exclude_kwargs=[],
            )

            # Try to get from cache
            cached_result = self.cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute method and cache result
            result = self.func(obj, *args, **kwargs)
            self.cache_instance.set(cache_key, result, self.ttl)

            return result

        return wrapper
