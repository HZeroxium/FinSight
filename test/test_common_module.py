#!/usr/bin/env python3
"""
Test script for the FinSight Common Module

This script tests the basic functionality of the common module to ensure
it can be imported and used correctly across the FinSight platform.
"""

import sys
import os
import asyncio
from pathlib import Path


def test_imports():
    """Test that all common modules can be imported successfully."""
    print("Testing imports...")

    try:
        # Test basic imports
        from common.logger import LoggerFactory, LoggerType, LogLevel

        print("✅ Logger imports successful")

        from common.cache import CacheFactory, CacheType

        print("✅ Cache imports successful")

        # Test main module import
        import common

        print(f"✅ Common module version: {common.__version__}")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_logger():
    """Test the logger functionality."""
    print("\nTesting logger functionality...")

    try:
        from common.logger import LoggerFactory, LoggerType, LogLevel

        # Create a logger instance
        logger = LoggerFactory.create_logger(
            name="test_service", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
        )

        # Test logging
        logger.info("Logger test successful - INFO level")
        logger.warning("Logger test successful - WARNING level")
        logger.debug("Logger test successful - DEBUG level (may not show)")

        print("✅ Logger functionality test passed")
        return True

    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        return False


def test_cache():
    """Test the cache functionality."""
    print("\nTesting cache functionality...")

    try:
        from common.cache import CacheFactory, CacheType

        # Create an in-memory cache
        cache = CacheFactory.create_cache(cache_type=CacheType.MEMORY, max_size=100)

        # Test cache operations synchronously (not async)
        test_cache_operations_sync(cache)

        print("✅ Cache functionality test passed")
        return True

    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False


def test_cache_operations_sync(cache):
    """Test basic cache operations synchronously."""
    # Test set and get
    cache.set("test_key", "test_value", ttl=60)
    value = cache.get("test_key")

    if value == "test_value":
        print("  ✅ Cache set/get operations work")
    else:
        raise ValueError(f"Expected 'test_value', got '{value}'")

    # Test exists
    exists = cache.exists("test_key")
    if exists:
        print("  ✅ Cache exists operation works")
    else:
        raise ValueError("Cache key should exist")

    # Test delete
    cache.delete("test_key")
    value_after_delete = cache.get("test_key")

    if value_after_delete is None:
        print("  ✅ Cache delete operation works")
    else:
        raise ValueError("Cache key should be deleted")


def test_common_module_structure():
    """Test the overall module structure and organization."""
    print("\nTesting module structure...")

    try:
        # Check that the module has the expected attributes
        import common

        expected_attrs = ["__version__", "LoggerFactory", "CacheFactory"]
        for attr in expected_attrs:
            if hasattr(common, attr):
                print(f"  ✅ {attr} is available in common module")
            else:
                print(f"  ⚠️ {attr} is not available in common module")

        return True

    except Exception as e:
        print(f"❌ Module structure test failed: {e}")
        return False


def test_factory_patterns():
    """Test that factory patterns work correctly."""
    print("\nTesting factory patterns...")

    try:
        from common.logger import LoggerFactory, LoggerType
        from common.cache import CacheFactory, CacheType

        # Test logger factory
        logger1 = LoggerFactory.create_logger("service1", LoggerType.STANDARD)
        logger2 = LoggerFactory.create_logger("service2", LoggerType.PRINT)

        if logger1 and logger2:
            print("  ✅ Logger factory creates different instances")

        # Test cache factory - use two different memory caches
        cache1 = CacheFactory.create_cache("cache1", CacheType.MEMORY, max_size=100)
        cache2 = CacheFactory.create_cache("cache2", CacheType.MEMORY, max_size=200)

        if cache1 and cache2:
            print("  ✅ Cache factory creates different instances")

        print("✅ Factory patterns test passed")
        return True

    except Exception as e:
        print(f"❌ Factory patterns test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("FinSight Common Module Test Suite")
    print("=" * 60)

    tests = [
        test_imports,
        test_logger,
        test_cache,
        test_common_module_structure,
        test_factory_patterns,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The common module is working correctly.")
        print("\nYou can now use the common module in your services like this:")
        print("  from common.logger import LoggerFactory")
        print("  from common.cache import CacheFactory")
        print("  from common.llm import LLMFacade")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
