"""
FinSight Common Module

This module provides shared utilities and components for the FinSight platform,
including logging, caching, LLM integrations, and other common functionality.

Usage:
    from common.logger import LoggerFactory
    from common.cache import CacheFactory
    from common.llm import LLMFacade
"""

__version__ = "0.1.0"
__author__ = "FinSight Team"
__email__ = "team@finsight.com"

# Import key components for easy access
try:
    from .cache import CacheFactory, CacheType
    from .llm import LLMFacade
    from .logger import LoggerFactory, LoggerType, LogLevel
except ImportError as e:
    # Handle graceful degradation if some dependencies are missing
    import warnings

    warnings.warn(f"Some common modules could not be imported: {e}", ImportWarning)

__all__ = [
    "__version__",
    "LoggerFactory",
    "LoggerType",
    "LogLevel",
    "CacheFactory",
    "CacheType",
    "LLMFacade",
]
