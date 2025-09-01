# utils/rate_limiting.py

"""
Rate limiting utilities for the news service.
Provides client identification, rate limit configuration, and helper functions.
"""

import re
from typing import Optional

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from common.logger import LoggerFactory, LoggerType, LogLevel

from ..core.config import settings


class RateLimitUtils:
    """Utility class for rate limiting configuration and client identification."""

    def __init__(self):
        """Initialize rate limiting utilities."""
        self.logger = LoggerFactory.get_logger(
            name="rate-limit-utils",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file=f"{settings.log_file_path}rate_limiting.log",
        )

    def get_client_identifier(self, request: Request) -> str:
        """
        Get a unique identifier for the client based on configuration.

        Priority order:
        1. API key (if enabled and available)
        2. Real IP address (if enabled)
        3. Fallback to default identification

        Args:
            request: FastAPI request object

        Returns:
            str: Unique client identifier
        """
        try:
            # Try to get API key first if enabled
            if settings.rate_limit_by_api_key:
                api_key = self._extract_api_key(request)
                if api_key:
                    self.logger.debug(
                        f"Using API key for client identification: {api_key[:8]}..."
                    )
                    return f"api_key:{api_key}"

            # Fall back to IP address if enabled
            if settings.rate_limit_by_ip:
                real_ip = self._get_real_ip_address(request)
                self.logger.debug(
                    f"Using IP address for client identification: {real_ip}"
                )
                return f"ip:{real_ip}"

            # Fallback to default identification
            fallback_id = get_remote_address(request)
            self.logger.debug(f"Using fallback identification: {fallback_id}")
            return f"fallback:{fallback_id}"

        except Exception as e:
            self.logger.error(f"Error getting client identifier: {e}")
            # Return a safe fallback
            return "unknown:fallback"

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """
        Extract API key from request headers or query parameters.

        Args:
            request: FastAPI request object

        Returns:
            Optional[str]: API key if found, None otherwise
        """
        try:
            # Check Authorization header (Bearer token)
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                api_key = auth_header[7:]  # Remove "Bearer " prefix
                if api_key and len(api_key) > 0:
                    return api_key

            # Check X-API-Key header
            api_key_header = request.headers.get("X-API-Key")
            if api_key_header and len(api_key_header) > 0:
                return api_key_header

            # Check query parameter
            api_key_param = request.query_params.get("api_key")
            if api_key_param and len(api_key_param) > 0:
                return api_key_param

            return None

        except Exception as e:
            self.logger.error(f"Error extracting API key: {e}")
            return None

    def _get_real_ip_address(self, request: Request) -> str:
        """
        Get the real IP address of the client, handling proxy headers.

        Args:
            request: FastAPI request object

        Returns:
            str: Real IP address
        """
        try:
            if not settings.rate_limit_trust_proxy:
                # Don't trust proxy headers, use direct client IP
                return request.client.host if request.client else "unknown"

            # Check X-Forwarded-For header (most common)
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                # X-Forwarded-For can contain multiple IPs: "client, proxy1, proxy2"
                # The first IP is the original client
                client_ip = forwarded_for.split(",")[0].strip()
                if self._is_valid_ip(client_ip):
                    return client_ip

            # Check X-Real-IP header
            real_ip = request.headers.get("X-Real-IP")
            if real_ip and self._is_valid_ip(real_ip):
                return real_ip

            # Check X-Client-IP header
            client_ip = request.headers.get("X-Client-IP")
            if client_ip and self._is_valid_ip(client_ip):
                return client_ip

            # Fallback to direct client IP
            return request.client.host if request.client else "unknown"

        except Exception as e:
            self.logger.error(f"Error getting real IP address: {e}")
            return "unknown"

    def _is_valid_ip(self, ip: str) -> bool:
        """
        Validate if a string is a valid IP address.

        Args:
            ip: IP address string to validate

        Returns:
            bool: True if valid IP, False otherwise
        """
        try:
            # Simple IP validation regex
            ipv4_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
            ipv6_pattern = r"^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"

            if re.match(ipv4_pattern, ip) or re.match(ipv6_pattern, ip):
                return True

            # Check for localhost variants
            if ip.lower() in ["localhost", "127.0.0.1", "::1"]:
                return True

            return False

        except Exception:
            return False

    def is_endpoint_exempt(self, request: Request) -> bool:
        """
        Check if an endpoint is exempt from rate limiting.

        Args:
            request: FastAPI request object

        Returns:
            bool: True if exempt, False otherwise
        """
        try:
            path = request.url.path

            # Check exact matches
            if path in settings.rate_limit_exempt_endpoints:
                return True

            # Check pattern matches (for dynamic routes)
            for exempt_pattern in settings.rate_limit_exempt_endpoints:
                if exempt_pattern.endswith("*"):
                    # Wildcard pattern
                    base_pattern = exempt_pattern[:-1]
                    if path.startswith(base_pattern):
                        return True
                elif exempt_pattern.startswith("*"):
                    # Suffix pattern
                    suffix_pattern = exempt_pattern[1:]
                    if path.endswith(suffix_pattern):
                        return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking endpoint exemption: {e}")
            return False

    def get_rate_limit_string(
        self, per_minute: int, per_hour: int, per_day: Optional[int] = None
    ) -> str:
        """
        Generate rate limit string for slowapi.

        Args:
            per_minute: Requests per minute
            per_hour: Requests per hour
            per_day: Requests per day (optional)

        Returns:
            str: Rate limit string in slowapi format
        """
        try:
            limits = [f"{per_minute}/minute", f"{per_hour}/hour"]

            if per_day:
                limits.append(f"{per_day}/day")

            return "; ".join(limits)

        except Exception as e:
            self.logger.error(f"Error generating rate limit string: {e}")
            return "100/minute; 1000/hour"

    def get_default_rate_limits(self) -> str:
        """
        Get default rate limits string.

        Returns:
            str: Default rate limits string
        """
        return self.get_rate_limit_string(
            per_minute=settings.rate_limit_requests_per_minute,
            per_hour=settings.rate_limit_requests_per_hour,
            per_day=settings.rate_limit_requests_per_day,
        )

    def get_news_search_rate_limits(self) -> str:
        """
        Get rate limits string for news search endpoints.

        Returns:
            str: News search rate limits string
        """
        return self.get_rate_limit_string(
            per_minute=settings.rate_limit_news_search_per_minute,
            per_hour=settings.rate_limit_news_search_per_hour,
        )

    def get_admin_rate_limits(self) -> str:
        """
        Get rate limits string for admin endpoints.

        Returns:
            str: Admin rate limits string
        """
        return self.get_rate_limit_string(
            per_minute=settings.rate_limit_admin_per_minute,
            per_hour=settings.rate_limit_admin_per_hour,
        )

    def get_cache_rate_limits(self) -> str:
        """
        Get rate limits string for cache management endpoints.

        Returns:
            str: Cache rate limits string
        """
        return self.get_rate_limit_string(
            per_minute=settings.rate_limit_cache_per_minute,
            per_hour=settings.rate_limit_cache_per_hour,
        )


def create_limiter() -> Limiter:
    """
    Create and configure the slowapi Limiter instance.

    Returns:
        Limiter: Configured limiter instance
    """
    try:
        # Create limiter with Redis backend
        limiter = Limiter(
            key_func=RateLimitUtils().get_client_identifier,
            default_limits=RateLimitUtils().get_default_rate_limits(),
            storage_uri=settings.rate_limit_storage_url,
            # strategy="fixed-window-elastic-expiry",
            headers_enabled=settings.rate_limit_include_headers,
            retry_after=settings.rate_limit_retry_after_header,
        )

        return limiter

    except Exception as e:
        # Fallback to memory storage if Redis is not available
        logger = LoggerFactory.get_logger(
            name="rate-limit-fallback",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.WARNING,
        )
        logger.warning(
            f"Failed to create Redis-based limiter, falling back to memory: {e}"
        )

        limiter = Limiter(
            key_func=RateLimitUtils().get_client_identifier,
            default_limits=RateLimitUtils().get_default_rate_limits(),
            # strategy="fixed-window-elastic-expiry",
            headers_enabled=settings.rate_limit_include_headers,
            retry_after=settings.rate_limit_retry_after_header,
        )

        return limiter


def get_rate_limit_decorator(rate_limit_string: str):
    """
    Create a rate limit decorator with custom limits.

    Args:
        rate_limit_string: Rate limit string (e.g., "60/minute; 500/hour")

    Returns:
        Callable: Decorator function
    """

    def decorator(func):
        """Rate limit decorator."""
        # This will be applied by the limiter instance
        func._rate_limit = rate_limit_string
        return func

    return decorator


# Global instances
rate_limit_utils = RateLimitUtils()
limiter = create_limiter()
