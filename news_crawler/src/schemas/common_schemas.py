"""
Common API schemas used across the service.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class HealthCheckSchema(BaseModel):
    """Health check response DTO."""

    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field("1.0.0", description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dependencies: Optional[Dict[str, str]] = Field(
        None, description="Dependency health status"
    )
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")


class ErrorResponseSchema(BaseModel):
    """Generic error response DTO."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class PaginationSchema(BaseModel):
    """Pagination parameters DTO."""

    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(10, ge=1, le=100, description="Items per page")
    total: Optional[int] = Field(None, description="Total items count")
    has_next: Optional[bool] = Field(None, description="Has next page")
    has_prev: Optional[bool] = Field(None, description="Has previous page")


class MetricsSchema(BaseModel):
    """Service metrics DTO."""

    requests_total: int = Field(0, description="Total requests processed")
    requests_success: int = Field(0, description="Successful requests")
    requests_error: int = Field(0, description="Failed requests")
    average_response_time: float = Field(
        0.0, description="Average response time in seconds"
    )
    active_connections: int = Field(0, description="Active connections")
    memory_usage_mb: float = Field(0.0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(0.0, description="CPU usage percentage")
