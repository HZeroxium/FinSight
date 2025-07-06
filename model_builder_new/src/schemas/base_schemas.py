# schemas/base_schemas.py

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class BaseResponse(BaseModel):
    """Base response schema for all API responses"""

    model_config = ConfigDict(validate_assignment=True)

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable message")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ErrorResponse(BaseResponse):
    """Error response schema"""

    model_config = ConfigDict(validate_assignment=True)

    success: bool = Field(False, description="Always False for error responses")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    error_details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )


class HealthResponse(BaseModel):
    """Health check response schema"""

    model_config = ConfigDict(validate_assignment=True)

    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="Service version")
    dependencies: Optional[Dict[str, str]] = Field(
        None, description="Status of dependencies"
    )


class ModelInfoResponse(BaseModel):
    """Model information response schema"""

    model_config = ConfigDict(validate_assignment=True)

    available_models: list[str] = Field(
        ..., description="List of available model types"
    )
    trained_models: Dict[str, Dict[str, Any]] = Field(
        ..., description="Information about trained models"
    )
    supported_timeframes: list[str] = Field(..., description="Supported timeframes")
    supported_symbols: list[str] = Field(..., description="Available symbols")
