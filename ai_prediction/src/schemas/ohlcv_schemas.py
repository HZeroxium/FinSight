# schemas/ohlcv_schemas.py

"""
OHLCV API schemas for request/response DTOs.

These schemas represent the logical structure of OHLCV data
at the abstraction layer, independent of storage implementation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class OHLCVSchema(BaseModel):
    """
    Base OHLCV data schema for API layer.

    This represents the standardized format for OHLCV data
    across all parts of the system.
    """

    timestamp: datetime = Field(..., description="Timestamp of the candle in UTC")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price during the period")
    low: float = Field(..., gt=0, description="Lowest price during the period")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Volume traded during the period")
    symbol: str = Field(..., min_length=1, description="Trading symbol (e.g., BTCUSDT)")
    exchange: str = Field(
        ..., min_length=1, description="Exchange name (e.g., binance)"
    )
    timeframe: str = Field(..., min_length=1, description="Timeframe (e.g., 1h, 1d)")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo else v.isoformat() + "+00:00"
            )
        },
        validate_assignment=True,
        use_enum_values=True,
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate OHLCV data integrity after initialization."""
        if self.high < max(self.open, self.close):
            raise ValueError("High price must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("Low price must be <= min(open, close)")


class OHLCVBatchSchema(BaseModel):
    """Schema for batch OHLCV operations."""

    records: List[OHLCVSchema] = Field(
        ..., min_length=1, description="List of OHLCV records"
    )
    exchange: str = Field(..., description="Exchange name for all records")
    symbol: str = Field(..., description="Symbol for all records")
    timeframe: str = Field(..., description="Timeframe for all records")

    model_config = ConfigDict(validate_assignment=True)


class OHLCVQuerySchema(BaseModel):
    """Schema for querying OHLCV data."""

    exchange: str = Field(..., description="Exchange name")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    start_date: datetime = Field(..., description="Start date for query")
    end_date: datetime = Field(..., description="End date for query")
    limit: Optional[int] = Field(
        None, ge=1, le=10000, description="Maximum number of records to return"
    )

    model_config = ConfigDict(validate_assignment=True)


class OHLCVResponseSchema(BaseModel):
    """Schema for OHLCV API responses."""

    data: List[OHLCVSchema] = Field(..., description="OHLCV records")
    count: int = Field(..., ge=0, description="Number of records returned")
    exchange: str = Field(..., description="Exchange name")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    start_date: datetime = Field(..., description="Query start date")
    end_date: datetime = Field(..., description="Query end date")
    has_more: bool = Field(..., description="Whether more data is available")

    model_config = ConfigDict(validate_assignment=True)


class OHLCVStatsSchema(BaseModel):
    """Schema for OHLCV statistics."""

    exchange: str = Field(..., description="Exchange name")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    total_records: int = Field(..., ge=0, description="Total number of records")
    date_range: Dict[str, datetime] = Field(
        ..., description="Date range with 'start' and 'end' keys"
    )
    price_range: Dict[str, float] = Field(
        ..., description="Price range with 'min' and 'max' keys"
    )
    volume_stats: Dict[str, float] = Field(
        ..., description="Volume statistics with 'min', 'max', 'avg' keys"
    )

    model_config = ConfigDict(validate_assignment=True)
