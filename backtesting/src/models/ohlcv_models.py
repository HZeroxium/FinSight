# models/ohlcv_models.py

"""
OHLCV data models for different database implementations.

These models represent the structure of OHLCV data as stored
in specific database systems (MongoDB, InfluxDB, CSV, etc.).
"""

from datetime import datetime, timezone
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2 MongoDB integration."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(ObjectId),
                    core_schema.chain_schema(
                        [
                            core_schema.str_schema(),
                            core_schema.no_info_plain_validator_function(cls.validate),
                        ]
                    ),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )

    @classmethod
    def validate(cls, v: Any) -> ObjectId:
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")


class OHLCVModelMongoDB(BaseModel):
    """MongoDB model for OHLCV data storage."""

    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    timestamp: datetime = Field(..., description="Timestamp of the candle in UTC")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price during the period")
    low: float = Field(..., gt=0, description="Lowest price during the period")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Volume traded during the period")
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    exchange: str = Field(..., min_length=1, description="Exchange name")
    timeframe: str = Field(..., min_length=1, description="Timeframe")

    # MongoDB specific fields
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation time",
    )
    updated_at: Optional[datetime] = Field(None, description="Record last update time")

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
        validate_assignment=True,
    )


class OHLCVModelInfluxDB(BaseModel):
    """InfluxDB model for OHLCV data storage."""

    # InfluxDB stores timestamp as the primary key
    timestamp: datetime = Field(..., description="Timestamp of the candle in UTC")

    # Fields (numeric values)
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price during the period")
    low: float = Field(..., gt=0, description="Lowest price during the period")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Volume traded during the period")

    # Tags (indexed string values)
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    exchange: str = Field(..., min_length=1, description="Exchange name")
    timeframe: str = Field(..., min_length=1, description="Timeframe")

    model_config = ConfigDict(validate_assignment=True)


class OHLCVModelCSV(BaseModel):
    """CSV model for OHLCV data storage."""

    timestamp: datetime = Field(..., description="Timestamp of the candle in UTC")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price during the period")
    low: float = Field(..., gt=0, description="Lowest price during the period")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Volume traded during the period")
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    exchange: str = Field(..., min_length=1, description="Exchange name")
    timeframe: str = Field(..., min_length=1, description="Timeframe")

    model_config = ConfigDict(validate_assignment=True)


class OHLCVModelTimeScaleDB(BaseModel):
    """TimeScaleDB/PostgreSQL model for OHLCV data storage."""

    id: Optional[int] = Field(None, description="Auto-increment primary key")
    timestamp: datetime = Field(..., description="Timestamp of the candle in UTC")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price during the period")
    low: float = Field(..., gt=0, description="Lowest price during the period")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Volume traded during the period")
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    exchange: str = Field(..., min_length=1, description="Exchange name")
    timeframe: str = Field(..., min_length=1, description="Timeframe")

    # TimeScaleDB specific fields
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation time",
    )

    model_config = ConfigDict(validate_assignment=True)


class OHLCVModelGeneric(BaseModel):
    """Generic model for OHLCV data that can be used with any storage system."""

    timestamp: datetime = Field(..., description="Timestamp of the candle in UTC")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price during the period")
    low: float = Field(..., gt=0, description="Lowest price during the period")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Volume traded during the period")
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    exchange: str = Field(..., min_length=1, description="Exchange name")
    timeframe: str = Field(..., min_length=1, description="Timeframe")

    # Optional metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    model_config = ConfigDict(validate_assignment=True)
