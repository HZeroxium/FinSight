# converters/ohlcv_converter.py

"""
OHLCV data converter for converting between schemas and models.

This module provides conversion utilities between the logical OHLCV schema
and various database-specific models, enabling reusability across different
storage implementations.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, TypeVar, Union

from ..schemas.ohlcv_schemas import OHLCVSchema
from ..models.ohlcv_models import (
    OHLCVModelMongoDB,
    OHLCVModelInfluxDB,
    OHLCVModelCSV,
    OHLCVModelTimeScaleDB,
    OHLCVModelGeneric,
)
from common.logger import LoggerFactory

# Type variables for generic typing
T = TypeVar("T")
ModelType = Union[
    OHLCVModelMongoDB,
    OHLCVModelInfluxDB,
    OHLCVModelCSV,
    OHLCVModelTimeScaleDB,
    OHLCVModelGeneric,
]


class OHLCVConverter:
    """
    Converter class for OHLCV data between schemas and models.

    Provides bidirectional conversion between the logical OHLCV schema
    and various database-specific models.
    """

    def __init__(self):
        """Initialize the OHLCV converter."""
        self.logger = LoggerFactory.get_logger(name="ohlcv_converter")

    # Schema to Model conversions

    def schema_to_mongodb_model(self, schema: OHLCVSchema) -> OHLCVModelMongoDB:
        """
        Convert OHLCV schema to MongoDB model.

        Args:
            schema: OHLCV schema instance

        Returns:
            MongoDB model instance
        """
        return OHLCVModelMongoDB(
            timestamp=self._ensure_utc_timezone(schema.timestamp),
            open=schema.open,
            high=schema.high,
            low=schema.low,
            close=schema.close,
            volume=schema.volume,
            symbol=schema.symbol,
            exchange=schema.exchange,
            timeframe=schema.timeframe,
        )

    def schema_to_influxdb_model(self, schema: OHLCVSchema) -> OHLCVModelInfluxDB:
        """
        Convert OHLCV schema to InfluxDB model.

        Args:
            schema: OHLCV schema instance

        Returns:
            InfluxDB model instance
        """
        return OHLCVModelInfluxDB(
            timestamp=self._ensure_utc_timezone(schema.timestamp),
            open=schema.open,
            high=schema.high,
            low=schema.low,
            close=schema.close,
            volume=schema.volume,
            symbol=schema.symbol,
            exchange=schema.exchange,
            timeframe=schema.timeframe,
        )

    def schema_to_csv_model(self, schema: OHLCVSchema) -> OHLCVModelCSV:
        """
        Convert OHLCV schema to CSV model.

        Args:
            schema: OHLCV schema instance

        Returns:
            CSV model instance
        """
        return OHLCVModelCSV(
            timestamp=self._ensure_utc_timezone(schema.timestamp),
            open=schema.open,
            high=schema.high,
            low=schema.low,
            close=schema.close,
            volume=schema.volume,
            symbol=schema.symbol,
            exchange=schema.exchange,
            timeframe=schema.timeframe,
        )

    def schema_to_timescaledb_model(self, schema: OHLCVSchema) -> OHLCVModelTimeScaleDB:
        """
        Convert OHLCV schema to TimeScaleDB model.

        Args:
            schema: OHLCV schema instance

        Returns:
            TimeScaleDB model instance
        """
        return OHLCVModelTimeScaleDB(
            timestamp=self._ensure_utc_timezone(schema.timestamp),
            open=schema.open,
            high=schema.high,
            low=schema.low,
            close=schema.close,
            volume=schema.volume,
            symbol=schema.symbol,
            exchange=schema.exchange,
            timeframe=schema.timeframe,
        )

    def schema_to_generic_model(self, schema: OHLCVSchema) -> OHLCVModelGeneric:
        """
        Convert OHLCV schema to generic model.

        Args:
            schema: OHLCV schema instance

        Returns:
            Generic model instance
        """
        return OHLCVModelGeneric(
            timestamp=self._ensure_utc_timezone(schema.timestamp),
            open=schema.open,
            high=schema.high,
            low=schema.low,
            close=schema.close,
            volume=schema.volume,
            symbol=schema.symbol,
            exchange=schema.exchange,
            timeframe=schema.timeframe,
        )

    # Model to Schema conversions

    def mongodb_model_to_schema(self, model: OHLCVModelMongoDB) -> OHLCVSchema:
        """
        Convert MongoDB model to OHLCV schema.

        Args:
            model: MongoDB model instance

        Returns:
            OHLCV schema instance
        """
        return OHLCVSchema(
            timestamp=self._ensure_utc_timezone(model.timestamp),
            open=model.open,
            high=model.high,
            low=model.low,
            close=model.close,
            volume=model.volume,
            symbol=model.symbol,
            exchange=model.exchange,
            timeframe=model.timeframe,
        )

    def influxdb_model_to_schema(self, model: OHLCVModelInfluxDB) -> OHLCVSchema:
        """
        Convert InfluxDB model to OHLCV schema.

        Args:
            model: InfluxDB model instance

        Returns:
            OHLCV schema instance
        """
        return OHLCVSchema(
            timestamp=self._ensure_utc_timezone(model.timestamp),
            open=model.open,
            high=model.high,
            low=model.low,
            close=model.close,
            volume=model.volume,
            symbol=model.symbol,
            exchange=model.exchange,
            timeframe=model.timeframe,
        )

    def csv_model_to_schema(self, model: OHLCVModelCSV) -> OHLCVSchema:
        """
        Convert CSV model to OHLCV schema.

        Args:
            model: CSV model instance

        Returns:
            OHLCV schema instance
        """
        return OHLCVSchema(
            timestamp=self._ensure_utc_timezone(model.timestamp),
            open=model.open,
            high=model.high,
            low=model.low,
            close=model.close,
            volume=model.volume,
            symbol=model.symbol,
            exchange=model.exchange,
            timeframe=model.timeframe,
        )

    def timescaledb_model_to_schema(self, model: OHLCVModelTimeScaleDB) -> OHLCVSchema:
        """
        Convert TimeScaleDB model to OHLCV schema.

        Args:
            model: TimeScaleDB model instance

        Returns:
            OHLCV schema instance
        """
        return OHLCVSchema(
            timestamp=self._ensure_utc_timezone(model.timestamp),
            open=model.open,
            high=model.high,
            low=model.low,
            close=model.close,
            volume=model.volume,
            symbol=model.symbol,
            exchange=model.exchange,
            timeframe=model.timeframe,
        )

    def generic_model_to_schema(self, model: OHLCVModelGeneric) -> OHLCVSchema:
        """
        Convert generic model to OHLCV schema.

        Args:
            model: Generic model instance

        Returns:
            OHLCV schema instance
        """
        return OHLCVSchema(
            timestamp=self._ensure_utc_timezone(model.timestamp),
            open=model.open,
            high=model.high,
            low=model.low,
            close=model.close,
            volume=model.volume,
            symbol=model.symbol,
            exchange=model.exchange,
            timeframe=model.timeframe,
        )

    # Batch conversions

    def schemas_to_models(
        self, schemas: List[OHLCVSchema], model_type: str
    ) -> List[ModelType]:
        """
        Convert a list of OHLCV schemas to models of specified type.

        Args:
            schemas: List of OHLCV schema instances
            model_type: Target model type ('mongodb', 'influxdb', 'csv', 'timescaledb', 'generic')

        Returns:
            List of model instances

        Raises:
            ValueError: If model_type is not supported
        """
        conversion_map = {
            "mongodb": self.schema_to_mongodb_model,
            "influxdb": self.schema_to_influxdb_model,
            "csv": self.schema_to_csv_model,
            "timescaledb": self.schema_to_timescaledb_model,
            "generic": self.schema_to_generic_model,
        }

        if model_type not in conversion_map:
            raise ValueError(f"Unsupported model type: {model_type}")

        converter_func = conversion_map[model_type]
        return [converter_func(schema) for schema in schemas]

    def models_to_schemas(self, models: List[ModelType]) -> List[OHLCVSchema]:
        """
        Convert a list of models to OHLCV schemas.

        Args:
            models: List of model instances

        Returns:
            List of OHLCV schema instances

        Raises:
            ValueError: If model type is not supported
        """
        schemas = []

        for model in models:
            if isinstance(model, OHLCVModelMongoDB):
                schemas.append(self.mongodb_model_to_schema(model))
            elif isinstance(model, OHLCVModelInfluxDB):
                schemas.append(self.influxdb_model_to_schema(model))
            elif isinstance(model, OHLCVModelCSV):
                schemas.append(self.csv_model_to_schema(model))
            elif isinstance(model, OHLCVModelTimeScaleDB):
                schemas.append(self.timescaledb_model_to_schema(model))
            elif isinstance(model, OHLCVModelGeneric):
                schemas.append(self.generic_model_to_schema(model))
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

        return schemas

    # Dictionary conversions for backward compatibility

    def dict_to_schema(self, data: Dict[str, Any]) -> OHLCVSchema:
        """
        Convert dictionary to OHLCV schema.

        Args:
            data: Dictionary with OHLCV data

        Returns:
            OHLCV schema instance
        """
        try:
            # Handle timestamp parsing
            timestamp = data.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif isinstance(timestamp, datetime):
                timestamp = self._ensure_utc_timezone(timestamp)
            elif timestamp is None:
                raise ValueError("Timestamp is required")

            return OHLCVSchema(
                timestamp=timestamp,
                open=float(data["open"]),
                high=float(data["high"]),
                low=float(data["low"]),
                close=float(data["close"]),
                volume=float(data["volume"]),
                symbol=str(data["symbol"]),
                exchange=str(data["exchange"]),
                timeframe=str(data["timeframe"]),
            )
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error converting dict to schema: {e}")
            raise ValueError(f"Invalid dictionary format for OHLCV data: {e}")

    def schema_to_dict(self, schema: OHLCVSchema) -> Dict[str, Any]:
        """
        Convert OHLCV schema to dictionary.

        Args:
            schema: OHLCV schema instance

        Returns:
            Dictionary representation
        """
        return {
            "timestamp": (
                schema.timestamp.isoformat() + "Z"
                if schema.timestamp.tzinfo
                else schema.timestamp.isoformat() + "+00:00"
            ),
            "open": schema.open,
            "high": schema.high,
            "low": schema.low,
            "close": schema.close,
            "volume": schema.volume,
            "symbol": schema.symbol,
            "exchange": schema.exchange,
            "timeframe": schema.timeframe,
        }

    def dicts_to_schemas(self, data_list: List[Dict[str, Any]]) -> List[OHLCVSchema]:
        """
        Convert list of dictionaries to OHLCV schemas.

        Args:
            data_list: List of dictionaries with OHLCV data

        Returns:
            List of OHLCV schema instances
        """
        return [self.dict_to_schema(data) for data in data_list]

    def schemas_to_dicts(self, schemas: List[OHLCVSchema]) -> List[Dict[str, Any]]:
        """
        Convert list of OHLCV schemas to dictionaries.

        Args:
            schemas: List of OHLCV schema instances

        Returns:
            List of dictionary representations
        """
        return [self.schema_to_dict(schema) for schema in schemas]

    # Utility methods

    def _ensure_utc_timezone(self, dt: datetime) -> datetime:
        """
        Ensure datetime has UTC timezone.

        Args:
            dt: Datetime instance

        Returns:
            UTC timezone-aware datetime
        """
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            return dt.astimezone(timezone.utc)
        return dt
