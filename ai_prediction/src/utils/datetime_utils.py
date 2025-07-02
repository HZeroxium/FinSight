"""
Centralized datetime utilities for consistent datetime handling across the system.

This module provides standardized functions for:
- Parsing and formatting datetime strings
- Converting between different datetime formats
- Validating datetime inputs
- Working with timezones consistently
"""

from datetime import datetime, timezone, timedelta
from typing import Union, Optional, Tuple
import pandas as pd

# Standard ISO 8601 format used throughout the system
ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
ISO_FORMAT_WITH_MS = "%Y-%m-%dT%H:%M:%S.%fZ"


class DateTimeUtils:
    """Centralized datetime utilities for the market data system."""

    @staticmethod
    def now_utc() -> datetime:
        """Get current UTC datetime with timezone info."""
        return datetime.now(timezone.utc)

    @staticmethod
    def now_iso() -> str:
        """Get current UTC datetime as ISO 8601 string."""
        return DateTimeUtils.now_utc().strftime(ISO_FORMAT)

    @staticmethod
    def to_utc_datetime(dt_input: Union[str, datetime, pd.Timestamp]) -> datetime:
        """
        Convert various datetime inputs to UTC datetime object.

        Args:
            dt_input: String, datetime, or pandas Timestamp

        Returns:
            UTC datetime object with timezone info

        Raises:
            ValueError: If input format is invalid
        """
        if isinstance(dt_input, str):
            return DateTimeUtils.parse_iso_string(dt_input)
        elif isinstance(dt_input, pd.Timestamp):
            if dt_input.tz is None:
                # Assume UTC if no timezone
                return dt_input.tz_localize("UTC").to_pydatetime()
            else:
                return dt_input.tz_convert("UTC").to_pydatetime()
        elif isinstance(dt_input, datetime):
            if dt_input.tzinfo is None:
                # Assume UTC if no timezone
                return dt_input.replace(tzinfo=timezone.utc)
            else:
                return dt_input.astimezone(timezone.utc)
        else:
            raise ValueError(f"Unsupported datetime input type: {type(dt_input)}")

    @staticmethod
    def to_iso_string(dt_input: Union[str, datetime, pd.Timestamp]) -> str:
        """
        Convert various datetime inputs to ISO 8601 string.

        Args:
            dt_input: String, datetime, or pandas Timestamp

        Returns:
            ISO 8601 formatted string (YYYY-MM-DDTHH:MM:SSZ)
        """
        utc_dt = DateTimeUtils.to_utc_datetime(dt_input)
        return utc_dt.strftime(ISO_FORMAT)

    @staticmethod
    def parse_iso_string(iso_string: str) -> datetime:
        """
        Parse ISO 8601 string to UTC datetime object.

        Args:
            iso_string: ISO 8601 formatted string

        Returns:
            UTC datetime object with timezone info

        Raises:
            ValueError: If string format is invalid
        """
        # Handle various ISO formats
        iso_string = iso_string.strip()

        try:
            # Try with Z suffix
            if iso_string.endswith("Z"):
                clean_string = iso_string[:-1]
                try:
                    # Try with microseconds
                    dt = datetime.fromisoformat(clean_string)
                except ValueError:
                    # Try without microseconds
                    if "." in clean_string:
                        clean_string = clean_string.split(".")[0]
                    dt = datetime.fromisoformat(clean_string)
                return dt.replace(tzinfo=timezone.utc)

            # Try with timezone info
            elif "+" in iso_string or iso_string.endswith(("Z", "+00:00")):
                dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc)

            # Assume UTC if no timezone info
            else:
                dt = datetime.fromisoformat(iso_string)
                return dt.replace(tzinfo=timezone.utc)

        except ValueError as e:
            raise ValueError(
                f"Invalid ISO 8601 datetime format: {iso_string}. Error: {e}"
            )

    @staticmethod
    def validate_date_range(
        start_date: str, end_date: str
    ) -> Tuple[datetime, datetime]:
        """
        Validate and parse date range strings.

        Args:
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format

        Returns:
            Tuple of (start_datetime, end_datetime) in UTC

        Raises:
            ValueError: If dates are invalid or start >= end
        """
        try:
            start_dt = DateTimeUtils.parse_iso_string(start_date)
            end_dt = DateTimeUtils.parse_iso_string(end_date)

            if start_dt >= end_dt:
                raise ValueError("Start date must be before end date")

            return start_dt, end_dt

        except ValueError as e:
            raise ValueError(f"Invalid date range: {e}")

    @staticmethod
    def parse_timeframe_to_timedelta(timeframe: str) -> Optional[timedelta]:
        """
        Convert timeframe string to timedelta object.

        Args:
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')

        Returns:
            timedelta object or None if invalid
        """
        timeframe_map = {
            "1m": timedelta(minutes=1),
            "3m": timedelta(minutes=3),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "2h": timedelta(hours=2),
            "4h": timedelta(hours=4),
            "6h": timedelta(hours=6),
            "8h": timedelta(hours=8),
            "12h": timedelta(hours=12),
            "1d": timedelta(days=1),
            "3d": timedelta(days=3),
            "1w": timedelta(weeks=1),
            "1M": timedelta(days=30),  # Approximate
        }
        return timeframe_map.get(timeframe)

    @staticmethod
    def generate_expected_timestamps(
        start_date: str, end_date: str, timeframe: str
    ) -> list[datetime]:
        """
        Generate expected timestamps for a date range and timeframe.

        Args:
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format
            timeframe: Timeframe string

        Returns:
            List of expected datetime objects
        """
        start_dt, end_dt = DateTimeUtils.validate_date_range(start_date, end_date)
        interval = DateTimeUtils.parse_timeframe_to_timedelta(timeframe)

        if not interval:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        timestamps = []
        current = start_dt
        while current <= end_dt:
            timestamps.append(current)
            current += interval

        return timestamps

    @staticmethod
    def format_timestamp_for_exchange(dt: datetime, exchange: str = "binance") -> str:
        """
        Format datetime for specific exchange API requirements.

        Args:
            dt: datetime object
            exchange: Exchange name

        Returns:
            Formatted timestamp string
        """
        if exchange.lower() == "binance":
            # Binance expects format like "1 Jan 2020"
            return dt.strftime("%d %b %Y %H:%M:%S")
        else:
            # Default to ISO format
            return DateTimeUtils.to_iso_string(dt)

    @staticmethod
    def calculate_time_range(
        days_back: int, end_date: Optional[datetime] = None
    ) -> Tuple[datetime, datetime]:
        """
        Calculate time range for historical data fetching.

        Args:
            days_back: Number of days to go back
            end_date: End date (default: now)

        Returns:
            Tuple of (start_time, end_time)
        """
        if end_date is None:
            end_date = DateTimeUtils.now_utc()

        start_date = end_date - timedelta(days=days_back)
        return start_date, end_date

    @staticmethod
    def to_pandas_timestamp(dt_input: Union[str, datetime]) -> pd.Timestamp:
        """
        Convert datetime input to pandas Timestamp with UTC timezone.

        Args:
            dt_input: String or datetime object

        Returns:
            pandas Timestamp with UTC timezone
        """
        utc_dt = DateTimeUtils.to_utc_datetime(dt_input)
        return pd.Timestamp(utc_dt, tz="UTC")

    @staticmethod
    def from_timestamp_ms(timestamp_ms: int) -> datetime:
        """
        Convert millisecond timestamp to UTC datetime.

        Args:
            timestamp_ms: Timestamp in milliseconds

        Returns:
            UTC datetime object
        """
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

    @staticmethod
    def to_timestamp_ms(dt: datetime) -> int:
        """
        Convert datetime to millisecond timestamp.

        Args:
            dt: datetime object

        Returns:
            Timestamp in milliseconds
        """
        return int(dt.timestamp() * 1000)
