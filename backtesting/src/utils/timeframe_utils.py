# utils/timeframe_utils.py

"""
Timeframe utility functions for handling timeframe-related operations.

This module provides utility functions for:
- Validating timeframes
- Converting timeframe strings to time deltas
- Calculating expected timestamps for given timeframes
- Timeframe arithmetic operations
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple

from ..schemas.enums import TimeFrame, TimeFrameMultiplier
from common.logger import LoggerFactory


class TimeFrameUtils:
    """Utility class for timeframe-related operations"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger("TimeFrameUtils")
        self._multipliers = self._build_multiplier_map()

    def _build_multiplier_map(self) -> Dict[str, int]:
        """Build mapping of timeframe strings to minute multipliers"""
        return {
            TimeFrame.MINUTE_1.value: TimeFrameMultiplier.MINUTE_1.value,
            TimeFrame.MINUTE_3.value: TimeFrameMultiplier.MINUTE_3.value,
            TimeFrame.MINUTE_5.value: TimeFrameMultiplier.MINUTE_5.value,
            TimeFrame.MINUTE_15.value: TimeFrameMultiplier.MINUTE_15.value,
            TimeFrame.MINUTE_30.value: TimeFrameMultiplier.MINUTE_30.value,
            TimeFrame.HOUR_1.value: TimeFrameMultiplier.HOUR_1.value,
            TimeFrame.HOUR_2.value: TimeFrameMultiplier.HOUR_2.value,
            TimeFrame.HOUR_4.value: TimeFrameMultiplier.HOUR_4.value,
            TimeFrame.HOUR_6.value: TimeFrameMultiplier.HOUR_6.value,
            TimeFrame.HOUR_8.value: TimeFrameMultiplier.HOUR_8.value,
            TimeFrame.HOUR_12.value: TimeFrameMultiplier.HOUR_12.value,
            TimeFrame.DAY_1.value: TimeFrameMultiplier.DAY_1.value,
            TimeFrame.DAY_3.value: TimeFrameMultiplier.DAY_3.value,
            TimeFrame.WEEK_1.value: TimeFrameMultiplier.WEEK_1.value,
            TimeFrame.MONTH_1.value: TimeFrameMultiplier.MONTH_1.value,
        }

    def is_valid_timeframe(self, timeframe: str) -> bool:
        """
        Check if a timeframe string is valid.

        Args:
            timeframe: Timeframe string to validate

        Returns:
            True if valid, False otherwise
        """
        return timeframe in self._multipliers

    def parse_timeframe_to_minutes(self, timeframe: str) -> Optional[int]:
        """
        Parse timeframe string to minutes.

        Args:
            timeframe: Timeframe string (e.g., "1h", "4h", "1d")

        Returns:
            Number of minutes or None if invalid
        """
        return self._multipliers.get(timeframe)

    def parse_timeframe_to_timedelta(self, timeframe: str) -> Optional[timedelta]:
        """
        Parse timeframe string to timedelta object.

        Args:
            timeframe: Timeframe string

        Returns:
            timedelta object or None if invalid
        """
        minutes = self.parse_timeframe_to_minutes(timeframe)
        return timedelta(minutes=minutes) if minutes else None

    def get_timeframe_category(self, timeframe: str) -> str:
        """
        Get the category of a timeframe (minute, hour, day, week, month).

        Args:
            timeframe: Timeframe string

        Returns:
            Category string
        """
        if timeframe.endswith("m"):
            return "minute"
        elif timeframe.endswith("h"):
            return "hour"
        elif timeframe.endswith("d"):
            return "day"
        elif timeframe.endswith("w"):
            return "week"
        elif timeframe.endswith("M"):
            return "month"
        else:
            return "unknown"

    def compare_timeframes(self, tf1: str, tf2: str) -> int:
        """
        Compare two timeframes.

        Args:
            tf1: First timeframe
            tf2: Second timeframe

        Returns:
            -1 if tf1 < tf2, 0 if tf1 == tf2, 1 if tf1 > tf2
        """
        minutes1 = self.parse_timeframe_to_minutes(tf1)
        minutes2 = self.parse_timeframe_to_minutes(tf2)

        if minutes1 is None or minutes2 is None:
            raise ValueError(f"Invalid timeframe(s): {tf1}, {tf2}")

        if minutes1 < minutes2:
            return -1
        elif minutes1 > minutes2:
            return 1
        else:
            return 0

    def get_smaller_timeframes(self, timeframe: str) -> List[str]:
        """
        Get all timeframes smaller than the given timeframe.

        Args:
            timeframe: Reference timeframe

        Returns:
            List of smaller timeframes
        """
        target_minutes = self.parse_timeframe_to_minutes(timeframe)
        if target_minutes is None:
            return []

        smaller = []
        for tf, minutes in self._multipliers.items():
            if minutes < target_minutes:
                smaller.append(tf)

        return sorted(smaller, key=lambda x: self._multipliers[x])

    def get_larger_timeframes(self, timeframe: str) -> List[str]:
        """
        Get all timeframes larger than the given timeframe.

        Args:
            timeframe: Reference timeframe

        Returns:
            List of larger timeframes
        """
        target_minutes = self.parse_timeframe_to_minutes(timeframe)
        if target_minutes is None:
            return []

        larger = []
        for tf, minutes in self._multipliers.items():
            if minutes > target_minutes:
                larger.append(tf)

        return sorted(larger, key=lambda x: self._multipliers[x])

    def align_timestamp_to_timeframe(
        self, timestamp: datetime, timeframe: str
    ) -> datetime:
        """
        Align timestamp to timeframe boundary.

        Args:
            timestamp: Timestamp to align
            timeframe: Target timeframe

        Returns:
            Aligned timestamp
        """
        if not self.is_valid_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Ensure UTC timezone
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Align based on timeframe type
        if timeframe.endswith("m"):
            # For minutes, align to the start of the interval
            minutes = int(timeframe[:-1])
            aligned_minute = (timestamp.minute // minutes) * minutes
            return timestamp.replace(minute=aligned_minute, second=0, microsecond=0)

        elif timeframe.endswith("h"):
            # For hours, align to the start of the hour interval
            hours = int(timeframe[:-1])
            aligned_hour = (timestamp.hour // hours) * hours
            return timestamp.replace(
                hour=aligned_hour, minute=0, second=0, microsecond=0
            )

        elif timeframe == "1d":
            # For daily, align to start of day
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)

        elif timeframe == "3d":
            # For 3-day, align to start of 3-day period
            days_since_epoch = (
                timestamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
            ).days
            aligned_days = (days_since_epoch // 3) * 3
            aligned_date = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
                days=aligned_days
            )
            return aligned_date.replace(hour=0, minute=0, second=0, microsecond=0)

        elif timeframe == "1w":
            # For weekly, align to start of week (Monday)
            days_since_monday = timestamp.weekday()
            aligned_date = timestamp - timedelta(days=days_since_monday)
            return aligned_date.replace(hour=0, minute=0, second=0, microsecond=0)

        elif timeframe == "1M":
            # For monthly, align to start of month
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        else:
            raise ValueError(f"Unsupported timeframe for alignment: {timeframe}")

    def generate_timeframe_range(
        self, start_time: datetime, end_time: datetime, timeframe: str
    ) -> List[datetime]:
        """
        Generate a list of aligned timestamps for the given timeframe and range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            timeframe: Timeframe to generate intervals for

        Returns:
            List of aligned timestamps
        """
        if not self.is_valid_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe: {timeframe}")

        interval = self.parse_timeframe_to_timedelta(timeframe)
        if not interval:
            raise ValueError(f"Could not parse timeframe: {timeframe}")

        # Align start time to timeframe boundary
        current = self.align_timestamp_to_timeframe(start_time, timeframe)
        timestamps = []

        while current <= end_time:
            timestamps.append(current)
            current += interval

        return timestamps

    def calculate_expected_records(
        self, start_time: datetime, end_time: datetime, timeframe: str
    ) -> int:
        """
        Calculate the expected number of records for a given time range and timeframe.

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            timeframe: Timeframe

        Returns:
            Expected number of records
        """
        timestamps = self.generate_timeframe_range(start_time, end_time, timeframe)
        return len(timestamps)

    def find_gaps_in_data(
        self, timestamps: List[datetime], timeframe: str
    ) -> List[Tuple[datetime, datetime]]:
        """
        Find gaps in a list of timestamps for the given timeframe.

        Args:
            timestamps: List of actual timestamps
            timeframe: Expected timeframe

        Returns:
            List of (gap_start, gap_end) tuples
        """
        if not timestamps:
            return []

        if not self.is_valid_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe: {timeframe}")

        interval = self.parse_timeframe_to_timedelta(timeframe)
        if not interval:
            raise ValueError(f"Could not parse timeframe: {timeframe}")

        # Sort timestamps
        sorted_timestamps = sorted(timestamps)
        gaps = []

        for i in range(len(sorted_timestamps) - 1):
            current = sorted_timestamps[i]
            next_ts = sorted_timestamps[i + 1]
            expected_next = current + interval

            if next_ts > expected_next:
                # There's a gap
                gaps.append((expected_next, next_ts - interval))

        return gaps

    def get_timeframe_statistics(self, timeframe: str) -> Dict[str, Any]:
        """
        Get statistics and information about a timeframe.

        Args:
            timeframe: Timeframe to analyze

        Returns:
            Dictionary with timeframe statistics
        """
        if not self.is_valid_timeframe(timeframe):
            return {"error": f"Invalid timeframe: {timeframe}"}

        minutes = self.parse_timeframe_to_minutes(timeframe)
        interval = self.parse_timeframe_to_timedelta(timeframe)
        category = self.get_timeframe_category(timeframe)

        return {
            "timeframe": timeframe,
            "category": category,
            "minutes": minutes,
            "hours": minutes / 60 if minutes else 0,
            "days": minutes / (60 * 24) if minutes else 0,
            "interval": str(interval) if interval else None,
            "smaller_timeframes": self.get_smaller_timeframes(timeframe),
            "larger_timeframes": self.get_larger_timeframes(timeframe),
            "records_per_day": 1440 / minutes if minutes else 0,
            "records_per_week": (1440 * 7) / minutes if minutes else 0,
            "records_per_month": (1440 * 30) / minutes if minutes else 0,
        }

    def get_all_supported_timeframes(self) -> List[str]:
        """
        Get all supported timeframes.

        Returns:
            List of all supported timeframe strings
        """
        return sorted(
            list(self._multipliers.keys()), key=lambda x: self._multipliers[x]
        )

    def get_common_conversion_pairs(self) -> List[Tuple[str, str]]:
        """
        Get common timeframe conversion pairs.

        Returns:
            List of (source, target) timeframe pairs that are commonly converted
        """
        common_pairs = [
            # From 1m
            ("1m", "3m"),
            ("1m", "5m"),
            ("1m", "15m"),
            ("1m", "30m"),
            ("1m", "1h"),
            # From 3m
            ("3m", "15m"),
            ("3m", "30m"),
            ("3m", "1h"),
            # From 5m
            ("5m", "15m"),
            ("5m", "30m"),
            ("5m", "1h"),
            # From 15m
            ("15m", "30m"),
            ("15m", "1h"),
            ("15m", "4h"),
            # From 30m
            ("30m", "1h"),
            ("30m", "2h"),
            ("30m", "4h"),
            # From 1h
            ("1h", "2h"),
            ("1h", "4h"),
            ("1h", "6h"),
            ("1h", "12h"),
            ("1h", "1d"),
            # From 2h
            ("2h", "4h"),
            ("2h", "6h"),
            ("2h", "12h"),
            ("2h", "1d"),
            # From 4h
            ("4h", "8h"),
            ("4h", "12h"),
            ("4h", "1d"),
            # From 12h
            ("12h", "1d"),
            # From 1d
            ("1d", "3d"),
            ("1d", "1w"),
            ("1d", "1M"),
        ]

        # Filter to only include valid pairs where both timeframes exist
        valid_pairs = []
        for source, target in common_pairs:
            if self.is_valid_timeframe(source) and self.is_valid_timeframe(target):
                valid_pairs.append((source, target))

        return valid_pairs

    def can_convert_timeframes(self, source: str, target: str) -> bool:
        """
        Check if conversion between two timeframes is possible.

        Args:
            source: Source timeframe
            target: Target timeframe

        Returns:
            True if conversion is possible (target >= source and evenly divisible)
        """
        source_minutes = self.parse_timeframe_to_minutes(source)
        target_minutes = self.parse_timeframe_to_minutes(target)

        if source_minutes is None or target_minutes is None:
            return False

        # Can only convert to larger timeframes that are evenly divisible
        return target_minutes >= source_minutes and target_minutes % source_minutes == 0

    def get_conversion_ratio(self, source: str, target: str) -> Optional[int]:
        """
        Get the conversion ratio between two timeframes.

        Args:
            source: Source timeframe
            target: Target timeframe

        Returns:
            Number of source intervals in one target interval, or None if invalid
        """
        if not self.can_convert_timeframes(source, target):
            return None

        source_minutes = self.parse_timeframe_to_minutes(source)
        target_minutes = self.parse_timeframe_to_minutes(target)

        if source_minutes and target_minutes:
            return target_minutes // source_minutes

        return None

    def get_pandas_frequency_string(self, timeframe: str) -> Optional[str]:
        """
        Convert timeframe to pandas frequency string for resampling.

        Args:
            timeframe: Timeframe string

        Returns:
            Pandas frequency string or None if not supported
        """
        frequency_map = {
            "1m": "1T",
            "3m": "3T",
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6H",
            "8h": "8H",
            "12h": "12H",
            "1d": "1D",
            "3d": "3D",
            "1w": "1W",
            "1M": "1M",
        }

        return frequency_map.get(timeframe)
