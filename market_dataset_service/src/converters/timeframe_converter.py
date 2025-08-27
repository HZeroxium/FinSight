# converters/timeframe_converter.py

"""
TimeFrame Converter for converting OHLCV data between different timeframes.

This module provides functionality to convert higher frequency data (e.g., 1h)
to lower frequency data (e.g., 4h, 1d) by aggregating OHLCV values appropriately.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from common.logger import LoggerFactory

from ..schemas.ohlcv_schemas import OHLCVSchema
from ..utils.timeframe_utils import TimeFrameUtils


class TimeFrameConverter:
    """
    Converts OHLCV data between different timeframes.

    Supports converting from higher frequency (smaller timeframe) to
    lower frequency (larger timeframe) data by proper aggregation.
    """

    def __init__(self, timeframe_utils: Optional[TimeFrameUtils] = None):
        """
        Initialize TimeFrameConverter.

        Args:
            timeframe_utils: Optional TimeFrameUtils instance. If not provided, creates new one.
        """
        self.logger = LoggerFactory.get_logger("TimeFrameConverter")
        self.timeframe_utils = timeframe_utils or TimeFrameUtils()

    def can_convert(self, source_timeframe: str, target_timeframe: str) -> bool:
        """
        Check if conversion from source to target timeframe is possible.

        Args:
            source_timeframe: Source timeframe (e.g., "1h")
            target_timeframe: Target timeframe (e.g., "4h")

        Returns:
            True if conversion is possible, False otherwise
        """
        try:
            return self.timeframe_utils.can_convert_timeframes(
                source_timeframe, target_timeframe
            )
        except Exception as e:
            self.logger.error(f"Error checking conversion compatibility: {e}")
            return False

    def get_conversion_ratio(
        self, source_timeframe: str, target_timeframe: str
    ) -> Optional[int]:
        """
        Get the conversion ratio between timeframes.

        Args:
            source_timeframe: Source timeframe
            target_timeframe: Target timeframe

        Returns:
            Number of source intervals in one target interval, or None if invalid
        """
        return self.timeframe_utils.get_conversion_ratio(
            source_timeframe, target_timeframe
        )

    def convert_ohlcv_data(
        self,
        data: List[OHLCVSchema],
        target_timeframe: str,
        source_timeframe: Optional[str] = None,
    ) -> List[OHLCVSchema]:
        """
        Convert OHLCV data to a different timeframe.

        Args:
            data: List of OHLCV data to convert
            target_timeframe: Target timeframe to convert to
            source_timeframe: Source timeframe (auto-detected if not provided)

        Returns:
            List of converted OHLCV data
        """
        if not data:
            return []

        # Auto-detect source timeframe if not provided
        if source_timeframe is None:
            source_timeframe = data[0].timeframe

        # Validate conversion is possible
        if not self.can_convert(source_timeframe, target_timeframe):
            raise ValueError(
                f"Cannot convert from {source_timeframe} to {target_timeframe}. "
                f"Target must be larger and evenly divisible by source timeframe."
            )

        # Convert to DataFrame for easier manipulation
        df = self._ohlcv_list_to_dataframe(data)

        # Perform timeframe conversion
        converted_df = self._aggregate_timeframe(df, target_timeframe, source_timeframe)

        # Convert back to OHLCVSchema objects
        converted_data = self._dataframe_to_ohlcv_list(
            converted_df, target_timeframe, data[0].exchange, data[0].symbol
        )

        self.logger.info(
            f"Converted {len(data)} records from {source_timeframe} to "
            f"{target_timeframe}, resulting in {len(converted_data)} records"
        )

        return converted_data

    def convert_multiple_symbols(
        self,
        symbol_data: Dict[str, List[OHLCVSchema]],
        target_timeframe: str,
        source_timeframe: Optional[str] = None,
    ) -> Dict[str, List[OHLCVSchema]]:
        """
        Convert OHLCV data for multiple symbols to a different timeframe.

        Args:
            symbol_data: Dictionary mapping symbol -> OHLCV data
            target_timeframe: Target timeframe to convert to
            source_timeframe: Source timeframe (auto-detected if not provided)

        Returns:
            Dictionary mapping symbol -> converted OHLCV data
        """
        converted_data = {}

        for symbol, data in symbol_data.items():
            try:
                converted_data[symbol] = self.convert_ohlcv_data(
                    data, target_timeframe, source_timeframe
                )
                self.logger.debug(f"Successfully converted data for symbol: {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to convert data for symbol {symbol}: {e}")
                converted_data[symbol] = []

        return converted_data

    def get_supported_conversions(self, source_timeframe: str) -> List[str]:
        """
        Get list of supported target timeframes for a given source timeframe.

        Args:
            source_timeframe: Source timeframe

        Returns:
            List of supported target timeframes
        """
        return self.timeframe_utils.get_larger_timeframes(source_timeframe)

    def _ohlcv_list_to_dataframe(self, data: List[OHLCVSchema]) -> pd.DataFrame:
        """Convert list of OHLCVSchema to pandas DataFrame"""
        records = []

        for item in data:
            records.append(
                {
                    "timestamp": item.timestamp,
                    "open": item.open,
                    "high": item.high,
                    "low": item.low,
                    "close": item.close,
                    "volume": item.volume,
                }
            )

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        return df

    def _aggregate_timeframe(
        self, df: pd.DataFrame, target_timeframe: str, source_timeframe: str
    ) -> pd.DataFrame:
        """Aggregate DataFrame to target timeframe"""

        # Define aggregation rules for OHLCV data
        agg_rules = {
            "open": "first",  # First open price in the period
            "high": "max",  # Maximum high price in the period
            "low": "min",  # Minimum low price in the period
            "close": "last",  # Last close price in the period
            "volume": "sum",  # Sum of all volumes in the period
        }

        # Get the appropriate resampling frequency using TimeFrameUtils
        freq = self.timeframe_utils.get_pandas_frequency_string(target_timeframe)
        if not freq:
            raise ValueError(
                f"Unsupported target timeframe for pandas resampling: {target_timeframe}"
            )

        # Resample and aggregate
        resampled_df = df.resample(freq).agg(agg_rules)

        # Remove rows with NaN values (incomplete periods)
        resampled_df = resampled_df.dropna()

        # Ensure data types
        for col in ["open", "high", "low", "close", "volume"]:
            resampled_df[col] = pd.to_numeric(resampled_df[col], errors="coerce")

        return resampled_df

    def _dataframe_to_ohlcv_list(
        self, df: pd.DataFrame, timeframe: str, exchange: str, symbol: str
    ) -> List[OHLCVSchema]:
        """Convert pandas DataFrame back to list of OHLCVSchema"""
        ohlcv_list = []

        for timestamp, row in df.iterrows():
            try:
                ohlcv = OHLCVSchema(
                    timestamp=timestamp.to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                )
                ohlcv_list.append(ohlcv)
            except Exception as e:
                self.logger.warning(f"Skipping invalid row at {timestamp}: {e}")
                continue

        return ohlcv_list

    def validate_timeframe_consistency(self, data: List[OHLCVSchema]) -> bool:
        """
        Validate that the data has consistent timeframe intervals.

        Args:
            data: OHLCV data to validate

        Returns:
            True if data has consistent intervals, False otherwise
        """
        if len(data) < 2:
            return True

        # Sort data by timestamp
        sorted_data = sorted(data, key=lambda x: x.timestamp)

        # Get expected interval from timeframe using TimeFrameUtils
        timeframe = sorted_data[0].timeframe
        expected_delta = self.timeframe_utils.parse_timeframe_to_timedelta(timeframe)

        if expected_delta is None:
            return False

        # Check intervals between consecutive records
        inconsistent_count = 0
        tolerance = timedelta(minutes=1)  # Allow 1 minute tolerance

        for i in range(1, len(sorted_data)):
            actual_delta = sorted_data[i].timestamp - sorted_data[i - 1].timestamp

            if abs(actual_delta - expected_delta) > tolerance:
                inconsistent_count += 1

        # Allow up to 5% inconsistency (for weekends, holidays, etc.)
        inconsistency_rate = inconsistent_count / (len(sorted_data) - 1)

        return inconsistency_rate <= 0.05

    def get_timeframe_info(self) -> Dict[str, Any]:
        """Get information about supported timeframes and conversions"""
        all_timeframes = self.timeframe_utils.get_all_supported_timeframes()

        return {
            "supported_timeframes": all_timeframes,
            "timeframe_statistics": {
                tf: self.timeframe_utils.get_timeframe_statistics(tf)
                for tf in all_timeframes
            },
            "conversion_matrix": self._build_conversion_matrix(),
            "common_conversion_pairs": self.timeframe_utils.get_common_conversion_pairs(),
        }

    def _build_conversion_matrix(self) -> Dict[str, List[str]]:
        """Build a matrix showing which conversions are supported"""
        matrix = {}
        all_timeframes = self.timeframe_utils.get_all_supported_timeframes()

        for source_tf in all_timeframes:
            matrix[source_tf] = self.get_supported_conversions(source_tf)

        return matrix

    def align_data_to_timeframe(
        self, data: List[OHLCVSchema], target_timeframe: str
    ) -> List[OHLCVSchema]:
        """
        Align timestamps in OHLCV data to timeframe boundaries.

        Args:
            data: OHLCV data to align
            target_timeframe: Target timeframe for alignment

        Returns:
            List of OHLCV data with aligned timestamps
        """
        aligned_data = []

        for record in data:
            try:
                aligned_timestamp = self.timeframe_utils.align_timestamp_to_timeframe(
                    record.timestamp, target_timeframe
                )

                aligned_record = OHLCVSchema(
                    timestamp=aligned_timestamp,
                    open=record.open,
                    high=record.high,
                    low=record.low,
                    close=record.close,
                    volume=record.volume,
                    symbol=record.symbol,
                    exchange=record.exchange,
                    timeframe=target_timeframe,
                )
                aligned_data.append(aligned_record)
            except Exception as e:
                self.logger.warning(f"Failed to align timestamp for record: {e}")
                continue

        return aligned_data

    def detect_missing_intervals(
        self, data: List[OHLCVSchema], expected_timeframe: Optional[str] = None
    ) -> List[Tuple[datetime, datetime]]:
        """
        Detect missing intervals in OHLCV data.

        Args:
            data: OHLCV data to analyze
            expected_timeframe: Expected timeframe (uses data timeframe if not specified)

        Returns:
            List of (missing_start, missing_end) timestamp tuples
        """
        if not data:
            return []

        timeframe = expected_timeframe or data[0].timeframe
        timestamps = [record.timestamp for record in data]

        return self.timeframe_utils.find_gaps_in_data(timestamps, timeframe)
