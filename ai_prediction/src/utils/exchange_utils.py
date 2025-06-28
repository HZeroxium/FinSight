# utils/exchange_utils.py

"""
Utility functions for exchange data collection operations.
Contains common patterns and helper functions used across different exchanges.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


class ExchangeUtils:
    """Utility functions for exchange operations"""

    @staticmethod
    def normalize_symbol(symbol: str, exchange_format: str = "ccxt") -> str:
        """
        Normalize symbol format for different exchanges

        Args:
            symbol: Original symbol
            exchange_format: Target format ('ccxt', 'binance', 'cryptofeed')

        Returns:
            Normalized symbol
        """
        if exchange_format == "ccxt":
            # CCXT uses BTC/USDT format
            if "/" not in symbol and "USDT" in symbol:
                # Convert BTCUSDT to BTC/USDT
                base = symbol.replace("USDT", "")
                return f"{base}/USDT"
            return symbol
        elif exchange_format == "binance":
            # Binance uses BTCUSDT format
            return symbol.replace("/", "")
        elif exchange_format == "cryptofeed":
            # Cryptofeed typically uses BTCUSDT format
            return symbol.replace("/", "")
        else:
            return symbol

    @staticmethod
    def calculate_time_range(days_back: int) -> Tuple[datetime, datetime, int]:
        """
        Calculate time range for historical data fetching

        Args:
            days_back: Number of days to go back

        Returns:
            Tuple of (start_time, end_time, since_timestamp_ms)
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        since_ms = int(start_time.timestamp() * 1000)

        return start_time, end_time, since_ms

    @staticmethod
    def create_filename(
        exchange: str,
        symbol: str,
        data_type: str,
        timeframe: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> str:
        """
        Create standardized filename for data storage

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Type of data (ohlcv, trades, etc.)
            timeframe: Optional timeframe
            suffix: Optional suffix

        Returns:
            Standardized filename
        """
        clean_symbol = symbol.replace("/", "_").replace("-", "_")
        parts = [exchange, clean_symbol, data_type]

        if timeframe:
            parts.append(timeframe)
        if suffix:
            parts.append(suffix)

        return "_".join(parts)

    @staticmethod
    def validate_symbol_format(symbol: str, exchange_type: str) -> bool:
        """
        Validate symbol format for specific exchange

        Args:
            symbol: Symbol to validate
            exchange_type: Exchange type

        Returns:
            True if valid format
        """
        if exchange_type == "ccxt":
            # CCXT expects BTC/USDT format for most operations
            return "/" in symbol
        elif exchange_type == "binance":
            # Binance expects BTCUSDT format
            return "/" not in symbol
        else:
            return True  # Accept any format for unknown exchanges

    @staticmethod
    def chunk_time_range(
        start_time: datetime, end_time: datetime, chunk_size_hours: int = 24
    ) -> List[Tuple[datetime, datetime]]:
        """
        Split time range into smaller chunks for API requests

        Args:
            start_time: Start time
            end_time: End time
            chunk_size_hours: Size of each chunk in hours

        Returns:
            List of (start, end) time tuples
        """
        chunks = []
        current_start = start_time
        chunk_delta = timedelta(hours=chunk_size_hours)

        while current_start < end_time:
            current_end = min(current_start + chunk_delta, end_time)
            chunks.append((current_start, current_end))
            current_start = current_end

        return chunks

    @staticmethod
    def merge_ohlcv_data(ohlcv_chunks: List[List[List]]) -> List[List]:
        """
        Merge multiple OHLCV data chunks and remove duplicates

        Args:
            ohlcv_chunks: List of OHLCV data chunks

        Returns:
            Merged and deduplicated OHLCV data
        """
        all_ohlcv = []
        for chunk in ohlcv_chunks:
            all_ohlcv.extend(chunk)

        # Remove duplicates based on timestamp and sort
        seen_timestamps = set()
        unique_ohlcv = []

        for candle in sorted(all_ohlcv, key=lambda x: x[0]):
            if candle[0] not in seen_timestamps:
                unique_ohlcv.append(candle)
                seen_timestamps.add(candle[0])

        return unique_ohlcv

    @staticmethod
    def prepare_data_with_metadata(
        data: Any, exchange_id: str, symbol: str, data_type: str, **additional_metadata
    ) -> Dict[str, Any]:
        """
        Prepare data with standardized metadata

        Args:
            data: Raw data
            exchange_id: Exchange identifier
            symbol: Trading symbol
            data_type: Type of data
            **additional_metadata: Additional metadata fields

        Returns:
            Data with metadata
        """
        metadata = {
            "exchange_id": exchange_id,
            "symbol": symbol,
            "data_type": data_type,
            "timestamp": datetime.now().isoformat(),
            "data_count": len(data) if hasattr(data, "__len__") else 1,
            **additional_metadata,
        }

        if isinstance(data, (list, dict)):
            return {**metadata, "data": data}
        else:
            return {**metadata, "raw_data": data}

    @staticmethod
    def calculate_collection_stats(
        collected_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate statistics from collected data

        Args:
            collected_data: Dictionary of collected data by symbol

        Returns:
            Statistics summary
        """
        stats = {
            "total_symbols": 0,
            "total_ohlcv_candles": 0,
            "total_trades": 0,
            "total_orderbook_levels": 0,
            "symbols_with_tickers": 0,
            "symbols_by_data_type": {
                "ohlcv": [],
                "trades": [],
                "orderbook": [],
                "ticker": [],
            },
        }

        for symbol, data in collected_data.items():
            # Skip validation data and other non-symbol entries
            if (
                symbol.endswith("_validations")
                or not isinstance(data, dict)
                or "ohlcv" not in data
            ):
                continue

            stats["total_symbols"] += 1

            # Count OHLCV data
            if isinstance(data.get("ohlcv"), dict):
                for timeframe, count in data["ohlcv"].items():
                    stats["total_ohlcv_candles"] += count
                stats["symbols_by_data_type"]["ohlcv"].append(symbol)
            elif data.get("ohlcv", 0) > 0:
                stats["total_ohlcv_candles"] += data["ohlcv"]
                stats["symbols_by_data_type"]["ohlcv"].append(symbol)

            # Count trades
            if data.get("trades", 0) > 0:
                stats["total_trades"] += data["trades"]
                stats["symbols_by_data_type"]["trades"].append(symbol)

            # Count orderbook levels
            if data.get("orderbook", 0) > 0:
                stats["total_orderbook_levels"] += data["orderbook"]
                stats["symbols_by_data_type"]["orderbook"].append(symbol)

            # Count tickers
            if data.get("ticker"):
                stats["symbols_with_tickers"] += 1
                stats["symbols_by_data_type"]["ticker"].append(symbol)

        # Calculate coverage percentages
        total_symbols = stats["total_symbols"]
        if total_symbols > 0:
            stats["coverage"] = {
                "ohlcv_coverage": len(stats["symbols_by_data_type"]["ohlcv"])
                / total_symbols,
                "trades_coverage": len(stats["symbols_by_data_type"]["trades"])
                / total_symbols,
                "orderbook_coverage": len(stats["symbols_by_data_type"]["orderbook"])
                / total_symbols,
                "ticker_coverage": len(stats["symbols_by_data_type"]["ticker"])
                / total_symbols,
            }

        return stats
