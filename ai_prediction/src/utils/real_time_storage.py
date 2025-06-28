"""
Real-time data storage utilities for handling streaming market data.
Supports buffering, batching, and efficient storage of real-time data streams.
"""

import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
import pandas as pd

from ..common.logger import LoggerFactory, LoggerType, LogLevel
from .data_storage import DataStorage


class RealTimeDataBuffer:
    """Buffer for real-time data with automatic flushing"""

    def __init__(
        self,
        max_size: int = 1000,
        flush_interval: int = 30,
        flush_callback: Optional[Callable] = None,
        logger_name: str = "realtime_buffer",
    ):
        """
        Initialize real-time data buffer

        Args:
            max_size: Maximum buffer size before auto-flush
            flush_interval: Interval in seconds for automatic flush
            flush_callback: Optional callback function for custom flush handling
            logger_name: Name for the logger instance
        """
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.flush_callback = flush_callback

        self.logger = LoggerFactory.get_logger(
            name=logger_name,
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            use_colors=True,
        )

        # Data buffers organized by data type and symbol
        self.buffers: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(deque)
        )
        self.buffer_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

        # Flush control
        self._stop_flush = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._auto_flush_worker, daemon=True
        )
        self._flush_thread.start()

    def add_data(self, data_type: str, symbol: str, data: Dict[str, Any]) -> None:
        """
        Add data to buffer

        Args:
            data_type: Type of data (trades, tickers, book, candles)
            symbol: Trading symbol
            data: Data to buffer
        """
        buffer_key = f"{data_type}_{symbol}"

        with self.buffer_locks[buffer_key]:
            self.buffers[data_type][symbol].append(data)

            # Check if buffer is full
            if len(self.buffers[data_type][symbol]) >= self.max_size:
                self._flush_buffer(data_type, symbol)

    def _flush_buffer(self, data_type: str, symbol: str) -> None:
        """Flush specific buffer"""
        buffer_key = f"{data_type}_{symbol}"

        with self.buffer_locks[buffer_key]:
            if not self.buffers[data_type][symbol]:
                return

            data_to_flush = list(self.buffers[data_type][symbol])
            self.buffers[data_type][symbol].clear()

        # Call flush callback if provided
        if self.flush_callback:
            self.flush_callback(data_type, symbol, data_to_flush)

        self.logger.debug(
            f"Flushed {len(data_to_flush)} items from {data_type}_{symbol} buffer"
        )

    def flush_all(self) -> None:
        """Flush all buffers"""
        for data_type in self.buffers:
            for symbol in self.buffers[data_type]:
                self._flush_buffer(data_type, symbol)

    def _auto_flush_worker(self) -> None:
        """Background worker for automatic buffer flushing"""
        while not self._stop_flush.wait(self.flush_interval):
            try:
                self.flush_all()
            except Exception as e:
                self.logger.error(f"Error in auto-flush worker: {e}")

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status"""
        status = {}

        for data_type in self.buffers:
            status[data_type] = {}
            for symbol in self.buffers[data_type]:
                buffer_key = f"{data_type}_{symbol}"
                with self.buffer_locks[buffer_key]:
                    status[data_type][symbol] = len(self.buffers[data_type][symbol])

        return status

    def stop(self) -> None:
        """Stop the buffer and flush all data"""
        self._stop_flush.set()
        self._flush_thread.join(timeout=5)
        self.flush_all()


class RealTimeDataStorage:
    """Enhanced storage system for real-time market data"""

    def __init__(
        self,
        base_dir: str = "data/realtime",
        buffer_size: int = 1000,
        flush_interval: int = 30,
        logger_name: str = "realtime_storage",
    ):
        """
        Initialize real-time data storage

        Args:
            base_dir: Base directory for data storage
            buffer_size: Size of data buffer
            flush_interval: Flush interval in seconds
            logger_name: Name for the logger instance
        """
        self.logger = LoggerFactory.get_logger(
            name=logger_name,
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            use_colors=True,
        )

        # Initialize storage
        self.storage = DataStorage(base_dir=base_dir)

        # Initialize buffer with flush callback
        self.buffer = RealTimeDataBuffer(
            max_size=buffer_size,
            flush_interval=flush_interval,
            flush_callback=self._save_buffered_data,
        )

        # Statistics tracking
        self.stats = {
            "total_messages": 0,
            "messages_by_type": defaultdict(int),
            "messages_by_symbol": defaultdict(int),
            "start_time": datetime.now(timezone.utc),
        }

    def store_trade(self, trade_data: Dict[str, Any]) -> None:
        """Store real-time trade data"""
        self._store_data("trades", trade_data)

    def store_ticker(self, ticker_data: Dict[str, Any]) -> None:
        """Store real-time ticker data"""
        self._store_data("tickers", ticker_data)

    def store_book(self, book_data: Dict[str, Any]) -> None:
        """Store real-time order book data"""
        self._store_data("book", book_data)

    def store_candle(self, candle_data: Dict[str, Any]) -> None:
        """Store real-time candle data"""
        self._store_data("candles", candle_data)

    def _store_data(self, data_type: str, data: Dict[str, Any]) -> None:
        """Internal method to store data"""
        symbol = data.get("symbol", "UNKNOWN")
        exchange = data.get("exchange", "UNKNOWN")

        # Add to buffer
        self.buffer.add_data(data_type, f"{exchange}_{symbol}", data)

        # Update statistics
        self.stats["total_messages"] += 1
        self.stats["messages_by_type"][data_type] += 1
        self.stats["messages_by_symbol"][f"{exchange}_{symbol}"] += 1

    def _save_buffered_data(
        self, data_type: str, symbol: str, data_list: List[Dict[str, Any]]
    ) -> None:
        """Save buffered data to storage"""
        if not data_list:
            return

        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{data_type}_{timestamp}"

            # Save as JSON
            self.storage.save_json(
                data_list,
                filename,
                subfolder=f"realtime/{data_type}",
                timestamp_suffix=False,
            )

            # Save as CSV for easy analysis
            df = pd.DataFrame(data_list)
            self.storage.save_csv(
                df,
                filename,
                subfolder=f"realtime/{data_type}",
                timestamp_suffix=False,
            )

            self.logger.info(f"Saved {len(data_list)} {data_type} records for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to save buffered data: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        now = datetime.now(timezone.utc)
        runtime = (now - self.stats["start_time"]).total_seconds()

        stats = {
            **self.stats,
            "runtime_seconds": runtime,
            "messages_per_second": (
                self.stats["total_messages"] / runtime if runtime > 0 else 0
            ),
            "buffer_status": self.buffer.get_buffer_status(),
            "current_time": now.isoformat(),
        }

        return stats

    def create_daily_summary(self) -> Dict[str, Any]:
        """Create daily summary of collected data"""
        stats = self.get_statistics()

        summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_messages": stats["total_messages"],
            "runtime_hours": stats["runtime_seconds"] / 3600,
            "average_messages_per_hour": (
                stats["total_messages"] / (stats["runtime_seconds"] / 3600)
                if stats["runtime_seconds"] > 0
                else 0
            ),
            "messages_by_type": dict(stats["messages_by_type"]),
            "messages_by_symbol": dict(stats["messages_by_symbol"]),
            "top_symbols": sorted(
                stats["messages_by_symbol"].items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

        # Save summary
        self.storage.save_json(
            summary,
            "daily_summary",
            subfolder="summaries",
        )

        return summary

    def stop(self) -> None:
        """Stop storage and flush all buffers"""
        self.logger.info("Stopping real-time storage...")
        self.buffer.stop()

        # Create final summary
        final_summary = self.create_daily_summary()
        self.logger.info(
            f"Final summary: {final_summary['total_messages']} messages processed"
        )
