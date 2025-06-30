# utils/base_data_collector.py

"""
Base data collector class providing common functionality for market data collection.
This class serves as a foundation for exchange-specific implementations.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps
import pandas as pd

from ..common.logger import LoggerFactory, LoggerType, LogLevel
from .data_storage import DataStorage
from .market_data_processor import MarketDataProcessor
from ..core.config import ConfigManager


def retry_on_error(
    max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)
):
    """Decorator for retrying operations on specific errors"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Log retry attempt
                        if hasattr(args[0], "logger"):
                            args[0].logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying..."
                            )
                        time.sleep(delay * (2**attempt))  # Exponential backoff
                        continue
                    break
                except Exception as e:
                    # Don't retry on other types of errors, but log them
                    if hasattr(args[0], "logger"):
                        args[0].logger.error(
                            f"Non-retryable error in {func.__name__}: {e}"
                        )
                    raise e

            # Log final failure
            if hasattr(args[0], "logger"):
                args[0].logger.error(
                    f"All {max_retries} attempts failed for {func.__name__}"
                )
            raise last_exception

        return wrapper

    return decorator


class BaseDataCollector(ABC):
    """Base class for market data collectors"""

    def __init__(
        self,
        exchange_name: str,
        base_dir: Optional[str] = None,
        logger_name: Optional[str] = None,
    ):
        """
        Initialize base data collector

        Args:
            exchange_name: Name of the exchange
            base_dir: Base directory for data storage
            logger_name: Name for the logger instance
        """
        self.exchange_name = exchange_name
        self.logger_name = logger_name or f"{exchange_name}_collector"

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name=self.logger_name,
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            use_colors=True,
            file_level=LogLevel.DEBUG,
            log_file=f"logs/{self.exchange_name}_collector.log",
        )

        # Initialize configuration
        self.config_manager = ConfigManager()
        self.exchange_config = self.config_manager.get_exchange_config(exchange_name)
        self.data_config = self.config_manager.get_data_collection_config()
        self.storage_config = self.config_manager.get_storage_config()

        # Initialize storage and processor
        storage_base_dir = base_dir or f"data/{exchange_name}"
        self.storage = DataStorage(base_dir=storage_base_dir)
        self.processor = MarketDataProcessor()

        # Create dataset structure
        self.dataset_structure = self.storage.create_dataset_structure(exchange_name)

        self.logger.info(f"Initialized {exchange_name} data collector")

    @abstractmethod
    def _initialize_client(self, **kwargs) -> None:
        """Initialize the exchange client - must be implemented by subclasses"""
        pass

    def get_default_symbols(self) -> List[str]:
        """Get default symbols for this exchange"""
        return self.config_manager.get_symbols_for_exchange(self.exchange_name)

    def get_default_timeframes(self) -> List[str]:
        """Get default timeframes for this exchange"""
        return self.config_manager.get_timeframes_for_exchange(self.exchange_name)

    def apply_rate_limiting(self, delay: float = 0.1) -> None:
        """Apply rate limiting delay"""
        time.sleep(delay)

    def save_collection_summary(
        self, summary: Dict[str, Any], filename: str = "collection_summary"
    ) -> None:
        """Save collection summary to storage"""
        self.storage.save_json(summary, filename, subfolder="summaries")

    def create_collection_summary(
        self, symbols: List[str], timeframes: List[str], **kwargs
    ) -> Dict[str, Any]:
        """Create base collection summary structure"""
        return {
            "exchange": self.exchange_name,
            "symbols": symbols,
            "timeframes": timeframes,
            "timestamp": datetime.now().isoformat(),
            "collected_data": {},
            "errors": [],
            "total_items": 0,
            **kwargs,
        }

    def log_collection_results(self, summary: Dict[str, Any]) -> None:
        """Log collection results in a standardized format"""
        self.logger.info("Collection Summary:")
        self.logger.info(f"  Exchange: {summary.get('exchange', 'Unknown')}")
        self.logger.info(f"  Total items collected: {summary.get('total_items', 0)}")
        self.logger.info(f"  Errors encountered: {len(summary.get('errors', []))}")

        for symbol, data in summary.get("collected_data", {}).items():
            # Skip non-symbol data like validations
            if not isinstance(data, dict) or symbol.endswith("_validations"):
                continue

            self.logger.info(f"  {symbol}:")
            if isinstance(data.get("ohlcv"), dict):
                self.logger.info(f"    - OHLCV: {data['ohlcv']}")
            else:
                self.logger.info(f"    - OHLCV: {data.get('ohlcv', 0)}")
            self.logger.info(f"    - Trades: {data.get('trades', 0)}")
            self.logger.info(f"    - Orderbook levels: {data.get('orderbook', 0)}")
            self.logger.info(f"    - Ticker: {'✓' if data.get('ticker') else '✗'}")

        if summary.get("errors"):
            self.logger.warning(
                f"Encountered {len(summary['errors'])} errors during collection"
            )

    def standardize_and_save_ohlcv(
        self,
        raw_ohlcv: List[List],
        symbol: str,
        timeframe: str,
        filename_prefix: str = "ohlcv",
    ) -> None:
        """Standardize and save OHLCV data"""
        if not raw_ohlcv:
            return

        # Process OHLCV data
        processed_ohlcv = self.processor.standardize_ohlcv(raw_ohlcv, symbol)

        # Create filename
        clean_symbol = symbol.replace("/", "_")
        filename = f"{self.exchange_name}_{clean_symbol}_{timeframe}_{filename_prefix}"

        # Save processed data
        self.storage.save_csv(
            processed_ohlcv, f"{filename}_processed", subfolder="ohlcv"
        )

        # Save as parquet for efficient storage
        self.storage.save_parquet(
            processed_ohlcv, f"{filename}_processed", subfolder="ohlcv"
        )

    def standardize_and_save_trades(
        self, raw_trades: List[Dict], symbol: str, filename_prefix: str = "trades"
    ) -> None:
        """Standardize and save trade data"""
        if not raw_trades:
            return

        # Process trades
        processed_trades = self.processor.standardize_trades(raw_trades, symbol)

        # Create filename
        clean_symbol = symbol.replace("/", "_")
        filename = f"{self.exchange_name}_{clean_symbol}_{filename_prefix}"

        # Save processed data
        if not processed_trades.empty:
            self.storage.save_csv(
                processed_trades, f"{filename}_processed", subfolder="trades"
            )

    def standardize_and_save_orderbook(
        self, raw_orderbook: Dict, symbol: str, filename_prefix: str = "orderbook"
    ) -> None:
        """Standardize and save orderbook data"""
        if not raw_orderbook:
            return

        # Process orderbook
        processed_orderbook = self.processor.standardize_orderbook(
            raw_orderbook, symbol
        )

        # Create filename
        clean_symbol = symbol.replace("/", "_")
        filename = f"{self.exchange_name}_{clean_symbol}_{filename_prefix}"

        # Save processed data - fix the DataFrame check
        for side, data in processed_orderbook.items():
            # Only process DataFrame objects (bids/asks), skip other metadata
            if isinstance(data, pd.DataFrame) and not data.empty:
                self.storage.save_csv(data, f"{filename}_{side}", subfolder="orderbook")
            elif side in ["spread_analysis"] and isinstance(data, dict):
                # Save spread analysis as JSON
                self.storage.save_json(
                    data, f"{filename}_{side}", subfolder="orderbook"
                )

    def standardize_and_save_ticker(
        self, raw_ticker: Dict, symbol: str, filename_prefix: str = "ticker"
    ) -> None:
        """Standardize and save ticker data"""
        if not raw_ticker:
            return

        # Standardize ticker data
        standardized_ticker = self.processor.standardize_ticker(raw_ticker, symbol)

        # Create filename
        clean_symbol = symbol.replace("/", "_")
        filename = f"{self.exchange_name}_{clean_symbol}_{filename_prefix}"

        # Save processed data
        self.storage.save_json(
            standardized_ticker, f"{filename}_processed", subfolder="tickers"
        )

    @abstractmethod
    def collect_comprehensive_data(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Collect comprehensive market data - must be implemented by subclasses"""
        pass

    def cleanup_and_finalize(self) -> None:
        """Cleanup resources and finalize collection"""
        self.logger.info(f"{self.exchange_name} data collection completed")
        self.logger.info(
            f"Check the 'data/{self.exchange_name}' directory for collected datasets"
        )
