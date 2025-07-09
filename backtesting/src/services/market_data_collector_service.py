# services/market_data_collector_service.py

"""
Market Data Collector Service

Orchestrates market data collection and storage operations.
Ensures data is kept up-to-date and handles collection scheduling.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import time

from ..interfaces.market_data_collector import MarketDataCollector
from ..services.market_data_service import MarketDataService
from common.logger import LoggerFactory
from ..utils.datetime_utils import DateTimeUtils


class MarketDataCollectorService:
    """
    Service for orchestrating market data collection and storage.

    Coordinates between MarketDataCollector and MarketDataService
    to ensure complete and up-to-date market data.
    """

    def __init__(
        self,
        collector: MarketDataCollector,
        data_service: MarketDataService,
        collection_interval_seconds: int = 3600,  # 1 hour default
    ):
        """
        Initialize market data collector service.

        Args:
            collector: MarketDataCollector implementation
            data_service: MarketDataService instance
            collection_interval_seconds: Interval between collection runs
        """
        self.collector = collector
        self.data_service = data_service
        self.collection_interval = collection_interval_seconds

        self.logger = LoggerFactory.get_logger(name="market_data_collector_service")

        self._running = False
        self._collection_stats = {
            "total_collections": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "last_collection_time": None,
            "collections_by_symbol": {},
        }

        self.logger.info("Market Data Collector Service initialized")

    async def collect_and_store_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        max_gap_days: int = 1,
    ) -> Dict[str, Any]:
        """
        Collect and store OHLCV data for specified range.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format
            max_gap_days: Maximum gap size to fill in single collection

        Returns:
            Collection result with statistics
        """
        collection_result = {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "records_collected": 0,
            "gaps_filled": 0,
            "success": False,
            "error": None,
            "collection_time": DateTimeUtils.now_iso(),
        }

        try:
            self.logger.info(
                f"Starting OHLCV collection for {exchange}/{symbol}/{timeframe} "
                f"from {start_date} to {end_date}"
            )

            # Collect data from exchange
            data = self.collector.collect_ohlcv(symbol, timeframe, start_date, end_date)

            if data:
                # Store collected data using schemas
                success = await self.data_service.save_ohlcv_data(
                    exchange, symbol, timeframe, data, validate=True
                )

                if success:
                    collection_result["records_collected"] = len(data)
                    collection_result["success"] = True

                    self.logger.info(
                        f"Successfully collected and stored {len(data)} OHLCV records"
                    )
                else:
                    collection_result["error"] = "Failed to save collected data"
                    self.logger.error("Failed to save collected data")
            else:
                collection_result["error"] = "No data collected from exchange"
                self.logger.warning("No data collected from exchange")

            # Update statistics
            self._update_collection_stats(symbol, collection_result["success"])

            return collection_result

        except Exception as e:
            error_msg = f"Collection failed: {str(e)}"
            collection_result["error"] = error_msg
            self.logger.error(error_msg)

            self._update_collection_stats(symbol, False)
            return collection_result

    async def ensure_data_completeness(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        max_gap_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Ensure OHLCV data is complete for specified range.
        Identifies and fills any gaps in existing data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format
            max_gap_days: Maximum gap size to fill automatically

        Returns:
            Completeness check result
        """
        result = {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "data_complete": False,
            "gaps_found": 0,
            "gaps_filled": 0,
            "total_records": 0,
            "errors": [],
        }

        try:
            self.logger.info(
                f"Checking data completeness for {exchange}/{symbol}/{timeframe}"
            )

            # Check for gaps in existing data
            gaps = self.data_service.get_data_gaps(
                exchange, symbol, timeframe, start_date, end_date
            )

            result["gaps_found"] = len(gaps)

            if not gaps:
                result["data_complete"] = True
                self.logger.info("Data is complete - no gaps found")
            else:
                self.logger.info(f"Found {len(gaps)} data gaps to fill")

                # Fill each gap
                for gap_start, gap_end in gaps:
                    try:
                        # Validate the gap dates
                        gap_start_dt = DateTimeUtils.to_utc_datetime(gap_start)
                        gap_end_dt = DateTimeUtils.to_utc_datetime(gap_end)

                        # Handle single-day gaps by extending end date by one interval
                        if gap_start_dt == gap_end_dt:
                            # For single timestamp gaps, extend by one interval
                            interval = DateTimeUtils.parse_timeframe_to_timedelta(
                                timeframe
                            )
                            if interval:
                                gap_end_dt = gap_start_dt + interval
                                gap_end = DateTimeUtils.to_iso_string(gap_end_dt)
                                self.logger.info(
                                    f"Extended single-day gap to: {gap_start} to {gap_end}"
                                )

                        # Ensure gap is valid
                        if gap_start_dt >= gap_end_dt:
                            self.logger.warning(
                                f"Invalid gap range: {gap_start} to {gap_end}, skipping"
                            )
                            continue

                        # Calculate gap size in days
                        gap_days = (gap_end_dt - gap_start_dt).days

                        if gap_days > max_gap_days:
                            self.logger.warning(
                                f"Gap too large ({gap_days} days), skipping: {gap_start} to {gap_end}"
                            )
                            continue

                        # Collect data for this gap
                        gap_result = await self.collect_and_store_ohlcv(
                            exchange, symbol, timeframe, gap_start, gap_end
                        )

                        if gap_result["success"]:
                            result["gaps_filled"] += 1
                            result["total_records"] += gap_result["records_collected"]
                        else:
                            result["errors"].append(
                                f"Failed to fill gap {gap_start}-{gap_end}: {gap_result['error']}"
                            )

                    except Exception as gap_error:
                        error_msg = (
                            f"Error filling gap {gap_start}-{gap_end}: {str(gap_error)}"
                        )
                        result["errors"].append(error_msg)
                        self.logger.error(error_msg)

                # Check if all gaps were filled
                result["data_complete"] = result["gaps_filled"] == result["gaps_found"]

            return result

        except Exception as e:
            error_msg = f"Error checking data completeness: {str(e)}"
            result["errors"].append(error_msg)
            self.logger.error(error_msg)
            return result

    async def update_to_latest(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        max_lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Update OHLCV data to the latest available timestamp.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Time interval
            max_lookback_days: Maximum days to look back if no existing data

        Returns:
            Update result
        """
        result = {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "records_added": 0,
            "success": False,
            "error": None,
            "last_timestamp_before": None,
            "last_timestamp_after": None,
        }

        try:
            # Get latest timestamp in database
            latest_timestamp = await self.data_service.get_latest_ohlcv_timestamp(
                exchange, symbol, timeframe
            )

            if latest_timestamp:
                result["last_timestamp_before"] = latest_timestamp
                start_date = latest_timestamp
            else:
                # No existing data, start from max_lookback_days ago
                start_dt = DateTimeUtils.now_utc() - timedelta(days=max_lookback_days)
                start_date = DateTimeUtils.to_iso_string(start_dt)

            # End date is now
            end_date = DateTimeUtils.now_iso()

            self.logger.info(
                f"Updating {exchange}/{symbol}/{timeframe} from {start_date} to {end_date}"
            )

            # Collect latest data
            collection_result = await self.collect_and_store_ohlcv(
                exchange, symbol, timeframe, start_date, end_date
            )

            if collection_result["success"]:
                result["records_added"] = collection_result["records_collected"]
                result["success"] = True

                # Get updated latest timestamp
                updated_timestamp = await self.data_service.get_latest_ohlcv_timestamp(
                    exchange, symbol, timeframe
                )
                result["last_timestamp_after"] = updated_timestamp

                self.logger.info(
                    f"Successfully updated with {result['records_added']} new records"
                )
            else:
                result["error"] = collection_result["error"]

            return result

        except Exception as e:
            error_msg = f"Error updating to latest: {str(e)}"
            result["error"] = error_msg
            self.logger.error(error_msg)
            return result

    async def scan_and_update_all_symbols(
        self,
        exchange: str,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        max_lookback_days: int = 30,
        update_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Scan and update all symbols to ensure data is current.
        This can be used as a background job or manual scan.

        Args:
            exchange: Exchange name
            symbols: List of symbols to process (None = get from collector)
            timeframes: List of timeframes to process (None = get from collector)
            max_lookback_days: Maximum days to look back for new symbols
            update_existing: Whether to update existing data to latest

        Returns:
            Scan and update results
        """
        scan_result = {
            "exchange": exchange,
            "scan_time": DateTimeUtils.now_iso(),
            "total_combinations": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "total_records_collected": 0,
            "processing_time_seconds": 0,
            "results": [],
            "errors": [],
        }

        start_time = time.time()

        try:
            # Get symbols and timeframes to process
            if symbols is None:
                symbols = self.data_service.get_available_symbols(exchange)
                if not symbols:
                    # Get from collector if none in database
                    symbols = self.collector.get_available_symbols()
                    symbols = symbols[:5]  # Limit for demo

            if timeframes is None:
                timeframes = self.collector.get_available_timeframes()
                timeframes = timeframes[:2]  # Limit for demo

            scan_result["total_combinations"] = len(symbols) * len(timeframes)

            self.logger.info(
                f"Starting scan for {len(symbols)} symbols × {len(timeframes)} timeframes = "
                f"{scan_result['total_combinations']} combinations"
            )

            # Process each combination
            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        if update_existing:
                            # Update existing data to latest
                            update_result = await self.update_to_latest(
                                exchange, symbol, timeframe, max_lookback_days
                            )
                        else:
                            # Just ensure completeness
                            end_date = DateTimeUtils.now_iso()
                            start_dt = DateTimeUtils.now_utc() - timedelta(
                                days=max_lookback_days
                            )
                            start_date = DateTimeUtils.to_iso_string(start_dt)

                            update_result = await self.ensure_data_completeness(
                                exchange, symbol, timeframe, start_date, end_date
                            )

                        scan_result["results"].append(update_result)

                        if update_result.get("success", False):
                            scan_result["successful_updates"] += 1
                            scan_result["total_records_collected"] += update_result.get(
                                "records_added", update_result.get("total_records", 0)
                            )
                        else:
                            scan_result["failed_updates"] += 1

                        # Add small delay to avoid overwhelming the exchange
                        await asyncio.sleep(0.1)

                    except Exception as e:
                        error_msg = f"Error processing {symbol}/{timeframe}: {str(e)}"
                        scan_result["errors"].append(error_msg)
                        scan_result["failed_updates"] += 1
                        self.logger.error(error_msg)

            scan_result["processing_time_seconds"] = time.time() - start_time

            self.logger.info(
                f"Scan completed: {scan_result['successful_updates']}/{scan_result['total_combinations']} successful, "
                f"{scan_result['total_records_collected']} total records collected"
            )

            return scan_result

        except Exception as e:
            error_msg = f"Error in scan and update: {str(e)}"
            scan_result["errors"].append(error_msg)
            self.logger.error(error_msg)
            return scan_result

    async def start_background_collection(
        self,
        exchange: str,
        symbols: List[str],
        timeframes: List[str],
        collection_interval_minutes: int = 60,
    ) -> None:
        """
        Start background collection service.

        Args:
            exchange: Exchange name
            symbols: List of symbols to monitor
            timeframes: List of timeframes to collect
            collection_interval_minutes: Minutes between collection runs
        """
        self._running = True
        interval_seconds = collection_interval_minutes * 60

        self.logger.info(
            f"Starting background collection for {len(symbols)} symbols, "
            f"interval: {collection_interval_minutes} minutes"
        )

        while self._running:
            try:
                # Update each symbol/timeframe combination to latest
                for symbol in symbols:
                    for timeframe in timeframes:
                        if not self._running:
                            break

                        await self.update_to_latest(exchange, symbol, timeframe)
                        await asyncio.sleep(1)  # Small delay between updates

                    if not self._running:
                        break

                # Wait for next collection cycle
                if self._running:
                    self.logger.info(
                        f"Background collection cycle completed. Next run in {collection_interval_minutes} minutes."
                    )
                    await asyncio.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in background collection: {str(e)}")
                if self._running:
                    await asyncio.sleep(60)  # Wait 1 minute before retrying

        self.logger.info("Background collection stopped")

    def stop_background_collection(self) -> None:
        """Stop background collection service."""
        self._running = False
        self.logger.info("Stopping background collection...")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self._collection_stats.copy()

    def _update_collection_stats(self, symbol: str, success: bool) -> None:
        """Update internal collection statistics."""
        self._collection_stats["total_collections"] += 1

        if success:
            self._collection_stats["successful_collections"] += 1
        else:
            self._collection_stats["failed_collections"] += 1

        self._collection_stats["last_collection_time"] = DateTimeUtils.now_iso()

        # Per-symbol stats
        if symbol not in self._collection_stats["collections_by_symbol"]:
            self._collection_stats["collections_by_symbol"][symbol] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
            }

        symbol_stats = self._collection_stats["collections_by_symbol"][symbol]
        symbol_stats["total"] += 1

        if success:
            symbol_stats["successful"] += 1
        else:
            symbol_stats["failed"] += 1


# Example usage and demonstration
async def main():
    """
    Demonstrate the Market Data Collector Service functionality.

    This shows how to use the service for various collection scenarios.
    """
    # This would be initialized with actual implementations
    # from ..adapters.binance_market_data_collector import BinanceMarketDataCollector
    # from ..adapters.csv_market_data_repository import CSVMarketDataRepository
    # from ..services.market_data_service import MarketDataService

    print("Market Data Collector Service Demo")
    print("=" * 50)

    # Example configuration
    exchange = "binance"
    symbols = ["BTCUSDT", "ETHUSDT"]
    timeframes = ["1h", "4h", "1d"]

    print(f"Exchange: {exchange}")
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print()

    # Simulate service operations
    print("Available operations:")
    print("1. collect_and_store_ohlcv() - Collect specific date range")
    print("2. ensure_data_completeness() - Fill gaps in existing data")
    print("3. update_to_latest() - Update to current timestamp")
    print("4. bulk_collection_scan() - Scan and update multiple symbols")
    print("5. start_background_collection() - Continuous monitoring")
    print()

    # Example date ranges
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    print(f"Example date range: {start_date} to {end_date}")
    print()

    print("Service would coordinate:")
    print("- MarketDataCollector: Fetch data from exchange APIs")
    print("- MarketDataService: Validate and manage storage")
    print("- Gap detection: Identify missing data periods")
    print("- Automatic updates: Keep data current")
    print("- Batch operations: Efficient bulk processing")


if __name__ == "__main__":
    asyncio.run(main())
    print("Service would coordinate:")
    print("- MarketDataCollector: Fetch data from exchange APIs")
    print("- MarketDataService: Validate and manage storage")
    print("- Gap detection: Identify missing data periods")
    print("- Automatic updates: Keep data current")
    print("- Batch operations: Efficient bulk processing")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main())
