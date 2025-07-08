# collect_new_symbol_data.py

"""
New Symbol Data Collection Script

Intelligent data collection for new symbols with automatic strategy selection:
- Collects complete historical data for new symbols
- Updates existing data to latest for known symbols
- Ensures data completeness for all symbol/timeframe pairs
"""

import asyncio
from datetime import date, timedelta
from typing import List, Tuple, Dict, Any, Optional

from .adapters.binance_market_data_collector import BinanceMarketDataCollector
from .services.market_data_service import MarketDataService
from .services.market_data_collector_service import MarketDataCollectorService
from .factories import create_repository
from .schemas.enums import CryptoSymbol, TimeFrame, Exchange
from .core.config import Settings
from .common.logger import LoggerFactory
from .utils.datetime_utils import DateTimeUtils


class NewSymbolDataCollector:
    """
    Intelligent data collector for new symbols with strategy selection
    """

    def __init__(
        self,
        exchange: str = Exchange.BINANCE.value,
        repository_type: str = "mongodb",
        repository_config: Optional[Dict[str, Any]] = None,
        max_lookback_days: int = 365,
        max_gap_days: int = 7,
    ):
        """
        Initialize the new symbol data collector

        Args:
            exchange: Exchange name (default: binance)
            repository_type: Type of repository (csv, mongodb, influxdb)
            repository_config: Repository configuration
            max_lookback_days: Maximum days to look back for historical data
            max_gap_days: Maximum gap size to fill automatically
        """
        self.exchange = exchange
        self.max_lookback_days = max_lookback_days
        self.max_gap_days = max_gap_days

        # Initialize logger
        self.logger = LoggerFactory.get_logger(name="new_symbol_collector")

        # Load settings
        self.settings = Settings()

        # Setup repository
        if repository_config is None:
            repository_config = self._get_default_repository_config(repository_type)

        self.repository = create_repository(repository_type, repository_config)

        # Initialize services
        self.market_data_collector = BinanceMarketDataCollector()
        self.market_data_service = MarketDataService(self.repository)
        self.collector_service = MarketDataCollectorService(
            self.market_data_collector, self.market_data_service
        )

        self.logger.info(
            f"Initialized NewSymbolDataCollector for {exchange} with {repository_type} repository"
        )

    def _get_default_repository_config(self, repository_type: str) -> Dict[str, Any]:
        """Get default repository configuration based on type"""
        if repository_type == "csv":
            return {"base_directory": self.settings.storage_config.base_directory}
        elif repository_type == "mongodb":
            return {
                "connection_string": "mongodb://localhost:27017/",
                "database_name": "finsight_market_data",
            }
        elif repository_type == "influxdb":
            return {
                "url": "http://localhost:8086",
                "token": "your-token",
                "org": "finsight",
                "bucket": "market_data",
            }
        else:
            raise ValueError(f"Unsupported repository type: {repository_type}")

    def get_default_symbol_timeframe_pairs(self) -> List[Tuple[str, str]]:
        """Get default symbol/timeframe pairs from configuration"""
        symbols = self.settings.data_collection_config.default_symbols
        timeframes = self.settings.data_collection_config.default_timeframes

        # Convert from enum-style to string values if needed
        symbol_values = [
            symbol.replace("/", "") if "/" in symbol else symbol for symbol in symbols
        ]

        # Create all combinations
        pairs = []
        for symbol in symbol_values:
            for timeframe in timeframes:
                pairs.append((symbol, timeframe))

        return pairs

    async def check_data_exists(
        self, symbol: str, timeframe: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if data exists for symbol/timeframe pair

        Returns:
            Tuple of (exists, latest_timestamp)
        """
        try:
            # Check if any data exists
            latest_timestamp = (
                await self.market_data_service.get_latest_ohlcv_timestamp(
                    self.exchange, symbol, timeframe
                )
            )
            return latest_timestamp is not None, latest_timestamp
        except Exception as e:
            self.logger.error(
                f"Error checking data existence for {symbol}/{timeframe}: {e}"
            )
            return False, None

    async def collect_symbol_data(
        self,
        symbol: str,
        timeframe: str,
        force_full_collection: bool = False,
        custom_start_date: Optional[str] = None,
        custom_end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Intelligent collection for a single symbol/timeframe pair

        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Time interval (e.g., 1h, 1d)
            force_full_collection: Force full historical collection
            custom_start_date: Custom start date (ISO format)
            custom_end_date: Custom end date (ISO format)

        Returns:
            Collection result dictionary
        """
        self.logger.info(f"🔍 Analyzing data for {symbol}/{timeframe}")

        # Check if data exists
        data_exists, latest_timestamp = await self.check_data_exists(symbol, timeframe)

        # Determine collection strategy
        if force_full_collection or not data_exists:
            # Strategy 1: Full historical collection
            return await self._collect_full_historical_data(
                symbol, timeframe, custom_start_date, custom_end_date
            )
        else:
            # Strategy 2: Update to latest + ensure completeness
            return await self._update_and_ensure_completeness(
                symbol, timeframe, latest_timestamp, custom_start_date, custom_end_date
            )

    async def _collect_full_historical_data(
        self,
        symbol: str,
        timeframe: str,
        custom_start_date: Optional[str] = None,
        custom_end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Collect full historical data for new symbol"""
        self.logger.info(f"📥 Collecting full historical data for {symbol}/{timeframe}")

        # Determine date range
        if custom_end_date:
            end_date = custom_end_date
        else:
            end_date = DateTimeUtils.now_iso()

        if custom_start_date:
            start_date = custom_start_date
        else:
            start_dt = DateTimeUtils.now_utc() - timedelta(days=self.max_lookback_days)
            start_date = DateTimeUtils.to_iso_string(start_dt)

        # Collect and store data
        result = await self.collector_service.collect_and_store_ohlcv(
            exchange=self.exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            max_gap_days=self.max_gap_days,
        )

        result["strategy"] = "full_historical_collection"
        return result

    async def _update_and_ensure_completeness(
        self,
        symbol: str,
        timeframe: str,
        latest_timestamp: str,
        custom_start_date: Optional[str] = None,
        custom_end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update existing data and ensure completeness"""
        self.logger.info(f"🔄 Updating existing data for {symbol}/{timeframe}")

        # Step 1: Update to latest
        update_result = await self.collector_service.update_to_latest(
            exchange=self.exchange,
            symbol=symbol,
            timeframe=timeframe,
            max_lookback_days=self.max_lookback_days,
        )

        # Step 2: Ensure data completeness
        if custom_end_date:
            end_date = custom_end_date
        else:
            end_date = DateTimeUtils.now_iso()

        if custom_start_date:
            start_date = custom_start_date
        else:
            # Use reasonable lookback for gap checking
            start_dt = DateTimeUtils.now_utc() - timedelta(days=self.max_lookback_days)
            start_date = DateTimeUtils.to_iso_string(start_dt)

        completeness_result = await self.collector_service.ensure_data_completeness(
            exchange=self.exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            max_gap_days=self.max_gap_days,
        )

        # Combine results
        combined_result = {
            "strategy": "update_and_completeness",
            "exchange": self.exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "update_result": update_result,
            "completeness_result": completeness_result,
            "total_records_added": (
                update_result.get("records_added", 0)
                + completeness_result.get("total_records", 0)
            ),
            "success": (
                update_result.get("success", False)
                and completeness_result.get("data_complete", False)
            ),
        }

        return combined_result

    async def collect_multiple_symbols(
        self,
        symbol_timeframe_pairs: Optional[List[Tuple[str, str]]] = None,
        force_full_collection: bool = False,
        custom_start_date: Optional[str] = None,
        custom_end_date: Optional[str] = None,
        max_concurrent: int = 3,
    ) -> Dict[str, Any]:
        """
        Collect data for multiple symbol/timeframe pairs

        Args:
            symbol_timeframe_pairs: List of (symbol, timeframe) tuples
            force_full_collection: Force full collection for all pairs
            custom_start_date: Custom start date
            custom_end_date: Custom end date
            max_concurrent: Maximum concurrent collections

        Returns:
            Overall collection results
        """
        if symbol_timeframe_pairs is None:
            symbol_timeframe_pairs = self.get_default_symbol_timeframe_pairs()

        self.logger.info(
            f"🚀 Starting collection for {len(symbol_timeframe_pairs)} symbol/timeframe pairs"
        )

        overall_result = {
            "total_pairs": len(symbol_timeframe_pairs),
            "successful_collections": 0,
            "failed_collections": 0,
            "total_records_collected": 0,
            "collection_results": [],
            "errors": [],
            "start_time": DateTimeUtils.now_iso(),
        }

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def collect_single_pair(symbol: str, timeframe: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.collect_symbol_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        force_full_collection=force_full_collection,
                        custom_start_date=custom_start_date,
                        custom_end_date=custom_end_date,
                    )
                except Exception as e:
                    error_msg = f"Error collecting {symbol}/{timeframe}: {str(e)}"
                    self.logger.error(error_msg)
                    return {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "success": False,
                        "error": error_msg,
                    }

        # Execute collections concurrently
        tasks = [
            collect_single_pair(symbol, timeframe)
            for symbol, timeframe in symbol_timeframe_pairs
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                overall_result["failed_collections"] += 1
                overall_result["errors"].append(str(result))
            else:
                overall_result["collection_results"].append(result)
                if result.get("success", False):
                    overall_result["successful_collections"] += 1
                    overall_result["total_records_collected"] += result.get(
                        "total_records_added", result.get("records_collected", 0)
                    )
                else:
                    overall_result["failed_collections"] += 1
                    if "error" in result:
                        overall_result["errors"].append(result["error"])

        overall_result["end_time"] = DateTimeUtils.now_iso()

        # Log summary
        self.logger.info(
            f"✅ Collection completed: {overall_result['successful_collections']}/{overall_result['total_pairs']} successful, "
            f"{overall_result['total_records_collected']} total records collected"
        )

        if overall_result["errors"]:
            self.logger.warning(
                f"⚠️ {len(overall_result['errors'])} errors occurred during collection"
            )

        return overall_result


async def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="New Symbol Data Collector")
    parser.add_argument(
        "--exchange", default="binance", help="Exchange name (default: binance)"
    )
    parser.add_argument(
        "--repository",
        choices=["csv", "mongodb", "influxdb"],
        default="mongodb",
        help="Repository type (default: mongodb)",
    )
    parser.add_argument("--symbol", help="Specific symbol to collect (e.g., BTCUSDT)")
    parser.add_argument("--timeframe", help="Specific timeframe to collect (e.g., 1h)")
    parser.add_argument(
        "--force-full", action="store_true", help="Force full historical collection"
    )
    parser.add_argument("--start-date", help="Custom start date (ISO format)")
    parser.add_argument("--end-date", help="Custom end date (ISO format)")
    parser.add_argument(
        "--max-concurrent", type=int, default=3, help="Maximum concurrent collections"
    )
    parser.add_argument(
        "--max-lookback-days",
        type=int,
        default=365,
        help="Maximum lookback days for historical data",
    )

    args = parser.parse_args()

    # Initialize collector
    collector = NewSymbolDataCollector(
        exchange=args.exchange,
        repository_type=args.repository,
        max_lookback_days=args.max_lookback_days,
    )

    try:
        if args.symbol and args.timeframe:
            # Collect single symbol/timeframe
            result = await collector.collect_symbol_data(
                symbol=args.symbol,
                timeframe=args.timeframe,
                force_full_collection=args.force_full,
                custom_start_date=args.start_date,
                custom_end_date=args.end_date,
            )
            collector.logger.info(f"Collection result: {result}")
        else:
            # Collect multiple symbols (default pairs)
            pairs = None
            if args.symbol:
                # Single symbol, all timeframes
                timeframes = (
                    collector.settings.data_collection_config.default_timeframes
                )
                pairs = [(args.symbol, tf) for tf in timeframes]
            elif args.timeframe:
                # All symbols, single timeframe
                symbols = [
                    s.replace("/", "")
                    for s in collector.settings.data_collection_config.default_symbols
                ]
                pairs = [(sym, args.timeframe) for sym in symbols]

            result = await collector.collect_multiple_symbols(
                symbol_timeframe_pairs=pairs,
                force_full_collection=args.force_full,
                custom_start_date=args.start_date,
                custom_end_date=args.end_date,
                max_concurrent=args.max_concurrent,
            )
            collector.logger.info(f"Overall collection result: {result}")

    except KeyboardInterrupt:
        collector.logger.info("Collection interrupted by user")
    except Exception as e:
        collector.logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
