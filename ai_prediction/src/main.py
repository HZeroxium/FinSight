# main.py

import asyncio
from .adapters.binance_market_data_collector import BinanceMarketDataCollector
from .adapters.csv_market_data_repository import CSVMarketDataRepository
from .services.market_data_service import MarketDataService
from .common.logger import LoggerFactory
from .services.market_data_collector_service import MarketDataCollectorService


async def main():
    # Initialize components
    logger = LoggerFactory.get_logger(name="main")
    market_data_collector = BinanceMarketDataCollector()
    market_data_repository = CSVMarketDataRepository()
    market_data_service = MarketDataService(market_data_repository)
    market_data_collector_service = MarketDataCollectorService(
        market_data_collector, market_data_service
    )

    # Start data collection
    # result = await market_data_collector_service.collect_and_store_ohlcv(
    #     exchange="binance",
    #     symbol="BTCUSDT",
    #     timeframe="12h",
    #     start_date="2015-08-17",
    #     end_date="2025-06-30",
    # )

    result = await market_data_collector_service.update_to_latest(
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1d",
    )

    # result = await market_data_collector_service.ensure_data_completeness(
    #     exchange="binance",
    #     symbol="BTCUSDT",
    #     timeframe="1d",
    #     start_date="2017-06-01",
    #     end_date="2025-06-30",
    # )

    if result:
        logger.info("Data collection and storage successful")
        logger.info(f"Collected data: {result}")
    else:
        logger.error("Data collection failed")


if __name__ == "__main__":
    asyncio.run(main())
