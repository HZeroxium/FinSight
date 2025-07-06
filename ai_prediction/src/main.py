# main.py

import asyncio
from .adapters.binance_market_data_collector import BinanceMarketDataCollector
from .services.market_data_service import MarketDataService
from .common.logger import LoggerFactory
from .services.market_data_collector_service import MarketDataCollectorService
from .factories import create_repository
from datetime import date, timedelta


def get_symbol_timeframe_pairs():
    return [
        # ("BTCUSDT", "1d"),
        # ("BTCUSDT", "12h"),
        # ("BTCUSDT", "6h"),
        # ("BTCUSDT", "4h"),
        # ("BTCUSDT", "2h"),
        # ("BTCUSDT", "1h"),
        # ("ETHUSDT", "1d"),
        # ("ETHUSDT", "12h"),
        # ("ETHUSDT", "6h"),
        # ("ETHUSDT", "4h"),
        # ("ETHUSDT", "2h"),
        # ("ETHUSDT", "1h"),
        # ("BNBUSDT", "1d"),
        # ("XRPUSDT", "1d"),
        # ("ADAUSDT", "1d"),
        # ("SOLUSDT", "1d"),
        # ("DOGEUSDT", "1d"),
        # ("DOTUSDT", "1d"),
        # ("MATICUSDT", "1d"),
        # ("LTCUSDT", "1d"),
        # Define with "1h" symbols for more frequent updates
        ("BTCUSDT", "1h"),
        ("ETHUSDT", "1h"),
        ("BNBUSDT", "1h"),
        ("XRPUSDT", "1h"),
        ("ADAUSDT", "1h"),
        ("SOLUSDT", "1h"),
        ("DOGEUSDT", "1h"),
        ("DOTUSDT", "1h"),
        ("MATICUSDT", "1h"),
        ("LTCUSDT", "1h"),
    ]


async def main():
    # Initialize components
    logger = LoggerFactory.get_logger(name="main")

    # Create repository using factory - easy to switch between different types
    # Option 1: CSV Repository
    # market_data_repository = create_repository("csv", {"base_directory": "data"})

    # Option 2: MongoDB Repository (uncomment to use)
    market_data_repository = create_repository(
        "mongodb",
        {
            "connection_string": "mongodb://localhost:27017/",
            "database_name": "finsight_market_data",
        },
    )

    # Option 3: InfluxDB Repository (uncomment to use)
    # market_data_repository = create_repository("influxdb", {
    #     "url": "http://localhost:8086",
    #     "token": "your-token",
    #     "org": "finsight",
    #     "bucket": "market_data"
    # })

    market_data_collector = BinanceMarketDataCollector()
    market_data_service = MarketDataService(market_data_repository)
    market_data_collector_service = MarketDataCollectorService(
        market_data_collector, market_data_service
    )

    end_date = date.today().isoformat()

    # Start date is very early to ensure we collect all historical data
    start_date = (date.today() - timedelta(days=365 * 20)).isoformat()  # 10 years ago

    pairs = get_symbol_timeframe_pairs()
    for symbol, timeframe in pairs:
        logger.info(
            f"Starting collection for {symbol}/{timeframe} from {start_date} to {end_date}"
        )
        try:
            # result = await market_data_collector_service.ensure_data_completeness(
            #     exchange="binance",
            #     symbol=symbol,
            #     timeframe=timeframe,
            #     start_date=start_date,
            #     end_date=end_date,
            # )
            result = await market_data_collector_service.collect_and_store_ohlcv(
                exchange="binance",
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            if result:
                logger.info(f"Collection successful for {symbol}/{timeframe}: {result}")
            else:
                logger.error(f"Collection failed for {symbol}/{timeframe}")
        except Exception as e:
            logger.error(f"Error during collection for {symbol}/{timeframe}: {e}")

    # # Start data collection
    # result = await market_data_collector_service.update_to_latest(
    #     exchange="binance",
    #     symbol="BTCUSDT",
    #     timeframe="1d",
    # )


if __name__ == "__main__":
    asyncio.run(main())
