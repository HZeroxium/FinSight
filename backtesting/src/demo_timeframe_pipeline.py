# demo_timeframe_pipeline.py

"""
Comprehensive demonstration of Cross-Repository Timeframe Pipeline

This script demonstrates:
1. Setting up cross-repository dependencies
2. Loading 1h OHLCV data from source repository (e.g., MongoDB)
3. Converting to multiple timeframes (2h, 4h, 6h, 12h, 1d)
4. Saving converted data to target repository (e.g., CSV)
5. Multiple repository combinations

Usage:
    python -m backtesting.src.demo_timeframe_pipeline
"""

import asyncio
from datetime import timedelta
import traceback

from .timeframe_load_convert_save import (
    CrossRepositoryConfig,
    create_cross_repository_pipeline,
)
from .schemas.enums import CryptoSymbol, TimeFrame, Exchange, RepositoryType
from .utils.datetime_utils import DateTimeUtils
from .common.logger import LoggerFactory


class CrossRepositoryDemoConfig:
    """Configuration class for demo scenarios"""

    # Default symbols for demo
    DEFAULT_SYMBOLS = [
        CryptoSymbol.BTCUSDT.value,
        CryptoSymbol.ETHUSDT.value,
        CryptoSymbol.BNBUSDT.value,
    ]

    # Default timeframes for conversion
    DEFAULT_TARGET_TIMEFRAMES = [
        TimeFrame.HOUR_2.value,
        TimeFrame.HOUR_4.value,
        TimeFrame.HOUR_12.value,
        TimeFrame.DAY_1.value,
    ]

    # Date range configurations
    DEMO_DATE_RANGES = {
        "short": {"days_back": 7, "description": "Last 7 days"},
        "medium": {"days_back": 30, "description": "Last 30 days"},
        "long": {"days_back": 90, "description": "Last 90 days"},
    }


async def demo_mongodb_to_csv_conversion():
    """Demonstrate MongoDB → CSV timeframe conversion"""

    logger = LoggerFactory.get_logger("MongoDBToCSVDemo")
    logger.info("Starting MongoDB to CSV cross-repository conversion demo")

    # Configure repositories
    config = CrossRepositoryConfig(
        source_repo_type=RepositoryType.MONGODB,
        target_repo_type=RepositoryType.CSV,
        source_repo_config={
            "connection_string": "mongodb://localhost:27017/",
            "database_name": "finsight_market_data",
            "collection_prefix": "ohlcv",
        },
        target_repo_config={"base_directory": "data/demo_converted/mongodb_to_csv"},
    )

    # Create pipeline
    pipeline = create_cross_repository_pipeline(
        config=config,
        source_timeframe=TimeFrame.HOUR_1.value,
        target_timeframes=CrossRepositoryDemoConfig.DEFAULT_TARGET_TIMEFRAMES,
    )

    # Date range for demo
    date_range = CrossRepositoryDemoConfig.DEMO_DATE_RANGES["medium"]
    end_date = DateTimeUtils.now_iso()
    start_date = DateTimeUtils.to_iso_string(
        DateTimeUtils.now_utc() - timedelta(days=date_range["days_back"])
    )

    logger.info(f"Processing {len(CrossRepositoryDemoConfig.DEFAULT_SYMBOLS)} symbols")
    logger.info(f"Date range: {date_range['description']} ({start_date} to {end_date})")

    try:
        # Run the pipeline
        results = await pipeline.run_cross_repository_pipeline(
            symbols=CrossRepositoryDemoConfig.DEFAULT_SYMBOLS,
            exchange=Exchange.BINANCE.value,
            start_date=start_date,
            end_date=end_date,
            overwrite_existing=False,
        )

        # Display results
        print("\n" + "=" * 80)
        print("MONGODB → CSV CONVERSION RESULTS")
        print("=" * 80)
        print(
            f"Source: {results.get('source_repository_info', {}).get('storage_type', 'Unknown')}"
        )
        print(
            f"Target: {results.get('target_repository_info', {}).get('storage_type', 'Unknown')}"
        )
        print(f"Symbols processed: {results.get('symbols_processed', 0)}")
        print(f"Success rate: {results.get('success_rate', 0):.1f}%")
        print(
            f"Total conversions: {results.get('successful_conversions', 0)}/{results.get('total_conversions', 0)}"
        )

        if "symbol_results" in results:
            print("\nPer-symbol results:")
            for symbol, symbol_result in results["symbol_results"].items():
                conversions = symbol_result.get("conversions_successful", 0)
                source_records = symbol_result.get("source_records", 0)
                print(
                    f"  {symbol}: {conversions} conversions, {source_records} source records"
                )

                if symbol_result.get("converted_records"):
                    for timeframe, count in symbol_result["converted_records"].items():
                        if isinstance(count, int):
                            print(f"    → {timeframe}: {count} records")

        return results

    except Exception as e:
        logger.error(f"MongoDB to CSV demo failed: {str(e)}")
        print(f"MongoDB to CSV Demo Error: {str(e)}")
        return None


async def demo_csv_to_mongodb_conversion():
    """Demonstrate CSV → MongoDB timeframe conversion"""

    logger = LoggerFactory.get_logger("CSVToMongoDBDemo")
    logger.info("Starting CSV to MongoDB cross-repository conversion demo")

    # Configure repositories
    config = CrossRepositoryConfig(
        source_repo_type=RepositoryType.CSV,
        target_repo_type=RepositoryType.MONGODB,
        source_repo_config={"base_directory": "data"},
        target_repo_config={
            "connection_string": "mongodb://localhost:27017/",
            "database_name": "finsight_demo_converted",
            "collection_prefix": "ohlcv_converted",
        },
    )

    # Create pipeline (smaller set for CSV demo)
    pipeline = create_cross_repository_pipeline(
        config=config,
        source_timeframe=TimeFrame.HOUR_1.value,
        target_timeframes=[
            TimeFrame.HOUR_4.value,
            TimeFrame.DAY_1.value,
        ],
    )

    # Test symbols (smaller set for CSV demo)
    symbols = [
        CryptoSymbol.BTCUSDT.value,
        CryptoSymbol.ETHUSDT.value,
    ]

    # Date range for demo
    date_range = CrossRepositoryDemoConfig.DEMO_DATE_RANGES["short"]
    end_date = DateTimeUtils.now_iso()
    start_date = DateTimeUtils.to_iso_string(
        DateTimeUtils.now_utc() - timedelta(days=date_range["days_back"])
    )

    logger.info(f"Processing {len(symbols)} symbols")
    logger.info(f"Date range: {date_range['description']} ({start_date} to {end_date})")

    try:
        # Run the pipeline
        results = await pipeline.run_cross_repository_pipeline(
            symbols=symbols,
            exchange=Exchange.BINANCE.value,
            start_date=start_date,
            end_date=end_date,
            overwrite_existing=False,
        )

        # Display results
        print("\n" + "=" * 80)
        print("CSV → MONGODB CONVERSION RESULTS")
        print("=" * 80)
        print(
            f"Source: {results.get('source_repository_info', {}).get('storage_type', 'Unknown')}"
        )
        print(
            f"Target: {results.get('target_repository_info', {}).get('storage_type', 'Unknown')}"
        )
        print(f"Symbols processed: {results.get('symbols_processed', 0)}")
        print(f"Success rate: {results.get('success_rate', 0):.1f}%")
        print(
            f"Total conversions: {results.get('successful_conversions', 0)}/{results.get('total_conversions', 0)}"
        )

        if "symbol_results" in results:
            print("\nPer-symbol results:")
            for symbol, symbol_result in results["symbol_results"].items():
                conversions = symbol_result.get("conversions_successful", 0)
                print(f"  {symbol}: {conversions} conversions")

        return results

    except Exception as e:
        logger.error(f"CSV to MongoDB demo failed: {str(e)}")
        print(f"CSV to MongoDB Demo Error: {str(e)}")
        return None


async def demo_bidirectional_conversion():
    """Demonstrate bidirectional conversion between repositories"""

    logger = LoggerFactory.get_logger("BidirectionalDemo")
    logger.info("Starting bidirectional cross-repository conversion demo")

    print("\n" + "=" * 80)
    print("BIDIRECTIONAL CONVERSION DEMO")
    print("=" * 80)

    # Small symbol set for bidirectional demo
    symbols = [CryptoSymbol.BTCUSDT.value]

    # Short date range for demo
    date_range = CrossRepositoryDemoConfig.DEMO_DATE_RANGES["short"]
    end_date = DateTimeUtils.now_iso()
    start_date = DateTimeUtils.to_iso_string(
        DateTimeUtils.now_utc() - timedelta(days=date_range["days_back"])
    )

    # Step 1: MongoDB → CSV
    print("\n1. MongoDB → CSV")
    print("-" * 40)

    mongodb_to_csv_config = CrossRepositoryConfig(
        source_repo_type=RepositoryType.MONGODB,
        target_repo_type=RepositoryType.CSV,
        source_repo_config={
            "connection_string": "mongodb://localhost:27017/",
            "database_name": "finsight_market_data",
            "collection_prefix": "ohlcv",
        },
        target_repo_config={"base_directory": "data/demo_bidirectional/step1_csv"},
    )

    try:
        mongodb_to_csv_pipeline = create_cross_repository_pipeline(
            config=mongodb_to_csv_config,
            source_timeframe=TimeFrame.HOUR_1.value,
            target_timeframes=[TimeFrame.HOUR_4.value],
        )

        mongodb_to_csv_results = (
            await mongodb_to_csv_pipeline.run_cross_repository_pipeline(
                symbols=symbols,
                exchange=Exchange.BINANCE.value,
                start_date=start_date,
                end_date=end_date,
                overwrite_existing=True,
            )
        )

        print(
            f"✅ MongoDB → CSV: {mongodb_to_csv_results.get('success_rate', 0):.1f}% success"
        )

    except Exception as e:
        logger.warning(f"MongoDB → CSV step failed: {e}")
        print(f"⚠️  MongoDB → CSV step failed: {e}")

    # Step 2: CSV → MongoDB (different database)
    print("\n2. CSV → MongoDB (round-trip)")
    print("-" * 40)

    csv_to_mongodb_config = CrossRepositoryConfig(
        source_repo_type=RepositoryType.CSV,
        target_repo_type=RepositoryType.MONGODB,
        source_repo_config={"base_directory": "data/demo_bidirectional/step1_csv"},
        target_repo_config={
            "connection_string": "mongodb://localhost:27017/",
            "database_name": "finsight_demo_roundtrip",
            "collection_prefix": "ohlcv_roundtrip",
        },
    )

    try:
        csv_to_mongodb_pipeline = create_cross_repository_pipeline(
            config=csv_to_mongodb_config,
            source_timeframe=TimeFrame.HOUR_4.value,  # Use the converted data
            target_timeframes=[TimeFrame.DAY_1.value],  # Convert to daily
        )

        csv_to_mongodb_results = (
            await csv_to_mongodb_pipeline.run_cross_repository_pipeline(
                symbols=symbols,
                exchange=Exchange.BINANCE.value,
                start_date=start_date,
                end_date=end_date,
                overwrite_existing=True,
            )
        )

        print(
            f"✅ CSV → MongoDB: {csv_to_mongodb_results.get('success_rate', 0):.1f}% success"
        )

    except Exception as e:
        logger.warning(f"CSV → MongoDB step failed: {e}")
        print(f"⚠️  CSV → MongoDB step failed: {e}")

    print("\n✨ Bidirectional conversion completed!")


async def demo_pipeline_statistics():
    """Demonstrate pipeline statistics and capabilities"""

    logger = LoggerFactory.get_logger("PipelineStatsDemo")
    logger.info("Demonstrating pipeline statistics and capabilities")

    print("\n" + "=" * 80)
    print("PIPELINE CAPABILITIES & STATISTICS")
    print("=" * 80)

    # Create a sample pipeline to show capabilities
    config = CrossRepositoryConfig(
        source_repo_type=RepositoryType.MONGODB,
        target_repo_type=RepositoryType.CSV,
    )

    pipeline = create_cross_repository_pipeline(
        config=config,
        source_timeframe=TimeFrame.HOUR_1.value,
        target_timeframes=CrossRepositoryDemoConfig.DEFAULT_TARGET_TIMEFRAMES,
    )

    # Get and display statistics
    stats = pipeline.get_pipeline_statistics()

    print(f"Source timeframe: {stats['source_timeframe']}")
    print(f"Target timeframes: {stats['target_timeframes']}")
    print(f"Source repository: {stats['source_repository_type']}")
    print(f"Target repository: {stats['target_repository_type']}")

    print("\nSupported conversions:")
    for timeframe, supported in stats["supported_conversions"].items():
        ratio = stats["conversion_ratios"].get(timeframe, "N/A")
        status = "✅" if supported else "❌"
        print(f"  {stats['source_timeframe']} → {timeframe}: {status} (ratio: {ratio})")

    # Show available repository combinations
    print(f"\nAvailable repository combinations:")
    repo_types = [RepositoryType.CSV, RepositoryType.MONGODB, RepositoryType.INFLUXDB]

    for source_type in repo_types:
        for target_type in repo_types:
            if source_type != target_type:
                print(f"  {source_type.value} → {target_type.value}")


async def demo_comprehensive_cross_repository():
    """Run comprehensive demonstration of cross-repository features"""

    logger = LoggerFactory.get_logger("ComprehensiveCrossRepoDemo")
    logger.info(
        "Starting comprehensive cross-repository timeframe pipeline demonstration"
    )

    print("🚀 FinSight Cross-Repository Timeframe Conversion Pipeline Demo")
    print("=" * 90)

    try:
        # 1. Show pipeline capabilities
        await demo_pipeline_statistics()

        # 2. Demonstrate MongoDB → CSV conversion
        print(f"\n{'🔄 MongoDB → CSV Conversion':<70}")
        print("-" * 90)
        mongodb_to_csv_results = await demo_mongodb_to_csv_conversion()

        # 3. Demonstrate CSV → MongoDB conversion
        print(f"\n{'🔄 CSV → MongoDB Conversion':<70}")
        print("-" * 90)
        try:
            csv_to_mongodb_results = await demo_csv_to_mongodb_conversion()
        except Exception as e:
            logger.warning(f"CSV to MongoDB demo skipped: {e}")
            print("⚠️  CSV → MongoDB demo skipped (data not available)")
            csv_to_mongodb_results = None

        # 4. Demonstrate bidirectional conversion
        print(f"\n{'🔄 Bidirectional Conversion':<70}")
        print("-" * 90)
        try:
            await demo_bidirectional_conversion()
        except Exception as e:
            logger.warning(f"Bidirectional demo skipped: {e}")
            print("⚠️  Bidirectional demo skipped")

        # 5. Summary
        print("\n" + "=" * 90)
        print("📊 CROSS-REPOSITORY DEMO SUMMARY")
        print("=" * 90)

        if mongodb_to_csv_results:
            print(
                f"✅ MongoDB → CSV: {mongodb_to_csv_results.get('success_rate', 0):.1f}% success rate"
            )
        else:
            print("❌ MongoDB → CSV: Failed")

        if csv_to_mongodb_results:
            print(
                f"✅ CSV → MongoDB: {csv_to_mongodb_results.get('success_rate', 0):.1f}% success rate"
            )
        else:
            print("⚠️  CSV → MongoDB: Skipped or failed")

        print("\n✨ Cross-repository demonstration completed successfully!")

    except Exception as e:
        logger.error(f"Comprehensive cross-repository demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
        traceback.print_exc()


async def main():
    """Main demonstration entry point"""

    logger = LoggerFactory.get_logger("CrossRepoDemoMain")
    logger.info("Starting cross-repository timeframe pipeline demonstration")

    print("Starting FinSight Cross-Repository Timeframe Conversion Pipeline Demo...")
    print("This demo will showcase:")
    print("1. 📊 Pipeline capabilities and statistics")
    print("2. 🔄 MongoDB → CSV conversion")
    print("3. 🔄 CSV → MongoDB conversion")
    print("4. ↔️  Bidirectional conversion workflows")
    print("5. 📈 Cross-repository timeframe aggregation")
    print()

    try:
        await demo_comprehensive_cross_repository()

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        logger.info("Demo interrupted by user")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        logger.error(f"Demo failed: {str(e)}")
        traceback.print_exc()

    finally:
        logger.info("Cross-repository demo cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
