# timeframe_load_convert_save.py

"""
Cross-Repository Timeframe Load-Convert-Save Pipeline

This module demonstrates a complete pipeline for:
1. Loading 1h OHLCV data from a source repository (e.g., MongoDB)
2. Converting to multiple larger timeframes (2h, 4h, 6h, 12h, 1d)
3. Saving converted data to a target repository (e.g., CSV)

The pipeline is designed to avoid rate limits by only fetching 1h data
and generating larger timeframes through aggregation across different storage systems.
"""

import asyncio
import traceback
from typing import Any, Dict, List, Optional

from common.logger import LoggerFactory

from ..converters.timeframe_converter import TimeFrameConverter
from ..factories.market_data_repository_factory import \
    MarketDataRepositoryFactory
from ..interfaces.market_data_repository import MarketDataRepository
from ..schemas.enums import CryptoSymbol, Exchange, RepositoryType, TimeFrame
from ..schemas.ohlcv_schemas import OHLCVQuerySchema, OHLCVSchema
from ..services.market_data_service import MarketDataService
from ..utils.datetime_utils import DateTimeUtils
from ..utils.timeframe_utils import TimeFrameUtils


class CrossRepositoryTimeFramePipeline:
    """
    Pipeline for loading, converting, and saving timeframe data across different repositories.

    Provides a high-level interface for cross-repository timeframe conversion workflow.
    """

    def __init__(
        self,
        source_repository: MarketDataRepository,
        target_repository: MarketDataRepository,
        source_timeframe: str = TimeFrame.HOUR_1.value,
        target_timeframes: Optional[List[str]] = None,
        timeframe_utils: Optional[TimeFrameUtils] = None,
    ):
        """
        Initialize the cross-repository timeframe pipeline.

        Args:
            source_repository: Repository to load source data from
            target_repository: Repository to save converted data to
            source_timeframe: Source timeframe for loading data
            target_timeframes: List of target timeframes to convert to
            timeframe_utils: Optional TimeFrameUtils instance
        """
        self.source_repository = source_repository
        self.target_repository = target_repository
        self.source_timeframe = source_timeframe
        self.target_timeframes = target_timeframes or [
            TimeFrame.HOUR_2.value,
            TimeFrame.HOUR_4.value,
            TimeFrame.HOUR_6.value,
            TimeFrame.HOUR_12.value,
            TimeFrame.DAY_1.value,
        ]

        # Initialize services and utilities
        self.source_service = MarketDataService(self.source_repository)
        self.target_service = MarketDataService(self.target_repository)

        # Use provided TimeFrameUtils or create new one
        self.timeframe_utils = timeframe_utils or TimeFrameUtils()
        self.timeframe_converter = TimeFrameConverter(self.timeframe_utils)

        self.logger = LoggerFactory.get_logger("CrossRepositoryTimeFramePipeline")

    async def run_cross_repository_pipeline(
        self,
        symbols: List[str],
        exchange: str = Exchange.BINANCE.value,
        start_date: str = "2013-01-01T00:00:00Z",
        end_date: Optional[str] = None,
        overwrite_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete cross-repository timeframe conversion pipeline.

        Args:
            symbols: List of symbols to process
            exchange: Exchange name
            start_date: Start date in ISO format
            end_date: End date in ISO format (defaults to now)
            overwrite_existing: Whether to overwrite existing data

        Returns:
            Dictionary with pipeline results and statistics
        """
        if end_date is None:
            end_date = DateTimeUtils.now_iso()

        self.logger.info(
            f"Starting cross-repository timeframe pipeline for {len(symbols)} symbols"
        )
        self.logger.info(f"Source timeframe: {self.source_timeframe}")
        self.logger.info(f"Target timeframes: {self.target_timeframes}")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Source repository: {type(self.source_repository).__name__}")
        self.logger.info(f"Target repository: {type(self.target_repository).__name__}")

        # Validate source timeframe and target timeframes compatibility
        for target_tf in self.target_timeframes:
            if not self.timeframe_converter.can_convert(
                self.source_timeframe, target_tf
            ):
                self.logger.warning(
                    f"Cannot convert from {self.source_timeframe} to {target_tf}, skipping"
                )

        results = {
            "pipeline_start": DateTimeUtils.now_iso(),
            "symbols_processed": 0,
            "symbols_failed": 0,
            "total_conversions": 0,
            "successful_conversions": 0,
            "errors": [],
            "symbol_results": {},
            "source_repository_info": await self._get_repository_info(
                self.source_repository
            ),
            "target_repository_info": await self._get_repository_info(
                self.target_repository
            ),
            "timeframe_statistics": {
                "source": self.timeframe_utils.get_timeframe_statistics(
                    self.source_timeframe
                ),
                "targets": {
                    tf: self.timeframe_utils.get_timeframe_statistics(tf)
                    for tf in self.target_timeframes
                },
                "conversion_ratios": {
                    tf: self.timeframe_utils.get_conversion_ratio(
                        self.source_timeframe, tf
                    )
                    for tf in self.target_timeframes
                    if self.timeframe_utils.can_convert_timeframes(
                        self.source_timeframe, tf
                    )
                },
            },
        }

        try:
            # Process each symbol
            for symbol in symbols:
                try:
                    symbol_result = await self.process_symbol_cross_repository(
                        symbol=symbol,
                        exchange=exchange,
                        start_date=start_date,
                        end_date=end_date,
                        overwrite_existing=overwrite_existing,
                    )

                    results["symbol_results"][symbol] = symbol_result
                    results["symbols_processed"] += 1
                    results["total_conversions"] += symbol_result.get(
                        "conversions_attempted", 0
                    )
                    results["successful_conversions"] += symbol_result.get(
                        "conversions_successful", 0
                    )

                    self.logger.info(
                        f"Completed cross-repository processing for symbol: {symbol}"
                    )

                except Exception as e:
                    error_msg = f"Failed to process symbol {symbol}: {str(e)}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
                    results["symbols_failed"] += 1
                    results["symbol_results"][symbol] = {"error": str(e)}

            results["pipeline_end"] = DateTimeUtils.now_iso()
            results["success_rate"] = (
                results["successful_conversions"] / max(results["total_conversions"], 1)
            ) * 100

            self.logger.info(f"Cross-repository pipeline completed successfully")
            self.logger.info(
                f"Processed: {results['symbols_processed']}/{len(symbols)} symbols"
            )
            self.logger.info(f"Conversion success rate: {results['success_rate']:.1f}%")

            return results

        except Exception as e:
            self.logger.error(f"Cross-repository pipeline failed with error: {str(e)}")
            results["pipeline_error"] = str(e)
            results["pipeline_end"] = DateTimeUtils.now_iso()
            return results

    async def process_symbol_cross_repository(
        self,
        symbol: str,
        exchange: str,
        start_date: str,
        end_date: str,
        overwrite_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a single symbol through the cross-repository timeframe conversion pipeline.

        Args:
            symbol: Symbol to process
            exchange: Exchange name
            start_date: Start date in ISO format
            end_date: End date in ISO format
            overwrite_existing: Whether to overwrite existing data

        Returns:
            Dictionary with processing results
        """
        result = {
            "symbol": symbol,
            "source_timeframe": self.source_timeframe,
            "target_timeframes": self.target_timeframes,
            "conversions_attempted": 0,
            "conversions_successful": 0,
            "source_records": 0,
            "converted_records": {},
            "errors": [],
        }

        try:
            # Step 1: Load source data from source repository
            self.logger.info(
                f"Loading {self.source_timeframe} data for {symbol} from source repository"
            )

            source_data = await self.load_source_data_from_repository(
                symbol=symbol,
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
            )

            if not source_data:
                result["errors"].append("No source data found in source repository")
                return result

            result["source_records"] = len(source_data)
            self.logger.info(
                f"Loaded {len(source_data)} records for {symbol} from source repository"
            )

            # Validate data consistency using TimeFrameUtils
            if not self.timeframe_converter.validate_timeframe_consistency(source_data):
                self.logger.warning(
                    f"Source data for {symbol} has inconsistent intervals"
                )

            # Step 2: Convert to each target timeframe and save to target repository
            for target_timeframe in self.target_timeframes:
                result["conversions_attempted"] += 1

                try:
                    # Check if conversion is supported
                    if not self.timeframe_converter.can_convert(
                        self.source_timeframe, target_timeframe
                    ):
                        error_msg = f"Conversion not supported: {self.source_timeframe} -> {target_timeframe}"
                        result["errors"].append(error_msg)
                        continue

                    # Check if data already exists in target repository (if not overwriting)
                    if not overwrite_existing:
                        existing_data = await self.check_existing_data_in_target(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=target_timeframe,
                            start_date=start_date,
                            end_date=end_date,
                        )

                        if existing_data:
                            self.logger.info(
                                f"Skipping {target_timeframe} for {symbol} - data already exists in target repository"
                            )
                            result["converted_records"][
                                target_timeframe
                            ] = "skipped_existing"
                            result["conversions_successful"] += 1
                            continue

                    # Convert data
                    self.logger.info(f"Converting {symbol} to {target_timeframe}")

                    converted_data = self.timeframe_converter.convert_ohlcv_data(
                        data=source_data,
                        target_timeframe=target_timeframe,
                        source_timeframe=self.source_timeframe,
                    )

                    if not converted_data:
                        result["errors"].append(
                            f"No converted data for {target_timeframe}"
                        )
                        continue

                    # Save converted data to target repository
                    await self.save_converted_data_to_target(
                        data=converted_data,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=target_timeframe,
                    )

                    result["converted_records"][target_timeframe] = len(converted_data)
                    result["conversions_successful"] += 1

                    self.logger.info(
                        f"Successfully converted and saved {len(converted_data)} records "
                        f"for {symbol} {target_timeframe} to target repository"
                    )

                except Exception as e:
                    error_msg = f"Failed conversion to {target_timeframe}: {str(e)}"
                    result["errors"].append(error_msg)
                    self.logger.error(error_msg)

            return result

        except Exception as e:
            result["errors"].append(f"Symbol processing failed: {str(e)}")
            self.logger.error(f"Failed to process symbol {symbol}: {str(e)}")
            return result

    async def load_source_data_from_repository(
        self, symbol: str, exchange: str, start_date: str, end_date: str
    ) -> List[OHLCVSchema]:
        """Load source timeframe data from source repository"""

        query = OHLCVQuerySchema(
            symbol=symbol,
            exchange=exchange,
            timeframe=self.source_timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=None,  # Load all available data
        )

        response = await self.source_service.get_ohlcv_data(
            exchange=exchange,
            symbol=symbol,
            timeframe=self.source_timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        return response.data

    async def check_existing_data_in_target(
        self, symbol: str, exchange: str, timeframe: str, start_date: str, end_date: str
    ) -> bool:
        """Check if data already exists in target repository for the given parameters"""

        try:
            existing_response = await self.target_service.get_ohlcv_data(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                limit=1,  # Just check if any data exists
            )

            return len(existing_response.data) > 0

        except Exception:
            # If check fails, assume no data exists
            return False

    async def save_converted_data_to_target(
        self, data: List[OHLCVSchema], symbol: str, exchange: str, timeframe: str
    ) -> bool:
        """Save converted data to target repository"""

        return await self.target_service.save_ohlcv_data(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            data=data,
            validate=True,
        )

    async def _get_repository_info(
        self, repository: MarketDataRepository
    ) -> Dict[str, Any]:
        """Get repository information for logging"""
        try:
            storage_info = await repository.get_storage_info()
            return {
                "storage_type": storage_info.get("storage_type", "unknown"),
                "location": storage_info.get("location", "unknown"),
            }
        except Exception:
            return {"storage_type": "unknown", "location": "unknown"}

    async def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get statistics about the pipeline configuration"""

        source_repo_info = await self._get_repository_info(self.source_repository)
        target_repo_info = await self._get_repository_info(self.target_repository)

        stats = {
            "source_timeframe": self.source_timeframe,
            "target_timeframes": self.target_timeframes,
            "supported_conversions": {},
            "conversion_ratios": {},
            "source_repository_type": source_repo_info["storage_type"],
            "target_repository_type": target_repo_info["storage_type"],
            "timeframe_info": self.timeframe_converter.get_timeframe_info(),
        }

        for target_tf in self.target_timeframes:
            is_supported = self.timeframe_converter.can_convert(
                self.source_timeframe, target_tf
            )
            stats["supported_conversions"][target_tf] = is_supported

            if is_supported:
                ratio = self.timeframe_converter.get_conversion_ratio(
                    self.source_timeframe, target_tf
                )
                stats["conversion_ratios"][target_tf] = ratio

        return stats


class CrossRepositoryConfig:
    """Configuration for cross-repository operations"""

    def __init__(
        self,
        source_repo_type: RepositoryType,
        target_repo_type: RepositoryType,
        source_repo_config: Optional[Dict[str, Any]] = None,
        target_repo_config: Optional[Dict[str, Any]] = None,
    ):
        self.source_repo_type = source_repo_type
        self.target_repo_type = target_repo_type
        self.source_repo_config = source_repo_config or {}
        self.target_repo_config = target_repo_config or {}


def create_cross_repository_pipeline(
    config: "CrossRepositoryConfig",
    source_timeframe: str = TimeFrame.HOUR_1.value,
    target_timeframes: Optional[List[str]] = None,
    timeframe_utils: Optional[TimeFrameUtils] = None,
) -> CrossRepositoryTimeFramePipeline:
    """
    Factory function to create cross-repository pipeline.

    Args:
        config: Cross-repository configuration
        source_timeframe: Source timeframe for loading data
        target_timeframes: List of target timeframes to convert to
        timeframe_utils: Optional TimeFrameUtils instance for dependency injection

    Returns:
        CrossRepositoryTimeFramePipeline instance
    """
    factory = MarketDataRepositoryFactory()

    # Create source repository
    source_repository = factory.create_repository(
        config.source_repo_type, config.source_repo_config
    )

    # Create target repository
    target_repository = factory.create_repository(
        config.target_repo_type, config.target_repo_config
    )

    return CrossRepositoryTimeFramePipeline(
        source_repository=source_repository,
        target_repository=target_repository,
        source_timeframe=source_timeframe,
        target_timeframes=target_timeframes,
        timeframe_utils=timeframe_utils,
    )


async def run_mongodb_to_csv_example():
    """Example: Load from MongoDB, save to CSV"""

    logger = LoggerFactory.get_logger("MongoDBToCSVExample")
    logger.info("Starting MongoDB to CSV conversion example")

    # Configure cross-repository setup
    config = CrossRepositoryConfig(
        source_repo_type=RepositoryType.MONGODB,
        target_repo_type=RepositoryType.CSV,
    )

    # Create pipeline
    pipeline = create_cross_repository_pipeline(
        config=config,
        source_timeframe=TimeFrame.HOUR_1.value,
        target_timeframes=[
            # TimeFrame.HOUR_2.value,
            TimeFrame.HOUR_4.value,
            TimeFrame.HOUR_6.value,
            TimeFrame.HOUR_12.value,
            TimeFrame.DAY_1.value,
        ],
    )

    # Define symbols to process
    symbols = [
        CryptoSymbol.BTCUSDT.value,
        CryptoSymbol.ETHUSDT.value,
        CryptoSymbol.BNBUSDT.value,
    ]

    # Run pipeline
    results = await pipeline.run_cross_repository_pipeline(
        symbols=symbols,
        exchange=Exchange.BINANCE.value,
        start_date="2013-01-01T00:00:00Z",
        # end_date="2024-12-31T23:59:59Z",
        overwrite_existing=False,
    )

    logger.info("MongoDB to CSV conversion results:")
    logger.info(f"Symbols processed: {results['symbols_processed']}")
    logger.info(f"Success rate: {results.get('success_rate', 0):.1f}%")
    logger.info(
        f"Total conversions: {results['successful_conversions']}/{results['total_conversions']}"
    )

    return results


async def run_csv_to_mongodb_example():
    """Example: Load from CSV, save to MongoDB"""

    logger = LoggerFactory.get_logger("CSVToMongoDBExample")
    logger.info("Starting CSV to MongoDB conversion example")

    # Configure cross-repository setup
    config = CrossRepositoryConfig(
        source_repo_type=RepositoryType.CSV,
        target_repo_type=RepositoryType.MONGODB,
        source_repo_config={"base_directory": "data"},
        target_repo_config={
            "connection_string": "mongodb://localhost:27017/",
            "database_name": "finsight_converted_data",
            "ohlcv_collection": "ohlcv_converted",
        },
    )

    # Create pipeline
    pipeline = create_cross_repository_pipeline(
        config=config,
        source_timeframe=TimeFrame.HOUR_1.value,
        target_timeframes=[
            TimeFrame.HOUR_4.value,
            TimeFrame.DAY_1.value,
        ],
    )

    # Define symbols to process
    symbols = [
        CryptoSymbol.BTCUSDT.value,
        CryptoSymbol.ETHUSDT.value,
    ]

    # Run pipeline
    results = await pipeline.run_cross_repository_pipeline(
        symbols=symbols,
        exchange=Exchange.BINANCE.value,
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-06-30T23:59:59Z",
        overwrite_existing=False,
    )

    logger.info("CSV to MongoDB conversion results:")
    logger.info(f"Symbols processed: {results['symbols_processed']}")
    logger.info(f"Success rate: {results.get('success_rate', 0):.1f}%")
    logger.info(
        f"Total conversions: {results['successful_conversions']}/{results['total_conversions']}"
    )

    return results


async def main():
    """Main function to demonstrate cross-repository timeframe conversion"""

    logger = LoggerFactory.get_logger("CrossRepositoryMain")
    logger.info("Starting cross-repository timeframe conversion demonstration")

    try:
        print("🚀 Cross-Repository Timeframe Conversion Demo")
        print("=" * 80)

        # Run MongoDB to CSV example
        print("\n1. MongoDB → CSV Conversion")
        print("-" * 40)
        try:
            mongodb_to_csv_results = await run_mongodb_to_csv_example()
            print(
                f"✅ MongoDB to CSV: {mongodb_to_csv_results.get('success_rate', 0):.1f}% success rate"
            )
        except Exception as e:
            logger.warning(f"MongoDB to CSV conversion failed: {e}")
            print(f"⚠️  MongoDB to CSV conversion skipped: {e}")

        # Run CSV to MongoDB example
        # print("\n2. CSV → MongoDB Conversion")
        # print("-" * 40)
        # try:
        #     csv_to_mongodb_results = await run_csv_to_mongodb_example()
        #     print(
        #         f"✅ CSV to MongoDB: {csv_to_mongodb_results.get('success_rate', 0):.1f}% success rate"
        #     )
        # except Exception as e:
        #     logger.warning(f"CSV to MongoDB conversion failed: {e}")
        #     print(f"⚠️  CSV to MongoDB conversion skipped: {e}")

        print("\n✨ Cross-repository demonstration completed!")
        logger.info("Cross-repository demonstration completed successfully")

    except Exception as e:
        logger.error(f"Cross-repository demonstration failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
