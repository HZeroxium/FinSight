# ccxt_demo.py

"""
Comprehensive CCXT Demo for Market Data Collection

This module demonstrates all major public endpoints available across different
exchanges using the CCXT library for AI/ML dataset creation.

Features:
- Multi-exchange support with unified interface
- Comprehensive market data fetching (OHLCV, tickers, trades, order books)
- Exchange metadata and capabilities discovery
- Data standardization and storage
- Rate limiting and error handling
- Real-time data collection capabilities

Requirements:
    pip install ccxt pandas numpy ta pyarrow

Author: Expert Software Architect
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import ccxt
from ccxt.base.errors import NetworkError, RequestTimeout, RateLimitExceeded

from .common.logger import LoggerFactory, LoggerType, LogLevel
from .utils import (
    BaseDataCollector,
    ExchangeUtils,
    retry_on_error,
    DataValidator,
)

import pandas as pd


class CCXTDataCollector(BaseDataCollector):
    """Comprehensive CCXT data collector for AI/ML datasets"""

    def __init__(
        self,
        exchange_ids: Optional[List[str]] = None,
        enable_rate_limit: bool = True,
        sandbox: bool = False,
        logger_name: str = "ccxt_collector",
    ):
        """
        Initialize CCXT data collector

        Args:
            exchange_ids: List of exchange IDs to use
            enable_rate_limit: Whether to enable rate limiting
            sandbox: Whether to use sandbox/testnet when available
            enable_realtime: Whether to enable real-time data collection
            logger_name: Name for the logger instance
        """
        super().__init__(exchange_name="ccxt", logger_name=logger_name)

        self.exchange_ids = exchange_ids or ["binance", "kraken"]
        self.enable_rate_limit = enable_rate_limit
        self.sandbox = sandbox
        self.exchanges: Dict[str, ccxt.Exchange] = {}

        # Initialize data validator
        self.validator = DataValidator()

        # Initialize exchanges
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize CCXT exchange instances"""
        for exchange_id in self.exchange_ids:
            try:
                if hasattr(ccxt, exchange_id):
                    exchange_class = getattr(ccxt, exchange_id)

                    # Get exchange-specific configuration
                    exchange_config = self.config_manager.get_exchange_config(
                        f"ccxt_{exchange_id}"
                    )

                    # Exchange configuration
                    config = {
                        "enableRateLimit": self.enable_rate_limit,
                        "timeout": exchange_config.timeout,
                        "options": {
                            "adjustForTimeDifference": exchange_config.adjust_for_time_difference,
                        },
                    }

                    # Enable sandbox if supported
                    if self.sandbox and exchange_id in ["binance", "kraken"]:
                        config["sandbox"] = True

                    exchange = exchange_class(config)
                    self.exchanges[exchange_id] = exchange

                    self.logger.info(f"Initialized {exchange_id} exchange")
                else:
                    self.logger.warning(f"Exchange {exchange_id} not found in CCXT")

            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_id}: {e}")

        self.logger.info(
            f"Initialized CCXT collector with {len(self.exchanges)} exchanges"
        )

    def get_supported_exchanges(self) -> List[str]:
        """Get list of all supported exchanges in CCXT"""
        return ccxt.exchanges

    def get_exchange_capabilities(self, exchange_id: str) -> Dict[str, Any]:
        """Get capabilities and features of a specific exchange"""
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not initialized")

        exchange = self.exchanges[exchange_id]

        capabilities = {
            "id": exchange.id,
            "name": exchange.name,
            "countries": getattr(exchange, "countries", []),
            "urls": getattr(exchange, "urls", {}),
            "version": getattr(exchange, "version", None),
            "has": exchange.has,
            "timeframes": getattr(exchange, "timeframes", {}),
            "limits": getattr(exchange, "limits", {}),
            "fees": getattr(exchange, "fees", {}),
            "rateLimit": exchange.rateLimit,
            "certified": getattr(exchange, "certified", False),
        }

        # Save capabilities data
        self.storage.save_json(
            capabilities,
            f"{exchange_id}_capabilities",
            subfolder=f"{exchange_id}/exchange_info",
        )

        self.logger.info(f"Retrieved capabilities for {exchange_id}")
        return capabilities

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(NetworkError, RequestTimeout, RateLimitExceeded),
    )
    def fetch_markets(self, exchange_id: str, reload: bool = False) -> Dict[str, Any]:
        """Fetch all available markets for an exchange"""
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not initialized")

        exchange = self.exchanges[exchange_id]

        if reload or not exchange.markets:
            markets = exchange.load_markets(reload=reload)
        else:
            markets = exchange.markets

        # Save markets data
        markets_data = ExchangeUtils.prepare_data_with_metadata(
            markets, exchange_id, "ALL", "markets", markets_count=len(markets)
        )

        self.storage.save_json(
            markets_data, f"{exchange_id}_markets", subfolder=f"{exchange_id}/markets"
        )

        self.logger.info(f"Fetched {len(markets)} markets for {exchange_id}")
        return markets

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(NetworkError, RequestTimeout, RateLimitExceeded),
    )
    def fetch_ticker(
        self, exchange_id: str, symbol: str, validate: bool = True
    ) -> Dict[str, Any]:
        """Fetch ticker data for a specific symbol with validation"""
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not initialized")

        exchange = self.exchanges[exchange_id]

        if not exchange.has.get("fetchTicker"):
            raise NotImplementedError(f"{exchange_id} does not support fetchTicker")

        # Ensure symbol is in correct format
        symbol = ExchangeUtils.normalize_symbol(symbol, "ccxt")

        ticker = exchange.fetch_ticker(symbol)
        ticker["exchange_id"] = exchange_id
        ticker["fetch_timestamp"] = datetime.now().isoformat()

        # Validate data if requested
        if validate:
            validation_report = self.validator.validate_ticker_data(ticker, symbol)
            if not validation_report["is_valid"]:
                self.logger.warning(
                    f"Ticker validation failed for {symbol} on {exchange_id}: {validation_report['issues']}"
                )

        # Save raw data
        filename = ExchangeUtils.create_filename(exchange_id, symbol, "ticker")
        self.storage.save_json(
            ticker, f"{filename}_raw", subfolder=f"{exchange_id}/tickers"
        )

        # Standardize and save
        self.standardize_and_save_ticker(ticker, symbol, f"{exchange_id}_ticker")

        # Store in real-time storage if enabled
        if self.realtime_storage:
            ticker["exchange"] = exchange_id
            self.realtime_storage.store_ticker(ticker)

        self.logger.debug(f"Fetched ticker for {symbol} on {exchange_id}")
        return ticker

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(NetworkError, RequestTimeout, RateLimitExceeded),
    )
    def fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[List[float]]:
        """Fetch OHLCV candlestick data"""
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not initialized")

        exchange = self.exchanges[exchange_id]

        if not exchange.has.get("fetchOHLCV"):
            raise NotImplementedError(f"{exchange_id} does not support fetchOHLCV")

        # Use configured limit if not specified
        if limit is None:
            limit = self.data_config.max_ohlcv_limit

        # Ensure symbol is in correct format
        symbol = ExchangeUtils.normalize_symbol(symbol, "ccxt")

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

        if not ohlcv:
            self.logger.warning(f"No OHLCV data returned for {symbol} on {exchange_id}")
            return ohlcv

        # Save raw data
        filename = ExchangeUtils.create_filename(
            exchange_id, symbol, "ohlcv", timeframe
        )
        ohlcv_data = ExchangeUtils.prepare_data_with_metadata(
            ohlcv,
            exchange_id,
            symbol,
            "ohlcv",
            timeframe=timeframe,
            since=since,
            limit=limit,
        )
        self.storage.save_json(
            ohlcv_data, f"{filename}_raw", subfolder=f"{exchange_id}/ohlcv"
        )

        # Standardize and save
        self.standardize_and_save_ohlcv(ohlcv, symbol, timeframe)

        self.logger.debug(
            f"Fetched {len(ohlcv)} OHLCV candles for {symbol} ({timeframe}) on {exchange_id}"
        )
        return ohlcv

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(NetworkError, RequestTimeout, RateLimitExceeded),
    )
    def fetch_trades(
        self,
        exchange_id: str,
        symbol: str,
        since: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch recent public trades"""
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not initialized")

        exchange = self.exchanges[exchange_id]

        if not exchange.has.get("fetchTrades"):
            raise NotImplementedError(f"{exchange_id} does not support fetchTrades")

        # Use configured limit if not specified
        if limit is None:
            limit = self.data_config.max_trades_limit

        # Ensure symbol is in correct format
        symbol = ExchangeUtils.normalize_symbol(symbol, "ccxt")

        trades = exchange.fetch_trades(symbol, since, limit)

        if not trades:
            self.logger.warning(f"No trades returned for {symbol} on {exchange_id}")
            return trades

        # Add metadata to trades
        for trade in trades:
            trade["exchange_id"] = exchange_id
            trade["fetch_timestamp"] = datetime.now().isoformat()

        # Save raw data
        filename = ExchangeUtils.create_filename(exchange_id, symbol, "trades")
        trades_data = ExchangeUtils.prepare_data_with_metadata(
            trades, exchange_id, symbol, "trades", since=since, limit=limit
        )
        self.storage.save_json(
            trades_data, f"{filename}_raw", subfolder=f"{exchange_id}/trades"
        )

        # Standardize and save
        self.standardize_and_save_trades(trades, symbol)

        self.logger.debug(f"Fetched {len(trades)} trades for {symbol} on {exchange_id}")
        return trades

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(NetworkError, RequestTimeout, RateLimitExceeded),
    )
    def fetch_order_book(
        self, exchange_id: str, symbol: str, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch order book data"""
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not initialized")

        exchange = self.exchanges[exchange_id]

        if not exchange.has.get("fetchOrderBook"):
            raise NotImplementedError(f"{exchange_id} does not support fetchOrderBook")

        # Use configured limit if not specified
        if limit is None:
            limit = self.data_config.max_orderbook_limit

        # Ensure symbol is in correct format
        symbol = ExchangeUtils.normalize_symbol(symbol, "ccxt")

        order_book = exchange.fetch_order_book(symbol, limit)
        order_book["exchange_id"] = exchange_id
        order_book["symbol"] = symbol
        order_book["fetch_timestamp"] = datetime.now().isoformat()

        # Save raw data
        filename = ExchangeUtils.create_filename(exchange_id, symbol, "orderbook")
        self.storage.save_json(
            order_book, f"{filename}_raw", subfolder=f"{exchange_id}/orderbook"
        )

        # Standardize and save
        self.standardize_and_save_orderbook(order_book, symbol)

        self.logger.debug(f"Fetched order book for {symbol} on {exchange_id}")
        return order_book

    def fetch_historical_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        days_back: int = 30,
        chunk_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Fetch historical OHLCV data by chunking requests"""
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not initialized")

        exchange = self.exchanges[exchange_id]

        if not exchange.has.get("fetchOHLCV"):
            raise NotImplementedError(f"{exchange_id} does not support fetchOHLCV")

        # Calculate time range
        start_time, end_time, since = ExchangeUtils.calculate_time_range(days_back)

        # Use configured chunk size if not specified
        if chunk_size is None:
            chunk_size = min(self.data_config.max_ohlcv_limit, 1000)

        all_ohlcv = []
        current_since = since

        self.logger.info(
            f"Fetching {days_back} days of {timeframe} data for {symbol} on {exchange_id}"
        )

        while current_since < int(end_time.timestamp() * 1000):
            try:
                ohlcv_chunk = self.fetch_ohlcv(
                    exchange_id, symbol, timeframe, current_since, chunk_size
                )

                if not ohlcv_chunk:
                    break

                all_ohlcv.extend(ohlcv_chunk)

                # Update since to last timestamp + 1
                current_since = ohlcv_chunk[-1][0] + 1

                # Rate limiting
                time.sleep(exchange.rateLimit / 1000)

            except Exception as e:
                self.logger.error(f"Error fetching historical data chunk: {e}")
                break

        # Remove duplicates and sort
        unique_ohlcv = ExchangeUtils.merge_ohlcv_data([all_ohlcv])

        self.logger.info(f"Fetched {len(unique_ohlcv)} historical candles for {symbol}")
        return unique_ohlcv

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(NetworkError, RequestTimeout, RateLimitExceeded),
    )
    def fetch_ohlcv_with_chunking(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        days_back: int = 30,
        validate: bool = True,
    ) -> List[List[float]]:
        """Fetch OHLCV data with intelligent chunking for large datasets"""
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not initialized")

        exchange = self.exchanges[exchange_id]

        if not exchange.has.get("fetchOHLCV"):
            raise NotImplementedError(f"{exchange_id} does not support fetchOHLCV")

        # Normalize symbol
        symbol = ExchangeUtils.normalize_symbol(symbol, "ccxt")

        # Calculate time range
        start_time, end_time, since = ExchangeUtils.calculate_time_range(days_back)

        # Use exchange-specific chunking
        chunk_size = min(self.data_config.max_ohlcv_limit, 1000)
        all_ohlcv = []
        current_since = since

        self.logger.info(
            f"Fetching {days_back} days of {timeframe} data for {symbol} on {exchange_id}"
        )

        while current_since < int(end_time.timestamp() * 1000):
            try:
                ohlcv_chunk = exchange.fetch_ohlcv(
                    symbol, timeframe, current_since, chunk_size
                )

                if not ohlcv_chunk:
                    break

                all_ohlcv.extend(ohlcv_chunk)

                # Update since to last timestamp + 1
                current_since = ohlcv_chunk[-1][0] + 1

                # Rate limiting
                time.sleep(exchange.rateLimit / 1000)

            except Exception as e:
                self.logger.error(f"Error fetching OHLCV chunk: {e}")
                break

        # Merge and deduplicate
        unique_ohlcv = ExchangeUtils.merge_ohlcv_data([all_ohlcv])

        # Validate merged data if requested
        if validate and unique_ohlcv:
            df = pd.DataFrame(
                unique_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            validation_report = self.validator.validate_ohlcv_data(df, symbol)
            self.logger.info(
                f"OHLCV validation score for {symbol}: {validation_report['quality_score']:.2f}"
            )

        # Save processed data
        if unique_ohlcv:
            filename = ExchangeUtils.create_filename(
                exchange_id, symbol, "ohlcv", timeframe
            )

            # Save raw data with metadata
            ohlcv_data = ExchangeUtils.prepare_data_with_metadata(
                unique_ohlcv,
                exchange_id,
                symbol,
                "ohlcv",
                timeframe=timeframe,
                days_back=days_back,
            )
            self.storage.save_json(
                ohlcv_data, f"{filename}_raw", subfolder=f"{exchange_id}/ohlcv"
            )

            # Standardize and save
            self.standardize_and_save_ohlcv(
                unique_ohlcv, symbol, timeframe, f"{exchange_id}_historical"
            )

        self.logger.info(
            f"Fetched {len(unique_ohlcv)} historical candles for {symbol} on {exchange_id}"
        )
        return unique_ohlcv

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(NetworkError, RequestTimeout, RateLimitExceeded),
    )
    def fetch_trades_with_validation(
        self,
        exchange_id: str,
        symbol: str,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        validate: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fetch recent public trades with optional validation"""
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not initialized")

        exchange = self.exchanges[exchange_id]

        if not exchange.has.get("fetchTrades"):
            raise NotImplementedError(f"{exchange_id} does not support fetchTrades")

        # Use configured limit if not specified
        if limit is None:
            limit = self.data_config.max_trades_limit

        # Ensure symbol is in correct format
        symbol = ExchangeUtils.normalize_symbol(symbol, "ccxt")

        trades = exchange.fetch_trades(symbol, since, limit)

        if not trades:
            self.logger.warning(f"No trades returned for {symbol} on {exchange_id}")
            return trades

        # Add metadata to trades
        for trade in trades:
            trade["exchange_id"] = exchange_id
            trade["fetch_timestamp"] = datetime.now().isoformat()

        # Validate data if requested
        if validate:
            try:
                import pandas as pd

                trades_df = pd.DataFrame(trades)
                validation_report = self.validator.validate_trade_data(
                    trades_df, symbol
                )
                self.logger.info(
                    f"Trades validation score for {symbol}: {validation_report['quality_score']:.2f}"
                )
            except Exception as validation_error:
                self.logger.warning(
                    f"Trade validation error for {symbol}: {validation_error}"
                )

        # Save raw data
        filename = ExchangeUtils.create_filename(exchange_id, symbol, "trades")
        trades_data = ExchangeUtils.prepare_data_with_metadata(
            trades, exchange_id, symbol, "trades", since=since, limit=limit
        )
        self.storage.save_json(
            trades_data, f"{filename}_raw", subfolder=f"{exchange_id}/trades"
        )

        # Standardize and save
        self.standardize_and_save_trades(trades, symbol)

        self.logger.debug(f"Fetched {len(trades)} trades for {symbol} on {exchange_id}")
        return trades

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        exceptions=(NetworkError, RequestTimeout, RateLimitExceeded),
    )
    def fetch_order_book_with_validation(
        self,
        exchange_id: str,
        symbol: str,
        limit: Optional[int] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Fetch order book data with validation"""
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not initialized")

        exchange = self.exchanges[exchange_id]

        if not exchange.has.get("fetchOrderBook"):
            raise NotImplementedError(f"{exchange_id} does not support fetchOrderBook")

        # Use configured limit if not specified
        if limit is None:
            limit = self.data_config.max_orderbook_limit

        # Ensure symbol is in correct format
        symbol = ExchangeUtils.normalize_symbol(symbol, "ccxt")

        order_book = exchange.fetch_order_book(symbol, limit)
        order_book["exchange_id"] = exchange_id
        order_book["symbol"] = symbol
        order_book["fetch_timestamp"] = datetime.now().isoformat()

        # Validate data if requested
        if validate:
            try:
                processed_orderbook = self.processor.standardize_orderbook(
                    order_book, symbol
                )
                ob_validation = self.validator.validate_orderbook_data(
                    processed_orderbook, symbol
                )
                self.logger.info(
                    f"Orderbook validation score for {symbol}: {ob_validation['quality_score']:.2f}"
                )
            except Exception as validation_error:
                self.logger.warning(
                    f"Orderbook validation error for {symbol}: {validation_error}"
                )

        # Save raw data
        filename = ExchangeUtils.create_filename(exchange_id, symbol, "orderbook")
        self.storage.save_json(
            order_book, f"{filename}_raw", subfolder=f"{exchange_id}/orderbook"
        )

        # Standardize and save
        try:
            self.standardize_and_save_orderbook(order_book, symbol)
        except Exception as processing_error:
            self.logger.warning(
                f"Orderbook processing error for {symbol}: {processing_error}"
            )

        self.logger.debug(f"Fetched order book for {symbol} on {exchange_id}")
        return order_book

    def compare_exchanges_advanced(
        self,
        symbol: str,
        exchanges: Optional[List[str]] = None,
        include_orderbook: bool = True,
    ) -> Dict[str, Any]:
        """
        Advanced comparison of market data across multiple exchanges with validation
        """
        exchanges = exchanges or list(self.exchanges.keys())

        comparison_data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "exchanges": {},
            "summary": {},
            "quality_scores": {},
            "spread_analysis": {},
        }

        # Normalize symbol
        normalized_symbol = ExchangeUtils.normalize_symbol(symbol, "ccxt")

        for exchange_id in exchanges:
            if exchange_id not in self.exchanges:
                continue

            try:
                exchange_data = {"ticker": None, "orderbook": None, "validation": {}}

                # Fetch ticker for comparison
                ticker = self.fetch_ticker(
                    exchange_id, normalized_symbol, validate=True
                )
                exchange_data["ticker"] = {
                    "last": ticker.get("last"),
                    "bid": ticker.get("bid"),
                    "ask": ticker.get("ask"),
                    "volume": ticker.get("baseVolume"),
                    "change": ticker.get("change"),
                    "percentage": ticker.get("percentage"),
                    "timestamp": ticker.get("timestamp"),
                }

                # Validate ticker
                validation_report = self.validator.validate_ticker_data(
                    ticker, normalized_symbol
                )
                exchange_data["validation"]["ticker"] = validation_report
                comparison_data["quality_scores"][exchange_id] = validation_report[
                    "quality_score"
                ]

                # Fetch orderbook if requested
                if include_orderbook:
                    try:
                        orderbook = self.fetch_order_book(
                            exchange_id, normalized_symbol
                        )
                        processed_orderbook = self.processor.standardize_orderbook(
                            orderbook, normalized_symbol
                        )

                        if "spread_analysis" in processed_orderbook:
                            comparison_data["spread_analysis"][exchange_id] = (
                                processed_orderbook["spread_analysis"]
                            )

                        exchange_data["orderbook"] = {
                            "bid_levels": len(orderbook.get("bids", [])),
                            "ask_levels": len(orderbook.get("asks", [])),
                            "best_bid": (
                                orderbook.get("bids", [[0]])[0][0]
                                if orderbook.get("bids")
                                else None
                            ),
                            "best_ask": (
                                orderbook.get("asks", [[0]])[0][0]
                                if orderbook.get("asks")
                                else None
                            ),
                        }

                        # Validate orderbook
                        ob_validation = self.validator.validate_orderbook_data(
                            processed_orderbook, normalized_symbol
                        )
                        exchange_data["validation"]["orderbook"] = ob_validation

                    except Exception as e:
                        self.logger.warning(
                            f"Could not fetch orderbook from {exchange_id}: {e}"
                        )

                comparison_data["exchanges"][exchange_id] = exchange_data

            except Exception as e:
                self.logger.error(f"Failed to fetch data from {exchange_id}: {e}")
                comparison_data["exchanges"][exchange_id] = {"error": str(e)}

        # Calculate enhanced summary statistics
        valid_tickers = {
            ex_id: data["ticker"]
            for ex_id, data in comparison_data["exchanges"].items()
            if data.get("ticker") and data["ticker"].get("last") is not None
        }

        if valid_tickers:
            prices = [ticker["last"] for ticker in valid_tickers.values()]
            volumes = [
                ticker.get("volume", 0)
                for ticker in valid_tickers.values()
                if ticker.get("volume")
            ]

            comparison_data["summary"] = {
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": sum(prices) / len(prices),
                "price_spread": max(prices) - min(prices),
                "price_spread_percentage": ((max(prices) - min(prices)) / min(prices))
                * 100,
                "exchanges_count": len(valid_tickers),
                "avg_volume": sum(volumes) / len(volumes) if volumes else 0,
                "avg_quality_score": (
                    sum(comparison_data["quality_scores"].values())
                    / len(comparison_data["quality_scores"])
                    if comparison_data["quality_scores"]
                    else 0
                ),
            }

        # Save comparison data
        filename = ExchangeUtils.create_filename("multi_exchange", symbol, "comparison")
        self.storage.save_json(comparison_data, filename, subfolder="comparisons")

        return comparison_data

    def collect_comprehensive_data(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        days_back: int = 7,
        include_trades: bool = True,
        include_orderbook: bool = True,
        enable_validation: bool = True,
        exchange_id: Optional[str] = None,
        enable_comparison: bool = True,
    ) -> Dict[str, Any]:
        """Collect comprehensive market data with enhanced features"""
        # Use defaults if not specified
        symbols = symbols or self.get_default_symbols()[:2]  # Limit for demo
        timeframes = timeframes or ["1h", "4h"]
        exchanges_to_use = (
            [exchange_id] if exchange_id else list(self.exchanges.keys())[:2]
        )  # Limit for demo

        # Create enhanced collection summary
        collection_summary = self.create_collection_summary(
            symbols=symbols,
            timeframes=timeframes,
            days_back=days_back,
            include_trades=include_trades,
            include_orderbook=include_orderbook,
            enable_validation=enable_validation,
            enable_comparison=enable_comparison,
        )

        self.logger.info(
            f"Starting comprehensive data collection for {len(exchanges_to_use)} exchanges"
        )

        # Collect data for each exchange
        all_validations = []

        for exchange_id in exchanges_to_use:
            if exchange_id not in self.exchanges:
                continue

            try:
                exchange_data = self.collect_exchange_data(
                    exchange_id,
                    symbols,
                    timeframes,
                    days_back,
                    include_trades,
                    include_orderbook,
                    enable_validation,
                )

                # Merge into collection summary with proper key structure
                for symbol, data in exchange_data.items():
                    # Skip non-symbol data
                    if not isinstance(data, dict):
                        continue

                    symbol_key = f"{exchange_id}_{symbol}"
                    collection_summary["collected_data"][symbol_key] = data

                    # Update total items
                    if isinstance(data.get("ohlcv"), dict):
                        collection_summary["total_items"] += sum(data["ohlcv"].values())
                    else:
                        collection_summary["total_items"] += data.get("ohlcv", 0)

                    collection_summary["total_items"] += data.get("trades", 0)
                    collection_summary["total_items"] += data.get("orderbook", 0)
                    if data.get("ticker"):
                        collection_summary["total_items"] += 1

            except Exception as e:
                error_msg = f"Failed to collect data from {exchange_id}: {e}"
                self.logger.error(error_msg)
                collection_summary["errors"].append(error_msg)

        # Perform cross-exchange comparisons if enabled
        if enable_comparison and len(exchanges_to_use) > 1:
            comparison_results = {}

            for symbol in symbols:
                try:
                    comparison = self.compare_exchanges_advanced(
                        symbol, exchanges_to_use, include_orderbook=include_orderbook
                    )
                    comparison_results[symbol] = comparison

                except Exception as e:
                    self.logger.error(f"Failed to compare exchanges for {symbol}: {e}")

            collection_summary["exchange_comparisons"] = comparison_results

        # Calculate comprehensive statistics
        collection_stats = ExchangeUtils.calculate_collection_stats(
            collection_summary["collected_data"]
        )
        collection_summary["statistics"] = collection_stats

        # Add validation summary if available
        if all_validations:
            validation_summary = self.validator.create_validation_summary(
                all_validations
            )
            collection_summary["validation_summary"] = validation_summary

        # Save collection summary
        self.save_collection_summary(
            collection_summary, "ccxt_comprehensive_collection"
        )

        # Log enhanced results
        self.log_collection_results(collection_summary)

        if "validation_summary" in collection_summary:
            val_summary = collection_summary["validation_summary"]
            self.logger.info(
                f"Data quality: {val_summary['valid_datasets']}/{val_summary['total_datasets']} valid datasets"
            )
            self.logger.info(
                f"Average quality score: {val_summary['average_quality_score']:.2f}"
            )

        return collection_summary

    def collect_exchange_data(
        self,
        exchange_id: str,
        symbols: List[str],
        timeframes: List[str],
        days_back: int,
        include_trades: bool = True,
        include_orderbook: bool = True,
        enable_validation: bool = True,
    ) -> Dict[str, Any]:
        """Collect data for a single exchange with validation"""
        exchange_data = {}
        all_validations = []

        for symbol in symbols:
            symbol_data = {"ohlcv": {}, "trades": 0, "orderbook": 0, "ticker": False}

            try:
                # Normalize symbol for this exchange
                normalized_symbol = ExchangeUtils.normalize_symbol(symbol, "ccxt")

                # Collect OHLCV for each timeframe with chunking
                for timeframe in timeframes:
                    try:
                        ohlcv = self.fetch_ohlcv_with_chunking(
                            exchange_id,
                            normalized_symbol,
                            timeframe,
                            days_back,
                            enable_validation,
                        )
                        symbol_data["ohlcv"][timeframe] = len(ohlcv)

                    except Exception as e:
                        error_msg = (
                            f"Failed to fetch {timeframe} OHLCV for {symbol}: {e}"
                        )
                        self.logger.error(error_msg)

                # Collect recent trades with validation
                if include_trades:
                    try:
                        trades = self.fetch_trades(exchange_id, normalized_symbol)
                        symbol_data["trades"] = len(trades)

                        if enable_validation and trades:
                            import pandas as pd

                            trades_df = pd.DataFrame(trades)
                            trade_validation = self.validator.validate_trade_data(
                                trades_df, normalized_symbol
                            )
                            all_validations.append(trade_validation)

                    except Exception as e:
                        self.logger.error(f"Failed to fetch trades for {symbol}: {e}")

                # Collect order book with validation
                if include_orderbook:
                    try:
                        orderbook = self.fetch_order_book(
                            exchange_id, normalized_symbol
                        )
                        symbol_data["orderbook"] = len(orderbook.get("bids", [])) + len(
                            orderbook.get("asks", [])
                        )

                        if enable_validation:
                            try:
                                processed_orderbook = (
                                    self.processor.standardize_orderbook(
                                        orderbook, normalized_symbol
                                    )
                                )
                                ob_validation = self.validator.validate_orderbook_data(
                                    processed_orderbook, normalized_symbol
                                )
                                all_validations.append(ob_validation)
                            except Exception as ob_error:
                                self.logger.warning(
                                    f"Orderbook validation failed for {symbol}: {ob_error}"
                                )

                    except Exception as e:
                        self.logger.error(
                            f"Failed to fetch order book for {symbol}: {e}"
                        )

                # Collect ticker with validation
                try:
                    ticker = self.fetch_ticker(
                        exchange_id, normalized_symbol, validate=enable_validation
                    )
                    symbol_data["ticker"] = True

                    if enable_validation:
                        ticker_validation = self.validator.validate_ticker_data(
                            ticker, normalized_symbol
                        )
                        all_validations.append(ticker_validation)

                except Exception as e:
                    self.logger.error(f"Failed to fetch ticker for {symbol}: {e}")

                exchange_data[symbol] = symbol_data
                self.logger.info(
                    f"Completed data collection for {symbol} on {exchange_id}"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to collect data for {symbol} on {exchange_id}: {e}"
                )

        # Store validation results separately to avoid structure conflicts
        if all_validations:
            # Don't add validations to the main exchange_data to avoid structure issues
            # Instead, return them separately or store them in a different way
            pass

        return exchange_data


def main():
    """Demonstrate enhanced CCXT data collection capabilities"""
    # Initialize logger for main function
    logger = LoggerFactory.get_logger(
        name="ccxt_demo_main",
        logger_type=LoggerType.STANDARD,
        level=LogLevel.INFO,
        use_colors=True,
    )

    try:
        logger.info("Starting Enhanced CCXT Demo")

        # Initialize data collector with enhanced features
        exchanges = ["binance", "kraken"]  # Limit for demo
        collector = CCXTDataCollector(
            exchange_ids=exchanges,
            enable_rate_limit=True,
            sandbox=False,
            enable_realtime=True,  # Enable real-time capabilities
        )

        logger.info(
            f"Initialized collector with exchanges: {list(collector.exchanges.keys())}"
        )

        # Demonstrate capabilities discovery
        for exchange_id in collector.exchanges.keys():
            try:
                capabilities = collector.get_exchange_capabilities(exchange_id)
                logger.info(
                    f"{exchange_id} capabilities: {len(capabilities['has'])} features"
                )

                # Show some key capabilities
                key_features = [
                    "fetchTicker",
                    "fetchOHLCV",
                    "fetchTrades",
                    "fetchOrderBook",
                ]
                available_features = [
                    f for f in key_features if capabilities["has"].get(f)
                ]
                logger.info(f"  Available features: {available_features}")

            except Exception as e:
                logger.error(f"Failed to get capabilities for {exchange_id}: {e}")

        # Demonstrate enhanced data collection
        demo_symbols = ["BTC/USDT", "ETH/USDT"]

        logger.info("Demonstrating comprehensive data collection with validation")

        collection_summary = collector.collect_comprehensive_data(
            symbols=demo_symbols,
            timeframes=["1h", "4h"],
            days_back=3,
            include_trades=True,
            include_orderbook=True,
            enable_validation=True,
            enable_comparison=True,  # Enable cross-exchange comparison
        )

        # Display enhanced results
        logger.info("Enhanced Collection Results:")
        logger.info(f"  Total items: {collection_summary['total_items']}")

        if "validation_summary" in collection_summary:
            val_summary = collection_summary["validation_summary"]
            logger.info(f"  Data quality: {val_summary['average_quality_score']:.2f}")

        if "exchange_comparisons" in collection_summary:
            logger.info("  Cross-exchange price comparison completed")
            for symbol, comparison in collection_summary[
                "exchange_comparisons"
            ].items():
                if comparison.get("summary"):
                    spread = comparison["summary"].get("price_spread_percentage", 0)
                    logger.info(f"    {symbol} price spread: {spread:.2f}%")

        # Cleanup and finalize
        collector.cleanup_and_finalize()

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
