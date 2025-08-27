# backtesting_demo_pipeline.py

"""
Comprehensive Backtesting Demo Pipeline

This demo showcases the complete backtesting system interaction flow:
1. Frontend sends backtest request to Backtesting Service
2. Backtesting Service retrieves market data via Market Data Service
3. Strategy is instantiated and configured
4. Backtest is executed via Backtrader Adapter
5. Results are processed and returned

Demonstrates the Ports & Adapters (Hexagonal Architecture) and Strategy patterns.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict

from common.logger import LoggerFactory

from ..adapters.backtesting.backtrader_adapter import BacktraderAdapter
from ..adapters.binance_market_data_collector import BinanceMarketDataCollector
from ..factories import create_repository
from ..schemas.backtesting_schemas import (BacktestRequest, BacktestResult,
                                           StrategyType)
from ..schemas.enums import CryptoSymbol, Exchange, TimeFrame
from ..schemas.ohlcv_schemas import OHLCVQuerySchema
from ..services.backtesting_service import BacktestingService
from ..services.market_data_collector_service import MarketDataCollectorService
from ..services.market_data_service import MarketDataService


class BacktestingDemoPipeline:
    """
    Demo pipeline showcasing complete backtesting workflow.

    Demonstrates the interaction between all components in the system:
    - Market Data Service (data retrieval)
    - Strategy Factory (strategy creation)
    - Backtrader Adapter (backtesting execution)
    - Backtesting Service (orchestration)
    """

    def __init__(self):
        """Initialize the demo pipeline with all required components."""
        self.logger = LoggerFactory.get_logger(name="backtesting_demo")

        # Initialize components following dependency injection pattern
        self._setup_components()

    def _setup_components(self):
        """Setup all required components with proper dependency injection."""
        self.logger.info("Setting up backtesting demo components...")

        # 1. Market Data Repository (switchable via factory pattern)
        self.market_data_repository = create_repository(
            "mongodb",
            {
                "connection_string": "mongodb://localhost:27017/",
                "database_name": "finsight_market_data",
            },
        )

        # 2. Market Data Services
        self.market_data_collector = BinanceMarketDataCollector()
        self.market_data_service = MarketDataService(self.market_data_repository)
        self.market_data_collector_service = MarketDataCollectorService(
            self.market_data_collector, self.market_data_service
        )

        # 3. Backtesting Engine (using Adapter pattern)
        self.backtesting_engine = BacktraderAdapter()

        # 4. Backtesting Service (Aggregator pattern)
        self.backtesting_service = BacktestingService(
            self.market_data_service, self.backtesting_engine
        )

        self.logger.info("All components initialized successfully")

    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """
        Run comprehensive demo showcasing all strategies and features.

        Returns:
            Dictionary containing all demo results
        """
        self.logger.info("🚀 Starting comprehensive backtesting demo...")

        demo_results = {
            "demo_info": {
                "start_time": datetime.now(),
                "strategies_tested": [],
                "symbols_tested": [],
                "performance_summary": {},
            },
            "strategy_results": {},
            "errors": [],
        }

        try:
            # Demo configuration
            symbols = [CryptoSymbol.BTCUSDT.value, CryptoSymbol.ETHUSDT.value]
            timeframe = TimeFrame.HOUR_1.value
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days

            # Demo strategies with different parameters
            strategies_to_test = [
                {
                    "name": "Moving Average Crossover",
                    "type": StrategyType.MOVING_AVERAGE_CROSSOVER,
                    "params": {
                        "fast_period": 10,
                        "slow_period": 20,
                        "ma_type": "sma",
                        "position_sizing": "percentage",
                        "position_size": 0.5,  # 50% of capital per trade
                    },
                },
                {
                    "name": "RSI Strategy",
                    "type": StrategyType.RSI_STRATEGY,
                    "params": {
                        "rsi_period": 14,
                        "oversold_threshold": 30,
                        "overbought_threshold": 70,
                        "position_sizing": "percentage",
                        "position_size": 0.4,
                    },
                },
                {
                    "name": "Bollinger Bands Mean Reversion",
                    "type": StrategyType.BOLLINGER_BANDS,
                    "params": {
                        "bb_period": 20,
                        "bb_std_dev": 2.0,
                        "bb_position": "mean_reversion",
                        "position_sizing": "percentage",
                        "position_size": 0.6,
                    },
                },
                {
                    "name": "MACD Strategy",
                    "type": StrategyType.MACD_STRATEGY,
                    "params": {
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                        "histogram_threshold": 0.001,
                        "position_sizing": "percentage",
                        "position_size": 0.5,
                    },
                },
                {
                    "name": "Simple Buy & Hold",
                    "type": StrategyType.SIMPLE_BUY_HOLD,
                    "params": {
                        "position_sizing": "percentage",
                        "position_size": 1.0,  # 100% investment
                    },
                },
            ]

            # Test each strategy on each symbol
            for symbol in symbols:
                demo_results["strategy_results"][symbol] = {}
                demo_results["demo_info"]["symbols_tested"].append(symbol)

                self.logger.info(f"\n📊 Testing strategies for {symbol}...")

                # Ensure we have market data
                await self._ensure_market_data(symbol, timeframe, start_date, end_date)

                for strategy_config in strategies_to_test:
                    strategy_name = strategy_config["name"]
                    self.logger.info(f"  🔍 Testing {strategy_name}...")

                    try:
                        # Run backtest for this strategy
                        result = await self._run_single_backtest(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date,
                            strategy_type=strategy_config["type"],
                            strategy_params=strategy_config["params"],
                        )

                        demo_results["strategy_results"][symbol][strategy_name] = result
                        demo_results["demo_info"]["strategies_tested"].append(
                            f"{symbol}:{strategy_name}"
                        )

                        # Log results
                        metrics = result.metrics
                        self.logger.info(
                            f"    ✅ {strategy_name}: "
                            f"Return: {metrics.total_return:.2f}%, "
                            f"Sharpe: {metrics.sharpe_ratio:.3f}, "
                            f"Max DD: {metrics.max_drawdown:.2f}%, "
                            f"Trades: {metrics.total_trades}"
                        )

                    except Exception as e:
                        error_msg = (
                            f"Error testing {strategy_name} on {symbol}: {str(e)}"
                        )
                        self.logger.error(f"    ❌ {error_msg}")
                        demo_results["errors"].append(error_msg)

            # Generate performance summary
            demo_results["performance_summary"] = self._generate_performance_summary(
                demo_results["strategy_results"]
            )

            demo_results["demo_info"]["end_time"] = datetime.now()
            demo_results["demo_info"]["duration_seconds"] = (
                demo_results["demo_info"]["end_time"]
                - demo_results["demo_info"]["start_time"]
            ).total_seconds()

            self.logger.info(
                f"\n🎉 Demo completed in {demo_results['demo_info']['duration_seconds']:.2f} seconds!"
            )
            self._print_summary(demo_results)

            return demo_results

        except Exception as e:
            self.logger.error(f"Demo failed: {str(e)}")
            demo_results["errors"].append(f"Demo execution failed: {str(e)}")
            return demo_results

    async def _ensure_market_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ):
        """Ensure we have market data for the specified period."""
        self.logger.info(f"  📈 Ensuring market data for {symbol} ({timeframe})...")

        try:
            # Check if we have data
            query = OHLCVQuerySchema(
                exchange=Exchange.BINANCE.value,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                limit=100,
            )

            existing_data = self.market_data_service.get_ohlcv_data(query)

            if len(existing_data) < 50:  # Need minimum data for strategies
                self.logger.info(f"    📥 Collecting market data for {symbol}...")

                await self.market_data_collector_service.collect_and_store_ohlcv(
                    exchange=Exchange.BINANCE.value,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )
                self.logger.info(
                    f"    ✅ Market data collection completed for {symbol}"
                )
            else:
                self.logger.info(
                    f"    ✅ Sufficient market data available for {symbol}"
                )

        except Exception as e:
            self.logger.warning(
                f"    ⚠️ Market data collection failed for {symbol}: {str(e)}"
            )

    async def _run_single_backtest(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        strategy_type: StrategyType,
        strategy_params: Dict[str, Any],
    ) -> BacktestResult:
        """Run a single backtest with specified parameters."""

        # Create backtest request
        request = BacktestRequest(
            symbol=symbol,
            timeframe=timeframe,
            exchange=Exchange.BINANCE.value,
            start_date=start_date,
            end_date=end_date,
            strategy_type=strategy_type,
            strategy_params=strategy_params,
            initial_capital=10000.0,  # $10K starting capital
            commission=0.001,  # 0.1% commission
            max_position_size=1.0,  # Max 100% position
        )

        # Execute backtest via service
        result = await self.backtesting_service.run_backtest(request)
        return result

    def _generate_performance_summary(self, strategy_results: Dict) -> Dict[str, Any]:
        """Generate performance summary across all strategies and symbols."""
        summary = {
            "best_strategy": {"name": "", "symbol": "", "return": float("-inf")},
            "worst_strategy": {"name": "", "symbol": "", "return": float("inf")},
            "strategy_rankings": [],
            "symbol_performance": {},
            "average_metrics": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            },
        }

        all_results = []

        for symbol, strategies in strategy_results.items():
            symbol_perf = {"strategies": 0, "avg_return": 0.0, "best_strategy": ""}
            symbol_returns = []

            for strategy_name, result in strategies.items():
                if hasattr(result, "metrics"):
                    metrics = result.metrics
                    return_pct = metrics.total_return

                    # Track best and worst
                    if return_pct > summary["best_strategy"]["return"]:
                        summary["best_strategy"] = {
                            "name": strategy_name,
                            "symbol": symbol,
                            "return": return_pct,
                        }

                    if return_pct < summary["worst_strategy"]["return"]:
                        summary["worst_strategy"] = {
                            "name": strategy_name,
                            "symbol": symbol,
                            "return": return_pct,
                        }

                    # Add to rankings
                    summary["strategy_rankings"].append(
                        {
                            "strategy": strategy_name,
                            "symbol": symbol,
                            "return": return_pct,
                            "sharpe": metrics.sharpe_ratio or 0.0,
                            "max_drawdown": metrics.max_drawdown,
                            "trades": metrics.total_trades,
                        }
                    )

                    all_results.append(metrics)
                    symbol_returns.append(return_pct)

            if symbol_returns:
                symbol_perf["strategies"] = len(symbol_returns)
                symbol_perf["avg_return"] = sum(symbol_returns) / len(symbol_returns)
                symbol_perf["best_strategy"] = max(
                    strategies.keys(),
                    key=lambda x: (
                        strategies[x].metrics.total_return
                        if hasattr(strategies[x], "metrics")
                        else 0
                    ),
                )

            summary["symbol_performance"][symbol] = symbol_perf

        # Calculate averages
        if all_results:
            summary["average_metrics"]["total_return"] = sum(
                r.total_return for r in all_results
            ) / len(all_results)
            summary["average_metrics"]["sharpe_ratio"] = sum(
                r.sharpe_ratio or 0 for r in all_results
            ) / len(all_results)
            summary["average_metrics"]["max_drawdown"] = sum(
                r.max_drawdown for r in all_results
            ) / len(all_results)
            summary["average_metrics"]["win_rate"] = sum(
                r.win_rate for r in all_results
            ) / len(all_results)

        # Sort rankings by return
        summary["strategy_rankings"].sort(key=lambda x: x["return"], reverse=True)

        return summary

    def _print_summary(self, demo_results: Dict[str, Any]):
        """Print a comprehensive summary of demo results."""
        print("\n" + "=" * 80)
        print("🎯 BACKTESTING DEMO SUMMARY")
        print("=" * 80)

        summary = demo_results["performance_summary"]
        demo_info = demo_results["demo_info"]

        print(f"\n📊 EXECUTION INFO:")
        print(f"   Duration: {demo_info['duration_seconds']:.2f} seconds")
        print(f"   Symbols tested: {len(demo_info['symbols_tested'])}")
        print(f"   Strategies tested: {len(demo_info['strategies_tested'])}")
        print(f"   Errors: {len(demo_results['errors'])}")

        if summary["strategy_rankings"]:
            print(f"\n🏆 TOP PERFORMING STRATEGIES:")
            for i, ranking in enumerate(summary["strategy_rankings"][:5], 1):
                print(
                    f"   {i}. {ranking['strategy']} ({ranking['symbol']}): "
                    f"{ranking['return']:.2f}% return, "
                    f"Sharpe: {ranking['sharpe']:.3f}"
                )

            print(f"\n📈 BEST STRATEGY:")
            best = summary["best_strategy"]
            print(
                f"   {best['name']} on {best['symbol']}: {best['return']:.2f}% return"
            )

            print(f"\n📉 WORST STRATEGY:")
            worst = summary["worst_strategy"]
            print(
                f"   {worst['name']} on {worst['symbol']}: {worst['return']:.2f}% return"
            )

            print(f"\n📊 AVERAGE METRICS:")
            avg = summary["average_metrics"]
            print(f"   Return: {avg['total_return']:.2f}%")
            print(f"   Sharpe Ratio: {avg['sharpe_ratio']:.3f}")
            print(f"   Max Drawdown: {avg['max_drawdown']:.2f}%")
            print(f"   Win Rate: {avg['win_rate']:.1f}%")

        if demo_results["errors"]:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            for error in demo_results["errors"]:
                print(f"   - {error}")

        print("\n" + "=" * 80)

    async def run_simple_demo(self, symbol: str = "BTCUSDT") -> BacktestResult:
        """
        Run a simple demo with a single strategy for quick testing.

        Args:
            symbol: Trading symbol to test

        Returns:
            Backtest result
        """
        self.logger.info(f"🚀 Running simple demo for {symbol}...")

        try:
            # Demo parameters
            timeframe = TimeFrame.HOUR_1.value
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Last 7 days

            # Ensure market data
            await self._ensure_market_data(symbol, timeframe, start_date, end_date)

            # Run simple moving average strategy
            result = await self._run_single_backtest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy_type=StrategyType.MOVING_AVERAGE_CROSSOVER,
                strategy_params={
                    "fast_period": 5,
                    "slow_period": 10,
                    "ma_type": "sma",
                    "position_sizing": "percentage",
                    "position_size": 0.8,
                },
            )

            self.logger.info(f"✅ Simple demo completed for {symbol}")
            self.logger.info(f"   Return: {result.metrics.total_return:.2f}%")
            self.logger.info(f"   Trades: {result.metrics.total_trades}")
            self.logger.info(f"   Win Rate: {result.metrics.win_rate:.1f}%")

            return result

        except Exception as e:
            self.logger.error(f"Simple demo failed: {str(e)}")
            raise


# Demo execution functions
async def run_comprehensive_demo():
    """Run the comprehensive backtesting demo."""
    pipeline = BacktestingDemoPipeline()
    return await pipeline.run_comprehensive_demo()


async def run_simple_demo(symbol: str = "BTCUSDT"):
    """Run a simple demo for quick testing."""
    pipeline = BacktestingDemoPipeline()
    return await pipeline.run_simple_demo(symbol)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtesting Demo Pipeline")
    parser.add_argument(
        "--mode",
        choices=["simple", "comprehensive"],
        default="simple",
        help="Demo mode to run",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol for simple demo")

    args = parser.parse_args()

    if args.mode == "comprehensive":
        print("Running comprehensive backtesting demo...")
        asyncio.run(run_comprehensive_demo())
    else:
        print(f"Running simple backtesting demo for {args.symbol}...")
        asyncio.run(run_simple_demo(args.symbol))
