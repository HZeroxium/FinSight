# adapters/backtesting/backtrader_adapter.py

"""
Backtrader adapter implementation.
Implements Adapter Pattern to integrate Backtrader engine.
"""

import backtrader as bt
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from ...interfaces.backtesting_engine import BacktestingEngine, BacktestingEngineError
from ...schemas.backtesting_schemas import (
    BacktestRequest,
    BacktestResult,
    StrategyConfig,
    TradeResult,
    PerformanceMetrics,
    EquityCurvePoint,
    PositionSide,
)
from ...schemas.ohlcv_schemas import OHLCVSchema
from ...strategies.strategy_factory import StrategyFactory
from common.logger import LoggerFactory


class BacktraderStrategy(bt.Strategy):
    """
    Backtrader strategy wrapper that uses our strategy implementations.
    """

    params = (("strategy_instance", None),)

    def __init__(self):
        self.strategy = self.params.strategy_instance
        self.logger = LoggerFactory.get_logger(name="backtrader_strategy")
        self.data_index = 0
        self.trades_log = []

        # Convert backtrader data to our format for strategy
        self.ohlcv_data = self._convert_bt_data_to_ohlcv()

    def _convert_bt_data_to_ohlcv(self) -> List[OHLCVSchema]:
        """Convert backtrader data to our OHLCV schema format."""
        ohlcv_data = []

        # Get all available data points
        for i in range(len(self.data)):
            try:
                timestamp = bt.num2date(self.data.datetime[i])

                ohlcv = OHLCVSchema(
                    timestamp=timestamp,
                    open=float(self.data.open[i]),
                    high=float(self.data.high[i]),
                    low=float(self.data.low[i]),
                    close=float(self.data.close[i]),
                    volume=float(self.data.volume[i]),
                    symbol="",  # Will be set by adapter
                    exchange="",  # Will be set by adapter
                    timeframe="",  # Will be set by adapter
                )
                ohlcv_data.append(ohlcv)
            except Exception as e:
                self.logger.warning(f"Error converting data point {i}: {e}")
                continue

        return ohlcv_data

    def next(self):
        """Called for each bar in backtrader."""
        try:
            # Update our strategy's portfolio value
            current_price = float(self.data.close[0])
            self.strategy.update_portfolio_value(current_price)

            # Generate signal using our strategy
            signal = self.strategy.generate_signals(self.ohlcv_data, self.data_index)

            # Execute trades based on signal
            if signal.get("action") == "buy" and not self.position:
                size = self._calculate_position_size(current_price)
                if size > 0:
                    order = self.buy(size=size)
                    self.trades_log.append(
                        {
                            "timestamp": bt.num2date(self.data.datetime[0]),
                            "action": "buy",
                            "price": current_price,
                            "size": size,
                            "reason": signal.get("reason", "Signal"),
                            "order": order,
                        }
                    )

            elif signal.get("action") == "sell" and self.position:
                order = self.sell(size=self.position.size)
                self.trades_log.append(
                    {
                        "timestamp": bt.num2date(self.data.datetime[0]),
                        "action": "sell",
                        "price": current_price,
                        "size": self.position.size,
                        "reason": signal.get("reason", "Signal"),
                        "order": order,
                    }
                )

            self.data_index += 1

        except Exception as e:
            self.logger.error(f"Error in strategy execution: {e}")

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on available cash."""
        available_cash = self.broker.getcash()
        max_position_value = available_cash * 0.95  # Use 95% of available cash
        return max_position_value / price

    def notify_trade(self, trade):
        """Called when a trade is completed."""
        if trade.isclosed:
            self.logger.info(
                f"Trade completed: PnL: {trade.pnl:.2f}, "
                f"Commission: {trade.commission:.2f}"
            )


class BacktraderAdapter(BacktestingEngine):
    """
    Backtrader engine adapter implementation.

    Converts our backtesting interface to Backtrader's format and vice versa.
    """

    def __init__(self):
        """Initialize Backtrader adapter."""
        self.logger = LoggerFactory.get_logger(name="backtrader_adapter")
        self.logger.info("BacktraderAdapter initialized")

    async def run_backtest(
        self,
        request: BacktestRequest,
        market_data: List[OHLCVSchema],
        strategy_config: StrategyConfig,
    ) -> BacktestResult:
        """
        Execute backtest using Backtrader engine.

        Args:
            request: Backtest parameters
            market_data: Historical OHLCV data
            strategy_config: Strategy configuration

        Returns:
            Comprehensive backtest results
        """
        start_time = datetime.now()

        try:
            # Validate inputs
            if not market_data:
                raise BacktestingEngineError("No market data provided")

            # Create and configure Cerebro
            cerebro = bt.Cerebro()

            # Set initial capital
            cerebro.broker.setcash(request.initial_capital)

            # Set commission
            cerebro.broker.setcommission(commission=request.commission)

            # Convert our data to Backtrader format
            bt_data = self._convert_ohlcv_to_backtrader(
                market_data, request.symbol, request.exchange, request.timeframe
            )
            cerebro.adddata(bt_data)

            # Create strategy instance
            strategy_instance = StrategyFactory.create_strategy(strategy_config)
            strategy_instance.initialize(request.initial_capital)

            # Update OHLCV data with request metadata
            for ohlcv in market_data:
                ohlcv.symbol = request.symbol
                ohlcv.exchange = request.exchange
                ohlcv.timeframe = request.timeframe

            # Add strategy to Cerebro
            cerebro.addstrategy(BacktraderStrategy, strategy_instance=strategy_instance)

            # Add analyzers for performance metrics
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            cerebro.addanalyzer(bt.analyzers.Calmar, _name="calmar")

            # Run backtest
            self.logger.info(f"Starting backtest for {request.symbol}")
            results = cerebro.run()

            if not results:
                raise BacktestingEngineError("Backtest failed to produce results")

            # Extract results
            strategy_result = results[0]
            final_value = cerebro.broker.getvalue()

            # Build comprehensive result
            backtest_result = self._build_backtest_result(
                request=request,
                strategy_config=strategy_config,
                strategy_result=strategy_result,
                initial_capital=request.initial_capital,
                final_capital=final_value,
                market_data=market_data,
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

            self.logger.info(
                f"Backtest completed for {request.symbol}. "
                f"Return: {backtest_result.metrics.total_return:.2f}%"
            )

            return backtest_result

        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}")
            raise BacktestingEngineError(f"Backtest failed: {str(e)}")

    def _convert_ohlcv_to_backtrader(
        self, ohlcv_data: List[OHLCVSchema], symbol: str, exchange: str, timeframe: str
    ) -> bt.feeds.PandasData:
        """
        Convert OHLCV data to Backtrader pandas data feed.

        Args:
            ohlcv_data: Our OHLCV data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            Backtrader data feed
        """
        # Convert to pandas DataFrame
        data_list = []
        for ohlcv in ohlcv_data:
            data_list.append(
                {
                    "datetime": ohlcv.timestamp,
                    "open": ohlcv.open,
                    "high": ohlcv.high,
                    "low": ohlcv.low,
                    "close": ohlcv.close,
                    "volume": ohlcv.volume,
                }
            )

        df = pd.DataFrame(data_list)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        # Create Backtrader data feed
        data_feed = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=None,
        )

        return data_feed

    def _build_backtest_result(
        self,
        request: BacktestRequest,
        strategy_config: StrategyConfig,
        strategy_result: bt.Strategy,
        initial_capital: float,
        final_capital: float,
        market_data: List[OHLCVSchema],
        execution_time: float,
    ) -> BacktestResult:
        """Build comprehensive backtest result from Backtrader output."""

        # Calculate duration
        start_date = request.start_date
        end_date = request.end_date
        duration_days = (end_date - start_date).days

        # Extract analyzer results
        trades_analyzer = strategy_result.analyzers.trades.get_analysis()
        sharpe_analyzer = strategy_result.analyzers.sharpe.get_analysis()
        drawdown_analyzer = strategy_result.analyzers.drawdown.get_analysis()
        returns_analyzer = strategy_result.analyzers.returns.get_analysis()

        # Build performance metrics
        metrics = self._build_performance_metrics(
            trades_analyzer=trades_analyzer,
            sharpe_analyzer=sharpe_analyzer,
            drawdown_analyzer=drawdown_analyzer,
            returns_analyzer=returns_analyzer,
            initial_capital=initial_capital,
            final_capital=final_capital,
            duration_days=duration_days,
        )

        # Extract trades
        trades = self._extract_trades(strategy_result)

        # Build equity curve
        equity_curve = self._build_equity_curve(market_data, initial_capital)

        return BacktestResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            exchange=request.exchange,
            strategy_type=strategy_config.strategy_type,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            initial_capital=initial_capital,
            final_capital=final_capital,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            execution_time_seconds=execution_time,
            engine_used="backtrader",
            is_valid=True,
            warnings=[],
            errors=[],
        )

    def _build_performance_metrics(
        self,
        trades_analyzer: Dict[str, Any],
        sharpe_analyzer: Dict[str, Any],
        drawdown_analyzer: Dict[str, Any],
        returns_analyzer: Dict[str, Any],
        initial_capital: float,
        final_capital: float,
        duration_days: int,
    ) -> PerformanceMetrics:
        """Build performance metrics from analyzer results."""

        # Basic calculations
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        annual_return = (total_return / duration_days) * 365 if duration_days > 0 else 0

        # Trade statistics
        total_trades = trades_analyzer.get("total", {}).get("total", 0)
        winning_trades = trades_analyzer.get("won", {}).get("total", 0)
        losing_trades = trades_analyzer.get("lost", {}).get("total", 0)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Average win/loss
        average_win = trades_analyzer.get("won", {}).get("pnl", {}).get("average", 0)
        average_loss = abs(
            trades_analyzer.get("lost", {}).get("pnl", {}).get("average", 0)
        )

        # Profit factor
        gross_profit = trades_analyzer.get("won", {}).get("pnl", {}).get("total", 0)
        gross_loss = abs(trades_analyzer.get("lost", {}).get("pnl", {}).get("total", 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Risk metrics
        max_drawdown = drawdown_analyzer.get("max", {}).get("drawdown", 0)
        sharpe_ratio = sharpe_analyzer.get("sharperatio", None)

        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=None,  # Not available in basic analyzers
            calmar_ratio=None,  # Calculate separately if needed
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            volatility=0.0,  # Calculate separately if needed
            var_95=None,
            total_commission=0.0,  # Calculate separately if needed
            holding_period_avg=None,
        )

    def _extract_trades(self, strategy_result: bt.Strategy) -> List[TradeResult]:
        """Extract trade results from strategy."""
        trades = []

        # Access trades from strategy if available
        if hasattr(strategy_result, "trades_log"):
            trade_pairs = []
            open_trade = None

            for trade_log in strategy_result.trades_log:
                if trade_log["action"] == "buy":
                    open_trade = trade_log
                elif trade_log["action"] == "sell" and open_trade:
                    # Calculate PnL
                    entry_value = open_trade["price"] * open_trade["size"]
                    exit_value = trade_log["price"] * trade_log["size"]
                    pnl = exit_value - entry_value
                    pnl_percentage = (pnl / entry_value) * 100

                    trade_result = TradeResult(
                        entry_date=open_trade["timestamp"],
                        exit_date=trade_log["timestamp"],
                        entry_price=open_trade["price"],
                        exit_price=trade_log["price"],
                        position_side=PositionSide.LONG,
                        quantity=open_trade["size"],
                        pnl=pnl,
                        pnl_percentage=pnl_percentage,
                        entry_reason=open_trade["reason"],
                        exit_reason=trade_log["reason"],
                        is_open=False,
                    )
                    trades.append(trade_result)
                    open_trade = None

        return trades

    def _build_equity_curve(
        self, market_data: List[OHLCVSchema], initial_capital: float
    ) -> List[EquityCurvePoint]:
        """Build equity curve from market data."""
        equity_curve = []
        portfolio_value = initial_capital
        peak_value = initial_capital

        for ohlcv in market_data:
            # Simple equity curve - would need actual portfolio tracking for accuracy
            drawdown = (
                ((peak_value - portfolio_value) / peak_value) * 100
                if peak_value > 0
                else 0
            )

            if portfolio_value > peak_value:
                peak_value = portfolio_value
                drawdown = 0

            point = EquityCurvePoint(
                timestamp=ohlcv.timestamp,
                portfolio_value=portfolio_value,
                cash=portfolio_value,  # Simplified
                position_value=0.0,  # Simplified
                drawdown=drawdown,
            )
            equity_curve.append(point)

        return equity_curve

    def validate_strategy_config(self, strategy_config: StrategyConfig) -> bool:
        """Validate strategy configuration for Backtrader."""
        try:
            # Use strategy factory to validate
            StrategyFactory.validate_strategy_config(strategy_config)
            return True
        except Exception as e:
            raise BacktestingEngineError(f"Strategy validation failed: {str(e)}")

    def get_supported_strategies(self) -> List[str]:
        """Get list of supported strategy types."""
        supported = StrategyFactory.get_supported_strategies()
        return [strategy.value for strategy in supported]

    def get_engine_info(self) -> Dict[str, Any]:
        """Get Backtrader engine information."""
        return {
            "name": "Backtrader",
            "version": bt.__version__,
            "description": "Professional Python backtesting library",
            "supported_strategies": self.get_supported_strategies(),
            "features": [
                "Commission modeling",
                "Slippage modeling",
                "Multiple data feeds",
                "Live trading",
                "Comprehensive analyzers",
                "Plotting capabilities",
            ],
            "adapter_version": "1.0.0",
        }
