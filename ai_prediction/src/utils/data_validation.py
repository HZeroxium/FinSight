# utils/data_validation.py

"""
Data validation utilities for market data quality assurance.
Provides comprehensive validation for different types of market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from ..common.logger import LoggerFactory, LoggerType, LogLevel


class DataValidator:
    """Comprehensive data validation for market data"""

    def __init__(self, logger_name: str = "data_validator"):
        """
        Initialize DataValidator

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = LoggerFactory.get_logger(
            name=logger_name,
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            use_colors=True,
        )

    def validate_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive validation for OHLCV data

        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            Validation report
        """
        report = {
            "symbol": symbol,
            "data_type": "ohlcv",
            "total_records": len(df),
            "validation_timestamp": datetime.now().isoformat(),
            "issues": [],
            "warnings": [],
            "quality_score": 0.0,
            "is_valid": True,
        }

        if len(df) == 0:
            report["issues"].append("Empty dataset")
            report["is_valid"] = False
            return report

        # Required columns check
        required_columns = ["open", "high", "low", "close", "volume", "timestamp"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            report["issues"].append(f"Missing required columns: {missing_columns}")
            report["is_valid"] = False

        if not report["is_valid"]:
            return report

        # Data type validation
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                report["issues"].append(f"Column {col} is not numeric")

        # OHLC logic validation
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        ).sum()

        if invalid_ohlc > 0:
            report["issues"].append(
                f"Invalid OHLC relationships in {invalid_ohlc} records"
            )

        # Price validation
        negative_prices = (
            (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
        ).sum()

        if negative_prices > 0:
            report["issues"].append(f"Non-positive prices in {negative_prices} records")

        # Volume validation
        negative_volume = (df["volume"] < 0).sum()
        if negative_volume > 0:
            report["issues"].append(f"Negative volume in {negative_volume} records")

        # Timestamp validation
        if "timestamp" in df.columns:
            duplicate_timestamps = df["timestamp"].duplicated().sum()
            if duplicate_timestamps > 0:
                report["warnings"].append(
                    f"Duplicate timestamps: {duplicate_timestamps}"
                )

            # Check for timestamp gaps
            if len(df) > 1:
                df_sorted = df.sort_values("timestamp")
                time_diffs = df_sorted["timestamp"].diff().dropna()
                if len(time_diffs) > 0:
                    median_diff = time_diffs.median()
                    large_gaps = (time_diffs > median_diff * 3).sum()
                    if large_gaps > 0:
                        report["warnings"].append(
                            f"Large timestamp gaps detected: {large_gaps}"
                        )

        # Price continuity validation
        if len(df) > 1:
            price_changes = df["close"].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # >50% change
            if extreme_changes > 0:
                report["warnings"].append(
                    f"Extreme price changes detected: {extreme_changes}"
                )

        # Calculate quality score
        total_issues = len(report["issues"]) + len(report["warnings"]) * 0.5
        report["quality_score"] = max(0, 1 - (total_issues / 10))  # Scale to 0-1

        if len(report["issues"]) == 0:
            report["is_valid"] = True

        self.logger.info(
            f"OHLCV validation for {symbol}: {report['quality_score']:.2f} quality score"
        )
        return report

    def validate_trade_data(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Validate trade data

        Args:
            df: Trade DataFrame
            symbol: Trading symbol

        Returns:
            Validation report
        """
        report = {
            "symbol": symbol,
            "data_type": "trades",
            "total_records": len(df),
            "validation_timestamp": datetime.now().isoformat(),
            "issues": [],
            "warnings": [],
            "quality_score": 0.0,
            "is_valid": True,
        }

        if len(df) == 0:
            report["issues"].append("Empty dataset")
            report["is_valid"] = False
            return report

        # Required columns
        required_columns = [
            "price",
            "qty",
        ]  # Changed from "amount" to "qty" for Binance compatibility
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            report["issues"].append(f"Missing required columns: {missing_columns}")

        # Price validation - ensure we're working with numeric types
        if "price" in df.columns:
            try:
                # Convert to numeric if it's not already
                if not pd.api.types.is_numeric_dtype(df["price"]):
                    price_series = pd.to_numeric(df["price"], errors="coerce")
                else:
                    price_series = df["price"]

                invalid_prices = (price_series <= 0).sum()
                if invalid_prices > 0:
                    report["issues"].append(
                        f"Invalid prices in {invalid_prices} records"
                    )
            except Exception as e:
                report["issues"].append(f"Price validation error: {str(e)}")

        # Amount/Quantity validation - handle both "amount" and "qty" columns
        amount_col = "qty" if "qty" in df.columns else "amount"
        if amount_col in df.columns:
            try:
                # Convert to numeric if it's not already
                if not pd.api.types.is_numeric_dtype(df[amount_col]):
                    amount_series = pd.to_numeric(df[amount_col], errors="coerce")
                else:
                    amount_series = df[amount_col]

                invalid_amounts = (amount_series <= 0).sum()
                if invalid_amounts > 0:
                    report["issues"].append(
                        f"Invalid amounts in {invalid_amounts} records"
                    )
            except Exception as e:
                report["issues"].append(f"Amount validation error: {str(e)}")

        # Side validation - handle different formats
        if "side" in df.columns:
            valid_sides = {"buy", "sell", "BUY", "SELL", "b", "s", True, False}
            try:
                invalid_sides = (~df["side"].isin(valid_sides)).sum()
                if invalid_sides > 0:
                    report["warnings"].append(
                        f"Invalid trade sides in {invalid_sides} records"
                    )
            except Exception as e:
                report["warnings"].append(f"Side validation error: {str(e)}")

        report["quality_score"] = max(0, 1 - len(report["issues"]) * 0.2)
        report["is_valid"] = len(report["issues"]) == 0

        return report

    def validate_orderbook_data(
        self, orderbook: Dict[str, pd.DataFrame], symbol: str
    ) -> Dict[str, Any]:
        """
        Validate orderbook data

        Args:
            orderbook: Orderbook data with 'bids' and 'asks'
            symbol: Trading symbol

        Returns:
            Validation report
        """
        report = {
            "symbol": symbol,
            "data_type": "orderbook",
            "validation_timestamp": datetime.now().isoformat(),
            "issues": [],
            "warnings": [],
            "quality_score": 0.0,
            "is_valid": True,
        }

        required_sides = ["bids", "asks"]
        for side in required_sides:
            if side not in orderbook:
                report["issues"].append(f"Missing {side} data")
                continue

            df = orderbook[side]
            if len(df) == 0:
                report["warnings"].append(f"Empty {side} data")
                continue

            # Price and amount validation
            if "price" in df.columns and "amount" in df.columns:
                invalid_prices = (df["price"] <= 0).sum()
                invalid_amounts = (df["amount"] <= 0).sum()

                if invalid_prices > 0:
                    report["issues"].append(
                        f"Invalid prices in {side}: {invalid_prices}"
                    )
                if invalid_amounts > 0:
                    report["issues"].append(
                        f"Invalid amounts in {side}: {invalid_amounts}"
                    )

                # Price ordering validation
                if side == "bids":
                    # Bids should be in descending order
                    if not df["price"].is_monotonic_decreasing:
                        report["warnings"].append(f"Bids not in descending price order")
                else:
                    # Asks should be in ascending order
                    if not df["price"].is_monotonic_increasing:
                        report["warnings"].append(f"Asks not in ascending price order")

        # Spread validation
        if "bids" in orderbook and "asks" in orderbook:
            if len(orderbook["bids"]) > 0 and len(orderbook["asks"]) > 0:
                best_bid = orderbook["bids"]["price"].max()
                best_ask = orderbook["asks"]["price"].min()

                if best_bid >= best_ask:
                    report["issues"].append("Invalid spread: bid >= ask")

        report["quality_score"] = max(0, 1 - len(report["issues"]) * 0.3)
        report["is_valid"] = len(report["issues"]) == 0

        return report

    def validate_ticker_data(
        self, ticker: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Validate ticker data

        Args:
            ticker: Ticker data
            symbol: Trading symbol

        Returns:
            Validation report
        """
        report = {
            "symbol": symbol,
            "data_type": "ticker",
            "validation_timestamp": datetime.now().isoformat(),
            "issues": [],
            "warnings": [],
            "quality_score": 0.0,
            "is_valid": True,
        }

        # Price fields validation
        price_fields = ["last", "bid", "ask", "high", "low", "open", "close"]
        for field in price_fields:
            if field in ticker and ticker[field] is not None:
                try:
                    price = float(ticker[field])
                    if price <= 0:
                        report["issues"].append(f"Invalid {field} price: {price}")
                except (ValueError, TypeError):
                    report["issues"].append(f"Non-numeric {field} price")

        # Spread validation
        if ticker.get("bid") and ticker.get("ask"):
            try:
                bid = float(ticker["bid"])
                ask = float(ticker["ask"])
                if bid >= ask:
                    report["issues"].append("Invalid spread: bid >= ask")
            except (ValueError, TypeError):
                report["warnings"].append("Could not validate spread due to data types")

        # Volume validation
        volume_fields = ["volume", "baseVolume", "quoteVolume"]
        for field in volume_fields:
            if field in ticker and ticker[field] is not None:
                try:
                    volume = float(ticker[field])
                    if volume < 0:
                        report["issues"].append(f"Negative {field}: {volume}")
                except (ValueError, TypeError):
                    report["warnings"].append(f"Non-numeric {field}")

        report["quality_score"] = max(0, 1 - len(report["issues"]) * 0.2)
        report["is_valid"] = len(report["issues"]) == 0

        return report

    def create_validation_summary(
        self, validation_reports: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create summary from multiple validation reports

        Args:
            validation_reports: List of validation reports

        Returns:
            Summary report
        """
        summary = {
            "total_datasets": len(validation_reports),
            "valid_datasets": 0,
            "average_quality_score": 0.0,
            "common_issues": {},
            "data_types": {},
            "validation_timestamp": datetime.now().isoformat(),
        }

        if not validation_reports:
            return summary

        total_quality = 0
        all_issues = []

        for report in validation_reports:
            if report.get("is_valid"):
                summary["valid_datasets"] += 1

            total_quality += report.get("quality_score", 0)
            all_issues.extend(report.get("issues", []))

            data_type = report.get("data_type", "unknown")
            summary["data_types"][data_type] = (
                summary["data_types"].get(data_type, 0) + 1
            )

        summary["average_quality_score"] = total_quality / len(validation_reports)

        # Count common issues
        from collections import Counter

        issue_counts = Counter(all_issues)
        summary["common_issues"] = dict(issue_counts.most_common(10))

        return summary
