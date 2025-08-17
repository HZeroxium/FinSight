# utils/data_fallback_utils.py

"""
Data Fallback Utilities - Handles intelligent data availability and fallback strategies

This module provides utilities for:
- Finding available datasets for specific symbol/timeframe combinations
- Implementing fallback strategies when exact datasets are not available
- Managing fallback priorities for timeframes (prioritizing DAY_1)
- Providing transparent feedback about which dataset was actually used
"""

from typing import Dict, Any, Optional, List, NamedTuple
from pathlib import Path
from datetime import datetime

from ..schemas.enums import (
    TimeFrame,
    CryptoSymbol,
    DataSelectionPriority,
    FallbackReason,
)
from ..core.constants import FallbackConstants
from ..interfaces.data_loader_interface import IDataLoader
from common.logger.logger_factory import LoggerFactory, LogLevel


class DataSelectionResult(NamedTuple):
    """Result of data selection with fallback information"""

    symbol: str
    timeframe: TimeFrame
    data_path: Optional[Path]
    selection_priority: DataSelectionPriority
    fallback_applied: bool
    original_request: Dict[str, Any]
    fallback_reason: Optional[FallbackReason]
    confidence_score: float


class DataFallbackUtils:
    """
    Utilities for intelligent data selection and fallback strategies.

    This class implements a hierarchical fallback system that ensures users
    always receive data while maintaining transparency about which dataset
    was actually used. It prioritizes timeframe fallback over symbol fallback,
    with DAY_1 timeframe being the most preferred fallback option.
    """

    def __init__(self, data_loader: IDataLoader):
        """
        Initialize the data fallback utilities.

        Args:
            data_loader: Data loader instance for checking data availability
        """
        self.logger = LoggerFactory.get_logger("DataFallbackUtils")
        self.data_loader = data_loader

        # Load fallback configuration
        self.timeframe_priority = FallbackConstants.TIMEFRAME_FALLBACK_PRIORITY
        self.symbol_priority = FallbackConstants.SYMBOL_FALLBACK_PRIORITY

        self.logger.info("Data fallback utilities initialized")

    async def find_available_data(
        self,
        requested_symbol: str,
        requested_timeframe: TimeFrame,
        enable_fallback: bool = True,
    ) -> DataSelectionResult:
        """
        Find available data for prediction, with intelligent fallback.

        Args:
            requested_symbol: The originally requested trading symbol
            requested_timeframe: The originally requested timeframe
            enable_fallback: Whether to enable fallback strategies

        Returns:
            DataSelectionResult containing the selected data and fallback information
        """
        import uuid

        search_id = str(uuid.uuid4())[:8]  # Short unique identifier

        self.logger.info(
            f"🚀 [{search_id}] Starting data availability search for {requested_symbol} {requested_timeframe.value}"
        )

        # Start with exact match attempt
        exact_match = await self._try_exact_data_match(
            requested_symbol, requested_timeframe, search_id
        )

        # Check if exact match was found (data_path can be None if data loader handles paths internally)
        if exact_match and exact_match.confidence_score > 0:
            self.logger.info(
                f"✅ [{search_id}] Found exact data match: {exact_match.symbol} {exact_match.timeframe.value} (confidence: {exact_match.confidence_score})"
            )
            # IMPORTANT: Return immediately when exact match is found
            # Do not continue with fallback strategies
            return exact_match

        self.logger.info(
            f"❌ [{search_id}] Exact match not found for {requested_symbol} {requested_timeframe.value}"
        )

        if not enable_fallback:
            self.logger.info(
                f"🔄 [{search_id}] Fallback disabled, returning exact match result for {requested_symbol} {requested_timeframe.value}"
            )
            # Return the exact match result even if data doesn't exist
            return exact_match or DataSelectionResult(
                symbol=requested_symbol,
                timeframe=requested_timeframe,
                data_path=None,
                selection_priority=DataSelectionPriority.EXACT_MATCH,
                fallback_applied=False,
                original_request={
                    "symbol": requested_symbol,
                    "timeframe": requested_timeframe.value,
                },
                fallback_reason=None,
                confidence_score=0.0,
            )

        # Apply fallback strategies - prioritize timeframe fallback over symbol fallback
        self.logger.info(
            f"🔄 [{search_id}] Applying fallback strategies for {requested_symbol} {requested_timeframe.value}"
        )
        fallback_result = await self._apply_data_fallback_strategies(
            requested_symbol, requested_timeframe
        )

        if fallback_result:
            self.logger.info(
                f"✅ [{search_id}] Applied data fallback: {requested_symbol} {requested_timeframe.value} -> "
                f"{fallback_result.symbol} {fallback_result.timeframe.value} "
                f"({fallback_result.fallback_reason})"
            )
            return fallback_result

        # If all fallback strategies fail, return the best available option
        self.logger.info(
            f"🔄 [{search_id}] All fallback strategies failed, trying best available for {requested_symbol} {requested_timeframe.value}"
        )
        best_available = await self._find_best_available_data_fallback(
            requested_symbol, requested_timeframe
        )

        if best_available:
            self.logger.warning(
                f"⚠️ [{search_id}] Using best available data fallback: {best_available.symbol} "
                f"{best_available.timeframe.value}"
            )
            return best_available

        # Final fallback - return the original request with no data found
        self.logger.error(
            f"❌ [{search_id}] No data available after all fallback attempts for {requested_symbol} {requested_timeframe.value}"
        )
        return DataSelectionResult(
            symbol=requested_symbol,
            timeframe=requested_timeframe,
            data_path=None,
            selection_priority=DataSelectionPriority.EXACT_MATCH,
            fallback_applied=False,
            original_request={
                "symbol": requested_symbol,
                "timeframe": requested_timeframe.value,
            },
            fallback_reason=FallbackReason.EXACT_MATCH_NOT_FOUND,
            confidence_score=0.0,
        )

    async def _try_exact_data_match(
        self, symbol: str, timeframe: TimeFrame, search_id: str
    ) -> Optional[DataSelectionResult]:
        """Try to find an exact match for the requested data."""
        try:
            self.logger.debug(
                f"🔍 [{search_id}] Checking exact data match for {symbol} {timeframe.value}"
            )

            # Check if data exists for the exact symbol/timeframe combination
            data_exists = await self.data_loader.check_data_exists(symbol, timeframe)
            self.logger.debug(
                f"📊 [{search_id}] Data exists check result: {data_exists} for {symbol} {timeframe.value}"
            )

            if data_exists:
                # Try to load a small sample to verify data is accessible
                try:
                    self.logger.debug(
                        f"📥 [{search_id}] Loading sample data for {symbol} {timeframe.value}"
                    )
                    sample_data = await self.data_loader.load_data(symbol, timeframe)
                    if not sample_data.empty:
                        self.logger.info(
                            f"✅ [{search_id}] Exact data match found: {symbol} {timeframe.value} ({len(sample_data)} records)"
                        )
                        return DataSelectionResult(
                            symbol=symbol,
                            timeframe=timeframe,
                            data_path=None,  # Data loader handles the path internally
                            selection_priority=DataSelectionPriority.EXACT_MATCH,
                            fallback_applied=False,
                            original_request={
                                "symbol": symbol,
                                "timeframe": timeframe.value,
                            },
                            fallback_reason=None,
                            confidence_score=1.0,
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ [{search_id}] Data exists but sample is empty: {symbol} {timeframe.value}"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ [{search_id}] Data exists but failed to load sample: {symbol} {timeframe.value} - {e}"
                    )
            else:
                self.logger.debug(
                    f"❌ [{search_id}] Data existence check failed for {symbol} {timeframe.value}"
                )

                # Even if existence check fails, try to load data directly
                # This handles the case where data was loaded from cloud but not yet cached locally
                try:
                    self.logger.debug(
                        f"🔄 [{search_id}] Attempting direct data load for {symbol} {timeframe.value}"
                    )
                    sample_data = await self.data_loader.load_data(symbol, timeframe)
                    if not sample_data.empty:
                        self.logger.info(
                            f"✅ [{search_id}] Direct data load successful: {symbol} {timeframe.value} ({len(sample_data)} records)"
                        )
                        return DataSelectionResult(
                            symbol=symbol,
                            timeframe=timeframe,
                            data_path=None,  # Data loader handles the path internally
                            selection_priority=DataSelectionPriority.EXACT_MATCH,
                            fallback_applied=False,
                            original_request={
                                "symbol": symbol,
                                "timeframe": timeframe.value,
                            },
                            fallback_reason=None,
                            confidence_score=1.0,
                        )
                    else:
                        self.logger.debug(
                            f"⚠️ [{search_id}] Direct data load returned empty data: {symbol} {timeframe.value}"
                        )
                except Exception as e:
                    self.logger.debug(
                        f"⚠️ [{search_id}] Direct data load failed: {symbol} {timeframe.value} - {e}"
                    )

            return DataSelectionResult(
                symbol=symbol,
                timeframe=timeframe,
                data_path=None,
                selection_priority=DataSelectionPriority.EXACT_MATCH,
                fallback_applied=False,
                original_request={
                    "symbol": symbol,
                    "timeframe": timeframe.value,
                },
                fallback_reason=FallbackReason.EXACT_MATCH_NOT_FOUND,
                confidence_score=0.0,
            )

        except Exception as e:
            self.logger.error(f"❌ [{search_id}] Error in exact data match: {e}")
            return None

    async def _apply_data_fallback_strategies(
        self, requested_symbol: str, requested_timeframe: TimeFrame
    ) -> Optional[DataSelectionResult]:
        """Apply configured data fallback strategies."""
        try:
            # Strategy 1: Timeframe fallback (same symbol, different timeframe)
            # This is the primary fallback strategy - prioritize DAY_1
            timeframe_fallback = await self._try_timeframe_data_fallback(
                requested_symbol, requested_timeframe
            )
            if timeframe_fallback:
                return timeframe_fallback

            # Strategy 2: Symbol fallback (different symbol, same timeframe)
            # This is secondary and only used if timeframe fallback fails
            # symbol_fallback = await self._try_symbol_data_fallback(
            #     requested_symbol, requested_timeframe
            # )
            # if symbol_fallback:
            #     return symbol_fallback

            # Strategy 3: Full fallback (different symbol, different timeframe)
            # This is the last resort
            # full_fallback = await self._try_full_data_fallback(
            #     requested_symbol, requested_timeframe
            # )
            # if full_fallback:
            #     return full_fallback

            return None

        except Exception as e:
            self.logger.error(f"Error applying data fallback strategies: {e}")
            return None

    async def _try_timeframe_data_fallback(
        self, symbol: str, original_timeframe: TimeFrame
    ) -> Optional[DataSelectionResult]:
        """Try to find data with the same symbol but different timeframe."""
        try:
            # Get timeframe priority list, prioritizing DAY_1
            fallback_timeframes = [
                tf for tf in self.timeframe_priority if tf != original_timeframe.value
            ]

            for tf_str in fallback_timeframes:
                try:
                    timeframe = TimeFrame(tf_str)

                    # Check if data exists for this symbol/timeframe combination
                    data_exists = await self.data_loader.check_data_exists(
                        symbol, timeframe
                    )

                    if data_exists:
                        # Try to load a small sample to verify data is accessible
                        try:
                            sample_data = await self.data_loader.load_data(
                                symbol, timeframe
                            )
                            if not sample_data.empty:
                                self.logger.info(
                                    f"✅ Timeframe data fallback found: {symbol} {original_timeframe.value} -> {timeframe.value}"
                                )

                                return DataSelectionResult(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    data_path=None,
                                    selection_priority=DataSelectionPriority.TIMEFRAME_FALLBACK,
                                    fallback_applied=True,
                                    original_request={
                                        "symbol": symbol,
                                        "timeframe": original_timeframe.value,
                                    },
                                    fallback_reason=FallbackReason.TIMEFRAME_NOT_AVAILABLE,
                                    confidence_score=0.9,  # High confidence for same symbol
                                )
                        except Exception as e:
                            self.logger.debug(
                                f"Data exists but failed to load sample for {symbol} {timeframe.value}: {e}"
                            )
                            continue

                except ValueError:
                    # Invalid timeframe, skip
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Error in timeframe data fallback: {e}")
            return None

    async def _try_symbol_data_fallback(
        self, original_symbol: str, timeframe: TimeFrame
    ) -> Optional[DataSelectionResult]:
        """Try to find data with the same timeframe but different symbol."""
        try:
            # Get symbol priority list, excluding the original symbol
            fallback_symbols = [
                sym for sym in self.symbol_priority if sym != original_symbol
            ]

            for symbol in fallback_symbols:
                try:
                    # Check if data exists for this symbol/timeframe combination
                    data_exists = await self.data_loader.check_data_exists(
                        symbol, timeframe
                    )

                    if data_exists:
                        # Try to load a small sample to verify data is accessible
                        try:
                            sample_data = await self.data_loader.load_data(
                                symbol, timeframe
                            )
                            if not sample_data.empty:
                                self.logger.info(
                                    f"✅ Symbol data fallback found: {original_symbol} -> {symbol} {timeframe.value}"
                                )

                                return DataSelectionResult(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    data_path=None,
                                    selection_priority=DataSelectionPriority.SYMBOL_FALLBACK,
                                    fallback_applied=True,
                                    original_request={
                                        "symbol": original_symbol,
                                        "timeframe": timeframe.value,
                                    },
                                    fallback_reason=FallbackReason.SYMBOL_NOT_AVAILABLE,
                                    confidence_score=0.7,  # Lower confidence for different symbol
                                )
                        except Exception as e:
                            self.logger.debug(
                                f"Data exists but failed to load sample for {symbol} {timeframe.value}: {e}"
                            )
                            continue

                except Exception as e:
                    self.logger.debug(f"Symbol data fallback failed for {symbol}: {e}")
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Error in symbol data fallback: {e}")
            return None

    async def _try_full_data_fallback(
        self, original_symbol: str, original_timeframe: TimeFrame
    ) -> Optional[DataSelectionResult]:
        """Try to find data with both different symbol and timeframe."""
        try:
            # Try combinations of fallback symbols and timeframes
            for tf_str in self.timeframe_priority:
                try:
                    timeframe = TimeFrame(tf_str)

                    for symbol in self.symbol_priority:
                        try:
                            # Check if data exists for this combination
                            data_exists = await self.data_loader.check_data_exists(
                                symbol, timeframe
                            )

                            if data_exists:
                                # Try to load a small sample to verify data is accessible
                                try:
                                    sample_data = await self.data_loader.load_data(
                                        symbol, timeframe
                                    )
                                    if not sample_data.empty:
                                        self.logger.info(
                                            f"✅ Full data fallback found: {original_symbol} {original_timeframe.value} -> {symbol} {timeframe.value}"
                                        )

                                        return DataSelectionResult(
                                            symbol=symbol,
                                            timeframe=timeframe,
                                            data_path=None,
                                            selection_priority=DataSelectionPriority.FULL_FALLBACK,
                                            fallback_applied=True,
                                            original_request={
                                                "symbol": original_symbol,
                                                "timeframe": original_timeframe.value,
                                            },
                                            fallback_reason=FallbackReason.EXACT_MATCH_NOT_FOUND,
                                            confidence_score=0.5,  # Lower confidence for full fallback
                                        )
                                except Exception as e:
                                    self.logger.debug(
                                        f"Data exists but failed to load sample for {symbol} {timeframe.value}: {e}"
                                    )
                                    continue

                        except Exception as e:
                            self.logger.debug(
                                f"Full fallback failed for {symbol} {timeframe.value}: {e}"
                            )
                            continue

                except ValueError:
                    # Invalid timeframe, skip
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Error in full data fallback: {e}")
            return None

    async def _find_best_available_data_fallback(
        self, requested_symbol: str, requested_timeframe: TimeFrame
    ) -> Optional[DataSelectionResult]:
        """Find the best available data as a last resort fallback."""
        try:
            # Scan all available data and find the best one
            best_data = None
            best_score = 0.0

            # Try all symbol/timeframe combinations
            for symbol in self.symbol_priority:
                for tf_str in self.timeframe_priority:
                    try:
                        timeframe = TimeFrame(tf_str)

                        # Check if data exists
                        data_exists = await self.data_loader.check_data_exists(
                            symbol, timeframe
                        )
                        if not data_exists:
                            continue

                        # Try to load sample data
                        try:
                            sample_data = await self.data_loader.load_data(
                                symbol, timeframe
                            )
                            if sample_data.empty:
                                continue

                            # Calculate score based on various criteria
                            score = self._calculate_data_score(
                                symbol, timeframe, requested_symbol, requested_timeframe
                            )

                            if score > best_score:
                                best_score = score
                                best_data = {
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "score": score,
                                }

                        except Exception as e:
                            self.logger.debug(
                                f"Failed to load sample data for {symbol} {timeframe.value}: {e}"
                            )
                            continue

                    except ValueError:
                        # Invalid timeframe, skip
                        continue

            if best_data and best_score > 0.0:
                return DataSelectionResult(
                    symbol=best_data["symbol"],
                    timeframe=best_data["timeframe"],
                    data_path=None,
                    selection_priority=DataSelectionPriority.BEST_AVAILABLE,
                    fallback_applied=True,
                    original_request={
                        "symbol": requested_symbol,
                        "timeframe": requested_timeframe.value,
                    },
                    fallback_reason=FallbackReason.BEST_AVAILABLE_OPTION,
                    confidence_score=best_score,
                )

            return None

        except Exception as e:
            self.logger.error(f"Error finding best available data fallback: {e}")
            return None

    def _calculate_data_score(
        self,
        available_symbol: str,
        available_timeframe: TimeFrame,
        requested_symbol: str,
        requested_timeframe: TimeFrame,
    ) -> float:
        """Calculate a score for data based on how well it matches the request."""
        try:
            score = 0.0

            # Symbol match (40% weight)
            if available_symbol == requested_symbol:
                score += 0.4
            elif available_symbol in self.symbol_priority:
                # Higher priority symbols get higher scores
                symbol_rank = self.symbol_priority.index(available_symbol)
                score += 0.3 * (1.0 - symbol_rank / len(self.symbol_priority))

            # Timeframe match (60% weight - prioritize timeframe over symbol)
            if available_timeframe.value == requested_timeframe.value:
                score += 0.6
            elif available_timeframe.value in self.timeframe_priority:
                # Higher priority timeframes get higher scores
                timeframe_rank = self.timeframe_priority.index(
                    available_timeframe.value
                )
                score += 0.5 * (1.0 - timeframe_rank / len(self.timeframe_priority))

            return score

        except Exception as e:
            self.logger.error(f"Error calculating data score: {e}")
            return 0.0

    def get_data_fallback_summary(self, result: DataSelectionResult) -> Dict[str, Any]:
        """Generate a summary of the data fallback operation for logging and response."""
        return {
            "original_request": result.original_request,
            "selected_data": {
                "symbol": result.symbol,
                "timeframe": result.timeframe.value,
            },
            "fallback_applied": result.fallback_applied,
            "selection_priority": result.selection_priority.value,
            "fallback_reason": (
                result.fallback_reason.value if result.fallback_reason else None
            ),
            "confidence_score": result.confidence_score,
            "timestamp": datetime.now().isoformat(),
        }

    def update_fallback_config(
        self,
        timeframe_priority: Optional[List[str]] = None,
        symbol_priority: Optional[List[str]] = None,
    ) -> None:
        """Update fallback configuration dynamically."""
        if timeframe_priority:
            self.timeframe_priority = timeframe_priority
            self.logger.info(f"Updated timeframe priority: {timeframe_priority}")

        if symbol_priority:
            self.symbol_priority = symbol_priority
            self.logger.info(f"Updated symbol priority: {symbol_priority}")


# Convenience functions for easy access
async def find_available_data(
    data_loader: IDataLoader,
    requested_symbol: str,
    requested_timeframe: TimeFrame,
    enable_fallback: bool = True,
) -> DataSelectionResult:
    """Convenience function to find available data."""
    utils = DataFallbackUtils(data_loader)
    return await utils.find_available_data(
        requested_symbol, requested_timeframe, enable_fallback
    )


def get_data_fallback_summary(result: DataSelectionResult) -> Dict[str, Any]:
    """Convenience function to get data fallback summary."""
    utils = DataFallbackUtils(None)  # We don't need the data_loader for this method
    return utils.get_data_fallback_summary(result)
