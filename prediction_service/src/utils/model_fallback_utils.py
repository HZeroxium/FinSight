# utils/model_fallback_utils.py

"""
Model Fallback Utilities - Handles intelligent model selection and fallback strategies

This module provides utilities for:
- Finding the best available model for prediction requests
- Implementing fallback strategies when exact models are not available
- Managing fallback priorities for timeframes, symbols, and model types
- Providing transparent feedback about which model was actually used
"""

from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from pathlib import Path
from datetime import datetime

from ..schemas.enums import (
    ModelType,
    TimeFrame,
    CryptoSymbol,
    FallbackStrategy,
    ModelSelectionPriority,
)
from ..core.constants import FallbackConstants
from .model_utils import ModelUtils
from common.logger.logger_factory import LoggerFactory, LogLevel


class ModelSelectionResult(NamedTuple):
    """Result of model selection with fallback information"""

    symbol: str
    timeframe: TimeFrame
    model_type: ModelType
    model_path: Optional[Path]
    selection_priority: ModelSelectionPriority
    fallback_applied: bool
    original_request: Dict[str, Any]
    fallback_reason: Optional[str]
    confidence_score: float


class ModelFallbackUtils:
    """
    Utilities for intelligent model selection and fallback strategies with cloud-first approach.

    This class implements a hierarchical fallback system that ensures users
    always receive predictions while maintaining transparency about which
    model was actually used. It now prioritizes cloud storage over local
    storage, automatically downloading models from cloud when needed.
    """

    def __init__(self):
        """Initialize the model fallback utilities."""
        self.logger = LoggerFactory.get_logger("ModelFallbackUtils")
        self.model_utils = ModelUtils()

        # Load fallback configuration
        self.fallback_strategy = FallbackStrategy(
            FallbackConstants.DEFAULT_FALLBACK_STRATEGY
        )
        self.timeframe_priority = FallbackConstants.TIMEFRAME_FALLBACK_PRIORITY
        self.symbol_priority = FallbackConstants.SYMBOL_FALLBACK_PRIORITY
        self.model_type_priority = FallbackConstants.MODEL_TYPE_FALLBACK_PRIORITY

        self.logger.info(
            f"Initialized with fallback strategy: {self.fallback_strategy.value}"
        )

    async def find_best_available_model(
        self,
        requested_symbol: str,
        requested_timeframe: TimeFrame,
        requested_model_type: Optional[ModelType] = None,
        enable_fallback: bool = True,
        fallback_strategy: Optional[FallbackStrategy] = None,
    ) -> ModelSelectionResult:
        """
        Find the best available model for prediction, with intelligent fallback.

        Args:
            requested_symbol: The originally requested trading symbol
            requested_timeframe: The originally requested timeframe
            requested_model_type: The originally requested model type (optional)
            enable_fallback: Whether to enable fallback strategies
            fallback_strategy: Specific fallback strategy to use

        Returns:
            ModelSelectionResult containing the selected model and fallback information
        """
        if fallback_strategy:
            self.fallback_strategy = fallback_strategy

        # Start with exact match attempt
        exact_match = await self._try_exact_match(
            requested_symbol, requested_timeframe, requested_model_type
        )

        if exact_match and exact_match.model_path and exact_match.model_path.exists():
            self.logger.info(
                f"Found exact match: {exact_match.symbol} {exact_match.timeframe.value} {exact_match.model_type.value}"
            )
            return exact_match

        if not enable_fallback:
            # Return the exact match result even if model doesn't exist
            return exact_match or ModelSelectionResult(
                symbol=requested_symbol,
                timeframe=requested_timeframe,
                model_type=requested_model_type or ModelType.PATCHTST,
                model_path=None,
                selection_priority=ModelSelectionPriority.EXACT_MATCH,
                fallback_applied=False,
                original_request={
                    "symbol": requested_symbol,
                    "timeframe": requested_timeframe.value,
                    "model_type": (
                        requested_model_type.value if requested_model_type else None
                    ),
                },
                fallback_reason=None,
                confidence_score=0.0,
            )

        # Apply fallback strategies based on configuration
        fallback_result = await self._apply_fallback_strategies(
            requested_symbol, requested_timeframe, requested_model_type
        )

        if fallback_result:
            self.logger.info(
                f"Applied fallback: {requested_symbol} {requested_timeframe.value} -> "
                f"{fallback_result.symbol} {fallback_result.timeframe.value} "
                f"({fallback_result.fallback_reason})"
            )
            return fallback_result

        # If all fallback strategies fail, return the best available option
        best_available = await self._find_best_available_fallback(
            requested_symbol, requested_timeframe, requested_model_type
        )

        if best_available:
            self.logger.warning(
                f"Using best available fallback: {best_available.symbol} "
                f"{best_available.timeframe.value} {best_available.model_type.value}"
            )
            return best_available

        # Final fallback - return the original request with no model found
        return ModelSelectionResult(
            symbol=requested_symbol,
            timeframe=requested_timeframe,
            model_type=requested_model_type or ModelType.PATCHTST,
            model_path=None,
            selection_priority=ModelSelectionPriority.EXACT_MATCH,
            fallback_applied=False,
            original_request={
                "symbol": requested_symbol,
                "timeframe": requested_timeframe.value,
                "model_type": (
                    requested_model_type.value if requested_model_type else None
                ),
            },
            fallback_reason="No models available for any fallback combination",
            confidence_score=0.0,
        )

    async def _try_exact_match(
        self,
        symbol: str,
        timeframe: TimeFrame,
        model_type: Optional[ModelType],
    ) -> Optional[ModelSelectionResult]:
        """Try to find an exact match for the requested model using cloud-first strategy."""
        try:
            # If no specific model type requested, try to find the best available
            if model_type is None:
                model_type = await self._find_best_model_type(symbol, timeframe)

            if model_type is None:
                return None

            # Use cloud-first strategy to check if model exists and potentially download it
            cloud_load_result = (
                await self.model_utils.cloud_ops.load_model_with_cloud_fallback(
                    symbol, timeframe, model_type, adapter_type="simple"
                )
            )

            if cloud_load_result.get("success", False):
                # Model exists and is available (either locally or downloaded from cloud)
                model_path = Path(cloud_load_result["path"])

                self.logger.info(
                    f"✅ Model found via cloud-first strategy: {symbol} {timeframe} {model_type} "
                    f"(source: {cloud_load_result.get('source', 'unknown')})"
                )

                return ModelSelectionResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    model_path=model_path,
                    selection_priority=ModelSelectionPriority.EXACT_MATCH,
                    fallback_applied=False,
                    original_request={
                        "symbol": symbol,
                        "timeframe": timeframe.value,
                        "model_type": model_type.value,
                    },
                    fallback_reason=None,
                    confidence_score=1.0,
                )
            else:
                # Model not found in cloud or locally
                self.logger.debug(
                    f"❌ Model not found via cloud-first strategy: {symbol} {timeframe} {model_type}"
                )

                return ModelSelectionResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    model_path=None,
                    selection_priority=ModelSelectionPriority.EXACT_MATCH,
                    fallback_applied=False,
                    original_request={
                        "symbol": symbol,
                        "timeframe": timeframe.value,
                        "model_type": model_type.value,
                    },
                    fallback_reason=None,
                    confidence_score=0.0,
                )

        except Exception as e:
            self.logger.error(f"Error in exact match with cloud-first strategy: {e}")
            return None

    async def _apply_fallback_strategies(
        self,
        requested_symbol: str,
        requested_timeframe: TimeFrame,
        requested_model_type: Optional[ModelType],
    ) -> Optional[ModelSelectionResult]:
        """Apply configured fallback strategies."""
        try:
            # Strategy 1: Timeframe fallback (same symbol, different timeframe)
            if self.fallback_strategy in [
                FallbackStrategy.TIMEFRAME_ONLY,
                FallbackStrategy.TIMEFRAME_AND_SYMBOL,
            ]:
                timeframe_fallback = await self._try_timeframe_fallback(
                    requested_symbol, requested_timeframe, requested_model_type
                )
                if timeframe_fallback:
                    return timeframe_fallback

            # Strategy 2: Symbol fallback (different symbol, same timeframe)
            if self.fallback_strategy in [
                FallbackStrategy.SYMBOL_ONLY,
                FallbackStrategy.TIMEFRAME_AND_SYMBOL,
            ]:
                symbol_fallback = await self._try_symbol_fallback(
                    requested_symbol, requested_timeframe, requested_model_type
                )
                if symbol_fallback:
                    return symbol_fallback

            # Strategy 3: Full fallback (different symbol, different timeframe)
            if self.fallback_strategy == FallbackStrategy.TIMEFRAME_AND_SYMBOL:
                full_fallback = await self._try_full_fallback(
                    requested_symbol, requested_timeframe, requested_model_type
                )
                if full_fallback:
                    return full_fallback

            return None

        except Exception as e:
            self.logger.error(f"Error applying fallback strategies: {e}")
            return None

    async def _try_timeframe_fallback(
        self,
        symbol: str,
        original_timeframe: TimeFrame,
        model_type: Optional[ModelType],
    ) -> Optional[ModelSelectionResult]:
        """Try to find a model with the same symbol but different timeframe using cloud-first strategy."""
        try:
            # Get timeframe priority list, excluding the original timeframe
            fallback_timeframes = [
                tf for tf in self.timeframe_priority if tf != original_timeframe.value
            ]

            for tf_str in fallback_timeframes:
                try:
                    timeframe = TimeFrame(tf_str)

                    # Try to find the best model type for this symbol/timeframe combination
                    if model_type is None:
                        best_model_type = await self._find_best_model_type(
                            symbol, timeframe
                        )
                    else:
                        best_model_type = model_type

                    if best_model_type is None:
                        continue

                    # Use cloud-first strategy to check if model exists
                    cloud_load_result = (
                        await self.model_utils.cloud_ops.load_model_with_cloud_fallback(
                            symbol, timeframe, best_model_type, adapter_type="simple"
                        )
                    )

                    if cloud_load_result.get("success", False):
                        model_path = Path(cloud_load_result["path"])

                        self.logger.info(
                            f"✅ Timeframe fallback found via cloud-first: {symbol} {original_timeframe} -> {timeframe} "
                            f"(source: {cloud_load_result.get('source', 'unknown')})"
                        )

                        return ModelSelectionResult(
                            symbol=symbol,
                            timeframe=timeframe,
                            model_type=best_model_type,
                            model_path=model_path,
                            selection_priority=ModelSelectionPriority.TIMEFRAME_FALLBACK,
                            fallback_applied=True,
                            original_request={
                                "symbol": symbol,
                                "timeframe": original_timeframe.value,
                                "model_type": model_type.value if model_type else None,
                            },
                            fallback_reason=f"Timeframe fallback: {original_timeframe.value} -> {timeframe.value}",
                            confidence_score=0.8,
                        )

                except ValueError:
                    # Invalid timeframe, skip
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Error in timeframe fallback: {e}")
            return None

    async def _try_symbol_fallback(
        self,
        original_symbol: str,
        timeframe: TimeFrame,
        model_type: Optional[ModelType],
    ) -> Optional[ModelSelectionResult]:
        """Try to find a model with the same timeframe but different symbol using cloud-first strategy."""
        try:
            # Get symbol priority list, excluding the original symbol
            fallback_symbols = [
                sym for sym in self.symbol_priority if sym != original_symbol
            ]

            for symbol in fallback_symbols:
                try:
                    # Try to find the best model type for this symbol/timeframe combination
                    if model_type is None:
                        best_model_type = await self._find_best_model_type(
                            symbol, timeframe
                        )
                    else:
                        best_model_type = model_type

                    if best_model_type is None:
                        continue

                    # Use cloud-first strategy to check if model exists
                    cloud_load_result = (
                        await self.model_utils.cloud_ops.load_model_with_cloud_fallback(
                            symbol, timeframe, best_model_type, adapter_type="simple"
                        )
                    )

                    if cloud_load_result.get("success", False):
                        model_path = Path(cloud_load_result["path"])

                        self.logger.info(
                            f"✅ Symbol fallback found via cloud-first: {original_symbol} -> {symbol} {timeframe} "
                            f"(source: {cloud_load_result.get('source', 'unknown')})"
                        )

                        return ModelSelectionResult(
                            symbol=symbol,
                            timeframe=timeframe,
                            model_type=best_model_type,
                            model_path=model_path,
                            selection_priority=ModelSelectionPriority.SYMBOL_FALLBACK,
                            fallback_applied=True,
                            original_request={
                                "symbol": original_symbol,
                                "timeframe": timeframe.value,
                                "model_type": model_type.value if model_type else None,
                            },
                            fallback_reason=f"Symbol fallback: {original_symbol} -> {symbol}",
                            confidence_score=0.7,
                        )

                except Exception as e:
                    self.logger.debug(f"Symbol fallback failed for {symbol}: {e}")
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Error in symbol fallback: {e}")
            return None

    async def _try_full_fallback(
        self,
        original_symbol: str,
        original_timeframe: TimeFrame,
        model_type: Optional[ModelType],
    ) -> Optional[ModelSelectionResult]:
        """Try to find a model with both different symbol and timeframe using cloud-first strategy."""
        try:
            # Try combinations of fallback symbols and timeframes
            for tf_str in self.timeframe_priority:
                try:
                    timeframe = TimeFrame(tf_str)

                    for symbol in self.symbol_priority:
                        try:
                            # Try to find the best model type for this combination
                            if model_type is None:
                                best_model_type = await self._find_best_model_type(
                                    symbol, timeframe
                                )
                            else:
                                best_model_type = model_type

                            if best_model_type is None:
                                continue

                            # Use cloud-first strategy to check if model exists
                            cloud_load_result = await self.model_utils.cloud_ops.load_model_with_cloud_fallback(
                                symbol,
                                timeframe,
                                best_model_type,
                                adapter_type="simple",
                            )

                            if cloud_load_result.get("success", False):
                                model_path = Path(cloud_load_result["path"])

                                self.logger.info(
                                    f"✅ Full fallback found via cloud-first: {original_symbol} {original_timeframe} -> {symbol} {timeframe} "
                                    f"(source: {cloud_load_result.get('source', 'unknown')})"
                                )

                                return ModelSelectionResult(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    model_type=best_model_type,
                                    model_path=model_path,
                                    selection_priority=ModelSelectionPriority.FULL_FALLBACK,
                                    fallback_applied=True,
                                    original_request={
                                        "symbol": original_symbol,
                                        "timeframe": original_timeframe.value,
                                        "model_type": (
                                            model_type.value if model_type else None
                                        ),
                                    },
                                    fallback_reason=f"Full fallback: {original_symbol} {original_timeframe.value} -> {symbol} {timeframe.value}",
                                    confidence_score=0.6,
                                )

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
            self.logger.error(f"Error in full fallback: {e}")
            return None

    async def _find_best_model_type(
        self, symbol: str, timeframe: TimeFrame
    ) -> Optional[ModelType]:
        """Find the best available model type for a symbol/timeframe combination using cloud-first strategy."""
        try:
            # Try model types in priority order
            for model_type_str in self.model_type_priority:
                try:
                    model_type = ModelType(model_type_str)

                    # Use cloud-first strategy to check if model exists
                    cloud_load_result = (
                        await self.model_utils.cloud_ops.load_model_with_cloud_fallback(
                            symbol, timeframe, model_type, adapter_type="simple"
                        )
                    )

                    if cloud_load_result.get("success", False):
                        self.logger.debug(
                            f"✅ Best model type found via cloud-first: {symbol} {timeframe} {model_type} "
                            f"(source: {cloud_load_result.get('source', 'unknown')})"
                        )
                        return model_type

                except (ValueError, AttributeError):
                    # Invalid model type, skip
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Error finding best model type: {e}")
            return None

    async def _find_best_available_fallback(
        self,
        requested_symbol: str,
        requested_timeframe: TimeFrame,
        requested_model_type: Optional[ModelType],
    ) -> Optional[ModelSelectionResult]:
        """Find the best available model as a last resort fallback."""
        try:
            # Scan all available models and find the best one
            available_models = self.model_utils.list_available_models()

            if not available_models:
                return None

            # Score models based on various criteria
            best_model = None
            best_score = 0.0

            for model_info in available_models:
                score = self._calculate_model_score(
                    model_info,
                    requested_symbol,
                    requested_timeframe,
                    requested_model_type,
                )

                if score > best_score:
                    best_score = score
                    best_model = model_info

            if best_model and best_score > 0.0:
                return ModelSelectionResult(
                    symbol=best_model["symbol"],
                    timeframe=TimeFrame(best_model["timeframe"]),
                    model_type=ModelType(best_model["model_type"]),
                    model_path=Path(best_model["model_path"]),
                    selection_priority=ModelSelectionPriority.FULL_FALLBACK,
                    fallback_applied=True,
                    original_request={
                        "symbol": requested_symbol,
                        "timeframe": requested_timeframe.value,
                        "model_type": (
                            requested_model_type.value if requested_model_type else None
                        ),
                    },
                    fallback_reason=f"Best available model fallback (score: {best_score:.2f})",
                    confidence_score=best_score,
                )

            return None

        except Exception as e:
            self.logger.error(f"Error finding best available fallback: {e}")
            return None

    def _calculate_model_score(
        self,
        model_info: Dict[str, Any],
        requested_symbol: str,
        requested_timeframe: TimeFrame,
        requested_model_type: Optional[ModelType],
    ) -> float:
        """Calculate a score for a model based on how well it matches the request."""
        try:
            score = 0.0

            # Symbol match (40% weight)
            if model_info["symbol"] == requested_symbol:
                score += 0.4
            elif model_info["symbol"] in self.symbol_priority:
                # Higher priority symbols get higher scores
                symbol_rank = self.symbol_priority.index(model_info["symbol"])
                score += 0.3 * (1.0 - symbol_rank / len(self.symbol_priority))

            # Timeframe match (35% weight)
            if model_info["timeframe"] == requested_timeframe.value:
                score += 0.35
            elif model_info["timeframe"] in self.timeframe_priority:
                # Higher priority timeframes get higher scores
                timeframe_rank = self.timeframe_priority.index(model_info["timeframe"])
                score += 0.25 * (1.0 - timeframe_rank / len(self.timeframe_priority))

            # Model type match (25% weight)
            if (
                requested_model_type
                and model_info["model_type"] == requested_model_type.value
            ):
                score += 0.25
            elif model_info["model_type"] in self.model_type_priority:
                # Higher priority model types get higher scores
                model_type_rank = self.model_type_priority.index(
                    model_info["model_type"]
                )
                score += 0.2 * (1.0 - model_type_rank / len(self.model_type_priority))

            return score

        except Exception as e:
            self.logger.error(f"Error calculating model score: {e}")
            return 0.0

    def get_fallback_summary(self, result: ModelSelectionResult) -> Dict[str, Any]:
        """Generate a summary of the fallback operation for logging and response."""
        return {
            "original_request": result.original_request,
            "selected_model": {
                "symbol": result.symbol,
                "timeframe": result.timeframe.value,
                "model_type": result.model_type.value,
                "model_path": str(result.model_path) if result.model_path else None,
            },
            "fallback_applied": result.fallback_applied,
            "selection_priority": result.selection_priority.value,
            "fallback_reason": result.fallback_reason,
            "confidence_score": result.confidence_score,
            "timestamp": datetime.now().isoformat(),
        }

    def update_fallback_config(
        self,
        fallback_strategy: Optional[FallbackStrategy] = None,
        timeframe_priority: Optional[List[str]] = None,
        symbol_priority: Optional[List[str]] = None,
        model_type_priority: Optional[List[str]] = None,
    ) -> None:
        """Update fallback configuration dynamically."""
        if fallback_strategy:
            self.fallback_strategy = fallback_strategy
            self.logger.info(f"Updated fallback strategy to: {fallback_strategy.value}")

        if timeframe_priority:
            self.timeframe_priority = timeframe_priority
            self.logger.info(f"Updated timeframe priority: {timeframe_priority}")

        if symbol_priority:
            self.symbol_priority = symbol_priority
            self.logger.info(f"Updated symbol priority: {symbol_priority}")

        if model_type_priority:
            self.model_type_priority = model_type_priority
            self.logger.info(f"Updated model type priority: {model_type_priority}")


# Convenience functions for easy access
async def find_best_available_model(
    requested_symbol: str,
    requested_timeframe: TimeFrame,
    requested_model_type: Optional[ModelType] = None,
    enable_fallback: bool = True,
    fallback_strategy: Optional[FallbackStrategy] = None,
) -> ModelSelectionResult:
    """Convenience function to find the best available model."""
    utils = ModelFallbackUtils()
    return await utils.find_best_available_model(
        requested_symbol,
        requested_timeframe,
        requested_model_type,
        enable_fallback,
        fallback_strategy,
    )


def get_fallback_summary(result: ModelSelectionResult) -> Dict[str, Any]:
    """Convenience function to get fallback summary."""
    utils = ModelFallbackUtils()
    return utils.get_fallback_summary(result)
