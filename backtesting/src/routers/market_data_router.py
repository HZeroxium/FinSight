# routers/market_data_router.py

"""
Market Data Router

Provides endpoints for market data operations including:
- OHLCV data retrieval and querying
- Data statistics and metadata
- Data validation and gap detection
- Storage management operations
"""

from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, status
from ..utils.datetime_utils import DateTimeUtils

from ..services.market_data_service import MarketDataService
from ..schemas.ohlcv_schemas import (
    OHLCVResponseSchema,
    OHLCVStatsSchema,
)
from common.logger import LoggerFactory
from ..factories.market_data_repository_factory import get_market_data_service
from ..utils.datetime_utils import DateTimeUtils


# Router configuration
router = APIRouter(
    prefix="/market-data",
    tags=["market-data"],
    responses={
        400: {"description": "Bad request - Invalid parameters"},
        404: {"description": "Not found - No data available"},
        500: {"description": "Internal server error"},
    },
)

# Logger
logger = LoggerFactory.get_logger(name="market_data_router")


@router.get("/ohlcv", response_model=OHLCVResponseSchema)
async def get_ohlcv_data(
    exchange: str = Query(..., description="Exchange name (e.g., binance)"),
    symbol: str = Query(..., description="Trading symbol (e.g., BTCUSDT)"),
    timeframe: str = Query(..., description="Timeframe (e.g., 1h, 1d)"),
    start_date: str = Query(..., description="Start date (ISO 8601 format)"),
    end_date: str = Query(..., description="End date (ISO 8601 format)"),
    limit: Optional[int] = Query(
        default=None, description="Maximum number of records to return", ge=1, le=10000
    ),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> OHLCVResponseSchema:
    """
    Retrieve OHLCV (candlestick) data for specified parameters.

    Returns historical price and volume data for the given symbol,
    exchange, and time range.
    """
    try:
        logger.info(
            f"OHLCV data requested: {symbol} on {exchange} "
            f"({timeframe}) from {start_date} to {end_date}"
        )

        # Validate date format
        try:
            DateTimeUtils.validate_date_range(start_date, end_date)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid date format: {str(e)}",
            )

        response = await market_data_service.get_ohlcv_data(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        logger.info(f"Retrieved {response.count} OHLCV records")
        return response

    except Exception as e:
        logger.error(f"Failed to retrieve OHLCV data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve OHLCV data: {str(e)}",
        )


@router.get("/ohlcv/stats", response_model=OHLCVStatsSchema)
async def get_ohlcv_stats(
    exchange: str = Query(..., description="Exchange name"),
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> OHLCVStatsSchema:
    """
    Get statistical information about OHLCV data.

    Returns metrics like total records, date range, price ranges,
    and volume statistics for the specified data series.
    """
    try:
        logger.info(f"OHLCV stats requested: {symbol} on {exchange} ({timeframe})")

        stats = await market_data_service.get_ohlcv_stats(exchange, symbol, timeframe)

        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data found for {symbol} on {exchange} ({timeframe})",
            )

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get OHLCV stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get OHLCV statistics: {str(e)}",
        )


@router.get("/ohlcv/latest-timestamp")
async def get_latest_timestamp(
    exchange: str = Query(..., description="Exchange name"),
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> Dict[str, Any]:
    """
    Get the timestamp of the latest OHLCV record.

    Useful for determining data freshness and identifying
    where to resume data collection.
    """
    try:
        logger.info(f"Latest timestamp requested: {symbol} on {exchange} ({timeframe})")

        latest_timestamp = await market_data_service.get_latest_ohlcv_timestamp(
            exchange, symbol, timeframe
        )

        return {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "latest_timestamp": latest_timestamp,
            "has_data": latest_timestamp is not None,
        }

    except Exception as e:
        logger.error(f"Failed to get latest timestamp: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get latest timestamp: {str(e)}",
        )


@router.get("/ohlcv/gaps")
async def get_data_gaps(
    exchange: str = Query(..., description="Exchange name"),
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe"),
    start_date: str = Query(..., description="Start date for gap analysis"),
    end_date: str = Query(..., description="End date for gap analysis"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> Dict[str, Any]:
    """
    Identify gaps in OHLCV data within specified date range.

    Returns a list of missing time periods that need to be collected
    to ensure data completeness.
    """
    try:
        logger.info(
            f"Gap analysis requested: {symbol} on {exchange} "
            f"({timeframe}) from {start_date} to {end_date}"
        )

        # Validate date format
        try:
            DateTimeUtils.validate_date_range(start_date, end_date)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid date format: {str(e)}",
            )

        gaps = await market_data_service.get_data_gaps(
            exchange, symbol, timeframe, start_date, end_date
        )

        return {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_period": {
                "start_date": start_date,
                "end_date": end_date,
            },
            "gaps_found": len(gaps),
            "gaps": [
                {
                    "start": gap[0],
                    "end": gap[1],
                }
                for gap in gaps
            ],
            "is_complete": len(gaps) == 0,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze data gaps: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze data gaps: {str(e)}",
        )


@router.get("/exchanges")
async def get_available_exchanges(
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> Dict[str, Any]:
    """
    Get list of available exchanges in the system.

    Returns all exchanges that have data stored in the repository.
    """
    try:
        logger.info("Available exchanges requested")

        exchanges = await market_data_service.get_available_exchanges()

        return {
            "exchanges": exchanges,
            "count": len(exchanges),
        }

    except Exception as e:
        logger.error(f"Failed to get available exchanges: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available exchanges: {str(e)}",
        )


@router.get("/symbols")
async def get_available_symbols(
    exchange: str = Query(..., description="Exchange name"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> Dict[str, Any]:
    """
    Get list of available symbols for an exchange.

    Returns all symbols that have data stored for the specified exchange.
    """
    try:
        logger.info(f"Available symbols requested for {exchange}")

        symbols = await market_data_service.get_available_symbols(exchange)

        return {
            "exchange": exchange,
            "symbols": symbols,
            "count": len(symbols),
        }

    except Exception as e:
        logger.error(f"Failed to get available symbols: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available symbols: {str(e)}",
        )


@router.get("/timeframes")
async def get_available_timeframes(
    exchange: str = Query(..., description="Exchange name"),
    symbol: str = Query(..., description="Trading symbol"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> Dict[str, Any]:
    """
    Get list of available timeframes for a symbol.

    Returns all timeframes that have data stored for the specified
    exchange and symbol combination.
    """
    try:
        logger.info(f"Available timeframes requested for {symbol} on {exchange}")

        timeframes = await market_data_service.get_available_timeframes(
            exchange, symbol
        )

        return {
            "exchange": exchange,
            "symbol": symbol,
            "timeframes": timeframes,
            "count": len(timeframes),
        }

    except Exception as e:
        logger.error(f"Failed to get available timeframes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available timeframes: {str(e)}",
        )


@router.get("/storage-info")
async def get_storage_info(
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> Dict[str, Any]:
    """
    Get information about the storage backend.

    Returns metadata about the underlying storage system,
    including type, capacity, and utilization.
    """
    try:
        logger.info("Storage info requested")

        storage_info = await market_data_service.get_storage_info()

        return {
            "storage_info": storage_info,
            "timestamp": DateTimeUtils.now_utc(),
        }

    except Exception as e:
        logger.error(f"Failed to get storage info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get storage information: {str(e)}",
        )
