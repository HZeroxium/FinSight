# FinSight Backtesting API

Advanced backtesting system for cryptocurrency trading strategies with comprehensive data management, strategy execution, and administrative operations.

## 🚀 Features

### Core Capabilities

- **Multi-Strategy Backtesting**: Support for multiple trading strategies (MA Crossover, RSI, Bollinger Bands, MACD, Buy & Hold)
- **Multi-Exchange Support**: Binance integration with extensible framework for additional exchanges
- **Cross-Timeframe Analysis**: Convert and analyze data across different timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- **Pluggable Storage**: Support for CSV, MongoDB, and InfluxDB storage backends
- **Administrative Operations**: Comprehensive data management and system monitoring

### Architecture

Built with **Ports & Adapters (Hexagonal Architecture)** for maximum flexibility:

- **Service Layer**: Business logic and orchestration
- **Repository Layer**: Pluggable storage adapters
- **Strategy Layer**: Extensible trading strategies
- **Adapter Layer**: Exchange connectors and backtesting engines

## 📋 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd FinSight/backtesting

# Install dependencies
pip install -r requirements.txt

# Create environment configuration
cp .env.example .env
# Edit .env with your settings
```

### 2. Start the Server

```bash
# Start the FastAPI server
python -m src.main

# Or using uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

### 3. Access Documentation

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>
- **API Root**: <http://localhost:8000/>

### 4. Run Admin Demo

```bash
# Test admin functionality
python admin_demo.py
```

## 🔧 API Endpoints

### Admin Endpoints

- `GET /admin/health` - System health check (public)
- `GET /admin/stats` - System statistics (auth required)
- `POST /admin/data/ensure` - Ensure data availability
- `POST /admin/data/convert-timeframe` - Convert timeframe data
- `POST /admin/data/cleanup` - Clean up old data
- `GET /admin/info` - Admin endpoint documentation

### Market Data Endpoints

- `GET /api/v1/market-data/ohlcv` - Retrieve OHLCV data
- `GET /api/v1/market-data/ohlcv/stats` - Get data statistics
- `GET /api/v1/market-data/ohlcv/gaps` - Identify data gaps
- `GET /api/v1/market-data/exchanges` - List available exchanges
- `GET /api/v1/market-data/symbols` - List available symbols
- `GET /api/v1/market-data/timeframes` - List available timeframes

### Backtesting Endpoints

- `GET /api/v1/backtesting/strategies` - List available strategies
- `GET /api/v1/backtesting/strategies/{name}/config` - Get strategy configuration
- `POST /api/v1/backtesting/run` - Execute backtest
- `GET /api/v1/backtesting/engines` - List backtesting engines

## 🔐 Authentication

Admin endpoints require API key authentication:

```bash
# Set in environment
export ADMIN_API_KEY="your-secure-api-key"

# Or in .env file
AI_PREDICTION_ADMIN_API_KEY=your-secure-api-key

# Use in requests
curl -H "Authorization: Bearer your-secure-api-key" \
     http://localhost:8000/admin/stats
```

## 📊 Data Management

### Supported Exchanges

- **Binance**: Full OHLCV data collection and real-time updates

### Supported Timeframes

- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `1h` - 1 hour
- `4h` - 4 hours
- `1d` - 1 day

### Storage Backends

- **CSV**: File-based storage for development and small datasets
- **MongoDB**: Document-based storage for production
- **InfluxDB**: Time-series database for high-performance scenarios

## 🎯 Trading Strategies

### Built-in Strategies

1. **Moving Average Crossover**: Buy/sell signals based on MA crossovers
2. **RSI Strategy**: Overbought/oversold signals using RSI indicator
3. **Bollinger Bands**: Mean reversion strategy using Bollinger Bands
4. **MACD Strategy**: Trend following using MACD indicator
5. **Simple Buy & Hold**: Baseline strategy for comparison

### Custom Strategies

Implement the `Strategy` interface to create custom strategies:

```python
from src.strategies.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def generate_signals(self, data: List[OHLCVSchema]) -> List[Dict[str, Any]]:
        # Implement your strategy logic
        return signals
```

## 🔧 Configuration

### Environment Variables

Key configuration options (see `.env.example` for complete list):

```bash
# Server
HOST=0.0.0.0
PORT=8000

# Application
AI_PREDICTION_DEBUG=true
AI_PREDICTION_ENVIRONMENT=development
AI_PREDICTION_ADMIN_API_KEY=your-api-key

# Storage
STORAGE_BASE_DIRECTORY=data
MONGODB_URL=mongodb://localhost:27017/

# Data Collection
DEFAULT_SYMBOLS=BTC/USDT,ETH/USDT,BNB/USDT
DEFAULT_TIMEFRAMES=1h,4h,1d
```

### Repository Configuration

Switch between storage backends in code:

```python
# CSV Repository
repository = create_repository("csv", {
    "base_directory": "data"
})

# MongoDB Repository
repository = create_repository("mongodb", {
    "connection_string": "mongodb://localhost:27017/",
    "database_name": "finsight_backtesting"
})

# InfluxDB Repository
repository = create_repository("influxdb", {
    "url": "http://localhost:8086",
    "token": "your-token",
    "org": "your-org",
    "bucket": "market_data"
})
```

## 📈 Usage Examples

### Ensure Data Availability

```bash
curl -X POST "http://localhost:8000/admin/data/ensure" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z",
    "force_refresh": false
  }'
```

### Convert Timeframe Data

```bash
curl -X POST "http://localhost:8000/admin/data/convert-timeframe" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "source_timeframe": "1h",
    "target_timeframe": "4h",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z",
    "save_converted": true
  }'
```

### Run Backtest

```bash
curl -X POST "http://localhost:8000/api/v1/backtesting/run" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "moving_average_crossover",
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z",
    "initial_capital": 10000,
    "strategy_params": {
      "fast_period": 10,
      "slow_period": 30
    }
  }'
```

### Retrieve Market Data

```bash
curl "http://localhost:8000/api/v1/market-data/ohlcv?exchange=binance&symbol=BTCUSDT&timeframe=1h&start_date=2024-01-01T00:00:00Z&end_date=2024-01-02T00:00:00Z"
```

## 📊 Data Collection & Management

### Starting the API Server

```bash
# Start the FastAPI server
python -m src.main

# Or using uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### New Symbol Data Collection

Use the intelligent data collection script for new symbols:

```bash
# Collect data for a specific symbol and timeframe
python -m src.collect_new_symbol_data --symbol BTCUSDT --timeframe 1h

# Collect data for a specific symbol (all default timeframes)
python -m src.collect_new_symbol_data --symbol BTCUSDT

# Collect data for all default symbols and timeframes
python -m src.collect_new_symbol_data

# Force full historical collection (ignores existing data)
python -m src.collect_new_symbol_data --symbol BTCUSDT --force-full

# Custom date range
python -m src.collect_new_symbol_data --symbol BTCUSDT --start-date 2024-01-01 --end-date 2024-12-31

# Advanced options
python -m src.collect_new_symbol_data \
  --repository mongodb \
  --max-concurrent 5 \
  --max-lookback-days 365
```

#### Collection Strategy

The script automatically selects the optimal collection strategy:

1. **New Symbol/Timeframe**: Collects full historical data using `collect_and_store_ohlcv`
2. **Existing Data**: Updates to latest using `update_to_latest` + ensures completeness with `ensure_data_completeness`
3. **Gap Detection**: Automatically identifies and fills data gaps

### Automated Data Collection (Cron Jobs)

#### Starting the Cron Job Service

```bash
# Start the background job service (runs continuously)
python -m src.market_data_job start

# Start with custom configuration
python -m src.market_data_job start --config-file custom_config.json

# Start with custom log file
python -m src.market_data_job start --log-file logs/market_data_custom.log
```

#### Managing Cron Jobs

```bash
# Check service status
python -m src.market_data_job status

# Stop the service
python -m src.market_data_job stop

# Run a manual collection job (without starting the scheduler)
python -m src.market_data_job run

# Run manual job with specific symbols
python -m src.market_data_job run --symbols BTCUSDT ETHUSDT --timeframes 1h 4h

# Update cron schedule
python -m src.market_data_job config --cron "0 */2 * * *"  # Every 2 hours
```

#### Cron Job Configuration

The cron job uses `scan_and_update_all_symbols` function to:

- Update existing data to latest timestamps
- Ensure data completeness for all symbol/timeframe pairs
- Handle rate limiting and error recovery
- Log detailed results for monitoring

Default configuration:

- **Schedule**: Every hour (`0 */1 * * *`)
- **Strategy**: Update existing data + ensure completeness
- **Priority Symbols**: BTCUSDT, ETHUSDT, BNBUSDT
- **Priority Timeframes**: 1h, 4h, 1d
- **Max Lookback**: 30 days

#### Configuration File Example

```json
{
  "cron_schedule": "0 */1 * * *",
  "timezone": "UTC",
  "exchange": "binance",
  "max_lookback_days": 30,
  "update_existing": true,
  "repository_type": "mongodb",
  "priority_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "priority_timeframes": ["1h", "4h", "1d"],
  "max_retries": 3,
  "enable_notifications": false
}
```

#### Process Management

The cron job service includes production-ready features:

- **PID File Management**: Prevents multiple instances
- **Graceful Shutdown**: Handles SIGINT/SIGTERM signals
- **Comprehensive Logging**: All operations logged to file
- **Error Recovery**: Automatic retries with exponential backoff
- **Health Monitoring**: Status reporting and metrics

#### Monitoring Logs

```bash
# Monitor live logs
tail -f logs/market_data_job.log

# Check recent logs
tail -100 logs/market_data_job.log

# Search for errors
grep -i error logs/market_data_job.log

# Check job statistics
grep -i "job.*completed" logs/market_data_job.log
```

### Data Collection Best Practices

1. **Initial Setup**: Use `collect_new_symbol_data.py` for initial historical data collection
2. **Ongoing Updates**: Use the cron job service for automated updates
3. **Gap Management**: The system automatically detects and fills gaps
4. **Rate Limiting**: Built-in Binance API rate limiting compliance
5. **Error Handling**: Comprehensive error recovery and logging
6. **Monitoring**: Regular log monitoring for collection health

## Usage Examples

### Ensure Data Availabilityy

Before running backtests, ensure you have the required market data:

```python
# Check if data exists for specific symbol/timeframe
from src.services.market_data_service import MarketDataService
from src.factories import create_repository

# Initialize service
repository = create_repository("mongodb", {
    "connection_string": "mongodb://localhost:27017/",
    "database_name": "finsight_market_data"
})
service = MarketDataService(repository)

# Check data availability
symbols = await service.get_available_symbols("binance")
timeframes = await service.get_available_timeframes("binance", "BTCUSDT")

print(f"Available symbols: {symbols}")
print(f"Available timeframes for BTCUSDT: {timeframes}")
```

### Collect Initial Data

```python
# Collect comprehensive historical data
from src.collect_new_symbol_data import NewSymbolDataCollector

collector = NewSymbolDataCollector(
    repository_type="mongodb",
    max_lookback_days=365
)

# Collect data for specific symbol
result = await collector.collect_symbol_data(
    symbol="BTCUSDT",
    timeframe="1h",
    force_full_collection=True
)

print(f"Collection result: {result}")
```

### Run Automated Collection

```bash
# Set up automated hourly collection
python -m src.market_data_job start

# Monitor the logs
tail -f logs/market_data_job.log
```

## 🚀 Quick Start

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Services**:

   ```bash
   # Start API server
   python -m src.main

   # Start cron job service (in another terminal)
   python -m src.market_data_job start
   ```

4. **Collect Initial Data**:

   ```bash
   # Collect data for key symbols
   python -m src.collect_new_symbol_data --symbol BTCUSDT
   python -m src.collect_new_symbol_data --symbol ETHUSDT
   ```

5. **Access API**:
   - API Documentation: <http://localhost:8000/docs>
   - Health Check: <http://localhost:8000/health>
   - Admin Panel: <http://localhost:8000/admin/status>
